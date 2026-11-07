"""Candidate origins on pseudo-val days with the CORRECTED selector rule (reproduces 75/80 official train windows):
ongoing needs an observed queue at T-5 and T-10; persistence IoU uses the clean state AT T; clean labels = block-mate
majority at the same stamp, then temporal interpolation. Days cache from ../ruler/fakeval/days (validation only)."""
#
# Role: reproduction pipeline (task1.sh step 5, cands_v2), after ruler/fakeval/build_fake.py and before
# select_write_v2.py.
# Inputs: ruler/fakeval/days/<panel>_<date>.npz/.parquet (validation days after the champion fill),
# ruler/blocks/my_blocks.npy (per panel, the block id of each link: groups of links that queue together),
# ruler/fakeval/cands_validation.parquet (for the masking regime of each day).
# Outputs: clean/<panel>_<date>.npy (clean queue state, links x steps) and cands_v2.parquet with columns panel,
# date, t (origin column), cov, hist_any, last_any, hor_any, piou, first, regime.
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
RUL=Path('/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924_ctm/ruler')
HERE=Path(__file__).resolve().parent
# B: panel -> {'links': link ids, 'bid': block id per link}.
B=np.load(RUL/'blocks/my_blocks.npy',allow_pickle=True).item()
rows=[]; (HERE/'clean').mkdir(exist_ok=True)
for f in sorted(p for p in glob.glob(str(RUL/'fakeval/days/*.npz')) if '/._' not in p):
    # File stem '<panel>_<date>'; the panel name itself contains underscores.
    z=np.load(f,allow_pickle=True); panel='_'.join(Path(f).stem.split('_')[:-1]); date=Path(f).stem.split('_')[-1]
    V,E,vc=z['V'],z['E'],z['vc']; links=z['links'].astype(str); times=[str(x) for x in z['times']]
    d=pd.read_parquet(str(f).replace('.npz','.parquet'),columns=['timestamp','speed_kmh'])
    # Gap stamps: stamps at which every link is blank, such as the 90-minute gaps after official windows.
    ts_null=d.groupby('timestamp').speed_kmh.apply(lambda s: s.isna().all()); gap=np.array([ts_null.get(t,False) for t in times])
    # Qobs: observed queue on eligible cells (speed at or below vc, 60% of free-flow speed).
    obs=~np.isnan(V); Qobs=(V<=vc)&E&obs
    # Clean queue state, first from speeds interpolated along time.
    Vi=pd.DataFrame(V).interpolate(axis=1,limit_direction='both').to_numpy(); Q=Vi<=vc
    bid=dict(zip(B[panel]['links'],B[panel]['bid'])); bl=np.array([bid.get(l,-1) for l in links])
    # Unobserved cells of a link in a block of 2+ links take the strict majority state of its observed
    # block-mates at the same stamp (ties count as free flow).
    q=(V<=vc)&obs
    for b in np.unique(bl[bl>=0]):
        ix=np.where(bl==b)[0]
        if len(ix)<2: continue
        no=obs[ix].sum(0); nq=q[ix].sum(0); maj=nq*2>no
        for j in ix:
            miss=~obs[j]&(no>0); Q[j,miss]=maj[miss]
    np.save(HERE/'clean'/f'{panel}_{date}.npy',Q)
    # The regime is joined from the v1 candidate table below.
    regime=None
    # Candidate origins t: history t-12..t-1 and horizon t+1..t+6, none at a gap stamp.
    for t in range(13,len(times)-6):
        if gap[t-12:t+7].any(): continue
        cov=E[:,t-12:t].mean(); hor=Q[:,t+1:t+7]
        # piou: IoU between the clean state at T held over the horizon (persistence) and the horizon itself.
        pers=np.repeat(Q[:,t][:,None],6,1); u=(pers|hor).sum(); piou=(pers&hor).sum()/u if u else 1.0
        first=int(np.argmax(hor.any(0)))+1 if hor.any() else 0
        # last_any: observed queue at both T-5 and T-10 (columns t-1 and t-2).
        rows.append((panel,date,t,cov,bool(Qobs[:,t-12:t].any()),bool(Qobs[:,t-1].any() and Qobs[:,t-2].any()),bool(hor.any()),piou,first))
c=pd.DataFrame(rows,columns=['panel','date','t','cov','hist_any','last_any','hor_any','piou','first'])
old=pd.read_parquet(RUL/'fakeval/cands_validation.parquet')[['panel','date','t','regime']]
c=c.merge(old,on=['panel','date','t'],how='left')
c.to_parquet(HERE/'cands_v2.parquet',index=False); print(len(c), c.regime.isna().mean())
