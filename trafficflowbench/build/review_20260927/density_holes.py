"""Density-target pilot in 18-step holes; same feature set as T1, fixed blends."""
# Research record; pipeline/task1.sh does not run this file. It tests a
# learned density correction for the hole speeds of T1 inside the 90-minute
# gaps (18 five-minute steps after each Task 2 forecast time). "T1": the
# LightGBM speed correction of round 3 (serve_lib.py), trained on pseudo
# holes cut into observed training data. Here a second LightGBM predicts the
# per-lane density k = flow / speed (veh/km/lane), and the hole speed is
# re-derived as flow / (blend of the T1 density and the model density).
# Candidates are scored on the February 2031 holes with the state and
# physics (fundamental diagram, LWR conservation) proxies of the harness.
# Outputs in density_holes/: the boosters (*.txt), inworld_predictions.parquet,
# inworld_density.npz, inworld_days.parquet (per panel-day scores) and
# inworld_results.json (parity check and per-panel score changes).
import os
# Caps the BLAS/OpenMP thread pools at 3 before numpy and LightGBM load.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='3'
from pathlib import Path
import sys,json,gc
import numpy as np,pandas as pd,lightgbm as lgb
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent/'density_holes';HERE.mkdir(exist_ok=True)
# R3: T1 serving code, in-world rows and predictions; R2: training pseudo
# holes; G2: G2v2 evaluation windows and cached day fields.
R3=ROOT/'build/review_20260925/round3/task1_trans_serving'
R2=ROOT/'build/review_20260925/round2/task1_trans'
G2=ROOT/'build/review_20260924_ctm/g2v2'
sys.path.insert(0,str(R3));import serve_lib as S
sys.path.insert(0,str(R3/'verify'));from v6_phys import acc,summarize,physics


def prepare(d):
    """Adds derived features, the parent density and its residual target.

    Args:
      d: Hole rows with the T1 features, the parent speed `ch_v` (km/h), the
        parent per-lane flow `ch_ql` (vph/lane), the link index `li` and,
        on labelled rows, the true values `y_v` and `y_ql`.

    Returns:
      The same frame with `base_k` (parent density, veh/km/lane), `r_k`
      (true minus parent density, labelled rows only) and `density_weight`
      (link length x lanes, scaled to mean 1).
    """
    S.derive(d)
    d['base_k']=d.ch_ql/d.ch_v.clip(lower=1)
    if 'y_v' in d:d['r_k']=d.y_ql/d.y_v.clip(lower=1)-d.base_k
    # Weights each row by the road space of its link (length x lanes), so
    # the loss counts vehicles rather than cells.
    weights=np.empty(len(d))
    for p,ix in d.groupby('panel').indices.items():
        zz=np.load(ROOT/f'build/review_20260924/work/g2/{p}_profile_all.npz',allow_pickle=True)
        par=physics.link_params(p,[str(l) for l in zz['links']])
        weights[ix]=(par['length_km']*np.maximum(par['lanes'],1))[d.li.to_numpy(int)[ix]]
    d['density_weight']=weights/weights.mean()
    return d


def fit(d,name):
    """Trains the L1 density-residual LightGBM and saves it as density_holes/<name>.txt.

    Args:
      d: Prepared training rows (see `prepare`).
      name: File stem of the saved booster.

    Returns:
      The trained booster.
    """
    m=lgb.train(dict(S.PAR,objective='regression_l1'),lgb.Dataset(d[S.FEATS+['base_k']],d.r_k,weight=d.density_weight),S.NR)
    m.save_model(str(HERE/f'{name}.txt'));return m


def pred(m,d):
    """Returns the model density (veh/km/lane): parent density plus the predicted residual, floored at 0.1."""
    return np.maximum(d.base_k.to_numpy()+m.predict(d[S.FEATS+['base_k']],num_threads=3),.1)


def blend(q,v,k,w):
    """Re-derives speed from flow and a blend of two densities.

    Args:
      q: Per-lane flow (vph/lane).
      v: Speed of the base candidate (km/h).
      k: Model density (veh/km/lane).
      w: Weight of the model density; 0 returns `v`, 1 uses `k` alone.

    Returns:
      Speed q / ((1 - w) * q / v + w * k) in km/h, clipped to the
      submission speed range [S.SPEED_FLOOR, S.SPEED_CAP].
    """
    return np.clip(q/np.maximum((1-w)*q/v+w*k,.1),S.SPEED_FLOOR,S.SPEED_CAP)


def run():
    """Fits the density models, builds the blended candidates and scores them."""
    A=prepare(pd.read_parquet(R3/'work/inworld_apply_rows.parquet'))
    F=S.load_ph([R2/'work'/f'ph_train_r2_{p}.parquet' for p in S.PANELS])
    # Test holes: the G2v2 February 2031 evaluation windows; a hole starts at
    # origin + 1 and lasts 18 steps.
    windows=pd.concat([pd.read_csv(G2/f'eval_windows_{p}_2031_02.csv')[['panel','date','origin']] for p in S.PANELS])
    holes={}
    for w in windows.itertuples():holes.setdefault((w.panel,w.date),[]).append(int(w.origin)+1)
    # Drops training pseudo holes whose span (13 steps before to 12 steps
    # after the 18-step hole) overlaps a test hole; keeps observed cells only.
    win=F[['panel','date','origin']].drop_duplicates()
    win['bad']=[any((o0+1-13<o+18)and(o<o0+1+18+12)for o in holes.get((p,d),[]))for p,d,o0 in zip(win.panel,win.date,win.origin)]
    F=F.merge(win,on=['panel','date','origin'],validate='many_to_one');F=prepare(F[(~F.bad)&F.obs].reset_index(drop=True))
    print('fit',len(F),'apply',len(A),'test_windows',len(windows),flush=True)
    # p: T1 hole speeds of the same rows ('serve': fit on all days;
    # 'strict': fit on the other day parity).
    p=np.load(R3/'verify/work/v5_inworld_preds.npz')
    K=A[['panel','date','origin','li','step_i','ch_ql','y_v','y_ql','is_target']].copy()
    # Serve density model on all training rows; strict models fit on odd
    # (even) days and predict the even (odd) days.
    m=fit(F,'inworld_serve');kp=pred(m,A);del m
    ks=np.full(len(A),np.nan)
    for odd in (True,False):
        mask=(F.day%2==1).to_numpy()==odd;test=(A.day%2==1).to_numpy()!=odd
        m=fit(F.loc[mask],f'inworld_strict_fitodd{int(odd)}');ks[test]=pred(m,A.loc[test]);del m;gc.collect()
    # Candidates: the T1 speed alone (`<mode>_base`) and the density blends
    # at w = .25, .5 and 1; hole flow stays the parent flow.
    for mode,k in [('serve',kp),('strict',ks)]:
        K[f'{mode}_base']=p[mode]
        for w in (.25,.5,1.):K[f'{mode}_w{w}']=blend(A.ch_ql.to_numpy(),p[mode],k,w)
    K.to_parquet(HERE/'inworld_predictions.parquet',index=False)
    np.savez_compressed(HERE/'inworld_density.npz',serve=kp,strict=ks)
    del F;gc.collect()
    # Scores every candidate per panel-day: hole cells take the candidate
    # speed and the parent flow, all other cells keep the cached day fields
    # (v0/q0 of the c0_cache).
    rows=[];parity={}
    names=['serve_base','strict_base']+[f'{mode}_w{w}'for mode in ('serve','strict')for w in (.25,.5,1.)]
    refdays=pd.read_csv(ROOT/'build/review_20260925/round4/task1_flow_trans/work/r3_inworld_days.csv')
    for panel in S.PANELS:
        z=dict(np.load(G2/'c0_cache'/f'{panel}_2031_02_v2.npz',allow_pickle=True));meta=json.loads(str(z.pop('meta')))
        par=physics.link_params(panel,meta['links']);keep=physics.scoreable_mask(panel,meta['links'],'train',meta['times'])
        lanes=np.maximum(par['lanes'],1.);ln=lanes[:,None];length=par['length_km'][:,None];vf=par['free_speed_kmh'][:,None]
        capl=par['capacity_vph'][:,None]/ln;kcl=par['critical_density'][:,None]/ln;kjl=par['k_jam'][:,None]/ln
        c=K[K.panel==panel]
        for di,date in enumerate(meta['dates']):
            key=f'd{di}';a=c[c.date==date];li=a.li.to_numpy(int);ti=(a.origin+1+a.step_i).to_numpy(int)
            for name in names:
                v=z[f'{key}_v0'].astype(float).copy();q=z[f'{key}_q0'].astype(float).copy()
                v[li,ti]=a[name];q[li,ti]=a.ch_ql.to_numpy()*lanes[li]
                r=acc(v,q,z,key,length,ln,vf,capl,kcl,kjl,keep)
                rows.append(dict(panel=panel,date=date,regime=meta['regimes'][di],cand=name,**r))
        print('scored',panel,flush=True)
        del z;gc.collect()
    days=pd.DataFrame(rows);days.to_parquet(HERE/'inworld_days.parquet',index=False)
    # Parity check: the base candidates must reproduce the earlier day
    # scores (r3_inworld_days.csv) to a relative error below 1e-5.
    for mode in ('serve','strict'):
        x=days[days.cand==f'{mode}_base'].merge(refdays[refdays.cand==f'{mode}_base'],on=['panel','date'],suffixes=('_new','_old'),validate='one_to_one')
        parity[mode]={col:float(np.max(np.abs(x[col+'_new']-x[col+'_old'])/np.maximum(np.abs(x[col+'_old']),1e-9)))for col in ['sse_v_in','numB','denB','fdnB']}
    assert max(v for d in parity.values() for v in d.values())<1e-5,parity
    result={'parity':parity}
    table=[]
    # Per-panel score change of each blend against its base;
    # total_proxy = .35 x state change + .15 x physics frame-B change.
    for mode in ('serve','strict'):
        for w in (.25,.5,1.):
            name=f'{mode}_w{w}';pp={}
            for panel in S.PANELS:
                sub=days[days.panel==panel];base=summarize(sub[sub.cand==f'{mode}_base']);new=summarize(sub[sub.cand==name])
                delta={k:float(new[k]-base[k])for k in ('S_state','S_phys_A','S_phys_B','S_LWR_B')}
                delta['total_proxy']=.35*delta['S_state']+.15*delta['S_phys_B'];pp[panel]=delta
            avg={key:float(np.mean([x[key]for x in pp.values()]))for key in next(iter(pp.values()))}
            result[name]={'mean':avg,'perpanel':pp}
            print(name,json.dumps(result[name]),flush=True)
    (HERE/'inworld_results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':run()
