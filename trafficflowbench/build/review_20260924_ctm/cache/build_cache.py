"""Cache train truth (mainline + ramps) per panel as dense arrays [day, link(milepost order), t]."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys, glob
import numpy as np, pandas as pd
from pathlib import Path
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
OUT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924_ctm/cache')
panel = sys.argv[1]
split = sys.argv[2] if len(sys.argv) > 2 else 'train'
if split == 'train':
    mfiles = sorted(p for p in (REL / 'corridors' / panel / 'train' / 'mainline_states').glob('*/*.parquet') if not p.name.startswith('._'))
else:
    mfiles = sorted(p for p in (REL / 'corridors' / panel / split / 'mainline_states_masked').glob('*/*.parquet') if not p.name.startswith('._'))
rfiles = {p.name.replace('synthetic_ramp_', '').replace('.parquet', ''): p for p in (REL / 'corridors' / panel / split / 'ramp_states').glob('*/*.parquet') if not p.name.startswith('._')}
links = None
Vs, Qs, Os, Es, Rs, RE, dates = [], [], [], [], [], [], []
for p in mfiles:
    d = pd.read_parquet(p, columns=['date', 'timestamp', 'link_id', 'milepost', 'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible'])
    d['link_id'] = d.link_id.astype(str)
    if links is None:
        order = d[['link_id', 'milepost']].drop_duplicates().sort_values('milepost')
        links = order.link_id.tolist(); mp = order.milepost.to_numpy(); li = {k: i for i, k in enumerate(links)}
    date = str(d.date.iloc[0]); dates.append(date)
    ts = pd.to_datetime(d.timestamp, utc=True)
    c = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(); r = d.link_id.map(li).to_numpy()
    n = (len(links), 288)
    def lay(col, dt=np.float32):
        g = np.full(n, np.nan, dt); g[r, c] = d[col].to_numpy(); return g
    Vs.append(lay('speed_kmh').astype(np.float16)); Qs.append(lay('flow_vph').astype(np.float16)); Os.append(lay('occupancy').astype(np.float16))
    e = np.zeros(n, bool); e[r, c] = d.is_score_eligible.to_numpy() == 1; Es.append(e)
    key = date.replace('-', '_')
    rp = rfiles.get(key)
    if rp is not None:
        rr = pd.read_parquet(rp, columns=['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph', 'is_score_eligible'])
        if 'ramps' not in dir():
            ramps = sorted(rr.ramp_link_id.astype(str).unique()); ri = {k: i for i, k in enumerate(ramps)}
            rtype = rr.drop_duplicates('ramp_link_id').set_index('ramp_link_id').ramp_type.reindex(ramps).to_numpy()
        tsr = pd.to_datetime(rr.timestamp, utc=True); cr = (tsr.dt.hour * 12 + tsr.dt.minute // 5).to_numpy()
        rix = rr.ramp_link_id.astype(str).map(ri).to_numpy()
        g = np.full((len(ramps), 288), np.nan, np.float32); g[rix, cr] = rr.flow_vph.to_numpy()
        ge = np.zeros((len(ramps), 288), bool); ge[rix, cr] = rr.is_score_eligible.to_numpy() == 1
        Rs.append(g.astype(np.float16)); RE.append(ge)
    else:
        Rs.append(None); RE.append(None)
nr = len(ramps)
Rs = [x if x is not None else np.full((nr, 288), np.nan, np.float16) for x in Rs]
RE = [x if x is not None else np.zeros((nr, 288), bool) for x in RE]
np.savez(OUT / f'{panel}_{split}.npz', V=np.stack(Vs), Q=np.stack(Qs), O=np.stack(Os), E=np.stack(Es), R=np.stack(Rs), RE=np.stack(RE),
         links=np.array(links), mp=mp, dates=np.array(dates), ramps=np.array(ramps), rtype=rtype)
print(panel, split, len(dates), len(links), nr, flush=True)
