"""Cache train truth (mainline + ramps) per panel as dense arrays [day, link(milepost order), t]."""
#
# For split 'train' the arrays hold the full truth (mainline_states); for 'validation' and 'private' they hold
# the released masked layer (mainline_states_masked), NaN where a cell is blanked.
#
# Role: reproduction pipeline (task1.sh step 4, onset.sh step 2, ongoing.sh) runs it for validation and
# private. Readers: the selector features (dense/serve_sel2.py), the onset site choice
# (ruler/onset_site/serve_official.py) and the ramp-based queue probability of the gap
# (review_20260929_target092/gap_structure/ramp_feats.py). The shipped train caches keep only the `links` array.
#
# Usage: build_cache.py <panel> [split]   (split: train (default), validation or private)
#
# Output: cache/<panel>_<split>.npz with
#   V, Q, O   speed (km/h), flow (veh/h), occupancy as float16, shape (days, links, 288 five-minute steps)
#   E         bool, cell is score-eligible
#   R, RE     ramp flow (veh/h, float16) and ramp eligibility, shape (days, ramps, 288)
#   links, mp link ids and mileposts in milepost order (index grows downstream)
#   dates     'YYYY-MM-DD' per day; ramps, rtype: ramp link ids and ramp types (e.g. OR = on-ramp)
import os
# One thread per process: the pipeline runs one cache build per panel and split.
os.environ['OMP_NUM_THREADS'] = '1'
import sys, glob
import numpy as np, pandas as pd
from pathlib import Path
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
OUT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924_ctm/cache')
panel = sys.argv[1]
split = sys.argv[2] if len(sys.argv) > 2 else 'train'
# Train caches the full truth; scored splits cache the masked layer that is released.
if split == 'train':
    mfiles = sorted(p for p in (REL / 'corridors' / panel / 'train' / 'mainline_states').glob('*/*.parquet') if not p.name.startswith('._'))
else:
    mfiles = sorted(p for p in (REL / 'corridors' / panel / split / 'mainline_states_masked').glob('*/*.parquet') if not p.name.startswith('._'))
# Ramp day files keyed by date 'YYYY_MM_DD', matched to the mainline day below.
rfiles = {p.name.replace('synthetic_ramp_', '').replace('.parquet', ''): p for p in (REL / 'corridors' / panel / split / 'ramp_states').glob('*/*.parquet') if not p.name.startswith('._')}
links = None
# Per-day lists: speed, flow, occupancy, eligibility, ramp flow, ramp eligibility, date.
Vs, Qs, Os, Es, Rs, RE, dates = [], [], [], [], [], [], []
for p in mfiles:
    d = pd.read_parquet(p, columns=['date', 'timestamp', 'link_id', 'milepost', 'speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible'])
    d['link_id'] = d.link_id.astype(str)
    # The first day fixes the link order (by milepost) for all days.
    if links is None:
        order = d[['link_id', 'milepost']].drop_duplicates().sort_values('milepost')
        links = order.link_id.tolist(); mp = order.milepost.to_numpy(); li = {k: i for i, k in enumerate(links)}
    date = str(d.date.iloc[0]); dates.append(date)
    ts = pd.to_datetime(d.timestamp, utc=True)
    # Column = 5-minute step of the day (hour * 12 + minute // 5); 288 steps per day.
    c = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(); r = d.link_id.map(li).to_numpy()
    n = (len(links), 288)
    def lay(col, dt=np.float32):
        """Returns column `col` of the day as a (links x 288) array, NaN where no row exists."""
        g = np.full(n, np.nan, dt); g[r, c] = d[col].to_numpy(); return g
    Vs.append(lay('speed_kmh').astype(np.float16)); Qs.append(lay('flow_vph').astype(np.float16)); Os.append(lay('occupancy').astype(np.float16))
    e = np.zeros(n, bool); e[r, c] = d.is_score_eligible.to_numpy() == 1; Es.append(e)
    key = date.replace('-', '_')
    rp = rfiles.get(key)
    if rp is not None:
        rr = pd.read_parquet(rp, columns=['timestamp', 'ramp_link_id', 'ramp_type', 'flow_vph', 'is_score_eligible'])
        # The first day that has a ramp file fixes the ramp order and the ramp types.
        if 'ramps' not in dir():
            ramps = sorted(rr.ramp_link_id.astype(str).unique()); ri = {k: i for i, k in enumerate(ramps)}
            rtype = rr.drop_duplicates('ramp_link_id').set_index('ramp_link_id').ramp_type.reindex(ramps).to_numpy()
        tsr = pd.to_datetime(rr.timestamp, utc=True); cr = (tsr.dt.hour * 12 + tsr.dt.minute // 5).to_numpy()
        rix = rr.ramp_link_id.astype(str).map(ri).to_numpy()
        g = np.full((len(ramps), 288), np.nan, np.float32); g[rix, cr] = rr.flow_vph.to_numpy()
        ge = np.zeros((len(ramps), 288), bool); ge[rix, cr] = rr.is_score_eligible.to_numpy() == 1
        Rs.append(g.astype(np.float16)); RE.append(ge)
    # Days without a ramp file get all-NaN ramp flow and no eligible ramp cells.
    else:
        Rs.append(None); RE.append(None)
nr = len(ramps)
Rs = [x if x is not None else np.full((nr, 288), np.nan, np.float16) for x in Rs]
RE = [x if x is not None else np.zeros((nr, 288), bool) for x in RE]
np.savez(OUT / f'{panel}_{split}.npz', V=np.stack(Vs), Q=np.stack(Qs), O=np.stack(Os), E=np.stack(Es), R=np.stack(Rs), RE=np.stack(RE),
         links=np.array(links), mp=mp, dates=np.array(dates), ramps=np.array(ramps), rtype=rtype)
print(panel, split, len(dates), len(links), nr, flush=True)
