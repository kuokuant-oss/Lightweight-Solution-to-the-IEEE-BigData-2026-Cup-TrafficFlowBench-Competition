"""Champion (G1) non-hole Task 1 state values on validation, keyed by (panel, timestamp, link_id).
G1 non-hole state rows == build/review_20260924_ctm/rl/rlb_state_rows.parquet (RLB-hybrid + k*), hole rows == g2v2F.
Output: work/champ_val_state.parquet (panel, timestamp, link_id, speed_kmh, flow_vph, hole flag)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
from src.tfb import release as R
HERE = Path(__file__).resolve().parent
parts = []
for z in pd.read_csv(R.RELEASE / 'submission_key.csv', usecols=['submission_id', 'task', 'split', 'panel', 'timestamp', 'link_id'],
                     dtype=str, chunksize=2_000_000):
    z = z[(z.task == 'state') & (z.split == 'validation')]
    parts.append(z[['submission_id', 'panel', 'timestamp', 'link_id']])
k = pd.concat(parts, ignore_index=True); k['submission_id'] = k.submission_id.astype(np.int64)
rl = pd.read_parquet(ROOT / 'build/review_20260924_ctm/rl/rlb_state_rows.parquet'); rl['hole'] = False
g2 = pd.read_parquet(ROOT / 'build/review_20260925/task1/g2v2F_state_rows.parquet'); g2['hole'] = True
ch = pd.concat([rl, g2], ignore_index=True)
k = k.merge(ch, on='submission_id', how='left')
print('validation state rows', len(k), 'missing champion', int(k.speed_kmh.isna().sum()), 'hole rows', int(k.hole.sum()))
k['panel'] = k.panel.astype('category')
k.to_parquet(HERE / 'work' / 'champ_val_state.parquet', index=False)
