"""Champion Task 1 predictions (validation split only) -> q/champ_state.parquet.
Source: candidate D (submission_D_onsetsite_blocks.csv, state rows) joined on submission_id to submission_key.csv.
If a (panel, timestamp, link_id) has several stations, speed/flow are averaged (build_fake reindexes by (timestamp, link_id))."""
#
# Role: reproduction pipeline (task1.sh step 5, champ_state). In the pipeline the candidate D CSV is the
# state-only parent rebuilt by pipeline/task1_state_csv.py (v3 stack, k* density projection, G2 holes).
# Output columns: panel, split, timestamp, link_id, speed_kmh, flow_vph, station_id ('+'-joined when several),
# mask_regime, n_station.
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
SUB = REPO / 'build/review_20260924_newworld/candidates/submission_D_onsetsite_blocks.csv'
# Key rows of the Task 1 state task on the validation split.
key = pd.read_csv(REL / 'submission_key.csv', usecols=['submission_id', 'task', 'split', 'panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'],
                  dtype={'submission_id': 'int64', 'task': str, 'split': str, 'panel': str, 'timestamp': str, 'station_id': str, 'link_id': str, 'mask_regime': str})
key = key[(key.task == 'state') & (key.split == 'validation')]
sub = pd.read_csv(SUB, usecols=['submission_id', 'task', 'speed_kmh', 'flow_vph'], dtype={'submission_id': 'int64', 'task': str})
sub = sub[sub.task == 'state'].drop(columns='task')
# Every key row must have a speed and a flow in the candidate.
m = key.merge(sub, on='submission_id', how='left', validate='one_to_one')
assert m.speed_kmh.notna().all() and m.flow_vph.notna().all(), m.speed_kmh.isna().sum()
print('state validation rows', len(m), 'panels', m.panel.nunique())
dup = m.duplicated(['panel', 'timestamp', 'link_id'], keep=False)
print('rows on links with >1 station', int(dup.sum()))
# One row per (panel, timestamp, link): mean over the link's stations.
g = m.groupby(['panel', 'split', 'timestamp', 'link_id'], sort=True)
out = g.agg(speed_kmh=('speed_kmh', 'mean'), flow_vph=('flow_vph', 'mean'), station_id=('station_id', lambda s: '+'.join(sorted(s))),
            mask_regime=('mask_regime', 'first'), n_station=('station_id', 'size')).reset_index()
out['timestamp'] = out.timestamp.astype(str)
out.to_parquet(HERE / 'champ_state.parquet', index=False)
print(out.shape, out.dtypes.to_dict(), out.n_station.value_counts().to_dict())
