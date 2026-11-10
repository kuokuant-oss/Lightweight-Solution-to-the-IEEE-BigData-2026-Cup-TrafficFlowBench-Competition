"""Package addition: final per-row source for Task 2 queue_ongoing.

Reads decisions_BNDT45_{validation,private}.npz (r26_serve_t45.py decide) and writes
<run root>/final_sources/ongoing.parquet: submission_id (int64), queue_pred (int8) = 'delta', validation rows
then private rows in SUBMISSION_MAP order. pipeline/assemble.py places these into the submission.
"""
# Runs in pipeline/ongoing.sh as the last step ("final_source").
import os
import numpy as np, pandas as pd
# Package edit (2026-11-10): the split list comes from TFB_SPLITS (default 'validation private', see README).
SPLITS = tuple(os.environ.get('TFB_SPLITS', 'validation private').split())
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[3] / 'final_sources' / 'ongoing.parquet'   # <run>/trafficflowbench/build/review_20261001_r26/serve -> <run>

parts = []
for s in SPLITS:
    z = np.load(HERE / f'decisions_BNDT45_{s}.npz')
    parts.append(pd.DataFrame({'submission_id': z['submission_id'].astype(np.int64), 'queue_pred': z['delta'].astype(np.int8)}))
d = pd.concat(parts, ignore_index=True)
# Package edit (2026-11-10): row and window counts follow window_index.csv instead of the fixed 5-window layout.
SS = HERE.parents[1] / 'review_20260930_round20/queue_data/serving_stage'
assert len(d) == sum(len(pd.read_parquet(SS / f'{s}_SUBMISSION_MAP.parquet')) for s in SPLITS) and d.submission_id.is_unique and set(np.unique(d.queue_pred)) <= {0, 1}
OUT.parent.mkdir(exist_ok=True); assert not OUT.exists(), OUT
d.to_parquet(OUT, index=False)
print('wrote', OUT, len(d), 'positives', int(d.queue_pred.sum()))
