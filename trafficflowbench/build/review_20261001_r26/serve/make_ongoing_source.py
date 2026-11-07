"""Package addition: final per-row source for Task 2 queue_ongoing.

Reads decisions_BNDT45_{validation,private}.npz (r26_serve_t45.py decide) and writes
<run root>/final_sources/ongoing.parquet: submission_id (int64), queue_pred (int8) = 'delta', validation rows
then private rows in SUBMISSION_MAP order. pipeline/assemble.py places these into the submission.
"""
# Runs in pipeline/ongoing.sh as the last step ("final_source").
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[3] / 'final_sources' / 'ongoing.parquet'   # <run>/trafficflowbench/build/review_20261001_r26/serve -> <run>

parts = []
for s in ('validation', 'private'):
    z = np.load(HERE / f'decisions_BNDT45_{s}.npz')
    parts.append(pd.DataFrame({'submission_id': z['submission_id'].astype(np.int64), 'queue_pred': z['delta'].astype(np.int8)}))
d = pd.concat(parts, ignore_index=True)
assert len(d) == 87000 and d.submission_id.is_unique and set(np.unique(d.queue_pred)) <= {0, 1}
OUT.parent.mkdir(exist_ok=True); assert not OUT.exists(), OUT
d.to_parquet(OUT, index=False)
print('wrote', OUT, len(d), 'positives', int(d.queue_pred.sum()))
