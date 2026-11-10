"""Package addition: write R16b_ongoing_bits.npz from the regenerated F2-operator decision.

s2_decide.r16b_bits() loads R16b_ongoing_bits.npz when it exists and otherwise reads the R16b submission CSV.
R16b's ongoing cells are R11c's (R16b changed state only), and R11c = CAP + serve_refit.py's override, i.e. the
refit-N6 F2 decision d1 on every official ongoing row (guard window = A-block decode). This script maps d1
(review_20260929_target092/main/ongoing_refitN6_d1.parquet) onto the rows of serving_stage/<split>_SUBMISSION_MAP
exactly as r16b_bits() maps the CSV column, and saves the same structure: int8 arrays keyed validation, private.
"""
# Runs in pipeline/ongoing.sh ("r16b_bits"), after serving_stage/prepare.py.
# Output: R16b_ongoing_bits.npz (int8 decision per SUBMISSION_MAP row).
import os
import numpy as np, pandas as pd
# Package edit (2026-11-10): the split list comes from TFB_SPLITS (default 'validation private', see README).
SPLITS = tuple(os.environ.get('TFB_SPLITS', 'validation private').split())
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SS = ROOT / 'build/review_20260930_round20/queue_data/serving_stage'
D1 = ROOT / 'build/review_20260929_target092/main/ongoing_refitN6_d1.parquet'

path = HERE / 'R16b_ongoing_bits.npz'
assert not path.exists(), path
d = pd.read_parquet(D1, columns=['submission_id', 'split', 'd1'])
q = dict(zip(d.submission_id.astype(np.int64), d.d1.astype(int)))
assert len(q) == len(d) == 43500 * len(SPLITS)
bits = {}
for s in SPLITS:
    m = pd.read_parquet(SS / f'{s}_SUBMISSION_MAP.parquet')
    b = np.full(len(m), -1, np.int8)
    for i, sid in enumerate(m.submission_id.to_numpy()):
        b[i] = int(q[int(sid)])
    assert (b >= 0).all() and set(d[d.split == s].submission_id) == set(m.submission_id)
    bits[s] = b
np.savez(path, **bits)
print({s: (len(b), int(b.sum())) for s, b in bits.items()})
