"""Package addition: write R21a_ongoing_bits.npz from the regenerated C1f decisions.

r22_serve.r21a_bits() loads R21a_ongoing_bits.npz when it exists; otherwise it reads the R21a submission CSV and
asserts that its ongoing cells equal review_20261001_fresh/serve/decisions_C1f_<split>.npz 'delta' (R21a was
written by s3_patch_csv.py from exactly that delta). This script saves that delta in the same structure
(int8 arrays keyed validation, private, in SUBMISSION_MAP row order).
"""
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
R21S = HERE.parents[1] / 'review_20261001_fresh/serve'

path = HERE / 'R21a_ongoing_bits.npz'
assert not path.exists(), path
bits = {s: np.load(R21S / f'decisions_C1f_{s}.npz')['delta'].astype(np.int8) for s in ('validation', 'private')}
assert all(len(b) == 43500 for b in bits.values())
np.savez(path, **bits)
print({s: (len(b), int(b.sum())) for s, b in bits.items()})
