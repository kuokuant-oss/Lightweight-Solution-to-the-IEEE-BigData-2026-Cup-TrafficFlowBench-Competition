"""Write review_20260927_research/strength/validation_predictions.parquet, exactly as the validation part of
build/review_20260927_research/strength.py does (lines 6-12 of that file, same column arithmetic, same asserts).

  python task1_strength_validation.py <run_root>/trafficflowbench

strength.py itself is not run: after this part it scores February with the HMP50 pool on the February c0 caches
(build/review_20260924_ctm/g2v2/c0_cache, about 1.8 GB of train-derived caches) and writes a research report.
Only this parquet is read downstream (gap_structure/c1_transfer.base_preds, used by c4_phead_transfer.py).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

SPEED_FLOOR, SPEED_CAP = 6.0, 120.74999999999999   # serve_lib.SPEED_FLOOR / SPEED_CAP
here = Path(sys.argv[1]) / 'build/review_20260927_research'
dest = here / 'strength'
dest.mkdir(exist_ok=True)
d = pd.read_parquet(here / 'full_pool/validation_predictions.parquet')
d['pool50'] = 2 * d.adaptive - d.t1; d['hm50'] = 2 * d.hm25 - d.t1
assert np.all(d[['pool50', 'hm50']].to_numpy() >= SPEED_FLOOR - 1e-8) and np.all(d[['pool50', 'hm50']].to_numpy() <= SPEED_CAP + 1e-8)
assert np.max(np.abs(d.pool50 - (.5 * d.t1 + .5 * (4 * d.adaptive - 3 * d.t1)))) < 1e-10
d.to_parquet(dest / 'validation_predictions.parquet', index=False)
print('wrote', dest / 'validation_predictions.parquet', len(d))
