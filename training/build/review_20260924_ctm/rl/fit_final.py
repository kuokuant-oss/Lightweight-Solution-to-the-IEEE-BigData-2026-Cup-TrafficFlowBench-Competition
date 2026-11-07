"""Final RL fit on the pooled train months (8 queue panels, all days), same recipe as rl_eval.fit.
Usage: fit_final.py --months 2031_01 2031_02 --variant RL --name RL_final"""
# Training record for the RLB and RL residual models. The reproduction does not run this
# program; it loads the shipped models. The program reads feature tables that are not
# shipped, and its paths resolve from its original place in the development repository,
# whose layout `trafficflowbench/` mirrors.
#
# Trains: two LightGBM regressors on the error of the base Task 1 state C1 (truth - C1) at
#   blanked cells outside Task 2 holes: speed (km/h, `y_v`) and per-lane flow (veh/h per
#   lane, `y_q`). RL uses 106 inputs without link or corridor identity; `--variant RLB` adds
#   24 co-switching block features (130).
# Data: the feature tables `feats/<corridor>_<month>.parquet` (plus `feats/blk_*` for RLB) of
#   the months in `--months`, 8 Task 2 corridors (`panel_i < 8`; the D12_I405 pair is left
#   out). Free-flow cells (C1 speed >= 0.85 x free-flow speed) are kept at 30% with weight
#   1/0.3 (`rl_eval.fit`, rng seed 0).
# Settings (`rl_eval.P`, `rl_eval.NROUND`): objective l2, learning rate .05, 63 leaves,
#   min_data_in_leaf 300, feature and bagging fraction .8, lambda_l2 10, 400 rounds, 3 threads.
# Output: `models/<name>/y_v.txt`, `y_q.txt`, `feats.json` and `manifest.json` (rows before
#   the free-cell sample, parameters, SHA256 of the files, seconds). Shipped as
#   `trafficflowbench/build/review_20260924_ctm/rl/models/RLB_final/` and `RL_final/`.
# Invocation: `fit_final.py --months 2031_01 2031_02 --variant RLB --name RLB_final` and
#   `fit_final.py --months 2031_01 2031_02 --variant RL --name RL_final` (January from
#   2031-01-04, and February 2031; 5,952,690 rows before the sample). It imports `rl_eval`
#   from its own folder (shipped next to the original).
import os, sys, argparse, json, hashlib, time
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import rl_eval as RE


def sha(p):
    """Returns the SHA256 hex digest of a file."""
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()


ap = argparse.ArgumentParser(); ap.add_argument('--months', nargs='+', required=True); ap.add_argument('--variant', default='RL'); ap.add_argument('--name', required=True)
a = ap.parse_args(); t0 = time.time()
FS = RE.featsets()[a.variant]
D = []
for i, m in enumerate(a.months):
    d = RE.load_month(m, a.variant != 'RL'); d = d[d.panel_i < 8]
    # Day index offset by 100 per month, so days of different months stay distinct.
    d['day'] = d.day.astype(int) + 100 * i; D.append(d)
D = pd.concat(D, ignore_index=True)
ms = RE.fit(D, np.ones(len(D), bool), FS)
md = HERE / 'models' / a.name; md.mkdir(parents=True, exist_ok=True)
for y, m in ms.items(): m.save_model(str(md / f'{y}.txt'))
json.dump(FS, open(md / 'feats.json', 'w'))
man = dict(variant=a.variant, months=a.months, panels=RE.EL.PANELS, rows=len(D), params=RE.P, rounds=RE.NROUND,
           free_subsample=0.3, files={f: sha(md / f) for f in ('y_v.txt', 'y_q.txt', 'feats.json')}, secs=round(time.time() - t0))
json.dump(man, open(md / 'manifest.json', 'w'), indent=1); print(json.dumps(man, indent=1))
