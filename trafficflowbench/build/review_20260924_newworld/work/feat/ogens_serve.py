"""Package addition (serve side of og_ens.py only): A25 ensemble probabilities on the official windows.

og_ens.py main() also loads the deleted training tables and the February calibration rows; none of that
is needed to serve. This script runs only og_ens.py's serving lines, with og_ens's own member list
(MEMBERS: name, feature list, seed) and og_exp.Q.predict, on the regenerated serve tables:
  serve_{tag}.parquet (build_serve_feats.py) + geom_serve_{tag}.parquet (build_geom.py serve)
  -> ogens_{tag}_prob.parquet (window_id, timestamp, link_id, prob = mean of the 6 member probabilities)
The frozen member models are read from models/og_{name}.txt (where they were archived after training).

  ogens_serve.py [tags...]      default: realval realpriv
"""
# Runs in the reproduction as `pipeline/ongoing.sh` step `ogens`. A25 is the
# location-free ongoing ensemble (A), which enters the final mix with weight
# 0.25; its probabilities also give the empty-window fallback
# (review_20260925/make_never_empty_fallback.py).
import os
for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[n] = '2'
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import og_ens as E  # noqa: E402  (member list and feature lists; its main() is not called)
# og_ens.py and og_exp.py change these variables at import (to 8, then 6);
# set them back to 2.
for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[n] = '2'
Q = E.Q


def serve(tag):
    """Writes the A-ensemble probabilities of the official ongoing rows of a split.

    Args:
      tag: 'realval' (validation) or 'realpriv' (private); selects
        serve_<tag>.parquet and geom_serve_<tag>.parquet and names the output
        ogens_<tag>_prob.parquet.

    Raises:
      AssertionError: when the output exists already or a member model's
        feature names differ from its MEMBERS list.
    """
    out = HERE / f'ogens_{tag}_prob.parquet'
    assert not out.exists(), out
    # identical to og_ens.main(): ongoing rows of the serve table, geometry merged by (window, link, step)
    s = pd.read_parquet(HERE / f'serve_{tag}.parquet'); s = s[s.condition.eq('queue_ongoing')].reset_index(drop=True)
    s = s.merge(pd.read_parquet(HERE / f'geom_serve_{tag}.parquet'), on=['window_id', 'link_id', 'step'], how='left')
    probs = []
    for name, FS, seed in E.MEMBERS:
        m = lgb.Booster(model_file=str(HERE / 'models' / f'og_{name}.txt'))
        assert m.feature_name() == list(FS), name
        probs.append(Q.predict(m, s, FS, False))
        print('member', name, m.current_iteration(), flush=True)
    p = np.mean(probs, axis=0); z = s[['window_id', 'timestamp', 'link_id']].copy(); z['prob'] = p
    z.to_parquet(out, index=False)
    print('wrote', out.name, len(z), 'windows', z.window_id.nunique(), flush=True)


if __name__ == '__main__':
    for t in (sys.argv[1:] or ['realval', 'realpriv']):
        serve(t)
