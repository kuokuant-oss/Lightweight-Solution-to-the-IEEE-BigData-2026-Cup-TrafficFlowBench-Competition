"""Task 4 (OD): run the original `build_odme` NNLS and write `<run>/final_sources/odme.parquet`.

`scripts/60_build_submission.py` cannot run as a whole for this purpose: its OUT is the protected
`build/submission_v1`, it also rebuilds the v1 Task 1/2 parts, and it has no OD-only mode. This wrapper
imports it unchanged and calls only `build_odme()`:
- per (split, panel): minimise ||A_m f - c||^2 + 0.05 ||f - b||^2 subject to f >= 0 (scipy nnls on the
  stacked system), from that split's released `synthetic_link_counts.csv` and `synthetic_weak_prior.csv`
  and the network's path-link incidence. No model, no seed, nothing from another split;
- the module's OUT is pointed at `<run>/work/od/`, where `path_flow_submission.csv` is written.
The CSV is then read back the way the official `merge_submissions.py` loads an odme file and keyed to
`submission_id` on (panel, departure_time, path_id). Run it through `bin/tfb-run`.
"""
# Reproduction pipeline: run by pipeline/od.sh (step 5 of run_reproduce.sh).
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Set by bin/tfb-run: the run tree root and its trafficflowbench/ copy.
RUN = Path(os.environ['TFB_WORKSPACE_ROOT'])
ROOT = Path(os.environ['TFB_REPO_ROOT'])
sys.path.insert(0, str(ROOT))
# Load scripts/60_build_submission.py unchanged as module `build60` (B). Loading runs its imports
# and its research-gate call (off unless TFB_RESEARCH_GATE=on), not its main(). B.R is its
# release-path module; B.R.RELEASE is the released data folder inside the run tree.
_spec = importlib.util.spec_from_file_location('build60', ROOT / 'scripts/60_build_submission.py')
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

KEYS = ['panel', 'departure_time', 'path_id']      # merge_submissions.KEYS['odme']
DEST = RUN / 'final_sources' / 'odme.parquet'


def main():
    """Runs build_odme() into <run>/work/od/ and writes final_sources/odme.parquet.

    The output holds submission_id (int64) and path_flow (float64), sorted by submission_id.

    Raises:
        SystemExit: the output parquet or the intermediate CSV already exists.
    """
    if DEST.exists():
        raise SystemExit(f'refusing to overwrite {DEST}')
    B.OUT = RUN / 'work' / 'od'
    if (B.OUT / 'path_flow_submission.csv').exists():
        raise SystemExit(f'refusing to overwrite {B.OUT / "path_flow_submission.csv"}')
    B.OUT.mkdir(parents=True, exist_ok=True)
    csv = B.build_odme()
    # As merge_submissions.load(): read_csv, keys as str, values to_numeric.
    d = pd.read_csv(csv)
    d = d[KEYS + ['path_flow']].copy()
    for c in KEYS:
        d[c] = d[c].astype(str)
    d['path_flow'] = pd.to_numeric(d.path_flow, errors='coerce')
    assert not d.duplicated(KEYS).any()
    # Key each (panel, departure_time, path_id) to its submission_id; every odme row of the key
    # must receive a finite, non-negative flow.
    key = pd.read_csv(B.R.RELEASE / 'submission_key.csv', usecols=['submission_id', 'task'] + KEYS, dtype=str)
    key = key[key.task.eq('odme')]
    m = key.merge(d, on=KEYS, how='left', validate='one_to_one')
    assert len(m) == len(d) and m.path_flow.notna().all() and np.isfinite(m.path_flow).all() and (m.path_flow >= 0).all()
    out = pd.DataFrame({'submission_id': m.submission_id.astype(np.int64).to_numpy(),
                        'path_flow': m.path_flow.astype(np.float64).to_numpy()})
    out = out.sort_values('submission_id', kind='stable').reset_index(drop=True)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DEST, index=False)
    print('wrote', DEST, 'rows', len(out), flush=True)


if __name__ == '__main__':
    main()
