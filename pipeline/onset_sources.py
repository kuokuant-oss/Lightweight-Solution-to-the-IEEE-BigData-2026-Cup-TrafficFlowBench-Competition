"""Write `<run>/final_sources/onset.parquet`: submission_id, queue_pred for all 87,000 Task 2 onset rows.

Source: `build/review_20260925/round2/synthesis/onset_FEA1_official.parquet` (14,500 step-6 rows written by
`sv2_official_FEA1.py`; prob is a 0/1 decision). Mapped through `submission_key.csv` on
(window_id, timestamp, link_id), as sv2 did when it patched its parent CSV: a decision is queued when
prob >= .5, and steps 1-5 (no row in the source) are 0. Onset windows are those with
condition == 'queue_onset' in the released `task2/<panel>/<split>/window_index.csv`.
Run it through `bin/tfb-run`.
"""
# Reproduction pipeline: step 4 of pipeline/onset.sh (log onset_source).
import os
from pathlib import Path

import numpy as np
import pandas as pd

# RUN, ROOT: the run tree and its trafficflowbench/ copy (set by bin/tfb-run); REL: released data.
RUN = Path(os.environ['TFB_WORKSPACE_ROOT'])
ROOT = Path(os.environ['TFB_REPO_ROOT'])
REL = RUN / 'tfb_competition_data' / 'kaggle_public'
SRC = ROOT / 'build/review_20260925/round2/synthesis/onset_FEA1_official.parquet'
DEST = RUN / 'final_sources' / 'onset.parquet'
# Package edit (2026-11-10): the split list comes from TFB_SPLITS (default 'validation private', see README).
SPLITS = tuple(os.environ.get('TFB_SPLITS', 'validation private').split())
# A Task 2 queue row is identified by window, 5-minute timestamp and link.
KEY = ['window_id', 'timestamp', 'link_id']


def main():
    """Maps the stage1 onset decisions to submission rows and writes DEST.

    Output columns: submission_id (int64) and queue_pred (int8, 1 = queued), sorted by id.

    Raises:
        SystemExit: DEST already exists.
    """
    if DEST.exists():
        raise SystemExit(f'refusing to overwrite {DEST}')
    f = pd.read_parquet(SRC)
    f['link_id'] = f.link_id.astype(str)
    assert len(f) == 7250 * len(SPLITS) and not f.duplicated(KEY).any() and set(np.unique(f.prob)) <= {0.0, 1.0}
    # Onset windows of both official splits, from the released window indexes (macOS ._ files skipped).
    onset = set()
    for p in sorted((REL / 'task2').glob('*/*/window_index.csv')):
        if p.name.startswith('._') or p.parent.name not in SPLITS:
            continue
        w = pd.read_csv(p)
        onset |= set(w.loc[w.condition.eq('queue_onset'), 'window_id'].astype(str))
    assert onset == set(f.window_id), 'onset windows differ from the stage1 source'
    # Every queue row of an onset window; rows without a stage1 decision (steps 1-5) become 0.
    key = pd.read_csv(REL / 'submission_key.csv', usecols=['submission_id', 'task'] + KEY, dtype=str)
    q = key[key.task.eq('queue') & key.window_id.isin(onset)]
    k = q.merge(f, on=KEY, how='left', validate='one_to_one')
    assert int(k.prob.notna().sum()) == len(f), 'not every stage1 decision maps to a submission row'
    out = pd.DataFrame({'submission_id': k.submission_id.astype(np.int64).to_numpy(),
                        'queue_pred': (k.prob.fillna(0.0) >= 0.5).astype(np.int8).to_numpy()})
    out = out.sort_values('submission_id', kind='stable').reset_index(drop=True)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DEST, index=False)
    print('wrote', DEST, 'rows', len(out), 'windows', len(onset), 'positives', int(out.queue_pred.sum()), flush=True)


if __name__ == '__main__':
    main()
