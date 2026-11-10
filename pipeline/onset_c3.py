"""Recompute D's C3 onset component (u-model site choice on the multi-site panels).

The original C3 file was written by `zoom_scripts/onset_zoomin/build_candidate.py 10 0.01 D7_I210_W` into
a session scratchpad that no longer exists, from two intermediates that were not preserved
(`u_official_C1.0.parquet`, `site_train.parquet`). This program rebuilds the same component with the
preserved code, the same way `round2/synthesis/sv2_official_FEA1.py` recomputes D's choice:
- `serve_official.official_rows(split)`: per window and candidate site, the bottleneck q/c features from
  the window's released history (T-60..T-5) and the masked layer at the origin stamp T;
- `serve_official.score_rows` with the frozen u-models (`u_models_C1.0.pkl`);
- `common.choices(rows, 'u')`: D's rule (sites with >= 10 train onsets, skip a window when the best
  eligible u < .01, D7_I210_W excluded);
- `serve_official.decode(prod, choice, tpl)` (prod_else_tpl) on the 0922 production component.

Writes `build/review_20260924_newworld/work/onset_site/
onset_component_submission_C3_onset_sitehazard_minev10_umin0.01_exclD7_I210_W.parquet`, the file name
`build_D_inputs.py` reads (refuses to overwrite). Run it through `bin/tfb-run`.
"""
# Reproduction pipeline: step 2 of pipeline/onset.sh (log c3).
import os
import sys
from pathlib import Path

import pandas as pd

# ROOT: the run tree's trafficflowbench/ copy (set by bin/tfb-run). OJ: the onset_joint folder whose
# common.py holds D's site-choice rule (choices), the multi-site panels (MS: D12_I5_N, D12_I5_S,
# D7_I405_S), the frozen u-models (UMODELS) and the train queue-extent templates (TPL).
ROOT = Path(os.environ['TFB_REPO_ROOT'])
OJ = ROOT / 'build/review_20260925/round2/onset_joint'
sys.path.insert(0, str(OJ))
import common as C  # noqa: E402
# SO is the serve_official module that common.py loads. Its HERE folder sets where it looks for the
# masked-layer caches; fakeHERE/cache links to the caches that onset.sh builds in step 2.
SO = C.SO; SO.HERE = C.ONS / 'fakeHERE'   # as sv2_official_FEA1.py: caches for both official splits
from serve_official import official_rows, score_rows, decode  # noqa: E402
# Package edit (2026-11-10): the split list comes from TFB_SPLITS (default 'validation private', see README).
SPLITS = tuple(os.environ.get('TFB_SPLITS', 'validation private').split())

DEST = ROOT / ('build/review_20260924_newworld/work/onset_site/'
               'onset_component_submission_C3_onset_sitehazard_minev10_umin0.01_exclD7_I210_W.parquet')


def main():
    """Rebuilds the C3 onset component for both official splits and writes it to DEST.

    Raises:
        SystemExit: DEST already exists.
    """
    if DEST.exists():
        raise SystemExit(f'refusing to overwrite {DEST}')
    # Step-6 (T+30) onset rows of the 0922 production component, with prob and queue_pred per link.
    comp = pd.read_parquet(ROOT / 'build/strongest_candidate_20260922/onset/production/onset_component.parquet')
    comp['link_id'] = comp.link_id.astype(str)
    choice = {}
    # Site choice per split on the multi-site panels; the two splits have disjoint window ids.
    for split in SPLITS:
        offi = official_rows(split)
        offi = offi[offi.panel.isin(C.MS)].reset_index(drop=True)
        ch = C.choices(score_rows(offi, C.UMODELS), 'u')
        assert not set(ch) & set(choice)
        choice.update(ch)
        print(split, 'site choices', len(ch), flush=True)
    # In each window with a choice, keep the production cells at the chosen site; if none is
    # queued there, use the site's template cells (>= .5), else its highest-probability link.
    # Windows without a choice keep the production component.
    new = decode(comp, choice, C.TPL)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(DEST, index=False)
    # Report how many cells and windows the site choice changed.
    m = comp.merge(new[['window_id', 'link_id', 'queue_pred']], on=['window_id', 'link_id'], suffixes=('', '_new'))
    ch = m[m.queue_pred != m.queue_pred_new]
    print('wrote', DEST, 'choices', len(choice), 'changed cells', len(ch), 'windows', ch.window_id.nunique(), flush=True)


if __name__ == '__main__':
    main()
