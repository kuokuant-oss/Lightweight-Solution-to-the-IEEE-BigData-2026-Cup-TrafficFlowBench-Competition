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
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ['TFB_REPO_ROOT'])
OJ = ROOT / 'build/review_20260925/round2/onset_joint'
sys.path.insert(0, str(OJ))
import common as C  # noqa: E402
SO = C.SO; SO.HERE = C.ONS / 'fakeHERE'   # as sv2_official_FEA1.py: caches for both official splits
from serve_official import official_rows, score_rows, decode  # noqa: E402

DEST = ROOT / ('build/review_20260924_newworld/work/onset_site/'
               'onset_component_submission_C3_onset_sitehazard_minev10_umin0.01_exclD7_I210_W.parquet')


def main():
    if DEST.exists():
        raise SystemExit(f'refusing to overwrite {DEST}')
    comp = pd.read_parquet(ROOT / 'build/strongest_candidate_20260922/onset/production/onset_component.parquet')
    comp['link_id'] = comp.link_id.astype(str)
    choice = {}
    for split in ('validation', 'private'):
        offi = official_rows(split)
        offi = offi[offi.panel.isin(C.MS)].reset_index(drop=True)
        ch = C.choices(score_rows(offi, C.UMODELS), 'u')
        assert not set(ch) & set(choice)
        choice.update(ch)
        print(split, 'site choices', len(ch), flush=True)
    new = decode(comp, choice, C.TPL)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(DEST, index=False)
    m = comp.merge(new[['window_id', 'link_id', 'queue_pred']], on=['window_id', 'link_id'], suffixes=('', '_new'))
    ch = m[m.queue_pred != m.queue_pred_new]
    print('wrote', DEST, 'choices', len(choice), 'changed cells', len(ch), 'windows', ch.window_id.nunique(), flush=True)


if __name__ == '__main__':
    main()
