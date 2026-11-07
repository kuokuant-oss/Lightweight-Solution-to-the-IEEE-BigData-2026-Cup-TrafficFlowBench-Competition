"""Serve the frozen 0922 onset head (M1) on the released validation/private onset windows.

This is the serving half of `scripts/145_onset_prior_forward.py`, case `final`, arm `feature_hazard`
(the block from `serve=[]` to `final_{arm}_serving.parquet`), run with the saved model instead of a
refit. Nothing is trained:
- model file, feature list, 145's own threshold, prior dates and link codes come from
  `final_feature_hazard_manifest.json`;
- the conditional-hazard tables are rebuilt with 145's own `Hazard` class from the train feature table
  `onset_features.parquet` over the same fit dates (2030-06-01..12-31). Serving dates are never fit
  dates, so 145's leave-one-day-out term is zero, as in the original run;
- features come from `task2infer.features_for_split` exactly as 145 called it: the window's released
  history (T-60..T-5) plus the origin stamp T of the masked layer, and the cached train queue prior.

Writes `build/final_sprint_review_20260922/prior/final_feature_hazard_serving.parquet` (refuses to
overwrite). Run it through `bin/tfb-run <run> pipeline/onset_head_serve.py`.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(os.environ['TFB_REPO_ROOT'])
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location('prior145', ROOT / 'scripts/145_onset_prior_forward.py')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
TI = M.TI
ARM = 'feature_hazard'


def main():
    manifest = json.loads((M.OUT / f'final_{ARM}_manifest.json').read_text())
    dest = M.OUT / f'final_{ARM}_serving.parquet'
    if dest.exists():
        raise SystemExit(f'refusing to overwrite {dest}')
    # Only the columns the Hazard tables and the link axis read.
    d = pd.read_parquet(M.OUT / 'onset_features.parquet',
                        columns=['panel', 'date', 'link_id', 'label', '_link_idx', '_tod_idx'])
    panels = sorted(d.panel.unique())
    linkmap = {}
    for panel in panels:
        for link in sorted(d.loc[d.panel.eq(panel), 'link_id'].astype(str).unique()):
            linkmap[(panel, link)] = len(linkmap)
    assert [dict(panel=p, link=l, code=c) for (p, l), c in linkmap.items()] == manifest['link_codes'], \
        'link codes differ from the model manifest'
    fitdates, fs, thr = manifest['prior_dates'], manifest['features'], manifest['threshold']
    assert manifest['arm'] == ARM and fitdates[-1] == '2030-12-31' and manifest['refit_after_calibration'] is False
    hazard = M.Hazard(d, fitdates)
    model = lgb.Booster(model_file=str(M.OUT / manifest['model']))
    assert model.feature_name() == fs, 'model features differ from the manifest'
    arm = ARM
    # ---- from 145 main(), case 'final' (unchanged apart from indentation) ----
    serve = []
    for split in ['validation', 'private']:
        for panel in panels:
            s = TI.features_for_split(panel, split, oblique=True, dispersion=True, clock=True, origin_obs=True, prior_dates=tuple(fitdates))
            s = s[s.condition.eq('queue_onset') & s.step.eq(6)].copy().reset_index(drop=True)
            # Serving's explicit cell identities map to training's physical link axis.
            axis = d[d.panel.eq(panel)].drop_duplicates('link_id').set_index('link_id')._link_idx.to_dict()
            s['_link_idx'] = s.link_id.map(axis).astype(int)
            ts = pd.to_datetime(s.timestamp, utc=True); s['_tod_idx'] = ts.dt.hour * 12 + ts.dt.minute // 5
            s['date'] = ts.dt.strftime('%Y-%m-%d'); s['split'] = split
            M.identity(s, panels, linkmap); hazard.apply(s)
            sp = M.predictions(model, s, fs, arm)
            serve.append(s[['window_id', 'timestamp', 'link_id', 'panel', 'split', 'step', 'queue_prior']].assign(prob=sp, queue_pred=M.decode(s, sp, thr).astype(int)))
            print('served', split, panel, len(s), flush=True)
    pd.concat(serve, ignore_index=True).to_parquet(dest, index=False)
    print('wrote', dest, flush=True)


if __name__ == '__main__':
    main()
