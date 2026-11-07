"""G2 train: one global LightGBM per channel for Task 1 cells inside a Task 2 gap.
fit = Jun-Dec 2030 gaps, early stopping = Jan 2031, test = Feb 2031 (never touched for any choice).
Reports test RMSE against simple predictors; saves models + feature order to g2/."""
# Training record for the G2 hole model. The reproduction does not run this program; it
# loads the shipped models. The program reads row tables that are not shipped, and its
# paths resolve from its original place in the development repository, whose layout
# `trafficflowbench/` mirrors.
#
# Trains: two LightGBM regressors over the 8 Task 2 corridors for every link at each of the
#   18 five-minute steps of a hole (the 90 minutes after a Task 2 origin T that the release
#   blanks): speed (km/h, `y_v`) and per-lane flow (veh/h per lane, `y_ql`).
# Data: `<out>/<corridor>_rows.parquet`, built by `g2_gap_data.py --panel <corridor>` (shipped):
#   holes cut at the replicated official windows of train days, history T-60 .. T-5 restored
#   from truth, 35% cell sample outside February. Fit = June-December 2030, early stopping =
#   January 2031, test = February 2031.
# Inputs: every column outside `NONFEAT` (51), including the corridor as the categorical
#   `panel_c`.
# Settings: `P` below (objective regression, learning rate .05, 127 leaves, min_data_in_leaf
#   200, feature and bagging fraction .8, lambda_l2 1, seed 1, 8 threads); at most 3,000
#   rounds, early stopping after 100 rounds without gain on January.
# Output in `<out>`: `gap_speed.txt`, `gap_flow.txt` (best iteration) and `gap_features.json`,
#   shipped under `trafficflowbench/build/review_20260924/work/g2/`; `g2_test_results.json`
#   and `g2_test_preds.parquet` (February test).
# Usage: `python g2_train.py [out_folder]`; the shipped models used the default folder `g2`.
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'g2')
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
df = pd.concat([pd.read_parquet(OUT / f'{p}_rows.parquet') for p in PANELS], ignore_index=True)
df['panel_c'] = pd.Categorical(df.panel, categories=PANELS)
# Keys, labels and flags; `is_target` marks Task 1 target cells, `is_g1` rows of the
# February windows listed in the `g1` window files.
NONFEAT = {'panel', 'date', 'origin', 'li', 'cond', 'regime', 'is_target', 'is_g1', 'y_v', 'y_q', 'y_ql'}
FEATS = [c for c in df.columns if c not in NONFEAT]
fit = df[df.date < '2031-01-01']; es = df[(df.date >= '2031-01-01') & (df.date < '2031-02-01')]; te = df[df.date >= '2031-02-01']
print('rows fit/es/test', len(fit), len(es), len(te), 'features', len(FEATS))
P = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8, seed=1)
res = {}; models = {}
for name, y in (('speed', 'y_v'), ('flow', 'y_ql')):
    dtr = lgb.Dataset(fit[FEATS], fit[y]); dva = lgb.Dataset(es[FEATS], es[y], reference=dtr)
    m = lgb.train(P, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    models[name] = m; m.save_model(str(OUT / f'gap_{name}.txt'))
    print(name, 'best_iter', m.best_iteration)
json.dump(FEATS, open(OUT / 'gap_features.json', 'w'))
te = te.copy()
te['gbm_v'] = models['speed'].predict(te[FEATS], num_iteration=models['speed'].best_iteration)
te['gbm_ql'] = models['flow'].predict(te[FEATS], num_iteration=models['flow'].best_iteration)
# Simple predictors from the row table, flow turned per lane: `lin_*` straight line across
# the hole, `h_*` weekday time-of-day profile, `hr_*` profile scaled by the ratio to the
# profile observed before and after the hole.
te['lin_ql'] = te.lin_q / te.lanes; te['hr_ql'] = te.hr_q / te.lanes; te['h_ql'] = te.h_q / te.lanes
rm = lambda a, b: float(np.sqrt(np.nanmean((a - b) ** 2)))
for sub, d in (('test_all', te), ('test_targets', te[te.is_target]), ('test_g1_targets', te[te.is_target & te.is_g1])):
    r = {'n': len(d)}
    for k in ('lin_v', 'h_v', 'gbm_v'): r[f'rmse_v_{k}'] = rm(d[k], d.y_v)
    for k in ('lin_ql', 'hr_ql', 'h_ql', 'gbm_ql'): r[f'rmse_ql_{k}'] = rm(d[k], d.y_ql)
    res[sub] = r
    print(sub, {k: round(v, 3) for k, v in r.items()})
# per panel on targets
pp = te[te.is_target].groupby('panel').apply(lambda d: pd.Series({'n': len(d), 'v_lin': rm(d.lin_v, d.y_v), 'v_gbm': rm(d.gbm_v, d.y_v),
                                                                 'ql_hr': rm(d.hr_ql, d.y_ql), 'ql_gbm': rm(d.gbm_ql, d.y_ql)}))
print(pp.round(2).to_string())
json.dump(res, open(OUT / 'g2_test_results.json', 'w'), indent=1)
te[['panel', 'date', 'origin', 'li', 'step', 'is_target', 'is_g1', 'y_v', 'y_ql', 'lin_v', 'lin_ql', 'hr_ql', 'gbm_v', 'gbm_ql']].to_parquet(OUT / 'g2_test_preds.parquet', index=False)
