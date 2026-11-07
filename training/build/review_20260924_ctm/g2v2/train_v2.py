"""Train a gap model (speed + per-lane flow LightGBM, G2 hyper-parameters) under a fit/ES/test arrangement.
  --arr F : fit Jun-Dec 2030, early stopping Jan 2031, test Feb 2031   (the shipped G2 arrangement)
  --arr J : fit Jun-Nov 2030 + Feb 2031, early stopping Dec 2030, test Jan 2031
  --rows v2    : rows/<panel>_rows.parquet (v2 windows, G2 + v2 features; 35% subsample via u<0.35 for fit/ES)
  --rows g2old : build/review_20260924/work/g2/<panel>_rows.parquet (the shipped G2 table; Feb rows subsampled 35%
                 with a fixed RNG when Feb is a fit month)
  --feats g2 | v2 | v2-<drop,...> : G2's 51 features, or G2 + V2_FEATS (optionally minus a comma list)
Saves <out>/gap_speed.txt, gap_flow.txt, gap_features.json, train_info.json; dense test-month predictions on the
v2 rows go to <out>/test_preds.parquet (every eligible hole cell of every v2 window in the test month)."""
# Training record for the G2v2 hole model v2_F. The reproduction does not run this program;
# it loads the shipped models. The program reads row tables that are not shipped, and its
# paths resolve from its original place in the development repository (plus the hard-coded
# development root in `G2`, mapped by the path layer).
#
# Trains: as G2 (`build/review_20260924/work/g2_train.py`), the speed (km/h, `y_v`) and
#   per-lane flow (veh/h per lane, `y_ql`) of every link at each of the 18 steps of a Task 2
#   hole, with 22 more inputs (`v2feat.V2_FEATS`: window condition, selector inputs, block
#   states, queue templates, distance to the nearest queued link): 73 in all.
# Data (`--rows v2`): `rows/<corridor>_rows.parquet`, built by `data_v2.py --panel <corridor>`
#   (shipped) at the corrected replicated windows of the 8 Task 2 corridors; fit and
#   early-stopping rows use the fixed 35% sample `u < 0.35`, test rows are all kept.
# Settings: `P` with the command-line seed, learning rate and thread count (at most 4);
#   at most 3,000 rounds, early stopping after 100 rounds without gain on the stopping month.
# Shipped: `trafficflowbench/build/review_20260924_ctm/g2v2/models/v2_F/` (`gap_speed.txt`,
#   `gap_flow.txt`, `gap_features.json`) from `--arr F --rows v2 --feats v2 --seed 1 --lr 0.05`
#   (recorded in its `train_info.json`; 3,598,488 fit rows, 553,555 stopping rows; both heads
#   ran all 3,000 rounds). The model headers record 3 threads (`--threads 3`).
#   `--arr J` gives the unshipped sibling v2_J, whose January test predictions are the base
#   of the January hole speed correction (`gap_structure/b3_near.py`).
import os, sys, json, argparse, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from v2feat import V2_FEATS
# G2 model folder: its feature list, and its row tables for `--rows g2old`.
G2 = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924/work/g2')
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
G2FEATS = json.load(open(G2 / 'gap_features.json'))
# Arrangement -> (is fit month, is early-stopping month, test month), on "YYYY-MM" strings.
ARR = {'F': (lambda m: m <= '2030-12', lambda m: m == '2031-01', '2031-02'),
       'J': (lambda m: (m <= '2030-11') | (m == '2031-02'), lambda m: m == '2030-12', '2031-01')}
P = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=1)


def feat_list(spec):
    """Returns the input list for a `--feats` spec.

    Args:
        spec: "g2" (G2's 51 inputs), "v2" (G2 + V2_FEATS) or "v2-a,b,..." (v2 without the
            listed inputs). For the v2 specs the corridor code `panel_c` goes last.
    """
    if spec == 'g2': return list(G2FEATS)
    base = [f for f in G2FEATS if f != 'panel_c'] + list(V2_FEATS)
    if spec.startswith('v2-'):
        drop = set(spec[3:].split(',')); base = [f for f in base if f not in drop]
    return base + ['panel_c']


def load(rows, arr, feats, part):
    """Loads the hole rows of the 8 corridors for one part of an arrangement.

    Args:
        rows: "v2" (rows/ tables, 35% sample by `u`) or "g2old" (G2 tables; their February
            rows, stored in full, are sampled at 35% with rng seed 3).
        arr: Arrangement key of `ARR`, "F" or "J".
        feats: Input list from `feat_list`.
        part: "train" (fit and early-stopping months, sampled) or "test" (every row of the
            test month).

    Returns:
        DataFrame with the inputs, labels and keys, `month` ("YYYY-MM"), `panel` and the
        categorical corridor code `panel_c`.
    """
    fit_m, es_m, te_m = ARR[arr]
    cols = list(dict.fromkeys([f for f in feats if f != 'panel_c'] + ['date', 'y_v', 'y_ql', 'is_target', 'cond', 'origin', 'li', 'step']))
    out = []
    for p in PANELS:
        if rows == 'v2':
            d = pd.read_parquet(HERE / 'rows' / f'{p}_rows.parquet', columns=cols + ['u'])
            m = d.date.astype(str).str[:7]
            sel = ((fit_m(m) | es_m(m)) & (d.u < 0.35)) if part == 'train' else (m == te_m)
        else:
            d = pd.read_parquet(G2 / f'{p}_rows.parquet', columns=cols)
            m = d.date.astype(str).str[:7]
            rng = np.random.default_rng(3); u = rng.random(len(d))
            sub = np.where(m == '2031-02', u < 0.35, True)     # only Feb is stored at 100% in the G2 table
            sel = ((fit_m(m) | es_m(m)) & sub) if part == 'train' else (m == te_m)
        d = d[np.asarray(sel)].copy(); d['month'] = d.date.astype(str).str[:7]; d['panel'] = p
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    df['panel_c'] = pd.Categorical(df.panel, categories=PANELS)
    return df


def main():
    """Trains both heads for one arrangement, then scores the test month densely."""
    ap = argparse.ArgumentParser(); ap.add_argument('--arr', required=True, choices=['F', 'J'])
    ap.add_argument('--rows', default='v2', choices=['v2', 'g2old']); ap.add_argument('--feats', default='v2')
    ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--lr', type=float, default=0.05)
    a = ap.parse_args(); t0 = time.time(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    feats = feat_list(a.feats); fit_m, es_m, te_m = ARR[a.arr]
    df = load(a.rows, a.arr, feats, 'train')
    fit = df[fit_m(df.month)]; es = df[es_m(df.month)]
    print('rows fit/es', len(fit), len(es), 'features', len(feats), flush=True)
    info = dict(arr=a.arr, rows=a.rows, feats=a.feats, n_fit=len(fit), n_es=len(es), seed=a.seed, lr=a.lr)
    models = {}
    for name, y in (('speed', 'y_v'), ('flow', 'y_ql')):
        dtr = lgb.Dataset(fit[feats], fit[y], free_raw_data=True); dva = lgb.Dataset(es[feats], es[y], reference=dtr)
        m = lgb.train(dict(P, seed=a.seed, num_threads=min(a.threads, 4), learning_rate=a.lr), dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        m.save_model(str(OUT / f'gap_{name}.txt'), num_iteration=m.best_iteration); models[name] = m
        info[f'best_iter_{name}'] = m.best_iteration; print(name, 'best_iter', m.best_iteration, round(time.time() - t0), flush=True)
    json.dump(feats, open(OUT / 'gap_features.json', 'w'))
    del df, fit, es
    te = load('v2', a.arr, feats, 'test')      # dense test month on the v2 windows
    te['p_v'] = models['speed'].predict(te[feats], num_iteration=models['speed'].best_iteration).astype(np.float32)
    te['p_ql'] = models['flow'].predict(te[feats], num_iteration=models['flow'].best_iteration).astype(np.float32)
    rm = lambda x, y: float(np.sqrt(np.mean((x - y) ** 2)))
    tt = te[te.is_target]
    info['dense_test'] = dict(n=len(tt), rmse_v=rm(tt.p_v, tt.y_v), rmse_ql=rm(tt.p_ql, tt.y_ql))
    for c, g in tt.groupby('cond', observed=True): info['dense_test'][c] = dict(n=len(g), rmse_v=rm(g.p_v, g.y_v), rmse_ql=rm(g.p_ql, g.y_ql))
    te[['panel', 'date', 'origin', 'li', 'step', 'cond', 'is_target', 'y_v', 'y_ql', 'p_v', 'p_ql']].to_parquet(OUT / 'test_preds.parquet', index=False)
    info['secs'] = round(time.time() - t0, 1)
    json.dump(info, open(OUT / 'train_info.json', 'w'), indent=1)
    print(json.dumps(info['dense_test'], indent=1))


if __name__ == '__main__':
    main()
