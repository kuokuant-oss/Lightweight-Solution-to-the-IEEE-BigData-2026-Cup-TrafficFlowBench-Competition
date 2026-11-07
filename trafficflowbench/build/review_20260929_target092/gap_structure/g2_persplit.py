"""G2: per-split transductive fitting (the T1 lesson) of the in-gap ramp information, strict odd/even halves.
March (validation) / April (private) pseudo 90-min holes of the HMP50 harness; labels = released visible speeds.
RR  = LightGBM residual on top of the strict HMP50 (pool50; OOF for both halves), features = G2v2 boundary feats +
      ramp feats + pool50 + HM pooled p; fit on one half of days, predict the other.  RR05 = half strength.
PRP = per-split ramp-aware queue-probability head (binary), group-pooled over labelled cells, inside HMP50 structure.
Baseline for comparison: 'resid' = frozen Jan-fit correction at lam .5 (the D1 candidate)."""
# Runs in pipeline/task1.sh step 9 once per split, after c4_phead_transfer.py.
# Its g2_<split>_feats.parquet (pseudo-hole rows with all features and the
# queue label yq) is the training table of the PR heads in d2_official_prp.py.
# "Transductive": fit on the released month itself, which Task 1 allows.
# Usage: g2_persplit.py <validation|private>
# Outputs: g2_<split>_feats.parquet, g2_<split>_preds.parquet and
# g2_<split>_results.json.
import os, sys, json, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ramp_feats as RF
from b2_ramp_resid import SP, FLOOR, CAP
from c1_transfer import rel, R2, R3, KEY, PANELS, FMAP
PARR = dict(objective='l2', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=10.0, num_threads=2, seed=9, verbose=-1)
PARB = dict(PARR, objective='binary')
NR = 400


def build(split):
    """Returns the feature table of one split's pseudo holes (cached in g2_<split>_feats.parquet).

    Starts from c4_<split>_preds.parquet and adds the G2v2 gap features (SP)
    and the ramp features of the split's own released ramp counts.
    """
    p = HERE / f'g2_{split}_feats.parquet'
    if p.exists(): return pd.read_parquet(p)
    src = R2 if split == 'validation' else R3
    c4 = pd.read_parquet(HERE / f'c4_{split}_preds.parquet')
    parts = []
    for panel in PANELS:
        ph = pd.read_parquet(src / f'ph_{split}_{panel}.parquet', columns=KEY + ['f_' + c for c in SP if c not in FMAP] + ['ch_v', 'ch_ql'])
        x = c4[c4.panel == panel].merge(ph, on=KEY, how='inner', validate='one_to_one')
        for c in SP:
            if c not in FMAP: x[c] = x['f_' + c].astype(np.float32)
        x = x.drop(columns=['f_' + c for c in SP if c not in FMAP])
        x['p_v'] = x.ch_v; x['p_ql'] = x.ch_ql; x = x.reset_index(drop=True)
        x = RF.attach(x, split); x = RF.attach_near(x, split)
        parts.append(x); print('feats', panel, len(x), flush=True); del ph; gc.collect()
    t = pd.concat(parts, ignore_index=True); t.to_parquet(p, index=False); return t


def main():
    """Fits RR and PR on odd/even halves of one split and scores the variants against pool50."""
    split = sys.argv[1]
    t = build(split)
    feats = SP + RF.RAMP_FEATS + RF.NEAR_FEATS + ['pool50', 'pp']
    t['r'] = t.y_v - t.pool50
    t['c_RR'] = np.nan; t['p_PR'] = np.nan
    odd = (t.day % 2 == 1).to_numpy()
    imps = {}
    for fit_odd in (True, False):
        tr = odd == fit_odd; te = ~tr
        m = lgb.train(PARR, lgb.Dataset(t.loc[tr, feats], t.loc[tr, 'r']), NR)
        t.loc[te, 'c_RR'] = m.predict(t.loc[te, feats], num_threads=2)
        imp = pd.Series(m.feature_importance('gain'), index=feats); imps[f'RR_fitodd{int(fit_odd)}'] = (imp / imp.sum()).sort_values(ascending=False).round(4).head(10).to_dict()
        mb = lgb.train(PARB, lgb.Dataset(t.loc[tr, feats], t.loc[tr, 'yq']), NR)
        t.loc[te, 'p_PR'] = mb.predict(t.loc[te, feats], num_threads=2)
        print('fit half', fit_odd, flush=True); del m, mb; gc.collect()
    t['p_PR_pool'] = t.groupby(['panel', 'date', 'origin', 'group', 'step_i']).p_PR.transform('mean')
    t['v_resid'] = np.clip(t.pool50 + 0.5 * t.c_sp_M2, FLOOR, CAP)
    t['v_RR1'] = np.clip(t.pool50 + t.c_RR, FLOOR, CAP); t['v_RR05'] = np.clip(t.pool50 + 0.5 * t.c_RR, FLOOR, CAP)
    t['v_PRP'] = 0.5 * t.t1 + 0.5 * np.clip((1 - t.p_PR_pool) * t.fv + t.p_PR_pool * t.qv, FLOOR, CAP)
    t['v_PRP_plus'] = np.clip(t.v_PRP + 0.5 * t.c_sp_M2, FLOOR, CAP)
    from f1_phead import ll
    out = {'ll_pp': ll(t.yq, t.pp), 'll_PR': ll(t.yq, t.p_PR), 'll_PR_pool': ll(t.yq, t.p_PR_pool), 'importance': imps}
    print({k: v for k, v in out.items() if k.startswith('ll')}, flush=True)
    for nm in ('resid', 'RR1', 'RR05', 'PRP', 'PRP_plus'):
        out[nm] = rel(t, 'y_v', 'pool50', 'v_' + nm)
        print(split, nm, json.dumps({k: out[nm][k] for k in ('rmse1', 'rel', 'ci', 'onset', 'ongoing', 'odd', 'even', 'early12', 'panels_improved')}), flush=True)
    json.dump(out, open(HERE / f'g2_{split}_results.json', 'w'), indent=1)
    t[KEY + ['y_v', 'day', 'cond', 'pool50', 'c_RR', 'p_PR', 'p_PR_pool', 'c_sp_M2']].to_parquet(HERE / f'g2_{split}_preds.parquet', index=False)


if __name__ == '__main__':
    main()
