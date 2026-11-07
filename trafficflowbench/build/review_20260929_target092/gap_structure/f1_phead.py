"""F1: structural integration - a ramp-aware queue-probability head replaces / joins the HMP50 probability.
Jan-fit (v2 rows, out of sample for Feb) binary LightGBM p_ramp(queue | G2v2 boundary feats + in-gap ramp feats).
Feb official-like test, HMP50 structure kept: speed = 0.5*T1 + 0.5*clip((1-p*)*fv + p*qv) with the HM heads fv/qv.
Pre-declared p* variants: R (p_ramp), RP (co-switch-group pooled p_ramp), MIX (0.5 pooled HM p + 0.5 pooled p_ramp).
Also the additive residual (b3 sp_M2, lam .5) on top of each, and log-loss / AUC of p vs the truth state."""
import os, sys, json, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ramp_feats as RF
from b2_ramp_resid import SP, rel, KEY, FLOOR, CAP
PARB = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=2, seed=5, verbose=-1)
NR = 500


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def main():
    feats = SP + RF.RAMP_FEATS + RF.NEAR_FEATS
    cols = list(dict.fromkeys(KEY + ['y_v', 'free_speed_kmh', 'step'] + feats))
    jan = pd.read_parquet(HERE / 'b3_jan.parquet', columns=[c for c in cols if c != 'step'])
    feb = pd.read_parquet(HERE / 'b3_feb.parquet', columns=[c for c in cols if c != 'step'] + ['is_target', 'cond'])
    for d in (jan, feb): d['step'] = d.step_i.astype(np.float32)
    jan = jan[jan.y_v.notna()]
    jan['yq'] = (jan.y_v <= 0.6 * jan.free_speed_kmh).astype(int)
    m = lgb.train(PARB, lgb.Dataset(jan[feats], jan.yq), NR); m.save_model(str(HERE / 'f1_phead_janfit.txt'))
    m0 = lgb.train(PARB, lgb.Dataset(jan[SP], jan.yq), NR); m0.save_model(str(HERE / 'f1_phead_noramp_janfit.txt'))
    feb['p_ramp'] = m.predict(feb[feats], num_threads=2); feb['p_noramp'] = m0.predict(feb[SP], num_threads=2)
    del jan; gc.collect()
    a1 = pd.read_parquet(HERE / 'a1_feb_rows.parquet', columns=KEY + ['serve_p', 'serve_pp', 'serve_fv', 'serve_qv', 'serve_t1', 'serve_group', 'serve_hmp50', 'lclass', 'yq'])
    c3 = pd.read_parquet(HERE / 'b3_feb_preds.parquet', columns=KEY + ['c_sp_M2'])
    k = feb.merge(a1, on=KEY, validate='one_to_one').merge(c3, on=KEY, validate='one_to_one')
    gk = ['panel', 'date', 'origin', 'serve_group', 'step_i']
    k['p_ramp_pool'] = k.groupby(gk).p_ramp.transform('mean')
    k['p_mix'] = 0.5 * k.serve_pp + 0.5 * k.p_ramp_pool
    out = {}
    ok = k.y_v.notna()
    for c in ('serve_p', 'serve_pp', 'p_noramp', 'p_ramp', 'p_ramp_pool', 'p_mix'):
        out['ll_' + c] = ll(k.yq[ok], k[c][ok]); out['auc_' + c] = auc(k.yq[ok], k[c][ok])
    print({kk: round(v, 5) for kk, v in out.items()}, flush=True)
    for nm, p in (('R', k.p_ramp), ('RP', k.p_ramp_pool), ('MIX', k.p_mix)):
        mix = np.clip((1 - p) * k.serve_fv + p * k.serve_qv, FLOOR, CAP)
        k['v_' + nm] = 0.5 * k.serve_t1 + 0.5 * mix
        k['v_' + nm + '_plus'] = np.clip(k['v_' + nm] + 0.5 * k.c_sp_M2, FLOOR, CAP)
    k['v_resid'] = np.clip(k.serve_hmp50 + 0.5 * k.c_sp_M2, FLOOR, CAP)
    t = k[k.is_target & k.y_v.notna()].copy(); t['y_ql'] = 0.0
    for nm in ('resid', 'R', 'RP', 'MIX', 'R_plus', 'RP_plus', 'MIX_plus'):
        out[nm] = rel(t, 'y_v', 'serve_hmp50', 'v_' + nm)
        print(nm, json.dumps({kk: out[nm][kk] for kk in ('rmse1', 'rel', 'ci', 'onset', 'ongoing', 'mixed', 'panels_improved')}), flush=True)
    json.dump(out, open(HERE / 'f1_results.json', 'w'), indent=1)
    k[KEY + ['p_ramp', 'p_ramp_pool', 'p_mix', 'v_R', 'v_RP', 'v_MIX', 'v_RP_plus', 'v_MIX_plus', 'v_resid']].to_parquet(HERE / 'f1_feb_preds.parquet', index=False)


if __name__ == '__main__':
    main()
