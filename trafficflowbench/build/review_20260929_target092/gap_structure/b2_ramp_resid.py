"""B2: residual LightGBM trained on January (v2_J out-of-sample residuals), tested on the February official-like gaps.
M0 = boundary/base features only (control), M1 = + in-gap ramp features.  Placebo: M1 applied to Feb rows whose ramp
features come from another Feb date (same panel/origin/link/step).  Reported on Feb target cells against
(a) the clean v2_F base and (b) the served champion gap values (flow: ch_ql = G2v2; speed: HMP50)."""
import os, sys, json, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ramp_feats as RF
KEY = ['panel', 'date', 'origin', 'li', 'step_i']
FLOOR, CAP = 6.0, 120.74999999999999
PAR = dict(objective='l2', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=2, seed=5, verbose=-1)
NR = 500
FL = ['p_ql', 'step', 'w', 'lin_q', 'h_q', 'pq', 'sq', 'pq3', 'sq3', 'rq_pre', 'rq_post', 'rq_all_pre', 'rq_all_post',
      'queue_pre', 'queue_post', 'tod', 'lanes', 'capacity_vph', 'rank', 'cond_on', 'p_v']
SP = ['p_v', 'step', 'w', 'lin_v', 'h_v', 'pv', 'sv', 'pv3', 'sv3', 'rv_pre', 'rv_post', 'queue_pre', 'queue_post', 'tod',
      'free_speed_kmh', 'cond_on', 'bq_pre0', 'bq_post', 'tmpl_stat', 'nq_pre', 'nq_post', 'rq_pre', 'rq_post', 'p_ql']
PHYS = ['pq_x_cfr', 'sq_x_cfr', 'blend_x_cfr']


def add_phys(k):
    if 'step' not in k: k['step'] = k.step_i.astype(np.float32)
    ln = k.lanes.clip(lower=1); w = (k.step + 1) / 19.0
    k['pq_x_cfr'] = (k.pq / ln * k.cFR_pre).astype(np.float32)
    k['sq_x_cfr'] = (k.sq / ln * k.cFR_post).astype(np.float32)
    k['blend_x_cfr'] = ((1 - w) * k.pq_x_cfr.fillna(k.p_ql) + w * k.sq_x_cfr.fillna(k.p_ql)).astype(np.float32)
    return k


def rel(t, y, base, new):
    e0 = (t[base] - t[y]) ** 2; e1 = (t[new] - t[y]) ** 2
    g = pd.DataFrame(dict(k=t.panel + '|' + t.date, e0=e0, e1=e1)).groupby('k')[['e0', 'e1']].sum()
    rng = np.random.default_rng(0); ix = rng.integers(0, len(g), (2000, len(g)))
    out = dict(rmse0=float(np.sqrt(e0.mean())), rmse1=float(np.sqrt(e1.mean())), rel=float(e1.sum() / e0.sum() - 1),
               ci=np.percentile(g.e1.to_numpy()[ix].sum(1) / g.e0.to_numpy()[ix].sum(1) - 1, [2.5, 97.5]).round(5).tolist())
    for nm, m in (('onset', t.cond == 'queue_onset'), ('ongoing', t.cond == 'queue_ongoing'), ('mixed', t.lclass == 'mixed'),
                  ('nonmixed', t.lclass != 'mixed')):
        out[nm] = round(float(e1[m].sum() / e0[m].sum() - 1), 5)
    out['perpanel'] = {p: round(float(((q[new] - q[y]) ** 2).sum() / ((q[base] - q[y]) ** 2).sum() - 1), 5) for p, q in t.groupby('panel')}
    out['panels_improved'] = int(sum(v < 0 for v in out['perpanel'].values()))
    return out


def main():
    jan = add_phys(pd.read_parquet(HERE / 'b1_jan.parquet'))
    feb = add_phys(pd.read_parquet(HERE / 'b1_feb.parquet'))
    a1 = pd.read_parquet(HERE / 'a1_feb_rows.parquet', columns=KEY + ['serve_hmp50', 'ch_ql', 'lclass'])
    feb = feb.merge(a1, on=KEY, how='left', validate='one_to_one'); assert feb.serve_hmp50.notna().all()
    # placebo ramp features for Feb: other date, same panel/origin/li/step
    dates = sorted(feb.date.unique()); shift = {d: dates[(i + 3) % len(dates)] for i, d in enumerate(dates)}
    fp = feb[KEY].copy(); fp['date'] = fp.date.map(shift); fp = RF.attach(fp.reset_index(drop=True), 'train')
    febp = feb.copy()
    for c in RF.RAMP_FEATS: febp[c] = fp[c].to_numpy()
    febp = add_phys(febp)
    jan['r_q'] = jan.y_ql - jan.p_ql; jan['r_v'] = jan.y_v - jan.p_v
    ramp = RF.RAMP_FEATS
    res = {}; imps = {}
    specs = {'fl_M0': (FL, 'r_q'), 'fl_M1': (FL + ramp + PHYS, 'r_q'), 'sp_M0': (SP, 'r_v'), 'sp_M1': (SP + ramp, 'r_v')}
    for nm, (feats, tgt) in specs.items():
        ok = jan[tgt].notna()
        m = lgb.train(PAR, lgb.Dataset(jan.loc[ok, feats], jan.loc[ok, tgt]), NR)
        feb['c_' + nm] = m.predict(feb[feats], num_threads=2)
        if nm.endswith('M1'): feb['c_' + nm + 's'] = m.predict(febp[feats], num_threads=2)
        imp = pd.Series(m.feature_importance('gain'), index=feats); imps[nm] = (imp / imp.sum()).sort_values(ascending=False).round(4).head(12).to_dict()
        m.save_model(str(HERE / f'b2_{nm}.txt'))
        print('fit', nm, flush=True); del m; gc.collect()
    t = feb[feb.is_target & feb.y_v.notna() & feb.y_ql.notna()].copy()
    for nm in ('fl_M0', 'fl_M1', 'fl_M1s'):
        t[nm + '_clean'] = np.clip(t.p_ql + t['c_' + nm], 0, None); t[nm + '_served'] = np.clip(t.ch_ql + t['c_' + nm], 0, None)
        res[nm + '_clean'] = rel(t, 'y_ql', 'p_ql', nm + '_clean'); res[nm + '_served'] = rel(t, 'y_ql', 'ch_ql', nm + '_served')
    for nm in ('sp_M0', 'sp_M1', 'sp_M1s'):
        t[nm + '_clean'] = np.clip(t.p_v + t['c_' + nm], FLOOR, CAP)
        res[nm + '_clean'] = rel(t, 'y_v', 'p_v', nm + '_clean')
        for lam in (0.5, 1.0):
            c = f'{nm}_hmp_{lam}'; t[c] = np.clip(t.serve_hmp50 + lam * t['c_' + nm], FLOOR, CAP)
            res[c] = rel(t, 'y_v', 'serve_hmp50', c)
    for k_, v in res.items(): print(k_, json.dumps({kk: v[kk] for kk in ('rmse0', 'rmse1', 'rel', 'ci', 'onset', 'ongoing', 'mixed', 'panels_improved')}), flush=True)
    res['importance'] = imps
    print(json.dumps(imps, indent=0))
    json.dump(res, open(HERE / 'b2_results.json', 'w'), indent=1)
    keep = KEY + ['cond', 'lclass', 'is_target', 'y_v', 'y_ql', 'p_v', 'p_ql', 'serve_hmp50', 'ch_ql'] + [c for c in feb.columns if c.startswith('c_')]
    feb[keep].to_parquet(HERE / 'b2_feb_preds.parquet', index=False)


if __name__ == '__main__':
    main()
