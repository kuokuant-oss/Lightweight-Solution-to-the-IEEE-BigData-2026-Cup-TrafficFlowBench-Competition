"""Served recipe of the round-2 task1_trans FS correction, SPEED ONLY (one code path for evaluation and serving).
Feature set, derived columns and hyperparameters are copied verbatim from round2/task1_trans/eval_holes.py (fixed in
its pre-registration).  fit(rows) -> booster on r_v = y_v - ch_v; apply(booster, rows) -> clip(ch_v + dv, floor, cap)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path

ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
HERE = Path(__file__).resolve().parent
PILOT = ROOT / 'build/review_20260925/round2/task1_trans'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
KEEP_F = ['step', 'w', 'lin_v', 'lin_q', 'h_v', 'h_q', 'pv', 'pvd', 'pq', 'sv', 'svd', 'sq', 'pv3', 'sv3', 'rv_pre', 'rv_post',
          'rq_pre', 'rq_post', 'rq_all_pre', 'rq_all_post', 'queue_pre', 'queue_post', 'tod', 'lanes', 'free_speed_kmh',
          'capacity_vph', 'cond_on', 'prev_rej', 'n_rej', 'blk1', 'bsize', 'bq_pre0', 'bq_hist', 'bq_post', 'bq_postd',
          'tmpl_dyn', 'tmpl_dyn_age', 'tmpl_post', 'tmpl_post_d', 'tmpl_stat', 'dq_up_pre', 'dq_dn_pre', 'nq_pre',
          'dq_up_post', 'dq_dn_post', 'nq_post']
BASE = ['ch_v', 'ch_ql'] + ['f_' + k for k in KEEP_F]
SPLIT = ['s_v', 's_ql', 's_qf', 's_n', 's_vq', 's_tq', 's_bqf', 's_v_pre', 's_qf_pre', 's_bqf_pre', 's_v_post', 's_qf_post',
         's_bqf_post', 'd_sv', 'd_sq', 'd_tq', 'd_vq', 'an_pre', 'an_post', 'd_ch_s']
FEATS = BASE + SPLIT
PAR = dict(objective='l2', learning_rate=0.05, num_leaves=31, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10.0, num_threads=3, seed=7, verbose=-1)
NR = 300
SPEED_FLOOR, SPEED_CAP = 6.0, 120.74999999999999   # src/tfb/predict1.py


def derive(D):
    with np.errstate(all='ignore'):
        D['d_sv'] = D.s_v - D.f_h_v; D['d_sq'] = D.s_ql - D.f_h_q / D.f_lanes.clip(lower=1)
        D['d_tq'] = D.s_tq - D.f_tmpl_stat; D['d_vq'] = D.s_vq - D.f_tmpl_stat
        D['an_pre'] = D.f_pv3 / D.s_v_pre; D['an_post'] = D.f_sv3 / D.s_v_post; D['d_ch_s'] = D.ch_v - D.s_v
    return D


def load_ph(paths):
    D = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    derive(D)
    D['r_v'] = D.y_v - D.ch_v; D['r_q'] = D.y_ql - D.ch_ql
    D['half'] = np.where(D.day % 2 == 1, 'A', 'B')
    D['wid'] = D.panel + '|' + D.date + '|' + D.origin.astype(str)
    D['pd_'] = D.panel + '|' + D.date
    return D


def fit(tr, seed=7):
    return lgb.train(dict(PAR, seed=seed), lgb.Dataset(tr[FEATS], tr['r_v']), NR)


def correction(model, X):
    return model.predict(X[FEATS], num_threads=3)


def apply(model, X, base_col='ch_v'):
    """Corrected speed = clip(base + dv, floor, cap); base_col lets serving add dv to the parent's shipped value."""
    return np.clip(X[base_col].to_numpy() + correction(model, X), SPEED_FLOOR, SPEED_CAP)


def ldo(D, seed=7, clip=True):
    """Odd/even calendar-day halves: fit on one half, predict the other.  Returns corrected speed per row."""
    out = np.full(len(D), np.nan)
    for hf, ht in (('A', 'B'), ('B', 'A')):
        m = fit(D[D.half == hf], seed); ix = np.where(D.half == ht)[0]
        dv = correction(m, D.iloc[ix]); v = D.ch_v.to_numpy()[ix] + dv
        out[ix] = np.clip(v, SPEED_FLOOR, SPEED_CAP) if clip else v
    return out


def rel_table(D, pv, boot=True, B=400):
    """Relative hole speed MSE change vs the parent (ch_v) by half / condition / early days + panel-day cluster bootstrap."""
    yv = D.y_v.to_numpy(); e0 = (D.ch_v.to_numpy() - yv) ** 2; e1 = (pv - yv) ** 2
    res = {}
    groups = {'A': (D.half == 'A').to_numpy(), 'B': (D.half == 'B').to_numpy(), 'all': np.ones(len(D), bool),
              'queue_onset': (D.cond == 'queue_onset').to_numpy(), 'queue_ongoing': (D.cond == 'queue_ongoing').to_numpy(),
              'days<=12': (D.day <= 12).to_numpy(), 'days<=17': (D.day <= 17).to_numpy()}
    for k, m in groups.items():
        if m.sum() == 0: continue
        res[k] = dict(n=int(m.sum()), windows=int(D.wid[m].nunique()), mse_v_parent=float(e0[m].mean()), mse_v_new=float(e1[m].mean()),
                      rel_v=float(e1[m].mean() / e0[m].mean() - 1), rel_q=0.0)
    if boot:
        g = D.pd_.to_numpy(); u, inv = np.unique(g, return_inverse=True)
        a0 = np.bincount(inv, e0, len(u)); a1 = np.bincount(inv, e1, len(u)); rng = np.random.default_rng(0); b = []
        for _ in range(B):
            s = rng.integers(0, len(u), len(u)); b.append(a1[s].sum() / a0[s].sum() - 1)
        res['boot_panelday'] = dict(n_clusters=len(u), ci_v=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    pp = []
    for p in PANELS:
        for h in ('A', 'B'):
            m = ((D.panel == p) & (D.half == h)).to_numpy()
            if m.any(): pp.append(dict(panel=p, half=h, n=int(m.sum()), rel_v=float(e1[m].mean() / e0[m].mean() - 1)))
    res['perpanel'] = pp
    res['panelhalves_speed_improve'] = f"{sum(x['rel_v'] < 0 for x in pp)}/{len(pp)}"
    return res


def desc(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return dict(n=len(x), mean=float(x.mean()), abs_mean=float(np.abs(x).mean()), p01=float(np.percentile(x, 1)),
                p99=float(np.percentile(x, 99)), abs_p90=float(np.percentile(np.abs(x), 90)), abs_p99=float(np.percentile(np.abs(x), 99)))
