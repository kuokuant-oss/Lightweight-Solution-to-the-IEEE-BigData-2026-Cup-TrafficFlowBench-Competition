"""Ongoing ensemble of no-ID/no-prior heads (with/without geometry), several seeds.
Trains missing members, calibrates ONE threshold on February (in-world), scores pseudo-val, serves real val/private."""
# Training record of the A ensemble, the location-free ongoing model (A25: it
# enters the final ongoing mix with weight 0.25). In the reproduction
# ogens_serve.py imports this file for MEMBERS only; main() is not run.
import os
for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[n] = '8'
import sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import og_exp as X
S = X.S; Q = X.Q
# ID: link/panel code features removed from the P1 set. GEOM: queue-geometry
# features of geom.py.
ID = ['link_code', 'panel_code']
GEOM = ['g_gap_km', 'g_gap_links', 'g_tail_vel', 'g_rh_w', 'g_eta_rh', 'g_eta_vel', 'g_block_len_km', 'g_block_links', 'g_head_age',
        'g_in_block', 'g_pos_in_block_km', 'g_clear_eta_rh', 'g_n_blocks', 'g_link_age', 'g_q_up', 'g_q_in', 'g_tail_age',
        'g_eta_rh_m_step', 'g_eta_vel_m_step', 'g_clear_m_step']
V2 = [f for f in X.FS0 if f not in ID + ['queue_prior']]
# MEMBERS: (name, feature list, seed). Three seeds of V2 (P1 without codes and
# queue prior) and three seeds of V2 plus the geometry features.
MEMBERS = [('V2_noID_noPrior', V2, 20260922), ('V2s2', V2, 20260929), ('V2s3', V2, 20260936),
           ('V6_V2geom', V2 + GEOM, 20260922), ('V6s2', V2 + GEOM, 20260929), ('V6s3', V2 + GEOM, 20260936)]


def main():
    """Trains missing members, calibrates one threshold and writes probabilities.

    The six member probabilities are averaged. One threshold is calibrated on
    February 2031 and stored with the pseudo-window scores in
    ogens_result.json. Writes ogens_<tag>_prob.parquet (window_id, timestamp,
    link_id, prob) for the pseudo windows (pv) and the official validation
    and private windows (realval, realpriv).
    """
    d = X.load_train()
    g = pd.read_parquet(HERE / 'geom_train_queue_ongoing.parquet'); d = d.merge(g, on=['window_id', 'link_id', 'step'], how='left')
    tr = d[d.period.eq('fit') & d._eligible]; es = d[d.period.eq('es') & d._eligible]; ca = d[d.period.eq('cal')].reset_index(drop=True)
    sets = {}
    for tag in ['pv', 'realval', 'realpriv']:
        s = pd.read_parquet(HERE / f'serve_{tag}.parquet'); s = s[s.condition.eq('queue_ongoing')].reset_index(drop=True)
        s = s.merge(pd.read_parquet(HERE / f'geom_serve_{tag}.parquet'), on=['window_id', 'link_id', 'step'], how='left')
        sets[tag] = s
    probs = {k: [] for k in ['cal', 'pv', 'realval', 'realpriv']}
    for name, FS, seed in MEMBERS:
        mf = HERE / f'og_{name}.txt'
        if mf.exists():
            m = lgb.Booster(model_file=str(mf))
        else:
            P = dict(X.PARAMS, seed=seed, num_threads=8)
            dt = lgb.Dataset(tr[FS], tr.label, weight=X.weights(tr)); dv = lgb.Dataset(es[FS], es.label, weight=X.weights(es), reference=dt)
            m = lgb.train(P, dt, num_boost_round=1000, valid_sets=[dv], callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            m.save_model(str(mf))
        probs['cal'].append(Q.predict(m, ca, FS, False))
        for tag in ['pv', 'realval', 'realpriv']: probs[tag].append(Q.predict(m, sets[tag], FS, False))
        print('member', name, m.current_iteration(), flush=True); gc.collect()
    cp = np.mean(probs['cal'], axis=0); thr, grid, pans = Q.calibrate(ca, cp)
    s_in, _ = X.inworld_score(ca, cp, thr)
    res = dict(members=[m[0] for m in MEMBERS], thr=thr, inworld_feb=round(s_in, 4))
    for tag in ['pv', 'realval', 'realpriv']:
        p = np.mean(probs[tag], axis=0); z = sets[tag][['window_id', 'timestamp', 'link_id']].copy(); z['prob'] = p
        z.to_parquet(HERE / f'ogens_{tag}_prob.parquet', index=False)
        if tag == 'pv':
            z['queue_pred'] = (p >= thr).astype(int); fr = S.score(z, 'validation'); fr = fr[fr.condition == 'queue_ongoing']
            res['pv_obs'] = round(S.agg(fr, 'iou')['queue_ongoing'], 4); res['pv_clean'] = round(S.agg(fr, 'iou_clean')['queue_ongoing'], 4)
            for t in [0.4, 0.5, 0.6]:
                z['queue_pred'] = (p >= t).astype(int); fr = S.score(z, 'validation'); fr = fr[fr.condition == 'queue_ongoing']
                res[f'pv_obs_{t}'] = round(S.agg(fr, 'iou')['queue_ongoing'], 4)
    json.dump(res, open(HERE / 'ogens_result.json', 'w'), indent=1); print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
