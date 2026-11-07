"""Ongoing variants: P1 recipe with feature subsets; evaluate in-world (Feb) and new-world (val pseudo)."""
import os
for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[n] = '6'
import sys, json, importlib.util, gc, glob
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 'fakeval'))
import score as S
from src.tfb import release as R
spec = importlib.util.spec_from_file_location('Q', ROOT / 'scripts/strongest_20260922_ongoing.py'); Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
man = json.loads((ROOT / 'build/review_20260923/ongoing_P1/manifest.json').read_text())
FS0 = man['features']; fitdates = man['fit_dates']
PARAMS = dict(Q.NEW_PARAMS, num_threads=6)


def weights(d):
    nc = d.groupby('window_id').label.transform('size').to_numpy(float)
    pwin = d[['panel', 'window_id']].drop_duplicates().groupby('panel').size()
    w = 1 / (nc * d.panel.map(pwin).to_numpy()); return w / w.mean()


def load_train():
    cache = HERE / 'og_train_all.parquet'
    if cache.exists(): return pd.read_parquet(cache)
    d = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(HERE / 'queue_ongoing_*.parquet')))], ignore_index=True)
    Q.identities(d)
    d['period'] = np.where(d.date <= '2030-12-31', 'fit', np.where(d.date <= '2031-01-31', 'es', 'cal'))
    Q.priors([d], fitdates, sorted(d.panel.unique()))
    d.attrs = {}; d.to_parquet(cache, index=False); return d


def inworld_score(ca, p, thr):
    fr = Q.frame(ca, p, thr)
    fr['condition'] = 'queue_ongoing'
    fr = fr.rename(columns={'IoU': 'iou'})
    return S.agg(fr, 'iou')['queue_ongoing'], fr


def run(name, FS, d, pv, extra_train=None):
    PARAMS['seed'] = 20260922 + (7 if 's2' in name else 0)
    tr = d[d.period.eq('fit') & d._eligible]; es = d[d.period.eq('es') & d._eligible]; ca = d[d.period.eq('cal')].reset_index(drop=True)
    if extra_train is not None: tr = pd.concat([tr, extra_train], ignore_index=True)
    cats = [x for x in ('panel_code', 'link_code') if x in FS]
    dt = lgb.Dataset(tr[FS], tr.label, weight=weights(tr), categorical_feature=cats)
    dv = lgb.Dataset(es[FS], es.label, weight=weights(es), categorical_feature=cats, reference=dt)
    m = lgb.train(PARAMS, dt, num_boost_round=1000, valid_sets=[dv], callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    cp = Q.predict(m, ca, FS, False); thr, grid, pans = Q.calibrate(ca, cp)
    s_in, _ = inworld_score(ca, cp, thr)
    pp = Q.predict(m, pv, FS, False)
    pred = pv[['window_id', 'timestamp', 'link_id']].copy(); pred['queue_pred'] = (pp >= thr).astype(int)
    fr = S.score(pred, 'validation'); fr = fr[fr.condition == 'queue_ongoing']
    a = S.agg(fr, 'iou'); b = S.agg(fr, 'iou_clean')
    # also fixed thr 0.5 on val pseudo
    pred['queue_pred'] = (pp >= 0.5).astype(int); fr5 = S.score(pred, 'validation'); fr5 = fr5[fr5.condition == 'queue_ongoing']
    res = dict(name=name, trees=m.current_iteration(), thr=thr, inworld_feb=round(s_in, 4), pv_obs=round(a['queue_ongoing'], 4), pv_clean=round(b['queue_ongoing'], 4),
               pv_obs_thr05=round(S.agg(fr5, 'iou')['queue_ongoing'], 4), nfeat=len(FS))
    print(json.dumps(res), flush=True)
    m.save_model(str(HERE / f'og_{name}.txt'))
    pv_out = pv[['window_id', 'timestamp', 'link_id']].copy(); pv_out['prob'] = pp; pv_out.to_parquet(HERE / f'og_{name}_pvprob.parquet', index=False)
    return m, thr, res


if __name__ == '__main__':
    d = load_train()
    pv = pd.read_parquet(HERE / 'serve_pv.parquet'); pv = pv[pv.condition.eq('queue_ongoing')].reset_index(drop=True)
    CLOCK = ['tod_min', 'tod_sin', 'tod_cos', 'origin_tod_min']
    ID = ['link_code', 'panel_code']
    variants = {
        'V0_P1': FS0,
        'V1_noID': [f for f in FS0 if f not in ID],
        'V2_noID_noPrior': [f for f in FS0 if f not in ID + ['queue_prior']],
        'V3_noID_noPrior_noClock': [f for f in FS0 if f not in ID + ['queue_prior'] + CLOCK],
        'V4_noMem': [f for f in FS0 if f not in ID + ['queue_prior'] + CLOCK + ['milepost']],
    }
    GEOM = ['g_gap_km', 'g_gap_links', 'g_tail_vel', 'g_rh_w', 'g_eta_rh', 'g_eta_vel', 'g_block_len_km', 'g_block_links', 'g_head_age',
            'g_in_block', 'g_pos_in_block_km', 'g_clear_eta_rh', 'g_n_blocks', 'g_link_age', 'g_q_up', 'g_q_in', 'g_tail_age',
            'g_eta_rh_m_step', 'g_eta_vel_m_step', 'g_clear_m_step']
    V2 = variants['V2_noID_noPrior']
    variants.update({'V5_noPrior': [f for f in FS0 if f != 'queue_prior'], 'V2s2': V2, 'V6_V2geom': V2 + GEOM, 'V7_P1geom': FS0 + GEOM})
    want = sys.argv[1:] or list(variants)
    if any('geom' in k for k in want):
        g = pd.read_parquet(HERE / 'geom_train_queue_ongoing.parquet')
        d = d.merge(g, on=['window_id', 'link_id', 'step'], how='left')
        gp = pd.read_parquet(HERE / 'geom_serve_pv.parquet')
        pv = pv.merge(gp, on=['window_id', 'link_id', 'step'], how='left')
        print('geom merged', d[GEOM[0]].notna().mean(), pv[GEOM[0]].notna().mean(), flush=True)
    out = []
    for k in want:
        _, _, r = run(k, variants[k], d, pv); out.append(r)
        gc.collect()
    pd.DataFrame(out).to_csv(HERE / f'og_results_{"_".join(want)[:60]}.csv', index=False)
