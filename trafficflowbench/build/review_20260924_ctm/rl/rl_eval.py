"""RL (and RLB) residual LightGBM on top of C1 -- paired in-world evaluation.
  --mode crossfit --month M          : 2-fold cross-fit by day halves (days 0-13 <-> 14-27) on month M
  --mode transfer --fit M1 --test M2 : fit on all days of M1, predict M2
Training panels: the 8 queue panels (no IDs).  D12_I405_N/S are scored as supplementary (predicted, never trained on).
Recipe = floor_audit/s15_longrange.py: L2, lr .05, 63 leaves, min_data_in_leaf 300, ff .8, bagging .8/1, l2 10, 400 rounds,
free cells (C1 v/vf >= .85) subsampled 30% with weight 1/.3.
Reports: MSE by stratum, per-panel dS_state / dS_LWR_A / dS_LWR_B / dS_FD, family-level summary (8 and 10 panels),
projected dLB = .35 dS_state + .15 dS_phys_B x [.73, .86], day bootstrap (200) of dLB_lo."""
import os, sys, argparse, time, json
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import rlfeat as RF, rleval as EL
P = dict(objective='l2', learning_rate=0.05, num_leaves=63, min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=10.0, num_threads=3, verbose=-1)
NROUND = int(os.environ.get('RL_NROUND', 400))
KSTAR = ['D7_I10_E', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
_ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
SEL = json.load(open(_ROOT / 'build/strongest_candidate_20260922/state/transfer/selection_before_held.json'))


def kstar(v, q, khat, lanes, alpha, weights):
    """= scripts/audit_strongest_state_20260922.apply(family='joint_aux')."""
    kk = np.maximum((1 - alpha) * q / v + alpha * khat, 0)
    wv = weights['speed_precision']; wq = weights['flow_lane_precision'] / np.square(lanes)
    vv = np.clip((wv * v + wq * kk * q) / (wv + wq * np.square(kk)), 6.0, 120.74999999999999)
    return vv, np.maximum(kk * vv, 50.0)


def featsets():
    fs = {'RL': list(RF.FEATS)}
    bf = HERE / 'rlb_feats.json'
    if bf.exists(): fs['RLB'] = list(RF.FEATS) + json.load(open(bf))
    return fs


def load_month(month, need_blk):
    parts = []
    for p in EL.PANELS10:
        d = pd.read_parquet(HERE / 'feats' / f'{p}_{month}.parquet')
        if need_blk:
            b = pd.read_parquet(HERE / 'feats' / f'blk_{p}_{month}.parquet'); assert len(b) == len(d)
            d = pd.concat([d, b], axis=1)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def fit(D, sub_mask, feats, seed=0):
    free = (D.p_r >= 0.85).to_numpy(); U = np.random.default_rng(seed).random(len(D))
    sub = sub_mask & (~free | (U < 0.3)); w = np.where(free, 1 / 0.3, 1.0)[sub]
    ms = {}
    for y in ('y_v', 'y_q'):
        ms[y] = lgb.train(P, lgb.Dataset(D.loc[sub, feats], D.loc[sub, y], weight=w), num_boost_round=NROUND)
    return ms


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['crossfit', 'transfer'], required=True)
    ap.add_argument('--month'); ap.add_argument('--fit'); ap.add_argument('--test'); ap.add_argument('--variants', nargs='+', default=['RL'])
    ap.add_argument('--boot', type=int, default=200); ap.add_argument('--kstar', action='store_true')
    a = ap.parse_args(); t0 = time.time()
    FS = featsets(); need_blk = any(v != 'RL' for v in a.variants)
    train8 = lambda D: D.panel_i.isin(range(8)).to_numpy()
    if a.mode == 'crossfit':
        tag = f'crossfit_{a.month}'; month = a.month
        D = load_month(month, need_blk); oof = {v: {y: np.zeros(len(D)) for y in ('y_v', 'y_q')} for v in a.variants}
        for fold in (0, 1):
            trm = ((D.day < 14) == (fold == 0)).to_numpy(); tem = ~trm
            for v in a.variants:
                ms = fit(D, trm & train8(D), FS[v])
                for y, m in ms.items(): oof[v][y][tem] = m.predict(D.loc[tem, FS[v]], num_threads=3)
            print('fold', fold, round(time.time() - t0), flush=True)
    else:
        tag = f'transfer_{a.fit}_to_{a.test}'; month = a.test
        Df = load_month(a.fit, need_blk); D = load_month(a.test, need_blk); oof = {}
        for v in a.variants:
            ms = fit(Df, train8(Df), FS[v]); oof[v] = {y: m.predict(D[FS[v]], num_threads=3) for y, m in ms.items()}
            md = HERE / 'models' / f'{v}_fit{a.fit}'
            if not (md / 'y_v.txt').exists():   # never overwrite a model another step already used
                md.mkdir(parents=True, exist_ok=True)
                for y, m in ms.items(): m.save_model(str(md / f'{y}.txt'))
                json.dump(FS[v], open(md / 'feats.json', 'w'))
            if v == 'RL':
                for y, m in ms.items():
                    imp = pd.Series(m.feature_importance('gain'), index=FS[v]); imp = imp / imp.sum()
                    print(v, y, 'lr_* gain share', round(float(imp[[c for c in FS[v] if c.startswith('lr_')]].sum()), 3), flush=True)
        del Df
        print('fit', round(time.time() - t0), flush=True)
    if a.kstar: tag += '_kstar'
    if len(a.variants) > 1 or a.variants != ['RL']: tag += '_' + '_'.join(a.variants)
    np.savez(HERE / 'results' / f'{tag}_oof.npz', **{f'{v}_{y}': oof[v][y] for v in oof for y in oof[v]})
    # ---- strata MSE (non-hole targets; hole stratum filled from the evaluator accumulators below)
    band = np.where(D.p_r >= 0.85, 'free', np.where(D.p_r >= 0.6, 'mid', 'cong'))
    rows = []
    for scope, pm in (('8p', train8(D)), ('10p', np.ones(len(D), bool))):
        for b in ('free', 'mid', 'cong', 'nonhole'):
            mm = pm & ((band == b) if b != 'nonhole' else True)
            r = dict(scope=scope, stratum=b, n=int(mm.sum()), C1_mse_v=float(np.mean(D.y_v[mm] ** 2)), C1_mse_q=float(np.mean(D.y_q[mm] ** 2)))
            for v in oof:
                r[f'{v}_mse_v'] = float(np.mean((D.y_v - oof[v]['y_v'])[mm] ** 2)); r[f'{v}_mse_q'] = float(np.mean((D.y_q - oof[v]['y_q'])[mm] ** 2))
            rows.append(r)
    # ---- paired official-form evaluation
    acc = []
    for pi, panel in enumerate(EL.PANELS10):
        r = D[D.panel_i == pi]; idx = r.index.to_numpy(); r = r[['day', 'li', 'ci', 'p_v', 'p_q']].reset_index(drop=True)
        cands = {'C1': (None, None)}
        for v in oof: cands[v] = ((r.p_v + oof[v]['y_v'][idx]).to_numpy(), (r.p_q + oof[v]['y_q'][idx]).to_numpy())
        ndays = len(EL.load(panel, month)[1]['dates'])
        if a.kstar:   # production composition on the 5 k* corridors: K = k*(C1), <v>K = k*(<v>(C1)); identity elsewhere
            ln = np.maximum(EL.load(panel, month)[0]['ln'], 1.0)[r.li.to_numpy(int)]
            base = {'C1': (r.p_v.to_numpy(float), r.p_q.to_numpy(float))}; base.update({v: cands[v] for v in oof})
            if panel in KSTAR:
                kz = np.load(HERE / 'cache' / f'{panel}_{month}_khat.npz'); kh = np.concatenate([kz[f'd{d}'] for d in range(ndays)]).astype(float)
                assert len(kh) == len(r)
                for nm, (bv, bq) in base.items():
                    vv = np.clip(bv, 6.0, 120.75); qq = np.maximum(bq * ln, 50.0)
                    kv, kq = kstar(vv, qq, kh, ln, SEL['panel_alphas'][panel], SEL['weights'][panel])
                    cands['K' if nm == 'C1' else f'{nm}K'] = (kv, kq / ln)
            else:
                for nm, (bv, bq) in base.items(): cands['K' if nm == 'C1' else f'{nm}K'] = (bv, bq)
        out = EL.evaluate(panel, month, list(range(ndays)), cands, r)
        for o in out: o['panel'] = panel
        acc += out
    A = pd.DataFrame(acc); A.to_csv(HERE / 'results' / f'{tag}_acc.csv', index=False)
    gh = A[A.panel.isin(EL.PANELS)].groupby('cand')[['n_gap', 'sse_v_gap', 'sse_q_gap']].sum()
    hole = dict(scope='8p', stratum='hole', n=int(gh.loc['C1', 'n_gap']), C1_mse_v=float(gh.loc['C1', 'sse_v_gap'] / gh.loc['C1', 'n_gap']), C1_mse_q=float(gh.loc['C1', 'sse_q_gap'] / gh.loc['C1', 'n_gap']))
    for v in oof: hole[f'{v}_mse_v'] = float(gh.loc[v, 'sse_v_gap'] / gh.loc[v, 'n_gap']); hole[f'{v}_mse_q'] = float(gh.loc[v, 'sse_q_gap'] / gh.loc[v, 'n_gap'])
    rows.append(hole)
    ST = pd.DataFrame(rows); ST.to_csv(HERE / 'results' / f'{tag}_strata.csv', index=False)
    pd.set_option('display.width', 250); print(ST.round(3).to_string())
    res = {}
    for scope, pan in (('8p', EL.PANELS), ('10p', EL.PANELS10)):
        g, p, f = EL.summarize(acc, pan)
        print(scope); print(f.round(5).to_string())
        base = p[p.cand == 'C1'].set_index('panel')
        extra = [c for c in p.cand.unique() if c.endswith('K') and c != 'K'] if a.kstar else []
        pairs = [('C1', v) for v in oof] + ([('K', c) for c in extra] + [('C1', 'K')] if a.kstar else [])
        for b0, v in pairs:
            d = p[p.cand == v].set_index('panel')[['S_state', 'S_LWR_A', 'S_LWR_B', 'S_FD']] - p[p.cand == b0].set_index('panel')[['S_state', 'S_LWR_A', 'S_LWR_B', 'S_FD']]
            d.to_csv(HERE / 'results' / f'{tag}_{v}_vs_{b0}_perpanel_{scope}.csv')
            print(v, f'per-panel delta vs {b0}'); print(d.round(5).to_string())
            lb = {k: float(x) for k, x in EL.lb_delta(f, b0, v).items()}
            lb.update(n_pos_state=int((d.S_state > 0).sum()), n_pos_lwrB=int((d.S_LWR_B > 0).sum()), n_panels=len(d))
            res[f'{v}_vs_{b0}_{scope}'] = dict(lb, levels={c: {k: float(x) for k, x in f.loc[c].items()} for c in (b0, v)})
            print(v, scope, {k: (round(x, 5) if isinstance(x, float) else x) for k, x in lb.items()})
        if 'RLB' in oof and 'RL' in oof:
            lb = {k: float(x) for k, x in EL.lb_delta(f, 'RL', 'RLB').items()}; res[f'RLB_minus_RL_{scope}'] = lb; print('RLB-RL', scope, {k: round(x, 5) for k, x in lb.items()})
    # ---- day bootstrap (8 panels, resample day indices jointly across panels)
    if a.boot:
        rng = np.random.default_rng(3); B = []; nd = int(A.day.max()) + 1
        A8 = A[A.panel.isin(EL.PANELS)]; byday = {d: A8[A8.day == d] for d in range(nd)}
        for b in range(a.boot):
            dd = rng.choice(nd, nd, replace=True); sel = pd.concat([byday[d] for d in dd])
            _, _, fb = EL.summarize(sel.to_dict('records'))
            r = {}
            for v in oof: x = EL.lb_delta(fb, 'C1', v); r[f'{v}_dLB_lo'] = x['dLB_lo']; r[f'{v}_dS_state'] = x['dS_state']; r[f'{v}_dS_LWR_B'] = fb.loc[v, 'S_LWR_B'] - fb.loc['C1', 'S_LWR_B']
            if 'RLB' in oof and 'RL' in oof: r['RLB_minus_RL_dLB_lo'] = EL.lb_delta(fb, 'RL', 'RLB')['dLB_lo']
            if a.kstar:
                for v in oof: r[f'{v}K_vs_K_dLB_lo'] = EL.lb_delta(fb, 'K', f'{v}K')['dLB_lo']
            B.append(r)
        B = pd.DataFrame(B); q = B.describe(percentiles=[.025, .975]).loc[['mean', '2.5%', '97.5%']]
        print(q.round(5).to_string()); res['bootstrap_8p'] = {c: {k: float(x) for k, x in q[c].items()} for c in q.columns}
    res['strata'] = ST.to_dict('records'); res['secs'] = round(time.time() - t0)
    json.dump(res, open(HERE / 'results' / f'{tag}.json', 'w'), indent=1)
    print('secs', round(time.time() - t0))


if __name__ == '__main__':
    main()
