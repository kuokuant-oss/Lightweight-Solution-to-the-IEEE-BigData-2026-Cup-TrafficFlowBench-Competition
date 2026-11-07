"""Paired train-holdout harness for gap models (extends build/review_20260924/work/g3_eval.py).

Train month M with official-like hole windows:
  --windows v2      eval_windows_<panel>_<M>.csv  (corrected v2 selector, sequence starting on the 1st of M)
  --windows v2x     eval_windows_<panel>_<M>x.csv (extended: sequences started on days 1/8/15/22, merged; ext_windows.py)
  --windows legacy  build/review_20260924/work/g3/<panel>_<M>_windows.csv (Feb only; the G2 report's windows)
Per window: hole T+1..T+18 blank on all links/channels, window history T-12..T-1 restored from truth at eligible
cells and removed from the target set (validation template).  C0 = shipped v3 stack pipeline at every target
(computed once and cached); each model candidate = C0 with the in-hole targets replaced by that gap model
(profile and static template exclude month M).
Scores: S_state (official per-regime form over ALL targets incl. hole), local LWR proxy (rhs == dN_true) in
frame A (all eligible cells, truth outside targets) and frame B (targets + masked-layer-observed cells only, as
evallib.py), S_FD over the same frames, S_phys = (S_FD + 2 S_LWR)/3 per regime.
Outputs: res/<panel>_<M>_<windows>_days.csv (accumulators), res/<panel>_<M>_<windows>_cells.parquet (hole cells)."""
import os, sys, time, argparse, json
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
G2W = ROOT / 'build/review_20260924/work'; sys.path.insert(0, str(G2W))
from src.tfb import release as R, grid, baselines as B, physics, predict1 as P1, task1stack as TS, score1
from g2_gap_data import gap_features
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from v2feat import v2_features, selector_feats, template_from_hist, LG, HISTN
from data_v2 import block_ids
CACHE = Path(__file__).resolve().parent / 'c0_cache'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']


def load_windows(panel, month, kind):
    if kind in ('v2', 'v2x'):
        w = pd.read_csv(HERE / f'eval_windows_{panel}_{month}{"x" if kind == "v2x" else ""}.csv')
        w['gap_min'] = w.tmin - w.tprev_min
    else:
        w = pd.read_csv(G2W / 'g3' / f'{panel}_{month}_windows.csv').sort_values('tmin').reset_index(drop=True)
        w['gap_min'] = w.tmin.diff()
    return w


def build_cache(panel, month, kind, wins):
    """Grids with holes + C0 (stack) predictions, cached as npz."""
    ym = month.replace('_', '-')
    files = [p for p in P1.split_masked_files(panel, 'train') if month in p.name]
    dates = [p.name[len('synthetic_mainline_'):-len('.parquet')].replace('_', '-') for p in files]
    regime = {p.name: rg for rg in R.REGIME_ORDER for p in R.masked_files(panel, 'train', rg)}
    truth_by = {p.name: p for p in R.truth_files(panel, 'train')}
    grids, gaps, real_target, wl = [], [], [], []
    for p, d in zip(files, dates):
        g = grid.build(panel, truth_by[p.name], p)
        rt = g.target.copy(); gm = np.zeros(rt.shape, bool); ws = []
        for _, w in wins[wins.date == d].iterrows():
            T = int(w.origin); hs = slice(max(T - HISTN, 0), T)
            for fld in ('speed_kmh', 'flow_vph', 'occupancy'):
                X = g.value[fld]; X[:, hs] = np.where(g.eligible[:, hs] & np.isnan(X[:, hs]), g.truth[fld][:, hs], X[:, hs])
            rt[:, hs] = False
            gm[:, T + 1:min(T + 1 + LG, gm.shape[1])] = True; ws.append(T + 1)
        for fld in ('speed_kmh', 'flow_vph', 'occupancy'):
            X = g.value[fld].copy(); X[gm] = np.nan; g.value[fld] = X
        g.blanked = (np.isnan(g.value['speed_kmh']) & np.isnan(g.value['flow_vph']))
        grids.append(g); gaps.append(gm); real_target.append(rt); wl.append(ws)
    W = P1.load_weights(); models = TS.load(); hist = P1.historical_mean(panel, P1.dev_train_files(panel))
    lrtc = {f: B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f], nan=0.0) for g in grids], axis=2), rho=P1.LRTC_KNOBS[f][0], theta=P1.LRTC_KNOBS[f][1]) for f in P1.FIELDS}
    links = [str(x) for x in grids[0].links]
    par = physics.link_params(panel, links)
    att, valid = physics.ramp_index(panel, 'train', official=True); lanes_d = score1.lanes_by_key(panel)
    z = {}
    for di, g in enumerate(grids):
        out = TS.predict_grid(g, di, lrtc, hist, W, par, models)
        for f, floor in (('speed_kmh', P1.SPEED_FLOOR), ('flow_vph', P1.FLOW_FLOOR)):
            h = hist.get((f, pd.Timestamp(g.times[0]).weekday())); bad = ~np.isfinite(out[f]) | (out[f] < floor)
            if bad.any() and h is not None: out[f] = np.where(bad & np.isfinite(h), h, out[f]); bad = ~np.isfinite(out[f]) | (out[f] < floor)
            if bad.any(): ob = g.value[f]; med = float(np.nanmedian(ob)) if np.isfinite(ob).any() else floor; out[f] = np.where(bad, med, out[f])
        k = f'd{di}'
        z[f'{k}_v0'] = np.clip(out['speed_kmh'], P1.SPEED_FLOOR, P1.SPEED_CAP).astype(np.float32)
        z[f'{k}_q0'] = np.maximum(out['flow_vph'], P1.FLOW_FLOOR).astype(np.float32)
        for fld, nm in (('speed_kmh', 'V'), ('flow_vph', 'Q'), ('occupancy', 'O')):
            z[f'{k}_{nm}'] = g.value[fld].astype(np.float32); z[f'{k}_T{nm}'] = g.truth[fld].astype(np.float32)
        z[f'{k}_E'] = g.eligible; z[f'{k}_rt'] = real_target[di]; z[f'{k}_gm'] = gaps[di]
        z[f'{k}_rv'] = physics.ramp_valid_grid(links, g.times, att, valid)
        z[f'{k}_ln'] = np.array([lanes_d.get(str(s), lanes_d.get(str(l), 1.0)) for s, l in zip(g.station, g.links)], dtype=np.float32)
        z[f'{k}_a'] = np.array(wl[di], int)
    meta = dict(dates=dates, regimes=[regime[p.name] for p in files], links=links, times=[str(t) for t in grids[0].times])
    return z, meta


def month_profiles(panel, ym):
    """Leave-month-out weekday truth profile (as g3_eval) and static templates excluding month ym."""
    allm = {}
    for rg in R.REGIME_ORDER:
        for p in R.masked_files(panel, 'train', rg): allm[p.name] = p
    S, C = {}, {}
    for nm, tp in {p.name: p for p in R.truth_files(panel, 'train')}.items():
        d = nm[len('synthetic_mainline_'):-len('.parquet')].replace('_', '-')
        if d.startswith(ym): continue
        g = grid.build(panel, tp, allm[nm]); wd = pd.Timestamp(d).weekday()
        ok = g.eligible & np.isfinite(g.truth['speed_kmh']) & np.isfinite(g.truth['flow_vph'])
        for fld in ('speed_kmh', 'flow_vph'):
            S[(fld, wd)] = S.get((fld, wd), 0) + np.where(ok, g.truth[fld], 0.0); C[(fld, wd)] = C.get((fld, wd), 0) + ok
    prof = {k: np.where(C[k] > 0, S[k] / np.maximum(C[k], 1), np.nan) for k in S}
    tz = np.load(HERE / 'rows' / f'{panel}_tmpl.npz', allow_pickle=True)
    H = sum(h for m, h in zip(tz['months'], tz['hist']) if str(m) != ym)
    return prof, template_from_hist(H), [str(x) for x in tz['links']]


def gap_matrix(V, Q, O, E, a_, nT, par, vcut, HV, HQ, bid, tmed, tiqr, win):
    f = gap_features(V.astype(float), Q.astype(float), O.astype(float), HV, HQ, a_, nT, par, vcut)
    pr, nr, b1 = selector_feats(V, E, vcut, a_ - 1, win['gap_min'], win['cond'])
    f.update(v2_features(V, E, a_, nT, vcut, bid, tmed, tiqr, f['queue_pre'][:, 0], f['queue_post'][:, 0],
                         dict(cond=win['cond'], prev_rej=pr, n_rej=nr, blk1=b1)))
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--panel', required=True); ap.add_argument('--month', default='2031_02')
    ap.add_argument('--windows', default='v2', choices=['v2', 'v2x', 'legacy'])
    ap.add_argument('--models', required=True, help='comma list name=<model dir>')
    ap.add_argument('--tag', default='run')
    a = ap.parse_args(); t0 = time.time(); panel = a.panel; ym = a.month.replace('_', '-')
    CACHE.mkdir(parents=True, exist_ok=True); RES = HERE / 'res'; RES.mkdir(exist_ok=True)
    wins = load_windows(panel, a.month, a.windows)
    cf = CACHE / f'{panel}_{a.month}_{a.windows}.npz'
    if cf.exists():
        z = dict(np.load(cf, allow_pickle=True)); meta = json.loads(str(z.pop('meta')))
    else:
        z, meta = build_cache(panel, a.month, a.windows, wins)
        np.savez(cf, meta=json.dumps(meta), **z); print('cached C0', round(time.time() - t0, 1), flush=True)
    links = meta['links']; par = physics.link_params(panel, links); keep = physics.scoreable_mask(panel, links, 'train', meta['times'])
    vcut = 0.6 * par['free_speed_kmh']; lanes = np.maximum(par['lanes'], 1.0); bid = block_ids(panel, links)
    prof, (tmed, tiqr), tl = month_profiles(panel, ym); assert tl == links
    GM = []
    for spec in a.models.split(','):
        nm, dd = spec.split('='); dd = Path(dd)
        fs = json.load(open(dd / 'gap_features.json'))   # optional per-channel lists (hybrid diagnostic dirs)
        fsv = json.load(open(dd / 'gap_features_speed.json')) if (dd / 'gap_features_speed.json').exists() else fs
        fsq = json.load(open(dd / 'gap_features_flow.json')) if (dd / 'gap_features_flow.json').exists() else fs
        GM.append((nm, (fsv, fsq), lgb.Booster(model_file=str(dd / 'gap_speed.txt')), lgb.Booster(model_file=str(dd / 'gap_flow.txt'))))
    L = par['length_km'][:, None]; lanesc = lanes[:, None]; vf = par['free_speed_kmh'][:, None]
    capl = par['capacity_vph'][:, None] / lanesc; kcl = par['critical_density'][:, None] / lanesc; kjl = par['k_jam'][:, None] / lanesc
    rows, cells = [], []
    for di, date in enumerate(meta['dates']):
        k = f'd{di}'; V, Q, O, E = z[f'{k}_V'], z[f'{k}_Q'], z[f'{k}_O'], z[f'{k}_E']
        v0, q0 = z[f'{k}_v0'].astype(float), z[f'{k}_q0'].astype(float); nT = v0.shape[1]
        wd = pd.Timestamp(date).weekday(); cands = [('C0', v0, q0)]
        preds = {nm: (v0.copy(), q0.copy()) for nm, *_ in GM}
        for a_ in z[f'{k}_a']:
            if a_ + LG > nT: continue
            w = wins[(wins.date == date) & (wins.origin == a_ - 1)].iloc[0]
            win = dict(cond=w.condition, gap_min=float(w.gap_min) if np.isfinite(w.gap_min) else np.nan)
            assert np.isnan(V[:, a_:a_ + LG]).all()
            f = gap_matrix(V, Q, O, E, a_, nT, par, vcut, prof[('speed_kmh', wd)], prof[('flow_vph', wd)], bid, tmed, tiqr, win)
            X = pd.DataFrame({c: np.asarray(v).reshape(-1).astype(np.float32) for c, v in f.items()})
            X['panel_c'] = pd.Categorical([panel] * len(X), categories=PANELS)
            tv, tq = z[f'{k}_TV'][:, a_:a_ + LG], z[f'{k}_TQ'][:, a_:a_ + LG]
            ok = E[:, a_:a_ + LG] & np.isfinite(tv) & np.isfinite(tq)
            li, si = np.where(ok)
            cr = dict(date=date, origin=a_ - 1, cond=w.condition, li=li.astype(np.int16), step=si.astype(np.int8),
                      is_target=z[f'{k}_rt'][:, a_:a_ + LG][li, si], y_v=tv[li, si], y_ql=tq[li, si] / lanes[li],
                      y_q_ln=tq[li, si] / np.maximum(z[f'{k}_ln'][li], 1.0), c0_v=v0[:, a_:a_ + LG][li, si].astype(np.float32),
                      c0_ql=(q0[:, a_:a_ + LG][li, si] / lanes[li]).astype(np.float32),
                      qpre=f['queue_pre'][li, si], qpost=f['queue_post'][li, si], vcut=vcut[li].astype(np.float32),
                      prev_rej=f['prev_rej'][li, si], blk1=f['blk1'][li, si])
            for nm, feats, ms, mf in GM:
                pv = np.clip(ms.predict(X[feats[0]]), P1.SPEED_FLOOR, P1.SPEED_CAP).reshape(f['step'].shape)
                pq = np.maximum(mf.predict(X[feats[1]]) * np.repeat(lanes[:, None], LG, 1).reshape(-1), P1.FLOW_FLOOR).reshape(f['step'].shape)
                preds[nm][0][:, a_:a_ + LG] = pv; preds[nm][1][:, a_:a_ + LG] = pq
                cr[f'{nm}_v'] = pv[li, si].astype(np.float32); cr[f'{nm}_ql'] = (pq[li, si] / lanes[li]).astype(np.float32)
            cells.append(pd.DataFrame(cr))
        cands += [(nm, v, q) for nm, (v, q) in preds.items()]
        ln = np.maximum(z[f'{k}_ln'].astype(float), 1.0)[:, None] * np.ones_like(v0)
        tvf, tqf = z[f'{k}_TV'].astype(float), z[f'{k}_TQ'].astype(float)
        rt = z[f'{k}_rt']; gm = z[f'{k}_gm']
        tg = rt & np.isfinite(tvf) & np.isfinite(tqf)
        okc = E & np.isfinite(tvf) & np.isfinite(tqf)
        vt = np.where(okc, tvf, np.nan); qt = np.where(okc, tqf, np.nan)
        obs = np.isfinite(V) & np.isfinite(Q)
        rv = z[f'{k}_rv']; Nt = qt / np.clip(vt, 1, None) * L
        memA = okc & np.isfinite(Nt) & keep[:, None]; memB = memA & (tg | obs)
        for cand, v, q in cands:
            r = dict(date=date, regime=meta['regimes'][di], cand=cand)
            for kind, m in (('in', tg & gm), ('out', tg & ~gm)):
                r[f'n_{kind}'] = int(m.sum()); r[f'sse_v_{kind}'] = float(((v[m] - vt[m]) ** 2).sum()); r[f'sse_q_{kind}'] = float((((q[m] - qt[m]) / ln[m]) ** 2).sum())
            vp = np.where(tg, v, vt); qp = np.where(tg, q, qt)
            Np = qp / np.clip(vp, 1, None) * L
            for fr, mem in (('A', memA), ('B', memB)):
                m = mem[:, 1:] & mem[:, :-1] & rv[:, :-1]
                dN = Np[:, 1:] - Np[:, :-1]; dNt = Nt[:, 1:] - Nt[:, :-1]
                r[f'num{fr}'] = float(np.abs(dN - dNt)[m].sum()); r[f'den{fr}'] = float(np.abs(dNt)[m].sum())
                kl = Np / L / lanesc; ql = qp / lanesc
                qfd = np.where(kl <= kcl, vf * kl, (capl / np.maximum(kjl - kcl, 1e-9)) * np.maximum(kjl - kl, 0))
                mm = mem & np.isfinite(ql) & np.isfinite(qfd)
                r[f'fdn{fr}'] = float(np.abs(ql - qfd)[mm].sum()); r[f'fdd{fr}'] = float(np.abs(ql)[mm].sum())
            rows.append(r)
    tag = f'{panel}_{a.month}_{a.windows}_{a.tag}'
    d = pd.DataFrame(rows); d.to_csv(RES / f'{tag}_days.csv', index=False)
    cl = pd.concat(cells, ignore_index=True); cl['panel'] = panel; cl.to_parquet(RES / f'{tag}_cells.parquet', index=False)
    res = {c: summarize(d[d.cand == c]) for c in d.cand.unique()}
    json.dump(res, open(RES / f'{tag}.json', 'w'), indent=1)
    for c, r in res.items(): print(panel, a.month, a.windows, c, {kk: round(float(vv), 4) for kk, vv in r.items()})
    print('secs', round(time.time() - t0, 1))


def summarize(x):
    s = x.groupby('regime').sum(numeric_only=True)
    n = s.n_in + s.n_out
    rv_ = np.sqrt((s.sse_v_in + s.sse_v_out) / n); rq_ = np.sqrt((s.sse_q_in + s.sse_q_out) / n)
    st = (0.54 * np.maximum(0, 1 - rv_ / 25) + 0.46 * np.maximum(0, 1 - rq_ / 600)).mean()
    out = dict(S_state=st)
    for fr in 'AB':
        lwr = (1 - s[f'num{fr}'] / s[f'den{fr}']).clip(lower=0); fd = 1 - s[f'fdn{fr}'] / s[f'fdd{fr}']
        out[f'S_phys_{fr}'] = ((fd + 2 * lwr) / 3).mean(); out[f'S_LWR_{fr}'] = lwr.mean()
    tot = x.sum(numeric_only=True)
    out.update(rmse_v_in=np.sqrt(tot.sse_v_in / tot.n_in), rmse_q_in=np.sqrt(tot.sse_q_in / tot.n_in), n_in=int(tot.n_in))
    return out


if __name__ == '__main__':
    main()
