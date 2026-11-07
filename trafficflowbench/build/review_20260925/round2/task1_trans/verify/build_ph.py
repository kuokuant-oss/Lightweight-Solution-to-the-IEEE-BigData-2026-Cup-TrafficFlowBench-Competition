"""Pseudo-hole cell table for the transductive Task 1 study (one panel in memory at a time).

--split validation : ruler2 pseudo windows (corrected replicated selector) on validation days; a window is kept only if
                     [T-12, T+1+18+12) does not touch an official Task 2 hole of that day; overlapping pseudo holes on
                     the same day are all kept (each window is blanked and evaluated on its own; 'dup' counts overlaps).  History T-12..T-1:
                     official-target cells filled with the champion (G1) Task 1 value (stands in for the republished
                     history).  Hole T+1..T+18 blanked on all links.  Truth = released observed value (observed cells only).
--split train      : Feb 2031, g2v2 extended windows (eval_windows_<panel>_2031_02x.csv), history restored from truth,
                     leave-month-out profile/template (eval_v2.month_profiles), truth at every eligible hole cell.
Champion hole prediction = G2v2 v2_F through eval_v2.gap_matrix (the serving feature path of serve_v2.py).
V0 = the same model with the train weekday profile replaced by the leave-own-day-out split profile (train profile where
the split profile is undefined).  Split statistics are leave-own-day-out over the whole split (tstats.SplitStats).
Output: work/ph_<split>_<panel>.parquet"""
import os, sys, time, argparse, json
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
G2W = ROOT / 'build/review_20260924/work'; sys.path.insert(0, str(G2W))
G2V2 = ROOT / 'build/review_20260924_ctm/g2v2'; sys.path.insert(0, str(G2V2))
from src.tfb import release as R, grid, physics, predict1 as P1
from v2feat import template_from_hist, LG, HISTN
from data_v2 import block_ids
from eval_v2 import gap_matrix, month_profiles
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from tstats import SplitStats
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
R2 = ROOT / 'build/review_20260924_ctm/ruler2'
MD = G2V2 / 'models/v2_F'
TSF = '%Y-%m-%dT%H:%M:%SZ'
KEEP_F = ['step', 'w', 'lin_v', 'lin_q', 'h_v', 'h_q', 'pv', 'pvd', 'pq', 'sv', 'svd', 'sq', 'pv3', 'sv3', 'rv_pre', 'rv_post',
          'rq_pre', 'rq_post', 'rq_all_pre', 'rq_all_post', 'queue_pre', 'queue_post', 'tod', 'lanes', 'free_speed_kmh',
          'capacity_vph', 'cond_on', 'prev_rej', 'n_rej', 'blk1', 'bsize', 'bq_pre0', 'bq_hist', 'bq_post', 'bq_postd',
          'tmpl_dyn', 'tmpl_dyn_age', 'tmpl_post', 'tmpl_post_d', 'tmpl_stat', 'dq_up_pre', 'dq_dn_pre', 'nq_pre',
          'dq_up_post', 'dq_dn_post', 'nq_post']


def pseudo_windows_validation(cands=None, check=True):
    """Re-run ruler2's selection (select_write_v2.select) recording the previous window of each sequence -> gap_min.
    cands: candidate table (default ruler2 cands_v2 on validation); check: assert equality with ruler2's windows."""
    c = pd.read_parquet(R2 / 'cands_v2.parquet') if cands is None else cands
    dates = sorted(c.date.unique()); d0 = pd.Timestamp(dates[0])
    by = {(p, d): g for (p, d), g in c.groupby(['panel', 'date'])}
    rows = []
    for panel in sorted(c.panel.unique()):
        for s in range(len(dates)):
            last = None; cnt = {'queue_onset': 0, 'queue_ongoing': 0}; done = False
            for date in dates[s:]:
                if done: break
                g = by.get((panel, date))
                if g is None: continue
                off = (pd.Timestamp(date) - d0).days * 1440
                for r in g.itertuples():
                    if cnt['queue_onset'] >= 5 and cnt['queue_ongoing'] >= 5: done = True; break
                    if r.cov < 0.7 or not r.hor_any: continue
                    if not r.hist_any: cond = 'queue_onset'
                    elif r.last_any: cond = 'queue_ongoing'
                    else: continue
                    if cond == 'queue_ongoing' and r.piou > 0.9: continue
                    if cnt[cond] >= 5: continue
                    tm = off + r.t * 5
                    if last is not None and tm - last < 360: continue
                    rows.append((panel, date, int(r.t), cond, s, tm - last if last is not None else np.nan)); cnt[cond] += 1; last = tm
    w = pd.DataFrame(rows, columns=['panel', 'date', 't', 'condition', 'start', 'gap_min'])
    u = w.sort_values('start').groupby(['panel', 'date', 't', 'condition']).agg(mult=('start', 'size'), gap_min=('gap_min', 'first')).reset_index()
    if not check: return u
    ref = pd.read_parquet(R2 / 'windows_validation.parquet')
    m = ref.merge(u, on=['panel', 'date', 't', 'condition'], how='outer', indicator=True)
    assert (m._merge == 'both').all() and (m.mult_x == m.mult_y).all(), 'selector re-run does not reproduce ruler2 windows'
    return u


def load_panel(panel, split, month):
    if split == 'validation':
        files = P1.split_masked_files(panel, 'validation')
        grids = [grid.build_masked(panel, p) for p in files]
    else:
        truth_by = {p.name: p for p in R.truth_files(panel, 'train')}
        files = [p for p in P1.split_masked_files(panel, 'train') if month in p.name]
        grids = [grid.build(panel, truth_by[p.name], p) for p in files]
    order = np.argsort([g.date for g in grids], kind='stable')
    return [grids[i] for i in order]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', required=True, choices=['validation', 'train'])
    ap.add_argument('--month', default='2031_02'); ap.add_argument('--panels', nargs='*', default=PANELS)
    ap.add_argument('--wins', default='x', choices=['x', 'r2'], help='train only: g2v2 extended windows (x) or the ruler2 selector run on the month (r2)')
    ap.add_argument('--hist', default='restore', choices=['restore', 'none'], help='train only: restore T-12..T-1 from truth (official-like) or leave the masked layer as is (fill-artifact diagnostic)')
    a = ap.parse_args(); t0 = time.time()
    feats = json.load(open(MD / 'gap_features.json'))
    ms = lgb.Booster(model_file=str(MD / 'gap_speed.txt')); mf = lgb.Booster(model_file=str(MD / 'gap_flow.txt'))
    if a.split == 'validation':
        PW = pseudo_windows_validation()
        CH = pd.read_parquet(HERE / 'work' / 'champ_val_state.parquet')
    for panel in a.panels:
        out = HERE / 'work' / (f'ph_{a.split}_{panel}.parquet' if a.split == 'validation' or a.wins == 'x' else f'ph_train_r2{"nh" if a.hist == "none" else ""}_{panel}.parquet')
        if out.exists(): print('skip', out.name, flush=True); continue
        grids = load_panel(panel, a.split, a.month)
        dates = [g.date for g in grids]; links = [str(x) for x in grids[0].links]; nL = len(links)
        times0 = [pd.Timestamp(t).strftime('%H:%M:%S') for t in grids[0].times]; nT = len(times0)
        par = physics.link_params(panel, links); vcut = 0.6 * par['free_speed_kmh']; lanes = np.maximum(par['lanes'], 1.0)
        bid = block_ids(panel, links)
        if a.split == 'validation':
            prof = np.load(G2W / 'g2' / f'{panel}_profile_all.npz', allow_pickle=True)
            tz = np.load(G2V2 / 'rows' / f'{panel}_tmpl.npz', allow_pickle=True)
            assert links == [str(x) for x in prof['links']] == [str(x) for x in tz['links']]
            tmed, tiqr = template_from_hist(tz['hist'].sum(0))
            PV = lambda wd: (prof[f'speed_kmh_{wd}'], prof[f'flow_vph_{wd}'])
            wi = pd.read_csv(R.RELEASE / 'task2' / panel / 'validation' / 'window_index.csv')
            off_holes = {}
            for _, w in wi.iterrows():
                a_ = times0.index(w.forecast_start[11:19]); off_holes.setdefault(w.date, []).append(a_)
            wins = PW[PW.panel == panel].copy(); wins['origin'] = wins.t
            ch = CH[CH.panel == panel]
            chd = {d: x for d, x in ch.assign(date=ch.timestamp.str[:10]).groupby('date')}
        else:
            prof_, (tmed, tiqr), tl = month_profiles(panel, a.month.replace('_', '-')); assert tl == links
            PV = lambda wd: (prof_[('speed_kmh', wd)], prof_[('flow_vph', wd)])
            if a.wins == 'x':
                wins = pd.read_csv(G2V2 / f'eval_windows_{panel}_{a.month}x.csv'); wins['gap_min'] = wins.tmin - wins.tprev_min
            else:   # ruler2 selector (sequences starting on every day of the month, within the month), as on validation
                cz = pd.read_parquet(ROOT / 'build/review_20260924_ctm/rep2/candidates_v2.parquet')
                cz = cz[(cz.panel == panel) & cz.date.str.startswith(a.month.replace('_', '-'))].sort_values(['date', 't'])
                wins = pseudo_windows_validation(cz, check=False); wins['origin'] = wins.t
            off_holes = {}
        Vs = np.stack([g.value['speed_kmh'] for g in grids]); Qs = np.stack([g.value['flow_vph'] for g in grids])
        Es = np.stack([g.eligible for g in grids])
        SS = SplitStats(Vs, Qs, Es, vcut, lanes, bid)
        del Vs, Qs, Es
        li_ = {l: j for j, l in enumerate(links)}; ti_ = {t: j for j, t in enumerate(times0)}
        parts = []; nw = dict(cand=len(wins), kept=0, off_overlap=0, dup=0, edge=0)
        for di, g in enumerate(grids):
            ws = wins[wins.date == g.date].sort_values(['mult', 'origin'], ascending=[False, True]) if 'mult' in wins.columns else wins[wins.date == g.date].sort_values('origin')
            if not len(ws): continue
            oh = off_holes.get(g.date, [])
            kept = []
            for _, w in ws.iterrows():
                a_ = int(w.origin) + 1
                if a_ - HISTN - 1 < 0 or a_ + LG + 12 > nT: nw['edge'] += 1; continue
                if any((a_ - 13 < o + LG) and (o < a_ + LG + 12) for o in oh): nw['off_overlap'] += 1; continue
                if any(abs(a_ - k) < LG for k, _ in kept): nw['dup'] += 1   # overlapping pseudo holes are kept (each evaluated alone)
                kept.append((a_, w))
            if not kept: continue
            wd = pd.Timestamp(g.date).weekday(); HV, HQ = PV(wd)
            st = SS.loo(di)
            HVs = np.where(np.isfinite(st['s_v']), st['s_v'], HV); HQs = np.where(np.isfinite(st['s_ql']), st['s_ql'] * lanes[:, None], HQ)
            if a.split == 'validation':
                cz = chd.get(g.date); CV = np.full((nL, nT), np.nan); CQ = np.full((nL, nT), np.nan)
                if cz is not None:
                    r_ = cz.link_id.map(li_).to_numpy(); c_ = cz.timestamp.str[11:19].map(ti_).to_numpy()
                    CV[r_, c_] = cz.speed_kmh.to_numpy(); CQ[r_, c_] = cz.flow_vph.to_numpy()
            for a_, w in kept:
                T = a_ - 1; hs = slice(T - HISTN, T); seg = slice(a_, a_ + LG)
                V = g.value['speed_kmh'].copy(); Q = g.value['flow_vph'].copy(); O = g.value['occupancy'].copy(); E = g.eligible
                if a.split == 'validation':
                    tv = V[:, seg].copy(); tq = Q[:, seg].copy()
                    for X, C in ((V, CV), (Q, CQ)):
                        X[:, hs] = np.where(np.isnan(X[:, hs]) & np.isfinite(C[:, hs]), C[:, hs], X[:, hs])
                    obs_t = np.isfinite(tv) & np.isfinite(tq)
                else:
                    tv = g.truth['speed_kmh'][:, seg].copy(); tq = g.truth['flow_vph'][:, seg].copy()
                    obs_t = np.isfinite(g.value['speed_kmh'][:, seg]) & np.isfinite(g.value['flow_vph'][:, seg])
                    for X, fld in ((V, 'speed_kmh'), (Q, 'flow_vph'), (O, 'occupancy')):
                        if a.hist == 'restore': X[:, hs] = np.where(E[:, hs] & np.isnan(X[:, hs]), g.truth[fld][:, hs], X[:, hs])
                for X in (V, Q, O): X[:, seg] = np.nan
                win = dict(cond=w.condition, gap_min=float(w.gap_min) if np.isfinite(w.gap_min) else np.nan)
                ok = E[:, seg] & np.isfinite(tv) & np.isfinite(tq)
                if not ok.any(): continue
                res = {}
                for nm, (hv, hq) in (('ch', (HV, HQ)), ('v0', (HVs, HQs))):
                    f = gap_matrix(V, Q, O, E, a_, nT, par, vcut, hv, hq, bid, tmed, tiqr, win)
                    X = pd.DataFrame({k: np.asarray(v).reshape(-1).astype(np.float32) for k, v in f.items()})
                    X['panel_c'] = pd.Categorical([panel] * len(X), categories=PANELS)
                    pv = np.clip(ms.predict(X[feats], num_threads=3), P1.SPEED_FLOOR, P1.SPEED_CAP).reshape(f['step'].shape)
                    pq = np.maximum(mf.predict(X[feats], num_threads=3) * np.repeat(lanes[:, None], LG, 1).reshape(-1), P1.FLOW_FLOOR).reshape(f['step'].shape)
                    res[nm] = (pv, pq, f)
                li, si = np.where(ok); f = res['ch'][2]
                d = {'date': g.date, 'day': np.int16(int(g.date[8:10])), 'origin': np.int16(T), 'cond': w.condition,
                     'li': li.astype(np.int16), 'step_i': si.astype(np.int8), 'obs': obs_t[li, si],
                     'y_v': tv[li, si].astype(np.float32), 'y_ql': (tq[li, si] / lanes[li]).astype(np.float32),
                     'ch_v': res['ch'][0][li, si].astype(np.float32), 'ch_ql': (res['ch'][1][li, si] / lanes[li]).astype(np.float32),
                     'v0_v': res['v0'][0][li, si].astype(np.float32), 'v0_ql': (res['v0'][1][li, si] / lanes[li]).astype(np.float32),
                     'vcut': vcut[li].astype(np.float32), 'bid': bid[li].astype(np.int16)}
                for k in KEEP_F: d['f_' + k] = np.asarray(f[k])[li, si].astype(np.float32)
                cc = a_ + si
                for k, x in st.items(): d[k] = x[li, cc].astype(np.float32)
                for k, c_ in (('pre', a_ - 1), ('post', a_ + LG)):
                    d[f's_v_{k}'] = st['s_v'][li, c_].astype(np.float32); d[f's_qf_{k}'] = st['s_qf'][li, c_].astype(np.float32)
                    d[f's_bqf_{k}'] = st['s_bqf'][li, c_].astype(np.float32)
                parts.append(pd.DataFrame(d)); nw['kept'] += 1
        D = pd.concat(parts, ignore_index=True); D.insert(0, 'panel', panel)
        D.to_parquet(out, index=False)
        print(panel, a.split, 'windows', nw, 'cells', len(D), 'obs', int(D.obs.sum()), round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
