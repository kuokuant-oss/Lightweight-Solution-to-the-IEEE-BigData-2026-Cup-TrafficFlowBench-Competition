"""Skeptic V4 (+ STRICT inputs) for the PRIVATE pseudo holes built by the pilot, one panel in memory at a time.
Per panel, from the released private masked layer only:
  (1) every ph_private row's date is a private date; y_v / y_ql*lanes == released masked speed / flow at (li, origin+1+step_i);
  (2) no row's window [T-12, T+31) touches an official private hole (recomputed from task2 window_index, same +-12 rule);
  (3) harness: SplitStats over all private days, leave-own-day-out, reproduces the stored s_* columns of ph_private AND of the
      pilot's official rows off_rows_private (so both came from private data only);
  (4) STRICT columns: split statistics over ONE day-parity half only -> verify/work/strict_private_<panel>.parquet.
Usage: v1b_strict_private.py [panels]"""
import os, sys, json
sys.dont_write_bytecode = True
os.environ['OMP_NUM_THREADS'] = '3'
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; LENS = HERE.parent
R2T = LENS.parent.parent / 'round2' / 'task1_trans'
sys.path.insert(0, str(R2T))
import build_ph as B                      # read-only import
from build_ph import R, grid, physics, P1, block_ids, LG
from tstats import SplitStats
RAWC = ['s_v', 's_ql', 's_qf', 's_n', 's_vq', 's_tq', 's_bqf']


def stats_at(st, li, cc, pre, post):
    o = {k: st[k][li, cc] for k in RAWC}
    for k, c_ in (('pre', pre), ('post', post)):
        o[f's_v_{k}'] = st['s_v'][li, c_]; o[f's_qf_{k}'] = st['s_qf'][li, c_]; o[f's_bqf_{k}'] = st['s_bqf'][li, c_]
    return o


def cmp(D, di, li, cc, pre, post, SS, tag):
    mx = 0.0; nanmis = 0
    for d in np.unique(di):
        m = di == d; o = stats_at(SS.loo(d), li[m], cc[m], pre[m], post[m])
        for k, x in o.items():
            y = D[k].to_numpy()[m].astype(np.float64); x = x.astype(np.float32).astype(np.float64)
            nanmis += int((np.isnan(x) != np.isnan(y)).sum())
            if np.isfinite(y).any(): mx = max(mx, float(np.nanmax(np.abs(x - y))))
    return {f'{tag}_max_abs': mx, f'{tag}_nan_mismatch': nanmis}


def main():
    panels = sys.argv[1:] or B.PANELS; allres = {}
    Of_all = pd.read_parquet(LENS / 'work' / 'off_rows_private.parquet')
    for panel in panels:
        D = pd.read_parquet(LENS / 'work' / f'ph_private_{panel}.parquet')
        files = P1.split_masked_files(panel, 'private')
        grids = [grid.build_masked(panel, p) for p in files]; grids = [grids[i] for i in np.argsort([g.date for g in grids], kind='stable')]
        dates = [g.date for g in grids]; links = [str(x) for x in grids[0].links]; times0 = [pd.Timestamp(t).strftime('%H:%M:%S') for t in grids[0].times]
        par = physics.link_params(panel, links); vcut = 0.6 * par['free_speed_kmh']; lanes = np.maximum(par['lanes'], 1.0); bid = block_ids(panel, links)
        Vs = np.stack([g.value['speed_kmh'] for g in grids]); Qs = np.stack([g.value['flow_vph'] for g in grids]); Es = np.stack([g.eligible for g in grids])
        del grids
        r = {'rows': len(D), 'dates_all_private': bool(set(D.date) <= set(dates)), 'n_private_days': len(dates),
             'first_date': dates[0], 'last_date': dates[-1]}
        di = D.date.map({d: i for i, d in enumerate(dates)}).to_numpy(); li = D.li.to_numpy().astype(int)
        cc = (D.origin.to_numpy() + 1 + D.step_i.to_numpy()).astype(int)
        pre = D.origin.to_numpy().astype(int); post = (D.origin.to_numpy() + 1 + LG).astype(int)
        # (1) truth = released observed private value
        tv = Vs[di, li, cc]; tq = Qs[di, li, cc]
        r['y_v_max_abs_vs_released'] = float(np.nanmax(np.abs(tv - D.y_v.to_numpy())))
        r['y_q_max_abs_vs_released'] = float(np.nanmax(np.abs(tq - D.y_ql.to_numpy() * lanes[li])))
        r['y_nan_or_unreleased'] = int((~np.isfinite(tv)).sum())
        r['eligible_all'] = bool(Es[di, li, cc].all())
        # (2) official private holes +-12
        wi = pd.read_csv(R.RELEASE / 'task2' / panel / 'private' / 'window_index.csv')
        oh = {}
        for _, w in wi.iterrows(): oh.setdefault(w.date, []).append(times0.index(w.forecast_start[11:19]))
        W = D[['date', 'origin']].drop_duplicates()
        bad = 0; mind = []
        for w in W.itertuples():
            a_ = int(w.origin) + 1
            for o in oh.get(w.date, []):
                if (a_ - 13 < o + LG) and (o < a_ + LG + 12): bad += 1
                mind.append(abs(a_ - o))
        r['windows'] = len(W); r['windows_touching_official_pm12'] = bad
        r['min_origin_distance_to_official_steps'] = int(min(mind)) if mind else None
        r['windows_on_official_days'] = int(sum(w.date in oh for w in W.itertuples()))
        # official hole cells are NaN in the released layer (so pseudo truth can never be an official hole cell)
        # (3) harness for ph_private and official rows
        SS = SplitStats(Vs, Qs, Es, vcut, lanes, bid)
        r.update(cmp(D, di, li, cc, pre, post, SS, 'ph_sstats'))
        Of = Of_all[Of_all.panel == panel]
        odi = Of.date.map({d: i for i, d in enumerate(dates)}).to_numpy(); oli = Of.li.to_numpy().astype(int)
        occ = (Of.origin.to_numpy() + 1 + Of.step_i.to_numpy()).astype(int)
        r['official_cells_released_nan'] = bool(np.isnan(Vs[odi, oli, occ]).all())
        r.update(cmp(Of, odi, oli, occ, Of.origin.to_numpy().astype(int), (Of.origin.to_numpy() + 1 + LG).astype(int), SS, 'off_sstats'))
        # (4) STRICT half-only stats
        out = {'panel': D.panel.to_numpy(), 'li': D.li.to_numpy(), 'origin': D.origin.to_numpy(), 'step_i': D.step_i.to_numpy()}
        dayno = np.array([int(d[8:10]) for d in dates])
        for h, pr in (('A', 1), ('B', 0)):
            idx = np.where(dayno % 2 == pr)[0]; nanday = np.full((1,) + Vs.shape[1:], np.nan)
            SSh = SplitStats(np.concatenate([Vs[idx], nanday]), np.concatenate([Qs[idx], nanday]),
                             np.concatenate([Es[idx], np.zeros((1,) + Es.shape[1:], bool)]), vcut, lanes, bid)
            pos = {int(d): j for j, d in enumerate(idx)}; dummy = len(idx); cols = {}
            for d in np.unique(di):
                m = di == d; o = stats_at(SSh.loo(pos.get(int(d), dummy)), li[m], cc[m], pre[m], post[m])
                for k, x in o.items(): cols.setdefault(k, np.full(len(D), np.nan, np.float32))[m] = x.astype(np.float32)
            for k, x in cols.items(): out[f'{k}_f{h}'] = x
        pd.DataFrame(out).to_parquet(HERE / 'work' / f'strict_private_{panel}.parquet', index=False)
        allres[panel] = r; print(panel, r, flush=True)
        del Vs, Qs, Es, SS
    json.dump(allres, open(HERE / 'results' / 'v4_private_legality.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
