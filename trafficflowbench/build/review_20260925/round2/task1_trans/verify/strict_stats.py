"""STRICT split statistics for the skeptic check: for each pseudo-hole row, the tstats.SplitStats features computed from the
days of ONE half only.  Column <stat>_fA = statistics over half-A (odd) days (leave-own-day-out if the row is itself an
A-day row, all A days otherwise); _fB likewise over even days.  Fitting on A and testing on B then uses _fA for both, so
no observation from a test-half day enters any feature of the fit rows or the test rows.
Harness check first: the same code over ALL days with leave-own-day-out must reproduce the stored s_* columns exactly.
Usage: strict_stats.py <validation|train_r2> [panels...]  -> verify/work/strict_<split>_<panel>.parquet"""
import os, sys
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import build_ph as B
from build_ph import physics, block_ids, LG
from tstats import SplitStats
PW = HERE.parent / 'work'
RAWC = ['s_v', 's_ql', 's_qf', 's_n', 's_vq', 's_tq', 's_bqf']


def stats_at(st, li, cc, pre, post):
    o = {k: st[k][li, cc] for k in RAWC}
    for k, c_ in (('pre', pre), ('post', post)):
        o[f's_v_{k}'] = st['s_v'][li, c_]; o[f's_qf_{k}'] = st['s_qf'][li, c_]; o[f's_bqf_{k}'] = st['s_bqf'][li, c_]
    return o


def main():
    split = sys.argv[1]; panels = sys.argv[2:] or B.PANELS
    for panel in panels:
        D = pd.read_parquet(PW / f'ph_{split}_{panel}.parquet', columns=['panel', 'date', 'day', 'origin', 'li', 'step_i'] + RAWC +
                            [f'{a}_{b}' for b in ('pre', 'post') for a in ('s_v', 's_qf', 's_bqf')])
        grids = B.load_panel(panel, 'validation' if split == 'validation' else 'train', '' if split == 'validation' else '2031_02')
        dates = [g.date for g in grids]; links = [str(x) for x in grids[0].links]
        par = physics.link_params(panel, links); vcut = 0.6 * par['free_speed_kmh']; lanes = np.maximum(par['lanes'], 1.0); bid = block_ids(panel, links)
        Vs = np.stack([g.value['speed_kmh'] for g in grids]); Qs = np.stack([g.value['flow_vph'] for g in grids]); Es = np.stack([g.eligible for g in grids])
        del grids
        di = D.date.map({d: i for i, d in enumerate(dates)}).to_numpy(); li = D.li.to_numpy().astype(int)
        cc = (D.origin.to_numpy() + 1 + D.step_i.to_numpy()).astype(int); pre = (D.origin.to_numpy()).astype(int); post = (D.origin.to_numpy() + 1 + LG).astype(int)
        # harness: all days, leave-own-day-out == stored columns
        SS = SplitStats(Vs, Qs, Es, vcut, lanes, bid); mx = 0.0
        for d in np.unique(di):
            m = di == d; o = stats_at(SS.loo(d), li[m], cc[m], pre[m], post[m])
            for k, x in o.items():
                y = D[k].to_numpy()[m].astype(np.float64); x = x.astype(np.float32).astype(np.float64)
                assert (np.isnan(x) == np.isnan(y)).all(), (panel, k)
                mx = max(mx, float(np.nanmax(np.abs(x - y))) if np.isfinite(y).any() else 0.0)
        print(panel, 'harness max |diff| vs stored s_* columns', mx, flush=True)
        out = {'panel': D.panel.to_numpy(), 'li': D.li.to_numpy(), 'origin': D.origin.to_numpy(), 'step_i': D.step_i.to_numpy()}
        dayno = np.array([int(d[8:10]) for d in dates])
        for h, par_ in (('A', 1), ('B', 0)):
            idx = np.where(dayno % 2 == par_)[0]
            nanday = np.full((1,) + Vs.shape[1:], np.nan)
            SSh = SplitStats(np.concatenate([Vs[idx], nanday]), np.concatenate([Qs[idx], nanday]),
                             np.concatenate([Es[idx], np.zeros((1,) + Es.shape[1:], bool)]), vcut, lanes, bid)
            pos = {int(d): j for j, d in enumerate(idx)}; dummy = len(idx)
            cols = {}
            for d in np.unique(di):
                m = di == d; st = SSh.loo(pos.get(int(d), dummy))
                o = stats_at(st, li[m], cc[m], pre[m], post[m])
                for k, x in o.items():
                    cols.setdefault(k, np.full(len(D), np.nan, np.float32))[m] = x.astype(np.float32)
            for k, x in cols.items(): out[f'{k}_f{h}'] = x
        pd.DataFrame(out).to_parquet(HERE / 'work' / f'strict_{split}_{panel}.parquet', index=False)
        print(panel, 'rows', len(D), 'written', flush=True)


if __name__ == '__main__':
    main()
