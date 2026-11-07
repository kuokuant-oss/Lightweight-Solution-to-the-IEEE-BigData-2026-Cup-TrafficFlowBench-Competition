"""G2v2 training rows: per panel, one row per eligible cell inside a simulated Task 2 hole on train.

Hole placement: corrected (v2) replicated official windows with their selector predecessor
(train_windows_v2seq.parquet, from sel_seq.py).  Per window: the masked layer outside the hole, the window history
T-12..T-1 restored from truth at eligible cells (the validation template republishes it), hole T+1..T+18 blank.
Features = the unchanged G2 features (build/review_20260924/work/g2_gap_data.gap_features, leave-month-out
weekday profile) + v2feat.v2_features (window condition, serving-identical selector features, block states,
templates [static template leave-month-out], distance to queued links).
Sampling: every cell of Jan/Feb 2031 windows (evaluation months), a u<0.35 subsample elsewhere; column u lets the
trainer apply the same 35% subsample to any month.
Output: rows/<panel>_rows.parquet, rows/<panel>_tmpl.npz (per-month template histograms)."""
#
# Role: record of how the G2v2 training rows and templates were built. `block_ids` is imported by eval_v2.py and
# serve_v2.py, and rows/<panel>_tmpl.npz is read by serve_v2.py (reproduction pipeline, task1.sh step 4).
# Usage: data_v2.py --panel <panel> [--sample 0.35] [--v3]
# Row columns: G2 and v2 (and v3) features, panel, date, origin T, li (link row), cond, regime, is_target, u
# (uniform draw used for subsampling), labels y_v (speed), y_q (flow) and y_ql (flow per lane).
import os, sys, argparse, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'build/review_20260924/work'))
from src.tfb import release as R, grid, physics
from g2_gap_data import gap_features
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from v2feat import v2_features, v3_features, selector_feats, template_from_hist, LG, HISTN, NB
# my_blocks: per panel, link ids and their block id (groups of links that queue together, built from train).
BLK = np.load(ROOT / 'build/review_20260924_newworld/work/blocks/my_blocks.npy', allow_pickle=True).item()
# Months whose hole cells are all kept (the evaluation months).
FULL_MONTHS = ('2031-01', '2031-02')


def block_ids(panel, links):
    """Returns the block id of each link (in the given order); -1 for a link without a block."""
    m = dict(zip([str(x) for x in BLK[panel]['links']], BLK[panel]['bid']))
    return np.array([m.get(l, -1) for l in links])


def main():
    """Builds the G2v2 training rows of one panel from the train months."""
    ap = argparse.ArgumentParser(); ap.add_argument('--panel', required=True); ap.add_argument('--sample', type=float, default=0.35)
    ap.add_argument('--v3', action='store_true', help='also v3 features; output rows3/')
    a = ap.parse_args(); t0 = time.time(); panel = a.panel
    OUT = HERE / ('rows3' if a.v3 else 'rows'); OUT.mkdir(exist_ok=True)
    tfiles = {p.name: p for p in R.truth_files(panel, 'train')}
    mfiles = {}
    for rg in R.REGIME_ORDER:
        for p in R.masked_files(panel, 'train', rg): mfiles[p.name] = (p, rg)
    names = sorted(tfiles); days = []
    for nm in names:
        g = grid.build(panel, tfiles[nm], mfiles[nm][0])
        days.append(dict(date=g.date, regime=mfiles[nm][1], V=g.value['speed_kmh'].astype(np.float32), Q=g.value['flow_vph'].astype(np.float32),
                         O=g.value['occupancy'].astype(np.float32), TV=g.truth['speed_kmh'].astype(np.float32), TQ=g.truth['flow_vph'].astype(np.float32),
                         TO=g.truth['occupancy'].astype(np.float32), E=g.eligible.copy(), tgt=g.target.copy()))
        links = [str(x) for x in g.links]
    nL, nT = days[0]['V'].shape
    par = physics.link_params(panel, links); vcut = 0.6 * par['free_speed_kmh']; lanes = np.maximum(par['lanes'], 1.0)
    bid = block_ids(panel, links); assert (bid >= 0).all(), 'links without block'
    # leave-month-out weekday profiles (as G2) and per-month queued-speed histograms (static templates)
    S, C = {}, {}; H = {}
    for d in days:
        mo, wd = d['date'][:7], pd.Timestamp(d['date']).weekday()
        ok = d['E'] & np.isfinite(d['TV']) & np.isfinite(d['TQ'])
        for fld, X in (('speed_kmh', d['TV']), ('flow_vph', d['TQ'])):
            k = (fld, mo, wd)
            S[k] = S.get(k, 0) + np.where(ok, X, 0.0); C[k] = C.get(k, 0) + ok
        # Queued truth cells add to the month's per-link speed histogram (1 km/h bins).
        qq = ok & (d['TV'] <= vcut[:, None])
        li, ci = np.where(qq); b = np.clip(np.round(d['TV'][li, ci]).astype(int), 0, NB - 1)
        h = H.setdefault(mo, np.zeros((nL, NB), np.int32)); np.add.at(h, (li, b), 1)
    months = sorted(H)
    np.savez_compressed(OUT / f'{panel}_tmpl.npz', months=np.array(months), hist=np.stack([H[m] for m in months]), links=np.array(links))
    def prof(fld, wd, excl):
        """Returns the mean truth profile of `fld` for weekday `wd` over all months except `excl`."""
        s = sum(v for k, v in S.items() if k[0] == fld and k[2] == wd and k[1] != excl)
        c = sum(v for k, v in C.items() if k[0] == fld and k[2] == wd and k[1] != excl)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    tm_cache = {}
    def tmpl(excl):
        """Returns the static template (median, IQR of queued speeds per link) from all months except `excl`."""
        if excl not in tm_cache: tm_cache[excl] = template_from_hist(sum(H[m] for m in months if m != excl))
        return tm_cache[excl]
    # Corrected (v2) replicated official windows with the minutes since their selector predecessor (gap_min).
    wins = pd.read_parquet(HERE / 'train_windows_v2seq.parquet'); wins = wins[wins.panel == panel]
    byd = {d['date']: i for i, d in enumerate(days)}
    rng = np.random.default_rng(11); rows = []; pc = {}
    for _, w in wins.iterrows():
        if w.date not in byd: continue
        d = days[byd[w.date]]; T = int(w.origin); a_ = T + 1
        if a_ + LG > nT or T - HISTN < 0: continue
        wd = pd.Timestamp(w.date).weekday(); mo = w.date[:7]
        if (wd, mo) not in pc: pc[(wd, mo)] = (prof('speed_kmh', wd, mo), prof('flow_vph', wd, mo))
        HV, HQ = pc[(wd, mo)]
        V, Q, O = d['V'].copy(), d['Q'].copy(), d['O'].copy()
        # Window history t-12..t-1 restored from truth at eligible cells, as the release republishes it.
        hs = slice(T - HISTN, T); el = d['E'][:, hs]
        for X, TX in ((V, d['TV']), (Q, d['TQ']), (O, d['TO'])):
            X[:, hs] = np.where(el & np.isnan(X[:, hs]), TX[:, hs], X[:, hs])
        # The hole itself is blank.
        for X in (V, Q, O): X[:, a_:a_ + LG] = np.nan
        f = gap_features(V.astype(float), Q.astype(float), O.astype(float), HV, HQ, a_, nT, par, vcut)
        # Selector features from the masked layer and gap_min; static template leave-month-out.
        pr, nr, b1 = selector_feats(V, d['E'], vcut, T, float(w.gap_min) if np.isfinite(w.gap_min) else np.nan, w.condition)
        tmed, tiqr = tmpl(mo)
        f.update(v2_features(V, d['E'], a_, nT, vcut, bid, tmed, tiqr, f['queue_pre'][:, 0], f['queue_post'][:, 0],
                             dict(cond=w.condition, prev_rej=pr, n_rej=nr, blk1=b1)))
        if a.v3: f.update(v3_features(V, a_, nT, vcut, bid, dict(cond=w.condition, blk1=b1), f['step']))
        yv = d['TV'][:, a_:a_ + LG]; yq = d['TQ'][:, a_:a_ + LG]
        el = d['E'][:, a_:a_ + LG] & np.isfinite(yv) & np.isfinite(yq)
        # Every hole cell of FULL_MONTHS is kept; elsewhere a `sample` share.
        u = rng.random(el.shape).astype(np.float32)
        keep = el & ((u < a.sample) | (mo in FULL_MONTHS))
        if not keep.any(): continue
        li, si = np.where(keep)
        r = {c: np.asarray(v)[li, si].astype(np.float32) for c, v in f.items()}
        r.update(panel=panel, date=w.date, origin=np.int16(T), li=li.astype(np.int16), cond=w.condition, regime=d['regime'],
                 is_target=d['tgt'][:, a_:a_ + LG][li, si], u=u[li, si], y_v=yv[li, si].astype(np.float32),
                 y_q=yq[li, si].astype(np.float32), y_ql=(yq[li, si] / lanes[li]).astype(np.float32))
        rows.append(pd.DataFrame(r))
    df = pd.concat(rows, ignore_index=True)
    for c in ('panel', 'date', 'cond', 'regime'): df[c] = df[c].astype('category')
    df.to_parquet(OUT / f'{panel}_rows.parquet', index=False)
    print(panel, 'rows', len(df), 'gaps', df.groupby(['date', 'origin'], observed=True).ngroups, 'secs', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
