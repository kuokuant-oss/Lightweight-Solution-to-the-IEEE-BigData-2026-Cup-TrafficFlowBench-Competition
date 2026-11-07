"""G2 data: per-panel gap feature tables on train, gaps placed where the (replicated) official selector puts
Task 2 windows.  One row per eligible cell inside a simulated gap (T+1..T+18, all links), labels = truth.

Information used is exactly what serving has on validation/private:
  * the masked layer outside the gap (regular Task 1 blanks stay blank),
  * the 12-step window history T-12..T-1 republished in full (restored from truth where the train mask blanked it),
  * a weekday x time-of-day truth profile from OTHER train months (leave-month-out; serving uses all 9 months).
Nothing inside the gap is read.  Output: g2/<panel>_rows.parquet
"""
# Role: `gap_features` and `LG` are imported by the reproduction pipeline (rl/serve_feats.py, Task 1 step 1)
# and by the G2v2 gap model code (g2v2/). `main()` is the record of how the original G2 gap-model training
# table was built; the pipeline does not run it.
import os, sys, argparse, time
import numpy as np, pandas as pd
from pathlib import Path
# Workspace root on the original development machine; the path layer of bin/tfb-run maps it onto the run tree.
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
# release: lists of raw day files; grid: per-day (link x 5-minute step) arrays with links in milepost order;
# physics: per-link lanes, free-flow speed, capacity and length.
from src.tfb import release as R, grid, physics, score1
HERE = Path(__file__).resolve().parent
LG = 18          # forecast horizon (6) + 60-minute buffer (12)
HISTN = 12       # window history length

def gap_features(V, Q, O, HV, HQ, a, nT, par, vcut):
    """Feature arrays (nL x LG) for a gap starting at column a. V/Q/O: observed layer (NaN = unseen)."""
    # Arguments: V/Q/O hold speed (km/h), flow (veh/h) and occupancy, shape (nL links x nT 5-minute steps),
    # links in milepost order (index grows downstream). HV/HQ: weekday x time-of-day profile of speed and
    # flow, same shape. a: first gap column (T+1). par: per-link parameters from physics.link_params.
    # vcut: per-link speed (km/h) at or below which a cell counts as queued.
    # Returns a dict of feature name -> float array (nL x LG); column s is gap step s (T+1+s).
    nL = V.shape[0]
    def last_finite(X, lo, hi):          # last finite value in columns [lo, hi), and its distance to a
        """Returns the last finite value per link in columns [lo, hi) and its distance (steps) to column a."""
        val = np.full(nL, np.nan); dist = np.full(nL, np.nan)
        for c in range(hi - 1, lo - 1, -1):
            if c < 0 or c >= nT: continue
            x = X[:, c]; m = np.isnan(val) & np.isfinite(x)
            val[m] = x[m]; dist[m] = a - c
        return val, dist
    def first_finite(X, lo, hi):
        """Returns the first finite value per link in columns [lo, hi) and its distance (steps) after the gap end."""
        val = np.full(nL, np.nan); dist = np.full(nL, np.nan)
        for c in range(lo, hi):
            if c < 0 or c >= nT: continue
            x = X[:, c]; m = np.isnan(val) & np.isfinite(x)
            val[m] = x[m]; dist[m] = c - (a + LG - 1)
        return val, dist
    def win_mean(X, lo, hi):
        """Returns the per-link NaN-mean over columns [lo, hi), clipped to the day; NaN when the range is empty."""
        lo, hi = max(lo, 0), min(hi, nT)
        if hi <= lo: return np.full(nL, np.nan)
        with np.errstate(all='ignore'):
            import warnings; warnings.simplefilter('ignore')
            return np.nanmean(X[:, lo:hi], axis=1)
    # e: first column after the gap.
    e = a + LG
    f = {}
    # Last observation before the gap (columns T-12..T) and first observation within the hour after the gap.
    pv, pvd = last_finite(V, a - 13, a); pq, _ = last_finite(Q, a - 13, a); po, _ = last_finite(O, a - 13, a)
    sv, svd = first_finite(V, e, e + 12); sq, _ = first_finite(Q, e, e + 12); so, _ = first_finite(O, e, e + 12)
    # 20-minute (4-step) means right before and right after the gap, for the observed layer and the profile.
    pv3, pq3 = win_mean(V, a - 4, a), win_mean(Q, a - 4, a)
    sv3, sq3 = win_mean(V, e, e + 4), win_mean(Q, e, e + 4)
    hv_pre, hq_pre = win_mean(HV, a - 4, a), win_mean(HQ, a - 4, a)
    hv_post, hq_post = win_mean(HV, e, e + 4), win_mean(HQ, e, e + 4)
    # Observed / profile ratios: how far demand (flow) and speed sit above or below the usual level.
    with np.errstate(all='ignore'):
        rq_pre, rq_post = pq3 / hq_pre, sq3 / hq_post
        rv_pre, rv_post = pv3 / hv_pre, sv3 / hv_post
    # corridor-wide demand level relative to the profile
    with np.errstate(all='ignore'):
        import warnings; warnings.simplefilter('ignore')
        rq_all_pre = np.nanmedian(rq_pre); rq_all_post = np.nanmedian(rq_post)
    rq_pre_f = np.where(np.isfinite(rq_pre), rq_pre, rq_all_pre); rq_post_f = np.where(np.isfinite(rq_post), rq_post, rq_all_post)
    # Lowest observed speed per link in the hour before the gap.
    vmin_hist = np.nanmin(np.where(np.isfinite(V[:, max(a-13,0):a]), V[:, max(a-13,0):a], np.inf), axis=1); vmin_hist[~np.isfinite(vmin_hist)] = np.nan
    # s: gap step 0..LG-1; w: weight of a straight line from the value before the gap to the value after it.
    s = np.arange(LG)[None, :]; w = (s + 1) / (LG + 1)
    cols = lambda x: np.repeat(np.asarray(x, float)[:, None], LG, 1)
    # Profile speed and flow at each gap cell.
    tidx = np.clip(a + s, 0, nT - 1).repeat(nL, 0)
    HVg, HQg = HV[np.arange(nL)[:, None], tidx], HQ[np.arange(nL)[:, None], tidx]
    # Fallback chain for the anchors: observation, then profile, then the anchor on the other side of the gap.
    pvf = np.where(np.isfinite(pv), pv, hv_pre); svf = np.where(np.isfinite(sv), sv, hv_post)
    pqf = np.where(np.isfinite(pq), pq, hq_pre); sqf = np.where(np.isfinite(sq), sq, hq_post)
    svf = np.where(np.isfinite(svf), svf, pvf); sqf = np.where(np.isfinite(sqf), sqf, pqf)
    # Linear interpolation across the gap, and the profile scaled by the interpolated demand ratio.
    f['step'] = np.repeat(s, nL, 0).astype(float); f['w'] = np.repeat(w, nL, 0)
    f['lin_v'] = cols(pvf) + (cols(svf) - cols(pvf)) * w
    f['lin_q'] = cols(pqf) + (cols(sqf) - cols(pqf)) * w
    r_int = cols(rq_pre_f) + (cols(rq_post_f) - cols(rq_pre_f)) * w
    f['hr_q'] = HQg * r_int
    f['h_v'] = HVg; f['h_q'] = HQg
    # Raw anchors and ratios, broadcast over the gap steps.
    for k, x in (('pv', pv), ('pvd', pvd), ('pq', pq), ('po', po), ('sv', sv), ('svd', svd), ('sq', sq), ('so', so),
                 ('pv3', pv3), ('pq3', pq3), ('sv3', sv3), ('sq3', sq3), ('rq_pre', rq_pre), ('rq_post', rq_post),
                 ('rv_pre', rv_pre), ('rv_post', rv_post), ('vmin_hist', vmin_hist)):
        f[k] = cols(x)
    f['rq_all_pre'] = np.full((nL, LG), rq_all_pre); f['rq_all_post'] = np.full((nL, LG), rq_all_post)
    for d in (-2, -1, 1, 2):            # upstream (-) / downstream (+) neighbours by milepost order
        for k, x in (('pv', pvf), ('sv', svf), ('pq', pqf), ('sq', sqf)):
            # y[i] = x[i + d]; neighbours past either end of the corridor are NaN.
            y = np.roll(x, -d);
            if d > 0: y[-d:] = np.nan
            else: y[:-d] = np.nan
            f[f'{k}_n{d:+d}'] = cols(y)
    # Queue flags: anchor speed at or below vcut before / after the gap.
    f['queue_pre'] = cols((pvf <= vcut).astype(float)); f['queue_post'] = cols((svf <= vcut).astype(float))
    # tod: time-of-day column index of each gap cell (5-minute steps from midnight).
    f['tod'] = (a + np.repeat(s, nL, 0)).astype(float)
    for c in ('lanes', 'free_speed_kmh', 'capacity_vph', 'length_km'):
        f[c] = cols(par[c])
    # rank: relative position along the corridor, 0 = most upstream link, 1 = most downstream.
    f['rank'] = cols(np.arange(nL) / max(nL - 1, 1))
    return f

def profile(days, month_of, lo_month=None):
    """Unused placeholder; returns None."""
    pass

def main():
    """Builds the G2 gap-model training table for one panel from the train months.

    For every replicated official Task 2 window on a train day, simulates the gap T+1..T+18, computes
    `gap_features` from the masked layer plus the fully restored window history, and keeps a random
    sample of the eligible gap cells with their true speed and flow as labels.

    Command-line args: --panel (corridor name, e.g. D7_I210_W) and --sample (share of eligible cells kept
    outside February 2031, default 0.35).

    Writes g2/<panel>_profile_all.npz (weekday profiles from all train months, used at serving time) and
    g2/<panel>_rows.parquet (one row per kept cell: features, panel, date, origin T, link index li,
    window condition, masking regime, is_target, is_g1, labels y_v, y_q and y_ql = flow per lane).
    """
    ap = argparse.ArgumentParser(); ap.add_argument('--panel', required=True); ap.add_argument('--sample', type=float, default=0.35)
    a = ap.parse_args(); t0 = time.time()
    OUT = HERE / 'g2'; OUT.mkdir(exist_ok=True)
    panel = a.panel
    # Truth and masked day files keyed by file name; each masked file carries its masking regime (R1..R3).
    tfiles = {p.name: p for p in R.truth_files(panel, 'train')}
    mfiles = {}
    for rg in R.REGIME_ORDER:
        for p in R.masked_files(panel, 'train', rg): mfiles[p.name] = (p, rg)
    names = sorted(tfiles)
    days = []
    for nm in names:
        g = grid.build(panel, tfiles[nm], mfiles[nm][0])
        days.append(g)
    links = [str(x) for x in days[0].links]; nL, nT = len(links), len(days[0].times)
    # vcut: a link is queued at or below 60% of its free-flow speed.
    par = physics.link_params(panel, links); vcut = 0.6 * par['free_speed_kmh']
    lanes = np.maximum(par['lanes'], 1.0)
    # profiles: sums per (month, weekday)
    months = sorted({g.date[:7] for g in days})
    S = {}; C = {}
    for g in days:
        key = (g.date[:7], pd.Timestamp(g.date).weekday())
        ok = g.eligible & np.isfinite(g.truth['speed_kmh']) & np.isfinite(g.truth['flow_vph'])
        for fld in ('speed_kmh', 'flow_vph'):
            S.setdefault((fld,) + key, np.zeros((nL, nT))); C.setdefault((fld,) + key, np.zeros((nL, nT)))
            S[(fld,) + key] += np.where(ok, g.truth[fld], 0.0); C[(fld,) + key] += ok
    def prof(fld, wd, exclude_month):
        """Returns the mean truth profile of `fld` for weekday `wd` over all months except `exclude_month`."""
        s = sum(v for k, v in S.items() if k[0] == fld and k[2] == wd and k[1] != exclude_month)
        c = sum(v for k, v in C.items() if k[0] == fld and k[2] == wd and k[1] != exclude_month)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    # serving profiles (all months) saved for validation/private
    # The npz holds one (nL x nT) array per field and weekday, named '<field>_<weekday 0-6>', plus 'links'.
    np.savez_compressed(OUT / f'{panel}_profile_all.npz', **{f'{fld}_{wd}': prof(fld, wd, None) for fld in ('speed_kmh', 'flow_vph') for wd in range(7)}, links=np.array(links))
    # Official Task 2 windows replayed on the train months by the replicated official selector.
    wins = pd.read_parquet(ROOT / 'build/review_20260923/work/e2/replicated_unique.parquet')
    wins = wins[wins.panel == panel]
    # Optional list of February 2031 windows that marks rows with is_g1.
    g1w = pd.read_csv(HERE / 'g1' / f'{panel}_2031_02_windows.csv') if (HERE / 'g1' / f'{panel}_2031_02_windows.csv').exists() else None
    byd = {g.date: i for i, g in enumerate(days)}
    # Fixed seed so the cell sample is reproducible.
    rng = np.random.default_rng(7)
    rows = []; pc = {}
    for _, w in wins.iterrows():
        if w.date not in byd: continue
        # The gap starts one step after the forecast origin T and needs a full history and a full gap in the day.
        g = days[byd[w.date]]; T = int(w.origin); a_ = T + 1
        if a_ + LG > nT or T - HISTN < 0: continue
        wd = pd.Timestamp(g.date).weekday(); mo = g.date[:7]
        k = (wd, mo)
        # Leave-month-out profile: the window's own month is excluded, so the label month is unseen.
        if k not in pc: pc[k] = (prof('speed_kmh', wd, mo), prof('flow_vph', wd, mo))
        HV, HQ = pc[k]
        V = g.value['speed_kmh'].copy(); Q = g.value['flow_vph'].copy(); O = g.value['occupancy'].copy()
        # window history republished in full: restore truth at eligible cells of T-12..T-1
        hs = slice(T - HISTN, T)
        el = g.eligible[:, hs]
        for X, fld in ((V, 'speed_kmh'), (Q, 'flow_vph'), (O, 'occupancy')):
            X[:, hs] = np.where(el & np.isnan(X[:, hs]), g.truth[fld][:, hs], X[:, hs])
        # the gap itself is never read, but blank it for safety
        for X in (V, Q, O): X[:, a_:a_ + LG] = np.nan
        f = gap_features(V, Q, O, HV, HQ, a_, nT, par, vcut)
        # Labels: true speed and flow inside the gap, on eligible cells only.
        yv = g.truth['speed_kmh'][:, a_:a_ + LG]; yq = g.truth['flow_vph'][:, a_:a_ + LG]
        el = g.eligible[:, a_:a_ + LG] & np.isfinite(yv) & np.isfinite(yq)
        tgt = g.target[:, a_:a_ + LG]
        # February 2031 is the held-out test month: all of its eligible cells are kept.
        is_test = g.date >= '2031-02-01'
        is_g1 = bool(g1w is not None and ((g1w.date == w.date) & (g1w.origin == w.origin)).any())
        keep = el & ((rng.random(el.shape) < a.sample) | is_test)
        if not keep.any(): continue
        li, si = np.where(keep)
        d = {c: v[li, si].astype(np.float32) for c, v in f.items()}
        d.update(panel=panel, date=g.date, origin=T, li=li.astype(np.int16), cond=w.condition,
                 regime=mfiles[names[byd[w.date]]][1], is_target=tgt[li, si], is_g1=is_g1,
                 y_v=yv[li, si].astype(np.float32), y_q=yq[li, si].astype(np.float32), y_ql=(yq[li, si] / lanes[li]).astype(np.float32))
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(OUT / f'{panel}_rows.parquet', index=False)
    print(panel, 'rows', len(df), 'gaps', df.groupby(['date', 'origin']).ngroups, 'secs', round(time.time() - t0, 1))

if __name__ == '__main__':
    main()
