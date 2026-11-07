"""Feb full-background frame builder + scorer (mirrors build/review_20260927_structural/density_experiment.evaluate, strict mode).

Candidate fields: 'cap' = CAP Feb confirmation candidate (model_capacity_20260927/state/confirmation/predictions.npz),
'dk50' = its parent (full DK50). Holes = HMP50 strict (round(2*strict_pool-strict_t1,3), flow ch_ql).
Scorer returns per (panel, day) sums; summarize() reproduces the official-form macro (panel x regime -> panel -> family).
"""
# Research helper (no main); state_density/j1_pilot.py imports it to build
# February 2031 grids and score candidates. CAP: the capacity-model state
# candidate (model_capacity_20260927); DK50: the density-head candidate of
# review_20260927_structural.
import os
# Caps the BLAS/OpenMP thread pools at 2 before numpy loads.
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']: os.environ[k] = '2'
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
STRUCT = ROOT / 'build/review_20260927_structural'; RL = ROOT / 'build/review_20260924_ctm/rl'
sys.path.insert(0, str(RL)); sys.path.insert(0, str(STRUCT)); sys.path.insert(0, str(ROOT))
import rleval as E
import matched_background as MATCH

# KEYS: general target cells of the February predictions; PRED: CAP and DK50
# values on them; HOLES: February HMP50 hole predictions. Each is loaded once
# (cached in _keys, _pred, _holes).
KEYS = STRUCT / 'density/feb_prediction_keys.parquet'
PRED = ROOT / 'build/model_capacity_20260927/state/confirmation/predictions.npz'
HOLES = ROOT / 'build/review_20260927_research/feb_pool/predictions.parquet'
_keys = None; _pred = None; _holes = None


def keys():
    """Returns the (panel_i, day, li, ci) keys of the February general target cells (cached)."""
    global _keys
    if _keys is None: _keys = pd.read_parquet(KEYS, columns=['panel_i', 'day', 'li', 'ci'])
    return _keys


def preds():
    """Returns the candidate arrays {'cap': (v, q), 'dk50': (v, q)} (speed km/h, per-lane flow; cached)."""
    global _pred
    if _pred is None:
        z = np.load(PRED); _pred = {'cap': (z['candidate_v'], z['candidate_q']), 'dk50': (z['parent_v'], z['parent_q'])}
    return _pred


def holes():
    """Returns the February hole predictions (strict HMP50 parts, parent flow, truth; cached)."""
    global _holes
    if _holes is None: _holes = pd.read_parquet(HOLES, columns=['panel', 'date', 'origin', 'li', 'step_i', 'ch_ql', 'y_v', 'y_ql', 'is_target', 'strict_pool', 'strict_t1'])
    return _holes


def frame(p, cand='cap'):
    """Return dict of full-month grids for panel p (links x days*288) with candidate target values filled."""
    # Returned keys: v / q (candidate grids, total flow), vt / qt (truth), Nt
    # (true storage, vehicles per link), masks (valid LWR transitions of frames
    # A and B), memA / memB (frame supports), fdp (per-lane FD parameters) and
    # sel / li / ti (index of the candidate target cells).
    pi = E.PANELS10.index(p); z, meta = MATCH.load(p, '2031_02'); nd = len(meta['dates'])
    stack = lambda f: np.concatenate([z[f'd{i}_{f}'] for i in range(nd)], axis=1)
    ln = np.maximum(z['ln'], 1)[:, None]; L = z['par_length_km'][:, None]; lanes = np.maximum(z['par_lanes'], 1)[:, None]
    ok = stack('ok'); rt = stack('rt') & ok; gm = stack('gm'); vt = stack('vt').astype(float); qt = stack('qt').astype(float)
    obs = np.isfinite(stack('obsv')) & np.isfinite(stack('obsq')); rv = stack('rv')
    k = keys(); sel = np.flatnonzero(k.panel_i.to_numpy() == pi); li = k.li.to_numpy(int)[sel]; ti = k.day.to_numpy(int)[sel] * 288 + k.ci.to_numpy(int)[sel]
    assert rt[li, ti].all() and not gm[li, ti].any()
    cv, cq = preds()[cand]
    v = stack('v1').astype(float); q = stack('q1').astype(float)
    h = holes(); h = h[h.panel == p]; dmap = {d: i for i, d in enumerate(meta['dates'])}
    hq = q.copy()
    if len(h):
        hli = h.li.to_numpy(int); hti = h.date.map(dmap).to_numpy(int) * 288 + h.origin.to_numpy(int) + 1 + h.step_i.to_numpy(int)
        assert gm[hli, hti].all() and np.array_equal(rt[hli, hti], h.is_target.to_numpy(bool))
        hq[hli, hti] = np.round(h.ch_ql.to_numpy() * ln[hli, 0], 3)
        hv = np.round(2 * h['strict_pool'] - h['strict_t1'], 3).to_numpy()
    q = hq
    v[li, ti] = cv[sel]; q[li, ti] = cq[sel] * ln[li, 0]
    if len(h): v[hli, hti] = hv
    v = np.where(rt, v, vt); q = np.where(rt, q, qt)
    # legal transition masks (frame A/B), cross-midnight kept only for consecutive same-regime days
    Nt = qt / np.maximum(vt, 1) * L
    memA = ok & np.isfinite(Nt); memB = memA & (rt | obs)
    masks = {}
    for f, mem in (('A', memA), ('B', memB)):
        mm = mem[:, :-1] & mem[:, 1:] & z['keep'][:, None] & rv[:, :-1]
        for i in range(nd - 1):
            if meta['regimes'][i] != meta['regimes'][i + 1] or (pd.Timestamp(meta['dates'][i + 1]) - pd.Timestamp(meta['dates'][i])).days != 1: mm[:, (i + 1) * 288 - 1] = False
        masks[f] = mm
    fdp = dict(vf=z['par_free_speed_kmh'][:, None], cap=z['par_capacity_vph'][:, None] / lanes, kc=z['par_critical_density'][:, None] / lanes, kj=z['par_k_jam'][:, None] / lanes)
    return dict(p=p, meta=meta, nd=nd, ln=ln, L=L, lanes=lanes, ok=ok, rt=rt, gm=gm, vt=vt, qt=qt, obs=obs, v=v, q=q, Nt=Nt, masks=masks, memB=memB, memA=memA, fdp=fdp,
                sel=sel, li=li, ti=ti)


def score(F, v=None, q=None, anatomy=False, cong=None):
    """Per-day sums for grids v,q (total flow). Defaults to F['v'],F['q']."""
    # Returns one dict per day: state SSEs (ssev; sseq per lane), LWR error
    # numerator / denominator (num / den) and FD error sums (fdn / fdd) for
    # frames A and B. With `anatomy`, the frame-B LWR error is also split by
    # target-target / target-observed transitions, hole / general cells and
    # congested (true speed < 0.6 x free-flow) / free cells.
    v = F['v'] if v is None else v; q = F['q'] if q is None else q
    rt, vt, qt, ln, L, lanes = F['rt'], F['vt'], F['qt'], F['ln'], F['L'], F['lanes']
    n = q / np.maximum(v, 1) * L; err = n - F['Nt']; de = np.diff(err, axis=1); dt = np.diff(F['Nt'], axis=1)
    kl = n / L / lanes; ql = q / lanes; fp = F['fdp']
    fd = np.where(kl <= fp['kc'], fp['vf'] * kl, fp['cap'] / np.maximum(fp['kj'] - fp['kc'], 1e-9) * np.maximum(fp['kj'] - kl, 0))
    out = []; T = n.shape[1]
    rl, rr = rt[:, :-1], rt[:, 1:]
    if cong is None: cong = F['vt'] < 0.6 * fp['vf']
    for di, date in enumerate(F['meta']['dates']):
        sl = slice(di * 288, (di + 1) * 288); es = slice(di * 288, min((di + 1) * 288, T - 1)); tg = rt[:, sl]
        row = dict(panel=F['p'], date=date, regime=F['meta']['regimes'][di], n=int(tg.sum()),
                   ssev=float(((v[:, sl] - vt[:, sl])[tg] ** 2).sum()), sseq=float((((q[:, sl] - qt[:, sl]) / ln)[tg] ** 2).sum()))
        for f in 'AB':
            tm = F['masks'][f][:, es]; fm = (F['memA'] if f == 'A' else F['memB'])[:, sl]
            row[f'num{f}'] = float(np.abs(de[:, es])[tm].sum()); row[f'den{f}'] = float(np.abs(dt[:, es])[tm].sum())
            row[f'fdn{f}'] = float(np.abs(ql[:, sl] - fd[:, sl])[fm].sum()); row[f'fdd{f}'] = float(np.abs(ql[:, sl])[fm].sum())
        if anatomy:
            tm = F['masks']['B'][:, es]; a = np.abs(de[:, es]); d0 = np.abs(dt[:, es]); L2 = rl[:, es]; R2 = rr[:, es]
            gmm = F['gm'][:, :-1][:, es] | F['gm'][:, 1:][:, es]; cg = cong[:, :-1][:, es] | cong[:, 1:][:, es]
            for kind, edge in (('tt', L2 & R2), ('to', L2 ^ R2)):
                for gname, gsel in (('hole', gmm), ('gen', ~gmm)):
                    for sname, ssel in (('cong', cg), ('free', ~cg)):
                        take = tm & edge & gsel & ssel
                        row[f'num_{kind}_{gname}_{sname}'] = float(a[take].sum()); row[f'den_{kind}_{gname}_{sname}'] = float(d0[take].sum()); row[f'n_{kind}_{gname}_{sname}'] = int(take.sum())
            take = tm & ~L2 & ~R2; row['den_oo'] = float(d0[take].sum()); row['num_oo'] = float(a[take].sum())
        out.append(row)
    return out


def summarize(a):
    """Turns per-day sums into proxy scores (same formula as density_experiment.summarize).

    Returns:
      (mean, perpanel, grouped): scores per candidate, per panel and
      candidate, and per panel/candidate/regime.
    """
    g = a.groupby(['panel', 'cand', 'regime']).sum(numeric_only=True)
    g['S_state'] = .54 * (1 - np.sqrt(g.ssev / g.n) / 25).clip(lower=0) + .46 * (1 - np.sqrt(g.sseq / g.n) / 600).clip(lower=0)
    for f in 'AB':
        g['S_FD_' + f] = (1 - g['fdn' + f] / g['fdd' + f]).clip(lower=0); g['S_LWR_' + f] = (1 - g['num' + f] / g['den' + f]).clip(lower=0)
        g['S_phys_' + f] = (g['S_FD_' + f] + 2 * g['S_LWR_' + f]) / 3; g['total_' + f] = .35 * g.S_state + .15 * g['S_phys_' + f]
    metrics = ['S_state'] + [c + '_' + f for f in 'AB' for c in ['S_FD', 'S_LWR', 'S_phys', 'total']]
    p = g[metrics].groupby(['panel', 'cand']).mean().reset_index(); p['family'] = p.panel.map(E.FAM)
    mean = p.groupby(['family', 'cand'])[metrics].mean().groupby('cand').mean()
    return mean, p, g
