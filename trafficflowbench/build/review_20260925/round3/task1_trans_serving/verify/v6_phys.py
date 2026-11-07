"""Skeptic V6: in-world Feb 2031 PHYSICS effect of the speed-only correction (the pilot did not recompute physics frames).
Per panel/day: rebuild the harness's v2F day fields = C0 (c0_cache v0/q0) with every hole cell of res/*_v2_cmp_cells
replaced by v2F_v / v2F_ql*lanes, recompute the eval_v2.py accumulators (sse_v_in, num/den/fdn/fdd for frames A, B) and
assert they reproduce res/*_v2_cmp_days.csv (cand v2F); then replace the hole speeds by the corrected speeds
(verify/work/v5_inworld_preds.npz: serve / strict), flow unchanged, and recompute.  S_state / S_phys via eval_v2.summarize.
-> verify/results/v6_phys.json"""
# Research record: main() needs February in-world tables (c0_cache, res/,
# inworld_apply_rows, v5_inworld_preds) that are not shipped. The
# reproduction imports this file through review_20260927/density_holes.py
# (for acc, summarize and physics) when the HM25 hole-mixture step loads
# iteration2/hole_mixture.py.
import os, sys, json
os.environ['OMP_NUM_THREADS'] = '3'
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; LENS = HERE.parent
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
G2V2 = ROOT / 'build/review_20260924_ctm/g2v2'; sys.path.insert(0, str(G2V2)); sys.path.insert(0, str(ROOT / 'build/review_20260924/work'))
from src.tfb import physics
from eval_v2 import summarize
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
# Hole length in 5-minute slots (90 min).
LG = 18


def acc(v, q, z, k, L, lanesc, vf, capl, kcl, kjl, keep):
    """Accumulates the state and physics score terms of one day.

    Args:
      v: (L, T) speed of the candidate state, km/h.
      q: (L, T) flow of the candidate state, veh/h.
      z: day cache with arrays '<k>_<name>': truth TV and TQ, released V and
        Q, eligibility E, lanes ln, and the masks rt, gm and rv.
      k: day key, 'd<day index>'.
      L: (L, 1) link lengths, km.
      lanesc: (L, 1) lanes.
      vf: (L, 1) free-flow speed, km/h.
      capl: (L, 1) capacity per lane, veh/h.
      kcl: (L, 1) critical density per lane, veh/km.
      kjl: (L, 1) jam density per lane, veh/km.
      keep: (L,) links that are scored.

    Returns:
      Dict of sums. n, sse_v, sse_q (flow per lane) over the target cells
      (rt with truth) where gm holds ('in') and where it does not ('out').
      For frames A and B: num/den, the absolute error of the change of the
      vehicle count N = q / v x L between consecutive slots against the
      truth change, and the truth total (LWR conservation); fdn/fdd, the
      absolute gap between per-lane flow and the triangular fundamental
      diagram flow at the same density, and the per-lane flow total. Frame A
      uses every eligible cell with truth on scored links; frame B only those
      that are targets or observed.
    """
    ln = np.maximum(z[f'{k}_ln'].astype(float), 1.0)[:, None] * np.ones_like(v)
    tvf, tqf = z[f'{k}_TV'].astype(float), z[f'{k}_TQ'].astype(float); E = z[f'{k}_E']; V, Q = z[f'{k}_V'], z[f'{k}_Q']
    rt = z[f'{k}_rt']; gm = z[f'{k}_gm']; rv = z[f'{k}_rv']
    tg = rt & np.isfinite(tvf) & np.isfinite(tqf); okc = E & np.isfinite(tvf) & np.isfinite(tqf)
    vt = np.where(okc, tvf, np.nan); qt = np.where(okc, tqf, np.nan); obs = np.isfinite(V) & np.isfinite(Q)
    Nt = qt / np.clip(vt, 1, None) * L
    memA = okc & np.isfinite(Nt) & keep[:, None]; memB = memA & (tg | obs)
    r = {}
    for kind, m in (('in', tg & gm), ('out', tg & ~gm)):
        r[f'n_{kind}'] = int(m.sum()); r[f'sse_v_{kind}'] = float(((v[m] - vt[m]) ** 2).sum()); r[f'sse_q_{kind}'] = float((((q[m] - qt[m]) / ln[m]) ** 2).sum())
    vp = np.where(tg, v, vt); qp = np.where(tg, q, qt); Np = qp / np.clip(vp, 1, None) * L
    for fr, mem in (('A', memA), ('B', memB)):
        m = mem[:, 1:] & mem[:, :-1] & rv[:, :-1]
        dN = Np[:, 1:] - Np[:, :-1]; dNt = Nt[:, 1:] - Nt[:, :-1]
        r[f'num{fr}'] = float(np.abs(dN - dNt)[m].sum()); r[f'den{fr}'] = float(np.abs(dNt)[m].sum())
        kl = Np / L / lanesc; ql = qp / lanesc
        qfd = np.where(kl <= kcl, vf * kl, (capl / np.maximum(kjl - kcl, 1e-9)) * np.maximum(kjl - kl, 0))
        mm = mem & np.isfinite(ql) & np.isfinite(qfd)
        r[f'fdn{fr}'] = float(np.abs(ql - qfd)[mm].sum()); r[f'fdd{fr}'] = float(np.abs(ql)[mm].sum())
    return r


def main():
    """Scores the parent and the corrected hole speeds on February and writes v6_phys.json."""
    A = pd.read_parquet(LENS / 'work' / 'inworld_apply_rows.parquet', columns=['panel', 'date', 'origin', 'li', 'step_i', 'ch_v'])
    P = np.load(HERE / 'work' / 'v5_inworld_preds.npz'); A['serve'] = P['serve']; A['strict'] = P['strict']
    out = {}; allrows = []
    for panel in PANELS:
        z = dict(np.load(G2V2 / 'c0_cache' / f'{panel}_2031_02_v2.npz', allow_pickle=True)); meta = json.loads(str(z.pop('meta')))
        links = meta['links']; par = physics.link_params(panel, links); keep = physics.scoreable_mask(panel, links, 'train', meta['times'])
        lanes = np.maximum(par['lanes'], 1.0); L = par['length_km'][:, None]; lanesc = lanes[:, None]; vf = par['free_speed_kmh'][:, None]
        capl = par['capacity_vph'][:, None] / lanesc; kcl = par['critical_density'][:, None] / lanesc; kjl = par['k_jam'][:, None] / lanesc
        cl = pd.read_parquet(G2V2 / 'res' / f'{panel}_2031_02_v2_cmp_cells.parquet', columns=['date', 'origin', 'li', 'step', 'v2F_v', 'v2F_ql'])
        cl = cl.merge(A[A.panel == panel].rename(columns={'step_i': 'step'}), on=['date', 'origin', 'li', 'step'], how='left', validate='one_to_one')
        assert cl.serve.notna().all() and np.abs(cl.ch_v - cl.v2F_v).max() == 0
        days = pd.read_csv(G2V2 / 'res' / f'{panel}_2031_02_v2_cmp_days.csv'); ref = days[days.cand == 'v2F'].set_index('date')
        rows = []; mxrel = 0.0
        for di, date in enumerate(meta['dates']):
            k = f'd{di}'; v0, q0 = z[f'{k}_v0'].astype(float), z[f'{k}_q0'].astype(float)
            c = cl[cl.date == date]; v = v0.copy(); q = q0.copy()
            if len(c):
                r_ = c.li.to_numpy().astype(int); t_ = (c.origin.to_numpy() + 1 + c.step.to_numpy()).astype(int)
                v[r_, t_] = c.v2F_v.to_numpy(); q[r_, t_] = c.v2F_ql.to_numpy().astype(float) * lanes[r_]
            base = acc(v, q, z, k, L, lanesc, vf, capl, kcl, kjl, keep)
            rr = ref.loc[date]
            for kk in ('sse_v_in', 'numB', 'denB', 'fdnB', 'fddB', 'numA', 'fdnA'):
                mxrel = max(mxrel, abs(base[kk] - rr[kk]) / max(abs(rr[kk]), 1e-9))
            rows.append(dict(date=date, regime=meta['regimes'][di], cand='v2F_rebuilt', **base))
            for nm in ('serve', 'strict'):
                vv = v.copy()
                if len(c): vv[r_, t_] = c[nm].to_numpy()
                rows.append(dict(date=date, regime=meta['regimes'][di], cand=nm, **acc(vv, q, z, k, L, lanesc, vf, capl, kcl, kjl, keep)))
        d = pd.DataFrame(rows); d['panel'] = panel; allrows.append(d)
        s = {c: summarize(d[d.cand == c]) for c in d.cand.unique()}
        o = {'rebuild_max_rel_vs_harness': mxrel}
        for nm in ('serve', 'strict'):
            for kk in ('S_state', 'S_phys_A', 'S_phys_B', 'S_LWR_B'):
                o[f'{nm}_d{kk}'] = float(s[nm][kk] - s['v2F_rebuilt'][kk])
        o['base_S_phys_B'] = float(s['v2F_rebuilt']['S_phys_B'])
        out[panel] = o; print(panel, {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in o.items()}, flush=True)
    for nm in ('serve', 'strict'):
        for kk in ('S_state', 'S_phys_A', 'S_phys_B'):
            out[f'mean_{nm}_d{kk}'] = float(np.mean([out[p][f'{nm}_d{kk}'] for p in PANELS]))
        out[f'{nm}_dLB_proxy_035dS_state_015dS_physB'] = 0.35 * out[f'mean_{nm}_dS_state'] + 0.15 * out[f'mean_{nm}_dS_phys_B']
    print({k: v for k, v in out.items() if k.startswith('mean') or 'dLB' in k})
    pd.concat(allrows).to_csv(HERE / 'work' / 'v6_phys_days.csv', index=False)
    json.dump(out, open(HERE / 'results' / 'v6_phys.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
