"""Paired evaluator on the rebuilt g3x grids (copy of floor_audit/evallib.py, repointed to ./cache and parameterised by
month; adds the two D12_I405 panels as a fifth family for the supplementary 10-panel view).
Official-form S_state (per panel x regime RMSE over ALL targets incl. holes), local LWR proxy (rhs == dN_true) in
frame A (all eligible) and frame B (masked-layer cells + targets), official-form S_FD over the frame-B rows.
Candidates override C1 at NON-hole targets only, via arrays aligned with the feature-table rows."""
import json, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
SP = HERE / 'cache'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
PANELS10 = PANELS + ['D12_I405_N', 'D12_I405_S']
FAM = {'D7_I10_E': 'I10', 'D7_I10_W': 'I10', 'D7_I210_E': 'I210', 'D7_I210_W': 'I210', 'D7_I405_N': 'I405', 'D7_I405_S': 'I405',
       'D12_I5_N': 'I5', 'D12_I5_S': 'I5', 'D12_I405_N': 'D12_I405', 'D12_I405_S': 'D12_I405'}
_cache = {}


def load(panel, month):
    key = (panel, month)
    if key not in _cache:
        _cache.clear(); _cache[key] = (np.load(SP / f'{panel}_{month}.npz'), json.load(open(SP / f'{panel}_{month}_meta.json')))
    return _cache[key]


def evaluate(panel, month, days, cands, rows):
    """cands: {name: (v array, q_per_lane array) | (None, None)} aligned with rows (DataFrame with day, li, ci; index 0..n-1)."""
    z, meta = load(panel, month)
    ln = np.maximum(z['ln'], 1.0)[:, None]; L = z['par_length_km'][:, None]; keep = z['keep']
    lanes = np.maximum(z['par_lanes'], 1.0)[:, None]; vf = z['par_free_speed_kmh'][:, None]
    capl = z['par_capacity_vph'][:, None] / lanes; kcl = z['par_critical_density'][:, None] / lanes; kjl = z['par_k_jam'][:, None] / lanes
    out = []
    for di in days:
        rg = meta['regimes'][di]; k = f'd{di}'; ok = z[f'{k}_ok']; rt = z[f'{k}_rt'] & ok; gm = z[f'{k}_gm']; rv = z[f'{k}_rv']
        vt = np.where(ok, z[f'{k}_vt'], np.nan).astype(float); qt = np.where(ok, z[f'{k}_qt'], np.nan).astype(float)
        v1 = z[f'{k}_v1'].astype(float); q1 = z[f'{k}_q1'].astype(float)
        obs = np.isfinite(z[f'{k}_obsv']) & np.isfinite(z[f'{k}_obsq'])
        Nt = qt / np.clip(vt, 1, None) * L
        memA = ok & np.isfinite(Nt) & keep[:, None]; memB = memA & (rt | obs)
        r = rows[rows.day == di]; li = r.li.to_numpy(int); ci = r.ci.to_numpy(int); sel = r.index.to_numpy()
        for name, (cv, cq) in cands.items():
            v = v1.copy(); q = q1.copy()
            if cv is not None:
                v[li, ci] = np.clip(cv[sel], 6.0, 120.75); q[li, ci] = np.maximum(cq[sel] * ln[li, 0], 50.0)
            vp = np.where(rt, v, vt); qp = np.where(rt, q, qt)
            A = dict(cand=name, regime=rg, day=di, n=int(rt.sum()), sse_v=float(np.nansum((v - vt)[rt] ** 2)), sse_q=float(np.nansum(((q - qt) / ln)[rt] ** 2)))
            mg = rt & gm
            A.update(n_gap=int(mg.sum()), sse_v_gap=float(np.nansum((v - vt)[mg] ** 2)), sse_q_gap=float(np.nansum(((q - qt) / ln)[mg] ** 2)))
            Np = qp / np.clip(vp, 1, None) * L
            for fr, mem in (('A', memA), ('B', memB)):
                m = mem[:, 1:] & mem[:, :-1] & rv[:, :-1]
                dN = Np[:, 1:] - Np[:, :-1]; dNt = Nt[:, 1:] - Nt[:, :-1]
                A[f'num{fr}'] = float(np.sum(np.abs(dN - dNt)[m])); A[f'den{fr}'] = float(np.sum(np.abs(dNt)[m]))
            kl = Np / L / lanes; ql = qp / lanes
            qfd = np.where(kl <= kcl, vf * kl, (capl / np.maximum(kjl - kcl, 1e-9)) * np.maximum(kjl - kl, 0))
            mm = memB & np.isfinite(ql) & np.isfinite(qfd)
            A['fd_num'] = float(np.abs(ql - qfd)[mm].sum()); A['fd_den'] = float(np.abs(ql)[mm].sum())
            out.append(A)
    return out


def summarize(acc, panels=None):
    d = pd.DataFrame(acc)
    if panels is not None: d = d[d.panel.isin(panels)]
    g = d.groupby(['panel', 'cand', 'regime']).sum(numeric_only=True).reset_index()
    g['S_state'] = 0.54 * (1 - np.sqrt(g.sse_v / g.n) / 25).clip(lower=0) + 0.46 * (1 - np.sqrt(g.sse_q / g.n) / 600).clip(lower=0)
    for fr in 'AB': g[f'S_LWR_{fr}'] = (1 - g[f'num{fr}'] / g[f'den{fr}']).clip(lower=0)
    g['S_FD'] = 1 - g.fd_num / g.fd_den
    g['S_phys_B'] = (g.S_FD + 2 * g.S_LWR_B) / 3; g['S_phys_A'] = (g.S_FD + 2 * g.S_LWR_A) / 3
    p = g.groupby(['panel', 'cand']).mean(numeric_only=True).reset_index(); p['fam'] = p.panel.map(FAM)
    f = p.groupby(['cand', 'fam']).mean(numeric_only=True).reset_index().groupby('cand').mean(numeric_only=True)
    return g, p, f[['S_state', 'S_LWR_A', 'S_LWR_B', 'S_FD', 'S_phys_A', 'S_phys_B']]


def lb_delta(f, base, cand, tr=(0.73, 0.86)):
    ds = f.loc[cand, 'S_state'] - f.loc[base, 'S_state']
    dpB = f.loc[cand, 'S_phys_B'] - f.loc[base, 'S_phys_B']; dpA = f.loc[cand, 'S_phys_A'] - f.loc[base, 'S_phys_A']
    return dict(dS_state=ds, dS_physB=dpB, dS_physA=dpA, dLB_lo=0.35 * ds + 0.15 * dpB * tr[0], dLB_hi=0.35 * ds + 0.15 * dpB * tr[1])
