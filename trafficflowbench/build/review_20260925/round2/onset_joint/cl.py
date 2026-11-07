"""Conditional-logit machinery + feature construction for per-site rows (train rows / pv rows / official rows)."""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from pathlib import Path
ME = Path(__file__).resolve().parent
PHASE = {k: v['phase'] for k, v in json.load(open(ME / 'site_phase.json')).items()}
BASEF = ['r6', 'rT', 'r3', 'r12', 'dr', 'rmx3']
GAPF = ['r6', 'rT', 'rmx3', 'r3']


def add_feats(d):
    """d: per (key, site) rows with base features, T, panel, site, date. Adds relative phase, relative-to-others, tod, weekend."""
    d = d.copy()
    for c in ['rT', 'rT1', 'rmx3']:
        d[c] = d[c].fillna(d['r3'])
    d['ph'] = (d['T'] + 6) % 6
    for k in range(6): d[f'ph{k}'] = (d.ph == k).astype(float)
    phi = np.array([PHASE[f'{p}|{s}'] for p, s in zip(d.panel, d.site)])
    d['kr'] = (d['T'].to_numpy() + 6 - phi) % 6          # 0 = demand step arrives at the bottleneck exactly at T+6
    for k in range(6): d[f'kr{k}'] = (d.kr == k).astype(float)
    d['stepin'] = ((d['T'].to_numpy() + 6 - phi) % 6 <= 5).astype(float)  # always 1 (a step lands in T+1..T+6); kept for clarity
    d['tod'] = d['T'] / 288.0
    d['wkend'] = (pd.to_datetime(d.date).dt.dayofweek >= 5).astype(float)
    g = d.groupby('key')
    for f in GAPF:
        mx = g[f].transform('max'); n = g[f].transform('size')
        # max over OTHER sites: if this row is the max, use the second largest
        srt = g[f].transform(lambda x: np.sort(x.to_numpy())[-2] if len(x) > 1 else np.nan)
        other = np.where(d[f] >= mx, srt, mx)
        d[f'gap_{f}'] = d[f] - other
        d[f'rk_{f}'] = g[f].rank(ascending=False, method='average') - 1
    return d


def fit_cl(X, grp, y, lam=1.0, w0=None):
    """Conditional logit; rows grouped by `grp` (any hashable); y 0/1, loss = -log sum_{pos} p.  X standardized outside."""
    o = np.argsort(grp, kind='stable'); X = X[o]; y = y[o].astype(float); g = np.asarray(grp)[o]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]; gi = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(g)]))
    keep = np.add.reduceat(y, starts) > 0
    rk = keep[gi]; X = X[rk]; y = y[rk]; g = g[rk]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]; gi = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(g)]))
    p_ = X.shape[1]

    def f(w):
        s = X @ w
        smax = np.maximum.reduceat(s, starts)[gi]; e = np.exp(s - smax)
        Z = np.add.reduceat(e, starts); P = np.add.reduceat(e * y, starts)
        loss = -np.sum(np.log(P) - np.log(Z)) + 0.5 * lam * w @ w
        pr = e / Z[gi]; q = y * e / P[gi]
        return loss, X.T @ (pr - q) + lam * w
    r = minimize(f, np.zeros(p_) if w0 is None else w0, jac=True, method='L-BFGS-B', options=dict(maxiter=2000))
    return r.x


class Std:
    def fit(self, X):
        self.m = np.nanmean(X, 0); self.s = np.nanstd(X, 0); self.s[~(self.s > 1e-9)] = 1.0; return self

    def tr(self, X):
        Z = (X - self.m) / self.s; Z[np.isnan(Z)] = 0.0; return Z


def design(d, spec, sites):
    """Build design matrix: spec dict with keys shared (list of cols), site_int (bool), site_slopes (list of cols interacted with site)."""
    cols = []; M = []
    for c in spec.get('shared', []): M.append(d[c].to_numpy(float)); cols.append(c)
    if spec.get('site_int', False):
        for s in sites[1:]: M.append((d.site == s).to_numpy(float)); cols.append(f'I_{s}')
    for c in spec.get('site_slopes', []):
        for s in sites: M.append(d[c].to_numpy(float) * (d.site == s).to_numpy(float)); cols.append(f'{c}x{s}')
    return np.column_stack(M), cols


class CLModel:
    def __init__(self, spec, lam=1.0): self.spec = spec; self.lam = lam

    def fit(self, d):
        self.sites = sorted(d.site.unique())
        X, self.cols = design(d, self.spec, self.sites); self.st = Std().fit(X)
        self.w = fit_cl(self.st.tr(X), d.key.to_numpy(), d.y.to_numpy(), self.lam); return self

    def score(self, d):
        X, _ = design(d, self.spec, self.sites); return self.st.tr(X) @ self.w
