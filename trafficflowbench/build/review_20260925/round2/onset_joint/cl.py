"""Conditional-logit machinery + feature construction for per-site rows (train rows / pv rows / official rows)."""
# Imported in the reproduction through s2_ruler.py and s6_fe_lib.py, which
# serve the FE site choice of synthesis/sv2_official_FEA1.py. A conditional
# logit compares the candidate sites of one window: P(site j) =
# exp(s_j) / sum over the window's sites of exp(s_k), with score s = X w.
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from pathlib import Path
ME = Path(__file__).resolve().parent
# PHASE: per 'panel|site', the 5-minute slot (0-5) of the 30-minute demand
# cycle in which a demand step reaches the bottleneck (site_phase.json).
# BASEF: load-ratio (flow / capacity) features at T: rT (at T), r3, r6, r12
# (means of the last 3, 6, 12 stamps), dr (r6 minus the 6 stamps before),
# rmx3 (max of the last 3). GAPF: features compared across a window's sites.
PHASE = {k: v['phase'] for k, v in json.load(open(ME / 'site_phase.json')).items()}
BASEF = ['r6', 'rT', 'r3', 'r12', 'dr', 'rmx3']
GAPF = ['r6', 'rT', 'rmx3', 'r3']


def add_feats(d):
    """d: per (key, site) rows with base features, T, panel, site, date. Adds relative phase, relative-to-others, tod, weekend."""
    # Added columns: ph0..ph5 (slot of T+30 in the cycle), kr0..kr5 (slots
    # from the latest demand-step arrival at the site to T+30), tod (T as a
    # fraction of the day), wkend, gap_<f> (value minus the largest value of
    # the window's other sites) and rk_<f> (rank in the window, 0 = largest).
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
    # Windows without a positive row are dropped. lam is the L2 penalty.
    # L-BFGS-B minimises the penalised negative log-likelihood with its
    # gradient. Returns the weight vector, one weight per column of X.
    o = np.argsort(grp, kind='stable'); X = X[o]; y = y[o].astype(float); g = np.asarray(grp)[o]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]; gi = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(g)]))
    keep = np.add.reduceat(y, starts) > 0
    rk = keep[gi]; X = X[rk]; y = y[rk]; g = g[rk]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]; gi = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(g)]))
    p_ = X.shape[1]

    def f(w):
        """Returns the penalised loss and its gradient at weights `w`."""
        s = X @ w
        smax = np.maximum.reduceat(s, starts)[gi]; e = np.exp(s - smax)
        Z = np.add.reduceat(e, starts); P = np.add.reduceat(e * y, starts)
        loss = -np.sum(np.log(P) - np.log(Z)) + 0.5 * lam * w @ w
        pr = e / Z[gi]; q = y * e / P[gi]
        return loss, X.T @ (pr - q) + lam * w
    r = minimize(f, np.zeros(p_) if w0 is None else w0, jac=True, method='L-BFGS-B', options=dict(maxiter=2000))
    return r.x


class Std:
    """Column standardiser that ignores NaN and maps NaN to 0 after scaling."""
    def fit(self, X):
        """Stores column means and standard deviations (1 for constant columns); returns self."""
        self.m = np.nanmean(X, 0); self.s = np.nanstd(X, 0); self.s[~(self.s > 1e-9)] = 1.0; return self

    def tr(self, X):
        """Returns standardised X with NaN set to 0 (the column mean)."""
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
    """Conditional-logit site scorer for one panel, or pooled panels.

    Attributes:
      spec: design spec (see design()); lam: L2 penalty.
      sites: sorted site names seen in fit; cols: design column names.
      st: fitted Std; w: fitted weights.
    """
    # Stores the design spec and the L2 penalty.
    def __init__(self, spec, lam=1.0): self.spec = spec; self.lam = lam

    def fit(self, d):
        """Fits the weights on rows with key (window), site and y (1 = queued); returns self."""
        self.sites = sorted(d.site.unique())
        X, self.cols = design(d, self.spec, self.sites); self.st = Std().fit(X)
        self.w = fit_cl(self.st.tr(X), d.key.to_numpy(), d.y.to_numpy(), self.lam); return self

    def score(self, d):
        """Returns the linear score of each row (higher = more likely the queuing site)."""
        X, _ = design(d, self.spec, self.sites); return self.st.tr(X) @ self.w
