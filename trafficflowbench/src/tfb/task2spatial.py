"""Data-weighted spatial aggregation, as a static per-(panel, link, link) lookup.

notes/18 C. SA-LSTM (arXiv:2402.05663) "integrat[es] Self-Attention (SA) on the
spatial dimension" over a matrix whose rows are road segments and columns are
consecutive time steps, and its argument is that a bottleneck creates
dependencies across NEIGHBOURING SEGMENTS as well as across time -- so the
model should decide which neighbours to look at rather than being told.

Our spatial features are the opposite of that: margin_up/dn_{1,2,4} sample a
fixed hop count, and margin_shock samples one fixed wave speed, which scripts/80
and 81 measured to be nearly inert whatever speed it uses. Nothing in the table
lets the data say which links matter to which.

A GBDT cannot learn attention weights, so the transferable piece is the
representation, not the mechanism: precompute how strongly each pair of links
moves together, from the fold's own training days, and use that as the weight in
a spatial aggregate. That is the static, training-free analogue of attention, and
it is isomorphic to queue_prior -- accumulated over the same fold-source days, so
nothing from the held-out fold enters.

Weights are correlations of the links' speed series, pooled over the training
days, with negative correlations clipped to zero and each row normalised. Two
declared choices, neither tuned:

  * clipping negatives to zero, because a neighbour that moves OPPOSITE to this
    link is not evidence about congestion arriving here, and keeping it would
    let two unrelated links cancel inside a weighted mean;
  * missing readings filled with the link's own mean before accumulating, so a
    detector outage contributes no deviation rather than dropping the day.

Aggregates are split upstream and downstream, because the two directions carry
different physics -- a congested front arrives from downstream while demand
arrives from upstream -- and a single symmetric mean would blend them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import release as R, score2, task2build as TB

# Role: task2feat.build_window_features adds these aggregates only when given
# `spatial_w`; the reproduction pipeline does not pass it, so this is a
# research block. EPS guards divisions by near-zero sums.
EPS = 1e-9


def panel_weights(panel: str, source_dates: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised non-negative correlation weights, split up / down.

    Returns (W_up, W_dn), each (n_links x n_links); row i holds the weights link
    i gives its upstream (lower milepost) or downstream neighbours, self excluded.
    """
    # Reads train-split truth days in `source_dates` only. Missing speeds are
    # filled with the link's own mean for that day, and the correlations pool
    # every stamp of every source day.
    # Raises RuntimeError when no source day matches.
    cut = score2.v_cut(panel)
    from .task2pool import date_of
    S = sx = sxx = None
    n = 0.0
    mp = None
    for path in R.truth_files(panel, "train"):
        if date_of(path) not in source_dates:
            continue
        links, milepost, V, Qf, O, E, Q, times = TB.day_arrays(panel, path, cut)
        if S is None:
            p = len(links)
            S = np.zeros((p, p))
            sx = np.zeros(p)
            sxx = np.zeros(p)
            mp = milepost
        X = V.copy()
        rm = np.nanmean(X, axis=1)
        rm = np.where(np.isfinite(rm), rm, 0.0)
        idx = ~np.isfinite(X)
        X[idx] = np.broadcast_to(rm[:, None], X.shape)[idx]
        S += X @ X.T
        sx += X.sum(axis=1)
        sxx += (X * X).sum(axis=1)
        n += X.shape[1]
    if S is None:
        raise RuntimeError(f"{panel}: no source day matched")
    mean = sx / n
    var = np.maximum(sxx / n - mean * mean, 0.0)
    sd = np.sqrt(var)
    cov = S / n - np.outer(mean, mean)
    denom = np.outer(sd, sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > EPS, cov / np.maximum(denom, EPS), 0.0)
    np.fill_diagonal(corr, 0.0)
    w = np.clip(np.nan_to_num(corr), 0.0, None)
    # Up and down follow milepost order (higher milepost = downstream);
    # task2dir's sign is not applied here.
    upper = mp[None, :] > mp[:, None]          # column j is downstream of row i
    lower = mp[None, :] < mp[:, None]
    def norm(m):
        """Keeps one side of the weights and normalises each row to sum 1."""
        x = w * m
        return x / np.maximum(x.sum(axis=1, keepdims=True), EPS)
    return norm(lower), norm(upper)


def aggregate(w_up: np.ndarray, w_dn: np.ndarray, margin_last: np.ndarray,
              trend: np.ndarray, queued: np.ndarray) -> dict[str, np.ndarray]:
    """Spatial aggregates under the data-derived weights."""
    # w_up, w_dn: (n_links x n_links) weights from panel_weights. margin_last,
    # trend, queued: per-link margin, 3-step speed trend and queued flag at
    # the last history stamp. Returns the nine FEATURES arrays.
    out = {
        "sa_margin_up": w_up @ margin_last,
        "sa_margin_dn": w_dn @ margin_last,
        "sa_trend_up": w_up @ trend,
        "sa_trend_dn": w_dn @ trend,
        "sa_queued_up": w_up @ queued,
        "sa_queued_dn": w_dn @ queued,
    }
    # how concentrated each link's dependency is: 1 means one neighbour carries
    # everything, near 0 means the link is coupled to the whole corridor
    out["sa_conc_up"] = (w_up ** 2).sum(axis=1)
    out["sa_conc_dn"] = (w_dn ** 2).sum(axis=1)
    # the gap the fixed-hop features cannot express: is the informative
    # neighbourhood upstream or downstream of this link right now
    out["sa_margin_gap"] = out["sa_margin_up"] - out["sa_margin_dn"]
    return out


FEATURES = ["sa_margin_up", "sa_margin_dn", "sa_trend_up", "sa_trend_dn",
            "sa_queued_up", "sa_queued_dn", "sa_conc_up", "sa_conc_dn",
            "sa_margin_gap"]


def weights_for(panels: list[str], source_dates: set[str]):
    """Returns per-panel (W_up, W_dn) weights and a diagnostics table.

    The table lists, per panel, the link count and the median number of
    non-zero and of effective (1 / sum of squared weights) neighbours.
    """
    out, diag = {}, []
    for p in panels:
        wu, wd = panel_weights(p, source_dates)
        out[p] = (wu, wd)
        nz = (wu > 0).sum(axis=1) + (wd > 0).sum(axis=1)
        eff = 1.0 / np.maximum((wu ** 2).sum(axis=1) + (wd ** 2).sum(axis=1), EPS)
        diag.append({"panel": p, "n_links": wu.shape[0],
                     "median_nonzero_neighbours": float(np.median(nz)),
                     "median_effective_neighbours": float(np.median(eff))})
    return out, pd.DataFrame(diag)
