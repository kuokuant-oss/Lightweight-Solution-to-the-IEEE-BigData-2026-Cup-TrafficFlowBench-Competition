"""Learned second-stage false-positive classifier for Task 2 components.

notes/10 A. Heng's Severstal kernel filters each connected component
individually -- ``if p.sum() > min_size`` -- rather than clearing the whole
image. The fixed-size version of that has already been tested here and
rejected: onset truth is about 3 cells, so any size floor deletes real
positives (calib 1 -> 0.870413, 2 -> 0.628050). This module is the learned
generalisation notes/10 A left unimplemented: keep the per-component decision,
replace ``size > min_size`` with a classifier that reads the component's shape,
its probability mass and its context.

Why removing a pure false-positive component is always right, and what that
makes the target: for a window with intersection I and union U, dropping a
component holding n cells of which i are true moves IoU to
(I-i) / (U - n + i). At i = 0 that is I / (U - n) > I / U for any n > 0. So a
component with no overlap can always be dropped, and the label is exactly
"does this component touch the truth at all".

Components use the same definition as task2eval.connected_filter -- scipy
ndimage.label over the (link x step) image with links in milepost order -- so
the learned rule and the rejected fixed-size rule act on the same objects.

Every feature is computed from stage-1 output and the static network only. No
label enters a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEEP_TAU = 0.5          # declared in advance; the guardrail forbids sweeping it
PARAMS = dict(boosting_type="gbdt", objective="binary", metric="binary_logloss",
              num_leaves=31, learning_rate=0.05, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=5, min_child_samples=20,
              verbose=-1, num_threads=0, seed=20260918, deterministic=True)
NUM_BOOST_ROUND = 300
FEATURES = ["n_cells", "n_links", "n_steps", "fill", "prob_mean", "prob_max",
            "prob_min", "prob_std", "prior_mean", "prior_max", "milepost_span",
            "step_min", "step_max", "n_components", "window_cells",
            "share_of_pred", "pred_cells"]

# notes/18 B. scripts/79 used FEATURES alone -- every one of them a description
# of the component's own interior -- and reached a removal precision of 0.469.
# Pure-FP and true components share a median size of 1 cell, so interior
# descriptors cannot separate them even in principle. The cascade literature
# says the same thing: a second stage built on descriptors from the same region
# is "not discriminative enough", and a context region is what lifts precision.
# These look OUTSIDE the component.
CONTEXT_FEATURES = [
    "ring_size", "ring_prob_mean", "ring_prob_max", "ring_prior_mean",
    "ring_minus_comp_prob",          # is the component a peak, or part of a slope
    "up_prob_mean", "dn_prob_mean",  # immediately upstream / downstream of the span
    "dist_links_nearest", "dist_steps_nearest",
    "prob_rank_in_window", "prob_gap_to_best", "share_of_prob_mass",
    "milepost_rel", "step_span_frac",
]
MULTISCALE_FEATURES = [f"ring{r}_{k}" for r in (2, 4)
                       for k in ("size", "prob_mean", "prior_mean", "minus_comp_prob")
                       ] + ["ring_falloff_1_4"]
ALL_FEATURES = FEATURES + CONTEXT_FEATURES
MULTISCALE_ALL = ALL_FEATURES + MULTISCALE_FEATURES


# notes/19 A. STDA-YOLOv8 (Sensors 25(9):2861) addresses "false positives and
# missed detections" with "multi-scale dilated convolutions, where convolutional
# kernels with different dilation rates capture contextual information from
# various receptive fields". Our single 1-ring is the single receptive field the
# source argues against.
#
# THE SOURCE'S RATES 1,3,5 ARE NOT COPYABLE and notes/19 A says so: the step axis
# here is only 6 cells, so radius 5 already spans the whole horizon. The declared
# transfer is to reuse the ladder this project's own feature table already uses
# for spatial neighbours, task2feat.NEIGHBOUR_HOPS = (1, 2, 4), rather than to
# import a number from a different grid. On the step axis a radius of r reaches
# r of the 6 cells, so 4 is the widest that still leaves the ring outside the
# component on that axis.
#
# What makes this affordable NOW and not before: notes/19 A flags feature count
# against stage-2 sample count as the risk, and scripts/96's inner CV moved that
# from ~700 components per fold to ~7,500, so sample count is no longer binding.
MULTISCALE_RADII = (2, 4)


def _context(img, lab, k, prob_img, prior_img, valid, multiscale: bool = False):
    """Features of the neighbourhood OUTSIDE component k. notes/18 B, notes/19 A."""
    from scipy import ndimage
    m = lab == k
    ring = ndimage.binary_dilation(m) & ~m & valid
    rp = prob_img[ring]
    out = {"ring_size": int(ring.sum()),
           "ring_prob_mean": float(rp.mean()) if rp.size else 0.0,
           "ring_prob_max": float(rp.max()) if rp.size else 0.0,
           "ring_prior_mean": float(prior_img[ring].mean()) if rp.size else 0.0}
    out["ring_minus_comp_prob"] = out["ring_prob_mean"] - float(prob_img[m].mean())
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    lo, hi = int(rows.min()), int(rows.max())
    seg = slice(int(cols.min()), int(cols.max()) + 1)
    up = prob_img[lo - 1, seg] if lo - 1 >= 0 else np.empty(0)
    dn = prob_img[hi + 1, seg] if hi + 1 < prob_img.shape[0] else np.empty(0)
    out["up_prob_mean"] = float(up.mean()) if up.size else 0.0
    out["dn_prob_mean"] = float(dn.mean()) if dn.size else 0.0
    other = (lab > 0) & ~m
    if other.any():
        orow, ocol = np.nonzero(other)
        out["dist_links_nearest"] = float(np.abs(orow[:, None] - rows[None, :]).min())
        out["dist_steps_nearest"] = float(np.abs(ocol[:, None] - cols[None, :]).min())
    else:
        out["dist_links_nearest"] = float(prob_img.shape[0])
        out["dist_steps_nearest"] = float(prob_img.shape[1])
    if multiscale:
        comp_p = float(prob_img[m].mean())
        for r in MULTISCALE_RADII:
            wide = ndimage.binary_dilation(m, iterations=r) & ~m & valid
            wp = prob_img[wide]
            out[f"ring{r}_size"] = int(wide.sum())
            out[f"ring{r}_prob_mean"] = float(wp.mean()) if wp.size else 0.0
            out[f"ring{r}_prior_mean"] = float(prior_img[wide].mean()) if wp.size else 0.0
            out[f"ring{r}_minus_comp_prob"] = (out[f"ring{r}_prob_mean"] - comp_p
                                               if wp.size else 0.0)
        # the multi-scale signal proper: how fast probability falls away from the
        # component. A true component sits on a slope; a pure FP sits on flat noise.
        out["ring_falloff_1_4"] = out["ring_prob_mean"] - out[f"ring{MULTISCALE_RADII[-1]}_prob_mean"]
    return out


def component_table(d: pd.DataFrame, pred: np.ndarray, with_label: bool = True,
                    context: bool = False, multiscale: bool = False):
    """One row per connected component, plus the cell -> component map.

    Returns (table, comp_of_row). comp_of_row is -1 for cells that are not
    predicted positive.
    """
    from scipy import ndimage
    comp_of_row = np.full(len(d), -1, dtype=np.int64)
    prob = d.prob.to_numpy()
    prior = d.queue_prior.to_numpy() if "queue_prior" in d.columns else np.zeros(len(d))
    lab_arr = d.label.to_numpy() if with_label else None
    mile = d.milepost.to_numpy()
    step = d.step.to_numpy()
    rows = []
    gid = 0
    for wid, idx in d.groupby("window_id", sort=False).indices.items():
        p = pred[idx]
        if p.sum() == 0:
            continue
        links = np.sort(pd.unique(mile[idx]))
        rank = {v: i for i, v in enumerate(links)}
        r = np.array([rank[v] for v in mile[idx]])
        c = step[idx] - 1
        img = np.zeros((len(links), int(c.max()) + 1), dtype=bool)
        img[r, c] = p > 0
        lab, n = ndimage.label(img)
        if n == 0:
            continue
        cl = lab[r, c]
        if context:
            shape = img.shape
            prob_img = np.zeros(shape)
            prior_img = np.zeros(shape)
            valid = np.zeros(shape, dtype=bool)
            prob_img[r, c] = prob[idx]
            prior_img[r, c] = prior[idx]
            valid[r, c] = True
        window_cells = len(idx)
        pred_cells = int((p > 0).sum())
        cond = d.condition.to_numpy()[idx][0]
        for k in range(1, n + 1):
            m = cl == k
            if not m.any():
                continue
            gi = idx[m]
            comp_of_row[gi] = gid
            pr = prob[gi]
            row = {"comp_id": gid, "window_id": wid, "condition": cond,
                   "n_cells": int(m.sum()),
                   "n_links": int(len(np.unique(r[m]))),
                   "n_steps": int(len(np.unique(c[m]))),
                   "prob_mean": float(pr.mean()), "prob_max": float(pr.max()),
                   "prob_min": float(pr.min()), "prob_std": float(pr.std()),
                   "prior_mean": float(prior[gi].mean()),
                   "prior_max": float(prior[gi].max()),
                   "milepost_span": float(mile[gi].max() - mile[gi].min()),
                   "step_min": int(step[gi].min()), "step_max": int(step[gi].max()),
                   "n_components": n, "window_cells": window_cells,
                   "pred_cells": pred_cells}
            row["fill"] = row["n_cells"] / max(row["n_links"] * row["n_steps"], 1)
            row["share_of_pred"] = row["n_cells"] / max(pred_cells, 1)
            if context:
                row.update(_context(img, lab, k, prob_img, prior_img, valid,
                                    multiscale=multiscale))
                row["share_of_prob_mass"] = float(pr.sum() / max(prob[idx][p > 0].sum(), 1e-9))
                row["milepost_rel"] = float(
                    (mile[gi].mean() - mile[idx].min())
                    / max(mile[idx].max() - mile[idx].min(), 1e-9))
                row["step_span_frac"] = row["n_steps"] / 6.0
            if with_label:
                row["y"] = int(lab_arr[gi].max() > 0)      # touches the truth at all
                row["n_true"] = int((lab_arr[gi] > 0).sum())
            rows.append(row)
            gid += 1
    tbl = pd.DataFrame(rows)
    if context and not tbl.empty:
        # rank and gap need every component in the window, so they come last
        g = tbl.groupby("window_id").prob_mean
        tbl["prob_rank_in_window"] = g.rank(pct=True)
        tbl["prob_gap_to_best"] = g.transform("max") - tbl.prob_mean
    return tbl, comp_of_row


def fit(tbl: pd.DataFrame, features: list[str] | None = None):
    """One booster per condition. Returns {condition: model}."""
    import lightgbm as lgb
    feats = FEATURES if features is None else features
    out = {}
    for cond, g in tbl.groupby("condition", sort=True):
        if g.y.nunique() < 2 or len(g) < 50:
            out[cond] = None                      # nothing to learn from
            continue
        ds = lgb.Dataset(g[feats], g.y)
        out[cond] = lgb.train(PARAMS, ds, num_boost_round=NUM_BOOST_ROUND)
    return out


def apply_filter(d: pd.DataFrame, pred: np.ndarray, models: dict,
                 tau: float = KEEP_TAU, features: list[str] | None = None) -> np.ndarray:
    """Drop components the classifier scores below ``tau``."""
    feats = FEATURES if features is None else features
    need_ctx = any(f in feats for f in CONTEXT_FEATURES)
    need_ms = any(f in feats for f in MULTISCALE_FEATURES)
    tbl, comp_of_row = component_table(d, pred, with_label=False, context=need_ctx,
                                       multiscale=need_ms)
    if tbl.empty:
        return pred
    keep_prob = np.ones(len(tbl))
    for cond, g in tbl.groupby("condition", sort=True):
        m = models.get(cond)
        if m is None:
            continue
        keep_prob[g.index.to_numpy()] = m.predict(g[feats])
    drop = set(tbl.comp_id.to_numpy()[keep_prob < tau])
    if not drop:
        return pred
    out = pred.copy()
    out[np.isin(comp_of_row, list(drop))] = 0
    return out


# ---------------------------------------------------------------- decision rule

def touch_ratio(tbl: pd.DataFrame) -> dict[str, float]:
    """Mean true-cell FRACTION among components that touch the truth, per condition.

    The one quantity the expected-IoU rule needs beyond the classifier, estimated
    on the same calibration block the classifier is fitted on. A fraction, not a
    count: a one-cell component that touches the truth holds exactly one true
    cell, so the expected content has to scale with the component's own size.
    """
    out = {}
    for cond, g in tbl.groupby("condition", sort=True):
        t = g[g.y > 0]
        out[cond] = float((t.n_true / t.n_cells).mean()) if len(t) else 1.0
    return out


def apply_expected_iou(d: pd.DataFrame, pred: np.ndarray, models: dict,
                       touch: dict[str, float], features: list[str]) -> np.ndarray:
    """Drop a component only when the plug-in expected IoU says it pays.

    tau = 0.5 asks "is this component more likely false than true", which is not
    the question the metric asks. Removing a pure false positive always helps --
    IoU goes from I/U to I/(U-n) -- but removing one that touches the truth costs
    intersection, and the cost grows with how much truth it holds. So the
    confidence needed to drop should depend on the component's SIZE, not sit at a
    constant. scripts/79 measured what a constant gives: 1,054 removals at 0.469
    precision.

    Plug-in estimates, using the same probabilities the model already produces
    (the failure analysis measured Sum(p)/n_true at 1.000 and 1.063, so the mass
    is close to the true set size):

        P   predicted cells            T = sum(p) over all cells
        I   sum(p) over predicted      U = P + T - I
        i_c = q_c * (mean true FRACTION among touchers) * n_c, capped at n_c

    and a component is kept when I/U >= (I - i_c) / (U - n_c + i_c).

    Each component is judged against the full prediction rather than sequentially,
    so the rule carries no ordering and no threshold.
    """
    need_ctx = any(f in features for f in CONTEXT_FEATURES)
    need_ms = any(f in features for f in MULTISCALE_FEATURES)
    tbl, comp_of_row = component_table(d, pred, with_label=False, context=need_ctx,
                                       multiscale=need_ms)
    if tbl.empty:
        return pred
    q = np.ones(len(tbl))
    for cond, g in tbl.groupby("condition", sort=True):
        m = models.get(cond)
        if m is not None:
            q[g.index.to_numpy()] = m.predict(g[features])
    prob = d.prob.to_numpy()
    drop = []
    for wid, idx in d.groupby("window_id", sort=False).indices.items():
        sel = tbl.window_id.to_numpy() == wid
        if not sel.any():
            continue
        p = pred[idx] > 0
        P = float(p.sum())
        if P == 0:
            continue
        T = float(prob[idx].sum())
        I = float(prob[idx][p].sum())
        U = P + T - I
        if U <= 0:
            continue
        base = I / U
        for j in np.flatnonzero(sel):
            row = tbl.iloc[j]
            n = float(row.n_cells)
            i_c = min(float(q[j]) * touch.get(row.condition, 1.0) * n, n)
            den = U - n + i_c
            if den <= 0:
                continue
            if (I - i_c) / den > base:
                drop.append(int(row.comp_id))
    if not drop:
        return pred
    out = pred.copy()
    out[np.isin(comp_of_row, drop)] = 0
    return out
