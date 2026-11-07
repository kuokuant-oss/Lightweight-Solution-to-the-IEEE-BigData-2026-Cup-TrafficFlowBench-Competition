"""Lovász hinge as a LightGBM custom objective, per window.

notes/18 A. Yu & Blaschko (arXiv:1512.07797) give "a novel surrogate loss
function for submodular losses, the Lovász hinge, which leads to O(p log p)
complexity with O(p) oracle accesses to the loss function to compute a gradient
or cutting-plane". The Jaccard loss is submodular, so this is a convex surrogate
for 1 - IoU itself.

Why this is the replacement for what was rejected. notes/16 G argued that a
set-level metric needs a set-level objective and that per-sample classification
losses are both misaligned and sensitive to the positive/negative ratio, but
concluded there was no GBDT precedent because AP-Loss and RS-Loss both need
backpropagation. What was missing was a gradient. The Lovász hinge supplies one
in O(p log p), and a gradient is all LightGBM's fobj asks for.

    balanced (rejected, -0.046055)  moved the class weight ratio
    focal    (rejected, -0.059847)  reweighted by how well-classified a sample is
    this                            optimises the set loss itself

Formulation follows bermanmaxim/LovaszSoftmax's lovasz_hinge_flat, with one
window playing the role of one image (notes/06 B: the official metric is
per-window IoU, so per_image=True is the right analogue):

    signs   = 2y - 1
    errors  = 1 - z * signs
    sort errors descending, permute the labels the same way
    g       = the Lovász extension's gradient of the Jaccard set function
    loss    = <relu(errors_sorted), g>

and therefore

    dloss/dz_i = -signs_i * g_{rank(i)} * 1[errors_i > 0]

Three choices this module makes, declared because none is transcription:

  * HESSIAN = |gradient|. The surrogate is piecewise linear, so the second
    derivative is 0 almost everywhere and LightGBM, which divides by the summed
    hessian to set a leaf, needs a substitute.

    A constant 1.0 was declared first and measured to stall training, so the
    reason for the change is recorded rather than the choice being quietly
    tuned. The hinge margin makes only the hard rows active: on a real onset
    fold at iteration 0, just 0.77% of rows carry a non-zero gradient, about
    10.5 per window, with |g| around 0.04 on those. With h = 1 every one of the
    other 99.2% still contributes hessian, so a 1000-row leaf sums h to 1000
    against sum|g| of roughly 0.7 and the leaf lands at 2.5e-4 -- four orders of
    magnitude under the margin scale of 1. Early stopping then fired at
    iteration 1.

    h = |g| instead makes a leaf -sum(g) / sum|g|, which lies in [-1, 1] and is
    exactly the hinge's own margin scale. It is scale-free, introduces no
    tunable quantity, and gives inactive rows h = 0 so they neither move nor
    dilute a leaf. scripts/84 verifies the gradient against numerical
    differentiation; the hessian is a substitute, not a derivative.

  * NO SAMPLE WEIGHTS. scripts/51 gives every window total weight 1 via
    1/n_cells. The Lovász gradient is already per-window normalised -- g sums to
    at most 1 over a window regardless of its size -- so applying 1/n_cells on
    top would normalise twice and make large windows nearly gradient-free. The
    two mechanisms express the same intent, so this objective carries the
    normalisation and the weights are dropped.

  * INIT_SCORE IS KEPT. The control boosts from logit(queue_prior) and so does
    this, leaving the objective as the only difference. Note the semantics
    change: hinge scores are margins, not log-odds, and logit(prior) has a scale
    of roughly 4 while the hinge margin is 1. A consequence, not a bug: the
    gradient is inactive wherever the margin is already satisfied, which
    concentrates it on positives and on negatives scoring above -1. That is
    hard-example focus arrived at by construction rather than by a knob.

Raw scores are no longer log-odds, so sigmoid(raw) is only a monotone squashing.
That is harmless here because every arm calibrates its own thresholds
(src/tfb/task2abl.py) -- which is exactly why fixing the threshold NUMBER rather
than the PROCEDURE had to be corrected before this candidate could be judged.
"""
from __future__ import annotations

import numpy as np

HESSIAN = "abs_grad"      # see the module docstring; 1.0 was tried and stalled


def lovasz_grad(gt_sorted: np.ndarray) -> np.ndarray:
    """Gradient of the Lovász extension of the Jaccard set function."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - np.cumsum(gt_sorted)
    union = gts + np.cumsum(1.0 - gt_sorted)
    jaccard = 1.0 - intersection / np.maximum(union, 1e-12)
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def window_loss_grad(z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Lovász hinge loss and its gradient for one window."""
    signs = 2.0 * y - 1.0
    errors = 1.0 - z * signs
    order = np.argsort(-errors, kind="mergesort")
    es = errors[order]
    g = lovasz_grad(y[order])
    active = es > 0
    loss = float(np.dot(np.where(active, es, 0.0), g))
    grad = np.empty_like(z, dtype=float)
    grad[order] = -signs[order] * g * active
    return loss, grad


def _bounds(groups: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) spans of equal group id."""
    cuts = np.flatnonzero(np.diff(groups)) + 1
    edges = np.concatenate([[0], cuts, [len(groups)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def loss_and_grad(z: np.ndarray, y: np.ndarray, spans) -> tuple[float, np.ndarray]:
    total = 0.0
    grad = np.zeros_like(z, dtype=float)
    for a, b in spans:
        l, gg = window_loss_grad(z[a:b], y[a:b])
        total += l
        grad[a:b] = gg
    return total, grad


def make_objective(groups: np.ndarray):
    """fobj over windows. ``groups`` must align row-for-row with the Dataset and
    be contiguous per window; task2lov_build sorts the frame to guarantee it."""
    spans = _bounds(groups)

    def fobj(preds, dataset):
        y = dataset.get_label()
        _, g = loss_and_grad(np.asarray(preds, dtype=float), y, spans)
        return g, np.abs(g)
    return fobj


def make_eval(groups: np.ndarray):
    """feval reporting the mean per-window Lovász hinge, the objective itself."""
    spans = _bounds(groups)

    def feval(preds, dataset):
        y = dataset.get_label()
        total, _ = loss_and_grad(np.asarray(preds, dtype=float), y, spans)
        return "lovasz", total / max(len(spans), 1), False
    return feval


def _prep(frame, feats):
    """Rows sorted so each window is contiguous, plus its group array."""
    import pandas as pd
    f = frame.sort_values("window_id", kind="mergesort")
    groups = pd.factorize(f.window_id, sort=False)[0]
    return f, groups


LOGIT_EPS = 1e-4
MARGIN_SCALE = float(np.log((1 - LOGIT_EPS) / LOGIT_EPS))   # 9.2103, |logit(eps)|


def scaled_init(prior: np.ndarray) -> np.ndarray:
    """logit(prior) mapped onto the hinge's own margin range, [-1, 1].

    scripts/84 and 85 rejected the Lovász hinge, but the measured cause was an
    interaction rather than the loss: the hinge margin is 1 while
    logit(queue_prior) has a median |z| of 9.21, because OOF.logit clips at
    1e-4 and |logit(1e-4)| is 9.2103. Nine tenths of the dynamic range sat in
    the flat part of the loss, leaving 2.79% of onset rows and 6.15% of ongoing
    rows with any gradient.

    Dividing by that same clip constant is not a tuned scale -- it is the exact
    factor that carries the existing encoding onto the interval the hinge
    measures margins in.
    """
    return OOF_logit(prior) / MARGIN_SCALE


def OOF_logit(p):
    from . import task2oof as OOF
    return OOF.logit(p)


def predictor_none(m, d, feats):
    """Scores for an arm trained WITHOUT init_score: no prior is added back."""
    raw = m.predict(d[feats], raw_score=True)
    return 1.0 / (1.0 + np.exp(-raw))


def predictor_scaled(m, d, feats):
    """Scores for an arm trained with the rescaled init_score."""
    raw = m.predict(d[feats], raw_score=True) + scaled_init(d.queue_prior.to_numpy())
    return 1.0 / (1.0 + np.exp(-raw))


def lgb_fitter(feval_factory=None, init: str = "logit"):
    """A fitter for task2abl.Arm.

    ``feval_factory`` is called with the validation group array and must return
    a LightGBM feval; the default is the Lovász hinge itself, which is the
    source-faithful pairing of objective and stopping metric. Passing a
    different factory is how scripts/85 aligns the stopping rule across arms.

    No sample weights: see the module docstring -- the Lovász gradient is
    already per-window normalised, and sum|grad| over a window is 1 whatever the
    window's size, which scripts/84 checks.
    """
    def _init(frame):
        if init == "none":
            return None
        if init == "scaled":
            return scaled_init(frame.queue_prior.to_numpy())
        return OOF_logit(frame.queue_prior.to_numpy())

    def fit(train, valid, feats):
        import lightgbm as lgb
        from . import task2oof as OOF
        tr, gtr = _prep(train, feats)
        va, gva = _prep(valid, feats)
        P = {k: v for k, v in OOF.PARAMS.items() if k not in ("objective", "metric")}
        P["objective"] = make_objective(gtr)
        P["metric"] = "None"
        dtrain = lgb.Dataset(tr[feats], tr.label, init_score=_init(tr))
        dvalid = lgb.Dataset(va[feats], va.label, init_score=_init(va),
                             reference=dtrain)
        fe = make_eval(gva) if feval_factory is None else feval_factory(va, gva)
        m = lgb.train(P, dtrain, num_boost_round=OOF.NUM_BOOST_ROUND,
                      valid_sets=[dvalid], feval=fe,
                      callbacks=[lgb.early_stopping(OOF.EARLY_STOPPING, verbose=False),
                                 lgb.log_evaluation(0)])
        return m, int(m.best_iteration or OOF.NUM_BOOST_ROUND)
    return fit
