"""Focal loss as a LightGBM custom objective, for Task 2.

notes/17 C. Lin et al., arXiv:1708.02002: the focal loss "down-weights the loss
assigned to well-classified examples". That mechanism is NOT the one already
rejected here -- per-window class-balanced weighting changed the positive/negative
total weight ratio and lost 0.046055. Focal leaves the class ratio alone and
re-weights by how well each example is already classified, which matches the
measured shape of the problem: 97.3% of windows are near-perfect and the loss
sits in the remaining 2.7%.

alpha and gamma are knobs and the guardrail forbids sweeping them, so only the
paper's alpha=0.25, gamma=2 is used.

Written analytically rather than with the finite differences of
jrzaurin/LightGBM-with-Focal-Loss: same loss, but exact and cheap over three
million rows. The derivation uses p_t = sigmoid(s*z) with s = 2y-1, which makes
both derivatives compact:

    q      = sigmoid(s*z)                     = p_t
    L      = -a_t * (1-q)^g * log q
    dL/dz  = s * a_t * [ g*q*(1-q)^g*log q - (1-q)^(g+1) ]
    d2L/dz2= a_t * q*(1-q) * [ g*( (1-q)^g*log q - g*q*(1-q)^(g-1)*log q
                                   + (1-q)^g ) + (g+1)*(1-q)^g ]

(s^2 = 1, so the second derivative needs no sign factor.)

scripts/77 checks both against numerical derivatives and checks that gamma=0,
alpha=0.5 reproduces LightGBM's built-in binary objective, which also pins down
whether sample weights reach a custom objective. They do not: on LightGBM 4.7 a
custom objective is handed unweighted gradients, so the 1/n_cells window
weighting has to be applied inside fobj or it silently disappears.
"""
from __future__ import annotations

import numpy as np

# Role: no program of the reproduction pipeline imports this module; it is the
# record of the focal-loss training experiment.
# ALPHA: weight of positive rows (negatives get 1 - ALPHA). GAMMA: focusing
# exponent; larger values down-weight well-classified rows more. EPS keeps
# probabilities away from 0 and 1 before the log.
ALPHA, GAMMA = 0.25, 2.0        # Lin et al. arXiv:1708.02002; not swept
EPS = 1e-9
# The focal loss is not convex in the raw score for gamma > 0 -- about 13% of
# points carry a negative second derivative. LightGBM divides by the summed
# hessian to set a leaf, so a negative sum produces a nonsense leaf value. The
# floor is a numerical safeguard, not a tuned quantity: it binds only where the
# loss is locally concave, and at gamma = 0 it never binds at all, which is why
# the equivalence check in scripts/77 still passes exactly.
HESS_FLOOR = 1e-6


def _parts(z: np.ndarray, y: np.ndarray, alpha: float, gamma: float):
    """Returns (s, a_t, q): label sign (+1/-1), class weight and clipped p_t.

    ``gamma`` is accepted for a uniform signature and is not used here.
    """
    s = 2.0 * y - 1.0                       # +1 for a positive, -1 for a negative
    a_t = np.where(y > 0, alpha, 1.0 - alpha)
    q = 1.0 / (1.0 + np.exp(-s * z))        # = p_t
    q = np.clip(q, EPS, 1.0 - EPS)
    return s, a_t, q


def grad_hess(z: np.ndarray, y: np.ndarray, alpha: float = ALPHA,
              gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray]:
    """Returns the focal-loss gradient and hessian with respect to raw scores.

    Args:
        z: Raw scores (log-odds), one per row.
        y: 0/1 labels.
        alpha: Weight of positive rows.
        gamma: Focusing exponent.

    Returns:
        (grad, hess), each shaped like ``z``; hess is floored at HESS_FLOOR.
    """
    s, a_t, q = _parts(z, y, alpha, gamma)
    lg = np.log(q)
    om = 1.0 - q
    om_g = om ** gamma
    grad = s * a_t * (gamma * q * om_g * lg - om * om_g)
    inner = gamma * (om_g * lg - gamma * q * (om ** (gamma - 1.0)) * lg + om_g) \
        + (gamma + 1.0) * om_g
    hess = a_t * q * om * inner
    return grad, np.maximum(hess, HESS_FLOOR)


def loss(z: np.ndarray, y: np.ndarray, alpha: float = ALPHA,
         gamma: float = GAMMA) -> np.ndarray:
    """Returns the per-row focal loss for raw scores ``z`` and 0/1 labels ``y``."""
    _, a_t, q = _parts(z, y, alpha, gamma)
    return -a_t * ((1.0 - q) ** gamma) * np.log(q)


def lgb_objective(alpha: float = ALPHA, gamma: float = GAMMA, weight=None):
    """fobj for lgb.train. Sample weights are applied here explicitly.

    LightGBM does not multiply a custom objective's gradients by the Dataset
    weights, so the 1/n_cells window weighting scripts/51 relies on has to be
    applied by hand or it silently disappears. scripts/77 verifies this.
    """
    # `weight`, when given, replaces the Dataset weights.
    def fobj(preds, dataset):
        """Returns weighted (grad, hess) for LightGBM's raw scores ``preds``."""
        y = dataset.get_label()
        g, h = grad_hess(preds, y, alpha, gamma)
        w = dataset.get_weight() if weight is None else weight
        if w is not None:
            g = g * w
            h = h * w
        return g, h
    return fobj


def logloss_objective():
    """Plain weighted binary logloss, written as a CUSTOM objective.

    Needed because LightGBM hands ``feval`` different quantities depending on
    the objective: probabilities under a built-in objective, raw scores under a
    custom one. An experiment that means to give two arms the same stopping
    metric therefore cannot pair a built-in control with a custom candidate --
    the shared feval would silently read a different quantity in each arm.
    Giving both arms a custom objective removes that asymmetry.

    This is focal at gamma=0, alpha=0.5, doubled, which scripts/78 checks
    reproduces LightGBM's built-in ``binary`` objective to 0.000e+00.
    """
    def fobj(preds, dataset):
        """Returns weighted logloss (grad, hess) for raw scores ``preds``."""
        g, h = grad_hess(preds, dataset.get_label(), 0.5, 0.0)
        w = dataset.get_weight()
        g, h = 2.0 * g, 2.0 * h
        if w is not None:
            g, h = g * w, h * w
        return g, h
    return fobj


def logloss_eval(weighted: bool = True):
    """Weighted binary logloss from raw scores, as a feval.

    Used to ALIGN the stopping rule across arms. It assumes RAW SCORES, so
    every arm it is applied to must use a custom objective -- see
    logloss_objective. Applied to a built-in objective it would sigmoid an
    already-sigmoided probability and silently measure the wrong thing.

    ``weighted=False`` is for aligning across arms whose Datasets carry
    different weights -- the Lovász arm deliberately carries none -- where a
    weighted metric would not be the same metric in both arms.
    """
    def feval(preds, dataset):
        """Returns (name, mean logloss, False); lower is better."""
        y = dataset.get_label()
        w = dataset.get_weight() if weighted else None
        p = np.clip(1.0 / (1.0 + np.exp(-preds)), EPS, 1.0 - EPS)
        l = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        v = float(np.average(l, weights=w) if w is not None else l.mean())
        return "wlogloss" if weighted else "logloss", v, False
    return feval


def lgb_eval(alpha: float = ALPHA, gamma: float = GAMMA):
    """feval for early stopping. A custom objective leaves preds as raw scores,
    so the built-in binary_logloss metric would be reading the wrong quantity."""
    def feval(preds, dataset):
        """Returns ("focal_loss", mean focal loss, False); lower is better."""
        y = dataset.get_label()
        w = dataset.get_weight()
        l = loss(preds, y, alpha, gamma)
        v = float(np.average(l, weights=w) if w is not None else l.mean())
        return "focal_loss", v, False
    return feval
