"""Candidate decision rules for Task 2.

The validation_v3 guardrail is frozen: its windows, its scorer, its bootstrap
and its promotion criteria live in task2eval/task2pool and are not touched
here. What was missing was a way to hand it a candidate that is not just a
threshold, which doc/plan/validation-v3.md §0 records as the interface gap.
This module is that interface.

Rules implemented so far:

  empty_fallback  When the thresholded prediction for a window is empty, take
                  the k highest-probability cells instead.

                  The official selector sets requires_queue_in_horizon = true
                  (config/synthetic_release_v1.json), so every scored window
                  has at least one queue cell in its horizon. An empty
                  prediction therefore cannot score: the union is non-empty and
                  the intersection is zero, so the window takes IoU = 0 with
                  certainty, while any non-empty guess has positive
                  probability of overlapping. The rule uses no label -- only
                  the model's own ranking -- and it changes nothing on windows
                  that already predict something.

                  notes/13 D: Vu et al.'s KiTS19 post-processing lowers the
                  threshold for empty and small first-pass masks. This is the
                  empty branch of that idea, not RankIoU: it predicts a fixed
                  small set rather than an expected-IoU-optimal set size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_iou_select(d: pd.DataFrame, prob_col: str = "prob", min_k: int = 1,
                        conditions: tuple[str, ...] | None = None,
                        fallback: np.ndarray | None = None) -> np.ndarray:
    """Per-window set size that maximises the plug-in expected IoU.

    Rank the window's cells by probability, then pick the prefix size k that
    maximises

        R(k) = E|S n Y| / E|S u Y| = S_k / (k + T - S_k)

    where S_k is the sum of the top-k probabilities and T the sum over the
    window. Both expectations follow from linearity alone, so this needs only
    MARGINAL probabilities -- it does not assume the conditional independence
    that notes/13 A records as RankSEG's blocking premise. It is still an
    approximation: a ratio of expectations is not the expectation of the ratio.

    Cost is a sort plus a prefix sum, O(m log m) per window. The o(m^3) figure
    in doc/plan/baseline-v2.md belongs to Dembczynski et al.'s EXACT
    joint-distribution algorithm, which this does not use.

    k = 0 is never chosen: the official selector sets
    requires_queue_in_horizon, so a scored window always has a positive cell
    and an empty prediction scores exactly 0.

    ``conditions`` restricts the rule to those conditions; windows of any other
    condition keep ``fallback`` (the control prediction).
    """
    out = (np.zeros(len(d), dtype=int) if fallback is None
           else np.asarray(fallback).astype(int).copy())
    sel = (np.ones(len(d), bool) if conditions is None
           else d.condition.isin(conditions).to_numpy())
    if not sel.any():
        return out
    order = np.lexsort((-d.loc[sel, prob_col].to_numpy(), d.loc[sel, "window_id"].factorize()[0]))
    idx = d.index[sel].to_numpy()[order]
    win = d.loc[idx, "window_id"].to_numpy()
    prob = d.loc[idx, prob_col].to_numpy(dtype=float)
    starts = np.flatnonzero(np.r_[True, win[1:] != win[:-1]])
    ends = np.r_[starts[1:], len(win)]
    take = []
    for a, b in zip(starts, ends):
        p = prob[a:b]
        T = p.sum()
        csum = np.cumsum(p)
        k = np.arange(1, len(p) + 1)
        R = csum / (k + T - csum)
        best = int(np.argmax(R)) + 1
        take.append(idx[a:a + max(best, min_k)])
    out[d.index.get_indexer(np.concatenate(take))] = 1
    return out


def empty_fallback(d: pd.DataFrame, pred: np.ndarray, k: int,
                   prob_col: str = "prob") -> np.ndarray:
    """Give every window whose prediction is empty its top-k cells by probability."""
    if k <= 0:
        return pred
    out = np.asarray(pred).copy()
    w = pd.Series(out).groupby(d.window_id.to_numpy()).transform("sum").to_numpy()
    empty = w == 0
    if not empty.any():
        return out
    sub = d.loc[empty, [prob_col]].assign(_w=d.loc[empty, "window_id"].to_numpy())
    take = (sub.sort_values(prob_col, ascending=False)
               .groupby("_w", sort=False).head(k).index)
    out[d.index.get_indexer(take)] = 1
    return out


def n_empty_windows(d: pd.DataFrame, pred: np.ndarray) -> int:
    return int((pd.Series(np.asarray(pred)).groupby(d.window_id.to_numpy()).sum() == 0).sum())
