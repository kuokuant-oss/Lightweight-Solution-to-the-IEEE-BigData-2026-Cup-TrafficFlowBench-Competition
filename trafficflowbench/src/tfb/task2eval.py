"""Paired relative evaluation for Task 2 candidates.

Everything a candidate is judged on lives here so that control and candidate
cannot diverge by accident: the same rows, the same scorer (score2.aggregate,
the official four-level equal-weight mean), the same per-window IoU, the same
bootstrap blocks.

The two estimators answer different questions and both are reported:

  * ``paired_bootstrap`` -- the delta on the whole eval window pool, with dates
    resampled as blocks. Windows inside one day share the day's demand and
    incidents, so the date is the dependence unit; resampling windows alone
    would understate the spread. This is the higher-power question: is the
    candidate better over the window distribution?
  * ``replicate_deltas`` -- the delta on R official-shaped 80-window manifests.
    The hidden score is measured on exactly one such set, and the organizers
    put its per-panel standard error at 0.1187 with 5 windows per condition, so
    a candidate can be better on the distribution and still lose on the one set
    that counts. This estimator shows how often that happens.

``min_size`` post-processing is the connected-component filter of scripts/51,
ported verbatim -- the frozen model ships min_size=1, i.e. no filtering, and a
candidate that changes it must be measured against that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import release as R, score2

# Role: the Task 2 training and audit programs (for example
# scripts/145_onset_prior_forward.py and scripts/strongest_20260922_ongoing.py)
# score with iou_frame and s_queue, the local copy of the official metric:
# per-window IoU, then equal-weight means over window -> (panel, condition)
# -> panel -> family. The other functions are the record of the
# candidate-comparison harness.
# CONDITIONS: the two window types, queue onset (no queue in the history) and
# queue ongoing (a queue visible in the history).
CONDITIONS = ("queue_onset", "queue_ongoing")


def logit(p, eps=1e-4):
    """scripts/51.logit, verbatim -- the init_score the frozen model was fit with."""
    # NaN counts as 0, and probabilities are clipped to [eps, 1 - eps].
    p = np.clip(np.nan_to_num(p, nan=0.0), eps, 1 - eps)
    return np.log(p / (1 - p))


def add_probs(d: pd.DataFrame, model_dir: Path, col: str = "prob") -> pd.DataFrame:
    """Per-condition frozen booster + the historical-prior init_score."""
    # Loads lgb_queue_onset.txt and lgb_queue_ongoing.txt from `model_dir`.
    # Each booster's raw score is added to logit(queue_prior), so the prior is
    # the starting log-odds. Rows of a condition without a model file get 0.
    # Returns a copy of `d` with the probabilities in column `col`.
    import lightgbm as lgb
    d = d.copy()
    d[col] = np.nan
    for cond in CONDITIONS:
        path = model_dir / f"lgb_{cond}.txt"
        if not path.exists():
            continue
        sel = (d.condition == cond).to_numpy()
        if not sel.any():
            continue
        m = lgb.Booster(model_file=str(path))
        raw = m.predict(d.loc[sel, m.feature_name()], raw_score=True) \
            + logit(d.loc[sel, "queue_prior"].to_numpy())
        d.loc[sel, col] = 1.0 / (1.0 + np.exp(-raw))
    d[col] = d[col].fillna(0.0)
    return d


def connected_filter(d: pd.DataFrame, pred_col: str, min_size: int) -> np.ndarray:
    """scripts/51.connected_filter, verbatim: drop blobs under min_size."""
    # Per window, the predicted cells form a (link x step) image with links
    # ranked by milepost; 4-connected components (scipy.ndimage.label
    # default) smaller than `min_size` cells are set to 0. Returns the
    # filtered 0/1 array aligned with `d`.
    from scipy import ndimage
    keep = d[pred_col].to_numpy().copy()
    for _, idx in d.groupby("window_id", sort=False).indices.items():
        sub = d.iloc[idx]
        links = pd.Index(sub.milepost.to_numpy()).unique()
        rank = {v: i for i, v in enumerate(np.sort(links))}
        r = sub.milepost.map(rank).to_numpy()
        c = sub.step.to_numpy() - 1
        img = np.zeros((len(rank), int(c.max()) + 1), dtype=bool)
        img[r, c] = d[pred_col].to_numpy()[idx] > 0
        lab, n = ndimage.label(img)
        if n == 0:
            continue
        sizes = ndimage.sum(img, lab, index=np.arange(1, n + 1))
        too_small = np.isin(lab[r, c], np.nonzero(sizes < min_size)[0] + 1)
        keep[idx[too_small]] = 0
    return keep


def decide(d: pd.DataFrame, thr: dict[str, float], min_size: int = 1,
           prob_col: str = "prob") -> np.ndarray:
    """Threshold per condition, then the optional component filter."""
    # `thr` maps condition -> probability threshold; a cell is predicted
    # queued when prob >= thr. Returns a 0/1 array aligned with `d`.
    p = (d[prob_col].to_numpy() >= d.condition.map(thr).to_numpy()).astype(int)
    if min_size and min_size > 1:
        return connected_filter(d.assign(_p=p), "_p", int(min_size))
    return p


def iou_frame(d: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Per-window IoU. scripts/51.iou_frame, with the prediction passed in."""
    # Returns one row per window: inter and union cell counts, IoU (1.0 when
    # the union is empty, as in the official scorer), family_id and the date
    # parsed from window_id ("panel|date|origin").
    p = np.asarray(pred) > 0
    y = d.label.to_numpy() > 0
    t = pd.DataFrame({"window_id": d.window_id.to_numpy(), "panel": d.panel.to_numpy(),
                      "condition": d.condition.to_numpy(),
                      "inter": np.logical_and(p, y), "union": np.logical_or(p, y)})
    o = t.groupby(["window_id", "panel", "condition"], sort=False, observed=True)[["inter", "union"]] \
         .sum().reset_index()
    o["IoU"] = np.where(o.union == 0, 1.0, o.inter / o.union.where(o.union != 0, 1))
    o["family_id"] = o.panel.map(R.families())
    o["date"] = [w.split("|")[1] for w in o.window_id]
    return o


def s_queue(frame: pd.DataFrame) -> float:
    """The official four-level equal-weight aggregate."""
    return score2.aggregate(frame)[0]


def subgroups(frame: pd.DataFrame) -> dict[str, float]:
    """Condition means and per-panel means, each aggregated the official way."""
    # Each per-panel value is the mean over conditions of the mean window IoU;
    # worst_panel is the lowest of them.
    out: dict[str, float] = {}
    for cond in CONDITIONS:
        g = frame[frame.condition == cond]
        out[cond] = s_queue(g) if len(g) else float("nan")
    for panel, g in frame.groupby("panel"):
        out[f"panel:{panel}"] = float(g.groupby("condition").IoU.mean().mean())
    panels = [v for k, v in out.items() if k.startswith("panel:")]
    out["worst_panel"] = float(min(panels)) if panels else float("nan")
    return out


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000,
                     seed: int = 0, block: str = "date") -> dict[str, float]:
    """Delta = S_queue(b) - S_queue(a), with ``block`` resampled with replacement.

    a and b must cover the same windows; the pairing is what removes the
    window-difficulty variance that dominates a single score's spread.
    """
    # Returns the point delta, the 2.5 and 97.5 percentiles of the resampled
    # deltas, their mean and sd, the share of resamples with delta > 0, and
    # the resample and block counts.
    # Raises RuntimeError when a and b cover different windows.
    a = a.sort_values("window_id").reset_index(drop=True)
    b = b.sort_values("window_id").reset_index(drop=True)
    if not a.window_id.equals(b.window_id):
        raise RuntimeError("paired_bootstrap needs identical window sets")
    point = s_queue(b) - s_queue(a)
    keys = np.sort(a[block].unique())
    idx_by_key = {k: np.flatnonzero((a[block] == k).to_numpy()) for k in keys}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([idx_by_key[k] for k in pick])
        out[i] = s_queue(b.iloc[rows]) - s_queue(a.iloc[rows])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"delta": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "boot_mean": float(out.mean()), "boot_sd": float(out.std(ddof=1)),
            "p_gt0": float((out > 0).mean()), "n_boot": n_boot, "n_blocks": len(keys)}


def replicate_deltas(a: pd.DataFrame, b: pd.DataFrame,
                     manifests: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """S_queue for both methods on each official-shaped manifest."""
    # `manifests` maps seed -> manifest with a window_id column. Returns one
    # row per seed with both scores and their delta.
    ia = a.set_index("window_id")
    ib = b.set_index("window_id")
    rows = []
    for seed, man in sorted(manifests.items()):
        ids = man.window_id.to_numpy()
        fa, fb = ia.loc[ids].reset_index(), ib.loc[ids].reset_index()
        sa, sb = s_queue(fa), s_queue(fb)
        rows.append({"seed": seed, "n_windows": len(ids), "control": sa,
                     "candidate": sb, "delta": sb - sa})
    return pd.DataFrame(rows)


def replicate_summary(rep: pd.DataFrame) -> dict[str, float]:
    """Summarises replicate deltas: mean, median, sd, 95% range, win/tie rate."""
    d = rep.delta.to_numpy()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"n_replicates": len(d), "delta_mean": float(d.mean()),
            "delta_median": float(np.median(d)), "delta_sd": float(d.std(ddof=1)),
            "q2.5": float(lo), "q97.5": float(hi),
            "win_rate": float((d > 0).mean()), "tie_rate": float((d == 0).mean()),
            "control_mean": float(rep.control.mean()), "candidate_mean": float(rep.candidate.mean())}


def choose_thresholds(d: pd.DataFrame, grid_t: np.ndarray | None = None,
                      passes: int = 2, min_size: int = 1,
                      prob_col: str = "prob") -> dict[str, float]:
    """scripts/51's coordinate sweep, on whatever rows are handed in.

    Only ever call this on calib rows (or, for a rolling-origin fold, on the
    dates strictly before that fold).
    """
    # Grid: 0.050 to 0.950 in steps of 0.025. Both thresholds start at 0.5;
    # each pass tunes one condition at a time with the other fixed and keeps
    # the first grid value with the best s_queue.
    grid_t = np.round(np.arange(0.05, 0.96, 0.025), 3) if grid_t is None else grid_t
    thr = {c: 0.5 for c in CONDITIONS}
    for _ in range(passes):
        for cond in CONDITIONS:
            best, bt = -1.0, thr[cond]
            for t in grid_t:
                cur = dict(thr)
                cur[cond] = float(t)
                s = s_queue(iou_frame(d, decide(d, cur, min_size, prob_col)))
                if s > best:
                    best, bt = s, float(t)
            thr[cond] = bt
    return thr


def persistence_iou_frame(man: pd.DataFrame) -> pd.DataFrame:
    """The official persistence reference, read off the window table.

    build_task2_persistence_submission.py predicts ``queue_now`` -- speed <=
    v_cut AND is_score_eligible at the last released history stamp -- held flat
    across the horizon, which is exactly the ``persistence_iou`` column the
    window enumeration already computes. Re-deriving it from features would
    drop the eligibility term and read high (scripts/51's note).
    """
    o = man[["window_id", "panel", "condition", "date"]].copy()
    o["IoU"] = man.persistence_iou.to_numpy()
    o["family_id"] = o.panel.map(R.families())
    return o
