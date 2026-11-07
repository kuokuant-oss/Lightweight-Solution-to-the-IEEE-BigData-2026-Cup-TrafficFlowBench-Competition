"""Task 2 local ruler.

The official score_task2.py needs task2/<split>/queue_targets.parquet, which
ships for NO split -- verified by find over the whole release. So the ruler has
to be rebuilt here, and the two things it must get right are:

  * the label:  queue = speed_kmh <= v_cut, v_cut from queue_utils.thresholds().
    fd_parameters.csv carries no v_cut column in this release (verified in
    scripts/20), so the fallback applies: v_cut = 0.60 * free_speed_kmh.
  * the aggregation: window -> (panel, condition) -> panel -> family -> overall,
    every level a plain unweighted mean, and IoU = 1.0 on an empty union.
    Copied from score_task2.py main().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import release as R


def v_cut(panel: str) -> dict[str, float]:
    """queue_utils.thresholds(): released v_cut if present, else 0.60*free_speed."""
    links = R.network(panel, "links")
    out = dict(zip(links.link_id.astype(str),
                   (pd.to_numeric(links.free_speed_kmh, errors="coerce").fillna(105.0) * 0.60).astype(float)))
    fd = R.network(panel, "fd_parameters")
    if "v_cut" in fd.columns:
        for row in fd.dropna(subset=["link_id", "v_cut"]).itertuples(index=False):
            out[str(row.link_id)] = float(row.v_cut)
    return out


def _task2_parts(split: str, name: str):
    """This release uses the per-panel layout task2/<panel>/<split>/<name>."""
    out = []
    for panel in R.panels():
        p = R.RELEASE / "task2" / panel / split / name
        if p.exists():
            out.append((panel, p))
    return out


def window_index(split: str = "train") -> pd.DataFrame:
    parts = []
    for panel, p in _task2_parts(split, "window_index.csv"):
        f = pd.read_csv(p)
        if "panel" not in f.columns:
            f["panel"] = panel
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def window_history(split: str = "train") -> pd.DataFrame:
    cols = ["window_id", "timestamp", "link_id", "speed_kmh", "is_score_eligible"]
    parts = []
    for panel, p in _task2_parts(split, "window_history.parquet"):
        f = pd.read_parquet(p, columns=cols)
        f["panel"] = panel
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    return 1.0 if union == 0 else inter / union


def aggregate(detail: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """window -> (panel, condition) -> panel -> family -> overall, all equal weight."""
    pc = detail.groupby(["panel", "family_id", "condition"], as_index=False).IoU.mean()
    pm = pc.groupby(["panel", "family_id"], as_index=False).IoU.mean()
    fam = pm.groupby("family_id", as_index=False).IoU.mean()
    return float(fam.IoU.mean()), fam
