"""v3 post-hoc recalibration of the Task 1 blend output.

ONE implementation, used by BOTH the harness (scripts/110) and the builder
(scripts/112). doc/plan/task2-parity.md found four columns where the training
and serving codepaths disagreed and priced two of them at 0.031654 and 0.012346
of S_queue; a second implementation of the same transform is exactly how that
happens again, so there is only this one.

The map is keyed on the PREDICTED value, which serving can see. It is fitted by
scripts/111 on the calibration month alone and frozen to CSV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import release as R

DIR = R.REPO / "data" / "interim" / "v3_calibration"


def load(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = DIR / f"{name}.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    return ((t.bin_lo + t.bin_hi) / 2.0).to_numpy(float), t.bias.to_numpy(float)


def apply(x: np.ndarray, tab: tuple[np.ndarray, np.ndarray] | None, lam: float) -> np.ndarray:
    """x - lam * bias(x), linearly interpolated between bin centres.

    np.interp holds the end values flat outside the range, so an extrapolated
    prediction is never given a correction larger than the edge bin's.
    """
    if tab is None or not lam:
        return x
    c, b = tab
    return x - lam * np.interp(x, c, b)


def calibrate_state(v: np.ndarray, q: np.ndarray, lam_v: float, lam_q: float,
                    speed_floor: float, speed_cap: float, flow_floor: float):
    vb, qb = load("speed_bias"), load("flow_bias")
    v2 = np.clip(apply(v, vb, lam_v), speed_floor, speed_cap)
    q2 = np.maximum(apply(q, qb, lam_q), flow_floor)
    return v2, q2
