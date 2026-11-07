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

# Frozen tables written by scripts/111: one row per predicted-value bin
# (bin_lo, bin_hi) with the bias of the predictions in that bin (`bias`, in the
# field's units); apply() subtracts it.
DIR = R.REPO / "data" / "interim" / "v3_calibration"


def load(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Loads one frozen bias table, `speed_bias` or `flow_bias`.

    Returns:
        (bin centres, bias per bin) as float arrays, or None when the CSV is
        absent, in which case apply() leaves the values unchanged.
    """
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
    """Applies the frozen speed and flow bias maps, then the output bounds.

    Args:
        v: predicted speed (km/h).
        q: predicted flow (veh/h).
        lam_v: share of the speed bias to remove (0 turns the speed map off).
        lam_q: share of the flow bias to remove (0 turns the flow map off).
        speed_floor: lower clip for the corrected speed.
        speed_cap: upper clip for the corrected speed.
        flow_floor: lower bound for the corrected flow.

    Returns:
        (v2, q2): corrected speed and flow arrays.
    """
    vb, qb = load("speed_bias"), load("flow_bias")
    v2 = np.clip(apply(v, vb, lam_v), speed_floor, speed_cap)
    q2 = np.maximum(apply(q, qb, lam_q), flow_floor)
    return v2, q2
