"""Task 1 prediction pipeline -- runs on any split, from the masked layer alone.

Composition and every constant come from doc/plan/baseline-v1.md §2, which
traces each choice to doc/reference/notes/:

  P1 LRTC-TNN      notes/08 B   speed rho 1e-4 theta 0.10 / flow rho 1e-5 theta 0.10
  P2 time interp   notes/01 D
  P3 space interp  notes/01 D
  P4 two-axis      notes/01 D
  P5 hist mean     notes/09 B
  blend            notes/09 B   NNLS weights, per channel, frozen from dev/2030-06
  clip             notes/09 A   speed <= 120.74999999999999
"""
#
# Role in the package: programs of the Task 1 reproduction pipeline (the C1
# rebuild and the k* builders of pipeline/task1.sh steps 1-2, and the hole
# models) read FIELDS, LRTC_KNOBS, the speed and flow bounds, load_weights,
# split_masked_files, dev_train_files and historical_mean from here.
# predict_panel and template form the complete first-version predictor that
# the submission builders scripts/60 and scripts/112 call.
#
# P1..P5 are the five blend members; their codes in METHODS (B0..B3) are the
# baseline families of baselines.py. "notes/NN X" and "doc/..." point to the
# team's development notes, which are not part of this package. "dev" is the
# development fold of the train days (data/processed/splits/day_split.csv).
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines as B
from . import grid
from . import release as R

# The two Task 1 channels: speed (km/h) and flow (veh/h, all lanes).
FIELDS = ("speed_kmh", "flow_vph")
# Blend members in the order of the blend weight vector; task1stack.MEMBERS
# lists the same five in the same order under short names.
METHODS = ("B3_lrtc_tnn", "B1b_temporal", "B1c_two_axis", "B1a_spatial", "B0_hist_mean")
# scripts/32 ranked speed (rho=1e-4, theta=0.10) first at RMSE 1.7888 and
# (1e-5, 0.10) second at 1.8099 -- a 0.021 RMSE gap, 0.0005 of S_state. But on
# the SCORED splits rho=1e-4 leaves 0.94% of target cells at ~0 (measured on
# D7_I10_E/validation: 3,286 of 349,187) while rho=1e-5 leaves none. A rounding
# level gain is not worth a degenerate failure mode, so speed uses 1e-5 too.
LRTC_KNOBS = {"speed_kmh": (1e-5, 0.10), "flow_vph": (1e-5, 0.10)}
# Upper clip for every predicted speed, km/h (notes/09 A in the docstring).
SPEED_CAP = 120.74999999999999
# Lowest speed anywhere in 10 panels x 20 days is 6.33 km/h; the generator never
# emits less. Anything under this floor is a reconstruction failure, not a
# prediction, so it falls back to the historical mean (notes/09 A: exploit the
# generator's known hard bounds; notes/09 B: that is what a baseline predictor
# is for).
SPEED_FLOOR = 6.0
# Lower bound for predicted flow, veh/h; equal to the Task 3 empty-flow floor
# (physics.EMPTY_FLOOR_VPH).
FLOW_FLOOR = 50.0
# Frozen NNLS blend weights: rows are METHODS, columns are FIELDS.
WEIGHTS_CSV = R.REPO / "data" / "interim" / "m11_task1_blend" / "blend_weights.csv"


def load_weights() -> dict[str, np.ndarray]:
    """Reads the frozen blend weights.

    Returns:
        Field name -> weight vector over METHODS, in METHODS order.
    """
    w = pd.read_csv(WEIGHTS_CSV, index_col=0)
    return {f: w.loc[list(METHODS), f].to_numpy(dtype=float) for f in FIELDS}


def split_masked_files(panel: str, split: str) -> list[Path]:
    """Lists a panel's masked mainline day files of one split.

    Args:
        panel: Corridor id, e.g. "D7_I210_W".
        split: "train", "validation" or "private".

    Returns:
        The day files of all three mask regimes (R1, R2, R3), sorted by file
        name, which orders them by date.
    """
    out = []
    for regime in R.REGIME_ORDER:
        out.extend(R.masked_files(panel, split, regime))
    return sorted(out, key=lambda p: p.name)


def dev_train_files(panel: str) -> list[Path]:
    """Train days that are in the dev fold ONLY.

    The 56-day val fold is held out of everything, including the predictor that
    feeds the real submission. A full-data refit is a deliberate later decision
    that needs explicit sign-off (doc/reference/validation.md), not a default.
    """
    # Day files are named synthetic_mainline_YYYY_MM_DD.parquet; day_split.csv
    # assigns each train date to a fold, "dev" or the held-out one.
    dev = set(R.day_split().query("fold == 'dev'").date)
    out = []
    for p in split_masked_files(panel, "train"):
        date = p.name[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-")
        if date in dev:
            out.append(p)
    return out


def historical_mean(panel: str, files: list[Path]) -> dict[tuple[str, int], np.ndarray]:
    """(link x weekday x time-of-day) mean over observed cells of the given days."""
    # Reads only the masked layer of the given days (observed cells). Returns
    # {(field, weekday): (link x time) mean}, weekday 0 = Monday, NaN where a
    # cell was never observed on that weekday. All days must share one grid
    # shape (the same links and stamps).
    acc, cnt = {}, {}
    for path in files:
        g = grid.build_masked(panel, path)
        wd = pd.Timestamp(g.times[0]).weekday()
        for f in FIELDS:
            v = g.value[f]
            ok = np.isfinite(v)
            k = (f, wd)
            if k not in acc:
                acc[k] = np.zeros_like(v)
                cnt[k] = np.zeros_like(v)
            acc[k] += np.where(ok, v, 0.0)
            cnt[k] += ok
    return {k: np.where(cnt[k] > 0, acc[k] / np.maximum(cnt[k], 1), np.nan) for k in acc}


def predict_panel(panel: str, split: str, hist: dict, weights: dict[str, np.ndarray]) -> pd.DataFrame:
    """Predictions for every (timestamp, station, link) the split's template asks about."""
    # Args: hist from historical_mean, weights from load_weights. LRTC runs once
    # on all days of the split stacked into a (link x time x day) tensor, so
    # each day borrows structure from the other days of the same split; NaN is
    # written as 0, LRTC's missing marker. Returns one row per (timestamp,
    # link) of every day, with speed_kmh and flow_vph.
    files = split_masked_files(panel, split)
    grids = [grid.build_masked(panel, p) for p in files]
    if not grids:
        return pd.DataFrame()
    lrtc = {}
    for f in FIELDS:
        rho, theta = LRTC_KNOBS[f]
        T = np.stack([np.nan_to_num(g.value[f], nan=0.0) for g in grids], axis=2)
        lrtc[f] = B.lrtc_tnn(T, rho=rho, theta=theta)

    rows = []
    for di, g in enumerate(grids):
        wd = pd.Timestamp(g.times[0]).weekday()
        out = {}
        for f in FIELDS:
            ti = B.two_axis(g.value[f], g.milepost)
            member = {"B3_lrtc_tnn": lrtc[f][:, :, di], "B1b_temporal": ti["temporal"],
                      "B1c_two_axis": ti["mean"], "B1a_spatial": ti["spatial"],
                      "B0_hist_mean": hist.get((f, wd), np.full_like(ti["mean"], np.nan))}
            # A member without a value at a cell enters the weighted sum as 0;
            # cells that end up below the floors take the fallback chain below.
            stack = np.stack([np.where(np.isfinite(member[m]), member[m], 0.0) for m in METHODS])
            out[f] = np.tensordot(weights[f], stack, axes=(0, 0))
        # fallback chain: blend -> historical mean -> panel median of the day
        for f, floor in (("speed_kmh", SPEED_FLOOR), ("flow_vph", FLOW_FLOOR)):
            h = hist.get((f, wd))
            bad = ~np.isfinite(out[f]) | (out[f] < floor)
            if bad.any() and h is not None:
                out[f] = np.where(bad & np.isfinite(h), h, out[f])
                bad = ~np.isfinite(out[f]) | (out[f] < floor)
            if bad.any():
                obs = g.value[f]
                med = float(np.nanmedian(obs)) if np.isfinite(obs).any() else floor
                out[f] = np.where(bad, med, out[f])
        out["speed_kmh"] = np.clip(out["speed_kmh"], SPEED_FLOOR, SPEED_CAP)
        out["flow_vph"] = np.maximum(out["flow_vph"], FLOW_FLOOR)
        # Flatten the (link x time) fields row by row: li is the link index
        # and ci the time index of each output row.
        nL, nT = out["speed_kmh"].shape
        li = np.repeat(np.arange(nL), nT)
        ci = np.tile(np.arange(nT), nL)
        rows.append(pd.DataFrame({
            "panel": panel,
            # the parquet stores timestamps as text; the template parses to UTC
            "timestamp": pd.to_datetime(pd.Series(np.asarray(g.times)[ci]), utc=True),
            "station_id": np.asarray(g.station, dtype=object)[li],
            "link_id": np.asarray(g.links, dtype=object)[li],
            "speed_kmh": out["speed_kmh"].reshape(-1),
            "flow_vph": out["flow_vph"].reshape(-1),
        }))
    return pd.concat(rows, ignore_index=True)


def template(panel: str, split: str) -> pd.DataFrame:
    """Reads a panel's Task 1 sample submission for one split.

    Returns:
        The cells the split asks about: panel, timestamp (parsed to UTC),
        station_id, link_id and mask_regime, the others kept as text.
    """
    t = pd.read_csv(R.RELEASE / "task1" / panel / split / "sample_submission_state.csv",
                    usecols=["panel", "timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
    t["timestamp"] = pd.to_datetime(t.timestamp, utc=True)
    return t
