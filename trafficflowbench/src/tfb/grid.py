"""Build the (link x timestamp) day grid that every Task 1 method works on.

The input layer is ``mainline_states_masked`` -- what a participant actually
holds at test time. Cells with pct_observed 20-67 keep their values there: they
are evidence, not holes (doc/reference/column-semantics.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import release as R

VALUE_COLS = ("speed_kmh", "flow_vph", "occupancy")


def link_order(panel: str) -> pd.DataFrame:
    """Links sorted by milepost -- the one-dimensional chain. Never link_id order."""
    fd = R.network(panel, "fd_parameters")
    links = R.network(panel, "links")
    cols = [c for c in ("link_id", "length_km", "lanes", "free_speed_kmh", "capacity_vph",
                        "critical_density", "k_jam") if c in fd.columns]
    out = fd[cols].copy()
    if "lanes" not in out.columns:
        out = out.merge(links[["link_id", "lanes"]], on="link_id", how="left")
    return out


class DayGrid:
    """One panel-day as (link x 5-minute timestamp) arrays, links in milepost order.

    Attributes:
        panel: panel ID.
        date: the day, 'YYYY-MM-DD'.
        links: link IDs in row order (milepost order).
        times: the day's sorted timestamps (column order).
        milepost: milepost of each link.
        value: field -> (n_links, n_times) float array from the masked layer
            (NaN where blanked or not reported); fields speed_kmh, flow_vph,
            occupancy.
        truth: the same layout from the unmasked layer; all NaN when the grid
            comes from build_masked().
        eligible: bool array, cells the official scorer counts
            (is_score_eligible).
        blanked: bool array, cells whose speed and flow are both missing in the
            masked layer.
        station: station ID of each link.
        pct: pct_observed of each cell (0 where the cell has no row).
    """
    __slots__ = ("panel", "date", "links", "times", "milepost", "value", "truth",
                 "eligible", "blanked", "station", "pct")

    def __init__(self, panel, date, links, times, milepost, value, truth, eligible, blanked, station, pct):
        """Stores the arrays described in the class docstring."""
        self.panel, self.date = panel, date
        self.links, self.times, self.milepost = links, times, milepost
        self.value, self.truth = value, truth
        self.eligible, self.blanked, self.station, self.pct = eligible, blanked, station, pct

    @property
    def target(self) -> np.ndarray:
        """Eligible cells the release blanked -- the scored set (verified in scripts/30)."""
        return self.eligible & self.blanked


def build(panel: str, truth_path: Path, masked_path: Path) -> DayGrid:
    """Builds a DayGrid for one train day from its unmasked and masked files.

    Args:
        panel: panel ID.
        truth_path: the day's `mainline_states` file (only train ships it).
        masked_path: the same day's `mainline_states_masked` file.

    Returns:
        DayGrid whose `value` holds the masked layer and `truth` the unmasked
        layer; eligibility and pct_observed come from the unmasked file.
    """
    truth = pd.read_parquet(truth_path, columns=["date", "timestamp", "station_id", "link_id", "milepost",
                                                 "speed_kmh", "flow_vph", "occupancy",
                                                 "pct_observed", "is_score_eligible"])
    masked = pd.read_parquet(masked_path, columns=["timestamp", "station_id", "link_id",
                                                   "speed_kmh", "flow_vph", "occupancy"])
    for f in (truth, masked):
        f["link_id"] = f.link_id.astype(str)
        f["station_id"] = f.station_id.astype(str)
    order = truth[["link_id", "milepost"]].drop_duplicates().sort_values("milepost").reset_index(drop=True)
    li = {k: i for i, k in enumerate(order.link_id)}
    times = np.sort(truth.timestamp.unique())
    ti = {k: i for i, k in enumerate(times)}
    nL, nT = len(li), len(times)

    r = truth.link_id.map(li).to_numpy()
    c = truth.timestamp.map(ti).to_numpy()
    station = np.empty((nL,), dtype=object)
    station[r] = truth.station_id.to_numpy()

    def lay(frame, col, idx_r, idx_c, fill=np.nan):
        """Scatters one column of `frame` into an (n_links, n_times) array, `fill` elsewhere."""
        g = np.full((nL, nT), fill, dtype=float)
        g[idx_r, idx_c] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        return g

    tvals = {c_: lay(truth, c_, r, c) for c_ in VALUE_COLS}
    pct = lay(truth, "pct_observed", r, c, 0.0)
    eligible = np.zeros((nL, nT), dtype=bool)
    eligible[r, c] = truth.is_score_eligible.astype(bool).to_numpy()

    mr = masked.link_id.map(li).to_numpy()
    mc = masked.timestamp.map(ti).to_numpy()
    mvals = {c_: lay(masked, c_, mr, mc) for c_ in VALUE_COLS}
    # A cell counts as blanked when the release removed both speed and flow.
    blanked = np.isnan(mvals["speed_kmh"]) & np.isnan(mvals["flow_vph"])

    return DayGrid(panel, str(truth.date.iloc[0]), list(order.link_id), times,
                   order.milepost.to_numpy(dtype=float), mvals, tvals, eligible, blanked, station, pct)


def build_masked(panel: str, masked_path: Path) -> DayGrid:
    """Grid built from the masked layer ALONE -- no truth.

    This is what validation and private ship (they carry no ``mainline_states``),
    so the prediction path must work from this. ``truth`` is left as NaN and
    ``target`` means "eligible and blanked", which on a scored split is the
    set the release asks about.
    """
    m = pd.read_parquet(masked_path, columns=["date", "timestamp", "station_id", "link_id", "milepost",
                                              "speed_kmh", "flow_vph", "occupancy",
                                              "pct_observed", "is_score_eligible"])
    m["link_id"] = m.link_id.astype(str)
    m["station_id"] = m.station_id.astype(str)
    order = m[["link_id", "milepost"]].drop_duplicates().sort_values("milepost").reset_index(drop=True)
    li = {k: i for i, k in enumerate(order.link_id)}
    times = np.sort(m.timestamp.unique())
    ti = {k: i for i, k in enumerate(times)}
    nL, nT = len(li), len(times)
    r = m.link_id.map(li).to_numpy()
    c = m.timestamp.map(ti).to_numpy()
    station = np.empty((nL,), dtype=object)
    station[r] = m.station_id.to_numpy()

    def lay(col, fill=np.nan):
        """Scatters column `col` of the masked file into an (n_links, n_times) array, `fill` elsewhere."""
        g = np.full((nL, nT), fill, dtype=float)
        g[r, c] = pd.to_numeric(m[col], errors="coerce").to_numpy(dtype=float)
        return g

    vals = {c_: lay(c_) for c_ in VALUE_COLS}
    pct = lay("pct_observed", 0.0)
    eligible = np.zeros((nL, nT), dtype=bool)
    eligible[r, c] = m.is_score_eligible.astype(bool).to_numpy()
    blanked = np.isnan(vals["speed_kmh"]) & np.isnan(vals["flow_vph"])
    truth = {c_: np.full((nL, nT), np.nan) for c_ in VALUE_COLS}
    return DayGrid(panel, str(m.date.iloc[0]), list(order.link_id), times,
                   order.milepost.to_numpy(dtype=float), vals, truth, eligible, blanked, station, pct)
