"""Local Task 1 scorer -- a line-by-line re-implementation of the official
``score_task1.py`` (pin c88cddf) with one addition: a day filter, so dev and
val can be scored separately (doc/reference/validation.md holds val out).

Verified against the official scorer by scripts/30_verify_scorer.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import release as R


def lanes_by_key(panel: str) -> dict[str, float]:
    """station_lanes() in baseline_task1_historical_mean.py: links.csv first,
    fd_parameters.csv second so it wins; lanes outside [1, 16] are dropped
    (links.csv marks zone connectors with lanes=99)."""
    out: dict[str, float] = {}
    for name, key in (("links", "link_id"), ("fd_parameters", "link_id"), ("fd_parameters", "station_id")):
        path = R.RELEASE / "corridors" / panel / "network" / f"{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if key not in frame.columns or "lanes" not in frame.columns:
            continue
        lanes = pd.to_numeric(frame["lanes"], errors="coerce")
        ok = lanes.notna() & (lanes >= 1) & (lanes <= 16)
        out.update(zip(frame[key][ok].astype(str), lanes[ok].astype(float)))
    return out


def lane_vector(station_id: pd.Series, link_id: pd.Series, lanes: dict[str, float]) -> np.ndarray:
    v = station_id.astype(str).map(lanes)
    v = v.fillna(link_id.astype(str).map(lanes))
    return v.fillna(1.0).clip(lower=1.0).to_numpy(dtype=float)


def targets_for_day(panel: str, regime: str, truth_path: Path, masked_path: Path) -> pd.DataFrame:
    """Eligible cells that the release blanked -- exactly what score_task1 scores."""
    truth = pd.read_parquet(truth_path, columns=["timestamp", "station_id", "link_id",
                                                 "speed_kmh", "flow_vph", "is_score_eligible"])
    truth["link_id"] = truth.link_id.astype(str)
    truth["station_id"] = truth.station_id.astype(str)
    truth["timestamp"] = pd.to_datetime(truth.timestamp, utc=True)
    blanked = pd.read_parquet(masked_path, columns=["timestamp", "station_id", "link_id",
                                                    "speed_kmh", "flow_vph"])
    blanked["link_id"] = blanked.link_id.astype(str)
    blanked["station_id"] = blanked.station_id.astype(str)
    blanked["timestamp"] = pd.to_datetime(blanked.timestamp, utc=True)
    blanked = blanked[blanked.speed_kmh.isna() & blanked.flow_vph.isna()][["timestamp", "station_id", "link_id"]]
    if blanked.empty:
        return blanked.assign(true_speed=np.nan, true_flow=np.nan)
    eligible = truth[truth.is_score_eligible.astype(bool)]
    out = eligible.merge(blanked, on=["timestamp", "station_id", "link_id"], how="inner", validate="one_to_one")
    return out.rename(columns={"speed_kmh": "true_speed", "flow_vph": "true_flow"})[
        ["timestamp", "station_id", "link_id", "true_speed", "true_flow"]]


class Accumulator:
    """Sums of squares per (panel, regime), aggregated the way the official
    scorer does: regime -> panel -> family -> overall, every level a plain mean."""

    def __init__(self) -> None:
        self.sq: dict[tuple[str, str], list[float]] = {}

    def add(self, panel: str, regime: str, d_speed: np.ndarray, d_flow_per_lane: np.ndarray) -> None:
        cell = self.sq.setdefault((panel, regime), [0.0, 0.0, 0])
        cell[0] += float(np.sum(d_speed * d_speed))
        cell[1] += float(np.sum(d_flow_per_lane * d_flow_per_lane))
        cell[2] += int(d_speed.size)

    def table(self) -> pd.DataFrame:
        rows = []
        for (panel, regime), (ss, sf, n) in sorted(self.sq.items()):
            if n == 0:
                rmse_v = rmse_q = np.nan
                s_v = s_q = s = 0.0
            else:
                rmse_v = float(np.sqrt(ss / n))
                rmse_q = float(np.sqrt(sf / n))
                s_v = max(0.0, 1.0 - rmse_v / R.SPEED_NORMALIZER)
                s_q = max(0.0, 1.0 - rmse_q / R.FLOW_NORMALIZER)
                s = R.SPEED_WEIGHT * s_v + R.FLOW_WEIGHT * s_q
            rows.append({"panel": panel, "regime": regime, "n": n, "rmse_speed": rmse_v,
                         "rmse_flow_per_lane": rmse_q, "S_speed": s_v, "S_flow": s_q, "S_state": s})
        return pd.DataFrame(rows)

    def overall(self) -> tuple[float, pd.DataFrame]:
        t = self.table()
        if t.empty:
            return 0.0, t
        per_panel = t.groupby("panel", sort=True).S_state.mean()
        fam = R.families()
        per_family = per_panel.rename_axis("panel").reset_index()
        per_family["family_id"] = per_family.panel.map(fam)
        overall = float(per_family.groupby("family_id").S_state.mean().mean())
        return overall, t


def pooled_rmse(acc: Accumulator) -> tuple[float, float]:
    """Cell-weighted RMSE over everything accumulated -- for comparing methods
    at a finer resolution than S_state, which saturates."""
    ss = sum(v[0] for v in acc.sq.values())
    sf = sum(v[1] for v in acc.sq.values())
    n = sum(v[2] for v in acc.sq.values())
    return (float(np.sqrt(ss / n)), float(np.sqrt(sf / n))) if n else (np.nan, np.nan)
