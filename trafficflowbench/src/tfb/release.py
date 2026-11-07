"""Release access helpers. Read-only; no modelling here.

Every path rule in this module was read off the official scorers in
``external/trafficflowbench-public/src`` (pin c88cddf), not guessed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# REPO: the `trafficflowbench/` folder. RELEASE: the released competition data
# (`kaggle_public`), kept in a sibling folder of REPO. OFFICIAL: the copy of the
# official scorer repository; this module reads only its panel manifest.
REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO.parent / "tfb_competition_data" / "kaggle_public"
OFFICIAL = REPO / "external" / "trafficflowbench-public"

# Task 1 score of one (panel, regime): S_speed = max(0, 1 - RMSE_speed / 25),
# S_flow = max(0, 1 - RMSE_flow_per_lane / 600) and
# S_state = .54 * S_speed + .46 * S_flow. R1..R3 are the release's mask regimes.
# score_task1.py / baseline_task1_historical_mean.py
SPEED_NORMALIZER = 25.0
FLOW_NORMALIZER = 600.0
SPEED_WEIGHT = 0.54
FLOW_WEIGHT = 0.46
REGIME_ORDER = ("R1", "R2", "R3")
# Task 3 empty-road rule: a (panel, regime) gets FD score 0 when more than
# EMPTY_SHARE of its target cells have flow below EMPTY_FLOOR_VPH (veh/h).
# score_task3.py
EMPTY_FLOOR_VPH = 50.0
EMPTY_SHARE = 0.20


def manifest() -> dict:
    """Returns the official panel manifest (`config/corridors.json`) as a dict."""
    return json.loads((OFFICIAL / "config" / "corridors.json").read_text(encoding="utf-8"))


def panels() -> list[str]:
    """Returns the panel IDs (e.g. `D7_I210_W`) in manifest order."""
    return [p["corridor_id"] for p in manifest()["panels"]]


def families() -> dict[str, str]:
    """Returns panel ID -> family ID; the official scores average panels within a family."""
    return {p["corridor_id"]: p["family_id"] for p in manifest()["panels"]}


def _clean(paths):
    """exFAT drops ``._*`` AppleDouble siblings next to every file; pyarrow chokes on them."""
    return sorted(p for p in paths if not p.name.startswith("._"))


def truth_files(panel: str, split: str = "train") -> list[Path]:
    """Unmasked observation layer -- the Task 1 truth. Only ``train`` ships it."""
    return _clean((RELEASE / "corridors" / panel / split / "mainline_states").glob("**/*.parquet"))


def masked_files(panel: str, split: str, regime: str) -> list[Path]:
    """Returns the masked-layer day files of one panel, split and mask regime.

    Args:
        panel: panel ID.
        split: 'train', 'validation' or 'private'.
        regime: mask regime, one of REGIME_ORDER.

    Returns:
        Sorted parquet paths, one file per day.
    """
    root = RELEASE / "corridors" / panel / split / "mainline_states_masked" / f"mask_regime={regime}"
    return _clean(root.glob("*.parquet"))


def network(panel: str, name: str) -> pd.DataFrame:
    """Reads one network table of a panel (e.g. `links`, `fd_parameters`), link IDs as strings."""
    return pd.read_csv(RELEASE / "corridors" / panel / "network" / f"{name}.csv", dtype={"link_id": str})


def day_split() -> pd.DataFrame:
    """Reads the project's day-to-fold table `data/processed/splits/day_split.csv` (all columns as strings)."""
    return pd.read_csv(REPO / "data" / "processed" / "splits" / "day_split.csv", dtype=str)
