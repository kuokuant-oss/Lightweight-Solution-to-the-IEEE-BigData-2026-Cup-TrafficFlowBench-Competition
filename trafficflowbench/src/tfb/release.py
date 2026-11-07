"""Release access helpers. Read-only; no modelling here.

Every path rule in this module was read off the official scorers in
``external/trafficflowbench-public/src`` (pin c88cddf), not guessed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO.parent / "tfb_competition_data" / "kaggle_public"
OFFICIAL = REPO / "external" / "trafficflowbench-public"

# score_task1.py / baseline_task1_historical_mean.py
SPEED_NORMALIZER = 25.0
FLOW_NORMALIZER = 600.0
SPEED_WEIGHT = 0.54
FLOW_WEIGHT = 0.46
REGIME_ORDER = ("R1", "R2", "R3")
# score_task3.py
EMPTY_FLOOR_VPH = 50.0
EMPTY_SHARE = 0.20


def manifest() -> dict:
    return json.loads((OFFICIAL / "config" / "corridors.json").read_text(encoding="utf-8"))


def panels() -> list[str]:
    return [p["corridor_id"] for p in manifest()["panels"]]


def families() -> dict[str, str]:
    return {p["corridor_id"]: p["family_id"] for p in manifest()["panels"]}


def _clean(paths):
    """exFAT drops ``._*`` AppleDouble siblings next to every file; pyarrow chokes on them."""
    return sorted(p for p in paths if not p.name.startswith("._"))


def truth_files(panel: str, split: str = "train") -> list[Path]:
    """Unmasked observation layer -- the Task 1 truth. Only ``train`` ships it."""
    return _clean((RELEASE / "corridors" / panel / split / "mainline_states").glob("**/*.parquet"))


def masked_files(panel: str, split: str, regime: str) -> list[Path]:
    root = RELEASE / "corridors" / panel / split / "mainline_states_masked" / f"mask_regime={regime}"
    return _clean(root.glob("*.parquet"))


def network(panel: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RELEASE / "corridors" / panel / "network" / f"{name}.csv", dtype={"link_id": str})


def day_split() -> pd.DataFrame:
    return pd.read_csv(REPO / "data" / "processed" / "splits" / "day_split.csv", dtype=str)
