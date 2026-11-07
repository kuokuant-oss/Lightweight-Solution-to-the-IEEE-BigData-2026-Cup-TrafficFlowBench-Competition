"""v3 Task 1 learned stack: ONE feature implementation, used by every caller.

The frozen Task 1 answer is a linear NNLS over five members with a single weight
vector for every panel, regime and condition. This nests those five members
inside a LightGBM together with local spatio-temporal context, so the candidate
can only differ from the blend by what the linear-and-context-free form gives up.

doc/plan/task2-parity.md priced two train/serving feature mismatches at 0.031654
and 0.012346 of S_queue. A second copy of this table is how that happens again,
so scripts/113 (fit), scripts/110 (score) and scripts/112 (build) all call here.

Task 3 has no submission of its own -- it is scored on the Task 1 answer -- so
every change here moves S_state and S_physics together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines as B, predict1 as P1, release as R

MEMBERS = ("lrtc", "temporal", "two_axis", "spatial", "hist")
OFFS = (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6)
NBR = (-3, -2, -1, 1, 2, 3)
DIR = R.REPO / "build" / "task1_stack"


def members(g, di, lrtc, hist, field):
    wd = pd.Timestamp(g.times[0]).weekday()
    ti = B.two_axis(g.value[field], g.milepost)
    return {"lrtc": lrtc[field][:, :, di], "temporal": ti["temporal"], "two_axis": ti["mean"],
            "spatial": ti["spatial"],
            "hist": hist.get((field, wd), np.full_like(ti["mean"], np.nan))}


def feature_table(g, di, lrtc, hist, W, par, mask=None, occ: bool = True) -> pd.DataFrame:
    """One row per cell in ``mask`` (default: the Task 1 target cells).

    ``occ`` adds the occupancy channel as context. Occupancy is blanked at target
    cells like speed and flow, but it survives at 73-75% of NON-target cells, and
    on observed cells it correlates 0.9958 / 0.9695 with the true density q/v --
    the very quantity S_LWR differences in time. No member of the frozen blend
    reads it: LRTC and the three interpolators run on speed and flow only.
    """
    m = g.target if mask is None else mask
    wd = pd.Timestamp(g.times[0]).weekday()
    cols = {}
    for f in P1.FIELDS:
        mem = members(g, di, lrtc, hist, f)
        st = np.stack([np.where(np.isfinite(mem[k]), mem[k], 0.0) for k in MEMBERS])
        cols[f"blend_{f}"] = np.tensordot(W[f], st, axes=(0, 0))[m]
        for k, v in mem.items():
            cols[f"{k}_{f}"] = v[m]
        obs = g.value[f]
        for o in OFFS:
            cols[f"t{o:+d}_{f}"] = np.roll(obs, -o, axis=1)[m]
        for o in NBR:
            cols[f"x{o:+d}_{f}"] = np.roll(obs, -o, axis=0)[m]
        cols[f"nobs_{f}"] = np.isfinite(obs).sum(axis=1, keepdims=True).repeat(obs.shape[1], 1)[m]
    if occ:
        o = g.value["occupancy"]
        cols["occ_t0"] = o[m]
        for k in OFFS:
            cols[f"t{k:+d}_occ"] = np.roll(o, -k, axis=1)[m]
        for k in NBR:
            cols[f"x{k:+d}_occ"] = np.roll(o, -k, axis=0)[m]
        # occupancy is a density proxy; pair it with the observed flow to get a
        # speed proxy (q/k) at the neighbours the target cell must be consistent with
        for k in (-1, 1):
            qn = np.roll(g.value["flow_vph"], -k, axis=1)
            on = np.roll(o, -k, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cols[f"vproxy_t{k:+d}"] = np.where(on > 1e-6, qn / on, np.nan)[m]
    nL, nT = m.shape
    cols["tod"] = np.tile(np.arange(nT), (nL, 1))[m]
    cols["weekday"] = np.full(int(m.sum()), wd)
    cols["milepost"] = np.repeat(g.milepost[:, None], nT, 1)[m]
    for c in ("lanes", "free_speed_kmh", "capacity_vph"):
        cols[c] = np.repeat(par[c][:, None], nT, 1)[m]
    # float32 halves the table; the 8-panel all-dev run was OOM-killed at float64
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()})


def load():
    """(speed booster, flow booster, feature order) or None when not yet fitted."""
    import lightgbm as lgb
    fs = DIR / "stack_features.csv"
    if not fs.exists():
        return None
    feats = pd.read_csv(fs).feature.tolist()
    try:
        return (lgb.Booster(model_file=str(DIR / "stack_speed.txt")),
                lgb.Booster(model_file=str(DIR / "stack_flow.txt")), feats)
    except Exception:
        return None


def predict_grid(g, di, lrtc, hist, W, par, models):
    """Stacked (speed, flow) fields: stack at target cells, blend elsewhere."""
    ms, mf, feats = models
    out = {}
    for f in P1.FIELDS:
        mem = members(g, di, lrtc, hist, f)
        st = np.stack([np.where(np.isfinite(mem[k]), mem[k], 0.0) for k in MEMBERS])
        out[f] = np.tensordot(W[f], st, axes=(0, 0))
    tgt = g.target
    if tgt.any():
        X = feature_table(g, di, lrtc, hist, W, par, mask=tgt)[feats]
        out["speed_kmh"] = out["speed_kmh"].copy(); out["flow_vph"] = out["flow_vph"].copy()
        out["speed_kmh"][tgt] = ms.predict(X)
        out["flow_vph"][tgt] = mf.predict(X)
    return out
