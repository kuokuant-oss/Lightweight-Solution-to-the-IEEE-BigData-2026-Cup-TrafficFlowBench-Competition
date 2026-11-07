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
#
# Role in the package: the Task 1 reproduction pipeline calls load and
# predict_grid in the C1 rebuild (build/review_20260924_ctm/rl/serve_feats.py,
# pipeline/task1.sh step 1) and feature_table in the k* builders (step 2).
# Each fitted booster is a LightGBM regressor that predicts a target cell's
# speed or flow directly from the five classical fills, their linear blend and
# the cell's observed neighbourhood. "v3" names this stacked Task 1 answer
# (the one scripts/112_build_submission_v3.py builds); S_state and S_physics
# are the Task 1 and Task 3 scores. scripts/110, scripts/113 and doc/plan/...
# are development records that are not part of this package.
from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines as B, predict1 as P1, release as R

# The five classical fills, in the order of predict1.METHODS and of the blend
# weight vector W: LRTC-TNN, temporal, two-axis mean, spatial, historical mean.
MEMBERS = ("lrtc", "temporal", "two_axis", "spatial", "hist")
# Time offsets in 5-minute steps for the neighbour features: the same link up
# to 30 minutes earlier (negative) or later (positive).
OFFS = (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6)
# Link offsets in milepost order for the neighbour features: positive = that
# many links downstream, negative = upstream, at the same stamp.
NBR = (-3, -2, -1, 1, 2, 3)
# Folder of the fitted stack: stack_speed.txt and stack_flow.txt (LightGBM
# boosters) and stack_features.csv (feature order).
DIR = R.REPO / "build" / "task1_stack"


def members(g, di, lrtc, hist, field):
    """Returns the five classical fills of one field on one day.

    Args:
        g: Day grid from grid.build_masked.
        di: Index of this day in the split's LRTC tensor (third axis).
        lrtc: Field -> (link x time x day) LRTC-TNN completion of the split.
        hist: Historical means keyed (field, weekday), from
            predict1.historical_mean.
        field: "speed_kmh" or "flow_vph".

    Returns:
        Member name (MEMBERS) -> (link x time) array, NaN where the member has
        no value.
    """
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
    # Further args: g, di, lrtc, hist as in members(); W: field -> blend weight
    # vector (predict1.load_weights); par: physics.link_params output.
    # Column families per field: blend_<field> (the frozen linear blend),
    # <member>_<field> (each fill), t<+-k>_<field> (observed value of the same
    # link k steps later / earlier), x<+-k>_<field> (observed value k links
    # downstream / upstream), nobs_<field> (observed cells of the link that
    # day). Then occupancy context, time of day (step index), weekday,
    # milepost and the static link parameters. Neighbour values come from the
    # masked layer, so they are NaN where those cells are blanked. np.roll wraps
    # around: near the ends of the day or of the corridor the offset columns
    # read the opposite end. The fitted boosters were trained on the same
    # construction.
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
    # Args as in feature_table; models is the (speed booster, flow booster,
    # feature order) triple from load(). Returns field -> (link x time) array:
    # the booster predictions at the target cells (eligible and blanked) and
    # the linear blend everywhere else, both unclipped.
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
