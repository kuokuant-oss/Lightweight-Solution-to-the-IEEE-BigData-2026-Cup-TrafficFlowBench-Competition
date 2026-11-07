"""M11 -- are the Task 1 predictors' errors orthogonal, and what does blending buy?

notes/09 B (BellKor, Netflix Prize): the Grand Prize solution's own list of what
it added that year ends with blending a set of predictors; and "First we further
improved the baseline predictors. This in turn improves our other models, which
incorporate those predictors."

Blend weights are fitted on the calibration month and frozen; the number that
counts is on a different month (doc/reference/validation.md).
"""
# Training record for the Task 1 blend weights. The reproduction does not run this program;
# it reads the shipped weights. The program runs from the root of the development repository
# (whose layout `trafficflowbench/` mirrors) and reads the train release through `src.tfb`.
#
# Trains: one non-negative weight per classical fill (`METHODS`) and channel, fitted with SciPy
#   NNLS on the pooled rows of every corridor of `release.panels()` (10 corridors). The
#   weighted sum of the fills is the "blend" input of the v3 Task 1 stack.
# Data: train split, dev-fold days only (`day_split.csv`, fold "dev"). Fit rows: the target
#   cells of the train masks on the dev days of June 2030 (`CALIB`). July 2030 (`EVAL`) is a
#   check month. Labels: true speed (km/h) and true flow (veh/h, all lanes).
# Fills: LRTC-TNN tensor completion over the month's days; temporal, two-axis and spatial
#   interpolation of the day; link x weekday x time-of-day mean of the observed (masked-layer)
#   values over all dev days.
# Output: `data/interim/m11_task1_blend/blend_weights.csv` (rows = fills, columns =
#   speed_kmh, flow_vph), shipped as
#   `trafficflowbench/data/interim/m11_task1_blend/blend_weights.csv` and read by
#   `src/tfb/predict1.py::load_weights`. Also writes `resid_corr_<field>.csv` (residual
#   correlations of the fills on June) and `eval.csv` (July scores).
# Invocation: `python scripts/34_task1_blend.py` (no arguments).
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tfb import baselines as B
from src.tfb import predict1 as P1, grid, release as R, score1

OUT = R.REPO / "data" / "interim" / "m11_task1_blend"
# CALIB: month whose dev days fit the weights; EVAL: month that checks them.
CALIB, EVAL = "2030_06", "2030_07"
FIELDS = ("speed_kmh", "flow_vph")
# The five fills, in the row order of blend_weights.csv: LRTC-TNN, temporal, two-axis
# and spatial interpolation, weekday time-of-day mean.
METHODS = ("B3_lrtc_tnn", "B1b_temporal", "B1c_two_axis", "B1a_spatial", "B0_hist_mean")


def collect(panel: str, month: str, hist, mindex, truth_by_name, dev):
    """Builds the five fills and the truth at the target cells of one month.

    LRTC-TNN is solved once on the stacked days of the month (links x times x days).

    Args:
        panel: Corridor id, e.g. "D7_I10_E".
        month: Month tag "YYYY_MM"; only its dev-fold days are used.
        hist: Weekday mean per (field, weekday): link x time array.
        mindex: Masked file name -> (regime, path).
        truth_by_name: Masked file name -> truth file path.
        dev: Dev-fold dates "YYYY-MM-DD".

    Returns:
        None when the month has no target cell, else a tuple
        (fills[field][method] -> 1-D array, truth[field] -> 1-D array,
        meta with the regime, station and link of each row). A missing
        fill value is set to 0.
    """
    days = sorted(n for n in mindex if n.startswith(f"synthetic_mainline_{month}")
                  and n[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-") in dev)
    grids = [grid.build(panel, truth_by_name[n], mindex[n][1]) for n in days]
    regimes = [mindex[n][0] for n in days]
    lrtc = {f: B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f], nan=0.0) for g in grids], axis=2),
                          rho=P1.LRTC_KNOBS[f][0], theta=P1.LRTC_KNOBS[f][1]) for f in FIELDS}
    P = {f: {m: [] for m in METHODS} for f in FIELDS}
    T = {f: [] for f in FIELDS}
    meta = {"regime": [], "station": [], "link": []}
    for di, (g, rg) in enumerate(zip(grids, regimes)):
        tgt = g.target
        if not tgt.any():
            continue
        wd = pd.Timestamp(g.times[0]).weekday()
        obs = np.isfinite(g.value["speed_kmh"])
        a = B.asm(g.value["speed_kmh"], obs, g.milepost, other={"flow_vph": g.value["flow_vph"]})
        li = np.where(tgt)[0]
        meta["regime"].append(np.full(int(tgt.sum()), rg))
        meta["station"].append(np.asarray(g.station)[li])
        meta["link"].append(np.asarray(g.links)[li])
        for f in FIELDS:
            ti = B.two_axis(g.value[f], g.milepost)
            src = {"B3_lrtc_tnn": lrtc[f][:, :, di], "B1b_temporal": ti["temporal"],
                   "B1c_two_axis": ti["mean"], "B1a_spatial": ti["spatial"],
                   "B0_hist_mean": hist[(f, wd)]}
            for m in METHODS:
                P[f][m].append(np.where(np.isfinite(src[m][tgt]), src[m][tgt], 0.0))
            T[f].append(g.truth[f][tgt])
    if not T[FIELDS[0]]:
        return None
    return ({f: {m: np.concatenate(v) for m, v in P[f].items()} for f in FIELDS},
            {f: np.concatenate(v) for f, v in T.items()},
            {k: np.concatenate(v) for k, v in meta.items()})


def main() -> None:
    """Fits the NNLS blend weights on June 2030 and scores them on July 2030."""
    OUT.mkdir(parents=True, exist_ok=True)
    dev = set(R.day_split().query("fold == 'dev'").date)
    corr_acc, weights, acc_blend, acc_best = {}, {}, score1.Accumulator(), {}
    all_calib, all_eval = {}, {}
    for panel in R.panels():
        mindex = {}
        for rg in R.REGIME_ORDER:
            for p in R.masked_files(panel, "train", rg):
                mindex[p.name] = (rg, p)
        truth_by_name = {p.name: p for p in R.truth_files(panel, "train")}
        hist = {}
        # dev days ONLY -- val is held out of training (doc/reference/validation.md)
        for n in sorted(n for n in mindex
                        if n[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-") in dev):
            g = grid.build(panel, truth_by_name[n], mindex[n][1])
            wd = pd.Timestamp(g.times[0]).weekday()
            for f in FIELDS:
                v = g.value[f]; ok = np.isfinite(v)
                k = (f, wd)
                if k not in hist:
                    hist[k] = [np.zeros_like(v), np.zeros_like(v)]
                hist[k][0] += np.where(ok, v, 0.0); hist[k][1] += ok
        # Mean over the observed cells; NaN where a link and time were never observed.
        hist = {k: np.where(c > 0, s / np.maximum(c, 1), np.nan) for k, (s, c) in hist.items()}
        all_calib[panel] = collect(panel, CALIB, hist, mindex, truth_by_name, dev)
        all_eval[panel] = collect(panel, EVAL, hist, mindex, truth_by_name, dev)
        print(f"  {panel} collected", flush=True)

    # ---- residual correlation on calib, pooled over panels
    print("\n===== residual correlation on calib (dev 2030-06) =====")
    for f in FIELDS:
        res = {m: np.concatenate([all_calib[p][0][f][m] - all_calib[p][1][f] for p in all_calib if all_calib[p]])
               for m in METHODS}
        C = pd.DataFrame({a: {b: float(np.corrcoef(res[a], res[b])[0, 1]) for b in METHODS} for a in METHODS})
        print(f"\n--- {f} ---"); print(C.round(3).to_string())
        C.to_csv(OUT / f"resid_corr_{f}.csv")

    # ---- non-negative least squares blend, fitted on calib, frozen
    from scipy.optimize import nnls
    for f in FIELDS:
        X = np.column_stack([np.concatenate([all_calib[p][0][f][m] for p in all_calib if all_calib[p]])
                             for m in METHODS])
        y = np.concatenate([all_calib[p][1][f] for p in all_calib if all_calib[p]])
        w, _ = nnls(X, y)
        weights[f] = w
        print(f"\nblend weights {f}: " + ", ".join(f"{m}={wi:.4f}" for m, wi in zip(METHODS, w))
              + f"  (sum {w.sum():.4f})")
    pd.DataFrame(weights, index=list(METHODS)).to_csv(OUT / "blend_weights.csv")

    # ---- evaluate on the eval month
    accs = {m: score1.Accumulator() for m in METHODS}
    accs["BLEND_nnls"] = score1.Accumulator()
    accs["BEST_per_channel"] = score1.Accumulator()
    for panel in R.panels():
        if not all_eval[panel]:
            continue
        P, T, meta = all_eval[panel]
        lanes = score1.lanes_by_key(panel)
        lv = score1.lane_vector(pd.Series(meta["station"]), pd.Series(meta["link"]), lanes)
        blend = {f: sum(weights[f][i] * P[f][m] for i, m in enumerate(METHODS)) for f in FIELDS}
        # Reference: the best single fill per channel (LRTC-TNN speed, two-axis flow).
        # Flow errors are scored per lane (divided by the link's lane count `lv`).
        best = {"speed_kmh": P["speed_kmh"]["B3_lrtc_tnn"], "flow_vph": P["flow_vph"]["B1c_two_axis"]}
        for rg in R.REGIME_ORDER:
            sel = meta["regime"] == rg
            if not sel.any():
                continue
            for m in METHODS:
                accs[m].add(panel, rg, P["speed_kmh"][m][sel] - T["speed_kmh"][sel],
                            (P["flow_vph"][m][sel] - T["flow_vph"][sel]) / lv[sel])
            accs["BLEND_nnls"].add(panel, rg, blend["speed_kmh"][sel] - T["speed_kmh"][sel],
                                   (blend["flow_vph"][sel] - T["flow_vph"][sel]) / lv[sel])
            accs["BEST_per_channel"].add(panel, rg, best["speed_kmh"][sel] - T["speed_kmh"][sel],
                                         (best["flow_vph"][sel] - T["flow_vph"][sel]) / lv[sel])
    rows = []
    for name, acc in accs.items():
        s, _ = acc.overall(); rv, rq = score1.pooled_rmse(acc)
        rows.append({"method": name, "S_state": s, "RMSE_speed": rv, "RMSE_flow_per_lane": rq})
    d = pd.DataFrame(rows).sort_values("S_state", ascending=False)
    d.to_csv(OUT / "eval.csv", index=False)
    print(f"\n===== eval on dev {EVAL} (weights frozen from {CALIB}) =====")
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
