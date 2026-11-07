"""v3 M3 -- is the frozen LINEAR NNLS blend leaving Task 1 value on the table?

Task 1 currently = NNLS over 5 members (LRTC-TNN, temporal, two-axis, spatial,
hist-mean), one weight vector for all panels/regimes/conditions, frozen since v1.
Task 3 has no submission of its own, so the same lever moves S_physics too:
S_state 0.9340->1.0 is 0.0231 of LB and S_physics 0.7346->0.9636 is 0.0344.

This trains a LightGBM stack on the SAME five members plus local spatio-temporal
context, so the blend is nested inside the candidate and the comparison measures
only what the linear-and-context-free form gives up. Fit days and eval days are
disjoint date blocks.

  .venv/bin/python scripts/113_task1_stack_probe.py --panels D7_I10_E D12_I5_N --fit 14 --eval 7
"""
# Training record for the v3 Task 1 stack. The reproduction does not run this program; it
# loads the shipped models. The program runs from the root of the development repository
# (whose layout `trafficflowbench/` mirrors) and reads the train release through `src.tfb`.
#
# Trains: two LightGBM regressors for a blanked Task 1 cell: speed (km/h) and all-lane flow
#   (veh/h). Inputs (`src/tfb/task1stack.py`, 71 columns): the five classical fills, their
#   blend (weights from `scripts/34_task1_blend.py`), the observed values of the same link up
#   to 30 minutes before and after and of up to 3 links on each side, occupancy context, and
#   link attributes.
# Data: train split, dev-fold days of the corridors in `--panels`. The last `--eval` dev days
#   are evaluation only; the fit days before them are cut into `--chunk`-day blocks and LRTC
#   is solved per block, as at serving (31 days). Rows: target cells of the train masks;
#   `--sample` keeps a random share of the fit rows (random_state 20260920).
# Settings: objective l2, learning rate .05, 127 leaves, min_data_in_leaf 200, feature and
#   bagging fraction .8 (every iteration), seed 20260920, `--rounds` rounds, no early
#   stopping, LightGBM's default thread count (all cores).
# Output with `--save`: `build/task1_stack/stack_speed.txt`, `stack_flow.txt` and
#   `stack_features.csv`, shipped under `trafficflowbench/build/task1_stack/`; every run
#   writes the stamp `113_task1_stack_probe.stamp.json`.
# Shipped models: `--panels` with the 10 corridors, `--fit 0 --eval 14 --chunk 31
#   --rounds 900 --sample 0.25 --save`. Fit days 2030-06-01 .. 2030-12-20 (5,860,097 rows),
#   evaluation days 2030-12-21 .. 2031-01-03 (1,589,413 rows); no February data. The command
#   was rebuilt from the run stamp.
# `gate.require()` checks the development repository's research gate; the package's
#   `src/tfb/gate.py` makes it a no-op unless TFB_RESEARCH_GATE=on.
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tfb import gate
gate.require()
import lightgbm as lgb
from src.tfb import baselines as B, grid, predict1 as P1, prov, release as R, score1, task1stack as TS

OUT = R.REPO / "build" / "task1_stack"
# Time offsets (5-minute steps) and link offsets of a neighbourhood. Not used in this
# file; the feature table is built by `task1stack.feature_table`.
OFFS = (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6)
NBR = (-3, -2, -1, 1, 2, 3)


def day_table(panel, g, di, lrtc, hist, W, lanes, par, occ=True):
    """Shared feature table (src/tfb/task1stack.py) plus the two targets."""
    # One row per target cell of the day; `lane_v` (lane count) turns flow errors into
    # per-lane errors in the report.
    tgt = g.target
    if not tgt.any():
        return None
    t = TS.feature_table(g, di, lrtc, hist, W, par, mask=tgt, occ=occ)
    nT = tgt.shape[1]
    t["y_speed"] = g.truth["speed_kmh"][tgt]
    t["y_flow"] = g.truth["flow_vph"][tgt]
    t["lane_v"] = np.repeat(lanes[:, None], nT, 1)[tgt]
    return t


def collect(panel, days_idx, mindex, tm, W, occ=True):
    """Builds the stack table for one block of consecutive days of a corridor.

    LRTC-TNN is solved on the stacked days of the block. The weekday mean uses the
    corridor's other dev-fold days, so the block's own days are left out.

    Args:
        panel: Corridor id.
        days_idx: Masked file names of the block, in date order.
        mindex: Masked file name -> (regime, path).
        tm: Masked file name -> truth file path.
        W: Blend weights per field (`predict1.load_weights()`).
        occ: Include the occupancy context block.

    Returns:
        DataFrame with one row per target cell: features, `y_speed`, `y_flow`, `lane_v`.
    """
    grids = [grid.build(panel, tm[n], mindex[n][1]) for n in days_idx]
    lrtc = {f: B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f], nan=0.0) for g in grids], axis=2),
                          rho=P1.LRTC_KNOBS[f][0], theta=P1.LRTC_KNOBS[f][1]) for f in P1.FIELDS}
    ev = {mindex[n][1] for n in days_idx}
    hist = P1.historical_mean(panel, [f for f in P1.dev_train_files(panel) if f not in ev])
    links = [str(x) for x in grids[0].links]
    from src.tfb import physics
    par = physics.link_params(panel, links)
    lm = score1.lanes_by_key(panel)
    lanes = pd.Series(links).map(lm).fillna(1.0).clip(lower=1.0).to_numpy(float)
    return pd.concat([t for t in (day_table(panel, g, i, lrtc, hist, W, lanes, par, occ)
                                  for i, g in enumerate(grids)) if t is not None], ignore_index=True)


def main():
    """Builds the fit and evaluation tables, trains both heads and compares them with the blend."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=["D7_I10_E", "D12_I5_N"])
    ap.add_argument("--fit", type=int, default=14)
    ap.add_argument("--eval", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=31,
                    help="LRTC tensor depth. Serving stacks a whole split (31 days), so the "
                         "members must be built at that depth; GBM training size is decoupled "
                         "from it by collecting many chunks.")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--no-occ", action="store_true", help="ablate the occupancy context block")
    ap.add_argument("--sample", type=float, default=1.0,
                    help="fraction of FIT rows to keep (eval is never subsampled). 10 panels x "
                         "200 days is ~27M rows; a GBM does not need them all and the full table "
                         "was OOM-killed.")
    ap.add_argument("--save", action="store_true", help="persist the two boosters + feature list")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    W = P1.load_weights(); dev = set(R.day_split().query("fold == 'dev'").date)
    FIT, EV = [], []
    for panel in a.panels:
        mindex = {}
        for rg in R.REGIME_ORDER:
            for p in R.masked_files(panel, "train", rg):
                mindex[p.name] = (rg, p)
        tm = {p.name: p for p in R.truth_files(panel, "train")}
        days = sorted(n for n in mindex
                      if n[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-") in dev
                      and n in tm)
        # eval block is the LAST chunk of dev days and is never used for fitting
        e_days = days[-a.eval:]
        pool = days[:-a.eval][:a.fit] if a.fit > 0 else days[:-a.eval]
        print(f"  [{panel}] fit {len(pool)} days in {a.chunk}-day LRTC chunks, eval {len(e_days)}",
              flush=True)
        # LRTC blocks of `--chunk` consecutive fit days; a last block under 5 days is skipped.
        for i in range(0, len(pool), a.chunk):
            blk = pool[i:i + a.chunk]
            if len(blk) < 5:
                continue
            t = collect(panel, blk, mindex, tm, W, not a.no_occ)
            if a.sample < 1.0:
                t = t.sample(frac=a.sample, random_state=20260920)
            FIT.append(t)
        EV.append(collect(panel, e_days, mindex, tm, W, not a.no_occ))
    fit, ev = pd.concat(FIT, ignore_index=True), pd.concat(EV, ignore_index=True)
    print(f"\nfit rows {len(fit):,}   eval rows {len(ev):,}   features {len([c for c in fit.columns if not c.startswith('y_') and c != 'lane_v'])}")
    feats = [c for c in fit.columns if not c.startswith("y_") and c != "lane_v"]
    res = {}
    for ch, y, base in (("speed", "y_speed", "blend_speed_kmh"), ("flow", "y_flow", "blend_flow_vph")):
        ok_f = fit[y].notna(); ok_e = ev[y].notna()
        m = lgb.train({"objective": "l2", "learning_rate": 0.05, "num_leaves": 127,
                       "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "verbose": -1, "seed": 20260920},
                      lgb.Dataset(fit.loc[ok_f, feats], fit.loc[ok_f, y]), num_boost_round=a.rounds)
        if a.save:
            OUT.mkdir(parents=True, exist_ok=True)
            m.save_model(str(OUT / f"stack_{ch}.txt"))
            pd.Series(feats).to_csv(OUT / "stack_features.csv", index=False, header=["feature"])
        pred = m.predict(ev.loc[ok_e, feats])
        truth = ev.loc[ok_e, y].to_numpy(); bl = ev.loc[ok_e, base].to_numpy()
        div = ev.loc[ok_e, "lane_v"].to_numpy() if ch == "flow" else 1.0
        res[ch] = (float(np.sqrt(np.mean(((bl - truth) / div) ** 2))),
                   float(np.sqrt(np.mean(((pred - truth) / div) ** 2))))
        print(f"  {ch:6s} RMSE  frozen blend {res[ch][0]:9.4f}   LGBM stack {res[ch][1]:9.4f}"
              f"   ({(res[ch][1]/res[ch][0]-1)*100:+.1f}%)")
    # Local Task 1 state score from the speed RMSE (km/h, scale 25) and the per-lane
    # flow RMSE (veh/h per lane, scale 600), weighted .54 / .46.
    def s_state(rv, rq): return 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)
    s0 = s_state(res["speed"][0], res["flow"][0]); s1 = s_state(res["speed"][1], res["flow"][1])
    print(f"\n  S_state  frozen {s0:.6f}  ->  stack {s1:.6f}   ({s1-s0:+.6f})")
    print(f"  Task 1 alone on LB: {0.35*(s1-s0):+.6f}   (Task 3 moves too, measured by scripts/110)")
    prov.write_stamp(OUT / "113_task1_stack_probe.stamp.json",
                     prov.stamp(kind="task1_stack_probe",
                                inputs={"panels": a.panels, "fit_days": a.fit, "eval_days": a.eval,
                                        "fit_rows": int(len(fit)), "eval_rows": int(len(ev))},
                                metrics={"rmse_speed_blend": res["speed"][0], "rmse_speed_stack": res["speed"][1],
                                         "rmse_flow_lane_blend": res["flow"][0], "rmse_flow_lane_stack": res["flow"][1],
                                         "S_state_blend": s0, "S_state_stack": s1}))


if __name__ == "__main__":
    main()
