"""Build the three submission files for validation + private, then merge.

Task 1  src/tfb/predict1.py        blend, frozen weights (doc/plan §2)
Task 2  src/tfb/task2infer.py      the two LightGBM models (doc/plan §5)
Task 3  no file -- scored on Task 1 (docs/SUBMISSION_SCHEMAS.md)
Task 4  official NNLS, REG_LAMBDA = 0.05 (doc/plan §4)

Nothing here reads the 56-day val fold.
"""

# Role: the first (v1) whole-submission builder, kept as the record of that
# build. The reproduction pipeline uses only build_odme(): pipeline/od.py
# imports this file unchanged, points OUT at the run tree and calls
# build_odme() for Task 4.
# Outputs under OUT: state_submission.csv, queue_submission.csv and
# path_flow_submission.csv, merged by the official merge_submissions.py into
# submission.csv.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Research gate (src/tfb/gate.py): returns at once unless TFB_RESEARCH_GATE=on.
from src.tfb import gate
gate.require()

import lightgbm as lgb
from src.tfb import predict1, release as R, score2, task2infer

# OUT: the v1 delivery folder. MODELS: the v1 Task 2 boosters
# (lgb_queue_onset.txt, lgb_queue_ongoing.txt) and their thresholds (config.csv).
OUT = R.REPO / "build" / "submission_v1"
MODELS = R.REPO / "data" / "interim" / "m14_task2_model"
SPLITS = ("validation", "private")


def build_state() -> Path:
    """Builds the v1 Task 1 state file with the frozen linear blend.

    For each split and panel, predict1.predict_panel fills the release
    template; its history-mean member uses the train days of the dev fold.

    Returns:
        Path of state_submission.csv (panel, timestamp, station_id, link_id,
        mask_regime, speed_kmh, flow_vph).
    """
    w = predict1.load_weights()
    parts = []
    hist_cache = {}
    for split in SPLITS:
        for panel in R.panels():
            if panel not in hist_cache:
                hist_cache[panel] = predict1.historical_mean(panel, predict1.dev_train_files(panel))
            hist = hist_cache[panel]
            pred = predict1.predict_panel(panel, split, hist, w)
            tmpl = predict1.template(panel, split)
            merged = tmpl.merge(pred, on=["panel", "timestamp", "station_id", "link_id"],
                                how="left", validate="one_to_one")
            miss = int(merged.speed_kmh.isna().sum())
            print(f"  state {split:10s} {panel:11s} rows {len(merged):>8,}  unmatched {miss}", flush=True)
            parts.append(merged)
    d = pd.concat(parts, ignore_index=True)
    d["timestamp"] = d.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    p = OUT / "state_submission.csv"
    d[["panel", "timestamp", "station_id", "link_id", "mask_regime",
       "speed_kmh", "flow_vph"]].to_csv(p, index=False)
    print(f"state rows {len(d):,} -> {p}")
    return p


def build_queue() -> Path:
    """Builds the v1 Task 2 queue file from the two condition boosters.

    Each booster's raw score is added to the logit of the train queue prior,
    so the trees learn a correction to the prior. A cell is queued when its
    probability reaches the threshold of its condition (onset or ongoing).

    Returns:
        Path of queue_submission.csv (window_id, timestamp, link_id,
        queue_pred 0/1).
    """
    cfg = pd.read_csv(MODELS / "config.csv", index_col=0).iloc[:, 0]
    thr = {c: float(cfg[f"thr_{c}"]) for c in ("queue_onset", "queue_ongoing")}
    models = {c: lgb.Booster(model_file=str(MODELS / f"lgb_{c}.txt"))
              for c in ("queue_onset", "queue_ongoing")}
    feats = models["queue_onset"].feature_name()
    parts = []
    for split in SPLITS:
        for panel in R.panels():
            if not (R.RELEASE / "task2" / panel / split).is_dir():
                continue
            f = task2infer.features_for_split(panel, split)
            if f.empty:
                continue
            prob = np.zeros(len(f))
            for cond, m in models.items():
                sel = (f.condition == cond).to_numpy()
                if not sel.any():
                    continue
                # Log-odds of the booster plus the log-odds of the train queue prior.
                raw = m.predict(f.loc[sel, feats], raw_score=True) + \
                    predict1_logit(f.loc[sel, "queue_prior"].to_numpy())
                prob[sel] = 1.0 / (1.0 + np.exp(-raw))
            f["queue_pred"] = (prob >= f.condition.map(thr).to_numpy()).astype(int)
            print(f"  queue {split:10s} {panel:11s} rows {len(f):>7,}  positives {int(f.queue_pred.sum()):>6,}",
                  flush=True)
            parts.append(f[["window_id", "timestamp", "link_id", "queue_pred"]])
    d = pd.concat(parts, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    p = OUT / "queue_submission.csv"
    d.to_csv(p, index=False)
    print(f"queue rows {len(d):,} -> {p}")
    return p


def predict1_logit(p, eps=1e-4):
    """Returns log(p / (1 - p)), with NaN read as 0 and p clipped to [eps, 1 - eps]."""
    p = np.clip(np.nan_to_num(p, nan=0.0), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_odme() -> Path:
    """Solves the Task 4 path flows of each split and panel by regularised NNLS.

    Minimises ||A_m f - c||^2 + lam * ||f - b||^2 subject to f >= 0, where A_m
    is the link x path incidence restricted to links with a released count,
    c the released link counts (synthetic_link_counts.csv), b the weak prior
    path flows (synthetic_weak_prior.csv) and lam = 0.05. Each split uses only
    its own released files.

    Returns:
        Path of path_flow_submission.csv (panel, departure_time, path_id,
        origin_zone, destination_zone, path_flow).
    """
    from importlib import util
    # `spec` (a module spec for scripts/39_task4_odme.py) is not used below.
    spec = util.spec_from_file_location("t4", R.REPO / "scripts" / "39_task4_odme.py")
    parts = []
    for split in SPLITS:
        for panel in R.panels():
            net = R.RELEASE / "corridors" / panel / "network"
            inc = pd.read_csv(net / "path_link_incidence.csv", dtype=str)
            paths = pd.read_csv(net / "path_set.csv", dtype=str)
            path_ids = sorted(paths.path_id.astype(str))
            link_ids = sorted(inc.link_id.astype(str).unique())
            pi = {q: i for i, q in enumerate(path_ids)}
            li = {l: i for i, l in enumerate(link_ids)}
            # A: link x path incidence matrix, 1 where the path uses the link.
            A = np.zeros((len(link_ids), len(path_ids)))
            A[inc.link_id.map(li).to_numpy(), inc.path_id.map(pi).to_numpy()] = 1.0
            counts = pd.read_csv(R.RELEASE / "task4" / panel / split / "synthetic_link_counts.csv",
                                 dtype={"link_id": str})
            prior = pd.read_csv(R.RELEASE / "task4" / panel / split / "synthetic_weak_prior.csv",
                                dtype={"path_id": str})
            b = prior.set_index("path_id").reindex(path_ids)
            # Use the first known prior-flow column name that the file has; missing paths get 0.
            bcol = [c for c in ("weak_prior_flow", "prior_flow", "path_flow", "flow") if c in b.columns][0]
            base = pd.to_numeric(b[bcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            # Only links with a released count enter the data term.
            observed = set(counts.link_id)
            measured = np.array([l in observed for l in link_ids], dtype=bool)
            c = counts.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(dtype=float)
            lam = 0.05
            # Stacked system [A_m; sqrt(lam) I] f ~ [c; sqrt(lam) b] solved by scipy's nnls.
            aa = np.vstack([A[measured], np.sqrt(lam) * np.eye(A.shape[1])])
            bb = np.concatenate([c[measured], np.sqrt(lam) * base])
            f, _ = nnls(aa, bb)
            meta = paths.set_index("path_id").reindex(path_ids)
            dep = pd.read_csv(R.RELEASE / "task4" / panel / split / "sample_submission_path_flow.csv",
                              dtype=str)
            parts.append(pd.DataFrame({
                "panel": panel, "departure_time": dep.departure_time.iloc[0],
                "path_id": path_ids,
                "origin_zone": meta.origin_zone.to_numpy(),
                "destination_zone": meta.destination_zone.to_numpy(),
                "path_flow": f}))
            print(f"  odme  {split:10s} {panel:11s} paths {len(path_ids):>5,}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    p = OUT / "path_flow_submission.csv"
    d.to_csv(p, index=False)
    print(f"odme rows {len(d):,} -> {p}")
    return p


def main() -> None:
    """Builds the three v1 parts and merges them with the official merge_submissions.py into OUT/submission.csv."""
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Task 1 ==="); state = build_state()
    print("=== Task 2 ==="); queue = build_queue()
    print("=== Task 4 ==="); odme = build_odme()
    print("=== merge (official script) ===")
    final = OUT / "submission.csv"
    cmd = [str(R.REPO / ".venv" / "bin" / "python"),
           str(R.OFFICIAL / "src" / "merge_submissions.py"),
           "--state", str(state), "--queue", str(queue), "--odme", str(odme),
           "--key", str(R.RELEASE / "submission_key.csv"), "--output", str(final)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:] or r.stderr[-3000:])
    r.check_returncode()
    print(f"\nWROTE {final}")


if __name__ == "__main__":
    main()
