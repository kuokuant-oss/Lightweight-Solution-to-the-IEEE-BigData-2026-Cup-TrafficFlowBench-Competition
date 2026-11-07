"""Build the v3 submission: v2 in every respect except the Task 1 calibration.

Only ONE thing changes from build/submission_v2: the Task 1 state answer gets
the scripts/111 recalibration applied to it (src/tfb/calib.py). Task 2 keeps the
v2 model and thresholds, Task 4 keeps the official NNLS at lambda=0.05, so the
leaderboard delta v2 -> v3 is attributable to that one change, the same
single-variable discipline the four clean anchors were built on.

Task 3 has no file of its own; it is scored on the Task 1 answer, which is
exactly why the calibration can move it.

Nothing here writes to build/submission_v1/ or build/submission_v2/.

  .venv/bin/python scripts/112_build_submission_v3.py --oblique --dispersion \
      --models m15_task2_model_v2 --lam-v 1.0 --lam-q 0.0
"""

# Role: record of the v3 build (the learned Task 1 stack with `--stack`, plus
# the bias calibration). The reproduction pipeline does not run this file;
# there, build/review_20260924_ctm/rl/serve_feats.py rebuilds the same stacked
# fill (as in predict_panel_stacked) for each split.
# Outputs under --out (default build/submission_v3): state_submission.csv, the
# v2 queue_submission.csv and path_flow_submission.csv (copied after a SHA-256
# check), the frozen calibration tables, submission.csv and a provenance stamp.

from __future__ import annotations

import argparse, shutil, subprocess, sys
from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Research gate (src/tfb/gate.py): returns at once unless TFB_RESEARCH_GATE=on.
from src.tfb import gate
gate.require()
# B: classical fills (LRTC-TNN and interpolations); physics: link FD
# parameters; predict1: the v1 blend, its fallbacks and bounds; TS: the
# learned Task 1 stack (task1stack).
from src.tfb import baselines as B, calib, grid, physics, predict1, prov, release as R, task1stack as TS

# FORBIDDEN: the delivered v1 and v2 folders, which this builder never writes.
OUT = R.REPO / "build" / "submission_v3"
FORBIDDEN = {(R.REPO / "build" / "submission_v1").resolve(),
             (R.REPO / "build" / "submission_v2").resolve()}


def _v2():
    """Loads scripts/71_build_submission_v2.py as a module (for its build_queue)."""
    spec = util.spec_from_file_location("b71", R.REPO / "scripts" / "71_build_submission_v2.py")
    m = util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def predict_panel_stacked(panel: str, split: str, hist, w, models):
    """predict_panel, but the target cells are filled by the learned stack.

    Same members, same fallback chain, same clips -- only the combination changes.
    The stack is nested over those members (src/tfb/task1stack.py), so this can
    differ from the frozen blend only by what the linear form gives up.
    """
    import numpy as np, pandas as pd
    files = predict1.split_masked_files(panel, split)
    grids = [grid.build_masked(panel, p) for p in files]
    if not grids:
        return pd.DataFrame()
    # LRTC-TNN low-rank tensor completion of each field over all days of the
    # split (link x time x day); day di of the result is one stack member.
    lrtc = {f: B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f], nan=0.0) for g in grids], axis=2),
                          rho=predict1.LRTC_KNOBS[f][0], theta=predict1.LRTC_KNOBS[f][1])
            for f in predict1.FIELDS}
    links = [str(x) for x in grids[0].links]
    par = physics.link_params(panel, links)
    rows = []
    for di, g in enumerate(grids):
        # Stack at the target cells, frozen blend elsewhere.
        out = TS.predict_grid(g, di, lrtc, hist, w, par, models)
        # Fallback for missing or below-floor values: the weekday x time-of-day
        # history mean, then the median of the day's observed values.
        for f, floor in (("speed_kmh", predict1.SPEED_FLOOR), ("flow_vph", predict1.FLOW_FLOOR)):
            h = hist.get((f, pd.Timestamp(g.times[0]).weekday()))
            bad = ~np.isfinite(out[f]) | (out[f] < floor)
            if bad.any() and h is not None:
                out[f] = np.where(bad & np.isfinite(h), h, out[f])
                bad = ~np.isfinite(out[f]) | (out[f] < floor)
            if bad.any():
                obs = g.value[f]
                med = float(np.nanmedian(obs)) if np.isfinite(obs).any() else floor
                out[f] = np.where(bad, med, out[f])
        # Clip speed to [SPEED_FLOOR, SPEED_CAP], floor flow, and flatten to rows (link-major).
        v = np.clip(out["speed_kmh"], predict1.SPEED_FLOOR, predict1.SPEED_CAP)
        q = np.maximum(out["flow_vph"], predict1.FLOW_FLOOR)
        nL, nT = v.shape
        li = np.repeat(np.arange(nL), nT); ci = np.tile(np.arange(nT), nL)
        rows.append(pd.DataFrame({
            "panel": panel,
            "timestamp": pd.to_datetime(pd.Series(np.asarray(g.times)[ci]), utc=True),
            "station_id": np.asarray(g.station, dtype=object)[li],
            "link_id": np.asarray(g.links, dtype=object)[li],
            "speed_kmh": v.reshape(-1), "flow_vph": q.reshape(-1)}))
    return pd.concat(rows, ignore_index=True)


def build_state(out: Path, lam_v: float, lam_q: float, use_stack: bool) -> Path:
    """Builds the calibrated Task 1 state file for validation and private.

    Args:
        out: output folder.
        lam_v: share of the speed bias to remove (calib.apply).
        lam_q: share of the flow bias to remove.
        use_stack: fill target cells with the learned stack instead of the
            frozen blend.

    Returns:
        Path of state_submission.csv (panel, timestamp, station_id, link_id,
        mask_regime, speed_kmh, flow_vph).

    Raises:
        SystemExit: `use_stack` is set but no fitted stack exists, or a
            template row has no prediction.
    """
    w = predict1.load_weights()
    models = TS.load() if use_stack else None
    if use_stack and models is None:
        raise SystemExit("--stack asked for but build/task1_stack/ has no fitted model")
    parts, hist_cache = [], {}
    for split in ("validation", "private"):
        for panel in R.panels():
            if panel not in hist_cache:
                hist_cache[panel] = predict1.historical_mean(panel, predict1.dev_train_files(panel))
            pred = (predict_panel_stacked(panel, split, hist_cache[panel], w, models)
                    if use_stack else predict1.predict_panel(panel, split, hist_cache[panel], w))
            v, q = calib.calibrate_state(pred.speed_kmh.to_numpy(float), pred.flow_vph.to_numpy(float),
                                         lam_v, lam_q, predict1.SPEED_FLOOR, predict1.SPEED_CAP,
                                         predict1.FLOW_FLOOR)
            moved = float(np.abs(v - pred.speed_kmh.to_numpy(float)).mean())
            pred = pred.assign(speed_kmh=v, flow_vph=q)
            tmpl = predict1.template(panel, split)
            merged = tmpl.merge(pred, on=["panel", "timestamp", "station_id", "link_id"],
                                how="left", validate="one_to_one")
            miss = int(merged.speed_kmh.isna().sum())
            print(f"  state {split:10s} {panel:11s} rows {len(merged):>8,}  unmatched {miss}"
                  f"  mean|dv| {moved:.4f}", flush=True)
            if miss:
                raise SystemExit(f"{panel}/{split}: {miss} template rows unmatched")
            parts.append(merged)
    d = pd.concat(parts, ignore_index=True)
    d["timestamp"] = d.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    p = out / "state_submission.csv"
    d[["panel", "timestamp", "station_id", "link_id", "mask_regime",
       "speed_kmh", "flow_vph"]].to_csv(p, index=False)
    print(f"state rows {len(d):,} -> {p}")
    return p


def main() -> None:
    """Builds the v3 submission.

    Parses the arguments, builds the calibrated state file, copies (or with
    --rebuild-queue rebuilds) the Task 2 file and copies the Task 4 file from
    v2, freezes the calibration tables, merges the parts with the official
    merge_submissions.py and writes a provenance stamp.

    Raises:
        SystemExit: the output folder is a delivered v1/v2 folder, or a v2
            file's SHA-256 differs from the expected value.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--models", default="m15_task2_model_v2")
    ap.add_argument("--oblique", action="store_true")
    ap.add_argument("--dispersion", action="store_true")
    ap.add_argument("--stack", action="store_true", help="use the learned Task 1 stack")
    ap.add_argument("--rebuild-queue", action="store_true",
                    help="rebuild Task 2 from --models instead of copying v2's file "
                         "byte-for-byte. Needed when the Task 2 model itself changed; "
                         "it also RETIRES the single-variable attribution against v2.")
    ap.add_argument("--lam-v", type=float, default=1.0)
    ap.add_argument("--lam-q", type=float, default=0.0)
    # Expected SHA-256 of v2's queue and path-flow files; a mismatch stops the build.
    ap.add_argument("--sha-queue", default="f3b3ff16a23996ceee8c15bf00bababed9fcaccaf109da4d24024de433d8a2f5")
    ap.add_argument("--sha-odme", default="d0344e71f9d20211ba2dcf18bde5fde13ad3ec5b16de6f5c6a98be658642f6c6")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else R.REPO / a.out
    if out.resolve() in FORBIDDEN:
        raise SystemExit(f"refusing to write a delivered namespace: {out}")
    out.mkdir(parents=True, exist_ok=True)
    models = R.REPO / "data" / "interim" / a.models
    v2 = _v2()
    print("=== Task 1 (calibrated) ===")
    sp = build_state(out, a.lam_v, a.lam_q, a.stack)
    # Task 2 and Task 4 must be byte-identical to v2 so the leaderboard delta is
    # attributable to Task 1 alone. Copy and verify rather than rebuild: a rebuild
    # that silently differed would break the attribution without announcing it.
    V2 = R.REPO / "build" / "submission_v2"
    qp, op = out / "queue_submission.csv", out / "path_flow_submission.csv"
    pairs = [(V2 / "path_flow_submission.csv", op, a.sha_odme)]
    if a.rebuild_queue:
        print("=== Task 2 (rebuilt from --models) ===")
        qp = v2.build_queue(out, models, a.oblique, a.dispersion)
    else:
        pairs.insert(0, (V2 / "queue_submission.csv", qp, a.sha_queue))
    for src, dst, want in pairs:
        got = prov.sha256_file(src)
        if got != want:
            raise SystemExit(f"{src.name}: v2 SHA {got} != expected {want}; refusing to build")
        shutil.copyfile(src, dst)
        print(f"  copied {dst.name} from v2, SHA verified {got[:16]}...")
    # The calibration table IS the v3 model. data/interim is gitignored; the
    # .gitignore exception is build/**/frozen/**, which is where v1's model lives.
    froz = out / "frozen" / "task1_calibration"
    froz.mkdir(parents=True, exist_ok=True)
    for f in ("speed_bias.csv", "flow_bias.csv", "111_fit_calibration_2030_06.stamp.json"):
        src = calib.DIR / f
        if src.exists():
            shutil.copyfile(src, froz / f)
            print(f"  froze {f}  {prov.sha256_file(froz / f)[:16]}...")
    print("=== merge (official script) ===")
    final = out / "submission.csv"
    cmd = [str(R.REPO / ".venv" / "bin" / "python"),
           str(R.OFFICIAL / "src" / "merge_submissions.py"),
           "--state", str(sp), "--queue", str(qp), "--odme", str(op),
           "--key", str(R.RELEASE / "submission_key.csv"), "--output", str(final)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1200:] or r.stderr[-2500:]); r.check_returncode()
    prov.write_stamp(out / "112_build_submission_v3.stamp.json",
                     prov.stamp(extra_sources=("src/tfb/calib.py",),
                                kind="submission_v3_build",
                                inputs={"models": a.models, "oblique": a.oblique, "stack": a.stack,
                                        "dispersion": a.dispersion,
                                        "lam_v": a.lam_v, "lam_q": a.lam_q},
                                outputs={p.name: prov.sha256_file(p)
                                         for p in [sp, qp, op, final] if p and p.exists()}))
    print("\nv3 build done")


if __name__ == "__main__":
    main()
