"""R21 probe M: world-invariance constraints on the R20 L0 LightGBM recipe (fit <= 2031-01-31, base196).

Variants (all: same rows, labels, weights, params, rounds and seeds as R20 L0; 2 threads):
  M0  exact L0 replication (sanity: must match the saved R20 L0 models)
  M1  + monotone constraints on dynamic queue-state features (physics sign priors, method=intermediate)
  M2  drop static link-identity features (milepost, lanes, free_speed, capacity, length_km, v_cut, dist_to_onramp_km)
  M3  M1 + M2
Outputs per variant: models + seed probabilities on held_matched, held_raw, march_v2, march_v1 (no decoding, no scoring).
"""
# Research record of the monotone ("mono") LightGBM probe; pipeline/ongoing.sh
# does not run it. s1_serve_prob.py imports it for `variant_cols`, which gives
# the feature columns of the served monotone models M1f / M1sf
# (models/M1f_seed*.txt, models/M1sf_seed*.txt; columns of variants M1 / M1s,
# where M1s is M1 without the queue-geometry features).
# Monotone constraints (MONO): lower speed or margin to the queue speed, and
# higher occupancy, queued neighbours or recency, can only raise the queue
# probability. base196: the 98 base features z-scored, then 98 missing flags
# (TRANSFORM_base196.json).
# Usage: m_train_infer.py VARIANT [VARIANT ...]   (or L0saved)
import os, sys
# Caps the BLAS/OpenMP thread pools at 2 before numpy and LightGBM load.
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "2"
sys.dont_write_bytecode = True
from pathlib import Path
import json, time, argparse, gc
import numpy as np, pandas as pd
import lightgbm as lgb

# Original absolute root; bin/tfb-run maps it onto the run tree.
R20 = Path("/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260930_round20")
QD = R20 / "queue_data"
sys.path.insert(0, str(QD))
from transform import transform_block  # noqa: E402

# CACHE: R20 fit and held-out tables; FS: the 98 base features; NAMES:
# base196 column names; CFG: the R20 L0 LightGBM params and rounds per seed.
CACHE = QD / "cache"
OUT = Path(__file__).resolve().parent
SPEC = json.loads((QD / "TRANSFORM_base196.json").read_text())
FS = SPEC["original_features"]
assert len(FS) == 98
NAMES = [f"z_{i}_{n}" for i, n in enumerate(FS)] + [f"missing_{i}_{n}" for i, n in enumerate(FS)]
CFG = json.loads((QD / "INPUT_MANIFEST.json").read_text())["gbdt_config"]

# MONO: monotone sign per feature (+1: a higher value can only raise the
# queue probability; -1: only lower it).
MONO = {  # sign prior for P(queued at a future step)
    "speed_origin": -1, "occ_origin": +1, "margin_origin": -1, "min_speed_hist": -1, "mean_speed_hist": -1,
    "min_margin_hist": -1, "trend_speed_3": -1, "trend_speed_6": -1, "trend_occ_3": +1, "is_queued_origin": +1,
    "n_queued_hist": +1, "margin_up_1": -1, "margin_dn_1": -1, "margin_up_2": -1, "margin_dn_2": -1,
    "margin_up_4": -1, "margin_dn_4": -1, "queued_up_1": +1, "queued_dn_1": +1, "occ_mean_up_1": +1,
    "origin_speed_T": -1, "origin_occ_T": +1, "origin_margin_T": -1, "origin_queued_T": +1,
    "origin_margin_T_up1": -1, "origin_margin_T_dn1": -1, "rec_s": +1, "rec_any": +1, "rec_n": +1,
}
# Feature groups that variants drop (DROPS): static link identity, ramp
# flows, time of day and queue geometry.
STATIC = ["milepost", "lanes", "free_speed", "capacity", "length_km", "v_cut", "dist_to_onramp_km"]
RAMP = ["onramp_flow", "offramp_flow", "onramp_flow_upstream", "onramp_flow_upstream_oblique", "origin_ramp_T",
        "origin_ramp_mask_T", "origin_dramp", "origin_on_ramp_T", "origin_off_ramp_T", "origin_net_ramp_T", "origin_dnet_ramp"]
TOD = ["tod_min", "tod_sin", "tod_cos", "origin_tod_min"]
GEOM = ["g_gap_km","g_gap_links","g_tail_vel","g_rh_w","g_eta_rh","g_eta_vel","g_block_len_km","g_block_links","g_head_age","g_in_block",
        "g_pos_in_block_km","g_clear_eta_rh","g_n_blocks","g_link_age","g_q_up","g_q_in","g_tail_age","g_eta_rh_m_step","g_eta_vel_m_step","g_clear_m_step"]
DROPS = {"M2": STATIC, "M3": STATIC, "M4": RAMP, "M5": TOD, "M6": STATIC + RAMP, "M0s": GEOM, "M1s": GEOM}


def variant_cols(v):
    """Returns the base196 column indices and monotone signs of a variant.

    Args:
      v: Variant name, e.g. 'M0', 'M1', 'M1s'.

    Returns:
      (cols, mono): column indices into base196 (kept z-scored columns, then
      their missing flags) and one constraint per column (0 = none; set only
      for M1, M3 and M1s).
    """
    drop = set(DROPS.get(v, []))
    keep = [i for i, n in enumerate(FS) if n not in drop]
    cols = keep + [98 + i for i in keep]
    mono = [0] * len(cols)
    if v in ("M1", "M3", "M1s"):
        for j, c in enumerate(cols):
            if c < 98 and FS[c] in MONO:
                mono[j] = MONO[FS[c]]
    return cols, mono


def train(v, seed):
    """Trains one seed of a variant on the R20 fit rows and saves models/<v>_seed<seed>.txt (skipped when present)."""
    cols, mono = variant_cols(v)
    names = [NAMES[c] for c in cols]
    out = OUT / "models" / f"{v}_seed{seed}.txt"
    if out.exists():
        return
    out.parent.mkdir(exist_ok=True)
    x = np.load(CACHE / "fit_matched_base196.npy", mmap_mode="r")
    md = np.load(CACHE / "fit_meta.npz")
    P = dict(CFG["params"], seed=20260922 + seed, num_threads=2)
    if any(mono):
        P.update(monotone_constraints=mono, monotone_constraints_method="intermediate")
    X = np.ascontiguousarray(x[:, cols]) if len(cols) < 196 else x
    t0 = time.time()
    ds = lgb.Dataset(X, label=md["label"], weight=md["weight"], feature_name=names, free_raw_data=True, params=P)
    ds.construct()
    del X, x; gc.collect()
    m = lgb.train(P, ds, num_boost_round=CFG["rounds"][str(seed)])
    m.save_model(str(out))
    json.dump(dict(variant=v, seed=seed, cols=len(cols), n_mono=int(sum(1 for z in mono if z)), seconds=time.time() - t0),
              open(out.with_suffix(".json"), "w"))
    print("TRAINED", v, seed, round(time.time() - t0, 1), flush=True)


def infer(v, models=None, tag=None):
    """Writes per-seed probabilities of a variant on the held-out and March tables (prob/<src>_<tag>_prob.npz).

    Args:
      v: Variant whose columns are used.
      models: Boosters to use; defaults to the three saved seeds of `v`.
      tag: Output name; defaults to `v`.
    """
    cols, _ = variant_cols(v)
    tag = tag or v
    if models is None:
        models = [lgb.Booster(model_file=str(OUT / "models" / f"{v}_seed{s}.txt")) for s in range(3)]
    for src in ("held_matched", "held_raw", "march_v2", "march_v1"):
        out = OUT / "prob" / f"{src}_{tag}_prob.npz"
        if out.exists():
            continue
        out.parent.mkdir(exist_ok=True)
        base = np.load(CACHE / f"{src}_base98.npy", mmap_mode="r")
        pred = np.zeros((len(base), 3))
        for b in range(0, len(base), 50000):
            e = min(b + 50000, len(base))
            xx = transform_block(base[b:e], SPEC)[:, cols]
            for s, m in enumerate(models):
                pred[b:e, s] = m.predict(xx, num_threads=2)
        np.savez(out, seed_prob=pred)
        print("INFER", tag, src, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    a = ap.parse_args()
    for v in a.variants:
        if v == "L0saved":  # inference-path parity with the saved R20 models
            ms = [lgb.Booster(model_file=str(QD / "models" / f"L0_geom_seed{s}.txt")) for s in range(3)]
            infer("M0", ms, tag="L0saved")
            continue
        for s in range(3):
            train(v, s)
        infer(v)
