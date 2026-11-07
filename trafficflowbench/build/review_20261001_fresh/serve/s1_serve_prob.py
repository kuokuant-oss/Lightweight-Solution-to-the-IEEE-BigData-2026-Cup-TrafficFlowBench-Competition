"""R21 serving step 1: probabilities on the 80 genuine official ongoing windows (R20 prepared inputs, <=T only).

kinds: parent (current through-Feb selrec3/geomselrec3/A25), L0 (R20 saved models), M<k> (probe-M models),
       T0 (R20 TabM, torch; mps if available else cpu).
Reads R20 serving_stage inputs read-only; writes only under this directory. No decoding, no CSV.
"""
import os, sys
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "2"
sys.dont_write_bytecode = True
from pathlib import Path
import json, argparse, time
import numpy as np, pandas as pd

R20 = Path("/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260930_round20")
Q = R20 / "queue_data"; SS = Q / "serving_stage"
sys.path.insert(0, str(Q))
from transform import transform_block  # noqa: E402
from inference_common import parent_paths  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "prob"; OUT.mkdir(exist_ok=True)
MONO = HERE.parent / "mono"
SPEC = json.loads((Q / "TRANSFORM_base196.json").read_text())
FS = SPEC["original_features"]


def base_of(split):
    b = np.load(SS / f"{split}_base98.npy", mmap_mode="r")
    ri = pd.read_parquet(SS / f"{split}_ROW_INDEX.parquet")
    assert len(b) == len(ri) == 43500
    return b, ri


def run(kind, split):
    out = OUT / f"{split}_{kind}_prob.npz"
    if out.exists():
        print("exists", out); return
    base, ri = base_of(split); t0 = time.time()
    if kind == "parent":
        import lightgbm as lgb
        prob = {}
        for fam, paths in parent_paths("march").items():
            arr = np.empty((len(base), len(paths)))
            for s, p in enumerate(paths):
                m = lgb.Booster(model_file=str(p)); fs = m.feature_name(); assert not set(fs) - set(FS)
                for b in range(0, len(base), 20000):
                    x = pd.DataFrame(np.asarray(base[b:b + 20000]), columns=FS)
                    arr[b:b + len(x), s] = m.predict(x[fs], num_threads=2)
            prob[{"selrec": "selrec_seed_prob", "geomselrec": "geom_seed_prob", "a25": "a25_seed_prob"}[fam]] = arr
        np.savez(out, **prob)
    elif kind == "L0" or kind.startswith("M"):
        import lightgbm as lgb
        sys.path.insert(0, str(MONO)); import m_train_infer as MT  # noqa: E402
        if kind == "L0":
            ms = [lgb.Booster(model_file=str(Q / "models" / f"L0_geom_seed{s}.txt")) for s in range(3)]; cols = list(range(196))
        else:
            ms = [lgb.Booster(model_file=str(MONO / "models" / f"{kind}_seed{s}.txt")) for s in range(3)]; cols, _ = MT.variant_cols({"M1f": "M1", "M1sf": "M1s"}.get(kind, kind))
        pred = np.zeros((len(base), 3))
        for b in range(0, len(base), 20000):
            e = min(b + 20000, len(base)); x = transform_block(np.asarray(base[b:e]), SPEC)[:, cols]
            for s, m in enumerate(ms):
                pred[b:e, s] = m.predict(x, num_threads=2)
        np.savez(out, seed_prob=pred)
    elif kind == "T0":
        import torch
        sys.path.insert(0, str(R20 / "tabm_models")); from model_embedded import make_model, probabilities  # noqa: E402
        torch.set_num_threads(2)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        ms = []
        for s in range(3):
            m = make_model(196); m.load_state_dict(torch.load(R20 / "tabm_models/models" / f"T0_seed{s}.pt", map_location="cpu", weights_only=True)["state_dict"])
            ms.append(m.to(dev).eval())
        pred = np.zeros((len(base), 3))
        with torch.no_grad():
            for b in range(0, len(base), 1024):
                e = min(b + 1024, len(base)); x = torch.from_numpy(transform_block(np.asarray(base[b:e]), SPEC)).to(dev)
                for s, m in enumerate(ms):
                    pred[b:e, s] = probabilities(m(x)).cpu().numpy()
        np.savez(out, seed_prob=pred, device=np.array(dev))
    else:
        raise SystemExit(kind)
    print("SERVED", kind, split, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kinds", nargs="+"); ap.add_argument("--splits", nargs="+", default=["validation", "private"])
    a = ap.parse_args()
    for k in a.kinds:
        for s in a.splits:
            run(k, s)
