"""R21 serving step 2: fixed R20 decoder on genuine windows + transport vs the R16b CSV.

For an arm replacing the geom3 slot: same-operator parent decision (selrec3+geom3, A25) vs arm decision
(selrec3+arm3, A25), both .75*N/6+.25*A, block mean >= .5 on strict Jan31 blocks, A25 never-empty guard.
Because the same operator does not replay R16b exactly, the candidate is a DELTA patch: R16b bits, changed only where
arm != same-operator parent. Writes decisions + per-window transport counts. No CSV.
# Runs in pipeline/ongoing.sh as `s2_decide.py P+M1f --sel P+M1sf --tag C1f`.
# C1f: each N slot is the per-seed mean of the production model and the
# monotone model (sel: selrec + M1sf; geom: geomselrec + M1f); N = mean of the
# 6 seed columns; link probability = .75 N + .25 A; a cell is queued when its
# link-group mean (strict blocks through 2031-01-31) at the same step is
# >= .5; never-empty fallback to the A decode. The served bits are the R16b
# bits, changed only where C1f differs from the same-operator parent.
# Outputs: decisions_<tag>_<split>.npz (delta, arm, sameop_parent, r16b,
# submission_id, submission_row_index), transport_<tag>_<split>.parquet and
# decide_<tag>.json. make_r21a_bits.py saves the C1f delta as the R21a bits.
# Usage: s2_decide.py ARM_EXPR [--sel SEL_EXPR] [--tag TAG]
"""
import os, sys
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "2"
sys.dont_write_bytecode = True
from pathlib import Path
import json, argparse
import numpy as np, pandas as pd

# Original absolute paths; bin/tfb-run maps them onto the run tree. R16B: the
# R16b submission CSV, read only when R16b_ongoing_bits.npz is missing.
R20 = Path("/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260930_round20")
SS = R20 / "queue_data/serving_stage"
HERE = Path(__file__).resolve().parent
R16B = Path("/Volumes/Crucial X9/trafficflow/trafficflowbench/build/deliverables/2026-09-30/submission_R16b_R11c_CAPDENS_C85.csv")
R16B_SHA = "05cebca599b04cba99802d2f97ab2a2145e6992cc3af12fa77c415a5770c5d64"
# Package edit (2026-11-10): the split list comes from TFB_SPLITS (default 'validation private', see README).
SPLITS = tuple(os.environ.get('TFB_SPLITS', 'validation private').split())


def pooled(ri, p):
    """Returns, for every row, the mean of `p` over its (window, step, link group)."""
    return pd.Series(np.asarray(p, np.float64)).groupby([ri.window_id.to_numpy(), ri.step.to_numpy(), ri.bid.to_numpy()]).transform("mean").to_numpy()


def decision(ri, p, pA):
    """Returns the queue decision: link-group mean of `p` >= .5, with the never-empty fallback to the same decode of `pA`."""
    d = pooled(ri, p) >= .5
    empty = pd.Series(d).groupby(ri.window_id.to_numpy()).transform("sum").to_numpy() == 0
    return np.where(empty, pooled(ri, pA) >= .5, d)


def r16b_bits():
    """Returns the R16b ongoing bits per split in SUBMISSION_MAP order.

    Loads R16b_ongoing_bits.npz (written by make_r16b_bits.py in the package
    run); otherwise reads the R16b CSV and saves that file.
    """
    path = HERE / "R16b_ongoing_bits.npz"
    if path.exists():
        z = np.load(path); return {s: z[s] for s in SPLITS}
    import hashlib
    h = hashlib.sha256()
    with R16B.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    assert h.hexdigest() == R16B_SHA
    maps = {s: pd.read_parquet(SS / f"{s}_SUBMISSION_MAP.parquet") for s in SPLITS}
    want = {}
    for s, m in maps.items():
        for i, sid in enumerate(m.submission_id.to_numpy()): want[int(sid)] = (s, i)
    bits = {s: np.full(len(m), -1, np.int8) for s, m in maps.items()}
    for ch in pd.read_csv(R16B, usecols=["submission_id", "task", "queue_pred"], chunksize=500000):
        g = ch[ch.submission_id.isin(want.keys())]
        assert (g.task == "queue").all()
        for sid, q in zip(g.submission_id.to_numpy(), g.queue_pred.to_numpy()):
            s, i = want[int(sid)]; bits[s][i] = int(q)
    assert all((b >= 0).all() for b in bits.values())
    np.savez(path, **bits)
    return bits


def run(arm_expr, tag, sel_expr="P"):
    """Decides one arm against the same-operator parent and writes the delta patch.

    Args:
      arm_expr: '+'-separated probability kinds averaged into the geom slot
        ('P' = the parent geomselrec probabilities).
      tag: Output name.
      sel_expr: The same for the selrec slot ('P' = the parent selrec).
    """
    bits = r16b_bits(); res = {}
    for split in SPLITS:
        ri = pd.read_parquet(SS / f"{split}_ROW_INDEX.parquet")
        mp = pd.read_parquet(SS / f"{split}_SUBMISSION_MAP.parquet")
        assert mp[["window_id", "link_id", "step"]].equals(ri[["window_id", "link_id", "step"]])
        par = np.load(HERE / "prob" / f"{split}_parent_prob.npz")
        sel, geom, a = par["selrec_seed_prob"], par["geom_seed_prob"], par["a25_seed_prob"]
        ld = lambda p, own: own if p == "P" else np.load(HERE / "prob" / f"{split}_{p}_prob.npz")["seed_prob"]
        parts = arm_expr.split("+"); arr = sum(ld(p, geom) for p in parts) / len(parts)
        sparts = sel_expr.split("+"); sel_new = sum(ld(p, sel) for p in sparts) / len(sparts)
        pA = a.mean(1)
        nP = sum([sel[:, i] for i in range(3)] + [geom[:, i] for i in range(3)])
        nC = sum([sel_new[:, i] for i in range(3)] + [arr[:, i] for i in range(3)])
        dP = decision(ri, .75 * (nP / 6) + .25 * pA, pA)
        dC = decision(ri, .75 * (nC / 6) + .25 * pA, pA)
        old = bits[split].astype(bool)
        delta = np.where(dC != dP, dC, old)
        st = ri[["window_id", "panel"]].copy()
        st["r16b"] = old; st["sameop_parent"] = dP; st["arm"] = dC; st["delta"] = delta
        st["r16b_vs_parent"] = old != dP; st["parent_vs_arm"] = dP != dC; st["r16b_vs_delta"] = old != delta
        w = st.groupby(["window_id", "panel"], sort=False)[["r16b", "sameop_parent", "arm", "delta", "r16b_vs_parent", "parent_vs_arm", "r16b_vs_delta"]].sum().reset_index()
        w.to_parquet(HERE / f"transport_{tag}_{split}.parquet", index=False)
        np.savez(HERE / f"decisions_{tag}_{split}.npz", delta=delta.astype(np.int8), arm=dC.astype(np.int8), sameop_parent=dP.astype(np.int8),
                 r16b=old.astype(np.int8), submission_id=mp.submission_id.to_numpy(np.int64), submission_row_index=mp.submission_row_index.to_numpy(np.int64))
        res[split] = dict(cells=len(ri), r16b_pos=int(old.sum()), sameop_parent_pos=int(dP.sum()), arm_pos=int(dC.sum()), delta_pos=int(delta.sum()),
                          r16b_vs_parent_cells=int((old != dP).sum()), r16b_vs_parent_windows=int((w.r16b_vs_parent > 0).sum()),
                          parent_vs_arm_cells=int((dP != dC).sum()), parent_vs_arm_windows=int((w.parent_vs_arm > 0).sum()),
                          delta_changed_cells=int((old != delta).sum()), delta_changed_windows=int((w.r16b_vs_delta > 0).sum()))
        print(split, json.dumps(res[split]), flush=True)
    json.dump(res, open(HERE / f"decide_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("arm"); ap.add_argument("--sel", default="P"); ap.add_argument("--tag"); a = ap.parse_args()
    run(a.arm, a.tag or a.arm.replace("+", "_"), a.sel)
