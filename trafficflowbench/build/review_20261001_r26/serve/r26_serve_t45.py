"""R26 BND + decoder threshold 0.45 (PREREG_R26.md probe DEC, selected on v3F odd days with the February twin,
confirmed on v3F even days and v3). Same slots as R26a (PB+M1sf, PB+M1f, A25); block mean >= 0.45; A fallback
unchanged (>= 0.5). Delta vs same-operator C1f (tau 0.5) onto R21a bits."""
# Runs in pipeline/ongoing.sh as `r26_serve_t45.py decide` (t45_decide).
# T45: the BND probabilities decoded with link-group mean >= 0.45 instead of
# 0.5; the never-empty fallback keeps the A decode at 0.5. R26c =
# where(T45 != C1f, T45, R21a); make_ongoing_source.py writes it as the final
# ongoing source. Outputs: decisions_BNDT45_<split>.npz and decide_BNDT45.json.
import os, sys
# Caps the BLAS/OpenMP thread pools at 2 before numpy loads.
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[k] = "2"
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import r26_serve as R  # noqa: E402
S = R.S
TAU = 0.45


def decision_tau(ri, sel, geom, a, tau):
    """Returns the R20-operator decision with link-group threshold `tau` (fallback: A decode at 0.5)."""
    pA = a.mean(1); n = sum([sel[:, i] for i in range(3)] + [geom[:, i] for i in range(3)])
    d = S.pooled(ri, .75 * (n / 6) + .25 * pA) >= tau
    empty = pd.Series(d).groupby(ri.window_id.to_numpy()).transform("sum").to_numpy() == 0
    return np.where(empty, S.pooled(ri, pA) >= .5, d)


def decide(tag="BNDT45"):
    """Decides T45 against the same-operator C1f and writes the delta onto the R21a bits.

    Raises:
      AssertionError: When the C1f or tau-0.5 decode does not reproduce the
        saved C1f / BND decisions.
    """
    bits = S.r21a_bits(); res = {}
    for split in R.SPLITS:
        ri = pd.read_parquet(S.SS / f"{split}_ROW_INDEX.parquet"); mp = pd.read_parquet(S.SS / f"{split}_SUBMISSION_MAP.parquet")
        par = np.load(S.R21S / "prob" / f"{split}_parent_prob.npz"); pb = np.load(R.PROB / f"{split}_PB_prob.npz")
        m1f = np.load(S.R21S / "prob" / f"{split}_M1f_prob.npz")["seed_prob"]; m1sf = np.load(S.R21S / "prob" / f"{split}_M1sf_prob.npz")["seed_prob"]
        A = par["a25_seed_prob"]
        dR = S.decision(ri, (par["selrec_seed_prob"] + m1sf) / 2, (par["geom_seed_prob"] + m1f) / 2, A)
        assert np.array_equal(dR, np.load(S.R21S / f"decisions_C1f_{split}.npz")["arm"].astype(bool)), split
        d50 = decision_tau(ri, (pb["selrec_seed_prob"] + m1sf) / 2, (pb["geom_seed_prob"] + m1f) / 2, A, 0.5)
        assert np.array_equal(d50, np.load(HERE / f"decisions_BND_{split}.npz")["cand"].astype(bool)), "tau 0.5 must reproduce BND"
        dC = decision_tau(ri, (pb["selrec_seed_prob"] + m1sf) / 2, (pb["geom_seed_prob"] + m1f) / 2, A, TAU)
        old = bits[split].astype(bool); delta = np.where(dC != dR, dC, old); r26a = np.load(HERE / f"decisions_BND_{split}.npz")["delta"].astype(bool)
        st = ri[["window_id", "panel"]].copy(); st["chg26a"] = r26a != delta; st["add"] = (~r26a) & delta; st["drop"] = r26a & (~delta)
        st["r26a"] = r26a; st["new"] = delta
        wsum = st.groupby(["window_id", "panel"], sort=False)[["r26a", "new", "chg26a", "add", "drop"]].sum().reset_index()
        np.savez(HERE / f"decisions_{tag}_{split}.npz", delta=delta.astype(np.int8), cand=dC.astype(np.int8), ref=dR.astype(np.int8),
                 r21a=old.astype(np.int8), submission_id=mp.submission_id.to_numpy(np.int64), submission_row_index=mp.submission_row_index.to_numpy(np.int64))
        res[split] = dict(vs_R26a_cells=int(st.chg26a.sum()), vs_R26a_windows=int((wsum.chg26a > 0).sum()), added=int(st["add"].sum()), dropped=int(st["drop"].sum()),
                          empty_windows=int((wsum.new == 0).sum()), vs_R21a_cells=int((old != delta).sum()))
        print(split, json.dumps(res[split]), flush=True)
        print(wsum[wsum.chg26a > 0][["window_id", "r26a", "new", "add", "drop"]].to_string(), flush=True)
    json.dump(dict(tag=tag, sel="PB+M1sf", geom="PB+M1f", a="A25", tau=TAU, **res), open(HERE / f"decide_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "decide": decide()
    elif cmd == "patch": S.HERE = HERE; S.patch("BNDT45", sys.argv[2])
