"""R26 BND serving on the 80 genuine official ongoing windows (R20 prepared <=T inputs), delta-patched onto R21a.

The parent selrec/geom models (rep2 training rows) were fit with truth-semantic prev_rej/n_rej (piou-based). Serving
fed the shape-rule values, which assume every shaped predecessor was considered and rejected. The official window index
(published, all origins <= T for the window itself and earlier windows) gives the spacing: predecessors earlier than
previous official origin + 360 min were never considered. LE (PREREG_R26.md, selected on ruler v3 odd days):
  k_known = clip((T - max(scan start, previous official origin + 360)) / 5, 0, 12)
  n_fix = n_srv if n_srv <= k_known; 0 if k_known == 0; else round(E[n_true | n_true >= k_known, n_srv]) (train table)
  prev_rej_fix = n_fix > 0.  Only the parent selrec/geom inputs change; M1f/M1sf/A25 unchanged.
  r26_serve.py prob        # PB probabilities (parent with fixed selector inputs)
  r26_serve.py decide      # candidate C1f-BND vs same-operator C1f; delta onto R21a bits
  r26_serve.py patch OUT   # stream R21a, replace only ongoing rows, verify, manifest
"""
import os, sys
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[k] = "2"
sys.dont_write_bytecode = True
from pathlib import Path
import json, glob
import numpy as np, pandas as pd
ROOT = Path("/Volumes/Crucial X9/trafficflow/trafficflowbench"); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "build/review_20261001_r22/serve")); import r22_serve as S  # noqa: E402
sys.path.insert(0, str(S.Q)); from inference_common import parent_paths  # noqa: E402
FS = S.FS; SPLITS = S.SPLITS; R21S = S.R21S; SS = S.SS
PROB = HERE / "prob"; PROB.mkdir(exist_ok=True)
TAB = {tuple(map(int, k.split(","))): v for k, v in json.load(open(HERE.parent / "bnd/le_table.json")).items()}
DATA = Path("/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public")


def le(k, n):
    if n <= k: return n
    if k == 0: return 0
    return int(round(TAB[(k, n)]))


def official_k():
    rows = []
    for split in SPLITS:
        for f in sorted(glob.glob(str(DATA / f"task2/*/{split}/window_index.csv"))): rows.append(pd.read_csv(f))
    w = pd.concat(rows, ignore_index=True); w["T"] = pd.to_datetime(w.forecast_origin, utc=True)
    w = w.sort_values(["panel", "split", "T"]).reset_index(drop=True)
    start = w.groupby(["panel", "split"]).date.transform("min").pipe(lambda s: pd.to_datetime(s, utc=True)) + pd.Timedelta(minutes=60)
    prev = w.groupby(["panel", "split"]).T.shift(1)
    b = np.maximum(start.astype("int64"), np.where(prev.notna(), (prev + pd.Timedelta(minutes=360)).astype("int64"), np.iinfo("int64").min))
    w["k_known"] = np.clip((w["T"].astype("int64") - b) // (5 * 60 * 10**9), 0, 12).astype(int)
    w["gap"] = (w["T"] - prev).dt.total_seconds() / 60
    return w[w.condition == "queue_ongoing"].reset_index(drop=True)


def prob():
    import lightgbm as lgb
    w = official_k(); info = {}
    for split in SPLITS:
        out = PROB / f"{split}_PB_prob.npz"
        base, ri = S.base_of(split)
        nsrv = pd.Series(np.asarray(base[:, FS.index("n_rej")])).groupby(ri.window_id.to_numpy()).first()
        ws = w[w.split == split].copy(); ws["n_srv"] = ws.window_id.map(nsrv).astype(int)
        ws["n_fix"] = [le(k, n) for k, n in zip(ws.k_known, ws.n_srv)]
        ws.to_parquet(HERE / f"official_k_{split}.parquet", index=False)
        chg = set(ws[ws.n_fix != ws.n_srv].window_id)
        info[split] = dict(ongoing=len(ws), k0=int((ws.k_known == 0).sum()), changed_windows=len(chg), changed=sorted(chg))
        print(split, json.dumps({k: v for k, v in info[split].items() if k != "changed"}), flush=True)
        print(ws[ws.window_id.isin(chg)][["window_id", "gap", "k_known", "n_srv", "n_fix"]].to_string(), flush=True)
        if out.exists(): continue
        par = np.load(R21S / "prob" / f"{split}_parent_prob.npz"); pb = {k: par[k].copy() for k in par.files}
        nm = ri.window_id.map(dict(zip(ws.window_id, ws.n_fix))).to_numpy(float)
        rows = np.flatnonzero(ri.window_id.isin(chg).to_numpy())
        xo = np.asarray(base[rows]).copy(); xo[:, FS.index("n_rej")] = nm[rows]; xo[:, FS.index("prev_rej")] = (nm[rows] > 0).astype(float)
        for fam, paths in parent_paths("march").items():
            if fam == "a25": continue
            sl = {"selrec": "selrec_seed_prob", "geomselrec": "geom_seed_prob"}[fam]
            for s, p in enumerate(paths):
                m = lgb.Booster(model_file=str(p)); fs = m.feature_name()
                pb[sl][rows, s] = m.predict(pd.DataFrame(xo, columns=FS)[fs], num_threads=2)
        unchanged = np.setdiff1d(np.arange(len(ri)), rows)
        for sl in ("selrec_seed_prob", "geom_seed_prob"): assert np.array_equal(pb[sl][unchanged], par[sl][unchanged])
        np.savez(out, **pb)
    json.dump(info, open(HERE / "official_k.json", "w"), indent=1)


def decide(tag="BND"):
    bits = S.r21a_bits(); res = {}
    for split in SPLITS:
        ri = pd.read_parquet(SS / f"{split}_ROW_INDEX.parquet"); mp = pd.read_parquet(SS / f"{split}_SUBMISSION_MAP.parquet")
        assert mp[["window_id", "link_id", "step"]].equals(ri[["window_id", "link_id", "step"]])
        par = np.load(R21S / "prob" / f"{split}_parent_prob.npz"); pb = np.load(PROB / f"{split}_PB_prob.npz")
        m1f = np.load(R21S / "prob" / f"{split}_M1f_prob.npz")["seed_prob"]; m1sf = np.load(R21S / "prob" / f"{split}_M1sf_prob.npz")["seed_prob"]
        A = par["a25_seed_prob"]
        dR = S.decision(ri, (par["selrec_seed_prob"] + m1sf) / 2, (par["geom_seed_prob"] + m1f) / 2, A)
        assert np.array_equal(dR, np.load(R21S / f"decisions_C1f_{split}.npz")["arm"].astype(bool)), split
        dC = S.decision(ri, (pb["selrec_seed_prob"] + m1sf) / 2, (pb["geom_seed_prob"] + m1f) / 2, A)
        old = bits[split].astype(bool); delta = np.where(dC != dR, dC, old)
        st = ri[["window_id", "panel"]].copy(); st["r21a"] = old; st["ref"] = dR; st["cand"] = dC; st["delta"] = delta
        st["ref_vs_cand"] = dR != dC; st["r21a_vs_delta"] = old != delta
        st["add"] = (~old) & delta; st["drop"] = old & (~delta)
        wsum = st.groupby(["window_id", "panel"], sort=False)[["r21a", "ref", "cand", "delta", "ref_vs_cand", "r21a_vs_delta", "add", "drop"]].sum().reset_index()
        wsum.to_parquet(HERE / f"transport_{tag}_{split}.parquet", index=False)
        np.savez(HERE / f"decisions_{tag}_{split}.npz", delta=delta.astype(np.int8), cand=dC.astype(np.int8), ref=dR.astype(np.int8),
                 r21a=old.astype(np.int8), submission_id=mp.submission_id.to_numpy(np.int64), submission_row_index=mp.submission_row_index.to_numpy(np.int64))
        res[split] = dict(cells=len(ri), r21a_pos=int(old.sum()), cand_pos=int(dC.sum()), delta_pos=int(delta.sum()),
                          changed_cells=int((old != delta).sum()), changed_windows=int((wsum.r21a_vs_delta > 0).sum()),
                          added=int(st["add"].sum()), dropped=int(st["drop"].sum()))
        print(split, json.dumps(res[split]), flush=True)
        print(wsum[wsum.r21a_vs_delta > 0][["window_id", "r21a", "delta", "add", "drop"]].to_string(), flush=True)
    json.dump(dict(tag=tag, sel="PB+M1sf", geom="PB+M1f", a="A25", **res), open(HERE / f"decide_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prob": prob()
    elif cmd == "decide": decide()
    elif cmd == "patch":
        S.HERE = HERE; S.patch("BND", sys.argv[2])
