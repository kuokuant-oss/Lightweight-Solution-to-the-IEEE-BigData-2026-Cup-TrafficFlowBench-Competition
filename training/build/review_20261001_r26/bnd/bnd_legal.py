"""R26 BND step 2 (PREREG_R26.md): parent selector inputs restricted to what the selection run makes known.
k_known = predecessors the selecting run considered = (tm(T) - max(run start, last selected + 360)) / 5, cap 12.
  L0: n_rej = min(n_srv, k_known); prev_rej = n_rej > 0
  LE: k_known = 0 -> 0; n_srv <= k_known -> n_srv (exact); else round(E[n_true | n_true >= k_known, n_srv]) tabulated on
      train candidates (rep2 candidates_v2, train months only).
Parent selrec/geom only; M1f/M1sf/A25 unchanged; C1f decode. v3 rows use the first run that found the window."""
# Training record for the BND expected-count table. The reproduction does not run this
# program; it reads the shipped table. The program's paths start at the hard-coded
# development root (`ROOT`, mapped by the path layer).
#
# Only `le_table()` produced a shipped file: `le_table.json`, under
#   `trafficflowbench/build/review_20261001_r26/bnd/`. It reads only the train candidate
#   origins `build/review_20260924_ctm/rep2/candidates_v2.parquet` (one row per corridor, train
#   date 2030-06-01 .. 2031-02-28 and 5-minute origin t, with the selector inputs; February
#   included). The serving code uses the table to replace a window's served count of rejected
#   predecessors (`n_rej`) by its expectation when the official scan knew only k of them (LE
#   rule above).
# `main()` evaluates the L0 and LE corrections on the March windows of the R23 ruler
#   (`build/review_20261001_r23/ruler3`, not shipped); this module imports that ruler's
#   `eval_v3`, `scorer` and `inference_common` at load time.
# Invocation: `python bnd_legal.py` (writes `le_table.json` first when it is missing).
import os, sys, json
for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[n] = "2"
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path("/Volumes/Crucial X9/trafficflow/trafficflowbench"); HERE = Path(__file__).resolve().parent
R3 = ROOT / "build/review_20261001_r23/ruler3"; sys.path.insert(0, str(R3))
import eval_v3 as V  # noqa: E402
B = V.B; FS = V.FS
from scorer import compare, family_score, score_native, KEY  # noqa: E402
from inference_common import parent_paths  # noqa: E402


def run_lengths(c, mask_pos, mask_run):
    """for rows where mask_pos: consecutive predecessors (same panel/date) satisfying mask_run (cap 12)"""
    key = {(p, d, t): i for i, (p, d, t) in enumerate(zip(c.panel, c.date, c.t))}
    P, D, T = c.panel.to_numpy(), c.date.to_numpy(), c.t.to_numpy(); out = np.zeros(len(c), int)
    for i in np.flatnonzero(mask_pos):
        k = 0
        while k < 12:
            j = key.get((P[i], D[i], T[i] - 1 - k))
            if j is None or not mask_run[j]: break
            k += 1
        out[i] = k
    return out


def le_table():
    """Returns the expected-count table, building and saving `le_table.json` on first use.

    A candidate origin has the ongoing shape when its history coverage is >= 0.7 and a
    queue is seen in the horizon, in the history and at the last two history steps. A
    shaped candidate whose queue changes (persistence overlap `piou` <= 0.9) can become an
    ongoing window; one with `piou` > 0.9 is rejected by the selector as unchanged. For each
    shaped candidate with `piou` <= 0.9, `ns` is the run of consecutive preceding origins of
    the same day that have the ongoing shape, and `nt` the run of consecutive preceding
    origins rejected as unchanged (both capped at 12).

    Returns:
        Dict (k, n) -> mean of `nt` over those candidates with ns = n and nt >= k, for
        1 <= k <= n <= 12; n when no candidate matches.
    """
    path = HERE / "le_table.json"
    if path.exists(): return {tuple(map(int, k.split(","))): v for k, v in json.load(open(path)).items()}
    c = pd.read_parquet(ROOT / "build/review_20260924_ctm/rep2/candidates_v2.parquet"); c["date"] = c.date.astype(str)
    c = c.sort_values(["panel", "date", "t"]).reset_index(drop=True)
    shape = ((c["cov"] >= 0.7) & c.hor_any.astype(bool) & c.hist_any.astype(bool) & c.last_any.astype(bool)).to_numpy()
    og = shape & (c.piou <= 0.9).to_numpy(); rej = shape & (c.piou > 0.9).to_numpy()
    ns = run_lengths(c, og, shape)[og]; nt = run_lengths(c, og, rej)[og]
    tab = {}
    for k in range(1, 13):
        for n in range(k, 13):
            s = nt[(ns == n) & (nt >= k)]
            tab[(k, n)] = float(s.mean()) if len(s) else float(n)
    json.dump({f"{k},{n}": v for (k, n), v in tab.items()}, open(path, "w"), indent=1)
    return tab


def main():
    """Evaluates the L0 and LE corrections of the parent selector inputs on the R23 March windows."""
    tab = le_table()
    x, ri = V.base(); w = pd.read_parquet(R3 / "windows_v3.parquet")
    w["tm"] = (pd.to_datetime(w.date) - pd.Timestamp("2031-03-01")).dt.days * 1440 + w.t * 5
    start = np.maximum(w.first_start_min.to_numpy(float), np.where(w.prev_selected_gap.notna(), w.tm - w.prev_selected_gap + 360, -1e9))
    w["k_known"] = np.clip((w.tm - start) // 5, 0, 12).astype(int)
    nsrv = pd.Series(np.asarray(x[:, FS.index("n_rej")])).groupby(ri.window_id.to_numpy()).first()
    w["n_srv"] = w.window_id.map(nsrv).astype(int)
    def fix(k, n, mode):
        """Returns the corrected count for k known predecessors and served count n."""
        if n <= k: return n
        if k == 0: return 0
        return k if mode == "L0" else int(round(tab[(k, n)]))
    for mode in ("L0", "LE"): w[f"n_{mode}"] = [fix(k, n, mode) for k, n in zip(w.k_known, w.n_srv)]
    print("k_known==0:", int((w.k_known == 0).sum()), " changed L0:", int((w.n_L0 != w.n_srv).sum()), " changed LE:", int((w.n_LE != w.n_srv).sum()), flush=True)
    w.to_parquet(HERE / "v3_kknown.parquet", index=False)
    wid = pd.Series(ri.window_id.to_numpy())
    par = B.load(R3 / "prob/parent.npz"); keys, elig, meta, pr = V.setup(); keep = ri.window_id.isin(set(keys.window_id)).to_numpy()
    m1f = B.load(R3 / "prob/M1f.npz")["seed_prob"][keep]; m1sf = B.load(R3 / "prob/M1sf.npz")["seed_prob"][keep]
    h = lambda a, b: (a + b) / 2
    def sc(P):
        """Decodes the ongoing mix with parent probabilities `P` and scores it per window."""
        d = keys[KEY].copy(); d["queue_pred"] = B.decode(keys, h(P["selrec_seed_prob"][keep], m1sf), h(P["geom_seed_prob"][keep], m1f), P["a25_seed_prob"][keep]); return score_native(d, elig, meta, keys)
    w0 = sc(par); res = {}
    odd = set(w[pd.to_datetime(w.date).dt.day % 2 == 1].window_id)
    new = set(w.window_id) - (set(pd.read_parquet(ROOT / "build/review_20260924_ctm/ruler2/windows_validation.parquet").window_id) | set(pd.read_parquet(ROOT / "build/review_20260924_ctm/ruler/fakeval/windows_validation.parquet").window_id))
    for mode in ("L0", "LE"):
        out = HERE / f"v3_parent_{mode}.npz"
        if out.exists(): alt = dict(B.load(out))
        else:
            nm = wid.map(dict(zip(w.window_id, w[f"n_{mode}"]))).to_numpy(float)
            rows = np.flatnonzero(nm != np.asarray(x[:, FS.index("n_rej")]))
            xo = np.asarray(x[rows]).copy(); xo[:, FS.index("n_rej")] = nm[rows]; xo[:, FS.index("prev_rej")] = (nm[rows] > 0).astype(float)
            alt = {k: v.copy() for k, v in par.items()}
            for fam, paths in parent_paths("march").items():
                if fam == "a25": continue
                slot = {"selrec": "selrec_seed_prob", "geomselrec": "geom_seed_prob"}[fam]
                for s, p in enumerate(paths):
                    m = lgb.Booster(model_file=str(p)); fs = m.feature_name()
                    alt[slot][rows, s] = m.predict(pd.DataFrame(xo, columns=FS)[fs], num_threads=2)
            np.savez(out, **alt)
        w1 = sc(alt); w1.to_parquet(HERE / f"perwin_v3_{mode}.parquet", index=False)
        chg = set(w[w[f"n_{mode}"] != w.n_srv].window_id)
        r = {}
        for name, sub in (("all", set(w.window_id)), ("changed", chg), ("odd", odd), ("even", set(w.window_id) - odd), ("changed_odd", chg & odd), ("changed_even", chg - odd), ("v3new", new), ("k0", set(w[w.k_known == 0].window_id))):
            a = w0[w0.window_id.isin(sub)].reset_index(drop=True); b = w1[w1.window_id.isin(sub)].reset_index(drop=True)
            c = compare(a, b, bootstrap_sizes=False)
            r[name] = dict(n=len(a), d=c["all"]["delta_family"], lo=c["all"]["date_bootstrap"]["lo"], hi=c["all"]["date_bootstrap"]["hi"], bw=[c["all"]["better"], c["all"]["worse"]])
            print(mode, name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r[name].items()}, flush=True)
        res[mode] = r
    json.dump(res, open(HERE / "bnd_legal_v3.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
