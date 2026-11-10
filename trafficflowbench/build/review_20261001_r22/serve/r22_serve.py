"""R22 serving on the 80 genuine official ongoing windows (R20 prepared <=T inputs), delta-patched onto R21a.

  r22_serve.py prob KIND...                 # probabilities for new kinds: M7f M7sf M8f M8sf (IC), N6M (refit), AM (A-mono)
  r22_serve.py decide TAG --sel E --geom E [--a E]
                                            # slot expressions: '+'-separated kinds averaged; P = parent slot,
                                            # M1f/M1sf = R21 probabilities, A25 = parent A slot (default)
  r22_serve.py patch TAG OUT.csv            # stream R21a, replace only the 87,000 ongoing rows, verify, manifest

Reference = the R21a recipe C1f (sel (P+M1sf)/2, geom (P+M1f)/2, A25) under the same R20 operator (strict Jan31 blocks,
block mean >= .5, A25 never-empty guard). Delta patch: R21a bits, changed only where candidate != same-operator C1f.
"""
# pipeline/ongoing.sh does not run this file; r26_serve.py and r26_serve_t45.py
# import it for the serving folders, `base_of`, `pooled`, `decision`,
# `r21a_bits` and `patch`. The other model kinds of `prob` and `decide` are
# the R22 research record (models under ../ic, ../n6mono and ../amono).
import os, sys
# Caps the BLAS/OpenMP thread pools at 2 before numpy and the models load.
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "2"
sys.dont_write_bytecode = True
from pathlib import Path
import json, argparse, hashlib, itertools, time
import numpy as np, pandas as pd

# Original absolute root; bin/tfb-run maps it onto the run tree. R21A: the
# R21a submission CSV, read when R21a_ongoing_bits.npz is missing and by
# `patch`.
ROOT = Path("/Volumes/Crucial X9/trafficflow/trafficflowbench")
R20 = ROOT / "build/review_20260930_round20"; Q = R20 / "queue_data"; SS = Q / "serving_stage"
R21S = ROOT / "build/review_20261001_fresh/serve"
HERE = Path(__file__).resolve().parent; R22 = HERE.parent
PROB = HERE / "prob"; PROB.mkdir(exist_ok=True)
R21A = ROOT / "build/deliverables/2026-10-01/submission_R21a_R16b_MONO_C1f.csv"
R21A_SHA = "28794f3cb42ff58379bec29061e84b0d9c0c73dcc5090853ddf40fe733b8b705"
SPEC = json.loads((Q / "TRANSFORM_base196.json").read_text()); FS = SPEC["original_features"]
# Package edit (2026-11-10): the split list comes from TFB_SPLITS (default 'validation private', see README).
SPLITS = tuple(os.environ.get("TFB_SPLITS", "validation private").split())


def sha(p):
    """Returns the SHA256 hex digest of file `p`."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def base_of(split):
    """Returns the 98 base features (memory-mapped) and the row index of one split."""
    b = np.load(SS / f"{split}_base98.npy", mmap_mode="r"); ri = pd.read_parquet(SS / f"{split}_ROW_INDEX.parquet")
    assert len(b) == len(ri)  # package edit (2026-11-10): was == 43500 (the released 5-window layout)
    return b, ri


def prob(kind):
    """Writes per-seed probabilities of one R22 model kind for both splits (prob/<split>_<kind>_prob.npz).

    Raises:
      SystemExit: For an unknown kind.
    """
    import lightgbm as lgb
    for split in SPLITS:
        out = PROB / f"{split}_{kind}_prob.npz"
        if out.exists():
            continue
        base, _ = base_of(split); t0 = time.time()
        if kind.startswith("M"):  # IC models (z-scored base196 columns of the variant)
            sys.path.insert(0, str(R22 / "ic")); import ic_train_infer as IC  # noqa: E402
            cols = IC.spec(kind)[0]
            ms = [lgb.Booster(model_file=str(R22 / "ic" / "models" / f"{kind}_seed{s}.txt")) for s in range(3)]
            pred = np.zeros((len(base), 3))
            for b in range(0, len(base), 20000):
                e = min(b + 20000, len(base)); x = IC.MT.transform_block(np.asarray(base[b:e]), SPEC)[:, cols]
                for s, m in enumerate(ms): pred[b:e, s] = m.predict(x, num_threads=2)
            np.savez(out, seed_prob=pred)
        elif kind == "N6M":  # production-recipe monotone refits (raw base98 by name)
            d = {}
            for var, key in (("R_V2selrec", "selrec_seed_prob"), ("R_V2geomselrec", "geom_seed_prob")):
                arr = np.empty((len(base), 3))
                for s in range(3):
                    m = lgb.Booster(model_file=str(R22 / "n6mono" / f"m_{var}_monorefit_seed{s}.txt")); fs = m.feature_name()
                    for b in range(0, len(base), 20000):
                        x = pd.DataFrame(np.asarray(base[b:b + 20000]), columns=FS); arr[b:b + len(x), s] = m.predict(x[fs], num_threads=2)
                d[key] = arr
            np.savez(out, **d)
        elif kind == "AM":
            mem = ["V2_noID_noPrior", "V2s2", "V2s3", "V6_V2geom", "V6s2", "V6s3"]
            arr = np.empty((len(base), 6))
            for s, n in enumerate(mem):
                m = lgb.Booster(model_file=str(R22 / "amono" / f"m_A_{n}_monoctrl.txt")); fs = m.feature_name()
                for b in range(0, len(base), 20000):
                    x = pd.DataFrame(np.asarray(base[b:b + 20000]), columns=FS); arr[b:b + len(x), s] = m.predict(x[fs], num_threads=2)
            np.savez(out, a25_seed_prob=arr)
        else:
            raise SystemExit(kind)
        print("SERVED", kind, split, round(time.time() - t0, 1), flush=True)


def slot(split, expr, which):
    """Returns the per-seed probabilities of one slot, averaged over the kinds in `expr`.

    Args:
      split: 'validation' or 'private'.
      expr: '+'-separated kinds; 'P' and 'A25' are the parent probabilities
        of the slot.
      which: Slot, 'sel', 'geom' or 'a'.
    """
    par = np.load(R21S / "prob" / f"{split}_parent_prob.npz")
    own = {"sel": par["selrec_seed_prob"], "geom": par["geom_seed_prob"], "a": par["a25_seed_prob"]}[which]
    parts = []
    for k in expr.split("+"):
        if k in ("P", "A25"):
            parts.append(own)
        elif k in ("M1f", "M1sf"):
            parts.append(np.load(R21S / "prob" / f"{split}_{k}_prob.npz")["seed_prob"])
        elif k == "N6M":
            parts.append(np.load(PROB / f"{split}_N6M_prob.npz")["selrec_seed_prob" if which == "sel" else "geom_seed_prob"])
        elif k == "AM":
            parts.append(np.load(PROB / f"{split}_AM_prob.npz")["a25_seed_prob"])
        else:
            parts.append(np.load(PROB / f"{split}_{k}_prob.npz")["seed_prob"])
    return sum(parts) / len(parts)


def pooled(ri, p):
    """Returns, for every row, the mean of `p` over its (window, step, link group)."""
    return pd.Series(np.asarray(p, np.float64)).groupby([ri.window_id.to_numpy(), ri.step.to_numpy(), ri.bid.to_numpy()]).transform("mean").to_numpy()


def decision(ri, sel, geom, a):
    """Returns the R20-operator queue decision.

    N = mean of the 3 sel and 3 geom seed columns; link probability =
    .75 N + .25 x mean of `a`. A cell is queued when its link-group mean at
    the same step is >= .5; a window with no queued cell takes the same
    decode of the A probability alone.
    """
    pA = a.mean(1); n = sum([sel[:, i] for i in range(3)] + [geom[:, i] for i in range(3)])
    d = pooled(ri, .75 * (n / 6) + .25 * pA) >= .5
    empty = pd.Series(d).groupby(ri.window_id.to_numpy()).transform("sum").to_numpy() == 0
    return np.where(empty, pooled(ri, pA) >= .5, d)


def r21a_bits():
    """Returns the R21a ongoing bits per split in SUBMISSION_MAP order.

    Loads R21a_ongoing_bits.npz (written by make_r21a_bits.py in the package
    run); otherwise reads the R21a CSV, checks it against the C1f delta and
    saves that file.
    """
    path = HERE / "R21a_ongoing_bits.npz"
    if path.exists():
        z = np.load(path); return {s: z[s] for s in SPLITS}
    assert sha(R21A) == R21A_SHA
    maps = {s: pd.read_parquet(SS / f"{s}_SUBMISSION_MAP.parquet") for s in SPLITS}
    want = {int(sid): (s, i) for s, m in maps.items() for i, sid in enumerate(m.submission_id.to_numpy())}
    bits = {s: np.full(len(m), -1, np.int8) for s, m in maps.items()}
    for ch in pd.read_csv(R21A, usecols=["submission_id", "task", "queue_pred"], chunksize=500000):
        g = ch[ch.submission_id.isin(want.keys())]; assert (g.task == "queue").all()
        for sid, q in zip(g.submission_id.to_numpy(), g.queue_pred.to_numpy()):
            s, i = want[int(sid)]; bits[s][i] = int(q)
    assert all((b >= 0).all() for b in bits.values())
    for s in SPLITS:  # R21a ongoing bits must equal R21's written C1f delta
        assert np.array_equal(bits[s], np.load(R21S / f"decisions_C1f_{s}.npz")["delta"].astype(np.int8)), s
    np.savez(path, **bits)
    return bits


def decide(tag, sel_e, geom_e, a_e):
    """Decides a candidate against the same-operator C1f and writes the delta onto the R21a bits.

    Args:
      tag: Output name.
      sel_e: Slot expression of the selrec slot (see `slot`).
      geom_e: Slot expression of the geomselrec slot.
      a_e: Slot expression of the A slot.
    """
    bits = r21a_bits(); res = {}
    for split in SPLITS:
        ri = pd.read_parquet(SS / f"{split}_ROW_INDEX.parquet"); mp = pd.read_parquet(SS / f"{split}_SUBMISSION_MAP.parquet")
        assert mp[["window_id", "link_id", "step"]].equals(ri[["window_id", "link_id", "step"]])
        dR = decision(ri, slot(split, "P+M1sf", "sel"), slot(split, "P+M1f", "geom"), slot(split, "A25", "a"))
        dC = decision(ri, slot(split, sel_e, "sel"), slot(split, geom_e, "geom"), slot(split, a_e, "a"))
        old = bits[split].astype(bool)
        # same-operator C1f must reproduce R21's arm decisions exactly
        assert np.array_equal(dR, np.load(R21S / f"decisions_C1f_{split}.npz")["arm"].astype(bool)), split
        delta = np.where(dC != dR, dC, old)
        st = ri[["window_id", "panel"]].copy(); st["r21a"] = old; st["ref"] = dR; st["cand"] = dC; st["delta"] = delta
        st["ref_vs_cand"] = dR != dC; st["r21a_vs_delta"] = old != delta
        w = st.groupby(["window_id", "panel"], sort=False)[["r21a", "ref", "cand", "delta", "ref_vs_cand", "r21a_vs_delta"]].sum().reset_index()
        w.to_parquet(HERE / f"transport_{tag}_{split}.parquet", index=False)
        np.savez(HERE / f"decisions_{tag}_{split}.npz", delta=delta.astype(np.int8), cand=dC.astype(np.int8), ref=dR.astype(np.int8),
                 r21a=old.astype(np.int8), submission_id=mp.submission_id.to_numpy(np.int64), submission_row_index=mp.submission_row_index.to_numpy(np.int64))
        res[split] = dict(cells=len(ri), r21a_pos=int(old.sum()), ref_pos=int(dR.sum()), cand_pos=int(dC.sum()), delta_pos=int(delta.sum()),
                          ref_vs_cand_cells=int((dR != dC).sum()), ref_vs_cand_windows=int((w.ref_vs_cand > 0).sum()),
                          changed_cells=int((old != delta).sum()), changed_windows=int((w.r21a_vs_delta > 0).sum()))
        print(split, json.dumps(res[split]), flush=True)
    json.dump(dict(tag=tag, sel=sel_e, geom=geom_e, a=a_e, **res), open(HERE / f"decide_{tag}.json", "w"), indent=1)


def patch(tag, out_name):
    """Writes a candidate CSV: the R21a CSV with the ongoing queue_pred values of decisions_<tag> applied.

    Checks that every other field and row stays byte-identical and writes a
    manifest next to the CSV.

    Args:
      tag: Decisions to apply.
      out_name: File name under candidates/.
    """
    assert sha(R21A) == R21A_SHA
    upd = {}
    for split in SPLITS:
        z = np.load(HERE / f"decisions_{tag}_{split}.npz")
        for row, sid, v, old in zip(z["submission_row_index"], z["submission_id"], z["delta"], z["r21a"]):
            upd[int(row)] = (int(sid), int(v), int(old))
    assert len(upd) == 87000
    out = HERE / "candidates" / out_name; out.parent.mkdir(exist_ok=True); assert not out.exists(), out
    part = out.with_suffix(".csv.partial"); assert not part.exists()
    changed = n = 0
    with R21A.open("rb") as src, part.open("xb") as dst:
        header = src.readline(); assert header.rstrip(b"\r\n") == b"submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow"; dst.write(header)
        for row, line in enumerate(src):
            if row in upd:
                sid, v, old = upd[row]
                f = line.rstrip(b"\r\n").split(b","); assert int(f[0]) == sid and f[1] == b"queue" and float(f[4]) == old
                if v != old:
                    eol = line[len(line.rstrip(b"\r\n")):]; f[4] = str(v).encode(); line = b",".join(f) + eol; changed += 1
            dst.write(line); n += 1
    assert n == 6980503
    diff = 0
    with R21A.open("rb") as a, part.open("rb") as b:
        assert a.readline() == b.readline()
        for row, (la, lb) in enumerate(itertools.zip_longest(a, b)):
            assert la is not None and lb is not None
            if la != lb:
                assert row in upd; fa = la.rstrip(b"\r\n").split(b","); fb = lb.rstrip(b"\r\n").split(b",")
                assert fa[:4] == fb[:4] and fa[5:] == fb[5:] and int(fb[4]) == upd[row][1] != upd[row][2]; diff += 1
    assert diff == changed
    part.rename(out)
    man = dict(candidate=str(out), sha256=sha(out), rows=n, parent=str(R21A), parent_sha256=R21A_SHA, tag=tag,
               changed_queue_rows=changed, changed_other_rows=0, decisions={s: sha(HERE / f"decisions_{tag}_{s}.npz") for s in SPLITS},
               decide=json.load(open(HERE / f"decide_{tag}.json")))
    json.dump(man, open(out.with_suffix(".csv.manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p1 = sp.add_parser("prob"); p1.add_argument("kinds", nargs="+")
    p2 = sp.add_parser("decide"); p2.add_argument("tag"); p2.add_argument("--sel", required=True); p2.add_argument("--geom", required=True); p2.add_argument("--a", default="A25")
    p3 = sp.add_parser("patch"); p3.add_argument("tag"); p3.add_argument("out_name")
    a = ap.parse_args()
    if a.cmd == "prob":
        for k in a.kinds: prob(k)
    elif a.cmd == "decide":
        decide(a.tag, a.sel, a.geom, a.a)
    else:
        patch(a.tag, a.out_name)
