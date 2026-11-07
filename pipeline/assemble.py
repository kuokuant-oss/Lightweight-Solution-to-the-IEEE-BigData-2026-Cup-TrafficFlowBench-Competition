"""Write the submission CSV from the final per-row sources, in submission_key order.

  python assemble.py --data <kaggle_public> --out submission.csv \
      --state A.parquet --state B.parquet ...   (applied in order; a later file overrides an earlier one)
      --onset onset.parquet --ongoing ongoing.parquet --odme odme.parquet [--expect-sha <sha256>]

State files: submission_id, speed_kmh, flow_vph. Every state row must be covered.
Queue files: submission_id, queue_pred (0/1). Queue rows not listed are 0.
OD file:     submission_id, path_flow. Every odme row must be covered.

Number format, the same as the delivered file:
- state speed/flow: round to 3 decimals, printed with exactly 3 decimals;
- path_flow: rounded to 4 decimals and printed as the shortest float repr (as pandas.to_csv does);
- unused numeric fields print as 0.0, queue_pred and the state/odme queue field as integers.
"""
import argparse, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

HEADER = "submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow\n"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", action="append", default=[])
    ap.add_argument("--onset", required=True)
    ap.add_argument("--ongoing", required=True)
    ap.add_argument("--odme", required=True)
    ap.add_argument("--expect-sha")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    key = pd.read_csv(Path(a.data) / "submission_key.csv", usecols=["submission_id", "task"])
    sid = key.submission_id.to_numpy(np.int64)
    task = key.task.to_numpy()
    n = int(sid.max()) + 1
    assert len(np.unique(sid)) == len(sid)

    v = np.full(n, np.nan); q = np.full(n, np.nan)
    for p in a.state:
        d = pd.read_parquet(p, columns=["submission_id", "speed_kmh", "flow_vph"])
        ii = d.submission_id.to_numpy(np.int64)
        assert (task[np.searchsorted(sid, ii)] == "state").all(), f"{p}: non-state ids"
        v[ii] = np.round(d.speed_kmh.to_numpy(np.float64), 3)
        q[ii] = np.round(d.flow_vph.to_numpy(np.float64), 3)
    st = sid[task == "state"]
    miss = int((~np.isfinite(v[st]) | ~np.isfinite(q[st])).sum())
    assert miss == 0, f"{miss} state rows not covered by the state sources"

    qp = np.zeros(n, np.int8); seen = np.zeros(n, bool)
    for p in (a.onset, a.ongoing):
        d = pd.read_parquet(p, columns=["submission_id", "queue_pred"])
        ii = d.submission_id.to_numpy(np.int64)
        assert (task[np.searchsorted(sid, ii)] == "queue").all(), f"{p}: non-queue ids"
        assert not seen[ii].any(), f"{p}: overlaps the other queue file"
        vals = d.queue_pred.to_numpy()
        assert set(np.unique(vals)) <= {0, 1}
        qp[ii] = vals; seen[ii] = True

    pf = np.full(n, np.nan)
    d = pd.read_parquet(a.odme, columns=["submission_id", "path_flow"])
    ii = d.submission_id.to_numpy(np.int64)
    pf[ii] = np.round(d.path_flow.to_numpy(np.float64), 4)
    od = sid[task == "odme"]
    assert np.isfinite(pf[od]).all(), "odme rows not covered"

    with open(out, "x") as f:
        f.write(HEADER)
        for s, t in zip(sid.tolist(), task.tolist()):
            if t == "state":
                f.write(f"{s},state,{v[s]:.3f},{q[s]:.3f},0,0.0\n")
            elif t == "queue":
                f.write(f"{s},queue,0.0,0.0,{int(qp[s])},0.0\n")
            elif t == "odme":
                f.write(f"{s},odme,0.0,0.0,0,{float(pf[s])!r}\n")
            else:
                raise ValueError(t)
    h = sha256(out)
    print(f"wrote {out}  rows {len(sid):,}  sha256 {h}")
    if a.expect_sha:
        print("MATCH" if h == a.expect_sha else f"MISMATCH (expected {a.expect_sha})")
        sys.exit(0 if h == a.expect_sha else 1)


if __name__ == "__main__":
    main()
