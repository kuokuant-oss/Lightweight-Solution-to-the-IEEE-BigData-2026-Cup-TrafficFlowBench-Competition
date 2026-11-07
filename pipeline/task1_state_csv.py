"""Write a state-only stand-in for a historical parent submission CSV, from regenerated keyed files.

  python task1_state_csv.py --data <kaggle_public> --out <csv> --layer SPEC [--layer SPEC ...]

SPEC = path.parquet,speed_col,flow_col,method[,mask_col]   (path may be a glob; the files are concatenated)
  - layers are applied in order; a later layer overrides an earlier one on the ids it lists;
  - flow_col (or speed_col) may be '-' to keep the value of the earlier layer;
  - method gives the number text the original builder wrote for that layer:
      np3  = numpy round to 3 decimals (scripts/107_compact_submission.py, the v3 compact file)
      fmt3 = format(x, '.3f')            (every keyed-patch builder of the chain)
      fmt6 = format(x, '.6f')            (scripts/build_strongest_submission_20260922.py, the k* rows)
  - mask_col (optional) keeps only rows where that boolean column is true.

Why: Task 1 programs of the chain read their parent's STATE values from the parent submission CSV
(read_csv, then task == 'state'). Queue and OD rows never feed Task 1, so they are written as
placeholders (0.0 / 0) to keep the original row count and order (some builders stream the parent CSV
and assert 6,980,503 rows). Every state row must be covered by the layers.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd

HEADER = "submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow\n"
CODES = {"np3": 1, "fmt3": 2, "fmt6": 3}


def text(x, code):
    if code == 1:
        return format(float(np.round(x, 3)), ".3f")
    if code == 2:
        return format(float(x), ".3f")
    if code == 3:
        return format(float(x), ".6f")
    raise ValueError(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", action="append", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    key = pd.read_csv(Path(a.data) / "submission_key.csv", usecols=["submission_id", "task"])
    sid = key.submission_id.to_numpy(np.int64); task = key.task.to_numpy()
    n = int(sid.max()) + 1
    v = np.full(n, np.nan); q = np.full(n, np.nan)
    mv = np.zeros(n, np.int8); mq = np.zeros(n, np.int8)
    for spec in a.layer:
        parts = spec.split(",")
        path, scol, fcol, meth = parts[:4]
        mask = parts[4] if len(parts) > 4 else None
        cols = ["submission_id"] + [c for c in (scol, fcol, mask) if c and c != "-"]
        files = sorted(glob.glob(path))
        assert files, f"no file matches {path}"
        d = pd.concat([pd.read_parquet(f, columns=list(dict.fromkeys(cols))) for f in files], ignore_index=True)
        if mask:
            d = d[d[mask].to_numpy(bool)]
        ii = d.submission_id.to_numpy(np.int64)
        assert len(np.unique(ii)) == len(ii), f"{path}: duplicate ids"
        assert (task[np.searchsorted(sid, ii)] == "state").all(), f"{path}: non-state ids"
        code = CODES[meth]
        if scol != "-":
            v[ii] = d[scol].to_numpy(np.float64); mv[ii] = code
        if fcol != "-":
            q[ii] = d[fcol].to_numpy(np.float64); mq[ii] = code
        print(f"layer {Path(path).name} ({len(files)} files): {len(ii):,} rows ({scol}, {fcol}, {meth}{', ' + mask if mask else ''})", flush=True)
    st = sid[task == "state"]
    miss = int(((mv[st] == 0) | (mq[st] == 0)).sum())
    assert miss == 0, f"{miss} state rows not covered"
    with open(out, "x") as f:
        f.write(HEADER)
        for s, t in zip(sid.tolist(), task.tolist()):
            if t == "state":
                f.write(f"{s},state,{text(v[s], mv[s])},{text(q[s], mq[s])},0,0.0\n")
            else:
                f.write(f"{s},{t},0.0,0.0,0,0.0\n")
    print(f"wrote {out}  rows {len(sid):,} (state {len(st):,}; queue/odme rows are placeholders)")


if __name__ == "__main__":
    main()
