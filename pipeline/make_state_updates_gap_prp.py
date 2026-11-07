"""Generator for the step-14 keyed file main/state_updates_gap_prp.parquet (PRP_plus hole speeds).

The original file (SHA 0e42af13..., 2026-09-29 19:39) was written without a saved script. Its content is
the PRP_plus column of gap_structure/d2_official_updates_prp.parquet (written by
gap_structure/d2_official_prp.py), in the keyed format that main/build_candidate.py reads:

  submission_id  int64   from d2.submission_id
  speed_kmh      float64 round(d2.new_speed_prp, 3)   <- PRP_plus hole speed
  flow_vph       float64 d2.champ_flow                 <- hole flow unchanged (G2v2, via the CAP parent)
  old_speed_kmh  float64 d2.champ_speed                <- parent (CAP) value, checked by the builder
  old_flow_vph   float64 d2.champ_flow

usage: python make_state_updates_gap_prp.py <d2_official_updates_prp.parquet> <out.parquet> [<reference.parquet>]
"""
import hashlib, sys
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else None
d = pd.read_parquet(src)
assert len(d) == 128465 and d.submission_id.notna().all() and (d.submission_id % 1 == 0).all()
k = pd.DataFrame({
    "submission_id": d.submission_id.astype(np.int64),
    "speed_kmh": np.round(d.new_speed_prp.to_numpy(np.float64), 3),
    "flow_vph": d.champ_flow.to_numpy(np.float64),
    "old_speed_kmh": d.champ_speed.to_numpy(np.float64),
    "old_flow_vph": d.champ_flow.to_numpy(np.float64),
})
assert k.submission_id.is_unique
k.to_parquet(out, index=False)
sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"wrote {out}  rows {len(k):,}  sha256 {sha}")
if ref:
    r = pd.read_parquet(ref)
    same_cols = list(r.columns) == list(k.columns) and all(r[c].dtype == k[c].dtype for c in r.columns)
    same_vals = r.reset_index(drop=True).equals(k.reset_index(drop=True))
    rs = hashlib.sha256(open(ref, "rb").read()).hexdigest()
    print(f"reference sha256 {rs}")
    print(f"columns+dtypes identical: {same_cols}   values identical (same row order): {same_vals}   "
          f"file bytes identical: {sha == rs}")
    if not same_vals:
        m = r.merge(k, on="submission_id", suffixes=("_ref", "_new"))
        for c in ["speed_kmh", "flow_vph", "old_speed_kmh", "old_flow_vph"]:
            print(c, "max |diff|", float(np.abs(m[c + "_ref"] - m[c + "_new"]).max()))
