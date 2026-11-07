"""Outcome-free <=Jan31 co-switch accumulation; no ML imports/fits/inference."""
# Training record for the strict block map of the R20 ongoing decode. The reproduction does
# not run this program; it loads the shipped map. Its paths resolve from its original place
# in the development repository (`ROOT`); `REL` is the release folder next to it.
#
# Same co-switching rule as `build/review_20260924_newworld/work/blocks/my_blocks.py`, on
#   train days up to 2031-01-31 only (`CUT`: 2030-06-01 .. 2031-01-31, 245 days, no February
#   data), 8 Task 2 corridors, true speeds of the released train files. Queue cut per link:
#   released v_cut, else 0.6 x free-flow speed (105 km/h when the free-flow speed is
#   missing); 63 km/h for a link absent from the network table.
# Output: `strict_blocks_through_20310131.npy` (dict corridor -> {links, co, den, bid}, as
#   `my_blocks.npy`), shipped under
#   `trafficflowbench/build/review_20260930_round20/evaluation_audit/`, and
#   `STRICT_BLOCKS_QA.json` (date range, blocks, differences from the all-day map, SHA256 of
#   every source file). Neither file is overwritten.
# Invocation: `python build_strict_blocks.py`.
import os, sys
sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"
from pathlib import Path
import hashlib, json, time, resource
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(2)
pa.set_io_thread_count(2)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REL = ROOT.parent / "tfb_competition_data/kaggle_public"
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
CUT = "2031-01-31"

def sha(path):
    """Returns the SHA256 hex digest of a file, read in 8 MB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def date_of(path):
    """Returns the date "YYYY-MM-DD" of a `synthetic_mainline_YYYY_MM_DD` file."""
    return path.stem.removeprefix("synthetic_mainline_").replace("_", "-")

def read_day(path, cutoff):
    """Reads one truth day as link x time arrays.

    Args:
        path: Truth parquet of one day.
        cutoff: Link id -> queue cut speed (km/h).

    Returns:
        (links in milepost order, speed V (km/h), eligibility E, queued = (V <= cut) & E).
    """
    # Same arrays as task2build.day_arrays, omitting only unused flow/occupancy.
    f = pd.read_parquet(path, columns=["timestamp", "link_id", "milepost", "speed_kmh", "is_score_eligible"])
    f["link_id"] = f.link_id.astype(str)
    order = f[["link_id", "milepost"]].drop_duplicates().sort_values("milepost").reset_index(drop=True)
    links = list(order.link_id)
    li = {k: i for i, k in enumerate(links)}
    times = np.sort(f.timestamp.unique())
    ti = {k: i for i, k in enumerate(times)}
    rr, cc = f.link_id.map(li).to_numpy(), f.timestamp.map(ti).to_numpy()
    shape = len(links), len(times)
    V = np.full(shape, np.nan)
    V[rr, cc] = pd.to_numeric(f.speed_kmh, errors="coerce").to_numpy()
    E = np.zeros(shape, bool)
    E[rr, cc] = f.is_score_eligible.astype(bool).to_numpy()
    vc = np.array([cutoff.get(k, 63.0) for k in links])[:, None]
    return links, V, E, (V <= vc) & E

def run():
    """Accumulates the co-switching shares per corridor and writes the map and its QA record."""
    start = time.time()
    outpath, metapath = HERE/"strict_blocks_through_20310131.npy", HERE/"STRICT_BLOCKS_QA.json"
    assert not outpath.exists() and not metapath.exists(), "refuse overwrite"
    oldpath = ROOT/"build/review_20260924_ctm/ruler/blocks/my_blocks.npy"
    old = np.load(oldpath, allow_pickle=True).item()
    output, records, sources = {}, [], {}
    for panel in PANELS:
        net = REL/"corridors"/panel/"network"
        links = pd.read_csv(net/"links.csv", dtype={"link_id":str})
        cut = dict(zip(links.link_id, pd.to_numeric(links.free_speed_kmh, errors="coerce").fillna(105.0)*.6))
        fd = pd.read_csv(net/"fd_parameters.csv", dtype={"link_id":str})
        if "v_cut" in fd:
            for r in fd.dropna(subset=["link_id", "v_cut"]).itertuples(index=False):
                cut[str(r.link_id)] = float(r.v_cut)
        for name in ["links.csv", "fd_parameters.csv"]:
            sources[str(net/name)] = sha(net/name)
        allfiles = sorted(p for p in (REL/"corridors"/panel/"train/mainline_states").glob("**/*.parquet") if not p.name.startswith("._"))
        files = [p for p in allfiles if date_of(p) <= CUT]
        assert files and all(date_of(p) <= CUT for p in files)
        order = None
        for path in files:
            L, V, E, q = read_day(path, cut)
            if order is None:
                order = L
                n = len(L)
                num, den = np.zeros(n-1), np.zeros(n-1)
            else:
                assert order == L
            ok = E & ~np.isnan(V)
            # Onset at (link, t): queued at t, free at t-1, observed at both steps.
            on = np.zeros(q.shape, bool)
            on[:, 1:] = q[:, 1:] & ~q[:, :-1] & ok[:, 1:] & ok[:, :-1]
            # Both directions of each adjacent pair i = (i, i+1): onset at one link while the
            # other is observed at t-1 and t and free at t-1; num counts the neighbour queued at t.
            for a, b in [(slice(0,n-1), slice(1,n)), (slice(1,n), slice(0,n-1))]:
                ev = on[a][:, 1:] & ok[b][:, 1:] & ok[b][:, :-1] & ~q[b][:, :-1]
                num += (ev & q[b][:, 1:]).sum(1)
                den += ev.sum(1)
            sources[str(path)] = sha(path)
        co = np.where(den > 0, num/np.maximum(den,1), np.nan)
        # A new block starts after every pair with co < 0.5 or no onset.
        bid = np.concatenate([[0], np.cumsum(~(co >= .5))])
        output[panel] = dict(links=order, co=co, den=den, bid=bid)
        assert old[panel]["links"] == order
        records.append(dict(panel=panel, first_source_date=min(map(date_of,files)), last_source_date=max(map(date_of,files)),
                            source_days=len(files), links=n, blocks=int(bid.max()+1),
                            historical_all_day_bid_mismatches=int((bid != old[panel]["bid"]).sum())))
        print("STRICT_BLOCK_PANEL", panel, records[-1], "seconds", round(time.time()-start,2), flush=True)
    np.save(outpath, output, allow_pickle=True)
    sources[str(oldpath)] = sha(oldpath)
    for p in [Path(__file__), HERE/"STRICT_BLOCKS_PROTOCOL.json", ROOT/"build/review_20260924_newworld/work/blocks/my_blocks.py", ROOT/"src/tfb/task2build.py", ROOT/"src/tfb/score2.py"]:
        sources[str(p)] = sha(p)
    meta = dict(PASS=True, fit_cut=CUT, outcome_free=True, no_model_imports=True, no_model_fit=True, no_model_inference=True,
                recipe="original my_blocks.py unchanged except explicit <=Jan31 source cut; speed/eligibility-only reader equivalent to task2build.day_arrays",
                same_map_for_all_arms_and_parent=True, output=str(outpath), output_sha256=sha(outpath), panels=records,
                source_sha256=sources, seconds=time.time()-start, peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9)
    metapath.write_text(json.dumps(meta,indent=2)+"\n")
    print("STRICT_BLOCKS_PASS", meta["output_sha256"], meta["seconds"], meta["peak_RSS_GB"], flush=True)

if __name__ == "__main__":
    run()
