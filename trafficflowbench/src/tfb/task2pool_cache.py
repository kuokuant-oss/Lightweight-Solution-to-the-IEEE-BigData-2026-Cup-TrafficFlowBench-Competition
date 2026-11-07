"""A window-pool cache that cannot silently serve a stale date range.

THE DEFECT THIS REPLACES. scripts/128 had

    if set(w.date) >= dates & set(w.date):
        return w

which is true for every possible pair of sets -- a set always contains its own
intersection with anything. The later ``w.date.isin(dates)`` can only DROP
dates, never add them, so a run that asked for four extra months of source
days would happily return the previous run's shorter pool and report the
longer date list. "Use more recent data" would then have changed only the
declaration.

TWO DATE SETS ARE RECORDED, because they are not the same thing:

    scanned_dates   every source day the sampler actually looked at
    produced_dates  the days that yielded at least one window

A legal source day can produce no window -- the official rules reject a day
with no queue in any horizon, or with too little history coverage. Concluding
"the cache is complete" from ``produced_dates == requested`` would therefore
re-enumerate those days forever, and concluding it from ``produced_dates``
being a subset would hide a genuinely missing day. Completeness is decided on
``scanned_dates`` alone.

The key also carries the panels, the sampler name and its parameters, so a
pool built with different spacing or a different panel list is a cache miss
rather than a silent reuse. A partial hit is completed by scanning only the
missing days and appending.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import task2pool as TP

# Role: no program of the reproduction pipeline imports this module; it is the
# record of the cached training-window pool. VERSION is hashed into the cache
# key, so a new version never reads an older cache.
VERSION = 2


def _key(panels: list[str], sampler: str, params: dict) -> str:
    """Returns the cache key: a SHA-256 prefix of panels, sampler and params.

    VERSION is hashed in as well, so any change to it is a cache miss.
    """
    blob = json.dumps({"panels": sorted(panels), "sampler": sampler,
                       "params": params, "version": VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def chain_pool(dates: set[str], panels: list[str], cache_dir: Path,
               name: str = "chain", rule=None, verbose: bool = True) -> pd.DataFrame:
    """The spacing-chain window pool over ``dates``, cached with its manifest."""
    # `dates`: YYYY-MM-DD train days wanted. `cache_dir` holds
    # <name>_<key>.parquet (the windows) and <name>_<key>.meta.json (scanned,
    # produced and barren dates). Only dates not yet scanned are enumerated
    # with task2pool and chained, then appended to the cache.
    # Returns the cached windows whose date is in `dates`.
    # Raises RuntimeError when a requested date is still unscanned.
    rule = rule or TP.rules()
    params = {"spacing_steps": int(rule.spacing_steps),
              "min_history_coverage": float(rule.min_history_coverage),
              "max_persistence_iou": float(rule.max_persistence_iou),
              "requires_queue_in_horizon": bool(rule.requires_queue_in_horizon)}
    key = _key(panels, "chain_windows", params)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wpath = cache_dir / f"{name}_{key}.parquet"
    mpath = cache_dir / f"{name}_{key}.meta.json"

    have = pd.DataFrame()
    scanned: set[str] = set()
    if wpath.exists() and mpath.exists():
        meta = json.loads(mpath.read_text())
        if meta.get("key") == key:
            have = pd.read_parquet(wpath)
            scanned = set(meta.get("scanned_dates", []))

    missing = sorted(set(dates) - scanned)
    if verbose:
        print(f"  cache {wpath.name}: {len(scanned)} dates scanned, "
              f"{len(missing)} of {len(dates)} requested still missing", flush=True)
    if missing:
        cand = TP.enumerate_candidates(set(missing), panel_list=panels, progress=verbose)
        add = TP.chain_windows(cand, rule) if len(cand) else cand
        have = pd.concat([have, add], ignore_index=True) if len(have) else add
        scanned |= set(missing)
        have = have.drop_duplicates(["panel", "date", "origin"]).sort_values(
            ["panel", "date", "origin"]).reset_index(drop=True)
        have.to_parquet(wpath, index=False)
        produced = sorted(set(have.date.astype(str)))
        mpath.write_text(json.dumps({
            "key": key, "version": VERSION, "panels": sorted(panels),
            "sampler": "chain_windows", "params": params,
            "scanned_dates": sorted(scanned), "n_scanned": len(scanned),
            "produced_dates": produced, "n_produced": len(produced),
            "barren_dates": sorted(scanned - set(produced))}, indent=1))

    out = have[have.date.isin(dates)].reset_index(drop=True)
    still = sorted(set(dates) - scanned)
    if still:
        raise RuntimeError(f"cache did not scan {len(still)} requested dates: {still[:5]}")
    if verbose:
        prod = sorted(set(out.date.astype(str)))
        print(f"  pool {len(out):,} windows; scanned {len(dates)} source dates, "
              f"{len(prod)} produced windows, {len(dates)-len(prod)} barren", flush=True)
    return out
