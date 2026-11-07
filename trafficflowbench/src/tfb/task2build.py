"""Task 2 features for an arbitrary window manifest.

scripts/50 builds the training table for one hardcoded window file and writes
one dataset. validation_v3 needs the same features for window sets that file
never contained, so the per-window build is lifted here unchanged. It is a port
of scripts/50, not a re-derivation, and scripts/65 proves the port by rebuilding
frozen windows and diffing every cell against ``m13_task2_dataset``.

Two details are load-bearing and easy to get wrong:

  * the queue prior. scripts/50 accumulates the (link x time-of-day) queue rate
    over the TRAIN-fold window dates only, and subtracts a day's own
    contribution when that day is itself a train day (leave-one-day-out). For a
    calib or eval date nothing is subtracted, so one array per panel serves
    every non-train date. This is NOT the prior src/tfb/task2infer.py uses --
    that one averages all dev days and feeds the submission path. Mixing them
    would shift the frozen model's init_score and make control and candidate
    incomparable, so this module refuses train-fold dates outright.
  * the link axis is milepost order, and the prior array shares it. Everything
    downstream (shift_rows, margin_shock) assumes the corridor chain.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import grid, release as R, score2, task2feat as TF

CACHE = R.REPO / "data" / "interim" / "validation_v3" / "priors"
PRIOR_MIN_DEN = 5.0          # scripts/50: den >= 5 keeps the cell, else global rate


def train_dates_of(spaced_path: Path, calib_from: str) -> list[str]:
    """The train-fold dates scripts/50 accumulated the prior over."""
    win = pd.read_parquet(spaced_path, columns=["date"])
    return sorted({d for d in win.date.astype(str) if d < calib_from})


def day_arrays(panel: str, path, cut: dict[str, float]):
    """scripts/50.day_arrays, verbatim: links in milepost order."""
    f = pd.read_parquet(path, columns=["timestamp", "link_id", "milepost", "speed_kmh",
                                       "flow_vph", "occupancy", "is_score_eligible"])
    f["link_id"] = f.link_id.astype(str)
    order = f[["link_id", "milepost"]].drop_duplicates().sort_values("milepost").reset_index(drop=True)
    links = list(order.link_id)
    li = {k: i for i, k in enumerate(links)}
    times = np.sort(f.timestamp.unique())
    ti = {k: i for i, k in enumerate(times)}
    r = f.link_id.map(li).to_numpy()
    c = f.timestamp.map(ti).to_numpy()
    n = (len(links), len(times))

    def lay(col):
        g = np.full(n, np.nan)
        g[r, c] = pd.to_numeric(f[col], errors="coerce").to_numpy()
        return g

    E = np.zeros(n, bool)
    E[r, c] = f.is_score_eligible.astype(bool).to_numpy()
    V, Qf, O = lay("speed_kmh"), lay("flow_vph"), lay("occupancy")
    vcut = np.array([cut.get(k, 63.0) for k in links])[:, None]
    return links, order.milepost.to_numpy(dtype=float), V, Qf, O, E, (V <= vcut) & E, times


def ramp_day(panel: str, name: str, split: str = "train") -> pd.DataFrame:
    """scripts/50.ramp_day, verbatim."""
    p = sorted((R.RELEASE / "corridors" / panel / split / "ramp_states")
               .glob(f"**/{name.replace('mainline', 'ramp')}"))
    p = [x for x in p if not x.name.startswith("._")]
    if not p:
        return pd.DataFrame(columns=["timestamp", "ramp_link_id", "flow_vph"])
    f = pd.read_parquet(p[0], columns=["timestamp", "ramp_link_id", "flow_vph"])
    f["ramp_link_id"] = f.ramp_link_id.astype(str)
    f["flow_vph"] = pd.to_numeric(f.flow_vph, errors="coerce")
    return f


def ramp_at(ramps_day: pd.DataFrame, ts) -> dict[str, float]:
    """scripts/50.ramp_at, verbatim."""
    f = ramps_day[ramps_day.timestamp == ts]
    return dict(zip(f.ramp_link_id, f.flow_vph))


def _cache_key(train_dates: list[str]) -> str:
    h = hashlib.sha256("\n".join(train_dates).encode()).hexdigest()[:16]
    return f"{len(train_dates)}d_{h}"


def nontrain_prior(panel: str, train_dates: list[str]) -> np.ndarray:
    """(link x time-of-day) queue rate over ``train_dates``, for a non-train date.

    Identical to what scripts/50 hands a calib or eval window: nothing is left
    out, because the date being featurised contributed nothing to the sum.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = _cache_key(train_dates)
    cache = CACHE / f"{panel}.{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["prior"]
    want = set(train_dates)
    cut = score2.v_cut(panel)
    tot_q = tot_n = None
    for path in R.truth_files(panel, "train"):
        from .task2pool import date_of
        if date_of(path) not in want:
            continue
        links, mp, V, Qf, O, E, Q, times = day_arrays(panel, path, cut)
        if tot_q is None:
            tot_q, tot_n = np.zeros_like(Q, float), np.zeros_like(Q, float)
        if Q.shape != tot_q.shape:
            raise RuntimeError(f"{panel}: day grid changed shape, {Q.shape} vs {tot_q.shape}")
        tot_q += Q.astype(float)
        tot_n += E.astype(float)
    if tot_q is None:
        raise RuntimeError(f"{panel}: no train-fold day matched, prior undefined")
    g = float(tot_q.sum() / max(tot_n.sum(), 1.0))
    prior = np.where(tot_n >= PRIOR_MIN_DEN, tot_q / np.maximum(tot_n, 1.0), g)
    np.savez_compressed(cache, prior=prior, train_dates=np.array(train_dates, dtype=object),
                        global_rate=g)
    return prior


def features_for_manifest(man: pd.DataFrame, train_dates: list[str],
                          progress: bool = False) -> pd.DataFrame:
    """One row per (window, link, step) that is score-eligible, plus its label.

    ``man`` needs panel / date / origin / condition. Any date in ``train_dates``
    is refused: its prior would need the leave-one-day-out subtraction, and
    validation_v3 evaluates on calib and eval dates only.
    """
    clash = sorted(set(man.date.astype(str)) & set(train_dates))
    if clash:
        raise RuntimeError(f"manifest contains train-fold dates, prior would not match: {clash[:5]}")
    parts = []
    for panel, wp in man.groupby("panel", sort=True):
        cut = score2.v_cut(panel)
        attach = R.network(panel, "ramp_attachment_map")
        prior = nontrain_prior(panel, train_dates)
        truth_by_date = {}
        for p in R.truth_files(panel, "train"):
            from .task2pool import date_of
            truth_by_date[date_of(p)] = p
        static_cache = None
        for date, wd in wp.groupby("date", sort=True):
            path = truth_by_date.get(date)
            if path is None:
                raise RuntimeError(f"{panel} {date}: no truth day file")
            links, mp, V, Qf, O, E, Q, times = day_arrays(panel, path, cut)
            if prior.shape[0] != len(links):
                raise RuntimeError(f"{panel}: prior links {prior.shape[0]} vs day links {len(links)}")
            rday = ramp_day(panel, path.name)
            if static_cache is None or len(static_cache) != len(links):
                static_cache = TF.static_table(panel, links, grid, R)
            li = {k: i for i, k in enumerate(links)}
            for w in wd.itertuples(index=False):
                t = int(w.origin)
                ramps = TF.ramp_vectors(panel, links, mp, ramp_at(rday, times[t - 1]), attach)
                feats = TF.build_window_features(V, Qf, O, E, links, mp, static_cache, ramps,
                                                 (t - TF.HIST, t), prior, t, len(times))
                step = feats.step.to_numpy()
                lidx = feats.link_id.map(li).to_numpy()
                cidx = t + step
                feats["label"] = Q[lidx, cidx].astype(np.int8)
                feats["eligible"] = E[lidx, cidx]
                feats["window_id"] = f"{panel}|{date}|{t}"
                feats["panel"] = panel
                feats["condition"] = w.condition
                parts.append(feats[feats.eligible].drop(columns=["eligible"]))
        if progress:
            print(f"  {panel}: {len(wp):,} windows featurised", flush=True)
    d = pd.concat(parts, ignore_index=True)
    for c in d.select_dtypes("float64").columns:
        d[c] = d[c].astype("float32")
    return d
