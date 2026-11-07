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

# Role: `day_arrays` and `ramp_day` are shared readers; the ongoing serving
# features of the reproduction pipeline read scored-split ramp flows through
# `ramp_day`, and the training-side builders, task2prior, task2spatial and
# task2wave read truth days through `day_arrays`. `nontrain_prior` and
# `features_for_manifest` are the record of the validation table builder.
#
# Grid convention: one day is a (link x 5-minute stamp) array, links in
# milepost order, 288 stamps on a complete day. A window with forecast time T
# at column t reads history columns t-12..t-1 (T-60..T-5 min) and the ramp
# flow at column t-1; its label is the queue mask at horizon columns
# t+1..t+6 (T+5..T+30 min) of the train truth day.
#
# CACHE: folder of cached `nontrain_prior` arrays, one .npz per (panel, date
# set).
CACHE = R.REPO / "data" / "interim" / "validation_v3" / "priors"
# PRIOR_MIN_DEN: fewest eligible observations a (link, time-of-day) cell
# needs to keep its own queue rate; sparser cells take the panel-wide rate.
PRIOR_MIN_DEN = 5.0          # scripts/50: den >= 5 keeps the cell, else global rate


def train_dates_of(spaced_path: Path, calib_from: str) -> list[str]:
    """The train-fold dates scripts/50 accumulated the prior over."""
    # `spaced_path`: parquet of training windows with a `date` column.
    # `calib_from`: first calibration date (YYYY-MM-DD); earlier window dates
    # form the train fold. Returns the sorted distinct train-fold dates.
    win = pd.read_parquet(spaced_path, columns=["date"])
    return sorted({d for d in win.date.astype(str) if d < calib_from})


def day_arrays(panel: str, path, cut: dict[str, float]):
    """scripts/50.day_arrays, verbatim: links in milepost order."""
    # Reads one full-truth mainline day file. Returns (links, mileposts, V, Qf,
    # O, E, Q, times): V speed (km/h), Qf flow (veh/h), O occupancy and E the
    # score-eligibility mask, each (n_links x n_stamps) with NaN where a cell
    # is absent; Q marks eligible cells with speed <= the link's v_cut
    # (63 km/h when `cut` lacks the link); times are the sorted stamps.
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
        """Returns column ``col`` as a NaN-filled (link x stamp) float array."""
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
    # Finds the ramp-state file of the same date as mainline file `name` in
    # `split` ("._" files are macOS metadata and are skipped). Returns columns
    # timestamp, ramp_link_id and flow_vph (veh/h); an empty frame when the
    # split has no ramp file for that date.
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
    # Returns {ramp_link_id: flow_vph} for the rows stamped exactly `ts`.
    f = ramps_day[ramps_day.timestamp == ts]
    return dict(zip(f.ramp_link_id, f.flow_vph))


def _cache_key(train_dates: list[str]) -> str:
    """Returns a cache tag: the date count plus a SHA-256 prefix of the dates."""
    h = hashlib.sha256("\n".join(train_dates).encode()).hexdigest()[:16]
    return f"{len(train_dates)}d_{h}"


def nontrain_prior(panel: str, train_dates: list[str]) -> np.ndarray:
    """(link x time-of-day) queue rate over ``train_dates``, for a non-train date.

    Identical to what scripts/50 hands a calib or eval window: nothing is left
    out, because the date being featurised contributed nothing to the sum.
    """
    # `train_dates`: YYYY-MM-DD train days whose truth files supply labels.
    # Returns a (n_links x n_stamps) float array in milepost order: queued
    # count / eligible count per (link, time of day), with the panel-wide rate
    # in cells under PRIOR_MIN_DEN observations. Reads train truth days only.
    # Raises RuntimeError when the day grids change shape or no day matches.
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
    # Panel-wide queue rate, the fallback for sparse cells.
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
    # Each manifest row is one window; `origin` is the grid column t of the
    # forecast time T in that day's truth file. Features read history columns
    # t-12..t-1 and ramp flow at column t-1 (the last released stamp); the
    # label of step s (1..6, 5 minutes each) is the queue mask at column t+s.
    # Returns the task2feat features plus label, window_id ("panel|date|t"),
    # panel and condition; float64 columns are stored as float32.
    # Raises RuntimeError on a train-fold date, a missing truth day or a link
    # count that differs from the prior.
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
            # Static link attributes are built once per panel and rebuilt only
            # when the day's link count differs.
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
                # Horizon column of each row: step s sits s stamps after T.
                cidx = t + step
                feats["label"] = Q[lidx, cidx].astype(np.int8)
                feats["eligible"] = E[lidx, cidx]
                feats["window_id"] = f"{panel}|{date}|{t}"
                feats["panel"] = panel
                feats["condition"] = w.condition
                # Keeps only cells the official scorer counts (eligible at the
                # horizon stamp).
                parts.append(feats[feats.eligible].drop(columns=["eligible"]))
        if progress:
            print(f"  {panel}: {len(wp):,} windows featurised", flush=True)
    d = pd.concat(parts, ignore_index=True)
    for c in d.select_dtypes("float64").columns:
        d[c] = d[c].astype("float32")
    return d
