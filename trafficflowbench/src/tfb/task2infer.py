"""Task 2 inference on a scored split.

On validation/private the only history a participant holds is
``task2/<panel>/<split>/window_history.parquet`` -- 12 stamps x all links.
The feature builder is the SAME one the training table used
(src/tfb/task2feat.py), so train and inference cannot drift apart.

The (link x time-of-day) queue prior is computed from dev-fold train days only.
val is never touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import grid, release as R, score2, task2long as TL, task2feat as TF

# Role: runs in the reproduction pipeline. pipeline/onset_head_serve.py and the
# ongoing serving feature builders call features_for_split with oblique=True,
# dispersion=True, clock=True, origin_obs=True and the model's own
# prior_dates; directed and long stay False.
# Time rule: per window it reads the released 12-stamp history (T-60..T-5),
# ramp flows at T-5 and, with origin_obs, the masked-layer cells and ramp
# flows at T itself. The queue prior is built from train-split truth days
# only; scored days never enter it.
# PRIOR_DIR: on-disk cache of queue_prior arrays. CALIB_FROM: first date after
# the 153-day train fold (2030-06-01..2030-10-31).
PRIOR_DIR = R.REPO / "data" / "interim" / "m13_task2_dataset" / "priors"
CALIB_FROM = "2030-11-01"        # scripts/50's split boundary; train fold ends here


def _day_arrays(panel: str, path, cut: dict[str, float]):
    """Reads one truth day as (links, Q, E) for the queue prior.

    Returns:
        links: Link ids sorted by milepost.
        Q: (n_links x n_stamps) bool, eligible cells with speed <= v_cut
            (63 km/h when ``cut`` lacks the link).
        E: (n_links x n_stamps) bool score-eligibility mask.
    """
    f = pd.read_parquet(path, columns=["timestamp", "link_id", "milepost", "speed_kmh",
                                       "is_score_eligible"])
    f["link_id"] = f.link_id.astype(str)
    order = f[["link_id", "milepost"]].drop_duplicates().sort_values("milepost").reset_index(drop=True)
    links = list(order.link_id)
    li = {k: i for i, k in enumerate(links)}
    times = np.sort(f.timestamp.unique())
    ti = {k: i for i, k in enumerate(times)}
    r = f.link_id.map(li).to_numpy(); c = f.timestamp.map(ti).to_numpy()
    n = (len(links), len(times))
    V = np.full(n, np.nan); V[r, c] = pd.to_numeric(f.speed_kmh, errors="coerce").to_numpy()
    E = np.zeros(n, bool); E[r, c] = f.is_score_eligible.astype(bool).to_numpy()
    vcut = np.array([cut.get(k, 63.0) for k in links])[:, None]
    return links, (V <= vcut) & E, E


def train_fold_dates() -> set[str]:
    """The 153 dev days scripts/50 accumulates the prior over.

    Identical to deriving them from m7_task2_windows/spaced.parquet (verified:
    same 153 dates, 2030-06-01..2030-10-31), but stated on the calendar so the
    serving path does not depend on a regenerable interim artifact.
    """
    return {d for d in R.day_split().query("fold == 'dev'").date if d < CALIB_FROM}


def _stamp_at(times, T):
    """The ramp row key for T, matching however this split writes timestamps."""
    # Returns T as an ISO-8601 UTC string ("YYYY-MM-DDTHH:MM:SSZ"); `times` is
    # accepted for the signature and not used.
    s1 = pd.Timestamp(T).strftime("%Y-%m-%dT%H:%M:%SZ")
    return s1


def queue_prior(panel: str, prior_end: str | None = None,
                prior_dates: tuple[str, ...] | None = None) -> tuple[list[str], np.ndarray]:
    """(link x 288) queue rate over the TRAIN-fold days. Cached on disk.

    Must match scripts/50 exactly: that is where the frozen model's init_score
    came from, and every tree was fit relative to it. Averaging all 217 dev days
    here instead of the 153 train-fold days shifts the logit origin of every
    cell under a model that cannot know it moved -- the training/serving skew
    of notes/15 A. Measured before the fix, on the released 80 windows: the two
    definitions disagreed on 33.35% of cells, and reverting this column alone
    recovered 0.031654 of S_queue (doc/plan/task2-parity.md).

    A calib or eval date subtracts nothing, exactly as scripts/50 does for a
    non-train day, so one array per panel serves every scored split.
    """
    # Source days, in priority order: exactly `prior_dates` when given (the
    # serving pipeline passes the model's manifest dates), else every train
    # day up to `prior_end`, else the 153-day train fold. Only train-split
    # truth files are read.
    # Returns (links in milepost order, (n_links x 288) array of queued /
    # eligible counts per time-of-day slot, with the panel-wide rate where a
    # cell has fewer than 5 eligible observations).
    # Raises RuntimeError when `prior_dates` names a date without a train
    # truth file.
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    # the filename carries the definition: the old dev-wide cache must not be
    # picked up silently by the corrected path, and an extended-date prior must
    # not be picked up by the frozen v3 path either
    if prior_dates is not None:
        # C3: the model's own manifest drives serving. The cache tag is the hash
        # of the exact date list, so a model can never be served with a prior
        # built from a different set of days -- including a set that merely
        # shares an end date.
        import hashlib
        tag = "man" + hashlib.sha256("\n".join(sorted(prior_dates)).encode()).hexdigest()[:16]
    elif prior_end is None:
        tag = "trainfold"
    else:
        tag = f"upto{prior_end.replace('-', '')}"
    cache = PRIOR_DIR / f"{panel}.{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["links"]), z["prior"]
    # prior_end accumulates over EVERY released train day up to that date, which
    # is what a model refit on all of them was fit against (scripts/128). The
    # default is the 153-day train fold the shipped v2/v3 model used; passing
    # anything else changes the init_score origin of every cell, so the model
    # that consumes it has to have been fit on the same definition.
    if prior_dates is not None:
        dev = set(prior_dates)
        available = {p.name[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-")
                     for p in R.truth_files(panel, "train")}
        missing = dev - available
        if missing:
            raise RuntimeError(f"{panel}: prior manifest names {len(missing)} dates with no "
                               f"released truth file, e.g. {sorted(missing)[:3]}")
    elif prior_end is None:
        dev = train_fold_dates()
    else:
        dev = {d for d in
               (p.name[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-")
                for p in R.truth_files(panel, "train")) if d <= prior_end}
    cut = score2.v_cut(panel)
    tot_q = tot_n = None
    links_ref = None
    for path in R.truth_files(panel, "train"):
        date = path.name[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-")
        if date not in dev:
            continue
        links, Q, E = _day_arrays(panel, path, cut)
        if links_ref is None:
            links_ref, tot_q, tot_n = links, np.zeros_like(Q, float), np.zeros_like(Q, float)
        tot_q += Q.astype(float); tot_n += E.astype(float)
    g = float(tot_q.sum() / max(tot_n.sum(), 1.0))
    prior = np.where(tot_n >= 5, tot_q / np.maximum(tot_n, 1.0), g)
    np.savez_compressed(cache, links=np.array(links_ref, dtype=object), prior=prior)
    return links_ref, prior


def features_for_split(panel: str, split: str, oblique: bool = False,
                       dispersion: bool = False, long: bool = False,
                       directed: bool = False, prior_end: str | None = None,
                       prior_dates: tuple[str, ...] | None = None,
                       clock: bool = False, origin_obs: bool = False) -> pd.DataFrame:
    """One row per (window, link, step) the template asks about.

    ``oblique`` and ``dispersion`` add the nine columns scripts/103 measured as
    cell A (+0.008421 on the dev OOF, dead cell removed): the oblique upstream
    on-ramp inflow of scripts/73 and the eight dispersion columns of
    scripts/104. BOTH DEFAULT TO FALSE, so every existing caller -- and in
    particular scripts/67's v1 parity baseline -- keeps producing exactly the
    41 columns the frozen model was fit on.

    Whichever flags are passed here must match the flags scripts/50 built the
    training table with. They are the same two flags in both places and they
    reach the same task2feat entry points, which is what keeps the two
    codepaths from drifting; scripts/67 is what proves they did not.
    """
    # Args: `split` is "validation" or "private"; the windows come from
    # task2/<panel>/<split>/window_index.csv. `long` adds the task2long block,
    # `directed` uses task2dir's travel sign, `prior_end` / `prior_dates`
    # choose the prior's source days (see queue_prior), `clock` adds
    # time-of-day columns and `origin_obs` adds the task2origin block of
    # observations at T.
    # Returns one row per (window, link, step) with the features, window_id,
    # panel, condition and the horizon timestamp of the step
    # (forecast_start + 5 min x (step - 1)); an empty frame when the panel has
    # no window in this split.
    from . import task2dir as TD
    sgn = TD.sign(panel) if directed else 1
    idx = score2.window_index(split)
    idx = idx[idx.panel == panel]
    if idx.empty:
        return pd.DataFrame()
    hist = pd.read_parquet(R.RELEASE / "task2" / panel / split / "window_history.parquet")
    hist["window_id"] = hist.window_id.astype(str)
    hist["link_id"] = hist.link_id.astype(str)
    cut = score2.v_cut(panel)
    prior_links, prior = queue_prior(panel, prior_end, prior_dates)
    mp_all = R.network(panel, "links")
    truth_mp = pd.read_parquet(R.truth_files(panel, "train")[0],
                               columns=["link_id", "milepost"]).drop_duplicates()
    truth_mp["link_id"] = truth_mp.link_id.astype(str)
    attach = R.network(panel, "ramp_attachment_map")
    ramp_files = sorted(p for p in (R.RELEASE / "corridors" / panel / split / "ramp_states").glob("**/*.parquet")
                        if not p.name.startswith("._"))
    ramp = pd.concat([pd.read_parquet(p, columns=["timestamp", "ramp_link_id", "flow_vph"])
                      for p in ramp_files], ignore_index=True)
    ramp["ramp_link_id"] = ramp.ramp_link_id.astype(str)

    split_dates = sorted(idx.date.astype(str).unique())
    if long:
        # the same list the training path uses: every date in this split, so the
        # recent-day prior looks back over the realization actually being scored
        from . import predict1 as _P1
        split_dates = sorted(p.name[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-")
                             for p in _P1.split_masked_files(panel, split))
    out = []
    for w in idx.itertuples(index=False):
        h = hist[hist.window_id == w.window_id]
        if h.empty:
            continue
        # Link order: the window's links sorted by the milepost of the first
        # train truth day.
        order = truth_mp[truth_mp.link_id.isin(set(h.link_id))].sort_values("milepost").reset_index(drop=True)
        links = list(order.link_id)
        mp = order.milepost.to_numpy(dtype=float)
        li = {k: i for i, k in enumerate(links)}
        times = np.sort(h.timestamp.unique())
        ti = {k: i for i, k in enumerate(times)}
        r = h.link_id.map(li).to_numpy(); c = h.timestamp.map(ti).to_numpy()
        # History rows of links without a milepost are dropped.
        ok = ~pd.isna(r)
        r = r[ok].astype(int); c = c[ok].astype(int)
        n = (len(links), len(times))
        def lay(col):
            """Returns history column ``col`` as a (link x stamp) array."""
            g = np.full(n, np.nan)
            g[r, c] = pd.to_numeric(h[col], errors="coerce").to_numpy()[ok]
            return g
        V, Qf, O = lay("speed_kmh"), lay("flow_vph"), lay("occupancy")
        E = np.zeros(n, bool); E[r, c] = h.is_score_eligible.astype(bool).to_numpy()[ok]
        static = TF.static_table(panel, links, grid, R)
        # T: the window's forecast origin in UTC.
        T = pd.Timestamp(w.forigin if hasattr(w, "forigin") else w.forecast_origin, tz="UTC")
        # Ramp flow is read at the LAST RELEASED HISTORY STAMP, not at
        # forecast_origin. scripts/50 trains on times[t-1]; forecast_origin is
        # times[t], one 5-minute step later, and it lies outside the 12 stamps
        # the window actually ships. Measured before the fix, on the released 80
        # windows: the three ramp columns disagreed on 24-97% of cells and
        # reverting them recovered 0.012346 of S_queue (doc/plan/task2-parity.md).
        T_ramp = pd.Timestamp(times[-1], tz="UTC") if pd.Timestamp(times[-1]).tz is None \
            else pd.Timestamp(times[-1])
        rf = ramp[ramp.timestamp == times[-1]]
        if rf.empty:
            rf = ramp[ramp.timestamp == T_ramp.strftime("%Y-%m-%dT%H:%M:%SZ")]
        if rf.empty:
            rf = ramp[pd.to_datetime(ramp.timestamp, utc=True) == T_ramp]
        ramps = TF.ramp_vectors(panel, links, mp,
                                dict(zip(rf.ramp_link_id, pd.to_numeric(rf.flow_vph, errors="coerce"))),
                                attach, oblique=oblique, sgn=sgn)
        og = None
        if origin_obs:
            # Observations AT the forecast origin, from this split's released
            # masked layer. discussion/742068 permits any released timestamp
            # <= T; T itself is the one stamp the shipped features never read.
            from . import task2origin as OG
            vc = static["v_cut"].to_numpy(dtype=float)
            rnow = ramp[ramp.timestamp == _stamp_at(times, T)]
            def _rv(fr_):
                """Sums ramp flows per ramp's nearest mainline link.

                On- and off-ramps are added alike; NaN where no attached ramp
                has a reading.
                """
                v = np.full(len(links), np.nan)
                idx = {k: i for i, k in enumerate(links)}
                amap = attach[["ramp_link_id", "nearest_mainline_link_id"]]
                d_ = dict(zip(fr_.ramp_link_id.astype(str),
                              pd.to_numeric(fr_.flow_vph, errors="coerce")))
                for rl, ml in zip(amap.ramp_link_id.astype(str),
                                  amap.nearest_mainline_link_id.astype(str)):
                    j = idx.get(ml)
                    if j is None:
                        continue
                    x = d_.get(rl)
                    if x is not None and np.isfinite(x):
                        v[j] = (0.0 if not np.isfinite(v[j]) else v[j]) + float(x)
                return v
            og = OG.from_masked(panel, split, str(w.date), links,
                                list(times) + [T], len(times),
                                vc, V[:, -1], Qf[:, -1], O[:, -1],
                                _rv(rnow), _rv(rf), sgn)
        # Prior rows in this window's link order; links missing from the prior
        # get NaN.
        pidx = {k: i for i, k in enumerate(prior_links)}
        rows_p = np.array([pidx.get(k, -1) for k in links])
        pr = np.where(rows_p[:, None] >= 0, prior[np.clip(rows_p, 0, None)], np.nan)
        # tod: time-of-day slot of T (0..287); step s reads prior column
        # tod + s, the slot of its horizon stamp.
        tod = T.hour * 12 + T.minute // 5
        # The whole released history (12 stamps) is the history slice.
        f = TF.build_window_features(V, Qf, O, E, links, mp, static, ramps,
                                     (0, len(times)), pr, tod, 288,
                                     dispersion=dispersion, sgn=sgn, clock=clock,
                                     origin=og)
        f["window_id"] = w.window_id
        f["panel"] = panel
        f["condition"] = w.condition
        f["timestamp"] = pd.to_datetime(w.forecast_start, utc=True) + \
            pd.to_timedelta((f.step.to_numpy() - 1) * 5, unit="m")
        if long:
            vc = np.array([cut.get(str(l), 63.0) for l in links], dtype=float)
            lf = TL.long_features(panel, split, str(w.date), [str(l) for l in links], vc,
                                  T, split_dates)
            f = f.join(lf, on="link_id")
        out.append(f)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
