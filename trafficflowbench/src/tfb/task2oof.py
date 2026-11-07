"""Serving-consistent out-of-fold predictions for Task 2.

Why this exists: every local number so far came from one train/calib/eval cut,
and its features were built the way scripts/50 builds TRAINING data -- the
queue prior with a leave-one-day-out term on train days. Serving never
subtracts anything (doc/plan/task2-parity.md), so the distribution the model
was fit on is not quite the one it is scored on, and a single cut gives one
noisy estimate over 515 windows with no coverage of the rest of dev.

This module builds the other thing: every dev window gets a prediction from a
model that never saw its day, with features computed exactly as the serving
path computes them.

Serving-consistency, concretely (each point verified in the parity audit):

  * the queue prior for a fold comes only from that fold's training days and
    takes no leave-one-day-out term -- the held-out day contributes nothing to
    it in the first place, so the training and serving definitions coincide;
  * ramp flow is read at the last history stamp, times[t-1];
  * the 12-step history and the link axis are the ones the released
    window_history carries -- scripts/67 showed 37 of 41 feature columns, the
    whole link set and every spatial feature already agreed cell for cell.

Only ``queue_prior`` depends on the fold, so the base features are built once
and that one column is recomputed per fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import grid, release as R, score2, task2build as TB, task2feat as TF

# scripts/51's Bolt configuration (notes/07 B), unchanged: the OOF baseline has
# to reproduce the frozen method, only out of fold.
PARAMS = dict(boosting_type="gbdt", objective="binary", metric="binary_logloss",
              num_leaves=5000, learning_rate=0.1, feature_fraction=1.0,
              bagging_fraction=0.8, bagging_freq=5,
              lambda_l1=8.544245989665272, lambda_l2=0.09577740930772316,
              min_child_samples=10, verbose=-1, num_threads=0)
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 200
CONDITIONS = ("queue_onset", "queue_ongoing")
DROP = ["link_id", "label", "window_id", "panel", "condition", "fold", "step", "date"]


def logit(p, eps=1e-4):
    p = np.clip(np.nan_to_num(p, nan=0.0), eps, 1 - eps)
    return np.log(p / (1 - p))


WEIGHT_SCHEMES = ("cells", "balanced")


def window_weights(d: pd.DataFrame, scheme: str = "cells") -> np.ndarray:
    """Training sample weights. Both schemes give every window total weight 1.

    cells     scripts/51's weighting: 1/(cells in the window). This equalises
              WINDOWS, which is what the four-level metric does, but inside a
              window the positives carry only P/n_cells of the mass -- measured
              on the OOF population, 0.3% for a queue_onset window and 17% for
              a queue_ongoing one.

    balanced  the same window total, split half to the positives and half to
              the negatives. Parameter-free.

              notes/07 D is the source of the question: the metric is a
              per-window IoU while per-cell logloss weights every cell alike,
              "兩者的落差有多大,取決於視窗大小的分布,尚未量測". The OOF error
              breakdown measures it -- windows with 1-15 true cells hold
              +0.152 of the +0.188 total headroom while windows over 200 true
              cells hold +0.005, which is the reverse of how the gradient mass
              is currently distributed. notes/14 A/B are the medical-imaging
              precedents for inverse-size reweighting, and notes/14 B argues
              specifically for a BOUNDED boost; splitting a fixed window budget
              is bounded by construction, since no window can exceed total
              weight 1.

              notes/14 E's boundary still applies: per-window soft-IoU is not
              row-additive, so this is a proxy that has to be measured, not a
              derivation of the metric.
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"scheme must be one of {WEIGHT_SCHEMES}, got {scheme!r}")
    n = d.groupby("window_id").link_id.transform("size").to_numpy(dtype=float)
    if scheme == "cells":
        w = 1.0 / np.maximum(n, 1.0)
    else:
        y = d.label.to_numpy() > 0
        npos = d.groupby("window_id").label.transform("sum").to_numpy(dtype=float)
        nneg = np.maximum(n - npos, 1.0)
        w = np.where(y, 0.5 / np.maximum(npos, 1.0), 0.5 / nneg)
    return w / w.mean()


def date_folds(dates: list[str], n_folds: int) -> dict[str, int]:
    """Contiguous blocks of dates, so a fold holds out whole days.

    Blocked rather than rolling-origin on purpose: the question here is where
    the error is, which needs every window to have an out-of-fold prediction.
    The per-fold scores are reported alongside, and a time trend shows up there.
    """
    blocks = np.array_split(np.array(sorted(dates)), n_folds)
    return {d: k for k, b in enumerate(blocks) for d in b}


def panel_prior(panel: str, source_dates: set[str]) -> np.ndarray:
    """(link x time-of-day) queue rate over ``source_dates``. No LOO -- see module docstring."""
    cut = score2.v_cut(panel)
    tot_q = tot_n = None
    from .task2pool import date_of
    for path in R.truth_files(panel, "train"):
        if date_of(path) not in source_dates:
            continue
        _, _, _, _, _, E, Q, _ = TB.day_arrays(panel, path, cut)
        if tot_q is None:
            tot_q, tot_n = np.zeros_like(Q, float), np.zeros_like(Q, float)
        tot_q += Q.astype(float)
        tot_n += E.astype(float)
    if tot_q is None:
        raise RuntimeError(f"{panel}: no source day for the prior")
    g = float(tot_q.sum() / max(tot_n.sum(), 1.0))
    return np.where(tot_n >= TB.PRIOR_MIN_DEN, tot_q / np.maximum(tot_n, 1.0), g)


def panel_prior_loo(panel: str, source_dates: set[str]):
    """(link x tod) queue rate over ``source_dates``, with a leave-one-day-out
    variant for each source day.

    Returns (full, per_day) where ``full`` is the plain rate over every source
    day -- what SERVING uses, because a scored day is never in the source set --
    and ``per_day[date]`` is the rate with that day's own contribution removed,
    which is what a TRAINING row on that day must see. This is scripts/50's
    rule, the one the shipped v2/v3 model was fit against; without it a refit
    over the full train period would hand every training row its own horizon
    through the init_score while serving gets no such term. The v2 parity audit
    priced one such prior mismatch at 0.031654 of S_queue.
    """
    cut = score2.v_cut(panel)
    from .task2pool import date_of
    tot_q = tot_n = None
    per = {}
    for path in R.truth_files(panel, "train"):
        date = date_of(path)
        if date not in source_dates:
            continue
        _, _, _, _, _, E, Q, _ = TB.day_arrays(panel, path, cut)
        q, n = Q.astype(float), E.astype(float)
        if tot_q is None:
            tot_q, tot_n = np.zeros_like(q), np.zeros_like(n)
        tot_q += q
        tot_n += n
        per[date] = (q, n)
    if tot_q is None:
        raise RuntimeError(f"{panel}: no source day for the prior")
    g = float(tot_q.sum() / max(tot_n.sum(), 1.0))
    full = np.where(tot_n >= TB.PRIOR_MIN_DEN, tot_q / np.maximum(tot_n, 1.0), g)
    out = {}
    for date, (q, n) in per.items():
        num, den = tot_q - q, tot_n - n
        gd = float(num.sum() / max(den.sum(), 1.0))
        out[date] = np.where(den >= TB.PRIOR_MIN_DEN, num / np.maximum(den, 1.0), gd)
    return full, out


def apply_prior_loo(d: pd.DataFrame, priors: dict[str, tuple]) -> np.ndarray:
    """Look up the leave-one-day-out prior for rows whose date is a source day,
    and the plain full-period prior for every other row -- exactly what serving
    sees for a date outside the prior's source set."""
    out = np.empty(len(d), dtype=np.float32)
    li = d._link_idx.to_numpy()
    ti = d._tod_idx.to_numpy()
    da = d.date.to_numpy()
    for panel, idx in d.groupby("panel", sort=False).indices.items():
        full, per = priors[panel]
        for date in np.unique(da[idx]):
            sel = idx[da[idx] == date]
            p = per.get(date, full)
            out[sel] = p[li[sel], ti[sel]]
    return out


def dow_group_7(w: int) -> int:
    """scripts/74's grouping: one group per day of week, Mon=0..Sun=6."""
    return int(w)


def dow_group_weekend(w: int) -> int:
    """Weekday / weekend. Mon-Fri -> 0, Sat-Sun -> 1.

    scripts/74 rejected the 7-way form at -0.047340 (0/8 panels, 0/5 folds). The
    diagnosed cause was support: 7 groups cut a cell from ~147 source days to
    ~21, and PRIOR_MIN_DEN = 5 never rejects a 21-day cell, so the noise went
    straight into the init_score. This key leaves ~105 / ~42.
    """
    return int(w >= 5)


def panel_prior_daytype(panel: str, source_dates: set[str], key=dow_group_7,
                        n_groups: int = 7) -> dict[int, np.ndarray]:
    """(link x time-of-day) queue rate per day type, over ``source_dates``.

    ``key`` maps a weekday index to a group and ``n_groups`` counts them. The
    defaults are scripts/74's 7-way grouping, so that run stays reproducible.

    notes/07 B is the mechanism: Bolt groups its target encoding by
    ``(edge, condition)`` rather than by edge alone, and swaps a sparse key for a
    fallback value. Here the extra key is the calendar day type -- notes/07 B
    measured Wed/Thu daily queue totals at 1.48-1.49x Monday's while the feature
    table carries no day-of-week at all, and notes/17 A2's source calls for
    capturing "traffic daily, weekly distribution".

    The fallback is hierarchical and reuses PRIOR_MIN_DEN, so no new threshold is
    introduced: a (link, tod, dow) cell with fewer than 5 readings falls back to
    the all-day (link, tod) rate, which itself falls back to the panel global
    rate. Where the conditioning has no support the candidate therefore returns
    exactly what panel_prior returns.
    """
    bad = {key(i) for i in range(7)} - set(range(n_groups))
    if bad:
        raise ValueError(f"key maps a weekday to {sorted(bad)}, outside range({n_groups}); "
                         f"the range(n_groups) fallback loop would drop it silently")
    cut = score2.v_cut(panel)
    from .task2pool import date_of
    tot_q = tot_n = None
    dow_q: dict[int, np.ndarray] = {}
    dow_n: dict[int, np.ndarray] = {}
    for path in R.truth_files(panel, "train"):
        date = date_of(path)
        if date not in source_dates:
            continue
        _, _, _, _, _, E, Q, _ = TB.day_arrays(panel, path, cut)
        if tot_q is None:
            tot_q, tot_n = np.zeros_like(Q, float), np.zeros_like(Q, float)
        q, n = Q.astype(float), E.astype(float)
        tot_q += q
        tot_n += n
        k = key(pd.Timestamp(date).dayofweek)
        if k not in dow_q:
            dow_q[k] = np.zeros_like(tot_q)
            dow_n[k] = np.zeros_like(tot_n)
        dow_q[k] += q
        dow_n[k] += n
    if tot_q is None:
        raise RuntimeError(f"{panel}: no source day for the prior")
    g = float(tot_q.sum() / max(tot_n.sum(), 1.0))
    all_day = np.where(tot_n >= TB.PRIOR_MIN_DEN, tot_q / np.maximum(tot_n, 1.0), g)
    out = {}
    for k in range(n_groups):
        if k not in dow_q:                   # day type absent from this fold
            out[k] = all_day
            continue
        rate = dow_q[k] / np.maximum(dow_n[k], 1.0)
        out[k] = np.where(dow_n[k] >= TB.PRIOR_MIN_DEN, rate, all_day)
    return out


def apply_prior_daytype(d: pd.DataFrame, priors: dict[str, dict[int, np.ndarray]],
                        key=dow_group_7) -> np.ndarray:
    """Look the fold's day-type prior up at each row's (link, time-of-day) index.

    ``key`` must be the one the priors were built with. With the default the
    group map is the identity, so this is elementwise what it was before.

    The horizon never crosses midnight (origins run to 280 of 288 five-minute
    slots and the horizon is 6), so the window's own date fixes the day type.
    """
    out = np.empty(len(d), dtype=np.float32)
    gmap = np.array([key(i) for i in range(7)], dtype=np.int64)
    grp = gmap[pd.to_datetime(d.date).dt.dayofweek.to_numpy()]
    li = d._link_idx.to_numpy()
    ti = d._tod_idx.to_numpy()
    for panel, idx in d.groupby("panel", sort=False).indices.items():
        pp = priors[panel]
        for k in np.unique(grp[idx]):
            sel = idx[grp[idx] == k]
            out[sel] = pp[int(k)][li[sel], ti[sel]]
    return out


def base_features(win: pd.DataFrame, progress: bool = False, oblique: bool = False,
                  dow: bool = False, newell: bool = False,
                  wave: dict[str, np.ndarray] | None = None,
                  spatial: dict | None = None,
                  time_oblique: bool = False,
                  accumulation: bool = False,
                  dispersion: bool = False,
                  directed: bool = False,
                  clock: bool = False,
                  origin_obs: bool = False,
                  split: str = "train",
                  include_ineligible: bool = False) -> pd.DataFrame:
    """Every feature except the fold-dependent prior, for the given windows.

    ``queue_prior`` is written as the (link, time-of-day) INDEX rather than a
    value, so a fold's prior can be substituted later without rebuilding.
    ``include_ineligible`` retains all physical query cells and exposes future
    eligibility only as ``_eligible`` evaluation metadata, never a feature.
    """
    parts = []
    from . import task2dir as TD
    if origin_obs:
        from . import task2origin as OG
    for panel, wp in win.groupby("panel", sort=True):
        sgn = TD.sign(panel) if directed else 1
        cut = score2.v_cut(panel)
        attach = R.network(panel, "ramp_attachment_map")
        from .task2pool import date_of
        by_date = {date_of(p): p for p in R.truth_files(panel, "train")}
        static_cache = fd_cache = None
        for date, wd in wp.groupby("date", sort=True):
            links, mp, V, Qf, O, E, Q, times = TB.day_arrays(panel, by_date[date], cut)
            n_times = len(times)
            dummy = np.zeros((len(links), n_times), float)
            rday = TB.ramp_day(panel, by_date[date].name)
            if static_cache is None or len(static_cache) != len(links):
                static_cache = TF.static_table(panel, links, grid, R)
                fd_cache = TF.newell_table(panel, links, grid) if newell else None
            li = {k: i for i, k in enumerate(links)}
            for w in wd.itertuples(index=False):
                t = int(w.origin)
                ramps = TF.ramp_vectors(panel, links, mp, TB.ramp_at(rday, times[t - 1]),
                                        attach, oblique=oblique, sgn=sgn)
                og = None
                if origin_obs:
                    rprev = TB.ramp_at(rday, times[t - 1])
                    rnow = TB.ramp_at(rday, times[t])
                    amap = attach[["ramp_link_id", "nearest_mainline_link_id"]]
                    def _rvec(d):
                        v = np.full(len(links), np.nan)
                        idx = {k: i for i, k in enumerate(links)}
                        for rl, ml in zip(amap.ramp_link_id.astype(str),
                                          amap.nearest_mainline_link_id.astype(str)):
                            j = idx.get(ml)
                            if j is None:
                                continue
                            x = d.get(rl)
                            if x is not None and np.isfinite(x):
                                v[j] = (0.0 if not np.isfinite(v[j]) else v[j]) + float(x)
                        return v
                    og = OG.from_masked(panel, split, str(date), links, times, t,
                                        static_cache["v_cut"].to_numpy(dtype=float),
                                        V[:, t - 1], Qf[:, t - 1], O[:, t - 1],
                                        _rvec(rnow), _rvec(rprev), sgn)
                f = TF.build_window_features(V, Qf, O, E, links, mp, static_cache, ramps,
                                             (t - TF.HIST, t), dummy, t, n_times,
                                             newell_fd=fd_cache if newell else None,
                                             wave_w=None if wave is None else wave[panel],
                                             spatial_w=None if spatial is None else spatial[panel],
                                             time_oblique=time_oblique,
                                             accumulation=accumulation,
                                             dispersion=dispersion,
                                             sgn=sgn, clock=clock, origin=og)
                step = f.step.to_numpy()
                lidx = f.link_id.map(li).to_numpy()
                f["label"] = Q[lidx, t + step].astype(np.int8)
                f["eligible"] = E[lidx, t + step]
                f["_link_idx"] = lidx
                f["_tod_idx"] = (t + step) % n_times
                f["window_id"] = f"{panel}|{date}|{t}"
                f["panel"] = panel
                f["date"] = date
                f["condition"] = w.condition
                if dow:
                    # Ordinary ordinal column, Mon=0..Sun=6. Nothing else moves --
                    # in particular queue_prior and the init_score are untouched,
                    # which is the difference from scripts/74. The ordering keeps
                    # the groups notes/07 B measured contiguous (Wed/Thu = 2,3;
                    # weekend = 5,6), so plain ordinal splits can isolate them.
                    f["dow"] = np.int8(pd.Timestamp(date).dayofweek)
                if include_ineligible:
                    parts.append(f.rename(columns={"eligible": "_eligible"}))
                else:
                    parts.append(f[f.eligible].drop(columns=["eligible"]))
        if progress:
            print(f"  {panel}: {len(wp):,} windows", flush=True)
    d = pd.concat(parts, ignore_index=True)
    for c in d.select_dtypes("float64").columns:
        d[c] = d[c].astype("float32")
    return d


def apply_prior(d: pd.DataFrame, priors: dict[str, np.ndarray]) -> np.ndarray:
    """Look the fold's prior up at each row's (link, time-of-day) index."""
    out = np.empty(len(d), dtype=np.float32)
    for panel, idx in d.groupby("panel", sort=False).indices.items():
        p = priors[panel]
        out[idx] = p[d._link_idx.to_numpy()[idx], d._tod_idx.to_numpy()[idx]]
    return out


def fit_fold(train: pd.DataFrame, valid: pd.DataFrame, feats: list[str],
             scheme: str = "cells"):
    """One condition, one fold. Returns the booster and its best iteration."""
    import lightgbm as lgb
    dtrain = lgb.Dataset(train[feats], train.label, weight=window_weights(train, scheme),
                         init_score=logit(train.queue_prior.to_numpy()))
    dvalid = lgb.Dataset(valid[feats], valid.label, weight=window_weights(valid, scheme),
                         init_score=logit(valid.queue_prior.to_numpy()), reference=dtrain)
    m = lgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                             lgb.log_evaluation(0)])
    return m, int(m.best_iteration or NUM_BOOST_ROUND)


def predict(m, d: pd.DataFrame, feats: list[str]) -> np.ndarray:
    raw = m.predict(d[feats], raw_score=True) + logit(d.queue_prior.to_numpy())
    return 1.0 / (1.0 + np.exp(-raw))
