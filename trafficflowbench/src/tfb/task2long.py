"""Task 2 long-history block: the legal input the pipeline never read.

The organizer ruled on 2026-09-20 (doc/reference/notes/22) that for a window with
forecast origin T a submission may use ANY released file with a timestamp at or
before T -- earlier times on day D, earlier days in the same split, and the whole
training split. task2infer.features_for_split reads window_history.parquet, i.e.
12 stamps; the same day holds a median of 9,931 observed cells at or before T
against the 1,063 the 60-minute box uses (1/9.3).

PARITY IS THE WHOLE DESIGN HERE.
Before the 60-minute window the only layer that exists on a scored split is
mainline_states_masked -- the Task 1 target cells are blanked there. scripts/50
builds its window features from the TRUTH layer, which on a scored split does not
exist. So this block reads mainline_states_masked on BOTH sides, and both callers
go through long_features() so there is one implementation, as
doc/plan/task2-parity.md requires (its two mismatches cost 0.031654 and 0.012346
of S_queue).

Every column is computed from cells strictly at or before T. The horizon and the
hour after it are already blanked in the released files (MASK_SPEC), so a
<= T filter cannot reach the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import release as R

# Role: task2infer.features_for_split adds this block only with long=True;
# the reproduction pipeline serves with long=False, so this is a research
# block.
# COLUMNS (lh_ = long history), per link, from masked-layer cells at or before
# T: mean flow over the last 3 h and over the day so far, mean and maximum
# occupancy over 3 h, lowest speed so far today and its ratio to v_cut, share
# of today's observed stamps that were queued, steps since the link was last
# queued, 3-hour speed slope (km/h per step), observed stamps today, the
# corridor mean of the queued share, and the recent-day queue rate at this
# time of day with the number of days behind it.
# STEPS_3H: 3 hours in 5-minute steps. RECENT_DAYS: earlier in-split days the
# recent prior looks back over. TOD_HALFWIDTH: slots (5 min each) on either
# side of T's time of day.
COLUMNS = ("lh_flow_mean_3h", "lh_flow_mean_day", "lh_occ_mean_3h", "lh_occ_max_3h",
           "lh_minspeed_day", "lh_margin_day", "lh_queued_frac_day", "lh_since_queue",
           "lh_speed_slope_3h", "lh_nobs_day", "lh_corr_queued_frac", "lh_recent_prior",
           "lh_recent_days")
STEPS_3H = 36          # 5-minute steps
RECENT_DAYS = 14       # in-split lookback for the current-realization prior
TOD_HALFWIDTH = 3      # +/- 15 minutes, so the recent prior has support to estimate on
# LRU cache of masked day arrays, keyed by (panel, split, date, link count).
_DAY: "OrderedDict" = __import__("collections").OrderedDict()
_DAY_MAX = 40          # days held; the recent prior only ever looks back RECENT_DAYS


def _masked_path(panel: str, split: str, date: str):
    """Returns the masked mainline file for ``date``, or None if absent.

    Searches the regime folders in R.REGIME_ORDER (R1, R2, R3) in order.
    """
    want = "synthetic_mainline_" + date.replace("-", "_") + ".parquet"
    for rg in R.REGIME_ORDER:
        for p in R.masked_files(panel, split, rg):
            if p.name == want:
                return p
    return None


def day_arrays(panel: str, split: str, date: str, links: list[str]):
    """(speed, flow, occ, times) on the MASKED layer, aligned to ``links``."""
    # Arrays are (len(links) x n_stamps), rows in the order of `links`, stamps
    # as tz-aware UTC; masked or absent cells are NaN. Returns None when the
    # masked file is missing. Results are kept in the _DAY cache.
    key = (panel, split, date, len(links))
    if key in _DAY:
        _DAY.move_to_end(key)
        return _DAY[key]
    while len(_DAY) >= _DAY_MAX:
        _DAY.popitem(last=False)
    p = _masked_path(panel, split, date)
    if p is None:
        _DAY[key] = None
        return None
    f = pd.read_parquet(p, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy"])
    f["link_id"] = f.link_id.astype(str)
    f["timestamp"] = pd.to_datetime(f.timestamp, utc=True)
    times = np.sort(f.timestamp.unique())
    ti = {t: i for i, t in enumerate(times)}
    li = {l: i for i, l in enumerate(links)}
    r = f.link_id.map(li).to_numpy(); c = f.timestamp.map(ti).to_numpy()
    ok = ~pd.isna(r)
    r = r[ok].astype(int); c = c[ok].astype(int)
    out = []
    for col in ("speed_kmh", "flow_vph", "occupancy"):
        g = np.full((len(links), len(times)), np.nan)
        g[r, c] = pd.to_numeric(f[col], errors="coerce").to_numpy()[ok]
        out.append(g)
    _DAY[key] = (out[0], out[1], out[2], times)
    return _DAY[key]


def _tod_of(ts) -> int:
    """Returns the 5-minute time-of-day slot (0..287) of ``ts``."""
    t = pd.Timestamp(ts)
    return t.hour * 12 + t.minute // 5


def _recent_prior(panel: str, split: str, date: str, links: list[str], vcut: np.ndarray,
                  tod: int, dates_before: list[str]) -> tuple[np.ndarray, float]:
    """Queue rate at this time-of-day over the most recent in-split days BEFORE `date`.

    This is the ONLY estimate of "where does a queue start here, now" that comes
    from the realization actually being scored. The shipped queue_prior is built
    from 2030-06..10; the organizer regenerated the validation and private months
    from new random seeds (notes/22 same announcement), and an ablation measured
    onset skill as 41.6% location memory (doc/plan/baseline-v3.md Z11).
    """
    # Reads masked-layer speeds of up to RECENT_DAYS earlier dates of the same
    # split, within TOD_HALFWIDTH slots of `tod`. Returns (per-link share of
    # observed cells that were queued, NaN where fewer than 3 cells were
    # observed; number of days considered).
    use = [d for d in dates_before if d < date][-RECENT_DAYS:]
    if not use:
        return np.full(len(links), np.nan), 0.0
    num = np.zeros(len(links)); den = np.zeros(len(links))
    for d in use:
        a = day_arrays(panel, split, d, links)
        if a is None:
            continue
        V, _, _, times = a
        # match on real time-of-day, not the column index: a day with a missing
        # stamp would otherwise shift every later column by one slot
        tods = np.array([_tod_of(t) for t in times])
        sel = np.abs(tods - tod) <= TOD_HALFWIDTH
        if not sel.any():
            continue
        blk = V[:, sel]
        seen = np.isfinite(blk)
        num += (seen & (blk <= vcut[:, None])).sum(axis=1).astype(float)
        den += seen.sum(axis=1).astype(float)
    return np.where(den >= 3, num / np.maximum(den, 1.0), np.nan), float(len(use))


def long_features(panel: str, split: str, date: str, links: list[str], vcut: np.ndarray,
                  origin_ts, dates_before: list[str]) -> pd.DataFrame:
    """One row per link, from cells at or before ``origin_ts``.

    Keyed on the TIMESTAMP, not a column index: the caller's column index comes
    from the truth layer (scripts/50) or from window_history (serving), and this
    block reads the masked layer. Resolving T here keeps the three aligned.
    """
    # `vcut`: per-link queue threshold (km/h) aligned with `links`.
    # `origin_ts`: the forecast time T. `dates_before`: the split's dates; only
    # those before `date` feed the recent prior.
    # Returns a frame indexed by link with the COLUMNS above; all NaN when the
    # day's masked file is missing.
    n = len(links)
    a = day_arrays(panel, split, date, links)
    if a is None:
        return pd.DataFrame({c: np.full(n, np.nan) for c in COLUMNS}, index=links)
    V, Q_, O_, times = a
    T = pd.Timestamp(origin_ts)
    if T.tzinfo is None:
        T = T.tz_localize("UTC")
    # `times` is a tz-aware datetime64 array; compare in the same frame
    hi = int(np.searchsorted(pd.DatetimeIndex(times), T, side="right"))   # inclusive of T
    lo3 = max(0, hi - STEPS_3H)
    with np.errstate(invalid="ignore", divide="ignore"):
        Vd, Qd, Od = V[:, :hi], Q_[:, :hi], O_[:, :hi]
        V3, O3 = V[:, lo3:hi], O_[:, lo3:hi]
        queued = Vd <= vcut[:, None]
        seen = np.isfinite(Vd)
        nobs = seen.sum(axis=1).astype(float)
        cols = {
            "lh_flow_mean_3h": np.nanmean(Q_[:, lo3:hi], axis=1),
            "lh_flow_mean_day": np.nanmean(Qd, axis=1),
            "lh_occ_mean_3h": np.nanmean(O3, axis=1),
            "lh_occ_max_3h": np.nanmax(O3, axis=1) if O3.size else np.full(n, np.nan),
            "lh_minspeed_day": np.nanmin(Vd, axis=1) if Vd.size else np.full(n, np.nan),
            "lh_nobs_day": nobs,
        }
        cols["lh_margin_day"] = cols["lh_minspeed_day"] / np.maximum(vcut, 1e-6)
        cols["lh_queued_frac_day"] = np.where(nobs > 0, (queued & seen).sum(axis=1) / np.maximum(nobs, 1), np.nan)
        # steps since this link was last queued today (hi = never seen queued)
        last = np.full(n, np.nan)
        for i in range(n):
            w = np.where(queued[i] & seen[i])[0]
            last[i] = hi - 1 - w[-1] if w.size else hi
        cols["lh_since_queue"] = last
        # 3h speed slope per link
        m = np.isfinite(V3)
        x = np.arange(V3.shape[1], dtype=float)
        slope = np.full(n, np.nan)
        for i in range(n):
            if m[i].sum() >= 4:
                slope[i] = np.polyfit(x[m[i]], V3[i][m[i]], 1)[0]
        cols["lh_speed_slope_3h"] = slope
        cols["lh_corr_queued_frac"] = np.full(n, float(np.nanmean(cols["lh_queued_frac_day"])))
        rp, nd = _recent_prior(panel, split, date, links, vcut, _tod_of(T), dates_before)
        cols["lh_recent_prior"] = rp
        cols["lh_recent_days"] = np.full(n, nd)
    return pd.DataFrame({c: np.asarray(cols[c], dtype=float) for c in COLUMNS}, index=links)
