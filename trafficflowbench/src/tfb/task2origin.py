"""Observations at the forecast origin T itself -- released, legal, and unused.

WHAT THIS ADDS. The shipped Task 2 features stop at the last released history
stamp, T-5: `task2infer` builds everything from `window_history.parquet` and
reads ramp flow at `times[-1]`. The organizer has since stated that a Task 2
forecast may use ANY released observation with timestamp <= T
(discussion/742068). `scripts/139` then measured, on all 160 scored windows,
that the stamp T is partly observable in the released masked layer:

    origin speed / flow / occupancy visible   57.67% public, 59.70% private
    origin speed visible AND eligible         51.18% public, 52.18% private
    origin ramp flow visible                  79.63% public, 79.15% private

and 160/160 windows have at least some of it. None of it reaches the model.

WHY THE MASKED LAYER ON BOTH SIDES. A feature is only usable if training sees
the same missingness serving sees. Serving can only read `mainline_states_masked`
for the scored split, so training reads `mainline_states_masked` for the train
split rather than the complete truth file. Measured on the train split, that
layer's speed visibility is 0.5885 / 0.5903 / 0.5968 on D7_I10_W / D7_I210_E /
D7_I405_N against 0.577-0.597 at serving -- the same mechanism, not a
simulation of it. Handing training a clean T and serving a 58%-visible one is
exactly the train/serving skew that cost 0.031654 of S_queue once before.

The history channels are NOT re-read from the masked layer: at serving they
come from `window_history.parquet`, which the parity audit matched cell for
cell against the truth day. Only the T column is masked-sourced.

Every column is NaN where the cell is unobserved, and `origin_obs_mask` says
which. LightGBM splits on NaN natively, so "not observed" stays distinct from
"observed and low" instead of being imputed into a value the sensor never saw.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from . import release as R

# Role: runs in the reproduction pipeline. task2infer.features_for_split
# (origin_obs=True) calls from_masked for every validation and private
# window. Time rule: the masked layer is read at the stamp T only; the T-5
# values come from the window history.
# COLUMNS: speed-visible mask at T; speed, flow and occupancy at T; margin
# (speed / v_cut) and queued flag at T; changes from T-5 to T; margin of the
# upstream and downstream neighbour at T; corridor shares of visible and of
# queued links; summed attached ramp flow at T with its mask and its change
# from T-5.
COLUMNS = ("origin_obs_mask", "origin_speed_T", "origin_flow_T", "origin_occ_T",
           "origin_margin_T", "origin_queued_T", "origin_dspeed", "origin_dflow",
           "origin_docc", "origin_margin_T_up1", "origin_margin_T_dn1",
           "origin_obs_frac_corridor", "origin_queued_frac_corridor",
           "origin_ramp_T", "origin_ramp_mask_T", "origin_dramp")


@functools.lru_cache(maxsize=64)
def masked_day(panel: str, split: str, date: str) -> tuple:
    """(link_id -> row, timestamp -> col, V, Q, O) from the MASKED layer."""
    # Returns None when the split has no masked file for `date`. Speeds are
    # km/h and flows veh/h; masked cells are NaN. Links keep their order of
    # first appearance; timestamp keys are normalised to tz-aware UTC by _utc.
    stem = f"synthetic_mainline_{date.replace('-', '_')}.parquet"
    paths = [p for r in R.REGIME_ORDER for p in R.masked_files(panel, split, r)
             if p.name == stem]
    if not paths:
        return None
    f = pd.read_parquet(paths[0], columns=["timestamp", "link_id", "speed_kmh",
                                           "flow_vph", "occupancy"])
    f["link_id"] = f.link_id.astype(str)
    links = list(dict.fromkeys(f.link_id))
    times = np.sort(f.timestamp.unique())
    li = {k: i for i, k in enumerate(links)}
    ti = {k: i for i, k in enumerate(times)}
    r = f.link_id.map(li).to_numpy()
    c = f.timestamp.map(ti).to_numpy()
    n = (len(links), len(times))

    def lay(col):
        """Returns column ``col`` as a NaN-filled (link x stamp) float array."""
        g = np.full(n, np.nan)
        g[r, c] = pd.to_numeric(f[col], errors="coerce").to_numpy()
        return g

    return li, {_utc(t): i for i, t in enumerate(times)}, lay("speed_kmh"), \
        lay("flow_vph"), lay("occupancy")


def _utc(t) -> pd.Timestamp:
    """One timestamp key for both codepaths.

    The released files store timestamps as strings ending in 'Z', so parsing
    them yields tz-AWARE UTC. The training path passes those strings straight
    through and matched; serving built its origin stamp from
    ``pd.Timestamp(forecast_origin, tz="UTC")`` and stripped the zone, so every
    lookup missed and all sixteen origin columns silently came back empty --
    0.0000 observed at T against the 0.5767 the audit measured. Normalising
    both sides here is what stops that being a silent feature outage rather
    than an error.
    """
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _neighbour(a: np.ndarray, k: int) -> np.ndarray:
    """Returns ``a`` shifted by ``k`` links, NaN-filled; row i holds row i-k.

    Same behaviour as task2feat.shift_rows.
    """
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def origin_columns(links, order_links, v_T, q_T, o_T, vcut,
                   v_prev, q_prev, o_prev, ramp_T, ramp_prev, sgn: int = 1) -> dict:
    """Assemble the origin-T block from already-extracted vectors."""
    # v_T, q_T, o_T: speed, flow and occupancy at T (NaN where masked).
    # v_prev, q_prev, o_prev: the same at T-5. ramp_T, ramp_prev: summed
    # attached ramp flow at T and T-5. vcut: per-link queue threshold (km/h).
    # sgn: +1 when traffic moves toward increasing milepost. `links` and
    # `order_links` are accepted for the signature only.
    # Returns the COLUMNS as float64 arrays.
    obs = np.isfinite(v_T).astype(np.float32)
    with np.errstate(invalid="ignore"):
        margin = v_T / vcut
        queued = (v_T <= vcut).astype(float)
    queued = np.where(np.isfinite(v_T), queued, np.nan)
    out = {
        "origin_obs_mask": obs,
        "origin_speed_T": v_T,
        "origin_flow_T": q_T,
        "origin_occ_T": o_T,
        "origin_margin_T": margin,
        "origin_queued_T": queued,
        "origin_dspeed": v_T - v_prev,
        "origin_dflow": q_T - q_prev,
        "origin_docc": o_T - o_prev,
        "origin_margin_T_up1": _neighbour(margin, sgn),
        "origin_margin_T_dn1": _neighbour(margin, -sgn),
        "origin_obs_frac_corridor": np.full(len(v_T), float(np.nanmean(obs))),
        "origin_queued_frac_corridor": np.full(
            len(v_T), float(np.nanmean(queued)) if np.isfinite(queued).any() else np.nan),
        "origin_ramp_T": ramp_T,
        "origin_ramp_mask_T": np.isfinite(ramp_T).astype(np.float32),
        "origin_dramp": ramp_T - ramp_prev,
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def from_masked(panel: str, split: str, date: str, links: list[str], times,
                t: int, vcut: np.ndarray, V_hist, Q_hist, O_hist,
                ramp_T: np.ndarray, ramp_prev: np.ndarray, sgn: int = 1) -> dict:
    """Origin-T block for one window, reading the released masked layer at T.

    ``V_hist/Q_hist/O_hist`` are the T-5 vectors from the history source, which
    is the window_history at serving and the truth day in training -- the two
    were matched cell for cell by the parity audit.
    """
    # `times` lists the window's stamps with T at position `t`; the T column
    # is looked up by timestamp in the masked day. Speed, flow and occupancy
    # at T are all NaN when the day file or the stamp is missing.
    md = masked_day(panel, split, str(date))
    n = len(links)
    nan = np.full(n, np.nan)
    if md is None:
        v_T = q_T = o_T = nan
    else:
        li, ti, MV, MQ, MO = md
        col = ti.get(_utc(times[t]))
        if col is None:
            v_T = q_T = o_T = nan
        else:
            rows = np.array([li.get(str(l), -1) for l in links])
            ok = rows >= 0
            v_T, q_T, o_T = nan.copy(), nan.copy(), nan.copy()
            v_T[ok] = MV[rows[ok], col]
            q_T[ok] = MQ[rows[ok], col]
            o_T[ok] = MO[rows[ok], col]
    return origin_columns(links, links, v_T, q_T, o_T, vcut,
                          V_hist, Q_hist, O_hist, ramp_T, ramp_prev, sgn)
