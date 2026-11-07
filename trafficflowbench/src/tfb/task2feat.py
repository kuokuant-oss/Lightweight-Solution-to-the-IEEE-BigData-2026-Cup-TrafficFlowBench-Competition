"""Task 2 feature construction.

Hard constraint: a feature may only use what a participant holds at forecast
time -- the released 12-step window history, the static network, and ramp
flows. Labels come from the horizon, which is withheld on the scored splits.

Feature provenance (doc/plan/baseline-v1.md §5):
  local / trend / static  notes/02 G  (the public competitor's feature set)
  upstream-downstream     notes/01 A  (ASM: information travels along
                                       characteristics, c_cong ~ -15 km/h)
  ramp + distance-to-ramp notes/02 H2 (Kerner arXiv:cond-mat/0502281: the
                                       nucleation rate is a function of the
                                       on-ramp and upstream mainline flows)
  historical queue rate   notes/07 B  (Bolt's init_score = a historical prior)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Role: runs in the reproduction pipeline. task2infer.features_for_split calls
# build_window_features for every validation and private window, and the
# training tables were built with the same functions.
#
# Array layout: V, Q_, O_, E are (n_links x n_stamps) arrays of speed (km/h),
# flow (veh/h), occupancy and score eligibility, links in milepost order.
# Features read only the history columns given by `hist_slice` (the 12 stamps
# T-60..T-5 before the forecast time T), static link data, ramp flows at the
# last history stamp, the time-of-day prior table and, when supplied, the
# task2origin block of observations at T. No horizon column is read.
#
# HIST: history length in 5-minute steps. HOR: forecast horizon in steps.
# DT_H: one step in hours. C_CONG_KMH: assumed speed of a congested front
# moving upstream (km/h). NEIGHBOUR_HOPS: link offsets of the neighbour
# features.
HIST, HOR = 12, 6
DT_H = 5.0 / 60.0
C_CONG_KMH = -15.0          # notes/01 A
NEIGHBOUR_HOPS = (1, 2, 4)


def static_table(panel: str, links: list[str], grid_mod, release_mod) -> pd.DataFrame:
    """Builds the static per-link attribute table of one panel.

    Args:
        panel: Corridor name, e.g. ``D7_I210_W``.
        links: Link ids in the row order of the day arrays (milepost order).
        grid_mod: The ``tfb.grid`` module.
        release_mod: The ``tfb.release`` module.

    Returns:
        DataFrame indexed by link id with lanes (at least 1), free_speed
        (km/h), capacity (veh/h), length_km and v_cut, the queue threshold
        speed 0.60 x free_speed (km/h).
    """
    lo = grid_mod.link_order(panel).set_index("link_id").reindex(links)
    mp = release_mod.network(panel, "links").set_index("link_id").reindex(links)
    out = pd.DataFrame(index=links)
    out["lanes"] = pd.to_numeric(lo.lanes, errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy()
    out["free_speed"] = pd.to_numeric(lo.free_speed_kmh, errors="coerce").to_numpy()
    out["capacity"] = pd.to_numeric(lo.capacity_vph, errors="coerce").to_numpy()
    out["length_km"] = pd.to_numeric(lo.length_km, errors="coerce").to_numpy()
    out["v_cut"] = out.free_speed * 0.60
    return out


def ramp_vectors(panel: str, links: list[str], milepost: np.ndarray,
                 ramp_flow_at_T: dict[str, float], attach: pd.DataFrame,
                 oblique: bool = False, sgn: int = 1) -> dict[str, np.ndarray]:
    """On/off-ramp flow mapped onto the mainline link each ramp attaches to,
    plus the distance to the nearest on-ramp (breakdown nucleates at the merge).

    ``oblique`` adds one extra column, ``onramp_flow_upstream_oblique``, and
    changes nothing else. Default False keeps every existing caller -- scripts
    50/51, the frozen model and the validation_v3 self-check -- bit-identical.
    """
    # `milepost`: per-link mileposts in the order of `links`.
    # `ramp_flow_at_T`: ramp_link_id -> flow (veh/h) at one stamp; callers pass
    # the last history stamp T-5. `attach`: the released ramp_attachment_map
    # (ramp -> nearest mainline link; ramp_type "OR" is an on-ramp, any other
    # type counts as an off-ramp). `sgn`: +1 when traffic moves toward
    # increasing milepost, -1 otherwise.
    # Returns per-link arrays: onramp_flow and offramp_flow (veh/h summed over
    # attached ramps), onramp_flow_upstream (on-ramp flow summed over all links
    # upstream of this one) and dist_to_onramp_km (NaN when the panel has no
    # on-ramp attachment).
    li = {k: i for i, k in enumerate(links)}
    n = len(links)
    on = np.zeros(n)
    off = np.zeros(n)
    on_pos = []
    for row in attach.itertuples(index=False):
        j = li.get(str(row.nearest_mainline_link_id))
        if j is None:
            continue
        f = ramp_flow_at_T.get(str(row.ramp_link_id), np.nan)
        if row.ramp_type == "OR":
            on_pos.append(milepost[j])
            if np.isfinite(f):
                on[j] += f
        elif np.isfinite(f):
            off[j] += f
    if on_pos:
        d = np.abs(milepost[:, None] - np.asarray(on_pos)[None, :]).min(axis=1)
    else:
        d = np.full(n, np.nan)
    # Upstream cumulative on-ramp inflow. Links are milepost-ascending; which
    # end of that chain is upstream is the corridor's travel direction, not a
    # property of the array (task2dir). sgn=+1 keeps the legacy prefix sum.
    if sgn >= 0:
        on_up = np.concatenate([[0.0], np.cumsum(on)[:-1]])
    else:
        on_up = np.concatenate([np.cumsum(on[::-1])[:-1][::-1], [0.0]])
    out = {"onramp_flow": on, "offramp_flow": off, "onramp_flow_upstream": on_up,
           "dist_to_onramp_km": d}
    if oblique:
        out["onramp_flow_upstream_oblique"] = _oblique(on, on_up, milepost, sgn)
    return out


def _oblique(on: np.ndarray, on_up: np.ndarray, milepost: np.ndarray,
             sgn: int = 1) -> np.ndarray:
    """Cumulative upstream inflow with its own background rate removed.

    notes/16 B: a cumulative curve hides its own changes because they are small
    against the mean trend, and the fix (Cassidy & Bertini's oblique axis, as
    quoted by Zheng et al. 2011) is to subtract that trend. R3 measured the same
    pathology on the SPACE axis instead of the time axis --
    corr(milepost, onramp_flow_upstream) is 0.905-0.978 across all eight panels,
    so a tree splits on milepost (15.74% gain) rather than on the inflow (2.67%).

    The background rate is the corridor's own average inflow density for this
    window, total on-ramp flow over corridor length. notes/16 B flags the choice
    of q0 as the risk in this transform: taking it from the window itself leaves
    no free parameter to tune, which is the point.
    """
    span = float(milepost[-1] - milepost[0])
    if not np.isfinite(span) or span <= 0:
        return on_up.copy()
    # q0: background on-ramp inflow per km of corridor in this window.
    q0 = float(np.nansum(on)) / span
    # distance travelled from the corridor's UPSTREAM end, which is milepost[0]
    # only when traffic runs toward increasing milepost.
    travelled = (milepost - milepost[0]) if sgn >= 0 else (milepost[-1] - milepost)
    return on_up - q0 * travelled


def newell_table(panel: str, links: list[str], grid_mod) -> dict[str, np.ndarray]:
    """Per-link triangular-FD parameters, plus the congested wave speed.

    ``fd_parameters.csv`` gives free_speed_kmh, capacity_vph, critical_density
    and k_jam, and qc/kc equals vf exactly (ratio 1.0000 measured on every
    panel), so the generator's FD really is triangular -- Newell's assumption.
    The congested wave speed follows from the same three numbers,
    w = qc / (kj - kc), so nothing is calibrated or tuned here.
    """
    # Returns per-link arrays: vf free-flow speed (km/h), qc capacity (veh/h),
    # kc critical density and kj jam density (veh/km), w congested wave speed
    # (km/h); NaN where the link table lacks a value.
    lo = grid_mod.link_order(panel).set_index("link_id").reindex(links)
    # num(c): column c of the link table as a float array.
    num = lambda c: pd.to_numeric(lo[c], errors="coerce").to_numpy(dtype=float)
    vf, qc, kc, kj = num("free_speed_kmh"), num("capacity_vph"), num("critical_density"), num("k_jam")
    with np.errstate(divide="ignore", invalid="ignore"):
        w = qc / (kj - kc)
    return {"vf": vf, "qc": qc, "kc": kc, "kj": kj, "w": w}


def newell_vectors(fd: dict[str, np.ndarray], milepost: np.ndarray, V: np.ndarray,
                   Q_: np.ndarray, t_last: int, sgn: int = 1) -> dict[str, np.ndarray]:
    """Newell's simplified theory as the demand/supply minimum at each link.

    notes/16 A: the state at a site is "the minimum of what would be expected
    along the free-flow characteristic and the congested characteristic", with
    upstream assumed uncongested and downstream congested -- exactly the onset
    geometry, where the 12-step history carries no queue and the question is
    whether the downstream congested characteristic reaches this link.

    Written in the demand/supply form rather than in cumulative counts. The
    source carries Newell on cumulative vehicle counts, but each link's count
    here can only be accumulated from the window start, leaving a per-link
    offset -- the vehicles already between the links -- that a 12-step window
    cannot observe, and that offset enters the free-minus-congested comparison
    directly. The demand/supply form is the same min over the same two
    characteristics with the same triangular FD, and is free of that nuisance:

        demand   D = min(vf * k_up,  qc)          along the free-flow characteristic
        supply   S = min(qc, w * (kj - k_dn))     along the congested characteristic
        Newell   q = min(D, S)

    D > S is the onset mechanism itself: upstream wants to send more than
    downstream can accept, so a queue forms. Density comes from k = q / v.

    Characteristic sources are located by distance, not by neighbour index.
    At 5-minute steps a free-flow characteristic travels vf * dt ~ 9.5 km
    (about 35 links at the 0.274 km median spacing) and a congested one
    w * dt ~ 1.9 km, so the adjacent link would be a sub-step lag and would not
    be on either characteristic. This mirrors margin_shock's searchsorted.

    notes/16 A item 2 requires the min and the gap to be explicit: an
    axis-parallel GBDT cannot form min(a, b) from a and b given separately.
    """
    # Reads only column `t_last` (the last history stamp, T-5) of V and Q_.
    # `fd` is the newell_table output; `sgn` orients upstream and downstream.
    # Returns per-link newell_demand, newell_supply, their minimum
    # (newell_min) and demand minus supply (newell_gap), all in veh/h.
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(V > 1.0, Q_ / V, np.nan)          # veh/km, guarded at crawl speed
    dt = DT_H
    kk = k[:, t_last]
    up_km = fd["vf"] * dt                               # free-flow reach in one step
    dn_km = fd["w"] * dt                                # congested reach in one step
    n = len(milepost)
    # iu: link one free-flow step upstream; idn: link one congested-wave step
    # downstream. searchsorted needs `milepost` in ascending order.
    iu = np.searchsorted(milepost, milepost - sgn * up_km).clip(0, n - 1)
    idn = np.searchsorted(milepost, milepost + sgn * dn_km).clip(0, n - 1)
    with np.errstate(invalid="ignore"):
        demand = np.minimum(fd["vf"] * kk[iu], fd["qc"])
        supply = np.minimum(fd["qc"], fd["w"] * (fd["kj"] - kk[idn]))
    demand = np.clip(demand, 0.0, None)                 # a flux cannot be negative
    supply = np.clip(supply, 0.0, None)
    return {"newell_demand": demand, "newell_supply": supply,
            "newell_min": np.minimum(demand, supply),
            "newell_gap": demand - supply}


def shift_rows(a: np.ndarray, k: int) -> np.ndarray:
    """Shift along the link axis by k, NaN-filled. k>0 pulls from upstream."""
    # Row i of the result holds row i-k of `a`; with links in milepost order
    # and sgn = +1, k > 0 reads the lower-milepost neighbour.
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def build_window_features(V, Q_, O_, E, links, milepost, static, ramps,
                          hist_slice, queue_prior, origin_col: int, n_times: int,
                          newell_fd: dict[str, np.ndarray] | None = None,
                          wave_w: np.ndarray | None = None,
                          spatial_w: tuple | None = None,
                          time_oblique: bool = False,
                          accumulation: bool = False,
                          dispersion: bool = False,
                          sgn: int = 1,
                          clock: bool = False,
                          origin: dict | None = None) -> pd.DataFrame:
    """One window -> (n_links x HOR) rows of features.

    V/Q_/O_/E are (n_links x n_times) speed / flow / occupancy / eligibility.
    hist_slice is the (start, end) column range of the released history.
    """
    # Other args: `links`/`milepost` give the link order; `static` is the
    # static_table frame; `ramps` the ramp_vectors dict; `queue_prior` a
    # (n_links x n_times) historical queue rate by time-of-day slot;
    # `origin_col` the slot or grid column of the forecast time T; `n_times`
    # the stamps per day (288). Optional blocks: `newell_fd` (newell_table),
    # `wave_w` (per-link wave speed, km/h), `spatial_w` ((W_up, W_dn) from
    # task2spatial), `time_oblique` / `accumulation` / `dispersion` (task2cum),
    # `clock` (explicit time-of-day columns) and `origin` (task2origin block
    # of observations at T). `sgn`: +1 when traffic moves toward increasing
    # milepost, -1 otherwise.
    # Time rule: values come from history columns hs..he-1 (T-60..T-5), static
    # data, the prior table, ramp flows passed in, or the origin-T block.
    # Returns one row per (link, step), step 1..HOR, with the per-link
    # features plus step, step_min, queue_prior, margin_shock and
    # queued_shock for that step.
    hs, he = hist_slice
    t_last = he - 1
    vcut = static["v_cut"].to_numpy()
    # margin: speed divided by the queue threshold v_cut; <= 1 means queued.
    with np.errstate(invalid="ignore"):
        margin_hist = V[:, hs:he] / vcut[:, None]
    v_last = V[:, t_last]
    margin_last = v_last / vcut

    # Per-link features shared by all horizon steps. trend_*_k: change over
    # the last k stamps (k x 5 minutes); *_hist: over the 12-stamp history.
    base = {
        "speed_origin": v_last,
        "flow_origin": Q_[:, t_last],
        "occ_origin": O_[:, t_last],
        "margin_origin": margin_last,
        "min_speed_hist": np.nanmin(V[:, hs:he], axis=1),
        "mean_speed_hist": np.nanmean(V[:, hs:he], axis=1),
        "min_margin_hist": np.nanmin(margin_hist, axis=1),
        "trend_speed_3": V[:, t_last] - V[:, t_last - 3],
        "trend_speed_6": V[:, t_last] - V[:, t_last - 6],
        "trend_occ_3": O_[:, t_last] - O_[:, t_last - 3],
        "trend_flow_3": Q_[:, t_last] - Q_[:, t_last - 3],
        "is_queued_origin": (v_last <= vcut).astype(float),
        "n_queued_hist": (V[:, hs:he] <= vcut[:, None]).sum(axis=1).astype(float),
        "eligible_rate_hist": E[:, hs:he].mean(axis=1),
        "milepost": milepost,
    }
    for c in static.columns:
        base[c] = static[c].to_numpy()
    for c, v in ramps.items():
        base[c] = v
    # spatial: ASM says information travels along characteristics (notes/01 A)
    for k in NEIGHBOUR_HOPS:
        base[f"margin_up_{k}"] = shift_rows(margin_last, sgn * k)
        base[f"margin_dn_{k}"] = shift_rows(margin_last, -sgn * k)
        base[f"trend_up_{k}"] = shift_rows(base["trend_speed_3"], sgn * k)
    base["queued_up_1"] = shift_rows(base["is_queued_origin"], sgn)
    base["queued_dn_1"] = shift_rows(base["is_queued_origin"], -sgn)
    if newell_fd is not None:
        base.update(newell_vectors(newell_fd, milepost, V, Q_, t_last, sgn))
    if spatial_w is not None:
        from . import task2spatial as SP
        base.update(SP.aggregate(spatial_w[0], spatial_w[1], margin_last,
                                 base["trend_speed_3"], base["is_queued_origin"]))
    if time_oblique or accumulation or dispersion:
        from . import task2cum as CU
        if time_oblique:
            base.update(CU.time_oblique(Q_, hist_slice))
        if accumulation:
            base.update(CU.link_accumulation(Q_, hist_slice, sgn))
        if dispersion:
            base.update(CU.dispersion(V, O_, Q_, hist_slice, sgn))
    if origin is not None:
        # Observations at T itself. Legal per discussion/742068, released, and
        # absent from every shipped model. NaN where the masked layer does not
        # show the cell; origin_obs_mask carries that explicitly.
        base.update(origin)
    # Corridor-wide context repeated on every link: number of queued links and
    # total mainline flow at the last history stamp.
    base["n_queued_corridor"] = np.full(len(links), float((v_last <= vcut).sum()))
    base["mainline_flow_corridor"] = np.full(len(links), float(np.nansum(Q_[:, t_last])))

    # One copy of the per-link frame per horizon step, plus step columns.
    frame = pd.DataFrame(base, index=pd.Index(links, name="link_id"))
    rows = []
    for step in range(1, HOR + 1):
        f = frame.copy()
        f["step"] = step
        f["step_min"] = step * 5
        if clock:
            # C1: step_min is a HORIZON OFFSET, not a clock time. The 50-column
            # table carries no absolute time at all, so queue_prior -- indexed
            # by (link, time-of-day) -- is the only column that knows when the
            # window is, and any "remove the prior" arm silently removes the
            # clock as well. These three columns give every arm the same
            # explicit clock so that comparison isolates the prior.
            mins = 1440.0 / n_times
            tod = ((origin_col + step) % n_times) * mins
            f["tod_min"] = tod
            f["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
            f["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
            o_tod = (origin_col % n_times) * mins
            f["origin_tod_min"] = o_tod
        # Out-of-day prior for the cell being predicted: the (link, time-of-day)
        # queue rate over OTHER days. notes/07 B -- Bolt computes its target
        # encoding out of sample and falls back for sparse keys; the in-day rate
        # would contain the horizon, i.e. the answer.
        f["queue_prior"] = queue_prior[:, (origin_col + step) % n_times]
        # shockwave arrival: a congested front travels upstream at c_cong.
        # After `step` intervals it has moved |c_cong| * step * dt km upstream,
        # so the informative neighbour is that far DOWNSTREAM of this link.
        # Per-link when a data-derived wave speed is supplied (notes/16 I),
        # otherwise the hardcoded ASM constant. Geometry is unchanged: a
        # congested front travels upstream, so the informative neighbour sits
        # that far DOWNSTREAM of this link.
        w_kmh = abs(C_CONG_KMH) if wave_w is None else wave_w
        lag_km = w_kmh * step * DT_H
        idx = np.searchsorted(milepost, milepost + sgn * lag_km).clip(0, len(links) - 1)
        f["margin_shock"] = margin_last[idx]
        f["queued_shock"] = (v_last <= vcut).astype(float)[idx]
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)
