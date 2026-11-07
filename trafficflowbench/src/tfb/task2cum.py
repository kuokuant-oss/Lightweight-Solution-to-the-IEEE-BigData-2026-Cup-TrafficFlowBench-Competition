"""Cumulative-count features on the TIME axis, per link.

Two families, both from the bottleneck-activation literature, both absent from
the current table for a recorded reason that this module re-examines.

A. TIME-AXIS OBLIQUE (Cassidy & Bertini, as carried by notes/16 B)

    Cumulative curves are plotted with an oblique time axis rather than a
    conventional orthogonal one to amplify temporal changes in traffic states,
    and a bottleneck activation time is identified where a sudden decrease in
    the slope of the cumulative curve occurs.

    task2feat._oblique already implements this transform, but its own docstring
    records that it was applied on the SPACE axis instead: "R3 measured the same
    pathology on the SPACE axis instead of the time axis". scripts/73 is that
    space-axis form. The source's own axis is time,
    and it has never been run. It is per-link and dynamic, which is exactly what
    R2 records as missing from the two heaviest onset features.

    It is also a deviation-from-own-trend quantity, so it gives the z-score idea
    of notes/17 A -- which notes/17 D records as having no external source -- a
    form that does have one.

B. PER-LINK ACCUMULATION CHANGE (Lindgren, Bertini, Helbing & Schonhof)

    That paper identifies bottleneck activations by "systematically probing the
    excess vehicle accumulation (spatial) and travel time (temporal) that arose
    between measurement locations".

    task2feat.newell_vectors records why cumulative counts were not used: "each
    link's count here can only be accumulated from the window start, leaving a
    per-link offset -- the vehicles already between the links -- that a 12-step
    window cannot observe, and that offset enters the free-minus-congested
    comparison directly."

    That reasoning is correct for Newell's free-minus-congested comparison,
    which consumes the absolute offset. It does not apply here. The unknown
    offset is a constant in time, so it cancels in every quantity below, all of
    which are differences of the accumulation rather than its level:

        A_i(t) = sum_{s=hs..t} [ Q_{i-1}(s) - Q_i(s) ]

    A_i(hs-1) is unobservable; A_i(t) - A_i(hs-1) is what this computes, and
    every feature is a further difference or extremum of that.

    DECLARED LIMITATION: ramps are available only at the window's last stamp
    (task2feat.ramp_vectors takes ramp_at(rday, times[t-1])), not as a series,
    so the accumulation here is MAINLINE ONLY. Conservation for a link with
    ramps is dA/dt = Q_up + on - off - Q_own; the on/off terms are missing.
    Where a link carries a ramp this is a biased estimate of its accumulation,
    and dist_to_onramp_km is already in the table for the model to condition on.

Both families take their background rate from the window itself, so neither
introduces a tunable quantity -- the same discipline task2feat._oblique states.
"""
from __future__ import annotations

import numpy as np

# Role: task2feat.build_window_features calls these blocks when its matching
# flag is set. The reproduction pipeline serves with dispersion=True, so
# `dispersion` runs there; `time_oblique` and `link_accumulation` are research
# blocks. Arrays are (n_links x n_stamps), links in milepost order, and every
# function reads only the history columns hist_slice = (hs, he), the 12
# stamps T-60..T-5 before the forecast time T.


def _rate(C: np.ndarray, n: int) -> np.ndarray:
    """Background rate per step: the window's own mean, one value per link."""
    return C[:, -1] / max(n, 1)


def time_oblique(Q_: np.ndarray, hist_slice: tuple[int, int]) -> dict[str, np.ndarray]:
    """Per-link cumulative flow over the history, detrended on the time axis.

    Ob_i(t) = C_i(t) - q0_i * (t - hs + 1), where C_i is the link's own
    cumulative count from the window start and q0_i is its own mean flow over
    the same window. Ob starts and ends at 0 by construction, so what it carries
    is the SHAPE: a link whose flow sagged mid-window dips negative, one that
    surged bulges positive. That is the amplification the source is after.
    """
    # Missing flows count as 0 veh/h in the cumulative count. Returns per-link
    # arrays: minimum, maximum and range of the oblique curve, position of its
    # minimum as a fraction of the window, last-step slope, sharpest one-step
    # slope drop and its position, and the change over the last three steps.
    hs, he = hist_slice
    q = np.nan_to_num(Q_[:, hs:he], nan=0.0)
    n = q.shape[1]
    C = np.cumsum(q, axis=1)
    q0 = _rate(C, n)
    steps = np.arange(1, n + 1, dtype=float)[None, :]
    ob = C - q0[:, None] * steps

    slope = q - q0[:, None]                      # d/dt of the oblique curve
    dslope = np.diff(slope, axis=1) if n > 1 else np.zeros((q.shape[0], 1))
    amin = ob.argmin(axis=1).astype(float)

    return {
        "tob_min": ob.min(axis=1),
        "tob_max": ob.max(axis=1),
        "tob_range": ob.max(axis=1) - ob.min(axis=1),
        "tob_argmin_frac": amin / max(n - 1, 1),
        "tob_slope_last": slope[:, -1],
        # the source's activation signature: the sharpest single-step slope drop
        "tob_slope_drop": dslope.min(axis=1),
        "tob_slope_drop_frac": dslope.argmin(axis=1).astype(float) / max(n - 2, 1),
        "tob_last3": ob[:, -1] - ob[:, max(n - 4, 0)],
    }


def link_accumulation(Q_: np.ndarray, hist_slice: tuple[int, int],
                      sgn: int = 1) -> dict[str, np.ndarray]:
    """Mainline accumulation change per link over the history window.

    Links are ordered by milepost and "upstream" is the lower-milepost side, the
    same convention task2feat.ramp_vectors uses for its cumulative inflow. The
    most upstream link has no measured inflow, so its net flow is taken as 0 --
    it is the one link where this feature carries no information, and the model
    sees milepost alongside it.
    """
    # `sgn` = -1 makes the high-milepost end upstream. Missing flows count as
    # 0. The accumulation is a running sum of per-step flow differences in
    # veh/h, not converted to vehicles. Returns per-link arrays: value at the
    # end, maximum and minimum over the window, last-step net inflow, change
    # over the last three steps, share of steps with positive net inflow, and
    # position of the maximum as a fraction of the window.
    hs, he = hist_slice
    q = np.nan_to_num(Q_[:, hs:he], nan=0.0)
    n = q.shape[1]
    if sgn >= 0:
        up = np.vstack([np.zeros((1, n)), q[:-1, :]])  # shift one link downstream
        head = 0
    else:                                               # upstream is the HIGH-milepost end
        up = np.vstack([q[1:, :], np.zeros((1, n))])
        head = -1
    net = up - q                                        # in minus out, per step
    net[head, :] = 0.0                                  # no measured inflow above the chain
    A = np.cumsum(net, axis=1)

    return {
        "acc_end": A[:, -1],
        "acc_max": A.max(axis=1),
        "acc_min": A.min(axis=1),
        "acc_net_last": net[:, -1],
        "acc_last3": A[:, -1] - A[:, max(n - 4, 0)],
        "acc_pos_frac": (net > 0).mean(axis=1),
        "acc_argmax_frac": A.argmax(axis=1).astype(float) / max(n - 1, 1),
    }


# Column names returned by each block, in output order.
COLUMNS_TIME_OBLIQUE = ("tob_min", "tob_max", "tob_range", "tob_argmin_frac",
                        "tob_slope_last", "tob_slope_drop", "tob_slope_drop_frac",
                        "tob_last3")
COLUMNS_ACCUMULATION = ("acc_end", "acc_max", "acc_min", "acc_net_last",
                        "acc_last3", "acc_pos_frac", "acc_argmax_frac")


# ---------------------------------------------------------------------------
# C. SPEED DISPERSION (Abdel-Aty et al.; Wang, Hu, Huang & Chen 2013)
#
# The 41-column table carries min, mean and two differences of speed, and NO
# dispersion measure of anything -- checked, there is no std, var or CV column.
#
# Abdel-Aty et al., matched case-control on freeway loop data: the logarithm of
# the coefficient of variation in speed in the 5-10 minutes before a crash is
# the most significant real-time traffic contributor, and the pairing they
# report is 5-min average occupancy at the UPSTREAM station with the 5-min
# coefficient of variation in speed at the DOWNSTREAM station.
#
# Wang, Hu, Huang & Chen, Traffic Injury Prevention 14(8) 2013, 448 crashes on a
# California freeway, splitting traffic into free flow / congested / back of
# queue / front of queue, verbatim from the abstract:
#
#     in FF, the CSV is positively related to collisions, whereas the SD is not
#     a good indicator for collisions; in CT and BQ, the SD and CSV have
#     positive impacts on the collision likelihood
#
# That distinction is why CV and SD are both emitted here rather than just one:
# the source predicts CV carries the free-flow signal and SD does not, and our
# onset windows are free-flow by construction -- the 12-step history holds no
# queue, which is what makes the six columns of R1 definitionally dead there.
# If CV earns gain at onset and SD does not, the source transferred; if both are
# inert, it did not.
#
# DECLARED TRANSFER GAP: that abstract says in terms that the study "concerns
# collisions only" and that congestion onset is absent from it. The chain from
# it to our target -- CV in free flow predicts crashes, crashes cause
# non-recurrent congestion, and notes/17 A1 measured our failures to be
# concentrated on low-recurrence locations (14.96% against 0.92-4.79%) -- is an
# inference of mine across three sourced links, not a single sourced claim.
# ---------------------------------------------------------------------------

def _cv(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sd, cv) along time. CV is sd/|mean|, 0 where the mean is ~0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.nanmean(x, axis=1)
        sd = np.nanstd(x, axis=1)
        cv = np.where(np.abs(m) > 1e-6, sd / np.maximum(np.abs(m), 1e-6), 0.0)
    return np.nan_to_num(sd), np.nan_to_num(cv)


def dispersion(V: np.ndarray, O_: np.ndarray, Q_: np.ndarray,
               hist_slice: tuple[int, int], sgn: int = 1) -> dict[str, np.ndarray]:
    """Speed / occupancy / flow dispersion over the released history, per link.

    ``shift_rows`` is re-implemented locally rather than imported to keep this
    module free of a task2feat cycle; links are milepost-ascending, so +1 is the
    upstream neighbour, matching task2feat.ramp_vectors' convention.
    """
    # V, O_, Q_: speed, occupancy and flow arrays. Returns per-link standard
    # deviation and coefficient of variation of speed and occupancy, the
    # coefficient of variation of flow, and the neighbour pairings below; a
    # neighbour beyond either end of the chain reads 0.
    hs, he = hist_slice
    v, o, q = V[:, hs:he], O_[:, hs:he], Q_[:, hs:he]
    sd_v, cv_v = _cv(v)
    sd_o, cv_o = _cv(o)
    _, cv_q = _cv(q)

    def _lo(a):                      # value from the lower-milepost neighbour
        """Returns ``a`` shifted one link: row i holds row i-1, row 0 is 0."""
        out = np.zeros_like(a)
        out[1:] = a[:-1]
        return out

    def _hi(a):                      # value from the higher-milepost neighbour
        """Returns ``a`` shifted one link: row i holds row i+1, last row is 0."""
        out = np.zeros_like(a)
        out[:-1] = a[1:]
        return out

    up, dn = (_lo, _hi) if sgn >= 0 else (_hi, _lo)

    mean_o = np.nan_to_num(np.nanmean(o, axis=1))
    return {
        "sd_speed_hist": sd_v,
        "cv_speed_hist": cv_v,
        # Abdel-Aty reports the LOGARITHM of the coefficient of variation. A
        # log_cv_speed_hist column was emitted and measured at exactly 0.0000
        # gain in scripts/99 and 100, which is forced rather than surprising:
        # log1p is strictly monotone in cv, so every split on one has an exact
        # counterpart on the other and a tree can extract nothing new. The
        # source's logarithm matters for a logistic regression's linearity, not
        # for an axis-parallel tree. Dropped; cv_speed_hist carries it.
        "sd_occ_hist": sd_o,
        "cv_occ_hist": cv_o,
        "cv_flow_hist": cv_q,
        # the source's reported pairing: upstream occupancy + downstream CV
        "cv_speed_dn_1": dn(cv_v),
        "occ_mean_up_1": up(mean_o),
        "cv_speed_up_1": up(cv_v),
    }


COLUMNS_DISPERSION = ("sd_speed_hist", "cv_speed_hist",
                      "sd_occ_hist", "cv_occ_hist", "cv_flow_hist",
                      "cv_speed_dn_1", "occ_mean_up_1", "cv_speed_up_1")
