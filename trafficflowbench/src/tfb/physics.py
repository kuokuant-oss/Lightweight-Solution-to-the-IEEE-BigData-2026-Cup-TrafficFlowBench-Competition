"""Local S_FD / S_LWR ruler for Task 3, scored on a Task 1 answer.

WHY THIS CAN EXIST AT ALL
-------------------------
scripts/38 and notes/03 both record that S_LWR "cannot be measured here",
because ``score_task3.load_boundary_flux`` needs organizer-held boundary flows.
That is true of the official evaluator and false of the quantity it computes.
The official docstring states the flux is

    "projected onto the released observations, so a participant who reproduces
     what was published satisfies the conservation identity exactly"

and that the topology fallback "put even the exact truth at S_LWR 0.0". Both
sentences say the same thing: on the organizer's evaluator the released truth
has residual zero, i.e.

    rhs(t) == dN_true(t)

so the boundary file is never needed to evaluate a *submission*. Substituting,

    E_LWR = sum|dN_pred - dN_true| / sum|dN_true|

which the train split can compute exactly, because train is the one split that
ships both ``mainline_states`` (truth) and ``mainline_states_masked``.

Everything else here is copied from score_task3.score_panel: the mixed field
(truth at non-target cells, the submission at target cells), the per-lane
triangular FD, the observability filter, the Mode A ramp-validity filter, the
forward difference and the 5.1/60 h transition bound.
"""
#
# Role in the package: link_params and scoreable_mask are read by programs of
# the Task 1 reproduction pipeline (for example the C1 rebuild in
# build/review_20260924_ctm/rl/serve_feats.py and the hole models). ramp_index
# and ramp_valid_grid are read by state audit and evaluation scripts. s_fd
# and lwr_parts form the local Task 3 ruler; no program in this package
# calls them.
#
# Terms: S_FD scores how closely the per-lane flow follows the triangular
# fundamental diagram (FD) at the density k = q / v. S_LWR scores vehicle
# conservation: N = k * length is the number of vehicles on a link, and its
# change between consecutive 5-minute stamps (dN) is compared with the true
# change. "Mode A" is the official rule that keeps a transition only where
# every ramp attached to the link has a valid released cell. score_task3 is
# the organizers' Task 3 scorer; scripts/38, notes/03 and the audits cited
# below are development records that are not part of this package.
from __future__ import annotations

import numpy as np
import pandas as pd

from . import release as R

# DT_MAX_H: score_task3's transition bound in hours (two stamps at most 5.1
# minutes apart form one transition). The functions here difference adjacent
# columns of a 5-minute grid and do not read it.
DT_MAX_H = 5.1 / 60.0
# score_task3's empty-submission guard: S_FD is 0 when more than EMPTY_SHARE
# of the submitted cells carry flow below EMPTY_FLOOR_VPH (veh/h).
EMPTY_FLOOR_VPH = 50.0
EMPTY_SHARE = 0.20


def link_params(panel: str, links: list[str]) -> dict[str, np.ndarray]:
    """network_parameters(): fd_parameters authoritative, k_crit recomputed."""
    # Args: panel id and the links in grid-row order. Returns a dict of arrays
    # aligned with `links`: length_km, free_speed_kmh (km/h), capacity_vph
    # (veh/h, all lanes), lanes, critical_density (veh/km, capacity / free
    # speed) and k_jam (veh/km). Missing values fall back to 1 km, 100 km/h,
    # 1800 veh/h per lane and 1 lane; fd_parameters.csv overrides links.csv
    # where present; k_jam defaults to 100 veh/km per lane and is kept at
    # least 5% above the critical density.
    pdir = R.RELEASE / "corridors" / panel
    n = pd.read_csv(pdir / "network" / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id")
    out = n.set_index("link_id")[["length_km", "free_speed_kmh", "capacity_vph", "lanes"]].copy()
    for c, d in (("length_km", 1.0), ("free_speed_kmh", 100.0)):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(d)
    out["lanes"] = pd.to_numeric(out["lanes"], errors="coerce").fillna(1.0).clip(lower=1.0)
    out["capacity_vph"] = pd.to_numeric(out["capacity_vph"], errors="coerce").fillna(1800.0 * out.lanes)
    fd_p = pdir / "network" / "fd_parameters.csv"
    if fd_p.exists():
        f = pd.read_csv(fd_p, dtype={"link_id": str})
        have = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in f.columns]
        m = f.groupby("link_id")[have].mean()
        for c in ("free_speed_kmh", "capacity_vph", "lanes"):
            if c in m:
                out[c] = m[c].reindex(out.index).fillna(out[c])
        out["k_jam"] = m["k_jam"].reindex(out.index) if "k_jam" in m else np.nan
    else:
        out["k_jam"] = np.nan
    out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    out["k_jam"] = out.k_jam.fillna(100.0 * out.lanes.clip(lower=1.0))
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    return {c: out[c].reindex(links).to_numpy(dtype=float) for c in
            ("length_km", "free_speed_kmh", "capacity_vph", "lanes", "critical_density", "k_jam")}


def scoreable_mask(panel: str, links: list[str], split: str, times) -> np.ndarray:
    """Which links enter S_LWR: observability filter + Mode A ramp validity.

    score_task3 drops a link whose topological neighbours are absent from the
    submitted field, and (modes A/B) a transition whose attached ramp cell is
    not valid. Both depend only on the release, never on the prediction, so
    they are computed once per panel.
    """
    # Only the observability filter is computed here: a link is kept when all
    # its incoming and outgoing neighbours that appear in
    # lwr_mainline_topology.csv are among `links`; a link absent from that file
    # is dropped. Every link is kept when the file is missing. `split` and
    # `times` are not read; Mode A ramp validity comes from ramp_index and
    # ramp_valid_grid. Returns a boolean vector aligned with `links`.
    pdir = R.RELEASE / "corridors" / panel
    topo_p = pdir / "network" / "lwr_mainline_topology.csv"
    keep = np.ones(len(links), dtype=bool)
    idx = {l: i for i, l in enumerate(links)}
    if topo_p.exists():
        topo = pd.read_csv(topo_p, dtype=str)
        topo_ids = set(topo.link_id.astype(str))
        have = set(links)
        obs = {}
        for row in topo.itertuples(index=False):
            inc = {v.strip() for v in str(getattr(row, "incoming_link_ids", "")).split(";") if v.strip()}
            outg = {v.strip() for v in str(getattr(row, "outgoing_link_ids", "")).split(";") if v.strip()}
            obs[str(row.link_id)] = (inc & topo_ids).issubset(have) and (outg & topo_ids).issubset(have)
        for l, i in idx.items():
            keep[i] = obs.get(l, False)
    return keep


def ramp_index(panel: str, split: str, official: bool = True) -> tuple[set, set]:
    """Read the ramp layer ONCE per panel.

    ``official=True`` returns ({"on": set, "off": set}, {"on": set, "off": set})
    -- the two ramp TYPES kept apart, which is what score_task3 does:

        state.on_ramp_valid  |= ~link_id.isin(on_attached)
        state.off_ramp_valid |= ~link_id.isin(off_attached)
        x = x[(x.on_ramp_valid >= 0.5) & (x.off_ramp_valid >= 0.5)]

    A link carrying both an on-ramp and an off-ramp needs BOTH released cells
    valid. Pooling the two types, as this function did, let one observed
    on-ramp vouch for an unobserved off-ramp on the same link, so the local
    ruler scored transitions the official evaluator drops. Measured over
    10 panels x 2 days (audit 2026-09-21 S7): 604,918 transitions here against
    482,192 by the published rule -- 137,524 extra and 14,798 missing.

    ``official=False`` reproduces the pooled behaviour that the v3 selection
    numbers were computed with.
    """
    # Returns (attached, valid). attached: the links that carry a ramp of each
    # type per lwr_mainline_topology.csv. valid: (timestamp string, mainline
    # link) pairs with a valid released ramp cell, mapped to the mainline
    # through ramp_attachment_map.csv. With official=False both are pooled:
    # one attached set and one valid set over both ramp types.
    pdir = R.RELEASE / "corridors" / panel
    TYPES = {"on": "on_ramp_link_ids", "off": "off_ramp_link_ids"}
    att = {k: set() for k in TYPES}
    topo_p = pdir / "network" / "lwr_mainline_topology.csv"
    if topo_p.exists():
        topo = pd.read_csv(topo_p, dtype=str)
        for k, c in TYPES.items():
            if c in topo.columns:
                att[k] = set(topo.loc[topo[c].fillna("").astype(str).str.len() > 0,
                                      "link_id"].astype(str))
    pooled = att["on"] | att["off"]
    amap_p = pdir / "network" / "ramp_attachment_map.csv"
    root = pdir / split / "ramp_states"
    if not amap_p.exists() or not root.exists():
        return (att, {k: set() for k in TYPES}) if official else (pooled, set())
    amap = pd.read_csv(amap_p, dtype=str).rename(columns={"nearest_mainline_link_id": "link_id"})
    pieces = []
    for p_ in sorted(x for x in root.glob("**/*.parquet") if not x.name.startswith("._")):
        f = pd.read_parquet(p_, columns=["timestamp", "ramp_link_id", "flow_vph",
                                         "pct_observed", "is_score_eligible"])
        f["ramp_link_id"] = f.ramp_link_id.astype(str)
        # A released ramp cell is valid when it is score-eligible, at least 75%
        # observed and carries a flow value.
        f = f[f.is_score_eligible.astype(bool)
              & (pd.to_numeric(f.pct_observed, errors="coerce") >= 75)
              & pd.to_numeric(f.flow_vph, errors="coerce").notna()]
        if not f.empty:
            pieces.append(f.merge(amap[["ramp_link_id", "link_id", "ramp_type"]],
                                  on="ramp_link_id", how="inner")
                          [["timestamp", "link_id", "ramp_type"]])
    if not pieces:
        return (att, {k: set() for k in TYPES}) if official else (pooled, set())
    v = pd.concat(pieces, ignore_index=True)
    if not official:
        return pooled, set(zip(v.timestamp.astype(str), v.link_id.astype(str)))
    # ramp_attachment_map marks on-ramps "OR"; anything else attaches as an off-ramp
    is_on = v.ramp_type.astype(str).str.upper().eq("OR")
    valid = {"on": set(zip(v.loc[is_on].timestamp.astype(str), v.loc[is_on].link_id.astype(str))),
             "off": set(zip(v.loc[~is_on].timestamp.astype(str), v.loc[~is_on].link_id.astype(str)))}
    return att, valid


def ramp_valid_grid(links: list[str], times, attached, valid) -> np.ndarray:
    """(link x time) Mode A validity for one day, from the per-panel index.

    With the official per-type index a cell is valid only when EVERY ramp type
    attached to that link has a valid released cell at that stamp -- the
    ``on_ramp_valid & off_ramp_valid`` conjunction of score_panel. The pooled
    (set, set) form is still accepted so the pre-fix numbers reproduce.
    """
    # Args: links in grid-row order, the day's timestamps and the
    # (attached, valid) pair from ramp_index. Timestamps are formatted as
    # YYYY-MM-DDTHH:MM:SSZ strings to look up the ramp index keys. Returns a
    # boolean (link x time) grid; links without an attached ramp are valid
    # at every stamp.
    ok = np.ones((len(links), len(times)), dtype=bool)
    ts = [pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ") for t in times]
    if isinstance(attached, dict):
        for kind in ("on", "off"):
            a, v = attached.get(kind, set()), valid.get(kind, set())
            for i, l in enumerate(links):
                if l in a:
                    ok[i, :] &= np.fromiter(((t, l) in v for t in ts), bool, len(ts))
        return ok
    for i, l in enumerate(links):
        if l in attached:
            ok[i, :] = [(t, l) in valid for t in ts]
    return ok


def s_fd(v: np.ndarray, q: np.ndarray, par: dict, submitted: np.ndarray) -> float:
    """Returns the local S_FD of one (link x time) grid.

    Converts every cell to per-lane flow and density (k = q / v, speed
    floored at 1 km/h) and compares the flow with the per-lane triangular FD
    at that density: vf * k up to the critical density, then a straight line
    from capacity down to 0 at jam density. The score is
    1 - sum|q - q_FD| / sum|q| over the finite cells, floored at 0.

    Args:
        v: Speed grid (link x time), km/h.
        q: Flow grid (link x time), veh/h over all lanes.
        par: Link parameters from link_params, aligned with the grid rows.
        submitted: Boolean (link x time) grid of the cells the submission
            fills; read only by the empty-submission guard.

    Returns:
        S_FD in [0, 1]; 0 when more than EMPTY_SHARE of the submitted cells
        carry flow below EMPTY_FLOOR_VPH.
    """
    lanes = np.maximum(par["lanes"], 1.0)[:, None]
    k = q / np.clip(v, 1.0, None)
    q_l, k_l = q / lanes, k / lanes
    cap_l = par["capacity_vph"][:, None] / lanes
    kc_l = par["critical_density"][:, None] / lanes
    kj_l = par["k_jam"][:, None] / lanes
    vf = par["free_speed_kmh"][:, None]
    q_fd = np.where(k_l <= kc_l, vf * k_l, (cap_l / np.maximum(kj_l - kc_l, 1e-9)) * np.maximum(kj_l - k_l, 0))
    ok = np.isfinite(q_l) & np.isfinite(q_fd)
    val = float(max(0.0, 1.0 - np.abs(q_l[ok] - q_fd[ok]).sum() / (np.abs(q_l[ok]).sum() + 1e-9)))
    if submitted.any() and float((q[submitted] < EMPTY_FLOOR_VPH).mean()) > EMPTY_SHARE:
        return 0.0
    return val


def lwr_parts(v: np.ndarray, q: np.ndarray, vt: np.ndarray, qt: np.ndarray,
              par: dict, keep: np.ndarray, rvalid: np.ndarray,
              official: bool = True) -> tuple[float, float]:
    """Return (numerator, denominator) of E_LWR for one day-grid.

    rhs == dN_true, so the residual is |dN_pred - dN_true| and the denominator
    is sum|dN_true| over exactly the same transitions.

    RAMP VALIDITY IS READ AT THE TRANSITION START. score_panel forms
    dN = N(t+1) - N(t) with a shift and then filters ROWS on
    on_ramp_valid & off_ramp_valid, so the surviving transition is the one
    whose START stamp is valid; the stamp at t+1 is not consulted. Requiring
    both, as this did, drops transitions the official evaluator keeps.
    ``official=False`` restores the both-ends rule.
    """
    # v, q: the field being scored (in score_task3, the mixed field: truth at
    # non-target cells and the submission at target cells); vt, qt: the true
    # field. All are (link x time) grids in km/h and veh/h. par: link_params
    # output; keep: scoreable_mask output; rvalid: ramp_valid_grid output.
    # N = (q / v) * length is the vehicle count on each link; speeds are
    # floored at 1 km/h before dividing.
    L = par["length_km"][:, None]
    N = (q / np.clip(v, 1.0, None)) * L
    Nt = (qt / np.clip(vt, 1.0, None)) * L
    finite = np.isfinite(N) & np.isfinite(Nt) & keep[:, None]
    if official:
        m = finite[:, 1:] & finite[:, :-1] & rvalid[:, :-1]
    else:
        ok = finite & rvalid
        m = ok[:, 1:] & ok[:, :-1]
    dN = np.where(m, N[:, 1:] - N[:, :-1], np.nan)
    dNt = np.where(m, Nt[:, 1:] - Nt[:, :-1], np.nan)
    return float(np.nansum(np.abs(dN - dNt))), float(np.nansum(np.abs(dNt)))
