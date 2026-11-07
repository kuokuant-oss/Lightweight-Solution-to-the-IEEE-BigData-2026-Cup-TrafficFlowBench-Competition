"""Congested wave speed estimated from data, per (panel, link).

notes/16 I. ``src/tfb/task2feat.C_CONG_KMH = -15.0`` is a hardcoded ASM
convention (notes/01 A) and three measurements disagree with it: the triangular
FD gives w = qc/(kj-kc) at a median of 23-27 km/h, regressing each link's first
queued time on milepost gives 7.20 upstream and 11.04 downstream, and the
margin_shock built from the constant has an AUC that changes sign across panels
(0.26 to 0.66). Three different numbers means no single global constant stands
up, whatever value it takes.

Chen & Rakha (arXiv:1911.07395) extract "activation and deactivation points,
shockwave speeds and traffic delay" automatically from exactly the structure
already built here -- "a binary matrix, where 1 represents a congestion roadway
section and 0 represents a free flow section", i.e. Q = (V <= v_cut) & E.

notes/16 I item 2 sets the transformation: estimate a static per-(panel, link)
lookup from the TRAINING days rather than anything inside a window's own
history. That leaks nothing and is isomorphic to queue_prior, which is
accumulated the same way and carries the same degeneracy rule.

Method, per panel and per source date set:

  * label connected congested regions in the (link x time) binary matrix;
  * inside a region, take each member link's first congested time and regress
    that time on milepost. A front moving upstream reaches lower mileposts
    later, so the slope dt/dx is negative and the speed is 1 / |slope|;
  * a region needs at least MIN_LINKS links to admit a slope with a residual;
  * a region whose front crosses it faster than its own resolution allows is
    CENSORED at that resolution, not discarded. This matters: at 5-minute steps
    a fast front lights up many links inside one step and so produces a slope of
    exactly zero. Dropping those would throw away precisely the fast waves and
    keep the slow ones, biasing every estimate downward -- the first version of
    this module did that and returned panel medians of 4.0-14.4 km/h, with one
    panel sitting on the 3.29 km/h resolution floor itself. A zero slope is
    evidence of "speed >= span / dt", so span / dt is substituted as a
    conservative lower bound and the region stays in the median;
  * per link, take the median speed over the regions containing it, requiring
    at least PRIOR_MIN_DEN regions, else the panel median, else the FD value.

The fallback chain mirrors queue_prior's, and PRIOR_MIN_DEN is reused rather
than a new threshold introduced.

Resolution bound, carried from notes/16 I: adjacent links sit a median 0.274 km
apart and steps are 5 minutes, so the smallest resolvable speed is 3.29 km/h
and resolving 24 km/h needs about 7 link intervals. Estimates are reported with
their spread rather than treated as exact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import grid, release as R, score2, task2build as TB, task2feat as TF

MIN_LINKS = 3               # fewest links that admit a slope plus a residual
DT_H = TF.DT_H


def fd_wave(panel: str, links: list[str]) -> np.ndarray:
    """w = qc / (kj - kc) from the triangular FD, the free prior of notes/16 I item 3."""
    lo = grid.link_order(panel).set_index("link_id").reindex(links)
    num = lambda c: pd.to_numeric(lo[c], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = num("capacity_vph") / (num("k_jam") - num("critical_density"))
    return w


def panel_wave(panel: str, source_dates: set[str]) -> tuple[np.ndarray, dict]:
    """Per-link congested wave speed in km/h, plus a diagnostic dict."""
    from scipy import ndimage
    cut = score2.v_cut(panel)
    from .task2pool import date_of
    per_link: list[list[float]] = []
    links = mp = None
    n_regions = n_used = n_censored = 0
    for path in R.truth_files(panel, "train"):
        if date_of(path) not in source_dates:
            continue
        lk, mpost, V, Qf, O, E, Q, times = TB.day_arrays(panel, path, cut)
        if links is None:
            links, mp = lk, mpost
            per_link = [[] for _ in links]
        lab, n = ndimage.label(Q)
        n_regions += n
        for k in range(1, n + 1):
            r, c = np.nonzero(lab == k)
            u = np.unique(r)
            if len(u) < MIN_LINKS:
                continue
            first = np.array([c[r == i].min() for i in u], dtype=float) * DT_H
            x = mp[u]
            if np.ptp(x) <= 0:
                continue
            span = float(np.ptp(x))
            ceiling = span / DT_H            # fastest this region can resolve
            slope = np.polyfit(x, first, 1)[0]          # hours per km
            if not np.isfinite(slope):
                continue
            speed = ceiling if slope == 0.0 else min(1.0 / abs(slope), ceiling)
            n_censored += slope == 0.0 or 1.0 / abs(slope) > ceiling
            n_used += 1
            for i in u:
                per_link[i].append(speed)
    if links is None:
        raise RuntimeError(f"{panel}: no source day matched")
    counts = np.array([len(v) for v in per_link])
    med = np.array([np.median(v) if v else np.nan for v in per_link])
    panel_med = float(np.nanmedian(med)) if np.isfinite(med).any() else np.nan
    fd = fd_wave(panel, links)
    out = np.where(counts >= TB.PRIOR_MIN_DEN, med, panel_med)
    out = np.where(np.isfinite(out), out, fd)
    out = np.where(np.isfinite(out), out, abs(TF.C_CONG_KMH))
    diag = {"panel": panel, "n_links": len(links), "n_regions": n_regions,
            "n_regions_used": n_used, "n_regions_censored": int(n_censored),
            "censored_share": float(n_censored / max(n_used, 1)),
            "links_with_own_estimate": int((counts >= TB.PRIOR_MIN_DEN).sum()),
            "panel_median_kmh": panel_med,
            "fd_median_kmh": float(np.nanmedian(fd)),
            "out_median_kmh": float(np.nanmedian(out)),
            "out_p10_kmh": float(np.nanpercentile(out, 10)),
            "out_p90_kmh": float(np.nanpercentile(out, 90))}
    return out, diag


def waves_for(panels: list[str], source_dates: set[str]):
    w, diags = {}, []
    for p in panels:
        arr, d = panel_wave(p, source_dates)
        w[p] = arr
        diags.append(d)
    return w, pd.DataFrame(diags)
