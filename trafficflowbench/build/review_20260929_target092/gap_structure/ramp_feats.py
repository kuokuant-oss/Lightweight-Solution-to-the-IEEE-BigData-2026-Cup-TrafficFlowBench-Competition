"""In-gap ramp features for 90-min gap rows (Task 1 is offline: ramp_states inside the gap are released and legal for
Task 1 per the Rules page 'Tasks 1, 3 and 4 are offline ... any released observation in the split, at any time').
Mainline inside the gap is never read.  Per (panel, date, origin): ramp counts at the gap columns a..a+17 normalised by
the same ramps' pre-gap (a-6..a-1) and post-gap (a+18..a+23) levels; corridor totals and local (upstream/downstream
W links) totals, split by ramp type.  3-step centred smoothing inside the gap only."""
# Helper module (no main); the gap_structure scripts import it to attach ramp
# features to gap rows. FR = off-ramp, OR = on-ramp. Traffic meaning: when the
# mainline queues back past an on-ramp, the ramp count drops, so in-gap ramp
# counts relative to their usual level act as a queue sensor.
# RAMP_FEATS: corridor ('c') and local ('up' / 'dn': ramps attached within
# W = 8 links upstream / downstream) ratios of in-gap ramp counts to the same
# ramps' pre-gap or post-gap level, plus a pre-to-post blend.
# NEAR_FEATS: in-gap counts of the nearest on-ramps over their pre-gap,
# post-gap and time-of-day profile levels.
# Input: the per-split cache written by build_cache.py (ramp counts per day,
# ramps x 288 five-minute slots).
import numpy as np, pandas as pd
from pathlib import Path
# Original absolute roots; bin/tfb-run maps them onto the run tree.
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
DATA = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
# LG: gap length in slots (18 x 5 min = 90 min). W: link reach of the local
# up/down ramp totals.
LG = 18
W = 8


def load_panel(panel, split):
    """Loads the ramp count cache of one panel and split.

    Returns:
      Dict with R (days x ramps x 288 ramp flows, NaN where not eligible),
      links (mainline link ids), rtype (ramp type per ramp, 'FR' or 'OR'),
      att (index of the attached mainline link, -1 if unknown) and dates
      (date string -> day index).
    """
    z = np.load(ROOT / f'build/review_20260924_ctm/cache/{panel}_{split}.npz', allow_pickle=True)
    R = z['R'].astype(np.float32); RE = z['RE']; R[~RE] = np.nan
    links = [str(x) for x in z['links']]; ramps = [str(x) for x in z['ramps']]; rtype = np.array(list(z['rtype']))
    amap = pd.read_csv(DATA / f'corridors/{panel}/network/ramp_attachment_map.csv', dtype=str).set_index('ramp_link_id')
    li_of = {l: i for i, l in enumerate(links)}
    att = np.array([li_of.get(amap.nearest_mainline_link_id.get(r, ''), -1) for r in ramps])
    dates = {str(d): i for i, d in enumerate(z['dates'])}
    return dict(R=R, links=links, rtype=rtype, att=att, dates=dates)


def _ratio(Rg, Rref, sel):
    """Sum over selected ramps observed in both the gap column and the reference window; ratio of sums."""
    # Rg: (nR, LG); Rref: (nR,) mean level (nan if unavailable)
    ok = np.isfinite(Rg) & np.isfinite(Rref)[:, None] & sel[:, None]
    num = np.where(ok, Rg, 0).sum(0); den = np.where(ok, Rref[:, None], 0).sum(0)
    with np.errstate(all='ignore'):
        return np.where(den > 0, num / den, np.nan), ok.sum(0)


def smooth(x):
    """Returns the centred 3-slot mean of `x`, ignoring NaN (NaN where all three are missing)."""
    y = x.copy()
    for s in range(len(x)):
        lo, hi = max(0, s - 1), min(len(x), s + 2)
        v = x[lo:hi]; v = v[np.isfinite(v)]
        y[s] = v.mean() if len(v) else np.nan
    return y


def window_feats(P, date, T, nL):
    """Return dict of arrays (nL, LG)."""
    # Gap slots a..a+17 with a = T + 1. Reference levels per ramp: mean of the
    # 6 slots before the gap (pre) and of the 6 slots after it (post).
    R = P['R'][P['dates'][date]]  # (nR, 288)
    a = T + 1
    g = R[:, a:a + LG]
    with np.errstate(all='ignore'):
        import warnings; warnings.simplefilter('ignore')
        pre = np.nanmean(R[:, max(a - 6, 0):a], axis=1)
        post = np.nanmean(R[:, a + LG:a + LG + 6], axis=1)
    out = {}
    for tp in ('FR', 'OR'):
        sel = P['rtype'] == tp
        for ref_name, ref in (('pre', pre), ('post', post)):
            r, n = _ratio(g, ref, sel & (P['att'] >= 0))
            out[f'c{tp}_{ref_name}'] = np.repeat(smooth(r)[None], nL, 0)
            out[f'c{tp}_{ref_name}_raw'] = np.repeat(r[None], nL, 0)
        # local upstream / downstream (by attachment index)
        for side in ('up', 'dn'):
            arr = {f'{side}{tp}_pre': np.full((nL, LG), np.nan), f'{side}{tp}_post': np.full((nL, LG), np.nan)}
            for li in range(nL):
                if side == 'up': loc = sel & (P['att'] >= li - W) & (P['att'] <= li)
                else: loc = sel & (P['att'] >= li) & (P['att'] <= li + W)
                if not loc.any(): continue
                r1, _ = _ratio(g, pre, loc); r2, _ = _ratio(g, post, loc)
                arr[f'{side}{tp}_pre'][li] = smooth(r1); arr[f'{side}{tp}_post'][li] = smooth(r2)
            out.update(arr)
    # interpolated expectation: pre ratio weighted toward start, post toward end
    w = (np.arange(LG) + 1) / (LG + 1)
    for tp in ('FR', 'OR'):
        out[f'c{tp}_blend'] = (1 - w)[None] * out[f'c{tp}_pre'] + w[None] * out[f'c{tp}_post']
    return out


RAMP_FEATS = [f'{p}{tp}_{r}' for tp in ('FR', 'OR') for p in ('c', 'up', 'dn') for r in ('pre', 'post')] + \
             ['cFR_blend', 'cOR_blend', 'cFR_pre_raw', 'cFR_post_raw']


def attach(rows, split):
    """rows: DataFrame with panel,date,origin,li,step_i. Adds RAMP_FEATS columns (float32)."""
    # Features are computed once per (panel, date, origin) window, then each
    # row picks its link and gap step.
    out = {c: np.full(len(rows), np.nan, np.float32) for c in RAMP_FEATS}
    for panel, gp in rows.groupby('panel'):
        P = load_panel(panel, split); nL = len(P['links'])
        for (date, T), gw in gp.groupby(['date', 'origin']):
            f = window_feats(P, date, int(T), nL)
            li = gw.li.to_numpy(int); si = gw.step_i.to_numpy(int); ix = gw.index.to_numpy()
            for c in RAMP_FEATS:
                out[c][ix] = f[c][li, si]
        del P
    for c in RAMP_FEATS:
        rows[c] = out[c]
    return rows


# ---- v2: nearest-ramp features (on-ramp counts collapse to ~15% while the adjacent upstream mainline is queued; a7) ----
NEAR_FEATS = [f'{nm}_{r}' for nm in ('nOR1', 'nOR2', 'uOR1') for r in ('pre', 'post', 'prof', 'rawprof')] + \
             ['nOR1_d', 'nOR2_d', 'uOR1_d', 'nOR_min_prof', 'cOR_prof']


def add_profile(P):
    """Adds Rsum / Rcnt: per ramp and slot, the sum and count of observed counts over all days of the split."""
    R = P['R']; ok = np.isfinite(R)
    P['Rsum'] = np.where(ok, R, 0).sum(0); P['Rcnt'] = ok.sum(0)
    return P


def near_feats(P, date, T, nL):
    """Builds the nearest-on-ramp features of one gap window.

    nOR1 / nOR2: the first / second on-ramp attached at the link or up to 3
    links downstream; uOR1: the first on-ramp up to 3 links upstream.

    Args:
      P: Panel cache from `load_panel` after `add_profile`.
      date: Window date.
      T: Forecast time slot; the gap covers slots T+1..T+18.
      nL: Number of mainline links.

    Returns:
      Dict NEAR_FEATS name -> array (nL, LG). '<ramp>_pre' / '_post' /
      '_prof': smoothed in-gap count over the pre-gap level, the post-gap
      level or the leave-this-day-out time-of-day mean ('_rawprof':
      unsmoothed); '<ramp>_d': link distance to the ramp; nOR_min_prof: the
      lower profile ratio of the two downstream ramps; cOR_prof: corridor
      on-ramp total over its profile.
    """
    di = P['dates'][date]; R = P['R'][di]; a = T + 1
    ok = np.isfinite(R)
    with np.errstate(all='ignore'):
        prof = (P['Rsum'] - np.where(ok, R, 0)) / np.maximum(P['Rcnt'] - ok, 1)   # leave-this-day-out tod mean
        import warnings; warnings.simplefilter('ignore')
        pre = np.nanmean(R[:, max(a - 6, 0):a], axis=1); post = np.nanmean(R[:, a + LG:a + LG + 6], axis=1)
    g = R[:, a:a + LG]; pg = prof[:, a:a + LG]
    orr = np.where((P['rtype'] == 'OR') & (P['att'] >= 0))[0]; att = P['att'][orr]
    out = {c: np.full((nL, LG), np.nan, np.float32) for c in NEAR_FEATS}
    def sm(x):
        """Returns `smooth(x)`."""
        return smooth(x)
    with np.errstate(all='ignore'):
        r_pre = g[orr] / pre[orr, None]; r_post = g[orr] / post[orr, None]; r_prof = g[orr] / pg[orr]
        # corridor on-ramp total vs profile
        okc = np.isfinite(g[orr]) & np.isfinite(pg[orr]) & (pg[orr] > 0)
        cprof = np.where(okc, g[orr], 0).sum(0) / np.maximum(np.where(okc, pg[orr], 0).sum(0), 1e-9)
    out['cOR_prof'][:] = smooth(cprof)[None]
    for li in range(nL):
        dn = np.where((att >= li) & (att <= li + 3))[0]; dn = dn[np.argsort(att[dn] - li, kind='stable')]
        up = np.where((att >= li - 3) & (att < li))[0]; up = up[np.argsort(li - att[up], kind='stable')]
        mins = []
        for nm, idx in (('nOR1', dn[:1]), ('nOR2', dn[1:2]), ('uOR1', up[:1])):
            if not len(idx): continue
            j = idx[0]
            out[f'{nm}_pre'][li] = sm(r_pre[j]); out[f'{nm}_post'][li] = sm(r_post[j]); out[f'{nm}_prof'][li] = sm(r_prof[j])
            out[f'{nm}_rawprof'][li] = r_prof[j]; out[f'{nm}_d'][li] = abs(att[j] - li)
            if nm != 'uOR1': mins.append(sm(r_prof[j]))
        if mins:
            out['nOR_min_prof'][li] = np.nanmin(np.vstack(mins), axis=0) if len(mins) > 1 else mins[0]
    return out


def attach_near(rows, split):
    """Adds the NEAR_FEATS columns (float32) to `rows` (panel, date, origin, li, step_i) and returns it."""
    out = {c: np.full(len(rows), np.nan, np.float32) for c in NEAR_FEATS}
    for panel, gp in rows.groupby('panel'):
        P = add_profile(load_panel(panel, split)); nL = len(P['links'])
        for (date, T), gw in gp.groupby(['date', 'origin']):
            f = near_feats(P, date, int(T), nL)
            li = gw.li.to_numpy(int); si = gw.step_i.to_numpy(int); ix = gw.index.to_numpy()
            for c in NEAR_FEATS:
                out[c][ix] = f[c][li, si]
        del P
    for c in NEAR_FEATS:
        rows[c] = out[c]
    return rows
