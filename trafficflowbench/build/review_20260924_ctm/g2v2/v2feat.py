"""G2v2 extra features for Task 1 cells inside a Task 2 hole.  Shared by training rows, the paired harness and serving.

Inputs are exactly what validation/private provide: the observed layer V (masked layer, window history T-60..T-5
republished, hole T+5..T+90 blank, post-hole observations kept), eligibility E, the public window_index
(condition, origins of all windows of the panel-split -> spacing), train-only blocks and train-only templates.
Nothing inside the hole is read (asserted by callers: the hole is NaN).

Conventions: t = slot of T (forecast_origin), hole = slots a..a+LG-1 with a = t+1, history = t-12..t-1.
"""
#
# G2v2: the second version of the gap model for the 90-minute masked gap after each Task 2 window (the "hole").
# Role: imported by g2v2/data_v2.py (training rows), g2v2/eval_v2.py (paired harness) and g2v2/serve_v2.py
# (reproduction pipeline, task1.sh step 4). Blocks are groups of links that queue together; a link is queued
# when its observed speed is at or below vcut (60% of free-flow speed).
import numpy as np
# LG: hole length in 5-minute steps (30-minute horizon + 60 minutes); HISTN: window history length in steps.
LG, HISTN = 18, 12
NB = 131   # template histogram bins: speed 0..130 km/h, 1 km/h


def selector_feats(V, E, vc, t, gap_min, cond):
    """Serving-identical re-implementation of build/review_20260924_ctm/dense/serve_sel2.py.
    prev_rej = 1 if origin T-5 was ongoing-shaped (cov>=.7 over its 12-step history, observed queue in that history
    and at its last two steps) and not spacing-blocked; NaN if T-5 was blocked. n_rej = run length of such
    predecessors (cap 12).  blk1 = 1 if T-5 was spacing-blocked (T - T_prev - 5 < 360), for both conditions.
    gap_min = T - T_prev in minutes (NaN if no previous window in the sequence/split)."""
    # Args: V speed (km/h) and E eligibility of the day (links x steps), vc per-link queue speed, t origin step,
    # gap_min minutes since the previous window, cond 'queue_onset' or 'queue_ongoing'.
    # Returns (prev_rej, n_rej, blk1).
    Qo = (V <= vc[:, None]) & E & np.isfinite(V)

    def shaped(o):
        """Returns True when origin step o has the shape of an ongoing candidate (see the docstring above)."""
        if o - 12 < 0: return False
        cov = E[:, o - 12:o].mean()
        return bool(cov >= 0.7 and Qo[:, o - 12:o].any() and Qo[:, o - 1].any() and Qo[:, o - 2].any())

    def blocked(o):
        """Returns True when origin step o falls within 360 minutes of the previous window."""
        return bool(np.isfinite(gap_min) and (gap_min - 5 * (t - o)) < 360)
    blk1 = float(blocked(t - 1))
    if cond != 'queue_ongoing':
        return np.nan, np.nan, blk1
    if blocked(t - 1):
        return np.nan, np.nan, blk1
    k = 0
    while k < 12 and not blocked(t - 1 - k) and shaped(t - 1 - k): k += 1
    return float(k > 0), float(k), blk1


def template_from_hist(H):
    """H: (nL, NB) counts of queued truth speeds -> (median, iqr) per link (NaN if < 20 cells)."""
    n = H.sum(1); cs = np.cumsum(H, 1)
    def q(p):
        """Returns the bin center of quantile p per link (NaN for links with fewer than 20 queued cells)."""
        idx = np.argmax(cs >= np.maximum(p * n, 1)[:, None], axis=1).astype(float) + 0.5
        return np.where(n >= 20, idx, np.nan)
    return q(0.5), q(0.75) - q(0.25)


def v2_features(V, E, a, nT, vcut, bid, tmpl_med, tmpl_iqr, qpre_link, qpost_link, win):
    """Extra feature arrays (nL x LG).  V: observed speed layer (hole blank).  bid: block id per link (grid order,
    -1 = unknown).  qpre_link / qpost_link: the base model's own pre/post queue flags (0/1 per link).
    win: dict(cond, prev_rej, n_rej, blk1)."""
    # Args: a first hole column (t+1), nT steps per day, vcut per-link queue speed (km/h), tmpl_med / tmpl_iqr
    # train-only queued-speed template per link (km/h). Returns a dict of feature name -> float32 (nL x LG).
    nL = V.shape[0]; t = a - 1; e = a + LG
    obs = np.isfinite(V); q = obs & (V <= vcut[:, None])
    f = {}
    cols = lambda x: np.repeat(np.asarray(x, np.float32)[:, None], LG, 1)
    # Window level: condition and selector features, the same for every cell of the window.
    # --- window level
    f['cond_on'] = np.full((nL, LG), 1.0 if win['cond'] == 'queue_ongoing' else 0.0, np.float32)
    f['prev_rej'] = np.full((nL, LG), win['prev_rej'], np.float32)
    f['n_rej'] = np.full((nL, LG), win['n_rej'], np.float32)
    f['blk1'] = np.full((nL, LG), win['blk1'], np.float32)
    # Block level: M is the (blocks x links) membership matrix.
    # --- block level
    ub = np.unique(bid[bid >= 0]); M = np.zeros((len(ub), nL), np.float32)
    bi = np.full(nL, -1)
    for k, b in enumerate(ub):
        ix = np.where(bid == b)[0]; M[k, ix] = 1; bi[ix] = k
    bsize = np.where(bi >= 0, M.sum(1)[np.maximum(bi, 0)], 1.0)
    def bfrac(c):
        """Returns per link the queued share of its block's observed links at step c.

        Links outside a block use their own observed state; NaN when nothing is observed.
        """
        if c < 0 or c >= nT: return np.full(nL, np.nan)
        no = M @ obs[:, c].astype(np.float32); nq = M @ q[:, c].astype(np.float32)
        fr = np.where(no > 0, nq / np.maximum(no, 1), np.nan)
        out = np.where(bi >= 0, fr[np.maximum(bi, 0)], np.where(obs[:, c], q[:, c].astype(float), np.nan))
        return out
    f['bsize'] = cols(bsize)
    # bq_pre0 / bq_pre1: block queue share at T and T-5; bq_hist: mean over the history t-12..t-1.
    f['bq_pre0'] = cols(bfrac(t)); f['bq_pre1'] = cols(bfrac(t - 1))
    with np.errstate(all='ignore'):
        import warnings; warnings.simplefilter('ignore')
        f['bq_hist'] = cols(np.nanmean(np.stack([bfrac(c) for c in range(max(t - HISTN, 0), t)]), 0))
    # bq_post: first observed block queue share within the hour after the hole, bq_postd: its distance in steps.
    bpost = np.full(nL, np.nan); bpostd = np.full(nL, np.nan)
    for c in range(e, min(e + 12, nT)):
        x = bfrac(c); m = np.isnan(bpost) & np.isfinite(x); bpost[m] = x[m]; bpostd[m] = c - (e - 1)
    f['bq_post'] = cols(bpost); f['bq_postd'] = cols(bpostd)
    # --- templates: own speed at the most recent pre-hole queued observation (same day), first post-hole one
    td = np.full(nL, np.nan); tda = np.full(nL, np.nan)
    for c in range(t, -1, -1):
        m = np.isnan(td) & q[:, c]; td[m] = V[m, c]; tda[m] = t - c
    tp = np.full(nL, np.nan); tpd = np.full(nL, np.nan)
    for c in range(e, nT):
        m = np.isnan(tp) & q[:, c]; tp[m] = V[m, c]; tpd[m] = c - (e - 1)
    f['tmpl_dyn'] = cols(td); f['tmpl_dyn_age'] = cols(tda); f['tmpl_post'] = cols(tp); f['tmpl_post_d'] = cols(tpd)
    # tmpl_stat / tmpl_iqr: train-only template (median and IQR of queued speeds).
    f['tmpl_stat'] = cols(tmpl_med); f['tmpl_iqr'] = cols(tmpl_iqr)
    # The block-mean template described on the next line is not computed here.
    # block-mean template at pre (mean own template over block-mates that have one) -- helps links without own history
    # --- distance (in links) to the nearest queued link, pre and post, up / down the milepost chain
    idx = np.arange(nL)
    # Links farther than 30 links, or no queued link at all, are capped at 30. nq_*: share of queued links.
    for nm, fl in (('pre', qpre_link), ('post', qpost_link)):
        fl = np.asarray(fl) > 0.5
        qi = idx[fl]
        if len(qi) == 0:
            up = np.full(nL, 30.0); dn = np.full(nL, 30.0)
        else:
            pos = np.searchsorted(qi, idx)           # first queued index >= i
            dn = np.where(pos < len(qi), qi[np.minimum(pos, len(qi) - 1)] - idx, 30.0)
            pos2 = np.searchsorted(qi, idx, side='right') - 1   # last queued index <= i
            up = np.where(pos2 >= 0, idx - qi[np.maximum(pos2, 0)], 30.0)
        f[f'dq_up_{nm}'] = cols(np.minimum(up, 30)); f[f'dq_dn_{nm}'] = cols(np.minimum(dn, 30))
        f[f'nq_{nm}'] = np.full((nL, LG), float(fl.mean()), np.float32)
    return f


V2_FEATS = ['cond_on', 'prev_rej', 'n_rej', 'blk1', 'bsize', 'bq_pre0', 'bq_pre1', 'bq_hist', 'bq_post', 'bq_postd',
            'tmpl_dyn', 'tmpl_dyn_age', 'tmpl_post', 'tmpl_post_d', 'tmpl_stat', 'tmpl_iqr',
            'dq_up_pre', 'dq_dn_pre', 'nq_pre', 'dq_up_post', 'dq_dn_post', 'nq_post']


def v3_features(V, a, nT, vcut, bid, win, step):
    """Second batch (v3): block state trend and run lengths around the hole, neighbouring blocks, onset timing flag."""
    # Args as in v2_features; step: hole step index per cell (nL x LG), used for the onset timing flag.
    # Per-block values are mapped to links through bi; a link outside every block has bi = -1, which picks the
    # last block's value.
    nL = V.shape[0]; t = a - 1; e = a + LG
    obs = np.isfinite(V); q = obs & (V <= vcut[:, None])
    ub = np.unique(bid[bid >= 0]); M = np.zeros((len(ub), nL), np.float32); bi = np.full(nL, -1)
    for k, b in enumerate(ub):
        ix = np.where(bid == b)[0]; M[k, ix] = 1; bi[ix] = k
    def bfr(c):   # per-block fraction (nB,)
        """Returns the queued share of observed links per block at step c (NaN outside the day)."""
        if c < 0 or c >= nT: return np.full(len(ub), np.nan)
        no = M @ obs[:, c].astype(np.float32); nq = M @ q[:, c].astype(np.float32)
        return np.where(no > 0, nq / np.maximum(no, 1), np.nan)
    cols = lambda x: np.repeat(np.asarray(x, np.float32)[:, None], LG, 1)
    f = {}
    # Block queue shares for up to 3 hours before T and 3 hours after the hole.
    pre = {c: bfr(c) for c in range(max(t - 36, 0), t + 1)}
    post = {c: bfr(c) for c in range(e, min(e + 36, nT))}
    for nm, c in (('bq_pre2', t - 2), ('bq_pre3', t - 3)):
        f[nm] = cols(pre.get(c, np.full(len(ub), np.nan))[bi])
    for nm, c in (('bq_e0', e), ('bq_e1', e + 1), ('bq_e2', e + 2), ('bq_e3', e + 3)):
        f[nm] = cols(post.get(c, np.full(len(ub), np.nan))[bi])
    # run lengths: consecutive stamps (observed-majority queued) ending at t-1 / starting at e; NaN-stamps skipped
    def run(seq):
        """Returns per block the number of consecutive queued stamps from the start of seq, skipping NaN stamps."""
        n = np.zeros(len(ub)); alive = np.ones(len(ub), bool)
        for x in seq:
            on = np.isfinite(x) & (x > 0.5); off = np.isfinite(x) & (x <= 0.5)
            n += alive & on; alive &= ~off
        return n
    f['brun_pre'] = cols(run([pre[c] for c in sorted(pre) if c <= t - 1][::-1])[bi])
    f['brun_post'] = cols(run([post[c] for c in sorted(post)])[bi])
    # neighbouring blocks (by block order along the milepost chain) at the last pre and first post stamps
    last_pre = pre.get(t - 1, np.full(len(ub), np.nan)); first_post = np.full(len(ub), np.nan)
    for c in sorted(post):
        m = np.isnan(first_post) & np.isfinite(post[c]); first_post[m] = post[c][m]
    for nm, x in (('pre', last_pre), ('post', first_post)):
        up = np.r_[np.nan, x[:-1]]; dn = np.r_[x[1:], np.nan]
        f[f'nbu_{nm}'] = cols(up[bi]); f[f'nbd_{nm}'] = cols(dn[bi])
    # onset_late: onset window with T-5 free of the spacing rule; flags hole steps 5 and later.
    on6 = float(win['cond'] == 'queue_onset' and win['blk1'] == 0)
    f['onset_late'] = (on6 * (step >= 5)).astype(np.float32)
    return f


V3_FEATS = ['bq_pre2', 'bq_pre3', 'bq_e0', 'bq_e1', 'bq_e2', 'bq_e3', 'brun_pre', 'brun_post', 'nbu_pre', 'nbd_pre',
            'nbu_post', 'nbd_post', 'onset_late']
