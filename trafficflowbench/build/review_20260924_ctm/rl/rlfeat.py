"""ONE feature implementation for the RL residual model (train cache, self-supervised check and serving all call this).
Reproduces floor_audit/s4_feats.py (local deltas / shocks) + s15_longrange.py (long-range shock bands) exactly; the
column order of FEATS is floor_audit/s18_feats.json.

Inputs per day (all L x T, links in milepost order):
  v1, q1   : C1 fields (stack at targets, G2 inside holes, blend elsewhere), q1 in veh/h/LANE
  ov, oq   : masked-layer observations (NaN where unseen), oq per lane;  oo: observed occupancy
  vf, ln, Lk: free speed, lanes (score1.lanes_by_key, >=1), link length -- length-L vectors
  tg       : boolean mask of the cells to featurise (non-hole Task-1 targets)
No IDs.  F = observed value where finite, else the C1 field."""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEATS = json.load(open(HERE / 's18_feats.json'))
BANDS = ((4, 8), (9, 16), (17, 32), (33, 400))


def sh(x, dl, dt):
    """value at (l+dl, t+dt) aligned to (l, t); NaN outside."""
    y = np.full(x.shape, np.nan, dtype=float); L, T = x.shape
    ls = slice(max(dl, 0), L + min(dl, 0)); ld = slice(max(-dl, 0), L + min(-dl, 0))
    ts = slice(max(dt, 0), T + min(dt, 0)); td = slice(max(-dt, 0), T + min(-dt, 0))
    y[ld, td] = x[ls, ts]; return y


def band_means(S, O):
    """S: shock field (nan where not observed); O: observed mask. (side, lo) -> mean over rows l+d, |d| in band."""
    L = S.shape[0]; val = np.where(O, S, 0.0); cnt = O.astype(float)
    cs = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(val, 0)]); cc = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(cnt, 0)])
    idx = np.arange(L)
    out = {}
    for lo, hi in BANDS:
        for side in (-1, 1):
            if side > 0: a = np.clip(idx + lo, 0, L); b = np.clip(idx + hi + 1, 0, L)
            else: a = np.clip(idx - hi, 0, L); b = np.clip(idx - lo + 1, 0, L)
            s = cs[b] - cs[a]; n = cc[b] - cc[a]
            out[(side, lo)] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    a3 = np.clip(idx - 3, 0, L); b3 = np.clip(idx + 4, 0, L)   # corridor-wide excluding |d|<=3
    s = cs[L][None, :] - (cs[b3] - cs[a3]); n = cc[L][None, :] - (cc[b3] - cc[a3])
    out['all'] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def _tmean(F):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean(np.stack([sh(F, 0, h) for h in (-2, -1, 1, 2)]), axis=0)


def day_features(v1, q1, ov, oq, oo, vf, ln, Lk, tg):
    """dict of 1-D float32 arrays (one entry per True cell of tg, row-major order == np.where(tg))."""
    vf = np.asarray(vf, float)[:, None]; ln = np.asarray(ln, float)[:, None]; Lk = np.asarray(Lk, float)[:, None]
    isob = np.isfinite(ov) & np.isfinite(oq)
    Fv = np.where(isob, ov, v1); Fq = np.where(isob, oq, q1)
    bv = _tmean(Fv); bq = _tmean(Fq)
    shv = Fv - bv; shq = Fq - bq
    li, ci = np.where(tg); c = {}
    c['p_v'] = v1[tg]; c['p_q'] = q1[tg]; c['p_k'] = q1[tg] / np.clip(v1[tg], 1, None); c['p_r'] = v1[tg] / vf[li, 0]
    c['vf'] = vf[li, 0]; c['lanes'] = ln[li, 0]; c['len'] = Lk[li, 0]; c['tod'] = ci.astype(float)
    c['own_shv'] = (v1 - bv)[tg]; c['own_shq'] = (q1 - bq)[tg]
    obf = isob.astype(float)
    for h in (-2, -1, 1, 2):
        c[f'dv_t{h:+d}'] = (sh(Fv, 0, h) - v1)[tg]; c[f'dq_t{h:+d}'] = (sh(Fq, 0, h) - q1)[tg]; c[f'ob_t{h:+d}'] = sh(obf, 0, h)[tg]
    for d in (-2, -1, 1, 2):
        c[f'dv_x{d:+d}'] = (sh(Fv, d, 0) - v1)[tg]; c[f'dq_x{d:+d}'] = (sh(Fq, d, 0) - q1)[tg]
        c[f'shv_x{d:+d}'] = sh(shv, d, 0)[tg]; c[f'shq_x{d:+d}'] = sh(shq, d, 0)[tg]; c[f'ob_x{d:+d}'] = sh(obf, d, 0)[tg]
        c[f'occ_x{d:+d}'] = sh(oo, d, 0)[tg]
    for d in (-1, 1):
        for h in (-1, 1):
            c[f'dv_x{d:+d}t{h:+d}'] = (sh(Fv, d, h) - v1)[tg]; c[f'dq_x{d:+d}t{h:+d}'] = (sh(Fq, d, h) - q1)[tg]
    for h in (-1, 1): c[f'occ_t{h:+d}'] = sh(oo, 0, h)[tg]
    for nm, F, b_ in (('v', Fv, bv), ('q', Fq, bq)):
        S_ = np.where(isob, F - b_, np.nan); Ob_ = isob & np.isfinite(S_)
        for h in (-1, 0, 1):
            bm = band_means(sh(S_, 0, h), sh(Ob_.astype(float), 0, h) > 0.5)
            for key, arr in bm.items():
                if key == 'all' and h != 0: continue
                c[f'lr_{nm}_h{h:+d}_' + ('all' if key == 'all' else f'{"dn" if key[0] > 0 else "up"}{key[1]}')] = arr[tg]
    assert set(FEATS) <= set(c), sorted(set(FEATS) - set(c))
    return {k: np.asarray(c[k], dtype=np.float32) for k in FEATS}


# ---------------------------------------------------------------- RLB block features (extension, see README)
BLK_FEATS = ['blk_has', 'blk_n', 'blk_fq', 'blk_mr', 'blk_st', 'blk_st_m1', 'blk_st_p1', 'blk_fq_m1', 'blk_fq_p1', 'blk_mr_m1', 'blk_mr_p1',
             'tpl_last', 'tpl_next', 'tpl_lastd', 'tpl_nextd', 'tpl_mean', 'tpl_dv',
             'tpl_day_q', 'tpl_day_f', 'tpl_day_q_r', 'tpl_day_f_r', 'tpl_day_same_dv', 'own_n_q', 'own_n_f']


def block_features(v1, ov, vf, tg, bid, win=6):
    """Co-switching block state + per-link queued-speed template, from OBSERVED speeds only (inference-time, no IDs).
    v1: C1 speed field; ov: observed speed (NaN unseen); vf: free speed per link; bid: block id per link (milepost order).
      blk_n / blk_fq / blk_mr : # observed block-mates at t, fraction of them queued (v <= .6 vf), their mean v/vf
      blk_st (= blk_fq >= .5) and the same at t-1 / t+1
      tpl_last / tpl_next     : own last / next observed speed within +-win steps taken when the mates' state equalled now
      tpl_day_q / tpl_day_f   : own median observed speed over the day while the mates were queued / free
      tpl_day_same_dv         : the day template for the CURRENT mates' state minus C1
    Returns dict of float32 arrays aligned with np.where(tg)."""
    import warnings
    L, T = v1.shape; vfc = np.asarray(vf, float)[:, None]; bid = np.asarray(bid)
    isob = np.isfinite(ov)
    r = np.where(isob, ov / vfc, 0.0); qd = np.where(isob, (ov <= 0.6 * vfc).astype(float), 0.0); o = isob.astype(float)
    nb = np.zeros((L, T)); sq = np.zeros((L, T)); sr = np.zeros((L, T))
    for b in np.unique(bid[bid >= 0]):
        m = np.where(bid == b)[0]
        if len(m) < 2: continue
        no = o[m].sum(0); nq = qd[m].sum(0); nr = r[m].sum(0)
        nb[m] = no[None, :] - o[m]; sq[m] = nq[None, :] - qd[m]; sr[m] = nr[None, :] - r[m]   # leave-own-link-out
    with np.errstate(all='ignore'):
        fq = np.where(nb > 0, sq / np.maximum(nb, 1), np.nan); mr = np.where(nb > 0, sr / np.maximum(nb, 1), np.nan)
    st = np.where(np.isfinite(fq), (fq >= 0.5).astype(float), np.nan)
    lastv = np.full((L, T), np.nan); nextv = np.full((L, T), np.nan); lastd = np.full((L, T), np.nan); nextd = np.full((L, T), np.nan)
    for k in range(1, win + 1):
        for arr, dist, dt in ((lastv, lastd, -k), (nextv, nextd, k)):
            ovs = sh(ov, 0, dt); sts = sh(st, 0, dt)
            ok = np.isnan(arr) & np.isfinite(ovs) & np.isfinite(st) & (sts == st)
            arr[ok] = ovs[ok]; dist[ok] = k
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tm = np.nanmean(np.stack([lastv, nextv]), 0)
        dq_ = np.nanmedian(np.where(isob & (st == 1), ov, np.nan), axis=1); df_ = np.nanmedian(np.where(isob & (st == 0), ov, np.nan), axis=1)
    nq_ = (isob & (st == 1)).sum(1).astype(float); nf_ = (isob & (st == 0)).sum(1).astype(float)
    same = np.where(st == 1, dq_[:, None], np.where(st == 0, df_[:, None], np.nan))
    li = np.where(tg)[0]
    c = {'blk_has': (bid[li] >= 0) & np.isin(bid[li], [b for b in np.unique(bid) if (bid == b).sum() > 1]),
         'blk_n': nb[tg], 'blk_fq': fq[tg], 'blk_mr': mr[tg], 'blk_st': st[tg],
         'blk_st_m1': sh(st, 0, -1)[tg], 'blk_st_p1': sh(st, 0, 1)[tg], 'blk_fq_m1': sh(fq, 0, -1)[tg], 'blk_fq_p1': sh(fq, 0, 1)[tg],
         'blk_mr_m1': sh(mr, 0, -1)[tg], 'blk_mr_p1': sh(mr, 0, 1)[tg],
         'tpl_last': lastv[tg], 'tpl_next': nextv[tg], 'tpl_lastd': lastd[tg], 'tpl_nextd': nextd[tg], 'tpl_mean': tm[tg], 'tpl_dv': (tm - v1)[tg],
         'tpl_day_q': dq_[li], 'tpl_day_f': df_[li], 'tpl_day_q_r': dq_[li] / vfc[li, 0], 'tpl_day_f_r': df_[li] / vfc[li, 0],
         'tpl_day_same_dv': (same - v1)[tg], 'own_n_q': nq_[li], 'own_n_f': nf_[li]}
    assert list(c) == BLK_FEATS
    return {k: np.asarray(v, dtype=np.float32) for k, v in c.items()}
