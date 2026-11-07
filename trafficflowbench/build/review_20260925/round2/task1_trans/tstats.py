"""Leave-own-day-out split statistics per (link, slot) from the released observed layer of one split.
All arrays are computed from the masked/observed layer only (what serving has); official Task 2 holes are NaN there.
Temporal smoothing: box of +-HW slots on the per-day sums before the leave-one-day-out subtraction."""
# Used by the month-specific hole-speed correction (T1): build_ph.py and
# round3/task1_trans_serving/s3_build_rows.py compute these statistics on each
# split, and the strict checks recompute them on one day-parity half. Arrays
# are (days D, links L, five-minute slots T).
import numpy as np

# Half-width of the time box: +-2 slots (+-10 min).
HW = 2


def _box(x, hw=HW):
    """Box-sum along the last axis (slots), same length."""
    if hw == 0: return x
    c = np.cumsum(np.pad(x, [(0, 0)] * (x.ndim - 1) + [(hw + 1, hw)]), axis=-1)
    return c[..., 2 * hw + 1:] - c[..., :-(2 * hw + 1)]


class SplitStats:
    """Per-split link x slot statistics with a leave-one-day-out view.

    A cell is observed when speed and flow are finite and the cell is
    score-eligible; it is queued when its speed is at most vcut. Every per-day
    sum is box-summed over +-HW slots, so loo(d) subtracts day d from the
    split totals.
    """
    def __init__(self, V, Q, E, vcut, lanes, bid):
        """V, Q: (D, L, T) observed speed / flow (NaN = unseen); E: eligibility; vcut, lanes: (L,); bid: block id (L,)."""
        # Per-day box sums (D, L, T): cnt observed cells, sv speed, sq flow
        # per lane, nq queued cells, svq speed of queued cells. tq_s, tq_n:
        # whole-day queued-speed sum and count per link (D, L). M: block
        # membership matrix (blocks x links); bi: block index of each link
        # (-1 = none). T: split totals of the box sums.
        ob = np.isfinite(V) & np.isfinite(Q) & E
        v = np.where(ob, V, 0.0); ql = np.where(ob, Q / lanes[None, :, None], 0.0)
        qd = ob & (V <= vcut[None, :, None])
        self.cnt = _box(ob.astype(np.float64)); self.sv = _box(v); self.sq = _box(ql); self.nq = _box(qd.astype(np.float64))
        self.svq = _box(np.where(qd, V, 0.0))
        # per-link whole-day queued-speed template (sum / count over the day)
        self.tq_s = np.where(qd, V, 0.0).sum(-1); self.tq_n = qd.sum(-1).astype(np.float64)
        ub = np.unique(bid[bid >= 0]); M = np.zeros((len(ub), len(bid))); bi = np.full(len(bid), -1)
        for k, b in enumerate(ub):
            ix = np.where(bid == b)[0]; M[k, ix] = 1; bi[ix] = k
        self.M, self.bi = M, bi
        self.T = {k: getattr(self, k).sum(0) for k in ('cnt', 'sv', 'sq', 'nq', 'svq')}
        self.tq_S, self.tq_N = self.tq_s.sum(0), self.tq_n.sum(0)
        self.D = V.shape[0]

    def loo(self, d):
        """Dict of (L, T) leave-day-d-out statistics."""
        # s_v: mean speed (km/h); s_ql: mean flow per lane (veh/h); s_qf:
        # queued share; s_n: observed share of the box cells on the other
        # days; s_vq: mean speed when queued; s_tq: whole-day queued-speed mean
        # of the link (needs >= 20 queued cells); s_bqf: queued share of the
        # link's block. Cells without any observation on the other days are
        # NaN, except s_n.
        c = self.T['cnt'] - self.cnt[d]
        with np.errstate(all='ignore'):
            o = dict(s_v=(self.T['sv'] - self.sv[d]) / c, s_ql=(self.T['sq'] - self.sq[d]) / c,
                     s_qf=(self.T['nq'] - self.nq[d]) / c, s_n=c / (2 * HW + 1) / max(self.D - 1, 1))
            nqq = self.T['nq'] - self.nq[d]
            o['s_vq'] = np.where(nqq > 0, (self.T['svq'] - self.svq[d]) / np.maximum(nqq, 1), np.nan)
            tn = self.tq_N - self.tq_n[d]
            o['s_tq'] = np.repeat(np.where(tn >= 20, (self.tq_S - self.tq_s[d]) / np.maximum(tn, 1), np.nan)[:, None], c.shape[1], 1)
            # block-level queued fraction (link's own value if it has no block)
            if self.M.shape[0] == 0:   # panel without blocks (D12_I405_N/S)
                o['s_bqf'] = o['s_qf'].copy()
            else:
                bn = self.M @ (self.T['nq'] - self.nq[d]); bc = self.M @ c
                bf = np.where(bc > 0, bn / np.maximum(bc, 1), np.nan)
                o['s_bqf'] = np.where(self.bi[:, None] >= 0, bf[np.maximum(self.bi, 0)], o['s_qf'])
        for k in o: o[k] = np.where(c > 0, o[k], np.nan) if k != 's_n' else o[k]
        return o
