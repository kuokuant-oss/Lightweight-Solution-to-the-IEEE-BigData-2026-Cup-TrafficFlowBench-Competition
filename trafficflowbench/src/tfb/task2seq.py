"""A small causal spatio-temporal model for Task 2, as a different hypothesis
class from the per-cell GBDT.

WHAT IS DIFFERENT. The 50-column table hands the tree one summary per cell:
last value, a mean, a couple of differences, a dispersion, three fixed
neighbour hops and a prior. It never sees the joint shape of the 12x(links)
history, and it predicts the six horizon steps as six independent rows that
share almost all their features. This model reads the whole window as a
(links x 12 x channels) tensor and emits all six steps at once.

STRUCTURE.
  temporal   a causal dilated TCN over the 12 history steps, per link, so step
             t only ever sees steps <= t. Dilations 1,2,4 give a receptive
             field of 12 steps -- the whole released history and no more.
  spatial    directed message passing along the link chain. Congestion travels
             UPSTREAM and demand travels DOWNSTREAM, so the two directions get
             separate kernels rather than one symmetric smoother, and which
             array direction is "upstream" comes from task2dir, not from the
             milepost order.
  head       one linear map to link x 6 logits.

TIME BOUNDARY. The input is exactly the released history columns t-12..t-1 and
the static network. The horizon t+1..t+6 is never read. Nothing is normalised
with statistics from the horizon or from a held-out fold.

The init_score prior is kept as an input channel and as an additive logit
offset, which is the same information the GBDT arm gets, so the comparison is
about the representation rather than about the prior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Role: no program of the reproduction pipeline imports this module; it is the
# record of the neural sequence-model experiment.
# CHANNELS: inputs per (link, history stamp) built by window_tensor, in order:
# speed / 100, flow / 2000, occupancy, eligibility, queued flag, margin
# (speed / v_cut), on-ramp and off-ramp flow / 2000, ramp-reading mask,
# capacity / 8000, free speed / 120, length (km), lanes / 5, and sine and
# cosine of the time of day. HIST, HOR: history and horizon length in steps.
CHANNELS = ("v_norm", "q_norm", "occ", "elig", "is_queued", "margin",
            "onramp", "offramp", "ramp_mask", "cap_norm", "vf_norm", "len_norm",
            "lanes_norm", "tod_sin", "tod_cos")
HIST, HOR = 12, 6


def ramp_history(panel, links, times, rday, attach, sgn: int = 1):
    """(n_links, n_times) on-ramp flow, off-ramp flow and an observed mask.

    The first version of this module took the ramp flow at the LAST history
    stamp and tiled it across all twelve steps, so the ramp channels were
    constant in time and the model could not see a ramp surge build -- the one
    thing the ramp layer is there to show. Kerner's nucleation argument is
    about the on-ramp flow RISING against the mainline, which a constant cannot
    express. This builds the real per-timestamp series, with a mask marking
    which (link, stamp) cells actually had a reading, and the caller slices
    only stamps at or before the forecast origin.
    """
    # `times`: the day's stamps; `rday`: the day's ramp rows (timestamp,
    # ramp_link_id, flow_vph), matched to `times` by exact value; `attach`:
    # the ramp attachment map. `sgn` is accepted and not used.
    # Returns three float32 (n_links x n_times) arrays: on-ramp flow and
    # off-ramp flow (veh/h, summed per attached mainline link) and a 0/1 mask
    # of cells with a reading.
    li = {k: i for i, k in enumerate(map(str, links))}
    n, T = len(links), len(times)
    on = np.zeros((n, T), np.float32)
    off = np.zeros((n, T), np.float32)
    seen = np.zeros((n, T), np.float32)
    if rday is None or len(rday) == 0:
        return on, off, seen
    amap = {str(r.ramp_link_id): (li.get(str(r.nearest_mainline_link_id)),
                                  str(r.ramp_type))
            for r in attach.itertuples(index=False)}
    ti = {t: i for i, t in enumerate(times)}
    rl = rday.ramp_link_id.astype(str).to_numpy()
    fv = pd.to_numeric(rday.flow_vph, errors="coerce").to_numpy()
    tcol = np.array([ti.get(t, -1) for t in rday.timestamp.to_numpy()])
    for k in range(len(rl)):
        c = tcol[k]
        if c < 0:
            continue
        hit = amap.get(rl[k])
        if hit is None or hit[0] is None:
            continue
        j, kind = hit
        f = fv[k]
        if not np.isfinite(f):
            continue
        if kind == "OR":
            on[j, c] += f
        else:
            off[j, c] += f
        seen[j, c] = 1.0
    return on, off, seen


def window_tensor(V, Qf, O, E, t: int, vcut, static, ramps,
                  ramp_hist=None, n_times: int = 288) -> np.ndarray:
    """(n_links, HIST, C) float32 for one window. Columns t-HIST..t-1 only."""
    # V, Qf, O, E: the day's (link x stamp) arrays; `t`: grid column of the
    # forecast time T. `static`: task2feat.static_table; `ramps`: the
    # task2feat.ramp_vectors dict, used only when `ramp_hist` (ramp_history
    # output) is None. The time-of-day channels assume column 0 is midnight
    # and n_times stamps per day. Missing values become 0.
    hs, he = t - HIST, t
    v = V[:, hs:he]
    q = Qf[:, hs:he]
    o = O[:, hs:he]
    e = E[:, hs:he].astype(np.float32)
    with np.errstate(invalid="ignore"):
        margin = v / vcut[:, None]
    n, T = v.shape
    def tile(x):
        """Repeats a per-link vector across the history stamps."""
        return np.repeat(np.asarray(x, dtype=np.float32)[:, None], T, axis=1)
    cap = static["capacity"].to_numpy(dtype=np.float32)
    vf = static["free_speed"].to_numpy(dtype=np.float32)
    ln = static["length_km"].to_numpy(dtype=np.float32)
    la = static["lanes"].to_numpy(dtype=np.float32)
    if ramp_hist is None:                       # legacy constant-ramp behaviour
        on_h = tile(np.nan_to_num(ramps["onramp_flow"] / 2000.0))
        off_h = tile(np.nan_to_num(ramps["offramp_flow"] / 2000.0))
        msk = np.ones_like(on_h)
    else:
        ON, OFF, SEEN = ramp_hist
        on_h = np.nan_to_num(ON[:, hs:he] / 2000.0)
        off_h = np.nan_to_num(OFF[:, hs:he] / 2000.0)
        msk = SEEN[:, hs:he]
    # absolute clock of each history stamp, so the net sees when the window is
    cols = (np.arange(hs, he) % n_times).astype(np.float32) * (1440.0 / n_times)
    ang = 2.0 * np.pi * cols / 1440.0
    out = np.stack([
        np.nan_to_num(v / 100.0, nan=0.0),
        np.nan_to_num(q / 2000.0, nan=0.0),
        np.nan_to_num(o, nan=0.0),
        e,
        np.nan_to_num((v <= vcut[:, None]).astype(np.float32), nan=0.0),
        np.nan_to_num(margin, nan=0.0),
        on_h, off_h, msk,
        tile(np.nan_to_num(cap / 8000.0)),
        tile(np.nan_to_num(vf / 120.0)),
        tile(np.nan_to_num(ln / 1.0)),
        tile(np.nan_to_num(la / 5.0)),
        np.broadcast_to(np.sin(ang)[None, :], (n, T)),
        np.broadcast_to(np.cos(ang)[None, :], (n, T)),
    ], axis=-1).astype(np.float32)
    return out


def build():
    """Defines the network modules and returns the QueueNet class.

    torch is imported here so that the rest of the module loads without it.

    Returns:
        QueueNet, a torch module mapping (x, prior_logit) to (B, L, HOR)
        logits.
    """
    import torch
    import torch.nn as nn

    class CausalTCN(nn.Module):
        """Dilated causal convolution over time, applied per link."""

        def __init__(self, c_in: int, width: int, dilations=(1, 2, 4)):
            """Stacks one kernel-3 convolution with GELU per dilation."""
            super().__init__()
            self.blocks = nn.ModuleList()
            self.pads = list(dilations)
            c = c_in
            for d in dilations:
                self.blocks.append(nn.Sequential(nn.Conv1d(c, width, 3, dilation=d), nn.GELU()))
                c = width
            self.width = width

        def forward(self, x):                      # x (B*L, C, T)
            """Applies the blocks with left padding, so step t sees steps <= t."""
            for blk, d in zip(self.blocks, self.pads):
                x = blk(torch.nn.functional.pad(x, (2 * d, 0)))   # left pad = causal
            return x

    class DirectedChain(nn.Module):
        """Two one-sided convolutions along the link axis: one reading from
        upstream, one reading from downstream. The caller orients the array so
        index 0 is the most UPSTREAM link, which is what makes the two kernels
        mean what their names say on every corridor."""

        def __init__(self, width: int, k: int = 5):
            """Creates the upstream and downstream kernels and a 1x1 mixer."""
            super().__init__()
            self.k = k
            self.up = nn.Conv1d(width, width, k)        # pads on the low side
            self.dn = nn.Conv1d(width, width, k)
            self.mix = nn.Sequential(nn.Conv1d(3 * width, width, 1), nn.GELU())

        def forward(self, h):                       # h (B, W, L)
            """Returns h plus a mixed update from both one-sided kernels."""
            k = self.k - 1
            u = self.up(torch.nn.functional.pad(h, (k, 0)))   # sees lower index = upstream
            d = self.dn(torch.nn.functional.pad(h, (0, k)))   # sees higher index = downstream
            return h + self.mix(torch.cat([h, u, d], dim=1))

    class QueueNet(nn.Module):
        """Causal TCN per link, directed chain layers, then a 1x1 head.

        The head also receives the mean prior logit of each link, and the
        prior logit is added to the output as an offset.
        """

        def __init__(self, c_in: int, width: int = 64, n_spatial: int = 3, hor: int = HOR):
            """Sets the hidden width, number of chain layers and horizon."""
            super().__init__()
            self.tcn = CausalTCN(c_in, width)
            self.spatial = nn.ModuleList([DirectedChain(width) for _ in range(n_spatial)])
            self.norm = nn.GroupNorm(1, width)
            self.head = nn.Sequential(nn.Conv1d(width + 1, width, 1), nn.GELU(),
                                      nn.Conv1d(width, hor, 1))

        def forward(self, x, prior_logit):
            """x (B, L, T, C) -> (B, L, HOR) logits. prior_logit (B, L, HOR)."""
            B, L, T, C = x.shape
            h = x.permute(0, 1, 3, 2).reshape(B * L, C, T)
            h = self.tcn(h)[:, :, -1]                      # last causal step
            h = h.reshape(B, L, -1).permute(0, 2, 1)       # (B, W, L)
            h = self.norm(h)
            for blk in self.spatial:
                h = blk(h)
            pm = prior_logit.mean(dim=2).unsqueeze(1)      # (B, 1, L)
            y = self.head(torch.cat([h, pm], dim=1))       # (B, HOR, L)
            return y.permute(0, 2, 1) + prior_logit

    return QueueNet


def soft_iou_loss(logit, y, mask, eps: float = 1.0):
    """1 - soft IoU per window, averaged. The metric is a per-window IoU, and a
    per-cell BCE spends its gradient on the ~99.6% negative cells instead.

    Kept alongside BCE rather than replacing it: notes/14 E's boundary is that a
    per-window soft IoU is not row-additive, so it is a proxy that has to be
    measured, not a derivation of the metric.
    """
    # logit, y, mask: (B, L, HOR) tensors; mask is 1 on scored cells. eps
    # smooths the ratio, so a window with an empty union scores 0 loss.
    import torch
    p = torch.sigmoid(logit) * mask
    y = y * mask
    inter = (p * y).sum(dim=(1, 2))
    union = p.sum(dim=(1, 2)) + y.sum(dim=(1, 2)) - inter
    return (1.0 - (inter + eps) / (union + eps)).mean()
