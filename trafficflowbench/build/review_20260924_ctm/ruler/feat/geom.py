"""Queue-geometry / shock features from a 12-stamp history (T-60..T-5).

Convention (measured on all 8 Task 2 corridors): queues grow toward LOWER link
index (lower milepost) from a head at the high-index end, so 'upstream' is the
lower index everywhere, which is what the legacy features (sgn=+1) assume too.
"""
#
# In this file a "block" is a contiguous stretch of queued links at one stamp (tail = lowest index = upstream
# end, head = highest index). Role: imported by build_geom.py (reproduction pipeline, ongoing.sh).
import numpy as np, pandas as pd

# HOR: number of horizon steps (6 x 5 minutes).
HOR = 6


def _blocks(q, gap=2):
    """Contiguous runs of True along axis 0, merging interior gaps <= gap. Returns list of (tail, head)."""
    idx = np.where(q)[0]
    if len(idx) == 0: return []
    out = []; s = idx[0]; p = idx[0]
    for i in idx[1:]:
        if i - p - 1 <= gap: p = i; continue
        out.append((s, p)); s = i; p = i
    out.append((s, p)); return out


def geom_features(V, Qf, E, vcut, length_km):
    """V, Qf: (L, H) speed / flow history, NaN = unobserved. Returns (L, HOR) dict of arrays."""
    # Args: vcut per-link queue speed (km/h); length_km per-link length. Each returned array has one value per
    # link (shape (L,)); step_frame repeats it over the HOR steps. Features of a link, taken from the queue nearest
    # downstream of it, or from the queue that contains it:
    #   g_gap_km / g_gap_links  distance from the link's downstream end to that queue's tail
    #   g_tail_vel              tail speed from a line fit over the last 7 stamps (km per 5 min, < 0 = growing
    #                           upstream)
    #   g_rh_w                  shock-wave speed (km/h) from flow and density just upstream of and inside the queue
    #   g_eta_rh / g_eta_vel    5-minute steps until the tail reaches the link at g_rh_w / g_tail_vel
    #   g_clear_eta_rh          steps until the queue clears the link when the tail moves downstream (99 otherwise)
    #   g_block_len_km/_links, g_pos_in_block_km, g_in_block, g_n_blocks  queue size, position and count
    #   g_head_age, g_tail_age, g_link_age  stamps the queue / tail link / link itself has been queued
    #   g_q_up, g_q_in          flow just upstream of the tail and inside the queue (veh/h)
    L, H = V.shape
    # Unobserved speeds are interpolated along time; a link with no observation counts as free flow.
    Vi = pd.DataFrame(V).T.interpolate(limit_direction='both').T.to_numpy()  # along time
    q = np.nan_to_num(Vi, nan=np.inf) <= vcut[:, None]
    xs = np.concatenate([[0.0], np.cumsum(length_km)])  # x_start[l]=xs[l], x_end[l]=xs[l+1]
    # tail position per stamp for block containing / nearest the final heads
    fin = _blocks(q[:, -1])
    hist_blocks = [_blocks(q[:, h]) for h in range(H)]
    feats = {k: np.full(L, np.nan) for k in ['g_gap_km', 'g_gap_links', 'g_tail_vel', 'g_rh_w', 'g_eta_rh', 'g_eta_vel',
                                              'g_block_len_km', 'g_block_links', 'g_head_age', 'g_in_block', 'g_pos_in_block_km',
                                              'g_clear_eta_rh', 'g_n_blocks', 'g_link_age', 'g_q_up', 'g_q_in', 'g_tail_age']}
    feats['g_n_blocks'][:] = len(fin)
    # link continuous-queued age at the last stamp
    age = np.zeros(L)
    for h in range(H - 1, -1, -1):
        still = q[:, h] & (age == (H - 1 - h))
        age[still] += 1
    feats['g_link_age'][:] = age
    feats['g_in_block'][:] = 0
    # Mean flow and speed over the last 3 stamps (T-15..T-5).
    Qm = np.nanmean(Qf[:, -3:], axis=1); Vm = np.nanmean(Vi[:, -3:], axis=1)
    for (tail, head) in fin:
        # history of this block's tail: for each stamp, block whose head is within 3 links of this head
        tails = []
        for h in range(H):
            cand = [b for b in hist_blocks[h] if abs(b[1] - head) <= 3 or (b[0] <= head <= b[1])]
            tails.append(min(b[0] for b in cand) if cand else np.nan)
        tails = np.array(tails, float)
        # Tail speed: slope of tail position (km) over the last 7 stamps, when at least 3 have a tail.
        ok = ~np.isnan(tails[-7:])
        vel = np.nan
        if ok.sum() >= 3:
            hh = np.arange(7)[ok]; xt = xs[tails[-7:][ok].astype(int)]
            vel = np.polyfit(hh, xt, 1)[0]  # km per 5 min (negative = tail moving upstream)
        head_age = int(np.sum(~np.isnan(tails)))
        tail_age = age[tail]
        # Upstream state: up to 3 links before the tail; queue state: the tail and up to 2 links after it.
        up = slice(max(0, tail - 3), tail); dn = slice(tail, min(head + 1, tail + 3))
        q_u = np.nanmean(Qm[up]) if tail > 0 else np.nan; v_u = np.nanmean(Vm[up]) if tail > 0 else np.nan
        q_c = np.nanmean(Qm[dn]); v_c = np.nanmean(Vm[dn])
        # Shock-wave speed w = (q_c - q_u) / (k_c - k_u) with densities k = q / v (veh/km).
        k_u = q_u / v_u if v_u and v_u > 0 else np.nan; k_c = q_c / v_c if v_c and v_c > 0 else np.nan
        w = (q_c - q_u) / (k_c - k_u) if np.isfinite(k_c - k_u) and abs(k_c - k_u) > 1e-6 else np.nan  # km/h
        blen = xs[head + 1] - xs[tail]
        # links upstream of this block (closer than any other block's tail): assign if nearer
        for l in range(0, tail):
            gap_km = xs[tail] - xs[l + 1]
            if np.isnan(feats['g_gap_km'][l]) or gap_km < feats['g_gap_km'][l]:
                feats['g_gap_km'][l] = gap_km; feats['g_gap_links'][l] = tail - l
                feats['g_tail_vel'][l] = vel; feats['g_rh_w'][l] = w
                # x 12 turns hours into 5-minute steps.
                feats['g_eta_rh'][l] = gap_km / max(-w, 0.5) * 12 if np.isfinite(w) else np.nan
                feats['g_eta_vel'][l] = gap_km / max(-vel, 0.01) if np.isfinite(vel) else np.nan
                feats['g_block_len_km'][l] = blen; feats['g_block_links'][l] = head - tail + 1
                feats['g_head_age'][l] = head_age; feats['g_tail_age'][l] = tail_age
                feats['g_q_up'][l] = q_u; feats['g_q_in'][l] = q_c
        for l in range(tail, head + 1):
            feats['g_in_block'][l] = 1; feats['g_pos_in_block_km'][l] = xs[l + 1] - xs[tail]
            feats['g_tail_vel'][l] = vel; feats['g_rh_w'][l] = w; feats['g_block_len_km'][l] = blen
            feats['g_block_links'][l] = head - tail + 1; feats['g_head_age'][l] = head_age; feats['g_tail_age'][l] = tail_age
            feats['g_q_up'][l] = q_u; feats['g_q_in'][l] = q_c
            feats['g_clear_eta_rh'][l] = (xs[l + 1] - xs[tail]) / max(w, 0.5) * 12 if np.isfinite(w) and w > 0 else 99.0
    return feats


def step_frame(links, feats):
    """Returns one row per link and horizon step 1..HOR, with the ETA features minus the step."""
    rows = []
    base = pd.DataFrame(feats); base['link_id'] = links
    for s in range(1, HOR + 1):
        f = base.copy(); f['step'] = s
        f['g_eta_rh_m_step'] = f.g_eta_rh - s; f['g_eta_vel_m_step'] = f.g_eta_vel - s
        f['g_clear_m_step'] = f.g_clear_eta_rh - s
        rows.append(f)
    return pd.concat(rows, ignore_index=True)
