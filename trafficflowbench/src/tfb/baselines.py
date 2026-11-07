"""Task 1 baseline families, each implemented from a source in doc/reference/notes.
Nothing here is invented: every method points at the note it came from.

  B0 historical mean   official baseline_task1_historical_mean.py          [官方]
  B1 two-axis interp   notes/01 D, the public competitor's feature recipe  [選手]
  B2 ASM               notes/01 A, Treiber/Kesting/Wilson arXiv:0909.4467  [文獻]
  B3 LRTC-TNN          notes/08 B, transdim imputer/LRTC-TNN.ipynb         [文獻]
"""
#
# Role in the package: the Task 1 reproduction pipeline (pipeline/task1.sh,
# steps 1 and 2) calls lrtc_tnn directly, and two_axis runs inside
# task1stack.members. asm (B2) is not called by any program in this package.
#
# Names in the docstring above: B0..B3 are the family codes that
# predict1.METHODS also uses. The bracketed tags give the kind of source:
# [官方] the organizers' starter code, [選手] a public competitor's feature
# recipe, [文獻] a published paper or its reference code. "notes/NN X" and
# "doc/..." point to the team's development notes, which are not part of this
# package. The historical mean (B0) is computed in predict1.historical_mean.
#
# lrtc_tnn and its three helpers are copied from Xinyu Chen's transdim project
# (imputer/LRTC-TNN.ipynb, MIT licence; see THIRD_PARTY_NOTICES.md).
from __future__ import annotations

import numpy as np

DT_H = 5.0 / 60.0  # 5-minute grid, in hours


# ---------------------------------------------------------------- B1 interp
def interp_axis(value: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    """Linear interpolation along one axis over the observed cells, edges held flat."""
    # value is a (link x time) grid with NaN at unobserved cells. axis=0
    # interpolates each time column across links at positions `coord`
    # (mileposts, km); axis=1 interpolates each link's row in time (`coord` =
    # step index). np.interp holds the nearest observed value beyond the first
    # and last observation; a line with no observation at all stays NaN.
    v = value if axis == 0 else value.T
    x = coord
    out = np.full_like(v, np.nan)
    for j in range(v.shape[1]):
        col = v[:, j]
        ok = np.isfinite(col)
        if ok.sum() == 0:
            continue
        out[:, j] = np.interp(x, x[ok], col[ok])
    return out if axis == 0 else out.T


def two_axis(value: np.ndarray, milepost: np.ndarray) -> dict[str, np.ndarray]:
    """notes/01 D: `spat_*` interpolates along space, `temp_*` along time."""
    # value: (link x time) grid with NaN at unobserved cells; milepost: link
    # positions in km, ascending. Returns three (link x time) fills: "spatial",
    # "temporal", and "mean", their cell-wise average (NaN only where both are).
    t_index = np.arange(value.shape[1], dtype=float)
    spat = interp_axis(value, milepost, axis=0)
    temp = interp_axis(value, t_index, axis=1)
    both = np.nanmean(np.stack([spat, temp]), axis=0)
    return {"spatial": spat, "temporal": temp, "mean": both}


# ---------------------------------------------------------------- B2 ASM
def _asm_field(value: np.ndarray, observed: np.ndarray, milepost: np.ndarray,
               sigma: float, tau: float, c_kmh: float, n_sigma: float = 4.0) -> np.ndarray:
    """One ASM characteristic field.

    V_c(x,t) = sum_i phi(x-x_i, t-t_i-(x-x_i)/c) v_i / sum_i phi(...)
    phi(x,t) = exp(-(|x|/sigma + |t|/tau)),  arXiv:0909.4467.

    Distances are the real mileposts (km): this corridor's link spacing runs
    0.01-3.08 km, so an index-based kernel would be a different method.
    Truncated at n_sigma decay lengths in both arguments (weight < e^-4).
    """
    # value / observed: (link x time) grid and its observed mask; milepost in
    # km. sigma (km) and tau (h) are the kernel decay lengths and c_kmh the
    # speed of the characteristic: positive for free-flow information carried
    # downstream, negative for congested waves carried upstream. Returns the
    # smoothed field, NaN where no observation lies inside the truncated kernel.
    nL, nT = value.shape
    v = np.where(observed, np.nan_to_num(value), 0.0)
    m = observed.astype(float)
    num = np.zeros((nL, nT))
    den = np.zeros((nL, nT))

    spacing = np.diff(milepost)
    typical = float(np.median(spacing)) if len(spacing) else 1.0
    max_d = max(1, int(np.ceil(n_sigma * sigma / max(typical, 1e-6))))
    max_d = min(max_d, nL - 1)
    # A characteristic crossing the spatial window shifts time by dx/c.
    max_shift_h = (n_sigma * sigma) / abs(c_kmh)
    max_u = int(np.ceil((n_sigma * tau + max_shift_h) / DT_H))

    # Sum the kernel over every link offset d (source row = target row + d)
    # and time offset u (in 5-minute steps), shifting whole slices at once.
    for d in range(-max_d, max_d + 1):
        lo_t, hi_t = max(0, -d), min(nL, nL - d)       # target rows
        src = slice(lo_t + d, hi_t + d)                # source rows
        tgt = slice(lo_t, hi_t)
        dx = milepost[tgt] - milepost[src]             # x_target - x_source, km
        if np.all(np.abs(dx) > n_sigma * sigma):
            continue
        for u in range(-max_u, max_u + 1):
            dtp = u * DT_H - dx / c_kmh                # t - t_i - (x-x_i)/c
            w = np.exp(-(np.abs(dx) / sigma + np.abs(dtp) / tau))
            w[np.abs(dx) > n_sigma * sigma] = 0.0
            w[np.abs(dtp) > n_sigma * tau] = 0.0
            if not w.any():
                continue
            lo_c, hi_c = max(0, -u), min(nT, nT - u)
            cs, ct = slice(lo_c + u, hi_c + u), slice(lo_c, hi_c)
            num[tgt, ct] += w[:, None] * v[src, cs]
            den[tgt, ct] += w[:, None] * m[src, cs]
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def asm(speed: np.ndarray, observed: np.ndarray, milepost: np.ndarray,
        other: dict[str, np.ndarray] | None = None,
        sigma: float | None = None, tau: float | None = None,
        c_free: float = 70.0, c_cong: float = -15.0,
        v_thr: float = 60.0, dv: float = 20.0) -> dict[str, np.ndarray]:
    """Adaptive Smoothing Method.

    Defaults are Table 1 of arXiv:0909.4467: sigma = dx/2, tau = dt/2,
    c_free = 70, c_cong = -15 km/h, V_thr = 60, dV = 20 km/h.

    `other` carries further fields (flow, occupancy). The paper's prescription
    for them: smooth with the same two kernels, but keep the nonlinear weight
    w on the velocity fields -- "use alternative data in the linear formulae
    but retain unchanged the dependence on velocity in the nonlinear component".
    """
    # speed: (link x time) speed grid, km/h; observed: its observed mask;
    # milepost: link positions, km. sigma defaults to half the median link
    # spacing and tau to half a 5-minute step. Returns {"speed_kmh": field}
    # plus one smoothed field per entry of `other`, all (link x time).
    spacing = np.diff(milepost)
    if sigma is None:
        sigma = float(np.median(spacing)) / 2.0 if len(spacing) else 0.5
    if tau is None:
        tau = DT_H / 2.0
    v_free = _asm_field(speed, observed, milepost, sigma, tau, c_free)
    v_cong = _asm_field(speed, observed, milepost, sigma, tau, c_cong)
    both = np.stack([v_free, v_cong])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vmin = np.nanmin(both, axis=0)
    # Congestion weight from the slower of the two velocity fields: close to 1
    # below v_thr (congested regime), close to 0 above it, over a width of dv.
    w = 0.5 * (1.0 + np.tanh((v_thr - vmin) / dv))
    w = np.where(np.isfinite(vmin), w, np.nan)

    def blend(free, cong):
        """Returns w * cong + (1 - w) * free, falling back to the finite field."""
        out = w * cong + (1.0 - w) * free
        out = np.where(np.isfinite(out), out, np.where(np.isfinite(free), free, cong))
        return out

    result = {"speed_kmh": blend(v_free, v_cong)}
    for name, field in (other or {}).items():
        q_free = _asm_field(field, observed, milepost, sigma, tau, c_free)
        q_cong = _asm_field(field, observed, milepost, sigma, tau, c_cong)
        result[name] = blend(q_free, q_cong)   # same w, from the velocity fields
    return result


# ---------------------------------------------------------------- B3 LRTC-TNN
def _ten2mat(tensor, mode):
    """Unfolds a tensor along `mode` into a matrix (Fortran order, as transdim)."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order="F")


def _mat2ten(mat, tensor_size, mode):
    """Folds a mode unfolding back to shape `tensor_size`; inverse of _ten2mat."""
    index = [mode] + [i for i in range(tensor_size.shape[0]) if i != mode]
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order="F"), 0, mode)


def _svt_tnn(mat, alpha, rho, theta):
    """Applies singular value thresholding for the truncated nuclear norm.

    Keeps only singular values above tau = alpha / rho; the largest `theta`
    of them stay unchanged and the rest are reduced by tau. A wide matrix
    (2m < n) is decomposed through its m x m Gram matrix and a tall one
    (m > 2n) through its transpose, which is cheaper than a full SVD.

    Args:
        mat: Matrix to threshold (one tensor unfolding).
        alpha: Weight of this unfolding in the objective.
        rho: Current ADMM penalty parameter.
        theta: Number of leading singular values left unshrunk.

    Returns:
        The thresholded matrix, same shape as `mat`.
    """
    tau = alpha / rho
    m, n = mat.shape
    if 2 * m < n:
        u, s, _ = np.linalg.svd(mat @ mat.T, full_matrices=0)
        s = np.sqrt(np.maximum(s, 0.0))
        idx = int(np.sum(s > tau))
        mid = np.zeros(idx)
        mid[:theta] = 1
        mid[theta:idx] = (s[theta:idx] - tau) / np.maximum(s[theta:idx], 1e-12)
        return (u[:, :idx] @ np.diag(mid)) @ (u[:, :idx].T @ mat)
    elif m > 2 * n:
        return _svt_tnn(mat.T, alpha, rho, theta).T
    u, s, v = np.linalg.svd(mat, full_matrices=0)
    idx = int(np.sum(s > tau))
    vec = s[:idx].copy()
    vec[theta:idx] = s[theta:idx] - tau
    return u[:, :idx] @ np.diag(vec) @ v[:idx, :]


def lrtc_tnn(sparse_tensor, alpha=None, rho=1e-5, theta=0.25, epsilon=1e-4, maxiter=100):
    """transdim imputer/LRTC-TNN.ipynb, verbatim. Missing entries are 0, not NaN."""
    # sparse_tensor: (link x time x day) array, 0 at missing cells; an observed
    # value equal to 0 also counts as missing. alpha: weights of the three
    # unfoldings (default 1/3 each). rho: starting ADMM penalty, raised 5% per
    # iteration up to 1e5. theta: share of each mode's size whose leading
    # singular values stay unshrunk. Stops when the relative change of the
    # estimate falls below epsilon, or after maxiter iterations. Returns the
    # low-rank estimate for every cell; observed cells are not reset to their
    # input values.
    alpha = np.ones(3) / 3 if alpha is None else np.asarray(alpha, dtype=float)
    dim = np.array(sparse_tensor.shape)
    pos_missing = np.where(sparse_tensor == 0)
    X = np.zeros(np.insert(dim, 0, len(dim)))
    T = np.zeros(np.insert(dim, 0, len(dim)))
    Z = sparse_tensor.copy()
    last = sparse_tensor.copy()
    snorm = np.sqrt(np.sum(sparse_tensor ** 2))
    it = 0
    while True:
        rho = min(rho * 1.05, 1e5)
        # ADMM step: low-rank update of each mode unfolding, then fill the
        # missing cells of Z with the average estimate, then the dual update.
        for k in range(len(dim)):
            X[k] = _mat2ten(_svt_tnn(_ten2mat(Z - T[k] / rho, k), alpha[k], rho,
                                     int(np.ceil(theta * dim[k]))), dim, k)
        Z[pos_missing] = np.mean(X + T / rho, axis=0)[pos_missing]
        T = T + rho * (X - np.broadcast_to(Z, np.insert(dim, 0, len(dim))))
        hat = np.einsum("k, kmnt -> mnt", alpha, X)
        tol = np.sqrt(np.sum((hat - last) ** 2)) / snorm
        last = hat.copy()
        it += 1
        if tol < epsilon or it >= maxiter:
            break
    return hat
