"""Design specifications of the conditional-logit onset site choice.

Each spec names the design columns (see cl.design): `shared` columns have one
weight for all sites, `site_int` adds per-site intercepts and `site_slopes`
columns get one weight per site. SPECS are fitted per panel, PSPECS pooled
over the multi-site panels without site identity. s2_ruler.py imports SPECS,
PSPECS and lg; in the reproduction sv2_official_FEA1.py calls
s2_ruler.prep_rows, which uses lg.
"""
import numpy as np
import cl as CL
# BF: base load-ratio features; KR: demand-phase indicators kr1..kr5 (kr0 is
# the reference level).
BF = CL.BASEF; KR = [f'kr{k}' for k in range(1, 6)]
# GAPS: gap to the best other site of the window; RKS: rank in the window.
GAPS = [f'gap_{f}' for f in CL.GAPF]; RKS = ['rk_r6', 'rk_rmx3']
SPECS = {
    'C1_base_kr_int': dict(shared=BF + KR, site_int=True),
    'C2_+gaps': dict(shared=BF + KR + GAPS + RKS, site_int=True),
    'C3_+gaps_todwk': dict(shared=BF + KR + GAPS + RKS, site_int=True, site_slopes=['tod', 'wkend']),
    'C4_stack_u_gaps': dict(shared=['ulogit'] + GAPS + RKS, site_int=True),
    'C4b_stack_u_gaps_kr': dict(shared=['ulogit'] + KR + GAPS + RKS, site_int=True),
    'C6_siteslopes': dict(site_int=True, site_slopes=BF + KR),
}
PSPECS = {  # pooled over the 4 multi-site panels, no site identity
    'P1_noID_base_kr': dict(shared=BF + KR),
    'P1b_noID_+gaps': dict(shared=BF + KR + GAPS + RKS),
}


# Returns logit(u), with u clipped to [1e-6, 1 - 1e-6] (ulogit feature).
def lg(u): u = np.clip(u, 1e-6, 1 - 1e-6); return np.log(u / (1 - u))
