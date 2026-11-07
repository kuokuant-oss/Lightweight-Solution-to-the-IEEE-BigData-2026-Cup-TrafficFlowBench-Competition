import numpy as np
import cl as CL
BF = CL.BASEF; KR = [f'kr{k}' for k in range(1, 6)]
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


def lg(u): u = np.clip(u, 1e-6, 1 - 1e-6); return np.log(u / (1 - u))
