"""R0 + R1: reproduce the parent / round-2 numbers on the validation pseudo holes, then the served recipe (speed only, clip)
in odd/even day halves.  Reads the pilot's pseudo-hole tables read-only.  -> results/r1_validation_ldo.json"""
import sys, json
sys.dont_write_bytecode = True
import numpy as np
import serve_lib as S

D = S.load_ph([S.PILOT / 'work' / f'ph_validation_{p}.parquet' for p in S.PANELS])
print('validation pseudo cells', len(D), 'windows', D.wid.nunique(), 'panel-days', D.pd_.nunique(), flush=True)
res = {'cells': len(D), 'windows': int(D.wid.nunique())}
# R0: parent MSE per half, round-2 FS (unclipped) reproduction
r0 = {}
for h in ('A', 'B'):
    m = (D.half == h).to_numpy(); r0[f'mse_v_parent_{h}'] = float(((D.ch_v - D.y_v) ** 2)[m].mean())
pv_u = S.ldo(D, clip=False); t = S.rel_table(D, pv_u, boot=False)
r0.update(fs_unclipped_rel_v_A=t['A']['rel_v'], fs_unclipped_rel_v_B=t['B']['rel_v'])
ok = abs(r0['mse_v_parent_A'] - 48.5156) < 1e-3 and abs(r0['mse_v_parent_B'] - 45.9596) < 1e-3 and \
     abs(r0['fs_unclipped_rel_v_A'] + 0.2303) < 5e-4 and abs(r0['fs_unclipped_rel_v_B'] + 0.2515) < 5e-4
r0['reproduced'] = bool(ok); res['R0'] = r0
print('R0', {k: round(v, 4) if isinstance(v, float) else v for k, v in r0.items()}, flush=True)
assert ok, 'R0 reproduction failed'
# R1: served recipe (clip) -- the pre-registered ruler
pv = S.ldo(D, clip=True); t = S.rel_table(D, pv)
res['R1'] = t
res['clip_changed_rows'] = int(np.sum(np.abs(pv - pv_u) > 1e-9))
res['heldout_dv'] = S.desc(pv - D.ch_v.to_numpy())
print('R1 served (clip):', {k: (v['n'], round(v['mse_v_parent'], 3), round(v['rel_v'], 4)) for k, v in t.items() if isinstance(v, dict) and 'rel_v' in v},
      t['boot_panelday'], t['panelhalves_speed_improve'], 'clip changed rows', res['clip_changed_rows'], flush=True)
res['acceptance_A'] = bool(t['A']['rel_v'] <= -0.15 and t['B']['rel_v'] <= -0.15)
print('acceptance A (<= -15% both halves):', res['acceptance_A'])
np.save(S.HERE / 'work' / 'r1_validation_ldo_pv.npy', pv)
json.dump(res, open(S.HERE / 'results' / 'r1_validation_ldo.json', 'w'), indent=1)
