"""Serve the speed-only correction on the official hole rows of each split, fitted ONLY on that split's pseudo holes.
  validation: fit on the pilot's 492 validation pseudo windows (round2/task1_trans/work/ph_validation_*)
  private   : fit on the private pseudo windows built here (work/ph_private_*)
Apply to work/off_rows_<split>.parquet (every eligible official hole cell), keep the key rows (hole state rows of the
parent), new speed = clip(parent speed + dv, 6.0, 120.75), flow = parent.
Also R2 = private leave-days-out (odd/even) on the private pseudo holes with the same recipe.
-> work/trans_state_rows_<split>.parquet, work/trans_state_rows.parquet (both splits, 128,465 rows), results/r6_serve.json"""
# Task 1 reproduction pipeline, step 6 (`pipeline/task1.sh`, log `t1_serve`).
# The correction dv comes from the recomputed hole-model features and is added
# to the parent's hole speed. pipeline/task1_state_csv.py writes
# trans_state_rows.parquet (the T1 layer) into the later parent CSVs; the HM25
# step (hole mixture) also loads the boosters work/fs_speed_<split>.txt.
import sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import serve_lib as S
from s2_cands import parent_state

# G2v2 v2_F hole rows: their submission_ids are the official hole cells.
G2F = S.ROOT / 'build/review_20260925/task1/g2v2F_state_rows.parquet'


def main():
    """Fits the correction per split, applies it to the official hole rows and writes the T1 rows and the report."""
    res = json.load(open(S.HERE / 'results' / 'r6_serve.json')) if (S.HERE / 'results' / 'r6_serve.json').exists() else {}
    holes = set(pd.read_parquet(G2F).submission_id.astype(np.int64))
    # ldo_dv is not used further.
    ldo_dv = {'validation': None}
    r1 = np.load(S.HERE / 'work' / 'r1_validation_ldo_pv.npy')
    outs = []
    for split in ('validation', 'private'):
        paths = [(S.PILOT if split == 'validation' else S.HERE) / 'work' / f'ph_{split}_{p}.parquet' for p in S.PANELS]
        D = S.load_ph(paths)
        r = {'fit_cells': len(D), 'fit_windows': int(D.wid.nunique()), 'fit_panel_days': int(D.pd_.nunique()),
             'fit_windows_per_panel': D.groupby('panel').wid.nunique().to_dict()}
        # leave-days-out on this split (validation: R1 already computed by s1 -> reuse its predictions; private: R2)
        if split == 'validation':
            pv_ldo = r1
        else:
            pv_ldo = S.ldo(D, clip=True)
            t = S.rel_table(D, pv_ldo); r['R2_private_ldo'] = t
            r['acceptance_D'] = bool(t['A']['rel_v'] <= -0.05 and t['B']['rel_v'] <= -0.05)
            print('R2 private LDO:', {k: (v['n'], v['windows'], round(v['mse_v_parent'], 3), round(v['rel_v'], 4)) for k, v in t.items() if isinstance(v, dict) and 'rel_v' in v},
                  t['boot_panelday'], t['panelhalves_speed_improve'], 'acceptance D', r['acceptance_D'], flush=True)
        r['heldout_dv'] = S.desc(pv_ldo - D.ch_v.to_numpy())
        # serve: fit on ALL pseudo cells of the split
        m = S.fit(D); m.save_model(str(S.HERE / 'work' / f'fs_speed_{split}.txt'))
        Of = pd.read_parquet(S.HERE / 'work' / f'off_rows_{split}.parquet'); S.derive(Of)
        Of['dv'] = S.correction(m, Of)
        # Parent hole rows of the split, each joined to its official-row
        # correction (every hole row must have one).
        ps = parent_state(split); ps = ps[ps.submission_id.isin(holes)]
        k = ps.merge(Of[['panel', 'timestamp', 'link_id', 'ch_v', 'ch_ql', 'dv', 'cond', 'window_id']], on=['panel', 'timestamp', 'link_id'],
                     how='left', validate='one_to_one')
        r['hole_rows'] = len(k); r['hole_rows_without_official_cell'] = int(k.dv.isna().sum())
        assert r['hole_rows_without_official_cell'] == 0
        r['parent_vs_recomputed_g2v2_speed_maxabs'] = float((k.speed_kmh - k.ch_v).abs().max())
        r['parent_vs_recomputed_g2v2_speed_p99abs'] = float(np.percentile((k.speed_kmh - k.ch_v).abs(), 99))
        k['speed_new'] = np.clip(k.speed_kmh.to_numpy() + k.dv.to_numpy(), S.SPEED_FLOOR, S.SPEED_CAP)
        r['official_dv_keyrows'] = S.desc(k.speed_new - k.speed_kmh)
        r['official_dv_allcells'] = S.desc(Of.dv)
        # Recorded check, not enforced: the 99th percentile of |dv| on the
        # official rows stays within 1.5 x the held-out one.
        r['abs_p99_ratio_official_vs_heldout'] = r['official_dv_keyrows']['abs_p99'] / r['heldout_dv']['abs_p99']
        r['integrity_p99_ok'] = bool(r['abs_p99_ratio_official_vs_heldout'] <= 1.5)
        r['official_dv_by_panel'] = k.assign(d=k.speed_new - k.speed_kmh).groupby('panel').d.agg(['size', 'mean', lambda x: float(np.abs(x).mean())]).rename(columns={'<lambda_0>': 'abs_mean'}).round(4).to_dict('index')
        r['official_dv_by_cond'] = k.assign(d=k.speed_new - k.speed_kmh).groupby('cond').d.agg(['size', 'mean']).round(4).to_dict('index')
        print(split, 'fit windows', r['fit_windows'], 'hole rows', len(k), 'dv keyrows', {a: round(b, 3) for a, b in r['official_dv_keyrows'].items()},
              'heldout', {a: round(b, 3) for a, b in r['heldout_dv'].items()}, 'p99 ratio', round(r['abs_p99_ratio_official_vs_heldout'], 3),
              'parent vs recomputed max', round(r['parent_vs_recomputed_g2v2_speed_maxabs'], 4), flush=True)
        o = k[['submission_id', 'speed_new', 'flow_vph']].rename(columns={'speed_new': 'speed_kmh'}).sort_values('submission_id')
        o.to_parquet(S.HERE / 'work' / f'trans_state_rows_{split}.parquet', index=False); outs.append(o)
        res[split] = r
    allr = pd.concat(outs, ignore_index=True).sort_values('submission_id')
    assert len(allr) == len(holes) and set(allr.submission_id) == holes
    allr.to_parquet(S.HERE / 'work' / 'trans_state_rows.parquet', index=False)
    res['rows_total'] = len(allr)
    json.dump(res, open(S.HERE / 'results' / 'r6_serve.json', 'w'), indent=1, default=float)
    print('rows', len(allr))


if __name__ == '__main__':
    main()
