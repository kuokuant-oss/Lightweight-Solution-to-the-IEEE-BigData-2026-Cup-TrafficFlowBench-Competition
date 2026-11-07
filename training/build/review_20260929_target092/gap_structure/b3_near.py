"""B3: add nearest-on-ramp features; retrain the Jan residual speed model and test on the Feb official-like gaps.
Also the reverse direction (fit on Feb 80 windows' v2_F residuals, test on Jan v2_J) as a second, independent check.
Pre-declared combinations with the served HMP50: HMP50 + lam*corr (lam .5, 1) and AVG = .5*HMP50 + .5*(v2 + corr)."""
# Training record for the January hole speed correction c_Jan. The reproduction does not
# run this program; it loads the shipped model. The program reads row tables that are not
# shipped, and its paths resolve from its original place in the development repository.
#
# Trains LightGBM residual models of the G2v2 hole prediction from the on-ramp and off-ramp
#   counts inside the hole (ramp detectors are not blanked; a mainline queue throttles its
#   on-ramps) and the nearest on-ramps.
# Data: `b1_jan.parquet` (January 2031 holes at the corrected replicated windows, 50% cell
#   sample; base = G2v2 v2_J, which did not see January but was fitted with February) and
#   `b1_feb.parquet` (February 2031 holes at the 80 official-like windows; base = v2_F),
#   both built by `b1_build.py` (not shipped). `ramp_feats.attach_near` adds the
#   nearest-on-ramp inputs; the results are cached as `b3_jan.parquet` / `b3_feb.parquet`.
#   Targets: `r_v` = true speed - base speed (km/h); `r_q` = true per-lane flow - base
#   (veh/h per lane).
# Settings (`b2_ramp_resid.PAR`, `NR`): objective l2, learning rate .05, 31 leaves,
#   min_data_in_leaf 500, feature and bagging fraction .8, lambda_l2 10, seed 5, 2 threads,
#   500 rounds, no early stopping.
# Shipped: `b3_sp_M2_janfit.txt` (c_Jan: January speed residual model, inputs SP +
#   RAMP_FEATS + NEAR_FEATS = 24 + 16 + 17 = 57) under
#   `trafficflowbench/build/review_20260929_target092/gap_structure/`. The other January fits
#   (`b3_sp_M1_janfit.txt`, `b3_sp_N_janfit.txt`, `b3_fl_M2_janfit.txt`) and the February fits
#   of the reverse check are not shipped.
# Other output: `b3_results.json` (February scores and the January reverse check) and
#   `b3_feb_preds.parquet`.
# Invocation: `python b3_near.py`. It imports `ramp_feats` and `b2_ramp_resid` from its own
#   folder (both shipped next to the original).
import os, sys, json, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ramp_feats as RF
from b2_ramp_resid import SP, FL, PAR, NR, rel, add_phys, KEY, FLOOR, CAP, PHYS


def main():
    """Builds the near-ramp tables, fits the January residual models and scores them on February."""
    out = {}
    for nm in ('jan', 'feb'):
        p = HERE / f'b3_{nm}.parquet'
        if not p.exists():
            x = pd.read_parquet(HERE / f'b1_{nm}.parquet'); x = RF.attach_near(x.reset_index(drop=True), 'train'); x.to_parquet(p, index=False)
            print('built', nm, {c: round(float(x[c].notna().mean()), 3) for c in ['nOR1_prof', 'uOR1_prof', 'cOR_prof']}, flush=True)
            del x; gc.collect()
    jan = add_phys(pd.read_parquet(HERE / 'b3_jan.parquet')); feb = add_phys(pd.read_parquet(HERE / 'b3_feb.parquet'))
    # February rows of the served hole speed (`serve_hmp50`) and flow (`ch_ql`), for comparison.
    a1 = pd.read_parquet(HERE / 'a1_feb_rows.parquet', columns=KEY + ['serve_hmp50', 'ch_ql', 'ch_v', 'lclass'])
    feb = feb.merge(a1, on=KEY, how='left', validate='one_to_one')
    for d in (jan, feb):
        d['r_v'] = d.y_v - d.p_v; d['r_q'] = d.y_ql - d.p_ql
    # Input sets: speed residual with in-hole ramp counts (M1), plus nearest on-ramps (M2,
    # shipped as c_Jan), nearest on-ramps only (N); flow residual with M2 inputs + physics (fl_M2).
    specs = {'sp_M1': SP + RF.RAMP_FEATS, 'sp_M2': SP + RF.RAMP_FEATS + RF.NEAR_FEATS, 'sp_N': SP + RF.NEAR_FEATS,
             'fl_M2': FL + RF.RAMP_FEATS + RF.NEAR_FEATS + PHYS}
    imps = {}
    for nm, feats in specs.items():
        tgt = 'r_q' if nm.startswith('fl') else 'r_v'
        ok = jan[tgt].notna()
        m = lgb.train(PAR, lgb.Dataset(jan.loc[ok, feats], jan.loc[ok, tgt]), NR)
        feb['c_' + nm] = m.predict(feb[feats], num_threads=2); m.save_model(str(HERE / f'b3_{nm}_janfit.txt'))
        imp = pd.Series(m.feature_importance('gain'), index=feats); imps[nm] = (imp / imp.sum()).sort_values(ascending=False).round(4).head(12).to_dict()
        # reverse: fit on Feb (all 80 windows), test on Jan
        okf = feb[tgt].notna()
        m2 = lgb.train(PAR, lgb.Dataset(feb.loc[okf, feats], feb.loc[okf, tgt]), NR)
        jan['c_' + nm] = m2.predict(jan[feats], num_threads=2)
        print('fit', nm, flush=True); del m, m2; gc.collect()
    t = feb[feb.is_target & feb.y_v.notna() & feb.y_ql.notna()].copy()
    tj = jan[jan.is_target & jan.y_v.notna() & jan.y_ql.notna()].copy(); tj['lclass'] = 'na'
    for nm in ('sp_M1', 'sp_M2', 'sp_N'):
        t[nm + '_clean'] = np.clip(t.p_v + t['c_' + nm], FLOOR, CAP); out['feb_' + nm + '_clean'] = rel(t, 'y_v', 'p_v', nm + '_clean')
        tj[nm + '_clean'] = np.clip(tj.p_v + tj['c_' + nm], FLOOR, CAP); out['janrev_' + nm + '_clean'] = rel(tj, 'y_v', 'p_v', nm + '_clean')
        for lam in (0.5, 1.0):
            c = f'{nm}_hmp_{lam}'; t[c] = np.clip(t.serve_hmp50 + lam * t['c_' + nm], FLOOR, CAP); out['feb_' + c] = rel(t, 'y_v', 'serve_hmp50', c)
        c = f'{nm}_avg'; t[c] = 0.5 * t.serve_hmp50 + 0.5 * t[nm + '_clean']; out['feb_' + c] = rel(t, 'y_v', 'serve_hmp50', c)
        t[nm + '_vsHMP_clean'] = t[nm + '_clean']; out['feb_' + nm + '_clean_vs_hmp'] = rel(t, 'y_v', 'serve_hmp50', nm + '_clean')
    t['fl_M2_served'] = np.clip(t.ch_ql + t.c_fl_M2, 0, None); out['feb_fl_M2_served'] = rel(t, 'y_ql', 'ch_ql', 'fl_M2_served')
    tj['fl_M2_clean'] = np.clip(tj.p_ql + tj.c_fl_M2, 0, None); out['janrev_fl_M2_clean'] = rel(tj, 'y_ql', 'p_ql', 'fl_M2_clean')
    for k_, v in out.items():
        print(k_, json.dumps({kk: v[kk] for kk in ('rmse0', 'rmse1', 'rel', 'ci', 'onset', 'ongoing', 'mixed', 'panels_improved')}), flush=True)
    out['importance'] = imps; print(json.dumps(imps))
    json.dump(out, open(HERE / 'b3_results.json', 'w'), indent=1)
    keep = KEY + ['cond', 'lclass', 'is_target', 'y_v', 'y_ql', 'p_v', 'p_ql', 'serve_hmp50', 'ch_ql', 'ch_v'] + [c for c in feb.columns if c.startswith('c_')]
    feb[keep].to_parquet(HERE / 'b3_feb_preds.parquet', index=False)


if __name__ == '__main__':
    main()
