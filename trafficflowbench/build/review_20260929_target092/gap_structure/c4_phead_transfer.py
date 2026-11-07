"""C4: transfer test of the ramp-aware queue-probability head (F1, Jan-fit, frozen) inside the HMP50 structure on the
March/April strict pseudo holes.  HM strict heads p/fv/qv and co-switch groups from review_20260927_research/full_pool
(full eligible grid; HM pooled p computed on the full grid exactly as pool50).  p_ramp pooled over the group's labelled
cells only (full-grid features were not stored; approximation, no label use).  Variants as in F1."""
# Runs in pipeline/task1.sh step 9, once per split. Its c4_<split>_preds.parquet
# (strict HMP50 parts t1 / fv / qv / pool50, cluster group, pooled p, queue
# label yq and the January residual c_sp_M2 per pseudo-hole cell) is the base
# table of g2_persplit.py and the held-out reference of d2_official_prp.py.
# F1 head (f1_phead_janfit.txt): January-fit queue-probability LightGBM with
# ramp features; a cell is queued when its speed is <= 0.6 x free-flow speed.
# Usage: c4_phead_transfer.py <validation|private>
# Outputs: c4_<split>_preds.parquet and c4_<split>_results.json.
import os, sys, json, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ramp_feats as RF
from b2_ramp_resid import SP, FLOOR, CAP
from c1_transfer import rel, R2, R3, KEY, PANELS, FMAP, base_preds
# Original absolute path of the HMP50 research folder; bin/tfb-run maps it
# onto the run tree.
RES = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260927_research')


def main():
    """Scores the January ramp heads inside the HMP50 structure on one split's pseudo holes."""
    split = sys.argv[1]; src = R2 if split == 'validation' else R3
    ph_m = lgb.Booster(model_file=str(HERE / 'f1_phead_janfit.txt')); pf = ph_m.feature_name()
    rs_m = lgb.Booster(model_file=str(HERE / 'b3_sp_M2_janfit.txt')); rf = rs_m.feature_name()
    base = base_preds(split); parts = []; maxdiff = 0.0
    for panel in PANELS:
        # Full eligible grid of the strict HM heads; pp = queue probability
        # pooled over cluster, window and step (as in HMP50).
        full = pd.read_parquet(RES / f'full_pool/{split}_{panel}.parquet')
        full['pp'] = full.groupby(['date', 'origin', 'group', 'step_i']).p.transform('mean')
        b = base[base.panel == panel].merge(full[KEY + ['p', 'fv', 'qv', 'group', 'pp']], on=KEY, how='left', validate='one_to_one')
        assert b.pp.notna().all()
        ph = pd.read_parquet(src / f'ph_{split}_{panel}.parquet', columns=KEY + ['f_' + c for c in SP if c not in FMAP] + ['ch_v', 'ch_ql'])
        x = b.merge(ph, on=KEY, how='inner', validate='one_to_one')
        for c in SP:
            if c not in FMAP: x[c] = x['f_' + c]
        x['p_v'] = x.ch_v; x['p_ql'] = x.ch_ql; x = x.reset_index(drop=True)
        x = RF.attach(x, split); x = RF.attach_near(x, split)
        x['p_ramp'] = ph_m.predict(x[pf], num_threads=2); x['c_sp_M2'] = rs_m.predict(x[rf], num_threads=2)
        # Rebuilds pool50 from its parts; the largest difference is reported.
        rec = 0.5 * x.t1 + 0.5 * np.clip((1 - x.pp) * x.fv + x.pp * x.qv, FLOOR, CAP)
        maxdiff = max(maxdiff, float(np.max(np.abs(rec - x.pool50))))
        x['yq'] = (x.y_v <= 0.6 * x.f_free_speed_kmh).astype(int)
        parts.append(x[KEY + ['y_v', 'yq', 'day', 'cond', 't1', 'pool50', 'p', 'fv', 'qv', 'group', 'pp', 'p_ramp', 'c_sp_M2']])
        print(panel, len(x), flush=True); del x, full, ph, b; gc.collect()
    t = pd.concat(parts, ignore_index=True)
    print('pool50 reconstruction max diff', maxdiff, flush=True)
    t['p_ramp_pool'] = t.groupby(['panel', 'date', 'origin', 'group', 'step_i']).p_ramp.transform('mean')
    t['p_mix'] = 0.5 * t.pp + 0.5 * t.p_ramp_pool
    out = {'pool50_reconstruction_maxdiff': maxdiff}
    from f1_phead import ll, auc
    for c in ('pp', 'p_ramp', 'p_ramp_pool', 'p_mix'):
        out['ll_' + c] = ll(t.yq, t[c]); out['auc_' + c] = auc(t.yq, t[c])
    print({k: round(v, 5) for k, v in out.items()}, flush=True)
    # p* variants: R (ramp head), RP (ramp head pooled per cluster), MIX (half
    # HM pooled p, half RP); '_plus' adds 0.5 x c_sp_M2; 'resid' = pool50 +
    # 0.5 x c_sp_M2.
    for nm, p in (('R', t.p_ramp), ('RP', t.p_ramp_pool), ('MIX', t.p_mix)):
        t['v_' + nm] = 0.5 * t.t1 + 0.5 * np.clip((1 - p) * t.fv + p * t.qv, FLOOR, CAP)
        t['v_' + nm + '_plus'] = np.clip(t['v_' + nm] + 0.5 * t.c_sp_M2, FLOOR, CAP)
    t['v_resid'] = np.clip(t.pool50 + 0.5 * t.c_sp_M2, FLOOR, CAP)
    for nm in ('resid', 'R', 'RP', 'MIX', 'R_plus', 'RP_plus', 'MIX_plus'):
        out[nm] = rel(t, 'y_v', 'pool50', 'v_' + nm)
        print(split, nm, json.dumps({k: out[nm][k] for k in ('rmse1', 'rel', 'ci', 'onset', 'ongoing', 'odd', 'even', 'early12', 'panels_improved')}), flush=True)
    json.dump(out, open(HERE / f'c4_{split}_results.json', 'w'), indent=1)
    t.to_parquet(HERE / f'c4_{split}_preds.parquet', index=False)


if __name__ == '__main__':
    main()
