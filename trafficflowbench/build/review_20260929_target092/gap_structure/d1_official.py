"""D1: keyed Task 1 gap-speed updates for the official validation/private 90-min gaps (NOT a submission).
new speed = clip(HMP50 speed + 0.5 * corr, 6, 120.75) where corr = frozen January-fit residual model b3_sp_M2_janfit
(validated: Feb official-like -15.1% gap MSE / total_B +0.000103; March -12.7%, April -13.1% vs HMP50 on strict pseudo
holes; no-ramp control and wrong-day placebo both worse).  Inputs: served G2v2 gap rows (off_rows_<split>) and the
split's released ramp_states (Task 1 is offline).  Flow unchanged.  Output: d1_official_updates.parquet keyed by
submission_id with the champion's current speed for verification."""
# Runs in pipeline/task1.sh step 9 ("d1_official"); d2_official_prp.py reads
# its output to check the correction and to carry the champion columns.
# "Champion": the HMP50 + DK50 + CAP candidate CSV, whose gap speeds equal the
# HMP50 speeds rounded to 3 decimals.
import os, sys, json, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ramp_feats as RF
from b2_ramp_resid import SP, FLOOR, CAP
# Original absolute root; bin/tfb-run maps it onto the run tree. HMP: HMP50
# serving rows; CHAMP: the candidate CSV after the CAP step; LAM: strength of
# the correction.
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
R3 = ROOT / 'build/review_20260925/round3/task1_trans_serving/work'
HMP = ROOT / 'build/review_20260927_research/serving/official_state_updates.parquet'
CHAMP = ROOT / 'build/deliverables/2026-09-27/capacity/submission_HMP50_DK50_CAP.csv'
FMAP = {'p_v': 'ch_v', 'p_ql': 'ch_ql'}
LAM = 0.5


def main():
    """Writes the D1 keyed gap-speed updates and the champion check for both splits."""
    bst = lgb.Booster(model_file=str(HERE / 'b3_sp_M2_janfit.txt')); fc = bst.feature_name()
    hm = pd.read_parquet(HMP)
    parts = []; rep = {}
    for split in ('validation', 'private'):
        d = pd.read_parquet(R3 / f'off_rows_{split}.parquet')
        for c in SP:
            if c not in FMAP: d[c] = d['f_' + c]
        d['p_v'] = d.ch_v; d['p_ql'] = d.ch_ql
        d = d.reset_index(drop=True)
        d = RF.attach(d, split); d = RF.attach_near(d, split)
        d['corr'] = bst.predict(d[fc], num_threads=2)
        # Joins the correction onto the HMP50 hole rows by panel, timestamp and link.
        keys = ['panel', 'timestamp', 'link_id']
        z = hm[hm.split == split].merge(d[keys + ['corr', 'step_i', 'origin', 'li', 'date']], on=keys, how='left', validate='one_to_one')
        assert z['corr'].notna().all(), (split, int(z['corr'].isna().sum()))
        z['new_speed'] = np.clip(z.speed_kmh + LAM * z['corr'], FLOOR, CAP)
        dv = z.new_speed - z.speed_kmh
        rep[split] = dict(all_gap_rows=len(d), keyed_rows=len(z), changed_3dp=int((np.round(z.new_speed, 3) != np.round(z.speed_kmh, 3)).sum()),
                          dv_mean=float(dv.mean()), dv_abs_mean=float(dv.abs().mean()), dv_p01=float(dv.quantile(.01)), dv_p99=float(dv.quantile(.99)),
                          abs_p99=float(dv.abs().quantile(.99)), ramp_cov_nOR1=float(d.nOR1_prof.notna().mean()))
        print(split, json.dumps(rep[split]), flush=True)
        parts.append(z); del d; gc.collect()
    out = pd.concat(parts, ignore_index=True)
    # champion check: gap speeds in the champion CSV equal round(HMP50, 3) for these ids
    ids = set(out.submission_id)
    ch = pd.concat([c[c.submission_id.isin(ids)][['submission_id', 'speed_kmh', 'flow_vph']] for c in pd.read_csv(CHAMP, chunksize=400000, usecols=['submission_id', 'speed_kmh', 'flow_vph'])], ignore_index=True)
    ch = ch.rename(columns={'speed_kmh': 'champ_speed', 'flow_vph': 'champ_flow'})
    out = out.merge(ch, on='submission_id', how='left', validate='one_to_one')
    rep['champion_rows_found'] = int(out.champ_speed.notna().sum())
    rep['champion_vs_round_hmp50_maxdiff'] = float(np.nanmax(np.abs(out.champ_speed - np.round(out.speed_kmh, 3))))
    rep['champion_flow_vs_parent_flow_maxdiff'] = float(np.nanmax(np.abs(out.champ_flow - out.parent_flow)))
    print(json.dumps({k: v for k, v in rep.items() if k.startswith('champ')}))
    out[['submission_id', 'split', 'panel', 'timestamp', 'link_id', 'window_id', 'cond', 'step_i', 'champ_speed', 'champ_flow',
         'speed_kmh', 'corr', 'new_speed']].rename(columns={'speed_kmh': 'hmp50_unrounded'}).to_parquet(HERE / 'd1_official_updates.parquet', index=False)
    json.dump(rep, open(HERE / 'd1_official_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
