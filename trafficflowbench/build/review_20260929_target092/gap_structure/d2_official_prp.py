"""D2: keyed official gap-speed updates for candidate PRP_plus (NOT a submission).
Per split: PR head = ramp-aware queue-probability LightGBM fitted on ALL of that split's pseudo holes (Task 1 offline,
per-split transductive, same recipe as the strict G2 test: March -20.0% / April -21.0% gap MSE vs HMP50).
Official rows: rebuild HMP50 on the full eligible official gap grid (official HM heads, released-topology groups, T1),
assert keyed parity with the shipped HMP50 speeds, then
  v = 0.5*T1 + 0.5*clip(g2 + (1-pPRpool)*rf + pPRpool*rq) + 0.5*corr_Jan   (clip 6..120.75), flow unchanged."""
# Runs in pipeline/task1.sh step 9 ("d2_official_prp"); the package helper
# pipeline/make_state_updates_gap_prp.py turns its output into the keyed
# PRP_plus layer. PR head: queue-probability LightGBM with ramp features, fit
# on all pseudo holes of the split (g2_<split>_feats.parquet). PRP: the HMP50
# mixture with the cluster-pooled PR probability. "_plus": + 0.5 x the
# January residual (b3_sp_M2_janfit.txt).
# Outputs: d2_official_updates_prp.parquet (submission_id, new_speed_prp,
# HMP50 speed, probabilities, correction and champion columns) and
# d2_official_report.json.
import os, sys, json, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ramp_feats as RF
from b2_ramp_resid import SP, FLOOR, CAP
from g2_persplit import PARB, NR
# Original absolute root; bin/tfb-run maps it onto the run tree. HM: HM25
# heads; RSV: HMP50 serving rows and cluster topology.
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
R3 = ROOT / 'build/review_20260925/round3/task1_trans_serving'
sys.path.insert(0, str(R3)); import serve_lib as S
HM = ROOT / 'build/review_20260927/iteration2/hole_mixture'
RSV = ROOT / 'build/review_20260927_research/serving'
FMAP = {'p_v': 'ch_v', 'p_ql': 'ch_ql'}


def main():
    """Fits the per-split PR heads and writes the PRP_plus gap speeds of the official holes.

    Raises:
      AssertionError: When a link has no cluster or the HMP50 rebuild differs
        by 1e-6 or more.
    """
    feats = SP + RF.RAMP_FEATS + RF.NEAR_FEATS + ['pool50', 'pp']
    corr_m = lgb.Booster(model_file=str(HERE / 'b3_sp_M2_janfit.txt')); cf = corr_m.feature_name()
    hist = pd.read_parquet(HM / 'official_state_updates.parquet')[['submission_id', 'parent_speed']].rename(columns={'parent_speed': 't1_speed'})
    hmp = pd.read_parquet(RSV / 'official_state_updates.parquet')
    d1 = pd.read_parquet(HERE / 'd1_official_updates.parquet', columns=['submission_id', 'champ_speed', 'champ_flow', 'corr', 'new_speed'])
    rep = {}; outs = []
    for split in ('validation', 'private'):
        # 1. per-split PR head on all pseudo holes of the split
        tr = pd.read_parquet(HERE / f'g2_{split}_feats.parquet', columns=list(dict.fromkeys(feats + ['yq'])))
        m = lgb.train(PARB, lgb.Dataset(tr[feats], tr.yq), NR); m.save_model(str(HERE / f'd2_PR_{split}.txt'))
        rep[split] = {'pr_fit_rows': len(tr), 'pr_fit_pos': int(tr.yq.sum())}; del tr; gc.collect()
        # 2. official full eligible gap grid, HMP50 rebuild
        d = pd.read_parquet(R3 / 'work' / f'off_rows_{split}.parquet'); S.derive(d)
        for kind, col in (('pqueue', 'p'), ('free', 'rf'), ('queued', 'rq')):
            b = lgb.Booster(model_file=str(HM / f'official_{split}_{kind}.txt')); d[col] = b.predict(d[S.FEATS], num_threads=2)
        topo = json.loads((RSV / f'{split}_topology.json').read_text())
        d['group'] = -1
        for panel, info in topo.items():
            ix = (d.panel == panel).to_numpy(); d.loc[ix, 'group'] = np.asarray(info['clusters'])[d.loc[ix, 'li'].to_numpy(int)]
        assert (d.group >= 0).all()
        gk = ['panel', 'date', 'origin', 'group', 'step_i']
        d['pp'] = d.groupby(gk).p.transform('mean')
        fs = lgb.Booster(model_file=str(R3 / 'work' / f'fs_speed_{split}.txt'))
        d['t1'] = np.clip(d.ch_v + fs.predict(d[S.FEATS], num_threads=2), S.SPEED_FLOOR, S.SPEED_CAP)
        # Keyed hole rows take the shipped T1 and G2 speeds; the other eligible
        # gap cells keep the recomputed T1 and the G2v2 value ch_v.
        key = ['panel', 'timestamp', 'link_id']
        g2p = pd.read_parquet(R3 / 'work' / f'parent_{split}_state.parquet', columns=['submission_id'] + key + ['speed_kmh'])
        g2p = g2p[g2p.submission_id.isin(set(hmp.submission_id))].rename(columns={'speed_kmh': 'g2_speed'})
        d = d.merge(g2p, on=key, how='left', validate='one_to_one')
        d = d.merge(hist, on='submission_id', how='left', validate='many_to_one')
        keyed = d.submission_id.notna().to_numpy()
        d.loc[keyed, 't1'] = d.loc[keyed, 't1_speed']
        d['g2'] = np.where(keyed, d.g2_speed, d.ch_v)
        d['hmp50'] = 0.5 * d.t1 + 0.5 * np.clip(d.g2 + (1 - d.pp) * d.rf + d.pp * d.rq, S.SPEED_FLOOR, S.SPEED_CAP)
        chk = d[keyed][['submission_id', 'hmp50']].merge(hmp[['submission_id', 'speed_kmh']], on='submission_id', validate='one_to_one')
        rep[split]['hmp50_rebuild_maxdiff'] = float(np.max(np.abs(chk.hmp50 - chk.speed_kmh)))
        assert rep[split]['hmp50_rebuild_maxdiff'] < 1e-6, rep[split]
        # 3. ramp features + PR head on the full grid, group pooling
        for c in SP:
            if c not in FMAP: d[c] = d['f_' + c]
        d['p_v'] = d.ch_v; d['p_ql'] = d.ch_ql; d['pool50'] = d.hmp50
        d = d.reset_index(drop=True); d = RF.attach(d, split); d = RF.attach_near(d, split)
        d['p_PR'] = m.predict(d[feats], num_threads=2); d['p_PR_pool'] = d.groupby(gk).p_PR.transform('mean')
        d['corr'] = corr_m.predict(d[cf], num_threads=2)
        d['v_prp'] = 0.5 * d.t1 + 0.5 * np.clip(d.g2 + (1 - d.p_PR_pool) * d.rf + d.p_PR_pool * d.rq, S.SPEED_FLOOR, S.SPEED_CAP)
        d['new_speed_prp'] = np.clip(d.v_prp + 0.5 * d['corr'], FLOOR, CAP)
        z = d[keyed][['submission_id', 'panel', 'cond', 'step_i', 'hmp50', 'pp', 'p_PR_pool', 'corr', 'new_speed_prp']].copy()
        z['split'] = split
        dv = z.new_speed_prp - z.hmp50
        rep[split].update(keyed_rows=len(z), dv_mean=float(dv.mean()), dv_abs_mean=float(dv.abs().mean()), dv_abs_p99=float(dv.abs().quantile(.99)),
                          p_change_abs_mean=float((z.p_PR_pool - z.pp).abs().mean()), all_grid_rows=len(d))
        print(split, json.dumps(rep[split]), flush=True)
        outs.append(z); del d, m; gc.collect()
    out = pd.concat(outs, ignore_index=True)
    chk = out.merge(d1.rename(columns={'corr': 'corr_d1'}), on='submission_id', validate='one_to_one')
    rep['corr_vs_d1_maxdiff'] = float(np.max(np.abs(chk['corr'] - chk.corr_d1)))
    out = chk
    out['changed_3dp_vs_champion'] = np.round(out.new_speed_prp, 3) != np.round(out.champ_speed, 3)
    rep['changed_3dp_vs_champion'] = int(out.changed_3dp_vs_champion.sum())
    # held-out reference of the same quantity (pseudo holes, strict)
    ref = []
    for split in ('validation', 'private'):
        c4 = pd.read_parquet(HERE / f'c4_{split}_preds.parquet', columns=['panel', 'date', 'origin', 'li', 'step_i', 't1', 'fv', 'qv', 'pool50', 'c_sp_M2'])
        g2 = pd.read_parquet(HERE / f'g2_{split}_preds.parquet', columns=['panel', 'date', 'origin', 'li', 'step_i', 'p_PR_pool'])
        t = c4.merge(g2, on=['panel', 'date', 'origin', 'li', 'step_i'])
        v = np.clip(0.5 * t.t1 + 0.5 * np.clip((1 - t.p_PR_pool) * t.fv + t.p_PR_pool * t.qv, FLOOR, CAP) + 0.5 * t.c_sp_M2, FLOOR, CAP)
        ref.append(float((v - t.pool50).abs().quantile(.99)))
    rep['heldout_dv_abs_p99'] = ref
    print(json.dumps({k: v for k, v in rep.items() if k not in ('validation', 'private')}))
    out.to_parquet(HERE / 'd2_official_updates_prp.parquet', index=False)
    json.dump(rep, open(HERE / 'd2_official_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
