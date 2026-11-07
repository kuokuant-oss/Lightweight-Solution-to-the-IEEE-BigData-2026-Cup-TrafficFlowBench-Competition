"""Apply a fitted RL model to the champion's NON-hole Task-1 target rows of validation + private (all 10 corridors).
Composition (matches how the champion was built):  C1 (stack; G2 inside holes)  ->  RL correction at non-hole rows
->  on the 5 k* corridors re-apply the k* projection  k = 0.9 q/v + 0.1 k_aux  (same k_aux per row, same alpha and
precision weights as build/strongest_candidate_20260922/state/transfer/serving)  ->  round to 3 decimals.
Hole rows are never touched (they keep G2).  Writes rl_state_rows.parquet (submission_id, speed_kmh, flow_vph) for every
state row whose 3-decimal value differs from the champion CSV, plus results/apply_<name>.json with change statistics.
Usage: apply_rl.py --model models/RL_final [--out rl_state_rows.parquet]"""
#
# Role: reproduction pipeline, task1.sh step 3: `--model models/RLB_final --fallback-model models/RL_final
# --fallback-panels D12_I405_N D12_I405_S --out rlb_state_rows.parquet`. RLB_final is the residual LightGBM with
# block features; the Task-1-only D12_I405 pair has no blocks and uses RL_final. In the pipeline the champion
# CSV is the state-only parent rebuilt by pipeline/task1_state_csv.py just before this step.
# Inputs: feats/serve_*.parquet and feats/blk_serve_*.parquet from serve_feats.py; a model folder holds
# feats.json (feature list), y_v.txt (speed residual model) and y_q.txt (per-lane flow residual model).
# Also writes results/apply_<out stem>.csv with the per-panel statistics.
import os, sys, argparse, json, time
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from serve_feats import P10, KSTAR, CHAMP, KUPD
# Output bounds: speed 6 to 120.75 km/h, flow at least 50 veh/h.
SPEED_FLOOR, SPEED_CAP, FLOW_FLOOR = 6.0, 120.74999999999999, 50.0
# k* settings chosen before the held-out check: per-panel alpha and precision weights.
SEL = json.load(open(ROOT / 'build/strongest_candidate_20260922/state/transfer/selection_before_held.json'))


def kstar(v, q, khat, lanes, alpha, weights):
    """= scripts/audit_strongest_state_20260922.apply(family='joint_aux') (copied; identity-checked below)."""
    # Args: v (km/h), q (veh/h), khat: auxiliary density estimate (veh/km), lanes, alpha: weight of khat,
    # weights: 'speed_precision' and 'flow_lane_precision'.
    # The target density mixes q/v with khat; the speed is then the precision-weighted least-squares fit of
    # (v, q) to that density, and the flow is density x speed. Returns (speed, flow).
    basek = q / v
    kk = np.maximum((1 - alpha) * basek + alpha * khat, 0)
    wv = weights['speed_precision']; wq = weights['flow_lane_precision'] / np.square(lanes)
    vv = (wv * v + wq * kk * q) / (wv + wq * np.square(kk))
    vv = np.clip(vv, SPEED_FLOOR, SPEED_CAP)
    return vv, np.maximum(kk * vv, FLOW_FLOOR)


def main():
    """Applies the residual model to every non-hole Task 1 row and writes the rows that change."""
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--out', default='rl_state_rows.parquet')
    ap.add_argument('--fallback-model', default=''); ap.add_argument('--fallback-panels', nargs='*', default=[])
    a = ap.parse_args(); t0 = time.time(); md = HERE / a.model
    out = HERE / a.out
    assert not out.exists(), f'refusing to overwrite {out}'

    def load_model(d):
        """Returns (feature list, speed residual booster, per-lane flow residual booster) of a model folder."""
        return json.load(open(d / 'feats.json')), lgb.Booster(model_file=str(d / 'y_v.txt')), lgb.Booster(model_file=str(d / 'y_q.txt'))
    M = {'main': load_model(md)}
    if a.fallback_model: M['fb'] = load_model(HERE / a.fallback_model)   # e.g. RL on the panels without co-switching blocks
    champ = pd.concat([z[z.task == 'state'][['submission_id', 'speed_kmh', 'flow_vph']] for z in
                       pd.read_csv(CHAMP, usecols=['submission_id', 'task', 'speed_kmh', 'flow_vph'], chunksize=1_000_000)]).set_index('submission_id')
    parts = []; stats = []
    for split in ('validation', 'private'):
        for panel in P10:
            feats, mv, mq = M['fb'] if panel in a.fallback_panels else M['main']
            T = pd.read_parquet(HERE / 'feats' / f'serve_{panel}_{split}.parquet')
            # RL uses the 106 rlfeat features; a longer list means RLB, which adds the block feature table.
            if len(feats) > 106: T = pd.concat([T, pd.read_parquet(HERE / 'feats' / f'blk_serve_{panel}_{split}.parquet')], axis=1)
            pv = mv.predict(T[feats], num_threads=3); pq = mq.predict(T[feats], num_threads=3)
            ln = T.lane_n.to_numpy(float); c1v = T.c1_v.to_numpy(float); c1q = T.c1_q.to_numpy(float)
            # Residuals are added to C1; the flow residual is per lane.
            v = np.clip(c1v + pv, SPEED_FLOOR, SPEED_CAP); q = np.maximum((c1q / ln + pq) * ln, FLOW_FLOOR)
            ch = champ.reindex(T.submission_id.to_numpy()); cv = ch.speed_kmh.to_numpy(); cq = ch.flow_vph.to_numpy()
            ident = np.nan
            if panel in KSTAR:
                u = pd.read_parquet(KUPD / f'updates_{panel}.parquet').set_index('submission_id').reindex(T.submission_id.to_numpy())
                assert u.aux_density_vpkm.notna().all()
                kh = u.aux_density_vpkm.to_numpy(); al = SEL['panel_alphas'][panel]; wts = SEL['weights'][panel]
                # the champion's k* was computed from the ROUNDED (3-dp) stack values; do the same so that a zero RL
                # correction reproduces the champion row exactly (identity check below)
                v0, q0 = kstar(np.round(c1v, 3), np.round(c1q, 3), kh, ln, al, wts)
                ident = float(max(np.abs(np.round(v0, 3) - cv).max(), np.abs(np.round(q0, 3) - cq).max()))
                v, q = kstar(np.round(v, 3), np.round(q, 3), kh, ln, al, wts)
            else:
                ident = float(max(np.abs(np.round(c1v, 3) - cv).max(), np.abs(np.round(c1q, 3) - cq).max()))
            v = np.round(v, 3); q = np.round(q, 3)
            chg = (v != cv) | (q != cq)
            dv = np.abs(v - cv); dq = np.abs(q - cq)
            stats.append(dict(split=split, panel=panel, model=(a.fallback_model if panel in a.fallback_panels else a.model), nonhole_rows=len(T), changed=int(chg.sum()), identity_max_abs=ident,
                              med_abs_dv=float(np.median(dv)), p90_abs_dv=float(np.quantile(dv, .9)), max_abs_dv=float(dv.max()),
                              med_abs_dq=float(np.median(dq)), p90_abs_dq=float(np.quantile(dq, .9)), max_abs_dq=float(dq.max()),
                              med_abs_dq_lane=float(np.median(dq / ln)), mean_dv=float(np.mean(v - cv)), mean_dq_lane=float(np.mean((q - cq) / ln)),
                              med_abs_rl_v=float(np.median(np.abs(pv))), med_abs_rl_q_lane=float(np.median(np.abs(pq)))))
            print(stats[-1], flush=True)
            parts.append(pd.DataFrame({'submission_id': T.submission_id.to_numpy()[chg], 'speed_kmh': v[chg], 'flow_vph': q[chg]}))
    D = pd.concat(parts, ignore_index=True); assert D.submission_id.is_unique
    D.to_parquet(out, index=False)
    S = pd.DataFrame(stats); S.to_csv(HERE / 'results' / f'apply_{out.stem}.csv', index=False)
    summ = {}
    for split, s in S.groupby('split'):
        summ[split] = dict(nonhole_rows=int(s.nonhole_rows.sum()), changed=int(s.changed.sum()), identity_max_abs=float(s.identity_max_abs.max()))
    summ['rows_total'] = len(D); summ['secs'] = round(time.time() - t0)
    json.dump(dict(summary=summ, per_panel=stats), open(HERE / 'results' / f'apply_{out.stem}.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
