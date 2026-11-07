"""D12_I405_N/S stage-2 residual on top of RL_final (production), as a ready keyed update + an out-of-world check.

Fit: January+February 2031 D12_I405 rows (rl/feats), base = RL_final prediction (RL_final never trained on D12_I405, so the base
residual on these rows is out-of-panel), target = y - base, recipe = d12_rl.py S2 (31 leaves, lr .05, min leaf 300, l2 10,
free-cell 30% subsample weight 1/.3), rounds 200 (pre-declared primary) and 100 (alternative, better on the Feb test).
Out-of-world check: validation self-supervised pseudo cells (10% of released observed eligible cells, seed 20260924,
truth = released value; rl/feats/selfsup_*_validation.parquet): MSE base vs base+S2.
Serve: validation + private non-hole D12 rows (rl/feats/serve_*): composition exactly apply_rl.py for non-k* panels.
Base reproduction is asserted against rl/rlb_state_rows.parquet (the D12 rows every later champion kept).
Writes d12_state_updates_S2_{200,100}.parquet (submission_id, speed_kmh, flow_vph) for rows whose 3-dp value changes.
"""
# Runs in pipeline/task1.sh step 8 ("d12"). The D12_I405 N/S pair has Task 1
# cells only. RL_final: the residual LightGBM (speed y_v, per-lane flow y_q)
# that corrects the stacked fill, fit without the D12_I405 pair; S2: the
# stage-2 heads of this file. The package run loads the frozen heads
# d12_prod_stage2_y_{v,q}.txt. The pipeline layer reads
# d12_state_updates_S2_100.parquet (first 100 trees of each head).
import os
# Caps the BLAS/OpenMP thread pools at 2 before numpy and LightGBM load.
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']: os.environ[k] = '2'
import sys, json, gc, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; RL = ROOT / 'build/review_20260924_ctm/rl'
D12 = ['D12_I405_N', 'D12_I405_S']
# P2: stage-2 LightGBM recipe.
P2 = dict(objective='l2', learning_rate=0.05, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
          lambda_l2=10.0, num_threads=2, verbose=-1, seed=0)


def main():
    """Loads (or fits) the D12_I405 stage-2 heads, checks them and writes the keyed updates.

    Raises:
      AssertionError: When the rebuilt RL_final rows differ from
        rlb_state_rows.parquet by more than 5e-4, or an output file already
        exists.
    """
    t0 = time.time(); md = RL / 'models/RL_final'; fs = json.load(open(md / 'feats.json'))
    base = {y: lgb.Booster(model_file=str(md / f'{y}.txt')) for y in ('y_v', 'y_q')}
    if all((HERE / f'd12_prod_stage2_{y}.txt').exists() for y in ('y_v', 'y_q')):   # package edit: serve the frozen stage-2 heads (the fit reads Jan+Feb train caches)
        S2 = {y: lgb.Booster(model_file=str(HERE / f'd12_prod_stage2_{y}.txt')) for y in ('y_v', 'y_q')}; print('frozen stage-2 heads', flush=True)
    else:
        tr = pd.concat([pd.read_parquet(RL / 'feats' / f'{p}_{m}.parquet', columns=fs + ['y_v', 'y_q']) for p in D12 for m in ('2031_01', '2031_02')], ignore_index=True)
        # Free-flow cells (p_r >= 0.85) enter as a 30 % sample with weight 1 / 0.3.
        free = (tr.p_r >= 0.85).to_numpy(); U = np.random.default_rng(0).random(len(tr)); sub = ~free | (U < 0.3); w = np.where(free, 1 / 0.3, 1.0)
        S2 = {}
        for y in ('y_v', 'y_q'):
            r = tr[y].to_numpy() - base[y].predict(tr[fs], num_threads=2)
            S2[y] = lgb.train(P2, lgb.Dataset(tr.loc[sub, fs], r[sub], weight=w[sub]), num_boost_round=200); S2[y].save_model(str(HERE / f'd12_prod_stage2_{y}.txt'))
        del tr; gc.collect(); print('fit', round(time.time() - t0), flush=True)
    rep = {}
    # out-of-world check on validation pseudo cells
    for p in [p for p in D12 if (RL / 'feats' / f'selfsup_{p}_validation.parquet').exists()]:   # package edit: research check, skipped when its table is absent
        s = pd.read_parquet(RL / 'feats' / f'selfsup_{p}_validation.parquet')
        bv = base['y_v'].predict(s[fs], num_threads=2); bq = base['y_q'].predict(s[fs], num_threads=2)
        out = dict(n=len(s), c1_mse_v=float(np.mean(s.y_v ** 2)), c1_mse_q=float(np.mean(s.y_q ** 2)), base_mse_v=float(np.mean((s.y_v - bv) ** 2)), base_mse_q=float(np.mean((s.y_q - bq) ** 2)))
        for k in (100, 200):
            sv = S2['y_v'].predict(s[fs], num_iteration=k, num_threads=2); sq = S2['y_q'].predict(s[fs], num_iteration=k, num_threads=2)
            out[f'S2_{k}_mse_v'] = float(np.mean((s.y_v - bv - sv) ** 2)); out[f'S2_{k}_mse_q'] = float(np.mean((s.y_q - bq - sq) ** 2))
            # proxy official-form S_state change on these cells (single panel, no regime split)
            out[f'S2_{k}_dS_proxy'] = .54 * (np.sqrt(out['base_mse_v']) - np.sqrt(out[f'S2_{k}_mse_v'])) / 25 + .46 * (np.sqrt(out['base_mse_q']) - np.sqrt(out[f'S2_{k}_mse_q'])) / 600
        rep[f'selfsup_validation_{p}'] = out; print(p, json.dumps(out), flush=True)
    # serving
    rlb = pd.read_parquet(RL / 'rlb_state_rows.parquet').set_index('submission_id')
    for k in (200, 100):
        parts = []; stats = []
        for split in ('validation', 'private'):
            for p in D12:
                T = pd.read_parquet(RL / 'feats' / f'serve_{p}_{split}.parquet')
                pv = base['y_v'].predict(T[fs], num_threads=2); pq = base['y_q'].predict(T[fs], num_threads=2)
                sv = S2['y_v'].predict(T[fs], num_iteration=k, num_threads=2); sq = S2['y_q'].predict(T[fs], num_iteration=k, num_threads=2)
                # RL_final composition (v0, q0) and with the stage-2 terms (v1, q1):
                # speed in [6, 120.75] km/h, total flow >= 50 vph, 3 decimals.
                ln = T.lane_n.to_numpy(float); c1v = T.c1_v.to_numpy(float); c1q = T.c1_q.to_numpy(float)
                v0 = np.round(np.clip(c1v + pv, 6.0, 120.74999999999999), 3); q0 = np.round(np.maximum((c1q / ln + pq) * ln, 50.0), 3)
                v1 = np.round(np.clip(c1v + pv + sv, 6.0, 120.74999999999999), 3); q1 = np.round(np.maximum((c1q / ln + pq + sq) * ln, 50.0), 3)
                ref = rlb.reindex(T.submission_id.to_numpy()); has = ref.speed_kmh.notna().to_numpy()
                dvr = float(np.abs(v0[has] - ref.speed_kmh.to_numpy()[has]).max()); dqr = float(np.abs(q0[has] - ref.flow_vph.to_numpy()[has]).max())
                assert dvr <= 5e-4 + 1e-9 and dqr <= 5e-4 + 1e-9, (split, p, dvr, dqr)
                chg = (v1 != v0) | (q1 != q0)
                stats.append(dict(split=split, panel=p, rows=len(T), base_rows_in_rlb=int(has.sum()), base_max_abs_diff_v=dvr, base_max_abs_diff_q=dqr, changed=int(chg.sum()),
                                  med_abs_dv=float(np.median(np.abs(v1 - v0))), p99_abs_dv=float(np.quantile(np.abs(v1 - v0), .99)), med_abs_dq_lane=float(np.median(np.abs(q1 - q0) / ln))))
                parts.append(pd.DataFrame({'submission_id': T.submission_id.to_numpy()[chg], 'speed_kmh': v1[chg], 'flow_vph': q1[chg], 'old_speed_kmh': v0[chg], 'old_flow_vph': q0[chg]}))
                del T; gc.collect()
        D = pd.concat(parts, ignore_index=True); assert D.submission_id.is_unique
        out = HERE / f'd12_state_updates_S2_{k}.parquet'; assert not out.exists(), out
        D.to_parquet(out, index=False); sha = hashlib.sha256(out.read_bytes()).hexdigest()
        rep[f'serve_S2_{k}'] = dict(file=str(out), sha256=sha, rows=len(D), stats=stats); print(json.dumps(rep[f'serve_S2_{k}'], indent=1), flush=True)
    json.dump(rep, open(HERE / 'd12_serve.json', 'w'), indent=1); print('seconds', round(time.time() - t0))


if __name__ == '__main__':
    main()
