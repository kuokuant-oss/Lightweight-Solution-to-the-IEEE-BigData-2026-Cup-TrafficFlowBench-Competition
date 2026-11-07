"""Rebuild the champion's C1 fields on a released split and featurise for RL.
C1 = shipped v3 stack (LRTC over the split's days + members + stack + fallback/clip, exactly scripts/112) with the Task-2
hole cells (forecast_start .. +18 steps, all links; window_index.csv) replaced by G2 exactly as g4_build.py
(profile = g2/<panel>_profile_all.npz).  k* is NOT part of C1 (it is re-applied after RL, see apply_rl.py).

--mode serve  : featurise every NON-hole state row of submission_key for (split, panel); check the rebuilt C1 against
                the champion CSV (non-k* panels) / the k* updates' old_* values (k* panels) and G2 inside holes.
                -> feats/serve_<panel>_<split>.parquet (submission_id, li, ci, day, FEATS, c1 v/q) + results/serve_check.csv
--mode selfsup: pseudo-mask 10% of observed eligible cells (seed 20260924, as floor_audit/s18_transfer.py), rerun the
                pipeline, featurise the pseudo cells; truth = released value.
                -> feats/selfsup_<panel>_<split>[_<month>].parquet
Usage: serve_feats.py --mode serve --split validation --panels ...   (one panel in memory at a time)"""
#
# Role: reproduction pipeline, task1.sh step 1 (`--mode serve --split validation|private`, all 10 panels). RL /
# RLB are the residual LightGBM models that correct the stacked fill (C1); this program rebuilds C1 and writes
# their input features. Outputs per panel and split, in feats/:
#   serve_<panel>_<split>.parquet      submission_id, day, li (link row), ci (5-minute step), RL features, c1_v,
#                                      c1_q, lane_n (lanes used for per-lane flow)
#   blk_serve_<panel>_<split>.parquet  RLB block features, same row order
#   c1all_serve_<panel>_<split>.parquet  every key row: stack_v/stack_q (v3 stack), c1_v/c1_q (after G2), hole
# The parity report is written to results/serve_check_<panel>_<split>.csv when the champion CSV exists.
import os, sys, time, argparse, json
sys.dont_write_bytecode = True
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
from src.tfb import release as R, grid, baselines as B, physics, predict1 as P1, task1stack as TS, score1
W3 = ROOT / 'build/review_20260924/work'; sys.path.insert(0, str(W3))
from g2_gap_data import gap_features, LG
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import rlfeat as RF
# QP: the 8 corridors with Task 2 windows (they have holes); P10 adds the Task-1-only D12_I405 pair.
# KSTAR: the 5 density corridors on which the k* density projection is applied after RL.
QP = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
P10 = QP + ['D12_I405_N', 'D12_I405_S']
KSTAR = ['D7_I10_E', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
# Champion CSV and k* updates: read only for the parity check.
CHAMP = ROOT / 'build/review_20260924_newworld/candidates/submission_D_onsetsite_blocks.csv'
KUPD = ROOT / 'build/strongest_candidate_20260922/state/transfer/serving'
TS_FMT = '%Y-%m-%dT%H:%M:%SZ'
STACK = {}   # package edit: day -> (stack v, stack q) before G2 replaces the holes


def c1_days(panel, split, month='', start='', pseudo=False):
    """Yields per day: (di, g, v1, q1, hole_mask, pm, truth_v, truth_q)."""
    # Args: panel, split; month / start keep only day files of that month or from that date on; pseudo=True
    # blanks a 10% sample of observed eligible cells (selfsup mode) and keeps their released values as truth.
    # v1/q1: C1 speed (km/h) and flow (veh/h), links x 5-minute steps; hole_mask marks the Task 2 gap cells;
    # pm is the pseudo mask (None in serve mode); truth_v/truth_q are None in serve mode.
    files = P1.split_masked_files(panel, split)
    if month: files = [p for p in files if month in p.name]
    if start: files = [p for p in files if p.name[len('synthetic_mainline_'):-len('.parquet')].replace('_', '-') >= start]
    grids = [grid.build_masked(panel, p) for p in files]
    rng = np.random.default_rng(20260924); pms = []; tvs = []
    for g in grids:
        if pseudo:
            obs = np.isfinite(g.value['speed_kmh']) & np.isfinite(g.value['flow_vph']) & g.eligible & ~g.blanked
            pm = obs & (rng.random(obs.shape) < 0.10)
            tvs.append({f: g.value[f].copy() for f in ('speed_kmh', 'flow_vph')})
            for f in ('speed_kmh', 'flow_vph', 'occupancy'):
                v = g.value[f].copy(); v[pm] = np.nan; g.value[f] = v
            g.blanked = g.blanked | pm
        else:
            pm = None; tvs.append(None)
        pms.append(pm)
    # Stack inputs: stack weights and models, the weekday x time-of-day mean of the development train files,
    # and LRTC-TNN (low-rank tensor completion) over all days of the split (links x steps x days).
    W = P1.load_weights(); models = TS.load(); hist = P1.historical_mean(panel, P1.dev_train_files(panel))
    lrtc = {f: B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f], nan=0.0) for g in grids], axis=2), rho=P1.LRTC_KNOBS[f][0], theta=P1.LRTC_KNOBS[f][1]) for f in P1.FIELDS}
    links = [str(x) for x in grids[0].links]; par = physics.link_params(panel, links)
    vcut = 0.6 * par['free_speed_kmh']; lanes = np.maximum(par['lanes'], 1.0)
    # Scored splits of the queue corridors: the G2 gap model fills the Task 2 holes.
    wins = pd.DataFrame()
    if split != 'train' and panel in QP:
        wins = pd.read_csv(R.RELEASE / 'task2' / panel / split / 'window_index.csv')
        prof = np.load(W3 / 'g2' / f'{panel}_profile_all.npz', allow_pickle=True)
        assert links == [str(x) for x in prof['links']]
        gfe = json.load(open(W3 / 'g2' / 'gap_features.json'))
        ms_ = lgb.Booster(model_file=str(W3 / 'g2' / 'gap_speed.txt')); mf_ = lgb.Booster(model_file=str(W3 / 'g2' / 'gap_flow.txt'))
    for di, g in enumerate(grids):
        # v3 stack prediction of the day; NaN or values below the floor fall back to the historical mean, then to
        # the day's median observation.
        out = TS.predict_grid(g, di, lrtc, hist, W, par, models)
        for f, floor in (('speed_kmh', P1.SPEED_FLOOR), ('flow_vph', P1.FLOW_FLOOR)):
            h = hist.get((f, pd.Timestamp(g.times[0]).weekday())); bad = ~np.isfinite(out[f]) | (out[f] < floor)
            if bad.any() and h is not None: out[f] = np.where(bad & np.isfinite(h), h, out[f]); bad = ~np.isfinite(out[f]) | (out[f] < floor)
            if bad.any(): ob = g.value[f]; med = float(np.nanmedian(ob)) if np.isfinite(ob).any() else floor; out[f] = np.where(bad, med, out[f])
        v1 = np.clip(out['speed_kmh'], P1.SPEED_FLOOR, P1.SPEED_CAP); q1 = np.maximum(out['flow_vph'], P1.FLOW_FLOOR)
        STACK[di] = (v1.copy(), q1.copy())   # package edit
        hole = np.zeros(v1.shape, bool)
        if len(wins):
            times = [pd.Timestamp(t).strftime(TS_FMT) for t in g.times]; nT = len(times); wd = pd.Timestamp(g.date).weekday()
            HV, HQ = prof[f'speed_kmh_{wd}'], prof[f'flow_vph_{wd}']
            # Hole of a window: forecast_start and the next LG-1 steps (30-minute horizon + 60 minutes), all links.
            for _, w in wins[wins.date == g.date].iterrows():
                a_ = times.index(w.forecast_start); seg = slice(a_, min(a_ + LG, nT)); ns = seg.stop - a_
                assert np.isnan(g.value['speed_kmh'][:, seg]).all() and np.isnan(g.value['flow_vph'][:, seg]).all(), 'hole not blank'
                f = gap_features(g.value['speed_kmh'], g.value['flow_vph'], g.value['occupancy'], HV, HQ, a_, nT, par, vcut)
                X = pd.DataFrame({k: v.reshape(-1).astype(np.float32) for k, v in f.items()})
                X['panel_c'] = pd.Categorical([panel] * len(X), categories=QP)
                pv = np.clip(ms_.predict(X[gfe]), P1.SPEED_FLOOR, P1.SPEED_CAP).reshape(f['step'].shape)
                # The G2 flow model predicts flow per lane; multiplied back by the lanes.
                pq = np.maximum(mf_.predict(X[gfe]) * np.repeat(lanes[:, None], LG, 1).reshape(-1), P1.FLOW_FLOOR).reshape(f['step'].shape)
                v1[:, seg] = pv[:, :ns]; q1[:, seg] = pq[:, :ns]; hole[:, seg] = True
        yield di, g, v1, q1, hole, pms[di], (tvs[di] or {}).get('speed_kmh'), (tvs[di] or {}).get('flow_vph')


def lanes_of(panel, g):
    """Returns the lanes per row of the day grid (by station id, else link id; at least 1)."""
    lanes_d = score1.lanes_by_key(panel)
    return np.maximum(np.array([lanes_d.get(str(s), lanes_d.get(str(l), 1.0)) for s, l in zip(g.station, g.links)], dtype=float), 1.0)


# my_blocks: per panel, the block id of each link (groups of links that queue together); -1 = no block.
BLOCKS = np.load(ROOT / 'build/review_20260924_newworld/work/blocks/my_blocks.npy', allow_pickle=True).item()


def blk_for(panel, g, v1, tg):
    """RLB block features (8 queue panels; no blocks elsewhere -> bid = -1)."""
    links = [str(x) for x in g.links]; par = physics.link_params(panel, links)
    if panel in BLOCKS:
        assert [str(x) for x in BLOCKS[panel]['links']] == links
        bid = np.asarray(BLOCKS[panel]['bid'])
    else:
        bid = np.full(len(links), -1)
    return RF.block_features(v1, g.value['speed_kmh'], par['free_speed_kmh'].astype(float), tg, bid)


def feats_for(panel, g, v1, q1, tg):
    """Returns the RL features (rlfeat.day_features) of the tg cells and the lanes per link; flows are per lane."""
    links = [str(x) for x in g.links]; par = physics.link_params(panel, links)
    ln = lanes_of(panel, g)
    c = RF.day_features(v1, q1 / ln[:, None], g.value['speed_kmh'], g.value['flow_vph'] / ln[:, None], g.value['occupancy'],
                        par['free_speed_kmh'].astype(float), ln, par['length_km'].astype(float), tg)
    return c, ln


def main():
    """Builds the RL / RLB feature tables for the panels of one split (serve or selfsup mode)."""
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['serve', 'selfsup'], required=True)
    ap.add_argument('--split', required=True); ap.add_argument('--panels', nargs='*', default=P10)
    ap.add_argument('--month', default=''); ap.add_argument('--start', default='')
    a = ap.parse_args(); t0 = time.time()
    if a.mode == 'serve':
        # State key rows of the split: the Task 1 cells to submit.
        key = pd.read_csv(R.RELEASE / 'submission_key.csv', usecols=['submission_id', 'task', 'split', 'panel', 'timestamp', 'link_id'], dtype=str)
        key = key[(key.task == 'state') & (key.split == a.split)]; key['submission_id'] = key.submission_id.astype(np.int64)
        champ = pd.concat([z[z.task == 'state'][['submission_id', 'speed_kmh', 'flow_vph']] for z in
                           pd.read_csv(CHAMP, usecols=['submission_id', 'task', 'speed_kmh', 'flow_vph'], chunksize=1_000_000)]).set_index('submission_id') if CHAMP.exists() else None   # package edit: parity input only; absent in a cold run
        checks = []
    for panel in a.panels:
        tag = f'{panel}_{a.split}' + (f'_{a.month}' if a.month else '')
        out = HERE / 'feats' / (f'serve_{tag}.parquet' if a.mode == 'serve' else f'selfsup_{tag}.parquet')
        if out.exists(): print('skip', out.name, flush=True); continue
        parts = []; bparts = []; aparts = []   # package edit: aparts
        if a.mode == 'serve':
            kp = key[key.panel == panel].copy(); kp['date'] = kp.timestamp.str[:10]
            ku = pd.read_parquet(KUPD / f'updates_{panel}.parquet').set_index('submission_id') if panel in KSTAR and champ is not None else None   # package edit: parity only
        for di, g, v1, q1, hole, pm, tv, tq in c1_days(panel, a.split, a.month, a.start, pseudo=(a.mode == 'selfsup')):
            if a.mode == 'serve':
                z = kp[kp.date == g.date]
                li_ = {str(x): j for j, x in enumerate(g.links)}; ti_ = {pd.Timestamp(t).strftime(TS_FMT): j for j, t in enumerate(g.times)}
                rr = z.link_id.map(li_); cc = z.timestamp.map(ti_); assert rr.notna().all() and cc.notna().all()
                rr = rr.to_numpy(int); cc = cc.to_numpy(int); assert g.target[rr, cc].all()
                # Key rows inside a Task 2 hole keep the G2 values; only the other rows are featurised.
                ih = hole[rr, cc]
                sv_, sq_ = STACK.pop(di)   # package edit: stack (v3) and C1 values of every key row -> c1all_serve_*.parquet
                aparts.append(pd.DataFrame({'submission_id': z.submission_id.to_numpy(), 'stack_v': sv_[rr, cc], 'stack_q': sq_[rr, cc],
                                            'c1_v': v1[rr, cc], 'c1_q': q1[rr, cc], 'hole': ih}))
                if champ is not None:   # package edit: the parity check runs only when the champion CSV is present
                    # parity of the rebuilt C1 with what the champion shipped
                    ch = champ.reindex(z.submission_id.to_numpy()); assert ch.notna().all().all()
                    cv, cq = ch.speed_kmh.to_numpy(), ch.flow_vph.to_numpy()
                    if ku is not None:
                        o = ku.reindex(z.submission_id.to_numpy()); assert o.notna().all().all()
                        bv, bq = o.old_speed_kmh.to_numpy(), o.old_flow_vph.to_numpy()
                        kv, kq = o.new_speed_kmh.to_numpy(), o.new_flow_vph.to_numpy()
                        ck_k = float(max(np.abs(kv[~ih] - cv[~ih]).max(initial=0), np.abs(kq[~ih] - cq[~ih]).max(initial=0)))
                    else:
                        bv, bq = cv, cq; ck_k = np.nan
                    dv = np.abs(v1[rr, cc] - bv); dq = np.abs(q1[rr, cc] - bq)
                    dvh = np.abs(v1[rr, cc] - cv)[ih]; dqh = np.abs(q1[rr, cc] - cq)[ih]
                    checks.append(dict(panel=panel, split=a.split, date=g.date, rows=len(z), hole_rows=int(ih.sum()),
                                       max_dv_nonhole=float(dv[~ih].max(initial=0)), max_dq_nonhole=float(dq[~ih].max(initial=0)),
                                       max_dv_hole=float(dvh.max(initial=0)), max_dq_hole=float(dqh.max(initial=0)), max_kstar_vs_champ_nonhole=ck_k))
                tg = np.zeros(g.target.shape, bool); tg[rr[~ih], cc[~ih]] = True
                order = np.argsort(rr[~ih] * len(g.times) + cc[~ih], kind='stable')   # np.where(tg) is row-major
                sid = z.submission_id.to_numpy()[~ih][order]
                c, ln = feats_for(panel, g, v1, q1, tg)
                li, ci = np.where(tg)
                assert (li == rr[~ih][order]).all() and (ci == cc[~ih][order]).all()
                d = pd.DataFrame(c); d.insert(0, 'submission_id', sid); d.insert(1, 'day', np.int16(di)); d.insert(2, 'li', li.astype(np.int16)); d.insert(3, 'ci', ci.astype(np.int16))
                d['c1_v'] = v1[tg]; d['c1_q'] = q1[tg]; d['lane_n'] = ln[li].astype(np.float32)
                parts.append(d); bparts.append(pd.DataFrame(blk_for(panel, g, v1, tg)))
            else:
                # selfsup: features on the pseudo-masked cells outside holes; labels are the residuals truth - C1
                # (flow residual per lane).
                tg = pm & ~hole
                c, ln = feats_for(panel, g, v1, q1, tg)
                li, ci = np.where(tg)
                d = pd.DataFrame(c); d.insert(0, 'day', np.int16(di)); d.insert(1, 'li', li.astype(np.int16)); d.insert(2, 'ci', ci.astype(np.int16))
                d['y_v'] = (tv - v1)[tg].astype(np.float32); d['y_q'] = ((tq - q1) / ln[:, None])[tg].astype(np.float32)
                d['near_hole'] = (np.convolve(hole.any(0).astype(float), np.ones(25), 'same') > 0)[ci]   # within 12 steps of a hole
                parts.append(d); bparts.append(pd.DataFrame(blk_for(panel, g, v1, tg)))
        D = pd.concat(parts, ignore_index=True); D.to_parquet(out, index=False)
        pd.concat(bparts, ignore_index=True).to_parquet(out.parent / f'blk_{out.name}', index=False)
        if aparts: pd.concat(aparts, ignore_index=True).to_parquet(out.parent / f'c1all_{out.name}', index=False)   # package edit
        print(a.mode, tag, D.shape, round(time.time() - t0), flush=True)
        if a.mode == 'serve' and champ is not None:   # package edit: parity report only when checked
            C = pd.DataFrame(checks); cp = C[C.panel == panel]
            cp.to_csv(HERE / 'results' / f'serve_check_{panel}_{a.split}.csv', index=False)
            print('  parity', panel, a.split, {k: float(cp[k].max()) for k in ('max_dv_nonhole', 'max_dq_nonhole', 'max_dv_hole', 'max_dq_hole', 'max_kstar_vs_champ_nonhole')}, flush=True)
    print('secs', round(time.time() - t0))


if __name__ == '__main__':
    main()
