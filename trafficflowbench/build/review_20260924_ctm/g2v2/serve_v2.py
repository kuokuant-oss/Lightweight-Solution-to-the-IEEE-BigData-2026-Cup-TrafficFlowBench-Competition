"""Serve a gap model on validation + private: Task 1 state rows inside every Task 2 hole of the 8 queue panels.

Gap-row rule (identical to build/review_20260924/work/g4_build.py, which produced the shipped G2 rows):
  task == 'state', split in {validation, private}, panel in the 8 queue panels, and timestamp in
  forecast_start .. forecast_start + 17 steps (= forecast_end + 60 min) of any window in
  task2/<panel>/<split>/window_index.csv, for every link of the panel, whenever that (split, panel, timestamp,
  link_id) exists in submission_key.csv.
Features: G2 base features (profile = g2/<panel>_profile_all.npz, all 9 train months), v2 features (static
template from all 9 train months, blocks from train, selector features from the masked layer + window
spacing in window_index).  Nothing inside the hole is read (asserted).
Usage: serve_v2.py --model <dir> --out <parquet> [--check-sel]"""
#
# Role: reproduction pipeline, task1.sh step 4: `--model build/review_20260924_ctm/g2v2/models/v2_F --out
# build/review_20260925/task1/g2v2F_state_rows.parquet`. v2_F is the final G2v2 model; its flow gives the final
# flow inside the holes. The model folder holds gap_features.json, gap_speed.txt (speed model) and gap_flow.txt
# (per-lane flow model).
# Outputs: <out> (submission_id, speed_kmh, flow_vph), <out stem>_windows.csv (rows per window and selector
# features) and <out stem>_selcheck.csv (selector features compared with dense/sel2_<split>.parquet).
import os, sys, json, argparse, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
G2W = ROOT / 'build/review_20260924/work'; sys.path.insert(0, str(G2W))
from src.tfb import release as R, grid, physics, predict1 as P1
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from v2feat import template_from_hist, LG
from data_v2 import block_ids
from eval_v2 import gap_matrix
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
SEL2 = ROOT / 'build/review_20260924_ctm/dense'


def main():
    """Predicts speed and flow for every Task 1 key row inside a Task 2 hole and writes them."""
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); assert not out.exists(), f'refusing to overwrite {out}'
    t0 = time.time(); md = Path(a.model)
    feats = json.load(open(md / 'gap_features.json'))
    ms = lgb.Booster(model_file=str(md / 'gap_speed.txt')); mf = lgb.Booster(model_file=str(md / 'gap_flow.txt'))
    key = pd.read_csv(R.RELEASE / 'submission_key.csv', usecols=['submission_id', 'task', 'split', 'panel', 'timestamp', 'link_id'], dtype=str)
    key = key[key.task == 'state']
    # (split, panel, timestamp, link_id) -> submission_id of the state rows.
    kidx = dict(zip(zip(key.split, key.panel, key.timestamp, key.link_id), key.submission_id.astype(int)))
    recs, audit, selchk = [], [], []
    for panel in PANELS:
        prof = np.load(G2W / 'g2' / f'{panel}_profile_all.npz', allow_pickle=True)
        # Static queued-speed template from all train months.
        tz = np.load(HERE / 'rows' / f'{panel}_tmpl.npz', allow_pickle=True)
        tmed, tiqr = template_from_hist(tz['hist'].sum(0))
        for split in ('validation', 'private'):
            wi = pd.read_csv(R.RELEASE / 'task2' / panel / split / 'window_index.csv')
            wi['T'] = pd.to_datetime(wi.forecast_origin, utc=True); wi = wi.sort_values('T').reset_index(drop=True)
            # Minutes since the previous official window of the panel (any condition); NaN for the first one.
            wi['gap_min'] = wi['T'].diff().dt.total_seconds() / 60
            # dense/sel2_<split>.parquet is read only to cross-check the selector features.
            sel2 = pd.read_parquet(SEL2 / f'sel2_{split}.parquet').set_index('window_id')
            files = {p.name[len('synthetic_mainline_'):-len('.parquet')].replace('_', '-'): p for p in P1.split_masked_files(panel, split)}
            for date, ws in wi.groupby('date'):
                g = grid.build_masked(panel, files[date]); links = [str(x) for x in g.links]
                assert links == [str(x) for x in prof['links']] == [str(x) for x in tz['links']], 'link order mismatch'
                par = physics.link_params(panel, links); vcut = 0.6 * par['free_speed_kmh']; lanes = np.maximum(par['lanes'], 1.0)
                bid = block_ids(panel, links)
                times = [str(t) for t in g.times]; nT = len(times); wd = pd.Timestamp(date).weekday()
                V = g.value['speed_kmh'].astype(np.float32); Q = g.value['flow_vph'].astype(np.float32); O = g.value['occupancy'].astype(np.float32)
                for _, w in ws.iterrows():
                    # The hole starts at forecast_start, the step after the forecast origin T.
                    a_ = times.index(w.forecast_start); assert times[a_ - 1] == w.forecast_origin
                    seg = slice(a_, min(a_ + LG, nT))
                    assert np.isnan(V[:, seg]).all() and np.isnan(Q[:, seg]).all(), 'hole not blank'
                    assert times[min(a_ + LG - 1, nT - 1)] <= (pd.Timestamp(w.forecast_end) + pd.Timedelta(minutes=60)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    win = dict(cond=w.condition, gap_min=float(w.gap_min) if np.isfinite(w.gap_min) else np.nan)
                    f = gap_matrix(V, Q, O, g.eligible, a_, nT, par, vcut, prof[f'speed_kmh_{wd}'], prof[f'flow_vph_{wd}'], bid, tmed, tiqr, win)
                    if w.condition == 'queue_ongoing':
                        s2 = sel2.loc[w.window_id]
                        selchk.append(dict(window_id=w.window_id, prev_rej=float(f['prev_rej'][0, 0]), n_rej=float(f['n_rej'][0, 0]),
                                           sel2_prev_rej=s2.prev_rej, sel2_n_rej=s2.n_rej))
                    X = pd.DataFrame({k: np.asarray(v).reshape(-1).astype(np.float32) for k, v in f.items()})
                    X['panel_c'] = pd.Categorical([panel] * len(X), categories=PANELS)
                    pv = np.clip(ms.predict(X[feats]), P1.SPEED_FLOOR, P1.SPEED_CAP).reshape(f['step'].shape)
                    # The flow model predicts flow per lane; multiplied back by the lanes.
                    pq = np.maximum(mf.predict(X[feats]) * np.repeat(lanes[:, None], LG, 1).reshape(-1), P1.FLOW_FLOOR).reshape(f['step'].shape)
                    # Keep the hole cells that exist in submission_key.csv.
                    n = 0
                    for s in range(min(LG, nT - a_)):
                        for li, l in enumerate(links):
                            sid = kidx.get((split, panel, times[a_ + s], l))
                            if sid is not None:
                                recs.append((sid, float(pv[li, s]), float(pq[li, s]))); n += 1
                    audit.append(dict(panel=panel, split=split, window_id=w.window_id, condition=w.condition, n_rows=n,
                                      prev_rej=float(f['prev_rej'][0, 0]), n_rej=float(f['n_rej'][0, 0]), blk1=float(f['blk1'][0, 0])))
        print(panel, 'rows so far', len(recs), round(time.time() - t0, 1), flush=True)
    d = pd.DataFrame(recs, columns=['submission_id', 'speed_kmh', 'flow_vph'])
    assert d.submission_id.is_unique
    d = d.sort_values('submission_id').reset_index(drop=True)
    d.to_parquet(out, index=False)
    au = pd.DataFrame(audit); au.to_csv(str(out).replace('.parquet', '_windows.csv'), index=False)
    sc = pd.DataFrame(selchk)
    kn = sc.sel2_prev_rej.notna() | sc.prev_rej.notna()
    agree = dict(n=len(sc), prev_rej_equal=int(((sc.prev_rej == sc.sel2_prev_rej) | (sc.prev_rej.isna() & sc.sel2_prev_rej.isna())).sum()),
                 n_rej_equal=int(((sc.n_rej == sc.sel2_n_rej) | (sc.n_rej.isna() & sc.sel2_n_rej.isna())).sum()))
    sc.to_csv(str(out).replace('.parquet', '_selcheck.csv'), index=False)
    print('rows', len(d), 'by split', au.groupby('split').n_rows.sum().to_dict(), 'selector check vs sel2', agree, 'secs', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
