"""Pseudo-truth candidate origins for a split, built exactly as ruler/fakeval/build_fake.py (Task 1 target cells of the
masked layer filled with a champion Task 1 value) + ruler2/cands_v2.py (corrected selector rule, block-mate majority
clean labels, temporal interpolation).
  s2_cands.py check <panel> ...   validation, fill = ruler/q/champ_state (what fakeval used); asserts equality with
                                  ruler2/cands_v2.parquet for those panels (harness proof)
  s2_cands.py private             private, fill = PARENT private state rows -> work/cands_private.parquet
                                  (+ work/parent_private_state.parquet: parent state rows keyed by panel/timestamp/link)"""
import os, sys, glob, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
RUL = ROOT / 'build/review_20260924_ctm/ruler'
PARENT = ROOT / 'build/review_20260925/round2/synthesis/submission_G1_FE_A1.csv'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
B = np.load(RUL / 'blocks/my_blocks.npy', allow_pickle=True).item()


def parent_state(split):
    out = HERE / 'work' / f'parent_{split}_state.parquet'
    if out.exists(): return pd.read_parquet(out)
    parts = []
    for z in pd.read_csv(REL / 'submission_key.csv', usecols=['submission_id', 'task', 'split', 'panel', 'timestamp', 'link_id'],
                         dtype=str, chunksize=2_000_000):
        parts.append(z[(z.task == 'state') & (z.split == split)][['submission_id', 'panel', 'timestamp', 'link_id']])
    k = pd.concat(parts, ignore_index=True); k['submission_id'] = k.submission_id.astype(np.int64)
    ps = []
    for z in pd.read_csv(PARENT, usecols=['submission_id', 'task', 'speed_kmh', 'flow_vph'], chunksize=2_000_000):
        ps.append(z[z.task == 'state'][['submission_id', 'speed_kmh', 'flow_vph']])
    P = pd.concat(ps, ignore_index=True)
    k = k.merge(P, on='submission_id', how='left')
    assert k.speed_kmh.notna().all(), 'parent missing state rows'
    k.to_parquet(out, index=False); print(split, 'parent state rows', len(k), flush=True)
    return k


def day_cands(panel, path, cpk, vcut_map):
    d = pd.read_parquet(path); d['link_id'] = d.link_id.astype(str); date = str(d.date.iloc[0])
    ts_null = d.groupby('timestamp').speed_kmh.apply(lambda s: s.isna().all()); gap_ts = set(ts_null[ts_null].index)
    target = (d.is_score_eligible == 1) & d.speed_kmh.isna() & ~d.timestamp.isin(gap_ts)
    key = pd.MultiIndex.from_frame(d.loc[target, ['timestamp', 'link_id']])
    got = cpk.reindex(key)
    d.loc[target, 'speed_kmh'] = got.speed_kmh.to_numpy(); nfill = int(got.speed_kmh.notna().sum())
    order = d[['link_id', 'milepost']].drop_duplicates().sort_values('milepost').link_id.tolist()
    li = {k: i for i, k in enumerate(order)}; times = np.sort(d.timestamp.unique()); ti = {k: i for i, k in enumerate(times)}
    r = d.link_id.map(li).to_numpy(); c = d.timestamp.map(ti).to_numpy(); n = (len(order), len(times))
    V = np.full(n, np.nan); V[r, c] = d.speed_kmh.to_numpy(); E = np.zeros(n, bool); E[r, c] = (d.is_score_eligible == 1).to_numpy()
    vc = np.array([vcut_map.get(k, 63.0) for k in order])[:, None]
    # --- cands_v2 logic (gap recomputed on the filled day, as cands_v2 does from the saved day parquet)
    ts_null2 = d.groupby('timestamp').speed_kmh.apply(lambda s: s.isna().all()); gap = np.array([ts_null2.get(t, False) for t in times])
    links = np.array(order).astype(str)
    obs = ~np.isnan(V); Qobs = (V <= vc) & E & obs
    Vi = pd.DataFrame(V).interpolate(axis=1, limit_direction='both').to_numpy(); Q = Vi <= vc
    bid = dict(zip(B[panel]['links'], B[panel]['bid'])); bl = np.array([bid.get(l, -1) for l in links])
    q = (V <= vc) & obs
    for b in np.unique(bl[bl >= 0]):
        ix = np.where(bl == b)[0]
        if len(ix) < 2: continue
        no = obs[ix].sum(0); nq = q[ix].sum(0); maj = nq * 2 > no
        for j in ix:
            miss = ~obs[j] & (no > 0); Q[j, miss] = maj[miss]
    rows = []
    for t in range(13, len(times) - 6):
        if gap[t - 12:t + 7].any(): continue
        cov = E[:, t - 12:t].mean(); hor = Q[:, t + 1:t + 7]
        pers = np.repeat(Q[:, t][:, None], 6, 1); u = (pers | hor).sum(); piou = (pers & hor).sum() / u if u else 1.0
        first = int(np.argmax(hor.any(0))) + 1 if hor.any() else 0
        rows.append((panel, date, t, cov, bool(Qobs[:, t - 12:t].any()), bool(Qobs[:, t - 1].any() and Qobs[:, t - 2].any()), bool(hor.any()), piou, first))
    return rows, dict(panel=panel, date=date, regime=path.parent.name.split('=')[1], filled=nfill, targets=int(target.sum()), gap_ts=len(gap_ts))


def run(split, panels, champ):
    rows, logs = [], []
    for panel in panels:
        links_csv = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str})
        vcut_map = dict(zip(links_csv.link_id, 0.6 * links_csv.free_speed_kmh.astype(float)))
        cpk = champ[champ.panel == panel].set_index(['timestamp', 'link_id'])[['speed_kmh', 'flow_vph']]
        files = sorted(p for p in (REL / 'corridors' / panel / split / 'mainline_states_masked').glob('*/*.parquet') if not p.name.startswith('._'))
        for path in files:
            r, lg = day_cands(panel, path, cpk, vcut_map); rows.extend(r); logs.append(lg)
        print(panel, split, 'days', len(files), 'filled', sum(x['filled'] for x in logs if x['panel'] == panel),
              'targets', sum(x['targets'] for x in logs if x['panel'] == panel), flush=True)
    c = pd.DataFrame(rows, columns=['panel', 'date', 't', 'cov', 'hist_any', 'last_any', 'hor_any', 'piou', 'first'])
    return c, pd.DataFrame(logs)


def main():
    mode = sys.argv[1]; t0 = time.time()
    if mode == 'check':
        panels = sys.argv[2:]
        ch = pd.read_parquet(RUL / 'q' / 'champ_state.parquet'); ch = ch[ch.split == 'validation']
        c, lg = run('validation', panels, ch)
        ref = pd.read_parquet(ROOT / 'build/review_20260924_ctm/ruler2/cands_v2.parquet')
        ref = ref[ref.panel.isin(panels)].drop(columns='regime')
        m = ref.merge(c, on=['panel', 'date', 't'], how='outer', indicator=True, suffixes=('_r', '_m'))
        res = dict(panels=panels, rows_ref=len(ref), rows_mine=len(c), both=int((m._merge == 'both').sum()))
        b = m[m._merge == 'both']
        for k in ('hist_any', 'last_any', 'hor_any', 'first'): res[f'mismatch_{k}'] = int((b[k + '_r'] != b[k + '_m']).sum())
        for k in ('cov', 'piou'): res[f'maxabs_{k}'] = float((b[k + '_r'] - b[k + '_m']).abs().max())
        res['identical'] = bool(res['both'] == len(ref) == len(c) and all(res[f'mismatch_{k}'] == 0 for k in ('hist_any', 'last_any', 'hor_any', 'first'))
                                and res['maxabs_cov'] < 1e-12 and res['maxabs_piou'] < 1e-12)
        print(json.dumps(res, indent=1)); json.dump(res, open(HERE / 'results' / 'r_cands_check_validation.json', 'w'), indent=1)
    else:
        ch = parent_state('private')
        c, lg = run('private', PANELS, ch)
        c.to_parquet(HERE / 'work' / 'cands_private.parquet', index=False); lg.to_csv(HERE / 'work' / 'cands_private_days.csv', index=False)
        print('private cands', len(c), 'fill share', round(lg.filled.sum() / lg.targets.sum(), 4))
    print('done', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
