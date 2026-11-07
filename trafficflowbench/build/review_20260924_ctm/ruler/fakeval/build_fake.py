"""Build a fake release root whose `validation` Task 2 split holds PSEUDO windows
drawn from validation days, so the exact serving code path can be scored on a
new world (validation month) with observed labels.

History cells that are Task 1 targets (blank in the masked layer) are filled
with the champion Task 1 prediction, because in the real release a window
history is republished without the Task 1 mask. Labels use only cells that are
observed AND eligible AND not filled.
"""
import os, sys, glob, json
import numpy as np, pandas as pd
from pathlib import Path

REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
SP = Path(__file__).resolve().parent
FAKE = SP / 'root'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
ALLP = PANELS + ['D12_I405_N', 'D12_I405_S']
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'validation'


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink():
        os.symlink(src, dst)


def skeleton():
    for name in ['config', 'task1', 'task4', 'README.md', 'sample_submission.csv', 'submission_key.csv']:
        link(REL / name, FAKE / name)
    for p in ALLP:
        link(REL / 'corridors' / p / 'network', FAKE / 'corridors' / p / 'network')
        link(REL / 'corridors' / p / 'train', FAKE / 'corridors' / p / 'train')
        for sp in ['validation', 'private']:
            for sub in ['ramp_states', 'mainline_states_masked']:
                link(REL / 'corridors' / p / sp / sub, FAKE / 'corridors' / p / sp / sub)
        if (REL / 'task2' / p).is_dir():
            link(REL / 'task2' / p / 'train', FAKE / 'task2' / p / 'train')
            other = 'private' if SPLIT == 'validation' else 'validation'
            link(REL / 'task2' / p / other, FAKE / 'task2' / p / other)


def occ_model(d, lanes):
    ok = d.speed_kmh.notna() & d.occupancy.notna() & (d.speed_kmh > 1)
    x = (d.flow_vph[ok] / d.speed_kmh[ok] / d.link_id[ok].map(lanes)).to_numpy()
    y = d.occupancy[ok].to_numpy()
    A = np.c_[x, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def main():
    skeleton()
    champ = pd.read_parquet(SP.parent / 'q' / 'champ_state.parquet')
    champ = champ[champ.split == SPLIT]
    cand_rows, win_rows, hist_parts, lab_parts = [], [], [], []
    for panel in PANELS:
        fd = pd.read_csv(REL / 'corridors' / panel / 'network' / 'fd_parameters.csv', dtype={'link_id': str})
        links_csv = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str})
        vcut_map = dict(zip(links_csv.link_id, 0.6 * links_csv.free_speed_kmh.astype(float)))
        lanes = dict(zip(fd.link_id, fd.lanes.astype(float)))
        cp = champ[champ.panel == panel]
        cpk = cp.set_index(['timestamp', 'link_id'])
        files = sorted(p for p in (REL / 'corridors' / panel / SPLIT / 'mainline_states_masked').glob('*/*.parquet') if not p.name.startswith('._'))
        coef = None
        for path in files:
            regime = path.parent.name.split('=')[1]
            d = pd.read_parquet(path)
            d['link_id'] = d.link_id.astype(str)
            date = str(d.date.iloc[0])
            if coef is None:
                coef = occ_model(d, lanes)
            ts_null = d.groupby('timestamp').speed_kmh.apply(lambda s: s.isna().all())
            gap_ts = set(ts_null[ts_null].index)
            target = (d.is_score_eligible == 1) & d.speed_kmh.isna() & ~d.timestamp.isin(gap_ts)
            key = pd.MultiIndex.from_frame(d.loc[target, ['timestamp', 'link_id']])
            got = cpk.reindex(key)
            d['filled'] = False
            d.loc[target, 'speed_kmh'] = got.speed_kmh.to_numpy()
            d.loc[target, 'flow_vph'] = got.flow_vph.to_numpy()
            d.loc[target, 'filled'] = got.speed_kmh.notna().to_numpy()
            fl = d.filled
            k_lane = d.flow_vph[fl] / d.speed_kmh[fl].clip(lower=1) / d.link_id[fl].map(lanes)
            d.loc[fl, 'occupancy'] = coef[0] * k_lane + coef[1]
            order = d[['link_id', 'milepost']].drop_duplicates().sort_values('milepost').link_id.tolist()
            li = {k: i for i, k in enumerate(order)}
            times = np.sort(d.timestamp.unique()); ti = {k: i for i, k in enumerate(times)}
            r = d.link_id.map(li).to_numpy(); c = d.timestamp.map(ti).to_numpy()
            n = (len(order), len(times))
            V = np.full(n, np.nan); V[r, c] = d.speed_kmh.to_numpy()
            E = np.zeros(n, bool); E[r, c] = (d.is_score_eligible == 1).to_numpy()
            F = np.zeros(n, bool); F[r, c] = d.filled.to_numpy()
            vc = np.array([vcut_map.get(k, 63.0) for k in order])[:, None]
            Qobs = (V <= vc) & E
            Vf = pd.DataFrame(V).interpolate(axis=1, limit_direction='both').to_numpy()
            Qc = Vf <= vc
            gap = np.array([t in gap_ts for t in times])
            LabM = E & ~F & ~np.isnan(V)  # observed, eligible, not filled
            for t in range(12, len(times) - 6):
                if gap[t - 12:t + 7].any():
                    continue
                cov = E[:, t - 12:t].mean()
                hor = Qc[:, t + 1:t + 7]; lastc = Qc[:, t - 1]
                pers = np.repeat(lastc[:, None], 6, axis=1)
                uni = (pers | hor).sum(); piou = (pers & hor).sum() / uni if uni else 1.0
                first = int(np.argmax(hor.any(0))) + 1 if hor.any() else 0
                cand_rows.append((panel, date, regime, t, cov, bool(Qobs[:, t - 12:t].any()), bool(Qobs[:, t - 1].any()),
                                  bool(hor.any()), piou, first))
            np.savez_compressed(SP / 'days' / f'{panel}_{date}.npz', V=V, E=E, F=F, Qc=Qc, LabM=LabM, vc=vc,
                                links=np.array(order), times=times.astype(str))
            d[['timestamp','link_id','speed_kmh','flow_vph','occupancy','pct_observed','is_score_eligible','filled']].to_parquet(SP / 'days' / f'{panel}_{date}.parquet', index=False)
            print(panel, date, regime, 'filled', int(d.filled.sum()), 'gap_ts', len(gap_ts), flush=True)
    c = pd.DataFrame(cand_rows, columns=['panel', 'date', 'regime', 't', 'cov', 'hist_any', 'last_any', 'hor_any', 'piou', 'first'])
    c.to_parquet(SP / f'cands_{SPLIT}.parquet', index=False)


if __name__ == '__main__':
    (SP / 'days').mkdir(exist_ok=True)
    main()
