"""Build a fake release root whose `validation` Task 2 split holds PSEUDO windows
drawn from validation days, so the exact serving code path can be scored on a
new world (validation month) with observed labels.

History cells that are Task 1 targets (blank in the masked layer) are filled
with the champion Task 1 prediction, because in the real release a window
history is republished without the Task 1 mask. Labels use only cells that are
observed AND eligible AND not filled.
"""
#
# Role: reproduction pipeline (task1.sh step 5, `build_fake.py validation`), run after
# ruler/q/build_champ_state.py. Its day caches feed ruler2/cands_v2.py and ruler2/select_write_v2.py, which
# cut the pseudo windows of the validation month.
# Usage: build_fake.py [split]   (default validation)
# Inputs: ../q/champ_state.parquet (champion state rows), the released masked layer, network links.csv and
# fd_parameters.csv. Outputs:
#   root/                       symlinked skeleton of the release (see skeleton())
#   days/<panel>_<date>.npz     V (speed, km/h, after the champion fill), E (eligible), F (filled), Qc (clean
#                               queue state), LabM (label mask), vc (queue speed), links, times; (links x steps)
#   days/<panel>_<date>.parquet the day's rows after the fill, with a `filled` flag
#   cands_<split>.parquet       candidate origins (see the columns at the end of main())
import os, sys, glob, json
import numpy as np, pandas as pd
from pathlib import Path

REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
SP = Path(__file__).resolve().parent
# FAKE: the fake release root. PANELS: the 8 corridors with Task 2 windows; ALLP adds the Task-1-only
# D12_I405 pair.
FAKE = SP / 'root'
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
ALLP = PANELS + ['D12_I405_N', 'D12_I405_S']
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'validation'


def link(src, dst):
    """Creates a symlink dst -> src (and its parent folders) unless dst already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink():
        os.symlink(src, dst)


def skeleton():
    """Links the parts of the real release that the pseudo split reuses into the fake root.

    Links config, task1, task4, the submission files, each corridor's network, train data and the masked
    layer and ramps of both scored splits, and the Task 2 train split and the other scored split. The Task 2
    folder of SPLIT itself is left for the pseudo windows.
    """
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
    """Fits occupancy as a straight line of per-lane density.

    Args:
        d: one day of mainline rows (speed_kmh, flow_vph, occupancy, link_id).
        lanes: link_id -> number of lanes.

    Returns:
        (slope, intercept) of occupancy = slope * flow / speed / lanes + intercept, least squares over rows
        with speed above 1 km/h.
    """
    ok = d.speed_kmh.notna() & d.occupancy.notna() & (d.speed_kmh > 1)
    x = (d.flow_vph[ok] / d.speed_kmh[ok] / d.link_id[ok].map(lanes)).to_numpy()
    y = d.occupancy[ok].to_numpy()
    A = np.c_[x, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def main():
    """Fills the Task 1 holes of every SPLIT day with the champion state and writes day caches and candidates."""
    skeleton()
    # Champion Task 1 state of the split (one row per timestamp and link, built by ../q/build_champ_state.py).
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
            # Folder name '<key>=<regime>' gives the masking regime of the day.
            regime = path.parent.name.split('=')[1]
            d = pd.read_parquet(path)
            d['link_id'] = d.link_id.astype(str)
            date = str(d.date.iloc[0])
            # The occupancy line is fitted once per panel, on its first day.
            if coef is None:
                coef = occ_model(d, lanes)
            # Gap stamps: stamps at which every link is blank, such as the 90-minute gaps after official windows;
            # they stay blank.
            ts_null = d.groupby('timestamp').speed_kmh.apply(lambda s: s.isna().all())
            gap_ts = set(ts_null[ts_null].index)
            # Task 1 holes outside the gap stamps take the champion speed and flow, as a republished window
            # history would show them.
            target = (d.is_score_eligible == 1) & d.speed_kmh.isna() & ~d.timestamp.isin(gap_ts)
            key = pd.MultiIndex.from_frame(d.loc[target, ['timestamp', 'link_id']])
            got = cpk.reindex(key)
            d['filled'] = False
            d.loc[target, 'speed_kmh'] = got.speed_kmh.to_numpy()
            d.loc[target, 'flow_vph'] = got.flow_vph.to_numpy()
            d.loc[target, 'filled'] = got.speed_kmh.notna().to_numpy()
            # Occupancy of filled cells from the fitted occupancy-density line.
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
            # vc: queue speed, 60% of free-flow speed (63 km/h for a link missing from links.csv).
            vc = np.array([vcut_map.get(k, 63.0) for k in order])[:, None]
            # Qobs: queued on eligible cells (observed or filled). Qc: the clean queue state, from speeds
            # interpolated along time, so every cell has a state.
            Qobs = (V <= vc) & E
            Vf = pd.DataFrame(V).interpolate(axis=1, limit_direction='both').to_numpy()
            Qc = Vf <= vc
            gap = np.array([t in gap_ts for t in times])
            LabM = E & ~F & ~np.isnan(V)  # observed, eligible, not filled
            # Candidate origins t: 12 history steps (t-12..t-1) and 6 horizon steps (t+1..t+6), none at a gap stamp.
            for t in range(12, len(times) - 6):
                if gap[t - 12:t + 7].any():
                    continue
                # cov: eligible share of the history; hor: clean queue state over the horizon.
                cov = E[:, t - 12:t].mean()
                hor = Qc[:, t + 1:t + 7]; lastc = Qc[:, t - 1]
                # piou: IoU between the last history state held over the horizon (persistence) and the horizon.
                pers = np.repeat(lastc[:, None], 6, axis=1)
                uni = (pers | hor).sum(); piou = (pers & hor).sum() / uni if uni else 1.0
                # first: first horizon step (1..6) with a clean queue, 0 when there is none.
                first = int(np.argmax(hor.any(0))) + 1 if hor.any() else 0
                cand_rows.append((panel, date, regime, t, cov, bool(Qobs[:, t - 12:t].any()), bool(Qobs[:, t - 1].any()),
                                  bool(hor.any()), piou, first))
            np.savez_compressed(SP / 'days' / f'{panel}_{date}.npz', V=V, E=E, F=F, Qc=Qc, LabM=LabM, vc=vc,
                                links=np.array(order), times=times.astype(str))
            d[['timestamp','link_id','speed_kmh','flow_vph','occupancy','pct_observed','is_score_eligible','filled']].to_parquet(SP / 'days' / f'{panel}_{date}.parquet', index=False)
            print(panel, date, regime, 'filled', int(d.filled.sum()), 'gap_ts', len(gap_ts), flush=True)
    # Candidate columns: panel, date, regime, t (origin column), cov, hist_any (observed queue in the history),
    # last_any (observed queue at t-1), hor_any (clean queue in the horizon), piou, first.
    c = pd.DataFrame(cand_rows, columns=['panel', 'date', 'regime', 't', 'cov', 'hist_any', 'last_any', 'hor_any', 'piou', 'first'])
    c.to_parquet(SP / f'cands_{SPLIT}.parquet', index=False)


if __name__ == '__main__':
    (SP / 'days').mkdir(exist_ok=True)
    main()
