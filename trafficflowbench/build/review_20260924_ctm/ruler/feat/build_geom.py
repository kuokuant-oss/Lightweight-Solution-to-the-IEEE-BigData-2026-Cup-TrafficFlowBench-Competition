"""Geometry features for train replicated windows (mode=train <cond>) or serve windows (mode=serve <root> <split> <tag>)."""
#
# Role: reproduction pipeline, ongoing.sh (steps ctm_geom_val / ctm_geom_pri): `build_geom.py serve <release root>
# validation real_validation` and `... private real_private`. Train mode is the record of the training table.
# Usage: build_geom.py train <cond> | serve <release root> <split> <tag>
# Outputs (in this folder): geom_train_<cond>.parquet or geom_serve_<tag>.parquet; one row per (window, link,
# horizon step) with the queue-geometry features of geom.py and window_id.
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from src.tfb import release as R, score2, task2build as TB
from src.tfb.task2pool import date_of
import geom as G


def static(panel, links):
    """Returns (queue speed v_cut per link, km/h, 63 when missing; link length in km, 0.3 when missing)."""
    fd = pd.read_csv(R.RELEASE / 'corridors' / panel / 'network' / 'fd_parameters.csv', dtype={'link_id': str}).set_index('link_id')
    cut = score2.v_cut(panel)
    return np.array([cut.get(k, 63.0) for k in links]), fd.length_km.reindex(links).fillna(0.3).to_numpy(float)


def train(cond):
    """Builds geometry features for the replicated official train windows of one condition from full truth."""
    u = pd.read_parquet(ROOT / 'build/review_20260923/work/e2/replicated_unique.parquet')
    u = u[u.condition == cond]
    out = []
    for panel, wp in u.groupby('panel'):
        cut = score2.v_cut(panel); by = {date_of(p): p for p in R.truth_files(panel, 'train')}
        for date, wd in wp.groupby('date'):
            links, mp, V, Qf, O, E, Q, times = TB.day_arrays(panel, by[date], cut)
            vc, ln = static(panel, links)
            for w in wd.itertuples():
                t = int(w.origin)
                # History t-12..t-1 (T-60..T-5).
                f = G.step_frame(links, G.geom_features(V[:, t - 12:t], Qf[:, t - 12:t], E[:, t - 12:t], vc, ln))
                f['window_id'] = f'{panel}|{date}|{t}'; out.append(f)
        print(panel, flush=True)
    d = pd.concat(out, ignore_index=True)
    for c in d.select_dtypes('float64').columns: d[c] = d[c].astype('float32')
    d.to_parquet(HERE / f'geom_train_{cond}.parquet', index=False)


def serve(root, split, tag):
    """Builds geometry features for the official windows of a scored split from their window history only.

    Links are ordered by the mileposts of the panel's first train day file.
    """
    R.RELEASE = Path(root)
    wi = score2.window_index(split)
    out = []
    for panel in sorted(wi.panel.unique()):
        hist = pd.read_parquet(R.RELEASE / 'task2' / panel / split / 'window_history.parquet')
        hist['link_id'] = hist.link_id.astype(str)
        truth_mp = pd.read_parquet(R.truth_files(panel, 'train')[0], columns=['link_id', 'milepost']).drop_duplicates()
        truth_mp['link_id'] = truth_mp.link_id.astype(str)
        for wid, h in hist.groupby('window_id'):
            order = truth_mp[truth_mp.link_id.isin(set(h.link_id))].sort_values('milepost')
            links = list(order.link_id); li = {k: i for i, k in enumerate(links)}
            times = np.sort(h.timestamp.unique()); ti = {k: i for i, k in enumerate(times)}
            r = h.link_id.map(li).to_numpy(); c = h.timestamp.map(ti).to_numpy()
            V = np.full((len(links), len(times)), np.nan); Qf = V.copy(); E = np.zeros(V.shape, bool)
            V[r, c] = h.speed_kmh.to_numpy(); Qf[r, c] = h.flow_vph.to_numpy(); E[r, c] = h.is_score_eligible.astype(bool).to_numpy()
            vc, ln = static(panel, links)
            f = G.step_frame(links, G.geom_features(V, Qf, E, vc, ln)); f['window_id'] = wid; out.append(f)
        print(panel, flush=True)
    d = pd.concat(out, ignore_index=True)
    for c in d.select_dtypes('float64').columns: d[c] = d[c].astype('float32')
    d.to_parquet(HERE / f'geom_serve_{tag}.parquet', index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'train': train(sys.argv[2])
    else: serve(sys.argv[2], sys.argv[3], sys.argv[4])
