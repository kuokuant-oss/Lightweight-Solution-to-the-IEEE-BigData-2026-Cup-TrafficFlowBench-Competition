"""Replicated official selector on pseudo-truth days; write pseudo Task 2 split files + labels."""
#
# Role: reproduction pipeline (task1.sh step 5, `select_write_v2.py validation`), after cands_v2.py. The pseudo
# windows feed the Task 1 harness (round2/task1_trans/build_ph.py builds pseudo holes from them).
# Usage: select_write_v2.py [split]   (default validation)
# Inputs: cands_v2.parquet, ruler/fakeval/days/<panel>_<date>.npz and .parquet.
# Outputs:
#   root/task2/<panel>/<split>/window_index.csv, window_history.parquet, sample_submission_queue.csv
#                                (pseudo Task 2 split in the released file format; sample predictions are 0)
#   labels_<split>.parquet       per window, horizon step and link: label, mask, clean, elig
#   windows_<split>.parquet      unique windows with mult and window_id
import sys, numpy as np, pandas as pd
from pathlib import Path
SP = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924_ctm/ruler/fakeval')
OUT2 = Path(__file__).resolve().parent
FAKE = OUT2 / 'root'
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'validation'
FAM = {'D7_I10_E': 'D7_I10', 'D7_I10_W': 'D7_I10', 'D7_I210_E': 'D7_I210', 'D7_I210_W': 'D7_I210', 'D7_I405_N': 'D7_I405',
       'D7_I405_S': 'D7_I405', 'D12_I5_N': 'D12_I5', 'D12_I5_S': 'D12_I5'}
c = pd.read_parquet(OUT2 / 'cands_v2.parquet')
# d0: first day of the month; candidate times are counted in minutes from its midnight for the spacing rule.
dates = sorted(c.date.unique()); d0 = pd.Timestamp(dates[0])
by = {(p, d): g for (p, d), g in c.groupby(['panel', 'date'])}


def select(panel, s):
    """Replays the official window selector on one panel from start day s.

    Scans days s, s+1, ... and the candidate origins of each day in time order. A candidate needs history
    coverage >= 0.7 and a clean queue in the horizon; it is an onset window when the history has no observed
    queue and an ongoing window when the last history steps do (ongoing windows with persistence IoU > 0.9
    are rejected). At most 5 windows per condition, and at least 360 minutes between two windows.

    Args:
        panel: corridor name.
        s: index of the start day in `dates`.

    Returns:
        List of (panel, date, regime, t, condition, s, piou, first) tuples.
    """
    sel = []; last = None; cnt = {'queue_onset': 0, 'queue_ongoing': 0}
    for date in dates[s:]:
        g = by.get((panel, date))
        if g is None: continue
        off = (pd.Timestamp(date) - d0).days * 1440
        for r in g.itertuples():
            if cnt['queue_onset'] >= 5 and cnt['queue_ongoing'] >= 5: return sel
            if r.cov < 0.7 or not r.hor_any: continue
            if not r.hist_any: cond = 'queue_onset'
            elif r.last_any: cond = 'queue_ongoing'
            else: continue
            if cond == 'queue_ongoing' and r.piou > 0.9: continue
            if cnt[cond] >= 5: continue
            tm = off + r.t * 5
            if last is not None and tm - last < 360: continue
            sel.append((panel, date, r.regime, int(r.t), cond, s, float(r.piou), int(r.first))); cnt[cond] += 1; last = tm
    return sel


# The selector is replayed once per start day, from every day of the month.
rows = []
for panel in sorted(c.panel.unique()):
    for s in range(len(dates)):
        rows.extend(select(panel, s))
w = pd.DataFrame(rows, columns=['panel', 'date', 'regime', 't', 'condition', 'start', 'piou', 'first'])
# u: unique windows; mult = number of start days whose replay picks the window.
u = w.groupby(['panel', 'date', 'regime', 't', 'condition', 'piou', 'first']).size().rename('mult').reset_index()
print(u.groupby('condition').size(), u.groupby(['condition', 'first']).size())
u['window_id'] = [f'PV_{p}_{d}_{t:03d}' for p, d, t in zip(u.panel, u.date, u.t)]
idx_rows, hist_parts, lab_parts, sub_parts = [], [], [], []
for (panel, date), g in u.groupby(['panel', 'date']):
    z = np.load(SP / 'days' / f'{panel}_{date}.npz', allow_pickle=True)
    V, E, F, Qc, LabM, vc = z['V'], z['E'], z['F'], z['Qc'], z['LabM'], z['vc']
    links = z['links'].astype(str); times = [str(x) for x in z['times']]
    day = pd.read_parquet(SP / 'days' / f'{panel}_{date}.parquet')
    for r in g.itertuples():
        t = r.t
        # Window history: the 12 steps t-12..t-1 of the filled day (as a republished history would be).
        hts = set(times[t - 12:t])
        h = day[day.timestamp.isin(hts)].copy(); h.insert(0, 'window_id', r.window_id)
        hist_parts.append(h[['window_id', 'timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'occupancy', 'pct_observed', 'is_score_eligible']])
        idx_rows.append(dict(window_id=r.window_id, panel=panel, family_id=FAM[panel], split=SPLIT, date=date,
                             forecast_origin=times[t], history_start=times[t - 12], history_end=times[t],
                             forecast_start=times[t + 1], forecast_end=times[t + 6], condition=r.condition,
                             history_coverage=round(float(E[:, t - 12:t].mean()), 4), future_coverage=1.0))
        # Labels: observed queue on label cells (observed, eligible, not filled); clean: time-interpolated state.
        for k in range(1, 7):
            lab = (V[:, t + k] <= vc[:, 0]) & LabM[:, t + k]
            lab_parts.append(pd.DataFrame(dict(window_id=r.window_id, timestamp=times[t + k], link_id=links, step=k,
                                               label=lab, mask=LabM[:, t + k], clean=Qc[:, t + k], elig=E[:, t + k])))
            sub_parts.append(pd.DataFrame(dict(window_id=r.window_id, timestamp=times[t + k], link_id=links, queue_pred=0)))
idx = pd.DataFrame(idx_rows); hist = pd.concat(hist_parts, ignore_index=True)
lab = pd.concat(lab_parts, ignore_index=True); sub = pd.concat(sub_parts, ignore_index=True)
for panel in sorted(u.panel.unique()):
    out = FAKE / 'task2' / panel / SPLIT; out.mkdir(parents=True, exist_ok=True)
    idx[idx.panel == panel].to_csv(out / 'window_index.csv', index=False)
    ids = set(idx.window_id[idx.panel == panel])
    hist[hist.window_id.isin(ids)].to_parquet(out / 'window_history.parquet', index=False)
    sub[sub.window_id.isin(ids)].to_csv(out / 'sample_submission_queue.csv', index=False)
lab.to_parquet(OUT2 / f'labels_{SPLIT}.parquet', index=False)
u.to_parquet(OUT2 / f'windows_{SPLIT}.parquet', index=False)
print(len(idx), 'windows', len(hist), 'hist rows', len(lab), 'label rows')
