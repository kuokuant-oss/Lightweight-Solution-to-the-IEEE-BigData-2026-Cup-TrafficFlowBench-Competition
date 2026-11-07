"""Score queue predictions on pseudo windows.  score(pred_df, split) -> dict, per-window frame."""
# Research scorer for Task 2 queue predictions on pseudo windows cut from the
# validation month (labels_<split>.parquet, windows_<split>.parquet and a
# pseudo release under root/task2/; these files are not shipped). og_exp.py and
# og_ens.py use it to compare ongoing variants. In the reproduction it is only
# imported (through og_exp.py); none of its functions run.
#
# Usage: score.py [split] [prediction.parquet ...]  (scores the persistence
# baseline, then each prediction file).
import numpy as np, pandas as pd
from pathlib import Path
SP = Path(__file__).resolve().parent
# FAM groups the two directions of a freeway into one corridor family; scores
# are averaged per panel, then per family, then over families.
FAM = {'D7_I10_E': 'D7_I10', 'D7_I10_W': 'D7_I10', 'D7_I210_E': 'D7_I210', 'D7_I210_W': 'D7_I210', 'D7_I405_N': 'D7_I405',
       'D7_I405_S': 'D7_I405', 'D12_I5_N': 'D12_I5', 'D12_I5_S': 'D12_I5'}
_LAB = {}


def labels(split='validation'):
    """Returns the cached (cell labels, window table) of a pseudo split."""
    if split not in _LAB:
        lab = pd.read_parquet(SP / f'labels_{split}.parquet')
        w = pd.read_parquet(SP / f'windows_{split}.parquet')[['window_id', 'panel', 'condition', 'regime', 'date', 'mult', 'first']]
        _LAB[split] = (lab, w)
    return _LAB[split]


def agg(fr, col='iou'):
    """Averages per-window scores into per-condition and overall queue scores.

    Args:
      fr: per-window frame with panel, condition and the score column.
      col: score column, 'iou' or 'iou_clean'.

    Returns:
      Dict {condition: score, 'S_queue': score}. Each value is the mean over
      corridor families of the family mean of the per-panel mean.
    """
    pc = fr.groupby(['panel', 'condition'])[col].mean().reset_index()
    out = {}
    for cond, g in pc.groupby('condition'):
        pm = g.set_index('panel')[col]; out[cond] = float(pm.groupby(pm.index.map(FAM)).mean().mean())
    pm = pc.groupby('panel')[col].mean(); out['S_queue'] = float(pm.groupby(pm.index.map(FAM)).mean().mean())
    return out


def score(pred, split='validation', windows=None):
    """Scores queue predictions window by window with IoU.

    Args:
      pred: frame with window_id, timestamp, link_id and queue_pred (0/1).
        Label cells without a prediction count as not queued.
      split: pseudo split name.
      windows: optional window ids to keep.

    Returns:
      One row per window: iou (over label cells with an observed label,
      `mask`), iou_clean (against the clean labels, all cells), n_true,
      n_pred, joined to the window table. IoU is 1 when both sets are empty.
    """
    lab, w = labels(split)
    p = pred[['window_id', 'timestamp', 'link_id', 'queue_pred']]
    m = lab.merge(p, on=['window_id', 'timestamp', 'link_id'], how='left')
    m['queue_pred'] = m.queue_pred.fillna(0).astype(bool)
    rows = []
    for wid, g in m.groupby('window_id', sort=False):
        pr = g.queue_pred.to_numpy(); mk = g['mask'].to_numpy(); lb = g.label.to_numpy(); cl = g.clean.to_numpy()
        a, b = pr[mk], lb[mk]; u = (a | b).sum(); iou = 1.0 if u == 0 else (a & b).sum() / u
        u2 = (pr | cl).sum(); iouc = 1.0 if u2 == 0 else (pr & cl).sum() / u2
        rows.append((wid, iou, iouc, int(b.sum()), int(a.sum())))
    fr = pd.DataFrame(rows, columns=['window_id', 'iou', 'iou_clean', 'n_true', 'n_pred']).merge(w, on='window_id')
    if windows is not None:
        fr = fr[fr.window_id.isin(windows)]
    return fr


def report(fr, name=''):
    """Prints the observed and clean scores of a per-window frame; returns both dicts."""
    a = agg(fr, 'iou'); b = agg(fr, 'iou_clean')
    print(f'{name:28s} obs: onset {a.get("queue_onset", np.nan):.4f} ongoing {a.get("queue_ongoing", np.nan):.4f} S {a["S_queue"]:.4f} | clean: onset {b.get("queue_onset", np.nan):.4f} ongoing {b.get("queue_ongoing", np.nan):.4f}  (n={fr.groupby("condition").size().to_dict()})')
    return a, b


def persistence(split='validation'):
    """Builds the persistence baseline for every pseudo window of a split.

    A link is predicted queued at all six horizon steps when its speed at the
    last history stamp (T-5 min) is at most 0.6 x free-flow speed and the cell
    is score-eligible.

    Returns:
      Frame with window_id, timestamp, link_id and queue_pred (0/1).
    """
    root = SP / 'root' / 'task2'
    out = []
    for panel in FAM:
        wi = pd.read_csv(root / panel / split / 'window_index.csv')
        h = pd.read_parquet(root / panel / split / 'window_history.parquet')
        links = pd.read_csv(f'/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public/corridors/{panel}/network/links.csv', dtype={'link_id': str})
        vc = dict(zip(links.link_id, 0.6 * links.free_speed_kmh))
        last = h[h.timestamp == h.groupby('window_id').timestamp.transform('max')].copy()
        last['q'] = (last.speed_kmh <= last.link_id.map(vc)) & (last.is_score_eligible == 1)
        sub = pd.read_csv(root / panel / split / 'sample_submission_queue.csv', dtype={'link_id': str})
        sub = sub.merge(last[['window_id', 'link_id', 'q']], on=['window_id', 'link_id'], how='left')
        sub['queue_pred'] = sub.q.fillna(False).astype(int)
        out.append(sub[['window_id', 'timestamp', 'link_id', 'queue_pred']])
    return pd.concat(out, ignore_index=True)


if __name__ == '__main__':
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'validation'
    fr = score(persistence(split), split); report(fr, 'persistence')
    for f in sys.argv[2:]:
        fr = score(pd.read_parquet(f), split); report(fr, Path(f).stem)
