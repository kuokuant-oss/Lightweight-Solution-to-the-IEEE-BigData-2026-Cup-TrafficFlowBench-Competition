"""Score queue predictions on pseudo windows.  score(pred_df, split) -> dict, per-window frame."""
import numpy as np, pandas as pd
from pathlib import Path
SP = Path(__file__).resolve().parent
FAM = {'D7_I10_E': 'D7_I10', 'D7_I10_W': 'D7_I10', 'D7_I210_E': 'D7_I210', 'D7_I210_W': 'D7_I210', 'D7_I405_N': 'D7_I405',
       'D7_I405_S': 'D7_I405', 'D12_I5_N': 'D12_I5', 'D12_I5_S': 'D12_I5'}
_LAB = {}


def labels(split='validation'):
    if split not in _LAB:
        lab = pd.read_parquet(SP / f'labels_{split}.parquet')
        w = pd.read_parquet(SP / f'windows_{split}.parquet')[['window_id', 'panel', 'condition', 'regime', 'date', 'mult', 'first']]
        _LAB[split] = (lab, w)
    return _LAB[split]


def agg(fr, col='iou'):
    pc = fr.groupby(['panel', 'condition'])[col].mean().reset_index()
    out = {}
    for cond, g in pc.groupby('condition'):
        pm = g.set_index('panel')[col]; out[cond] = float(pm.groupby(pm.index.map(FAM)).mean().mean())
    pm = pc.groupby('panel')[col].mean(); out['S_queue'] = float(pm.groupby(pm.index.map(FAM)).mean().mean())
    return out


def score(pred, split='validation', windows=None):
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
    a = agg(fr, 'iou'); b = agg(fr, 'iou_clean')
    print(f'{name:28s} obs: onset {a.get("queue_onset", np.nan):.4f} ongoing {a.get("queue_ongoing", np.nan):.4f} S {a["S_queue"]:.4f} | clean: onset {b.get("queue_onset", np.nan):.4f} ongoing {b.get("queue_ongoing", np.nan):.4f}  (n={fr.groupby("condition").size().to_dict()})')
    return a, b


def persistence(split='validation'):
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


def _norm(pred):
    """Normalise a prediction frame to the label keys: timestamp '%Y-%m-%dT%H:%M:%SZ' strings, link_id str, int queue_pred."""
    p = pred[['window_id', 'timestamp', 'link_id', 'queue_pred']].copy()
    p['window_id'] = p.window_id.astype(str); p['link_id'] = p.link_id.astype(str)
    p['timestamp'] = pd.to_datetime(p.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    p['queue_pred'] = p.queue_pred.astype(int)
    assert not p.duplicated(['window_id', 'timestamp', 'link_id']).any(), 'duplicate prediction keys'
    return p


def evaluate(pred, split='validation', conditions=None, verbose=False):
    """RULER ENTRY POINT.  pred: DataFrame(window_id, timestamp, link_id, queue_pred) on the pseudo windows
    (missing cells count as 0).  Returns (summary dict, per-window frame).

    summary keys (all use the official aggregation window IoU -> panel x condition -> panel -> family):
      onset         obs labels, windows with >=1 observed positive label (the REPORT's onset number; 155 windows)
      ongoing       obs labels, all 305 ongoing windows
      overall       S_queue over {onset obs>0 windows} + {all ongoing windows}
      onset_all     obs labels, all 213 onset windows (windows with no observed positive score 1 if pred empty)
      onset_clean / ongoing_clean   'clean' labels (time-interpolated speeds, no mask), all windows of the condition
      overall_clean S_queue on clean labels, all windows
    The per-window frame has iou, iou_clean, n_true, n_pred, panel, condition, regime, date, mult, first
    and a flag `onset_obs_pos` (condition == onset and n_true > 0)."""
    fr = score(_norm(pred), split)
    if conditions is not None:
        fr = fr[fr.condition.isin(conditions)]
    fr['onset_obs_pos'] = fr.condition.eq('queue_onset') & (fr.n_true > 0)
    keep = fr.condition.eq('queue_ongoing') | fr.onset_obs_pos
    a = agg(fr[keep], 'iou'); a_all = agg(fr, 'iou'); c = agg(fr, 'iou_clean')
    out = dict(onset=a.get('queue_onset', np.nan), ongoing=a.get('queue_ongoing', np.nan), overall=a['S_queue'],
               onset_all=a_all.get('queue_onset', np.nan), onset_clean=c.get('queue_onset', np.nan),
               ongoing_clean=c.get('queue_ongoing', np.nan), overall_clean=c['S_queue'],
               n_onset_obs_pos=int(fr.onset_obs_pos.sum()), n_onset=int(fr.condition.eq('queue_onset').sum()),
               n_ongoing=int(fr.condition.eq('queue_ongoing').sum()))
    if verbose:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    return out, fr


def paired_boot(fa, fb, condition, n=1000, seed=0, obs_pos_only=None):
    """Date-bootstrap CI of the aggregated difference b - a for one condition (per-window frames from evaluate())."""
    if obs_pos_only is None: obs_pos_only = condition == 'queue_onset'
    ka = fa[fa.condition.eq(condition)]; ka = ka[ka.n_true > 0] if obs_pos_only else ka
    m = ka[['window_id', 'date', 'panel', 'condition', 'iou']].merge(fb[['window_id', 'iou']], on='window_id', suffixes=('_a', '_b'))
    rng = np.random.default_rng(seed); dates = m.date.unique(); bs = []
    for _ in range(n):
        mm = pd.concat([m[m.date == x] for x in rng.choice(dates, len(dates))])
        bs.append(agg(mm.rename(columns={'iou_b': 'iou'}))[condition] - agg(mm.rename(columns={'iou_a': 'iou'}))[condition])
    d = m.iou_b - m.iou_a
    return dict(delta=agg(m.rename(columns={'iou_b': 'iou'}))[condition] - agg(m.rename(columns={'iou_a': 'iou'}))[condition],
                ci=tuple(np.percentile(bs, [2.5, 97.5])), better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()))


if __name__ == '__main__':
    import sys
    # python score.py validation [pred.parquet ...]            -> legacy report lines
    # python score.py eval pred.parquet [perwin_out.parquet]    -> evaluate() summary (+ per-window table)
    if len(sys.argv) > 2 and sys.argv[1] == 'eval':
        f = sys.argv[2]
        pred = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
        s, fr = evaluate(pred, 'validation', verbose=True)
        if len(sys.argv) > 3: fr.to_parquet(sys.argv[3], index=False)
        sys.exit(0)
    split = sys.argv[1] if len(sys.argv) > 1 else 'validation'
    fr = score(persistence(split), split); report(fr, 'persistence')
    for f in sys.argv[2:]:
        fr = score(pd.read_parquet(f), split); report(fr, Path(f).stem)
