"""Same-split recency queue prior (legal: only days strictly before the window's date, same split, masked layer).
rec_s   : share of prior days with an observed queue at (link, T+5s +/-10 min)
rec_any : share of prior days with an observed queue at the link anywhere in [T-30, T+60]
rec_n   : number of prior days used (0..KMAX)
Train and pseudo-val windows draw K in 0..KMAX from a hash so the distribution matches official windows (days 1-6).
Usage: build_recency.py train <cond> | pv <cond> | real <split> <cond>
"""
#
# Role: reproduction pipeline (ongoing.sh, steps rec_val / rec_pri: `real validation queue_ongoing` and
# `real private queue_ongoing`). The train and pv modes are the record of the training and ruler tables.
# Reads the released masked layer (mainline_states_masked) and network links.csv (free-flow speed).
# Output: dense/rec_<tag>.parquet with tag train_<cond>_s<stride>, pv_<cond> or real<split[:4]>_<cond>
# (e.g. realvali_queue_ongoing); columns window_id, link_id, step, rec_s, rec_any (float32), rec_n.
import sys, glob, hashlib, functools
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); REL = ROOT.parent / 'tfb_competition_data' / 'kaggle_public'
HERE = Path(__file__).resolve().parent
# RUL: the March pseudo-val ruler folder (pseudo windows cut from the validation month's observed cells).
RUL = ROOT / 'build/review_20260924_ctm/ruler'
# KMAX: at most 5 earlier days, the number an official window on day 6 of the month can see.
KMAX = 5


@functools.lru_cache(maxsize=None)
def day_index(panel, split):
    """Returns {date 'YYYY-MM-DD': masked-layer day file} for one panel and split (cached)."""
    fs = [p for p in glob.glob(str(REL / 'corridors' / panel / split / 'mainline_states_masked' / '*' / '*.parquet')) if '/._' not in p]
    return {Path(p).stem.replace('synthetic_mainline_', '').replace('_', '-'): p for p in fs}


@functools.lru_cache(maxsize=400)
def day_q(panel, split, date):
    """Returns the observed queue state of one masked-layer day (cached, up to 400 days).

    Args:
        panel: corridor name, e.g. 'D7_I210_W'.
        split: 'train', 'validation' or 'private'.
        date: 'YYYY-MM-DD'.

    Returns:
        (li, Q): li maps link_id to a row of Q; Q is int8 (links x 288 five-minute steps) with -1 for an
        unobserved or ineligible cell, 0 for free flow and 1 for queued (speed at or below 60% of the link's
        free-flow speed). Columns follow the day's sorted timestamps.
    """
    f = pd.read_parquet(day_index(panel, split)[date], columns=['timestamp', 'link_id', 'speed_kmh', 'is_score_eligible'])
    f['link_id'] = f.link_id.astype(str)
    links = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str})
    vc = dict(zip(links.link_id, 0.6 * links.free_speed_kmh))
    times = np.sort(f.timestamp.unique()); ti = {t: i for i, t in enumerate(times)}
    L = sorted(f.link_id.unique()); li = {l: i for i, l in enumerate(L)}
    Q = np.zeros((len(L), 288), np.int8) - 1   # -1 unobserved, 0 free, 1 queued
    ok = f.speed_kmh.notna() & (f.is_score_eligible == 1)
    g = f[ok]
    Q[g.link_id.map(li).to_numpy(), g.timestamp.map(ti).to_numpy()] = (g.speed_kmh.to_numpy() <= g.link_id.map(vc).to_numpy()).astype(np.int8)
    return li, Q


def kdraw(wid):
    """Returns a K in 0..KMAX drawn deterministically from the MD5 hash of the window id."""
    return int(hashlib.md5(wid.encode()).hexdigest(), 16) % (KMAX + 1)


def feats_for(panel, split, date, t, links, K):
    """Builds the recency features of one window from the K most recent earlier days of the same split.

    Args:
        panel: corridor name.
        split: split whose masked layer is read; only days strictly before `date` are used.
        date: window date 'YYYY-MM-DD'.
        t: forecast origin T as a 5-minute step of the day.
        links: link ids to report, one output row per link and horizon step.
        K: number of earlier days to use (0 gives NaN features).

    Returns:
        DataFrame with 6 rows per link (step 1..6 = T+5 .. T+30 min): link_id, step, rec_s, rec_any, rec_n.
    """
    dates = sorted(d for d in day_index(panel, split) if d < date)[-K:] if K > 0 else []
    n = len(links); rec = np.full((n, 6), np.nan); anyq = np.full(n, np.nan)
    if dates:
        num = np.zeros((n, 6)); den = np.zeros((n, 6)); na = np.zeros(n); da = np.zeros(n)
        for d in dates:
            li, Q = day_q(panel, split, d); idx = np.array([li.get(l, -1) for l in links])
            ok = idx >= 0
            # rec_s: per horizon step, a day counts as queued when any cell within +/-10 min (5 steps) is queued.
            for s in range(1, 7):
                c = t + s; lo, hi = max(0, c - 2), min(288, c + 3)
                blk = Q[idx[ok], lo:hi]
                seen = (blk >= 0).any(1); q = (blk == 1).any(1)
                num[ok, s - 1] += q; den[ok, s - 1] += seen
            # rec_any: any queued cell in [T-30, T+60] on that day.
            blk = Q[idx[ok], max(0, t - 6):min(288, t + 13)]
            na[ok] += (blk == 1).any(1); da[ok] += (blk >= 0).any(1)
        rec = np.where(den > 0, num / np.maximum(den, 1), np.nan); anyq = np.where(da > 0, na / np.maximum(da, 1), np.nan)
    rows = []
    for s in range(1, 7):
        rows.append(pd.DataFrame(dict(link_id=links, step=s, rec_s=rec[:, s - 1], rec_any=anyq, rec_n=len(dates))))
    return pd.concat(rows, ignore_index=True)


def main():
    """Builds the recency table for the mode given on the command line and writes rec_<tag>.parquet."""
    mode = sys.argv[1]; out = []
    # train: replicated train windows of the given stride; K from the hash so the spread of K matches days 1-6.
    if mode == 'train':
        cond = sys.argv[2]
        stride = sys.argv[3]
        ids = pd.concat([pd.read_parquet(p, columns=['window_id', 'panel', 'date']).drop_duplicates('window_id') for p in sorted(glob.glob(str(HERE / f'dense_s{stride}_*.parquet')))])
        ids['origin'] = ids.window_id.str.split('|').str[2].astype(int)
        u = ids
        for w in u.itertuples():
            links = pd.read_csv(REL / 'corridors' / w.panel / 'network' / 'links.csv', dtype={'link_id': str}).link_id.tolist()
            f = feats_for(w.panel, 'train', w.date, int(w.origin), links, kdraw(w.window_id)); f['window_id'] = w.window_id; out.append(f)
        tag = f'train_{cond}_s{stride}'
    else:
        # pv: March pseudo-val windows (validation split), K from the hash.
        if mode == 'pv':
            cond = sys.argv[2]; split = 'validation'
            # The first read is replaced by the glob over all panels on the next line.
            wi = pd.read_csv(RUL / 'fakeval' / 'root' / 'task2' / 'D7_I10_E' / 'validation' / 'window_index.csv')
            wi = pd.concat([pd.read_csv(p) for p in glob.glob(str(RUL / 'fakeval' / 'root' / 'task2' / '*' / 'validation' / 'window_index.csv'))])
            tag = f'pv_{cond}'
        # real: the released official windows of a scored split; K = KMAX, so day d of the month uses min(d-1, 5)
        # earlier days of that month.
        else:
            split, cond = sys.argv[2], sys.argv[3]
            wi = pd.concat([pd.read_csv(p) for p in glob.glob(str(REL / 'task2' / '*' / split / 'window_index.csv'))])
            tag = f'real{split[:4]}_{cond}'
        wi = wi[wi.condition == cond]
        for w in wi.itertuples():
            T = pd.Timestamp(w.forecast_origin); t = T.hour * 12 + T.minute // 5; date = T.strftime('%Y-%m-%d')
            links = pd.read_csv(REL / 'corridors' / w.panel / 'network' / 'links.csv', dtype={'link_id': str}).link_id.tolist()
            K = kdraw(w.window_id) if mode == 'pv' else KMAX
            f = feats_for(w.panel, split, date, t, links, K); f['window_id'] = w.window_id; out.append(f)
    d = pd.concat(out, ignore_index=True)
    for c in ['rec_s', 'rec_any']: d[c] = d[c].astype('float32')
    d.to_parquet(HERE / f'rec_{tag}.parquet', index=False); print(tag, len(d), d.rec_n.value_counts().sort_index().to_dict(), flush=True)


if __name__ == '__main__':
    main()
