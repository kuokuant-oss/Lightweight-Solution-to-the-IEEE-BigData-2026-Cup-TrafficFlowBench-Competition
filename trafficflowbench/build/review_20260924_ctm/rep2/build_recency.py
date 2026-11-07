"""Same-split recency queue prior (legal: only days strictly before the window's date, same split, masked layer).
rec_s   : share of prior days with an observed queue at (link, T+5s +/-10 min)
rec_any : share of prior days with an observed queue at the link anywhere in [T-30, T+60]
rec_n   : number of prior days used (0..KMAX)
Train and pseudo-val windows draw K in 0..KMAX from a hash so the distribution matches official windows (days 1-6).
Usage: build_recency.py train <cond> | pv <cond> | real <split> <cond>
"""
import sys, glob, hashlib, functools
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); REL = ROOT.parent / 'tfb_competition_data' / 'kaggle_public'
HERE = Path(__file__).resolve().parent
RUL = ROOT / 'build/review_20260924_ctm/ruler2'
KMAX = 5


@functools.lru_cache(maxsize=None)
def day_index(panel, split):
    fs = [p for p in glob.glob(str(REL / 'corridors' / panel / split / 'mainline_states_masked' / '*' / '*.parquet')) if '/._' not in p]
    return {Path(p).stem.replace('synthetic_mainline_', '').replace('_', '-'): p for p in fs}


@functools.lru_cache(maxsize=400)
def day_q(panel, split, date):
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
    return int(hashlib.md5(wid.encode()).hexdigest(), 16) % (KMAX + 1)


def feats_for(panel, split, date, t, links, K):
    dates = sorted(d for d in day_index(panel, split) if d < date)[-K:] if K > 0 else []
    n = len(links); rec = np.full((n, 6), np.nan); anyq = np.full(n, np.nan)
    if dates:
        num = np.zeros((n, 6)); den = np.zeros((n, 6)); na = np.zeros(n); da = np.zeros(n)
        for d in dates:
            li, Q = day_q(panel, split, d); idx = np.array([li.get(l, -1) for l in links])
            ok = idx >= 0
            for s in range(1, 7):
                c = t + s; lo, hi = max(0, c - 2), min(288, c + 3)
                blk = Q[idx[ok], lo:hi]
                seen = (blk >= 0).any(1); q = (blk == 1).any(1)
                num[ok, s - 1] += q; den[ok, s - 1] += seen
            blk = Q[idx[ok], max(0, t - 6):min(288, t + 13)]
            na[ok] += (blk == 1).any(1); da[ok] += (blk >= 0).any(1)
        rec = np.where(den > 0, num / np.maximum(den, 1), np.nan); anyq = np.where(da > 0, na / np.maximum(da, 1), np.nan)
    rows = []
    for s in range(1, 7):
        rows.append(pd.DataFrame(dict(link_id=links, step=s, rec_s=rec[:, s - 1], rec_any=anyq, rec_n=len(dates))))
    return pd.concat(rows, ignore_index=True)


def main():
    mode = sys.argv[1]; out = []
    if mode == 'train':
        cond = sys.argv[2]
        stride = sys.argv[3]
        ids = pd.concat([pd.read_parquet(p, columns=['window_id', 'panel', 'date']).drop_duplicates('window_id') for p in sorted(glob.glob(str(HERE / f'rep2_s{stride}_*.parquet')))])
        ids['origin'] = ids.window_id.str.split('|').str[2].astype(int)
        u = ids
        for w in u.itertuples():
            links = pd.read_csv(REL / 'corridors' / w.panel / 'network' / 'links.csv', dtype={'link_id': str}).link_id.tolist()
            f = feats_for(w.panel, 'train', w.date, int(w.origin), links, kdraw(w.window_id)); f['window_id'] = w.window_id; out.append(f)
        tag = f'train_{cond}_s{stride}'
    else:
        if mode == 'pv':
            cond = sys.argv[2]; split = 'validation'
            wi = pd.read_csv(RUL / 'root' / 'task2' / 'D7_I10_E' / 'validation' / 'window_index.csv')
            wi = pd.concat([pd.read_csv(p) for p in glob.glob(str(RUL / 'root' / 'task2' / '*' / 'validation' / 'window_index.csv'))])
            tag = f'pv2_{cond}'
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
    d.to_parquet(RUL / f'rec_{tag}.parquet', index=False); print(tag, len(d), d.rec_n.value_counts().sort_index().to_dict(), flush=True)


if __name__ == '__main__':
    main()
