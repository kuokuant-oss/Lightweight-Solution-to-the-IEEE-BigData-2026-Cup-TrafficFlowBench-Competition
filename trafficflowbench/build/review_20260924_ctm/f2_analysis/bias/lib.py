"""Shared helpers for the ruler-bias analysis (label: bias).
Reproduces dense/eval_rul.py + dense/mix_eval.py exactly (A-block baseline, selrec6, mix75, block-mean decode),
adds a vectorised date bootstrap identical in construction to score.paired_boot (same RNG draws -> same CI)
that also supports window subsets and importance weights."""
import os
for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[n] = '3'
import sys, importlib
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); CT = ROOT / 'build/review_20260924_ctm'
HERE = Path(__file__).resolve().parent
REL = ROOT.parent / 'tfb_competition_data' / 'kaggle_public'
AMOD = ROOT / 'build/review_20260924_newworld/work/feat/models'
A_NAMES = ['V2_noID_noPrior', 'V2s2', 'V2s3', 'V6_V2geom', 'V6s2', 'V6s3']
SR6 = [CT / f'rep2/m_R_V2selrec_s0_seed{i}.txt' for i in range(3)] + [CT / f'rep2/m_R_V2geomselrec_s0_seed{i}.txt' for i in range(3)]
RD = {'v1': CT / 'ruler/fakeval', 'v2': CT / 'ruler2'}
FAM = {'D7_I10_E': 'D7_I10', 'D7_I10_W': 'D7_I10', 'D7_I210_E': 'D7_I210', 'D7_I210_W': 'D7_I210', 'D7_I405_N': 'D7_I405',
       'D7_I405_S': 'D7_I405', 'D12_I5_N': 'D12_I5', 'D12_I5_S': 'D12_I5'}
_B = np.load(CT / 'ruler/blocks/my_blocks.npy', allow_pickle=True).item()
BID = {k: dict(zip(v['links'], v['bid'])) for k, v in _B.items()}
_BOOST = {}


def booster(p):
    p = str(p)
    if p not in _BOOST: _BOOST[p] = lgb.Booster(model_file=p)
    return _BOOST[p]


def score_mod(rv):
    sp = str(RD[rv])
    for k in list(sys.modules):
        if k == 'score': del sys.modules[k]
    sys.path.insert(0, sp); S = importlib.import_module('score'); sys.path.remove(sp)
    return S


def load_cache(rv):
    """pv cache exactly as eval_rul.py sees it (prev_rej/n_rej from truth candidates, float32 casts, block rec features)."""
    cols = set()
    for n in A_NAMES: cols |= set(booster(AMOD / f'og_{n}.txt').feature_name())
    for m in SR6: cols |= set(booster(m).feature_name())
    cols |= {'window_id', 'timestamp', 'link_id', 'panel', 'step', 'date', 'queue_prior'}
    s = pd.read_parquet(CT / f'dense/pv_cache_{rv}.parquet', columns=sorted(cols))
    s['prev_rej'] = s.prev_rej.astype('float32'); s['n_rej'] = s.n_rej.astype('float32')
    s['bid'] = np.array([BID[a][b] for a, b in zip(s.panel, s.link_id.astype(str))])
    return s


def nan_mask(s, frac, seed=7):
    """window mask used by eval_rul.py for nanfrac (same RNG call sequence)."""
    rng = np.random.default_rng(seed); wids = s.window_id.unique(); mw = set(wids[rng.random(len(wids)) < frac])
    return s.window_id.isin(mw).to_numpy()


def pred_models(s, models):
    return np.mean([booster(m).predict(s[booster(m).feature_name()], num_threads=3) for m in models], axis=0)


def predA(s):
    return pred_models(s, [AMOD / f'og_{n}.txt' for n in A_NAMES])


def predN(s):
    return pred_models(s, SR6)


def block_mean(s, p):
    z = pd.DataFrame({'window_id': s.window_id.values, 'timestamp': s.timestamp.values, 'bid': s.bid.values, 'prob': p})
    return z.groupby(['window_id', 'timestamp', 'bid']).prob.transform('mean').to_numpy()


def evaluate(S, s, qpred):
    z = s[['window_id', 'timestamp', 'link_id']].copy(); z['queue_pred'] = qpred.astype(int)
    summ, fr = S.evaluate(z, conditions=['queue_ongoing'])
    return summ, fr.reset_index(drop=True)


def agg_w(df, col, w=None):
    """official aggregation (panel mean -> family mean -> mean), optional per-window weights."""
    d = df.assign(_v=df[col].to_numpy(), _w=(np.ones(len(df)) if w is None else np.asarray(w, float)))
    d = d[d._w > 0]
    pm = d.groupby('panel').apply(lambda g: np.sum(g._v * g._w) / np.sum(g._w), include_groups=False)
    return float(pm.groupby(pm.index.map(FAM)).mean().mean())


def boot(fa, fb, col='iou', subset=None, w=None, n=1000, seed=0):
    """Date bootstrap of agg(b) - agg(a), same construction as score.paired_boot (dates = m.date.unique() order,
    rng.choice with seed 0).  subset: boolean over merged windows; w: importance weights (aligned with merge)."""
    m = fa[['window_id', 'date', 'panel', col]].merge(fb[['window_id', col]], on='window_id', suffixes=('_a', '_b'))
    if w is not None:
        if isinstance(w, pd.Series): w = m.window_id.map(w).to_numpy()
        m['_w'] = np.asarray(w, float)
    else:
        m['_w'] = 1.0
    if subset is not None:
        if isinstance(subset, (set, list, pd.Index)): subset = m.window_id.isin(subset).to_numpy()
        m = m[np.asarray(subset)].reset_index(drop=True)
    m['d'] = m[col + '_b'] - m[col + '_a']
    delta = agg_w(m, 'd', m._w)
    rng = np.random.default_rng(seed); dates = m.date.unique()
    di = pd.Index(dates).get_indexer(m.date)
    pan = sorted(m.panel.unique()); pi = pd.Index(pan).get_indexer(m.panel)
    fam = sorted(set(FAM[p] for p in pan)); fi = np.array([fam.index(FAM[p]) for p in pan])
    dv = m.d.to_numpy(); wv = m._w.to_numpy()
    bs = np.empty(n)
    for b in range(n):
        ch = rng.choice(dates, len(dates)); cnt = pd.Series(ch).value_counts()
        c = pd.Index(dates).map(lambda x: cnt.get(x, 0)).to_numpy()[di] * wv
        num = np.bincount(pi, c * dv, len(pan)); den = np.bincount(pi, c, len(pan))
        ok = den > 0; pmv = np.where(ok, num / np.where(ok, den, 1), np.nan)
        fv = [np.nanmean(pmv[(fi == k) & ok]) if ((fi == k) & ok).any() else np.nan for k in range(len(fam))]
        bs[b] = np.nanmean(fv)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(delta), lo=float(lo), hi=float(hi), n=int(len(m)), better=int((m.d > 1e-9).sum()), worse=int((m.d < -1e-9).sum()))


def fmt(r):
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] n={r['n']} {r['better']}/{r['worse']}"
