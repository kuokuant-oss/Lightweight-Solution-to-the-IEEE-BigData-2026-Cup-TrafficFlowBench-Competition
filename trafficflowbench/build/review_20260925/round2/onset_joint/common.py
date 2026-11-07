"""Shared harness for the onset_joint lens.
- ruler: v1 onset (pseudo-val) through candidate D's decoder (copy of onset/d_onset_adapt.py 'D' path);
  eval_choice(scores) replaces the site choice on the 3 multi-site panels and returns onset, perwin, paired boot vs D.
- in-world: replicated train onset windows (rep2/replicated_unique_v2.parquet) x per-site train rows (onset/train_rows_*.parquet).
Nothing here writes outside round2/onset_joint."""
import os, sys, pickle, warnings
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
ME = Path(__file__).resolve().parent
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
OS = ROOT / 'build/review_20260924_ctm/ruler/onset_site'; RUL = OS.parent
ONS = ROOT / 'build/review_20260925/onset'
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
sys.path.insert(0, str(OS)); sys.path.insert(0, str(RUL / 'fakeval'))
_cwd = os.getcwd(); os.chdir(OS)
import score as S
from pv_sites import SITES, site_of
import serve_official as SO
from serve_official import score_rows
from eval_template import apply, MULTI
from oracle_site import pred0
import site_feats as SF
os.chdir(_cwd)
MS = ['D12_I5_N', 'D12_I5_S', 'D7_I405_S']          # panels where D uses the site choice
BN = SF.BN

PK = pickle.load(open(OS / 'u_models_C1.0.pkl', 'rb')); UMODELS, TPL = PK['models'], PK['tpl']
SITE_PV = None  # package edit (2026-11-07): March pseudo-val site rows not loaded (ruler only, not shipped)

# ---- n train onset events per site (D's eligibility filter, >= 10) ----
_ev = pd.read_parquet(OS / 'events_train.parquet'); _rows = []
for (panel, di, tq), gg in _ev.groupby(['panel', 'di', 'tq']):
    if tq - 6 < 12: continue
    true = set()
    for r in gg.itertuples():
        for i in range(r.a, r.b + 1): true.add(site_of(panel, i))
    for s in BN[panel]: _rows.append((panel, s, int(s in true)))
NEV = pd.DataFrame(_rows, columns=['panel', 'site', 'y']).query('y == 1').groupby(['panel', 'site']).size()

B = np.load(RUL / 'blocks' / 'my_blocks.npy', allow_pickle=True).item(); BID = {p: dict(zip(v['links'], v['bid'])) for p, v in B.items()}


def block_decode(pred, t=0.7):
    p = pred[pred.condition == 'queue_onset'].copy(); p['bid'] = [BID[a][b] for a, b in zip(p.panel, p.link_id)]
    y = np.zeros(len(p), bool)
    for wid, ix in p.groupby('window_id', sort=False).indices.items():
        pr = p.prob.to_numpy()[ix]; b = p.bid.to_numpy()[ix]; s = pd.Series(pr).groupby(b).mean(); sel = set(s[s >= t].index) | {s.idxmax()}
        y[ix] = np.isin(b, list(sel))
    out = pred.copy(); out.loc[out.condition == 'queue_onset', 'queue_pred'] = y.astype(int); return out


def mix(site_pred, blk_pred):
    out = site_pred.copy(); on = out.condition == 'queue_onset'; nonmulti = on & ~out.panel.isin(MULTI)
    bb = blk_pred.set_index(['window_id', 'timestamp', 'link_id']).queue_pred
    key = pd.MultiIndex.from_frame(out.loc[nonmulti, ['window_id', 'timestamp', 'link_id']]); out.loc[nonmulti, 'queue_pred'] = bb.reindex(key).to_numpy(); return out


def onset_only(pred): return pred[pred.condition == 'queue_onset'][['window_id', 'timestamp', 'link_id', 'panel', 'prob', 'queue_pred']].reset_index(drop=True)


BLK = U_PV = None  # package edit (2026-11-07): March pseudo-val ruler inputs not built at import (ruler only)


def choices(uu, col='u', skip_from=None):
    """D's choice rule: argmax over sites with >=10 train events; skip window if max champion u < 0.01.
    skip_from: frame with champion u (default uu itself) so that the skip set is the same as D's."""
    ch = {}
    sk = uu if skip_from is None else skip_from
    sk = sk[[NEV.get((p, s), 0) >= 10 for p, s in zip(sk.panel, sk.site)]]
    umax = sk.groupby('key').u.max()
    for k, g in uu.groupby('key'):
        if k.startswith('PV_D7_I210_W') or g.panel.iloc[0] not in MS: continue
        g = g[[NEV.get((p, s), 0) >= 10 for p, s in zip(g.panel, g.site)]]
        if len(g) == 0 or umax.get(k, 0) < 0.01: continue
        ch[k] = {g.site.iloc[int(np.nanargmax(g[col].to_numpy()))]}
    return ch


def ruler(ch):
    pr = onset_only(mix(apply(pred0, ch, TPL, 'prod_else_tpl'), BLK))
    s, fr = S.evaluate(pr, conditions=['queue_onset'])
    return s, fr


CH_D = S_D = FR_D = None  # package edit (2026-11-07): D's score on the March pseudo-val ruler not computed at import


def eval_choice(scored, col, name=''):
    """scored: pv site rows with a score column `col` (higher = chosen). Returns dict with onset, vs_D boot, n changed windows."""
    ch = choices(scored, col, skip_from=U_PV)
    s, fr = ruler(ch)
    nchg = sum(1 for k in ch if ch[k] != CH_D.get(k))
    return dict(name=name, onset=round(s['onset'], 4), onset_clean=round(s['onset_clean'], 4), n_changed=nchg,
                vs_D=S.paired_boot(FR_D, fr, 'queue_onset')), fr, ch


def pv_top1(scored, col):
    """pv top-1 on the 3 D panels (windows with a true site), per panel + pooled."""
    out = {}; allv = []
    for p in MS:
        g0 = scored[scored.panel == p]
        a = [g.y.to_numpy()[np.nanargmax(g[col].to_numpy())] for k, g in g0.groupby('key') if g.y.sum() > 0]
        out[p] = (round(float(np.mean(a)), 3), len(a)); allv += a
    out['all3'] = (round(float(np.mean(allv)), 3), len(allv))
    return out


# ---------------- in-world rows ----------------
def train_rows(panel):
    t = pd.read_parquet(ONS / f'train_rows_{panel}.parquet')
    return t


def inworld_windows(panel, t=None):
    """replicated onset windows (train) joined to per-site rows; keeps windows with all sites present."""
    if t is None: t = train_rows(panel)
    u = pd.read_parquet(ROOT / 'build/review_20260924_ctm/rep2/replicated_unique_v2.parquet')
    u = u[(u.condition == 'queue_onset') & (u.panel == panel)][['date', 'origin']].rename(columns={'origin': 'T'})
    ev = t.merge(u, on=['date', 'T']); ev['key'] = ev.date + '|' + ev['T'].astype(str)
    ns = ev.groupby('key').site.nunique(); ev = ev[ev.key.isin(ns[ns == len(BN[panel])].index)]
    return ev


def top1(d, col):
    a = [g.y.to_numpy()[np.nanargmax(g[col].to_numpy())] for k, g in d.groupby('key') if g.y.sum() > 0]
    return float(np.mean(a)), len(a)


def top1_vec(d, col):
    """per-window hit (1/0) for windows with a true site; returns Series indexed by key."""
    return pd.Series({k: g.y.to_numpy()[np.nanargmax(g[col].to_numpy())] for k, g in d.groupby('key') if g.y.sum() > 0})
