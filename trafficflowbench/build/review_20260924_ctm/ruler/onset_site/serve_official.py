"""Fit the unconditional per-site physical breakdown hazard (u_phys_ph) on train, then serve it on
(a) pseudo-val onset windows (re-check) and (b) OFFICIAL validation/private onset windows.
Legal inputs per window: official window_history (T-60..T-5) + masked layer at T (origin timestamp).
Output: site scores per window and a modified onset component (prod_else_tpl decoder) for multi-site panels."""
#
# Role: official_rows, score_rows and decode are imported by pipeline/onset_c3.py (reproduction pipeline,
# onset.sh step 2), which scores with the frozen u-models and templates (u_models_C1.0.pkl). onset_c3.py points
# HERE to a folder whose cache/ holds the masked-layer caches of both official splits (cache/build_cache.py). The
# __main__ block is the record of how the models, templates and the site-choice component were made.
# Usage (record): serve_official.py [C]   (C: inverse regularisation strength of the logistic models)
import sys, pickle, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pv_sites import SITES, site_of
from site_feats import BN
from uncond import build_train, series_feats
from eval_template import templates
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
# MULTI: panels with more than one onset site. COLS: model inputs (bn-ratio series features and the 5-minute
# phase one-hot ph0..ph5).
MULTI = ['D12_I5_N', 'D12_I5_S', 'D7_I210_W', 'D7_I405_S']
COLS = ['r6', 'rT', 'r3', 'r12', 'dr', 'rmx3', 'ph0', 'ph1', 'ph2', 'ph3', 'ph4', 'ph5']
# C is read from the command line at import time (1.0 when no argument is given).
C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0


def addph(X):
    """Adds ph = T modulo 6 (5-minute phase of T within each half hour) and its one-hot columns ph0..ph5."""
    X['ph'] = (X['T'] + 6) % 6
    for k in range(6): X[f'ph{k}'] = (X.ph == k).astype(float)
    return X


def fit_models():
    """Fits one model per (panel, site) on the train pairs of uncond.build_train.

    Returns:
        {(panel, site): ('lr', scaler + logistic regression, feature medians)} or ('const', positive rate,
        medians) for sites with fewer than 5 train onsets.
    """
    models = {}
    for panel in MULTI:
        d = addph(build_train(panel))
        for s, g in d.groupby('site'):
            med = g[COLS].median()
            if g.y.sum() < 5: models[(panel, s)] = ('const', g.y.mean(), med); continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(g[COLS].fillna(med), g.y)
            models[(panel, s)] = ('lr', m, med)
    return models


def score_rows(rows, models):
    """Returns rows with column u: the breakdown hazard of each (window, site) from its model (NaN-filled with medians)."""
    rows = addph(rows.copy()); rows['u'] = np.nan
    for (panel, s), (kind, m, med) in models.items():
        ix = (rows.panel == panel) & (rows.site == s)
        if not ix.any(): continue
        rows.loc[ix, 'u'] = m if kind == 'const' else m.predict_proba(rows.loc[ix, COLS].fillna(med))[:, 1]
    return rows


def official_rows(split):
    """Builds one row per official onset window and site of the multi-site panels.

    Uses the window's released history (flows at T-60..T-5) and the masked layer at the origin stamp T from
    cache/<panel>_<split>.npz; features are series_feats of the bottleneck links' flow / capacity ratio.

    Args:
        split: 'validation' or 'private'.

    Returns:
        DataFrame with split, panel, key (window_id), date, T (origin step of the day), site and the features.
    """
    out = []
    for panel in MULTI:
        wi = pd.read_csv(REL / 'task2' / panel / split / 'window_index.csv'); wi = wi[wi.condition == 'queue_onset']
        h = pd.read_parquet(REL / 'task2' / panel / split / 'window_history.parquet')
        z = np.load(HERE / 'cache' / f'{panel}_{split}.npz', allow_pickle=True)
        links = z['links'].astype(str); li = {k: i for i, k in enumerate(links)}; dates = list(z['dates'].astype(str))
        lk = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str}).set_index('link_id').reindex(links)
        cap = lk.capacity_vph.to_numpy()
        for w in wi.itertuples():
            to = pd.Timestamp(w.forecast_origin); T = to.hour * 12 + to.minute // 5
            Qd = np.full((len(links), 288), np.nan)
            hh = h[h.window_id == w.window_id]
            ts = pd.to_datetime(hh.timestamp, utc=True); c = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            assert c.max() == T - 1 and c.min() == T - 12, (w.window_id, c.min(), c.max(), T)
            Qd[hh.link_id.astype(str).map(li).to_numpy(), c] = hh.flow_vph.to_numpy()
            di = dates.index(w.date)
            Qd[:, T] = z['Q'][di, :, T].astype(float)   # masked layer at the origin timestamp (Task 1 targets blank)
            for s, bl in BN[panel].items():
                rm = np.nanmean(Qd[bl, :] / cap[bl, None], 0)[None, :]
                f = {k: float(v[0]) for k, v in series_feats(rm, T).items()}
                out.append(dict(split=split, panel=panel, key=w.window_id, date=w.date, T=T, site=s, **f))
    return pd.DataFrame(out)


def decode(comp, choice, tpl, thr=0.5):
    """comp: onset component rows (step 6) with prob/queue_pred; returns modified copy (prod_else_tpl)."""
    # choice: window_id -> set of sites to keep; tpl: queue-extent templates. Inside the chosen sites the
    # production prediction is kept; when it is empty, template links with share >= thr; when still empty, the
    # most probable link of the chosen sites. Train-cache link order maps link ids to site indices.
    q = comp.copy()
    LI = {}
    for panel in MULTI:
        z = np.load(HERE / 'cache' / f'{panel}_train.npz', allow_pickle=True); LI[panel] = {k: i for i, k in enumerate(z['links'].astype(str))}
    q['site'] = [site_of(pa, LI[pa][str(l)]) if pa in LI else 'X' for pa, l in zip(q.panel, q.link_id)]
    parts = []
    for wid, g in q.groupby('window_id', sort=False):
        keep = choice.get(wid)
        if keep is None: parts.append(g); continue
        g = g.copy(); pa = g.panel.iloc[0]
        m = g.site.isin(keep).to_numpy(); y0 = g.queue_pred.to_numpy().astype(bool)
        y = y0 & m
        if not y.any():
            for s in keep:
                tp = tpl.get((pa, s), {})
                y |= np.array([tp.get(str(l), 0) >= thr for l in g.link_id]) & (g.site == s).to_numpy()
        if not y.any() and m.any():
            ix = np.where(m)[0]; y[ix[np.argmax(g.prob.to_numpy()[ix])]] = True
        g['queue_pred'] = y.astype('int8'); parts.append(g)
    return pd.concat(parts).drop(columns='site')


if __name__ == '__main__':
    models = fit_models()
    tpl = templates()
    with open(HERE / f'u_models_C{C}.pkl', 'wb') as f: pickle.dump(dict(models=models, tpl=tpl, cols=COLS), f)
    pv = score_rows(pd.read_parquet(HERE / 'site_pv.parquet'), models)
    pv.to_parquet(HERE / f'u_pv_C{C}.parquet', index=False)
    acc = [g.y.to_numpy()[np.nanargmax(g.u.to_numpy())] for k, g in pv[pv.panel.isin(MULTI)].groupby('key') if g.y.sum() > 0]
    print('pv top1', np.mean(acc), len(acc))
    offi = pd.concat([official_rows('validation'), official_rows('private')], ignore_index=True)
    offi = score_rows(offi, models); offi.to_parquet(HERE / f'u_official_C{C}.parquet', index=False)
    comp = pd.read_parquet(ROOT / 'build/strongest_candidate_20260922/onset/production/onset_component.parquet')
    comp['link_id'] = comp.link_id.astype(str)
    choice = {k: {g.site.iloc[int(np.nanargmax(g.u.to_numpy()))]} for k, g in offi.groupby('key')}
    new = decode(comp, choice, tpl)
    new.to_parquet(HERE / f'onset_component_siteu_C{C}.parquet', index=False)
    m = comp.merge(new[['window_id', 'link_id', 'queue_pred']], on=['window_id', 'link_id'], suffixes=('', '_new'))
    ch = m[m.queue_pred != m.queue_pred_new]
    print('changed cells', len(ch), 'windows', ch.window_id.nunique(), ch.groupby('split').window_id.nunique().to_dict())
    # production site vs chosen site per official window
    LI = {p: {k: i for i, k in enumerate(np.load(HERE / 'cache' / f'{p}_train.npz', allow_pickle=True)['links'].astype(str))} for p in MULTI}
    rows = []
    for wid, g in comp[comp.panel.isin(MULTI)].groupby('window_id'):
        pa = g.panel.iloc[0]
        ps = sorted({site_of(pa, LI[pa][l]) for l in g.link_id[g.queue_pred == 1]})
        rows.append(dict(window_id=wid, panel=pa, split=g.split.iloc[0], prod_sites='+'.join(ps), u_site=list(choice[wid])[0],
                         u=' '.join(f'{s}:{v:.2f}' for s, v in offi[offi.key == wid][['site', 'u']].itertuples(index=False))))
    r = pd.DataFrame(rows); pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
    print(r.to_string())
