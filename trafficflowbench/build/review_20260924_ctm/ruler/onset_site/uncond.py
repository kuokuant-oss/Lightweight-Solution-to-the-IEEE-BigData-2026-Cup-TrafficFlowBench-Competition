"""Unconditional per-site breakdown hazard vs physical state, fit on ALL train (site, origin) pairs where the
site region is queue-free over [t-12, t+5]; label = site region queued at t+6.
Apply to pseudo-val onset windows (site_pv rows) and compare top-1 site choice."""
#
# Role: `build_train` and `series_feats` are imported by serve_official.py; the reproduction pipeline
# (onset.sh step 2, through pipeline/onset_c3.py) calls series_feats only, on the official windows. The
# __main__ block is the record of the comparison table uncond_pv.parquet.
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pv_sites import SITES
from site_feats import BN
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
# Panels with more than one onset site.
MULTI = ['D12_I5_N', 'D12_I5_S', 'D7_I210_W', 'D7_I405_S']


def series_feats(rm, T):
    """rm: [D, 288] bn ratio; returns dict of arrays at origin T (vectorised over days)."""
    # rT: bn ratio at T; rT1/r3/r6/r12: means over the last 2/3/6/12 steps up to T; r6p: mean over the 6 steps
    # before that; rmx3: 3-step max; dr = r6 - r6p (rise of the load).
    f = {}
    f['rT'] = rm[:, T]
    f['rT1'] = np.nanmean(rm[:, T - 1:T + 1], 1)
    f['r3'] = np.nanmean(rm[:, T - 2:T + 1], 1)
    f['r6'] = np.nanmean(rm[:, T - 5:T + 1], 1)
    f['r12'] = np.nanmean(rm[:, T - 11:T + 1], 1)
    f['r6p'] = np.nanmean(rm[:, T - 11:T - 5], 1)
    f['rmx3'] = np.nanmax(rm[:, T - 2:T + 1], 1)
    f['dr'] = f['r6'] - f['r6p']
    return f


def build_train(panel):
    """Builds the train pairs (day, origin T, site) of one panel for the unconditional breakdown hazard.

    Origins T run from step 60 to 249 (05:00 to 20:45). A pair is kept when the site region (+-1 link) has no
    queued cell over T-12..T+5; its label y is 1 when the region is queued at T+6 (T+30 min).

    Returns:
        DataFrame with the series_feats columns, y, clear, T, site, date and panel.
    """
    z = np.load(HERE / 'cache' / f'{panel}_train.npz', allow_pickle=True)
    links = z['links'].astype(str)
    lk = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str}).set_index('link_id').reindex(links)
    cap = lk.capacity_vph.to_numpy(); vf = lk.free_speed_kmh.to_numpy()
    Q = z['Q'].astype(np.float32); V = z['V'].astype(np.float32); E = z['E']
    Qq = (V <= 0.6 * vf[None, :, None]) & E
    dates = z['dates']
    rows = []
    for s, bl in BN[panel].items():
        lo, hi = SITES[panel][s]
        rm = np.nanmean(Q[:, bl, :] / cap[None, bl, None], 1)
        qs = Qq[:, max(0, lo - 1):hi + 2, :].any(1)  # D x 288 site queued
        for T in range(60, 250):
            clear = ~qs[:, T - 12:T + 6].any(1)
            y = qs[:, T + 6]
            f = series_feats(rm, T)
            df = pd.DataFrame(f); df['y'] = y.astype(int); df['clear'] = clear; df['T'] = T; df['site'] = s; df['date'] = dates
            rows.append(df[df.clear])
    d = pd.concat(rows, ignore_index=True); d['panel'] = panel
    return d


if __name__ == '__main__':
    pv = pd.read_parquet(HERE / 'site_pv.parquet')
    tr_ev = pd.read_parquet(HERE / 'site_train.parquet')
    out = []
    for panel in MULTI:
        d = build_train(panel)
        # ph: T modulo 6, the 5-minute phase of T within each half hour (one-hot ph0..ph5).
        d['ph'] = (d['T'] + 6) % 6
        P = pv[pv.panel == panel].copy(); P['ph'] = (P['T'] + 6) % 6
        TE = tr_ev[tr_ev.panel == panel].copy(); TE['ph'] = (TE['T'] + 6) % 6
        for name, cols in {'u_phys': ['r6', 'rT', 'r3', 'r12', 'dr', 'rmx3'], 'u_r6': ['r6'],
                           'u_phys_ph': ['r6', 'rT', 'r3', 'r12', 'dr', 'rmx3', 'ph0', 'ph1', 'ph2', 'ph3', 'ph4', 'ph5']}.items():
            for X in (d, P, TE):
                for k in range(6): X[f'ph{k}'] = (X.ph == k).astype(float)
            P[name] = np.nan; TE[name] = np.nan
            for s, g in d.groupby('site'):
                if g.y.sum() < 5:
                    P.loc[P.site == s, name] = g.y.mean(); TE.loc[TE.site == s, name] = g.y.mean(); continue
                med = g[cols].median()
                m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(g[cols].fillna(med), g.y)
                for X in (P, TE):
                    ix = X.site == s
                    if ix.any(): X.loc[ix, name] = m.predict_proba(X.loc[ix, cols].fillna(med))[:, 1]
        out.append(P)
        print(panel, 'train pairs', len(d), 'pos', int(d.y.sum()))
        for name in ['u_r6', 'u_phys', 'u_phys_ph']:
            acc = [g.y.to_numpy()[np.nanargmax(g[name].to_numpy())] for k, g in P.groupby('key') if g.y.sum() > 0]
            acc_t = [g.y.to_numpy()[np.nanargmax(g[name].to_numpy())] for k, g in TE.groupby('key') if g.y.sum() > 0]
            print(f'   {name:10s} top1 pv {np.mean(acc):.3f} (n={len(acc)})  train-events(in-sample) {np.mean(acc_t):.3f}')
    pd.concat(out).to_parquet(HERE / 'uncond_pv.parquet', index=False)
