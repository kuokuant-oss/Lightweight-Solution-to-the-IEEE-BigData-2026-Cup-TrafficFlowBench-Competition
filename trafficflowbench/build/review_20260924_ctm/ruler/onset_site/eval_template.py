"""Site choice + extent: production links in chosen site, or a train template of the site's onset block."""
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
FV = Path(__file__).resolve().parent.parent / 'fakeval'
sys.path.insert(0, str(FV)); sys.path.insert(0, str(HERE))
import score as S
from pv_sites import SITES, site_of
from oracle_site import rep, pred0, LI

MULTI = ['D12_I5_N', 'D12_I5_S', 'D7_I210_W', 'D7_I405_S']
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')


def templates():
    ev = pd.read_parquet(HERE / 'events_train.parquet')
    tpl = {}
    for panel in MULTI:
        z = np.load(HERE / 'cache' / f'{panel}_train.npz', allow_pickle=True); links = z['links'].astype(str)
        lk = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str}).set_index('link_id').reindex(links)
        V = z['V']; E = z['E']; vf = lk.free_speed_kmh.to_numpy()
        g = ev[ev.panel == panel].copy(); g['site'] = [site_of(panel, b) for b in g.b]
        for s, gg in g.groupby('site'):
            if s == 'X': continue
            lo, hi = SITES[panel][s]; rng = np.arange(max(0, lo - 3), min(len(links), hi + 4))
            M = []
            for (di, tq), _ in gg.groupby(['di', 'tq']):
                v = V[di, rng, tq].astype(float); e = E[di, rng, tq]
                M.append(np.where(e, v <= 0.6 * vf[rng], np.nan))
            fr = np.nanmean(np.array(M), 0)
            tpl[(panel, s)] = {links[i]: f for i, f in zip(rng, fr)}
    return tpl


def apply(pred, choice, tpl, mode, thr=0.5):
    p = pred.copy(); on = p.condition.eq('queue_onset'); q = p[on].copy()
    q['site'] = [site_of(pa, LI[pa][l]) for pa, l in zip(q.panel, q.link_id)]
    out = []
    for wid, g in q.groupby('window_id', sort=False):
        keep = choice.get(wid)
        if keep is None: out.append(g); continue
        g = g.copy(); pa = g.panel.iloc[0]
        m = g.site.isin(keep).to_numpy(); y0 = g.queue_pred.to_numpy().astype(bool)
        t = np.zeros(len(g), bool)
        for s in keep:
            tp = tpl.get((pa, s), {})
            t |= np.array([tp.get(l, 0) >= thr for l in g.link_id]) & (g.site == s).to_numpy()
        if mode == 'prod_else_tpl':
            y = y0 & m
            if not y.any(): y = t
        elif mode == 'tpl': y = t
        elif mode == 'union': y = (y0 & m) | t
        if not y.any() and m.any():
            ix = np.where(m)[0]; y[ix[np.argmax(g.prob.to_numpy()[ix])]] = True
        g['queue_pred'] = y.astype('int8'); out.append(g)
    return pd.concat([p[~on], pd.concat(out).drop(columns='site')], ignore_index=True)


if __name__ == '__main__':
    tpl = templates()
    u = pd.read_parquet(HERE / 'uncond_pv.parquet')
    ps = pd.read_parquet(HERE / 'pv_sites.parquet').set_index('window_id')
    base = rep(pred0, 'production')
    ora = {wid: set(r.ysite.split('+')) - {'X'} for wid, r in ps.iterrows() if r.panel in MULTI and isinstance(r.ysite, str) and r.ysite and set(r.ysite.split('+')) - {'X'}}
    for mode in ['prod_else_tpl', 'tpl', 'union']:
        rep(apply(pred0, ora, tpl, mode), f'ORACLE site {mode}')
    rng = np.random.default_rng(0)
    for col in ['u_phys_ph', 'u_phys', 'u_r6']:
        ch = {k: {g.site.iloc[int(np.nanargmax(g[col].to_numpy()))]} for k, g in u.groupby('key')}
        for mode in ['prod_else_tpl', 'tpl', 'union']:
            fo = rep(apply(pred0, ch, tpl, mode), f'{col} {mode}')
            m = base[['window_id', 'iou', 'panel', 'date', 'condition']].merge(fo[['window_id', 'iou']], on='window_id', suffixes=('0', '1'))
            dates = m.date.unique(); diffs = []
            for b in range(300):
                dd = rng.choice(dates, len(dates)); mm = pd.concat([m[m.date == x] for x in dd])
                diffs.append(S.agg(mm.rename(columns={'iou1': 'iou'}))['queue_onset'] - S.agg(mm.rename(columns={'iou0': 'iou'}))['queue_onset'])
            print(f'      delta CI95 [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]  better {int((m.iou1 > m.iou0).sum())} worse {int((m.iou1 < m.iou0).sum())}')
