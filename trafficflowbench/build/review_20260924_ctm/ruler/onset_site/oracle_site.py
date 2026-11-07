"""Upper bound: restrict production onset predictions to the TRUE site(s); also to a chosen site from a rule."""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
FV = Path(__file__).resolve().parent.parent / 'fakeval'
sys.path.insert(0, str(FV)); sys.path.insert(0, str(HERE))
import score as S
from pv_sites import SITES, site_of

# Package edit (2026-11-07): the March pseudo-val ruler (fakeval labels, pv_prod, pv_sites) and the
# validation-cache link maps are no longer loaded at import. Only the ruler functions below use them;
# the reproduction never calls those, and the ruler files are validation-split derived (not shipped).
lab = w = pred0 = ps = None
LI = {}


def restrict(pred, choice, extra_top=True):
    """choice: window_id -> set of sites to keep (None = keep as is)."""
    p = pred.copy()
    on = p.condition.eq('queue_onset')
    q = p[on].copy()
    q['site'] = [site_of(pa, LI[pa][l]) for pa, l in zip(q.panel, q.link_id)]
    out = []
    for wid, g in q.groupby('window_id', sort=False):
        keep = choice.get(wid)
        if keep is None: out.append(g); continue
        g = g.copy(); m = g.site.isin(keep).to_numpy()
        y = g.queue_pred.to_numpy().astype(bool) & m
        if not y.any() and m.any() and extra_top:
            ix = np.where(m)[0]; y[ix[np.argmax(g.prob.to_numpy()[ix])]] = True
        g['queue_pred'] = y.astype('int8'); out.append(g)
    q2 = pd.concat(out)
    return pd.concat([p[~on], q2.drop(columns='site')], ignore_index=True)


def rep(pred, name):
    fr = S.score(pred, 'validation')
    fo = fr[fr.condition.eq('queue_onset') & (fr.n_true > 0)]
    a = S.agg(fo); b = S.agg(fo, 'iou_clean')
    print(f'{name:30s} onset(obs>0) {a["queue_onset"]:.4f} clean {b["queue_onset"]:.4f}  per-panel', fo.groupby('panel').iou.mean().round(3).to_dict())
    return fo


if __name__ == '__main__':
    rep(pred0, 'production')
    ora = {wid: set(r.ysite.split('+')) - {'X'} or None for wid, r in ps.iterrows() if isinstance(r.ysite, str) and r.ysite}
    rep(restrict(pred0, ora), 'oracle true site(s)')
    multi = ['D12_I5_N', 'D12_I5_S', 'D7_I210_W', 'D7_I405_S']
    ora2 = {k: v for k, v in ora.items() if ps.loc[k, 'panel'] in multi}
    rep(restrict(pred0, ora2), 'oracle site, multi-site panels')
