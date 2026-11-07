"""Per pseudo-val onset window: true site vs production predicted site, IoU."""
#
# SITES and site_of are imported by the onset site code (serve_official.py, site_feats.py, uncond.py,
# eval_template.py, oracle_site.py) and by round2/onset_joint/common.py; in the reproduction pipeline they
# load through pipeline/onset_c3.py (onset.sh step 2). The __main__ block is the record of the March
# pseudo-val site table (pv_sites.parquet); it reads ../fakeval/pv_prod.parquet and cache/<panel>_validation.npz.
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
FV = Path(__file__).resolve().parent.parent / 'fakeval'
sys.path.insert(0, str(FV))
import score as S
# Candidate onset sites per corridor: site name -> (lo, hi) link-index range of its queued region, with
# links in milepost order (index grows downstream).
SITES = {  # queued region [lo, hi] in milepost-order link index; bottleneck links
    'D12_I5_N': {'S1': (103, 105), 'S2': (236, 238), 'S3': (8, 14)},
    'D12_I5_S': {'S1': (43, 45), 'S2': (10, 15), 'S3': (89, 89), 'S4': (39, 41)},
    'D7_I10_E': {'S1': (4, 10), 'S2': (32, 39)},
    'D7_I10_W': {'S1': (24, 26)},
    'D7_I210_E': {'S1': (0, 4)},
    'D7_I210_W': {'S1': (29, 33), 'S2': (3, 7), 'S3': (45, 49)},
    'D7_I405_N': {'S1': (349, 351), 'S2': (63, 65)},
    'D7_I405_S': {'S1': (32, 39), 'S2': (11, 15)},
}


def site_of(panel, idx, tol=2):
    """Returns the site whose region contains or lies nearest to link index idx.

    Args:
        panel: corridor name (a key of SITES).
        idx: link index in milepost order.
        tol: largest distance in links still assigned to a site.

    Returns:
        Site name such as 'S1', or 'X' when no site region is within tol links.
    """
    best = 'X'; bd = 1e9
    for s, (lo, hi) in SITES[panel].items():
        d = 0 if lo <= idx <= hi else min(abs(idx - lo), abs(idx - hi))
        if d < bd: bd = d; best = s
    return best if bd <= tol else 'X'


if __name__ == '__main__':
    # Per pseudo-val onset window: sites of the clean labels at T+30 (step 6) vs sites of the production
    # prediction, the window IoU and the highest production probability per site.
    lab, w = S.labels('validation')
    pred = pd.read_parquet(FV / 'pv_prod.parquet'); pred = pred[pred.condition == 'queue_onset']
    wo = w[w.condition == 'queue_onset']
    L = lab[lab.window_id.isin(wo.window_id) & (lab.step == 6)].merge(pred[['window_id', 'link_id', 'queue_pred', 'prob']], on=['window_id', 'link_id'], how='left')
    rows = []
    for panel in SITES:
        z = np.load(HERE / 'cache' / f'{panel}_validation.npz', allow_pickle=True); li = {k: i for i, k in enumerate(z['links'].astype(str))}
        for wid, g in L[L.window_id.isin(wo.window_id[wo.panel == panel])].groupby('window_id'):
            g = g.assign(li=g.link_id.map(li)).sort_values('li')
            Y = g.clean.to_numpy(); P = g.queue_pred.fillna(0).astype(bool).to_numpy(); pr = g.prob.fillna(0).to_numpy(); ix = g.li.to_numpy()
            ys = sorted({site_of(panel, i) for i in ix[Y]}); ps = sorted({site_of(panel, i) for i in ix[P]})
            mk = g['mask'].to_numpy(); lb = g.label.to_numpy()
            u = (P[mk] | lb[mk]).sum(); iou = 1.0 if u == 0 else (P[mk] & lb[mk]).sum() / u
            # site-level max prob
            sp = {s: pr[[site_of(panel, i) == s for i in ix]].max() for s in SITES[panel]}
            rows.append(dict(window_id=wid, panel=panel, ysite='+'.join(ys), psite='+'.join(ps), iou=iou, nobs=int(lb[mk].sum()),
                             tod=pd.Timestamp(g.timestamp.iloc[0]).hour, **{f'p_{s}': v for s, v in sp.items()}))
    d = pd.DataFrame(rows).merge(wo[['window_id', 'date']], on='window_id')
    d.to_parquet(HERE / 'pv_sites.parquet', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    print(pd.crosstab([d.panel, d.ysite], d.psite))
    print(d[d.nobs > 0].groupby('panel').iou.mean().round(3))
    print(d[(d.nobs > 0) & (d.ysite != d.psite)].sort_values('panel')[['window_id', 'ysite', 'psite', 'iou', 'tod']])
