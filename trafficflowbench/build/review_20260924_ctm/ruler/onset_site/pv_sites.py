"""Per pseudo-val onset window: true site vs production predicted site, IoU."""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
FV = Path(__file__).resolve().parent.parent / 'fakeval'
sys.path.insert(0, str(FV))
import score as S
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
    best = 'X'; bd = 1e9
    for s, (lo, hi) in SITES[panel].items():
        d = 0 if lo <= idx <= hi else min(abs(idx - lo), abs(idx - hi))
        if d < bd: bd = d; best = s
    return best if bd <= tol else 'X'


if __name__ == '__main__':
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
