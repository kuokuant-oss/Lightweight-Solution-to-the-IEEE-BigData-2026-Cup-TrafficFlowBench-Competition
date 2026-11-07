"""Fit selected variants on ALL train (no LMO) and score (a) pv top-1, (b) v1 onset ruler through D's decoder with paired boot vs D.
PRIMARY / SECONDARY fixed by prereg.json + s1_inworld_table.csv (selection done in-world before this script ran)."""
import sys, json, numpy as np, pandas as pd
import common as C
import cl as CL
from s1_specs import SPECS, PSPECS, lg
VARS = [  # (name, role)
    ('C3_+gaps_todwk|rep|1.0', 'PRIMARY'),
    ('P1b_noID_+gaps|rep|20.0', 'SECONDARY'),
    ('C1_base_kr_int|rep|1.0', 'exploratory'), ('C2_+gaps|rep|1.0', 'exploratory'), ('C2_+gaps|rep|5.0', 'exploratory'),
    ('C3_+gaps_todwk|allpos|1.0', 'exploratory'), ('C4_stack_u_gaps|rep|5.0', 'exploratory'), ('C4b_stack_u_gaps_kr|rep|5.0', 'exploratory'),
    ('C6_siteslopes|allpos|1.0', 'exploratory'), ('P1_noID_base_kr|rep|1.0', 'exploratory'),
]
PANELS = C.MS + ['D7_I210_W']


def train_sets():
    rep, allp = [], []
    for panel in PANELS:
        tr = C.train_rows(panel); tr['key'] = tr.date + '|' + tr['T'].astype(str)
        ev = C.inworld_windows(panel, tr)
        ev['u'] = C.score_rows(ev[['panel', 'site', 'T'] + CL.BASEF].assign(), C.UMODELS).u.to_numpy() if False else None
        n = tr.groupby('key').site.transform('size'); pos = tr.groupby('key').y.transform('sum')
        ga = tr[(pos > 0) & (n >= 2)].copy()
        for d, L in ((ev, rep), (ga, allp)):
            d = d.copy(); d['u'] = C.score_rows(d, C.UMODELS).u.to_numpy(); d = CL.add_feats(d); d['ulogit'] = lg(d.u); d['key'] = panel + '|' + d.key
            L.append(d)
    return pd.concat(rep, ignore_index=True), pd.concat(allp, ignore_index=True)


def prep_rows(rows):
    d = C.score_rows(rows, C.UMODELS); d = CL.add_feats(d); d['ulogit'] = lg(d.u); return d


def fit_variant(name, REP, ALL):
    k, ts, lam = name.split('|'); lam = float(lam)
    D = REP if ts == 'rep' else ALL
    if k in PSPECS:
        return {'__pooled__': CL.CLModel(PSPECS[k], lam).fit(D[D.panel.isin(PANELS)])}
    return {p: CL.CLModel(SPECS[k], lam).fit(D[D.panel == p]) for p in PANELS}


def score_variant(models, rows):
    s = np.full(len(rows), np.nan)
    for p, m in models.items():
        ix = np.ones(len(rows), bool) if p == '__pooled__' else (rows.panel == p).to_numpy()
        ix &= rows.panel.isin(PANELS).to_numpy()
        if ix.any(): s[ix] = m.score(rows[ix])
    return s


if __name__ == '__main__':
    REP, ALL = train_sets()
    PV = prep_rows(C.SITE_PV)
    assert np.allclose(PV.u.to_numpy(), C.U_PV.u.to_numpy(), equal_nan=True)
    out = {'D': dict(onset=round(C.S_D['onset'], 4), pv_top1=C.pv_top1(C.U_PV, 'u'))}
    print('D', out['D'], flush=True)
    perwin = {}
    for name, role in VARS:
        mods = fit_variant(name, REP, ALL)
        PV['s'] = score_variant(mods, PV)
        r, fr, ch = C.eval_choice(PV, 's', name)
        r['role'] = role; r['pv_top1'] = C.pv_top1(PV, 's')
        out[name] = r; perwin[name] = fr; PV[f's|{name}'] = PV['s']
        print(role, json.dumps(r, default=float), flush=True)
    json.dump(out, open(C.ME / 's2_ruler.json', 'w'), indent=1, default=float)
    PV.to_parquet(C.ME / 's2_pv_scored.parquet', index=False)
    pd.concat([f.assign(variant=k) for k, f in perwin.items()]).to_parquet(C.ME / 's2_perwin.parquet', index=False)
