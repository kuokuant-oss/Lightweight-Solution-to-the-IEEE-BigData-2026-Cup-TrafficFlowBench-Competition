"""Family ensemble FE (defined after s5, before computing it): mean per-window softmax over the 24 relative-feature configs
{C2,C3,C4,C4b} x {rep,allpos} x lambda {1,5,20}.  Exploratory (the family was identified on the ruler in s5).
Outputs: in-world LMO top-1 (same folds as s1), pv top-1 + ruler vs D, official changed windows (+ files), diagnostics:
simple argmax rules and D7_I210_W inclusion."""
# Training record for the FE site-choice models. The reproduction does not run this
# program; it loads the shipped pickle. The program reads row tables that are not shipped,
# and its paths resolve from its original place in the development repository (`common.ROOT`).
#
# Trains the family ensemble FE: conditional logit models (`cl.CLModel`: inputs standardised,
#   L2 penalty lambda, L-BFGS-B from zero weights) that score the candidate bottleneck sites
#   of one onset window, P(site j) = exp(s_j) / sum_k exp(s_k). 24 configurations =
#   4 specifications (`s1_specs.SPECS`) x 2 training sets x lambda in {1, 5, 20}, each fitted
#   per corridor (D12_I5_N, D12_I5_S, D7_I405_S, D7_I210_W). The FE score is the mean softmax.
#   Specifications: C2 = site load ratios + demand-step phase + gaps and ranks against the
#   other sites; C3 = C2 + per-site slopes on time of day and weekend; C4 = u-model logit +
#   gaps and ranks; C4b = C4 + demand-step phase. Every specification has per-site intercepts.
# Training sets (`s2_ruler.train_sets`): "rep" = replicated train onset windows
#   (`build/review_20260924_ctm/rep2/replicated_unique_v2.parquet`); "allpos" = every train
#   (day, T) with a queued site and at least 2 sites. Rows: per-site train rows
#   `build/review_20260925/onset/train_rows_<corridor>.parquet` (train dates 2030-06-01 ..
#   2031-02-28, February included; demand profiles leave-one-month-out).
# Steps: leave-one-month-out check on the train windows; full-train fit of the 24
#   configurations; scores of the March pseudo windows `common.SITE_PV`; pickle.
# Shipped: `s6_FE_models.pkl` (dict (specification, training set, lambda) -> {corridor:
#   CLModel}) under `trafficflowbench/build/review_20260925/round2/onset_joint/`. Also writes
#   `s6_fe.json`, `s6_perwin_FE.parquet` and `s6_pv_scored_FE.parquet`.
# Invocation: `python s6_fe.py` in this folder, after `s0b_phase.py` (`cl.py` reads
#   `site_phase.json` at import). `common.py` loads the u-models and `SITE_PV` at import.
import sys, json, itertools, numpy as np, pandas as pd
import common as C
import cl as CL
import s2_ruler as R2
from s1_specs import SPECS, lg
sys.path.insert(0, str(C.ONS))
import site_mday as SM
FAM = ['C2_+gaps', 'C3_+gaps_todwk', 'C4_stack_u_gaps', 'C4b_stack_u_gaps_kr']
# The 24 configurations: (specification, training set, L2 penalty lambda).
CFG = [(k, ts, l) for k in FAM for ts in ('rep', 'allpos') for l in (1.0, 5.0, 20.0)]


def softmax_by(key, s):
    """Returns the softmax of scores `s` within each group of `key` (one group per window)."""
    s = pd.Series(s); m = s.groupby(key.to_numpy()).transform('max').to_numpy(); e = np.exp(s.to_numpy() - m)
    return e / pd.Series(e).groupby(key.to_numpy()).transform('sum').to_numpy()


out = {}
# ---------- in-world LMO ----------
hits = {}
for panel in C.MS + ['D7_I210_W']:
    tr = C.train_rows(panel); tr['key'] = tr.date + '|' + tr['T'].astype(str)
    ev0 = C.inworld_windows(panel, tr)
    n = tr.groupby('key').site.transform('size'); pos = tr.groupby('key').y.transform('sum'); ga = tr[(pos > 0) & (n >= 2)].copy()
    P = np.zeros(len(ev0)); U = np.full(len(ev0), np.nan)
    for mo in sorted(ev0.month.unique()):
        te = (ev0.month == mo).to_numpy(); mods = SM.fit(tr[(tr.month != mo) & (tr['T'] >= 60)], SM.BASE)
        et = ev0[te].copy(); et['u'] = SM.score(et, mods, SM.BASE); U[te] = et.u
        rt = ev0[~te].copy(); rt['u'] = SM.score(rt, mods, SM.BASE)
        gt = ga[ga.month != mo].copy(); gt['u'] = SM.score(gt, mods, SM.BASE)
        et, rt, gt = [CL.add_feats(d).assign(ulogit=lambda x: lg(x.u)) for d in (et, rt, gt)]
        acc = np.zeros(te.sum())
        for k, ts, l in CFG:
            m = CL.CLModel(SPECS[k], l).fit(rt if ts == 'rep' else gt)
            acc += softmax_by(et.key, m.score(et))
        P[te] = acc / len(CFG)
    ev0['key'] = panel + '|' + ev0.key
    hits[('U0', panel)] = C.top1_vec(ev0.assign(_s=U), '_s'); hits[('FE', panel)] = C.top1_vec(ev0.assign(_s=P), '_s')
    for f in ('r6', 'rmx3', 'rT'):
        hits[(f'argmax_{f}', panel)] = C.top1_vec(ev0.assign(_s=ev0[f].fillna(ev0.r3)), '_s')
    print(panel, {k: round(v.mean(), 3) for (k, p), v in hits.items() if p == panel}, flush=True)
b3 = pd.concat([hits[('U0', p)] for p in C.MS])
inw = {}
for v in ('U0', 'FE', 'argmax_r6', 'argmax_rmx3', 'argmax_rT'):
    h3 = pd.concat([hits[(v, p)] for p in C.MS]).reindex(b3.index)
    inw[v] = dict(pooled3=round(h3.mean(), 4), **{p: round(hits[(v, p)].mean(), 3) for p in C.MS + ['D7_I210_W']},
                  better=int(((h3 == 1) & (b3 == 0)).sum()), worse=int(((h3 == 0) & (b3 == 1)).sum()))
out['inworld'] = inw; print(json.dumps(inw), flush=True)
# ---------- full-train fit, pv + ruler ----------
REP, ALL = R2.train_sets(); PV = R2.prep_rows(C.SITE_PV)
# `fits` is always empty (it iterates over an empty list); the fit is the loop below.
fits = [(k, ts, l, (CL.CLModel(SPECS[k], l).fit(REP[REP.panel == p] if ts == 'rep' else ALL[ALL.panel == p]) for p in [None])) for k, ts, l in []]
MODELS = {}
for k, ts, l in CFG:
    D = REP if ts == 'rep' else ALL
    MODELS[(k, ts, l)] = {p: CL.CLModel(SPECS[k], l).fit(D[D.panel == p]) for p in C.MS + ['D7_I210_W']}


def fe_score(rows):
    """Returns the FE score of each site row: the mean softmax over the 24 configurations.

    Rows of corridors without a model score -50 inside the softmax and NaN in the result.

    Args:
        rows: Per (window key, site) rows with the model inputs, `panel` and `key`.
    """
    acc = np.zeros(len(rows)); ok = rows.panel.isin(C.MS + ['D7_I210_W']).to_numpy()
    for key, mods in MODELS.items():
        s = np.full(len(rows), -50.0)
        for p, m in mods.items():
            ix = (rows.panel == p).to_numpy()
            if ix.any(): s[ix] = m.score(rows[ix])
        acc += softmax_by(rows.key, s)
    acc /= len(MODELS); acc[~ok] = np.nan; return acc


PV['fe'] = fe_score(PV)
r, fr, ch = C.eval_choice(PV, 'fe', 'FE'); r['pv_top1'] = C.pv_top1(PV, 'fe'); out['ruler_FE'] = r; print('FE ruler', json.dumps(r, default=float), flush=True)
fr.to_parquet(C.ME / 's6_perwin_FE.parquet', index=False)
for f in ('r6', 'rmx3'):
    PV['_s'] = PV[f].fillna(PV.r3); rr, _, _ = C.eval_choice(PV, '_s', f'argmax_{f}'); rr['pv_top1'] = C.pv_top1(PV, '_s'); out[f'ruler_argmax_{f}'] = rr
    print(f'argmax_{f}', json.dumps(rr, default=float), flush=True)
# D7_I210_W inclusion diagnostic: pv top-1 on D7_I210_W, u vs FE (D keeps production there)
g = PV[PV.panel == 'D7_I210_W']
t_u = [x.y.to_numpy()[np.nanargmax(x.u.to_numpy())] for _, x in g.groupby('key') if x.y.sum() > 0]
t_f = [x.y.to_numpy()[np.nanargmax(x.fe.to_numpy())] for _, x in g.groupby('key') if x.y.sum() > 0]
out['D7_I210_W_pv_top1'] = dict(u=round(float(np.mean(t_u)), 3), FE=round(float(np.mean(t_f)), 3), n=len(t_u)); print(out['D7_I210_W_pv_top1'])
PV.to_parquet(C.ME / 's6_pv_scored_FE.parquet', index=False)
json.dump(out, open(C.ME / 's6_fe.json', 'w'), indent=1, default=float)
pd.to_pickle(MODELS, C.ME / 's6_FE_models.pkl')
