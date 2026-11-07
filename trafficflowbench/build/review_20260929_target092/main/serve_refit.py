"""Serve the Jan-Feb refit N ensemble (6 models, contract_audit/q4_refit) on the OFFICIAL ongoing windows.
Same feature assembly as build/review_20260924_ctm/serve/serve_ens.py; mix = .75 N + .25 A link probs, train-block mean >= .5;
the guard window keeps the champion's cells. Parity gate: production models must reproduce the champion CSV exactly.
Writes main/queue_override_ongoing_refitN6.parquet (only flipped cells). Never submits."""
# Runs in pipeline/ongoing.sh ("serve_refit"), the decision step of the F2
# operator. N6: six CTM ongoing boosters, 3 seeds each of two feature sets
# ("selrec": state features + selection-screen replay + recency; "geomselrec":
# the same plus queue geometry). The refit set is trained on January-February
# 2031 (contract_audit/q4_refit). A: the location-free new-world ensemble
# (ogens_<tag>_prob.parquet). Link probability = .75 x N + .25 x A; a cell is
# queued when the mean over its link group (my_blocks) at the same time is
# >= .5; a window with no queued cell takes its A-block decode (never-empty
# rule). All serving inputs come from data up to each window's forecast time.
# Outputs: queue_override_ongoing_refitN6.parquet (cells where the refit and
# production decodes differ), never_empty_fallback.json, serve_refit.json and
# ongoing_refitN6_d1.parquet (refit decision d1 on every official ongoing row,
# read by make_r16b_bits.py).
import os
os.environ['OMP_NUM_THREADS'] = '2'
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
# Original absolute roots; bin/tfb-run maps them onto the run tree.
# PROD / NEW: production and refit N6 boosters. B / BID: link group id per
# panel and link (my_blocks).
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); CT = ROOT / 'build/review_20260924_ctm'
HERE = Path(__file__).resolve().parent; RF = HERE.parent / 'contract_audit/q4_refit'
REL = ROOT.parent / 'tfb_competition_data/kaggle_public'
CHAMP = ROOT / 'build/deliverables/2026-09-27/capacity/submission_HMP50_DK50_CAP.csv'
PROD = [CT / f'rep2/m_R_V2selrec_s0_seed{i}.txt' for i in range(3)] + [CT / f'rep2/m_R_V2geomselrec_s0_seed{i}.txt' for i in range(3)]
NEW = [RF / f'm_R_V2selrec_refit_seed{i}.txt' for i in range(3)] + [RF / f'm_R_V2geomselrec_refit_seed{i}.txt' for i in range(3)]
# package edit: GUARD = 'Q2_D12_I5_N_validation_007' removed; no window is hard-coded (never-empty rule below)
B = np.load(CT / 'ruler/blocks/my_blocks.npy', allow_pickle=True).item(); BID = {k: dict(zip(v['links'], v['bid'])) for k, v in B.items()}


def ens(models, s):
    """Returns the mean probability of the boosters `models` on rows `s` (each reads its own feature columns)."""
    return np.mean([lgb.Booster(model_file=str(m)).predict(s[lgb.Booster(model_file=str(m)).feature_name()], num_threads=2) for m in models], 0)


# Per split: official queue_ongoing rows of the CTM serving table, joined with
# queue geometry, the selection-screen replay (prev_rej, n_rej) and the
# same-split recency features. pN0 / pN1: production / refit N6 mean;
# bp0 / bp1: link-group mean of .75 pN + .25 pA at each time.
parts = []
for sp, rt, tag in [('validation', 'realvali', 'realval'), ('private', 'realpriv', 'realpriv')]:
    s = pd.read_parquet(CT / f'serve/serve_real_{sp}.parquet'); s = s[s.condition.eq('queue_ongoing')].reset_index(drop=True)
    s = s.merge(pd.read_parquet(CT / f'ruler/feat/geom_serve_real_{sp}.parquet'), on=['window_id', 'link_id', 'step'], how='left')
    sel = pd.read_parquet(CT / f'dense/sel2_{sp}.parquet')[['window_id', 'prev_rej', 'n_rej']]; s = s.merge(sel, on='window_id', how='left')
    r = pd.read_parquet(CT / f'dense/rec_{rt}_queue_ongoing.parquet'); r['link_id'] = r.link_id.astype(str); s['link_id'] = s.link_id.astype(str)
    s = s.merge(r, on=['window_id', 'link_id', 'step'], how='left')
    s['prev_rej'] = s.prev_rej.astype('float32'); s['n_rej'] = s.n_rej.astype('float32')
    a = pd.read_parquet(ROOT / f'build/review_20260924_newworld/work/feat/ogens_{tag}_prob.parquet'); a['link_id'] = a.link_id.astype(str)
    z = s[['window_id', 'timestamp', 'link_id', 'panel']].copy()
    z['pN0'] = ens(PROD, s); z['pN1'] = ens(NEW, s)
    z = z.merge(a.rename(columns={'prob': 'pA'}), on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one')
    assert z.pA.notna().all()
    z['bid'] = [BID[p][l] for p, l in zip(z.panel, z.link_id)]
    for k in ('0', '1'):
        z['pl' + k] = .75 * z['pN' + k] + .25 * z.pA
        z['bp' + k] = z.groupby(['window_id', 'timestamp', 'bid'])['pl' + k].transform('mean')
    z['split'] = sp; parts.append(z)
d = pd.concat(parts, ignore_index=True)
# Attaches the official submission_id of every queue cell (submission_key.csv).
d['timestamp'] = pd.to_datetime(d.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
key = pd.read_csv(REL / 'submission_key.csv', usecols=['submission_id', 'task', 'timestamp', 'link_id', 'window_id'], dtype=str)
key = key[key.task == 'queue'].copy(); key['submission_id'] = key.submission_id.astype(int)
key['timestamp'] = pd.to_datetime(key.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
d = d.merge(key[['submission_id', 'window_id', 'timestamp', 'link_id']], on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one')
assert d.submission_id.notna().all()
# package edit: the hard-coded guard window and the champion-CSV parity gate (CAP, a whole earlier submission) are
# replaced by the generic never-empty rule of review_20260925/make_never_empty_fallback.py, applied to every window
# of both splits for both decodes: a window whose block decode (>= .5) has no queued cell takes its A-block decode.
import sys; sys.path.insert(0, str(ROOT / 'build/review_20260925'))
from make_never_empty_fallback import never_empty  # noqa: E402
fb = pd.read_parquet(ROOT / 'build/review_20260925/candidates/a_block_fallback.parquet', columns=['window_id', 'timestamp', 'link_id', 'queue_pred'])
d = d.merge(fb.rename(columns={'queue_pred': 'fallback'}), on=['window_id', 'timestamp', 'link_id'], how='left', validate='one_to_one')
assert d.fallback.notna().all()
fired = {}
for k, name in (('0', 'production'), ('1', 'refit')):
    d['d' + k], empty = never_empty(d.window_id.to_numpy(), (d['bp' + k] >= .5).astype(int), d.fallback)
    fired[name] = {sp: sorted(d.window_id[empty & (d.split == sp).to_numpy()].unique()) for sp in ('validation', 'private')}
    print('never-empty fallback fired,', name, 'decode:', {sp: len(v) for sp, v in fired[name].items()}, fired[name], flush=True)
json.dump(fired, open(HERE / 'never_empty_fallback.json', 'w'), indent=1)
# never-empty check on the new decision (outside the guard)
empty_new = d.groupby('window_id').d1.sum(); print('windows empty under refit:', list(empty_new[empty_new == 0].index))
ov = d[d.d1 != d.d0].copy(); ov['queue_pred'] = ov.d1; ov['parent_queue_pred'] = ov.d0
dst = HERE / 'queue_override_ongoing_refitN6.parquet'; assert not dst.exists()
ov[['submission_id', 'window_id', 'timestamp', 'link_id', 'split', 'queue_pred', 'parent_queue_pred', 'bp0', 'bp1']].to_parquet(dst, index=False)
w = ov.groupby(['split', 'window_id']).agg(n_add=('queue_pred', 'sum'), n=('queue_pred', 'size')).reset_index(); w['n_remove'] = w['n'] - w['n_add']
rep = dict(rows=len(ov), windows=int(ov.window_id.nunique()), by_split=ov.groupby('split').window_id.nunique().to_dict(),
           added=int(ov.queue_pred.sum()), removed=int((1 - ov.queue_pred).sum()), per_window=w.to_dict('records'),
           max_abs_dbp=float((d.bp1 - d.bp0).abs().max()), sha256=hashlib.sha256(open(dst, 'rb').read()).hexdigest())
json.dump(rep, open(HERE / 'serve_refit.json', 'w'), indent=1); print(json.dumps({k: v for k, v in rep.items() if k != 'per_window'}, indent=1))
# package addition: the refit-N6 decision on every official ongoing row (R16b ongoing = CAP cells + this override = d1)
d[['submission_id', 'window_id', 'timestamp', 'link_id', 'split', 'd1']].to_parquet(HERE / 'ongoing_refitN6_d1.parquet', index=False)
print(w.to_string())
