"""Synthesis step 2: official validation/private onset rows for FE site choice + A1 template decoder (3 D panels), and
candidate G1 + FE + A1 built from the champion G1 CSV (read only).
Asserts before writing anything:
  (1) champion choice + prod_else_tpl reproduces D's official onset (0 mismatches)            [as onset_joint s7 / onset_extent s3]
  (2) champion choice + template decoder reproduces onset_extent/onset_A1_official.parquet     [A1 alone]
  (3) FE choice + prod_else_tpl reproduces onset_joint/real/onset_FE_<split>.parquet            [FE alone]
  (4) G1's step-6 onset cells == D official onset; G1 has 0 positives at onset steps 1-5 in the touched windows
Writes only under round2/synthesis/.  Never submits."""
# Task 2 reproduction pipeline, `pipeline/onset.sh` step 3 (`stage1_fea1`).
# Output onset_FEA1_official.parquet: one row per official onset cell of both
# splits (window_id, timestamp = T+30, link_id, prob = 0/1 decision). On
# D12_I5_N, D12_I5_S and D7_I405_S the decision is FE+A1: the FE site choice
# (s6_fe_lib.fe_score) with the A1 extent (the chosen site's train template
# >= 0.5). Every other panel keeps candidate D's decision
# (work/blocks/D_onset_*.parquet). pipeline/onset_sources.py keys the file to
# the submission. The G1 candidate CSV part at the end is skipped here.
import os, sys, json, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '3')
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from pathlib import Path
ME = Path(__file__).resolve().parent
OJ = ME.parent / 'onset_joint'; OX = ME.parent / 'onset_extent'
sys.path.insert(0, str(OJ))
import common as C
import s2_ruler as R2
import s6_fe_lib as F
# serve_official reads its masked-layer caches from fakeHERE/cache, a link to
# the caches that onset.sh builds for both official splits.
SO = C.SO; SO.HERE = C.ONS / 'fakeHERE'
from serve_official import official_rows, decode
from pv_sites import site_of
ROOT = C.ROOT; REL = ROOT.parent / 'tfb_competition_data' / 'kaggle_public'
# Production onset component (0922 head + panel_shrink25 decoder) and
# candidate D's decisions (Dref: prob by window, timestamp and link).
prod = pd.read_parquet(ROOT / 'build/strongest_candidate_20260922/onset/production/onset_component.parquet'); prod['link_id'] = prod.link_id.astype(str)
prod['timestamp'] = pd.to_datetime(prod.timestamp, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
Dv = pd.read_parquet(ROOT / 'build/review_20260924_newworld/work/blocks/D_onset_realval.parquet'); Dp = pd.read_parquet(ROOT / 'build/review_20260924_newworld/work/blocks/D_onset_realpriv.parquet')
Dall = pd.concat([Dv, Dp], ignore_index=True); Dall['link_id'] = Dall.link_id.astype(str)
Dref = Dall.set_index(['window_id', 'timestamp', 'link_id']).prob
# Package edit (2026-11-07): the A1 and FE reference files are parity checks only (assertions (2), (3)); they are
# read and asserted when present, skipped when absent (a cold run does not have them).
A1ref = None
if (OX / 'onset_A1_official.parquet').exists():
    A1ref = pd.read_parquet(OX / 'onset_A1_official.parquet'); A1ref['link_id'] = A1ref.link_id.astype(str); A1ref = A1ref.set_index(['window_id', 'timestamp', 'link_id']).prob
# LIt: link index (milepost order) of each multi-site panel, from the train
# cache link list.
LIt = {p: {k: i for i, k in enumerate(np.load(C.OS / 'cache' / f'{p}_train.npz', allow_pickle=True)['links'].astype(str))} for p in C.MS}


def decode_tpl(comp, choice, thr=0.5):
    """= onset_extent/s3_real.decode_tpl (template >= thr inside the chosen site; D fallback if empty)."""
    # comp: production onset rows; choice: {window: {site}}. Returns a copy
    # with queue_pred = template >= thr inside the chosen site, or the
    # prod_else_tpl decode of serve_official when that is empty.
    parts = []
    for wid, g in comp.groupby('window_id', sort=False):
        keep = choice.get(wid)
        if keep is None: parts.append(g); continue
        g = g.copy(); pa = g.panel.iloc[0]; site = np.array([site_of(pa, LIt[pa][l]) for l in g.link_id])
        y = np.zeros(len(g), bool)
        for s in keep:
            tp = C.TPL.get((pa, s), {}); y |= np.array([tp.get(l, 0) >= thr for l in g.link_id]) & (site == s)
        if not y.any():
            y = decode(g.assign(), {wid: keep}, C.TPL).queue_pred.to_numpy().astype(bool)
        g['queue_pred'] = y.astype('int8'); parts.append(g)
    return pd.concat(parts)


def to_rows(comp, dec):
    """Returns D's rows of the windows in `comp`, with prob = decision on the multi-site panels.

    Args:
      comp: production onset rows of one split.
      dec: decoded rows (window_id, timestamp, link_id, queue_pred).

    Returns:
      Frame with window_id, timestamp, link_id and prob; panels outside
      common.MS keep D's prob.
    """
    new = Dall[Dall.window_id.isin(comp.window_id)].merge(dec[['window_id', 'timestamp', 'link_id', 'queue_pred']], on=['window_id', 'timestamp', 'link_id'], how='left')
    pan = new.window_id.str.extract(r'Q2_(.+)_(?:validation|private)_')[0]
    new['prob'] = np.where(pan.isin(C.MS), new.queue_pred.astype(float), new.prob)
    return new[['window_id', 'timestamp', 'link_id', 'prob']]


summary = {}; allnew = []; det = []
for split in ('validation', 'private'):
    # Per-site rows of the official onset windows (released history T-60 ..
    # T-5 min and the masked layer at T), with u-model and FE scores. ch0 is
    # D's choice (u-model); ch1 is the FE choice, skipping the same windows.
    offi = official_rows(split); offi = offi[offi.panel.isin(C.MS)].reset_index(drop=True)
    O = R2.prep_rows(offi); O['fe'] = F.fe_score(O)
    ch0 = C.choices(O, 'u'); ch1 = C.choices(O, 'fe', skip_from=O)
    comp = prod[prod.split == split]
    # Four decodes: D (u choice, prod_else_tpl), A1 (u choice, template), FE
    # (FE choice, prod_else_tpl) and FE+A1 (FE choice, template).
    rD = to_rows(comp, decode(comp, ch0, C.TPL)); rA1 = to_rows(comp, decode_tpl(comp, ch0))
    rFE = to_rows(comp, decode(comp, ch1, C.TPL)); rFA = to_rows(comp, decode_tpl(comp, ch1))
    key = lambda r: pd.MultiIndex.from_frame(r[['window_id', 'timestamp', 'link_id']])
    # Assertion (1): the recomputed D decode equals D's file on every cell.
    m1 = int(((rD.prob >= .5) != (Dref.reindex(key(rD)).to_numpy() >= .5)).sum()); assert m1 == 0, ('D repro', m1)
    m2 = m3 = None
    if A1ref is not None:
        m2 = int(((rA1.prob >= .5) != (A1ref.reindex(key(rA1)).to_numpy() >= .5)).sum()); assert m2 == 0, ('A1 repro', m2)
    if (OJ / 'real' / f'onset_FE_{split}.parquet').exists():
        fe_ref = pd.read_parquet(OJ / 'real' / f'onset_FE_{split}.parquet'); fe_ref['link_id'] = fe_ref.link_id.astype(str)
        fe_ref = fe_ref.set_index(['window_id', 'timestamp', 'link_id']).prob
        m3 = int(((rFE.prob >= .5) != (fe_ref.reindex(key(rFE)).to_numpy() >= .5)).sum()); assert m3 == 0, ('FE repro', m3)
    # Windows and cells where FE+A1 differs from D, for the change log.
    old = Dref.reindex(key(rFA)).to_numpy() >= .5; new = rFA.prob.to_numpy() >= .5
    chw = sorted(set(rFA.window_id[old != new]))
    for w in chw:
        g = rFA[rFA.window_id == w]; o = Dref.reindex(key(g)).to_numpy() >= .5; n = g.prob.to_numpy() >= .5
        gA = rA1[rA1.window_id == w].prob.to_numpy() >= .5; gF = rFE[rFE.window_id == w].prob.to_numpy() >= .5
        det.append(dict(split=split, window_id=w, D_site=sorted(ch0.get(w, {'-'}))[0], FE_site=sorted(ch1.get(w, {'-'}))[0],
                        n_D=int(o.sum()), n_A1=int(gA.sum()), n_FE=int(gF.sum()), n_FEA1=int(n.sum()), added=int((n & ~o).sum()), removed=int((~n & o).sum()),
                        links_FEA1=' '.join(g.link_id[n])))
    summary[split] = dict(D_repro_mismatch=m1, A1_repro_mismatch=m2, FE_repro_mismatch=m3, n_choice=len(ch0),
                          choice_changed=sorted(k for k in ch1 if ch1[k] != ch0.get(k)), changed_windows=chw, changed_cells=int((old != new).sum()))
    allnew.append(rFA)
    print(split, json.dumps(summary[split]), flush=True)
# sv2_official_changes.csv: per changed window, the D and FE sites and cell
# counts. The FE+A1 rows of both splits form the onset output.
DET = pd.DataFrame(det); DET.to_csv(ME / 'sv2_official_changes.csv', index=False); print(DET.to_string(index=False))
onset_new = pd.concat(allnew, ignore_index=True); onset_new.to_parquet(ME / 'onset_FEA1_official.parquet', index=False)

# ------------------------------------------------------------------ candidate from G1
def sha(p):
    """Returns the SHA256 hex digest of a file, read in 16 MB chunks."""
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


# Development record: the candidate CSV built on the G1 submission. In the
# reproduction the parent CSV is absent and the script stops below.
PARENT = ROOT / 'build/review_20260925/candidates/submission_G1_F2_g2v2_guard.csv'
PARENT_SHA = '76150ccc642315ac410b05df1384463044eb8e44cede7a317df94587144bc5e8'
if not PARENT.exists():  # package edit (2026-11-07): no G1 parent CSV in the reproduction; pipeline/assemble.py writes the submission
    print('parent CSV absent: stopping after onset_FEA1_official.parquet', flush=True); sys.exit(0)
assert sha(PARENT) == PARENT_SHA, 'parent SHA mismatch'
dst = ME / 'submission_G1_FE_A1.csv'; assert not dst.exists(), dst
kq = pd.read_csv(REL / 'submission_key.csv', usecols=['submission_id', 'task', 'split', 'timestamp', 'link_id', 'window_id'], dtype=str)
qk = kq[kq.task == 'queue'].copy(); qk['submission_id'] = qk.submission_id.astype(int); del kq
g1 = pd.read_csv(PARENT, usecols=['submission_id', 'task', 'queue_pred']); g1 = g1[g1.task == 'queue'].set_index('submission_id').queue_pred
k = qk.merge(Dall, on=['window_id', 'timestamp', 'link_id'], how='inner', validate='one_to_one'); assert len(k) == len(Dall)
k['g1'] = g1.reindex(k.submission_id).to_numpy(); mism = int((k.g1 != (k.prob >= 0.5).astype(int)).sum()); assert mism == 0, ('G1 vs D onset', mism)
wins = set(onset_new.window_id)
k2 = qk[qk.window_id.isin(wins)].merge(onset_new, on=['window_id', 'timestamp', 'link_id'], how='left')
k2['g1'] = g1.reindex(k2.submission_id).to_numpy(); k2['new'] = (k2.prob.fillna(0) >= 0.5).astype(int)
steps15_pos = int(k2[k2.prob.isna()].g1.sum()); assert steps15_pos == 0, ('G1 positives at onset steps 1-5', steps15_pos)
chg = k2[k2.g1 != k2.new]
chg[['submission_id', 'split', 'window_id', 'timestamp', 'link_id', 'g1', 'new']].rename(columns={'new': 'queue_pred'}).to_parquet(ME / 'queue_override_FEA1.parquet', index=False)
newq = dict(zip(chg.submission_id.astype(int), chg.new.astype(int)))
split_of = dict(zip(qk.submission_id, qk.split)); changed = {'queue_validation': 0, 'queue_private': 0}; n01 = n10 = 0
with open(PARENT) as fi, open(dst, 'w') as fo:
    fo.write(fi.readline())
    for line in fi:
        sid = int(line[:line.index(',')])
        if sid in newq:
            p = line.rstrip('\n').split(','); assert p[1] == 'queue'; v = str(newq[sid])
            if p[4] != v:
                n01 += int(v == '1'); n10 += int(v == '0'); p[4] = v; line = ','.join(p) + '\n'; changed['queue_' + split_of[sid]] += 1
        fo.write(line)
rec = dict(candidate=str(dst), sha256=sha(dst), parent=str(PARENT), parent_sha256=PARENT_SHA,
           change='onset FE+A1 on D12_I5_N/D12_I5_S/D7_I405_S: site = onset_joint FE (mean softmax of 24 relative-feature conditional logits), '
                  'extent = train site template >= .5 inside the chosen site (onset_extent A1). Everything else = G1.',
           g1_vs_D_onset_mismatch=mism, g1_onset_steps1to5_positives=steps15_pos, rows_changed=changed, zero_to_one=n01, one_to_zero=n10,
           windows=sorted(chg.window_id.unique()), official=summary)
json.dump(rec, open(str(dst) + '.receipt.json', 'w'), indent=1, default=str); print(json.dumps(rec, indent=1, default=str))
