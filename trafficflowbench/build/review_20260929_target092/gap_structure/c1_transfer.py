"""C1: transfer test of the in-gap ramp speed correction to the released scored months.
March (validation) and April (private) strict pseudo 90-min holes of the HMP50 harness (review_20260927_research):
857,083 / 849,387 visible-label cells; base = HMP50 strict (pool50).  The correction models are the January-fit
residual models from b3 (frozen before this script; no refit here).  Ramp features from the split's own released
ramp_states (Task 1 offline).  Labels are released visible mainline values inside the pseudo holes."""
# Research record; pipeline/task1.sh does not run it. c4_phead_transfer.py and
# g2_persplit.py import its paths and helpers (rel, base_preds, R2, R3, KEY,
# PANELS, FMAP).
# Models: b3_<name>_janfit.txt, speed-residual LightGBMs fit on January 2031
# gap rows; sp_M2 reads SP + RAMP_FEATS + NEAR_FEATS. Variants: pool50 + 0.5 x
# correction (h05), pool50 + 1.0 x correction (h10), and the mean of pool50
# and G2v2 + correction (avg).
# Usage: c1_transfer.py <validation|private>
# Outputs: c1_<split>_preds.parquet and c1_<split>_results.json.
import os, sys, json, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ramp_feats as RF
from b2_ramp_resid import SP, FLOOR, CAP
# Original absolute root; bin/tfb-run maps it onto the run tree. R2 / R3:
# pseudo-hole tables of the validation / private split.
ROOT = Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
RES = ROOT / 'build/review_20260927_research'
R2 = ROOT / 'build/review_20260925/round2/task1_trans/work'; R3 = ROOT / 'build/review_20260925/round3/task1_trans_serving/work'
KEY = ['panel', 'date', 'origin', 'li', 'step_i']
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']
MODELS = {nm: HERE / f'b3_{nm}_janfit.txt' for nm in ('sp_M1', 'sp_M2', 'sp_N')}
# FMAP: the base speed / flow features take the G2v2 hole values ch_v / ch_ql.
FMAP = {'p_v': 'ch_v', 'p_ql': 'ch_ql'}


def base_preds(split):
    """Loads the strict held-out HMP50 hole predictions of one split.

    Returns:
      Rows (KEY, y_v, day, cond, t1, pool50) of the split's pseudo holes;
      pool50 = .5 x T1 + .5 x pooled mixture (2 x adaptive - T1 for the
      private split, where adaptive is the 25 % blend).
    """
    if split == 'validation':
        d = pd.read_parquet(RES / 'strength/validation_predictions.parquet', columns=KEY + ['y_v', 'day', 'cond', 't1', 'pool50'])
    else:
        d = pd.read_parquet(RES / 'full_pool/private_predictions.parquet', columns=KEY + ['y_v', 'day', 'cond', 't1', 'adaptive'])
        d['pool50'] = 2 * d.adaptive - d.t1; d = d.drop(columns='adaptive')
    return d


def rel(t, y, base, new, B=2000):
    """Compares a candidate speed column with a base column on the same cells.

    Args:
      t: Rows with panel, date, cond and day.
      y: Truth column.
      base: Reference column.
      new: Candidate column.
      B: Number of panel-day bootstrap draws.

    Returns:
      Dict: cell count, both RMSEs, relative MSE change (new / base - 1), its
      95 % panel-day bootstrap interval, the change for onset / ongoing
      windows, odd / even days and days 1-12, per panel, and the number of
      panels that improve.
    """
    e0 = (t[base] - t[y]) ** 2; e1 = (t[new] - t[y]) ** 2
    g = pd.DataFrame(dict(k=t.panel + '|' + t.date, e0=e0, e1=e1)).groupby('k')[['e0', 'e1']].sum()
    rng = np.random.default_rng(0); ix = rng.integers(0, len(g), (B, len(g)))
    out = dict(n=len(t), rmse0=float(np.sqrt(e0.mean())), rmse1=float(np.sqrt(e1.mean())), rel=float(e1.sum() / e0.sum() - 1),
               ci=np.percentile(g.e1.to_numpy()[ix].sum(1) / g.e0.to_numpy()[ix].sum(1) - 1, [2.5, 97.5]).round(5).tolist())
    for nm, m in (('onset', t.cond == 'queue_onset'), ('ongoing', t.cond == 'queue_ongoing'), ('odd', t.day % 2 == 1),
                  ('even', t.day % 2 == 0), ('early12', t.day <= 12)):
        out[nm] = round(float(e1[m].sum() / e0[m].sum() - 1), 5)
    out['perpanel'] = {p: round(float(((q[new] - q[y]) ** 2).sum() / ((q[base] - q[y]) ** 2).sum() - 1), 5) for p, q in t.groupby('panel')}
    out['panels_improved'] = int(sum(v < 0 for v in out['perpanel'].values()))
    return out


def main():
    """Applies the frozen January corrections to one split's pseudo holes and scores the variants against pool50."""
    split = sys.argv[1]
    src = R2 if split == 'validation' else R3
    base = base_preds(split)
    boosters = {nm: lgb.Booster(model_file=str(p)) for nm, p in MODELS.items()}
    feat_cols = {nm: b.feature_name() for nm, b in boosters.items()}
    parts = []
    for panel in PANELS:
        cols = ['f_' + c for c in SP if c not in FMAP] + ['ch_v', 'ch_ql', 'y_ql']
        ph = pd.read_parquet(src / f'ph_{split}_{panel}.parquet', columns=KEY + cols)
        b = base[base.panel == panel]
        x = b.merge(ph, on=KEY, how='inner', validate='one_to_one')
        assert len(x) == len(b), (panel, len(x), len(b))
        for c in SP:
            if c not in FMAP: x[c] = x['f_' + c]
        x['p_v'] = x.ch_v; x['p_ql'] = x.ch_ql
        x = x.reset_index(drop=True)
        # Ramp counts from the split's own released ramp_states (Task 1 may
        # read the whole month).
        x = RF.attach(x, split); x = RF.attach_near(x, split)
        for nm, bst in boosters.items():
            x['c_' + nm] = bst.predict(x[feat_cols[nm]], num_threads=2)
        keep = KEY + ['y_v', 'y_ql', 'day', 'cond', 't1', 'pool50', 'ch_v', 'ch_ql'] + ['c_' + nm for nm in boosters] + \
               ['nOR1_prof', 'dnOR_post', 'cOR_prof']
        parts.append(x[keep]); print(panel, len(x), flush=True)
        del x, ph; gc.collect()
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(HERE / f'c1_{split}_preds.parquet', index=False)
    out = {}
    for nm in boosters:
        t[f'{nm}_h05'] = np.clip(t.pool50 + 0.5 * t['c_' + nm], FLOOR, CAP)
        t[f'{nm}_h10'] = np.clip(t.pool50 + 1.0 * t['c_' + nm], FLOOR, CAP)
        t[f'{nm}_avg'] = 0.5 * t.pool50 + 0.5 * np.clip(t.ch_v + t['c_' + nm], FLOOR, CAP)
        for v in ('h05', 'h10', 'avg'):
            out[f'{nm}_{v}'] = rel(t, 'y_v', 'pool50', f'{nm}_{v}')
            print(split, nm, v, json.dumps({k: out[f'{nm}_{v}'][k] for k in ('rmse0', 'rmse1', 'rel', 'ci', 'onset', 'ongoing', 'odd', 'even', 'early12', 'panels_improved')}), flush=True)
    t['g2_clean_M2'] = np.clip(t.ch_v + t.c_sp_M2, FLOOR, CAP); out['M2_clean_vs_g2'] = rel(t, 'y_v', 'ch_v', 'g2_clean_M2')
    print(split, 'M2 clean vs G2v2', json.dumps({k: out['M2_clean_vs_g2'][k] for k in ('rmse0', 'rmse1', 'rel', 'ci', 'panels_improved')}))
    json.dump(out, open(HERE / f'c1_{split}_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
