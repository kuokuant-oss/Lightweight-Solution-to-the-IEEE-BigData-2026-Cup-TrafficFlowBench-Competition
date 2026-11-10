"""Compare the Task 2 outputs of a truncated run with those of the full run, cell by cell.

  python audit/compare.py --release <kaggle_public> --view <view_dir> --run <truncated out_dir> \
      --ref <full-run out_dir> --cut <τ> [--drop-stamps N] [--probs <dir>] [--json <result.json>]

Decisions. A cell is one queue row of submission_key.csv (window, link, step). The compared windows are
those the truncated view still lists (forecast origin <= τ). For each, the 0/1 prediction of every cell
in <run>/run/final_sources/{onset,ongoing}.parquet must equal the one in the full run's files.

Probabilities (--probs: the folder where run_truncation.sh copied the truncated run's probability
files, at their run-tree paths). Each continuous output that feeds the decisions is compared value by
value with the same file of the full run (<ref>/run/trafficflowbench), on the rows of the compared
windows:
  onset_head       final_sprint_review_20260922/prior/final_feature_hazard_serving.parquet (prob)
  onset_component  strongest_candidate_20260922/onset/production/onset_component.parquet (prob)
  ongoing_A        review_20260924_newworld/work/feat/ogens_<tag>_prob.parquet (prob)
  ongoing_parent, ongoing_M1f, ongoing_M1sf   review_20261001_fresh/serve/prob/<split>_*_prob.npz
  ongoing_PB       review_20261001_r26/serve/prob/<split>_PB_prob.npz (the BND re-prediction)
The npz rows follow serving_stage/<split>_ROW_INDEX.parquet of each run. Reported per source: values
compared, values changed, max |Δp|, for all compared windows and for the window(s) whose origin is τ.

Printed and written to --json. With --drop-stamps (the negative control) the window(s) at τ are
expected to change.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

TAG = {'validation': 'realval', 'private': 'realpriv'}
B = 'build/'
PQ = {'onset_head': B + 'final_sprint_review_20260922/prior/final_feature_hazard_serving.parquet',
      'onset_component': B + 'strongest_candidate_20260922/onset/production/onset_component.parquet'}
NPZ = {'ongoing_parent': (B + 'review_20261001_fresh/serve/prob/{s}_parent_prob.npz', None),
       'ongoing_M1f': (B + 'review_20261001_fresh/serve/prob/{s}_M1f_prob.npz', None),
       'ongoing_M1sf': (B + 'review_20261001_fresh/serve/prob/{s}_M1sf_prob.npz', None),
       'ongoing_PB': (B + 'review_20261001_r26/serve/prob/{s}_PB_prob.npz', None)}
ROWIDX = B + 'review_20260930_round20/queue_data/serving_stage/{s}_ROW_INDEX.parquet'
# Relative paths run_truncation.sh copies before it deletes a run tree.
KEEP = list(PQ.values()) + [B + 'review_20260924_newworld/work/feat/ogens_{t}_prob.parquet'] + [v[0] for v in NPZ.values()] + [ROWIDX]


def stats(new, ref, at):
    """Counts and max |Δ| of two aligned float arrays (rows x columns); `at` marks the rows of the windows at τ."""
    new = np.asarray(new, float).reshape(len(new), -1); ref = np.asarray(ref, float).reshape(len(ref), -1)
    d = np.abs(new - ref); same = (new == ref) | (np.isnan(new) & np.isnan(ref)); d[same] = 0
    d = np.where(np.isnan(d), np.inf, d)   # NaN on one side only counts as a change
    out = dict(values=int(d.size), changed=int((~same).sum()), max_abs_diff=float(d.max()) if d.size else 0.0)
    a = d[at]; out.update(at_cut_values=int(a.size), at_cut_changed=int((~same[at]).sum()), at_cut_max_abs_diff=float(a.max()) if a.size else 0.0)
    e = d[~at]; out.update(earlier_changed=int((~same[~at]).sum()), earlier_max_abs_diff=float(e.max()) if e.size else 0.0)
    return out


def key(df, cols):
    """Returns a string key per row from the given columns (timestamps normalised to ISO UTC)."""
    parts = []
    for c in cols:
        v = df[c]
        if c == 'timestamp':
            v = pd.to_datetime(v, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        parts.append(v.astype(str))
    return parts[0].str.cat(parts[1:], sep='|')


def probs(pdir, ref, splits, wins, at_ids):
    """Compares every probability source of the truncated run (in pdir) with the full run (ref)."""
    res = {}
    def load_pq(path):
        d = pd.read_parquet(path); d = d[d.window_id.isin(wins)].copy(); d['k'] = key(d, ['window_id', 'timestamp', 'link_id'])
        return d.set_index('k')
    for name, rel in PQ.items():
        n, r = load_pq(pdir / rel), load_pq(ref / rel)
        assert set(n.index) == set(r.index), f'{name}: rows differ ({len(n)} vs {len(r)})'
        r = r.loc[n.index]; res[name] = stats(n.prob.to_numpy(), r.prob.to_numpy(), n.window_id.isin(at_ids).to_numpy())
    parts = {'ongoing_A': [], **{k: [] for k in NPZ}}
    for s in splits:
        rel = B + f'review_20260924_newworld/work/feat/ogens_{TAG.get(s, "real" + s)}_prob.parquet'
        n, r = load_pq(pdir / rel), load_pq(ref / rel)
        assert set(n.index) == set(r.index), f'ongoing_A {s}: rows differ'
        r = r.loc[n.index]; parts['ongoing_A'].append((n.prob.to_numpy(), r.prob.to_numpy(), n.window_id.isin(at_ids).to_numpy()))
        ri_n = pd.read_parquet(pdir / ROWIDX.format(s=s)); ri_r = pd.read_parquet(ref / ROWIDX.format(s=s))
        kn = key(ri_n, ['window_id', 'link_id', 'step']); kr = pd.Series(np.arange(len(ri_r)), index=key(ri_r, ['window_id', 'link_id', 'step']))
        assert ri_n.window_id.isin(wins).all() and kn.isin(kr.index).all()
        pos = kr.loc[kn].to_numpy(); at = ri_n.window_id.isin(at_ids).to_numpy()
        for name, (pat, _) in NPZ.items():
            zn, zr = np.load(pdir / pat.format(s=s)), np.load(ref / pat.format(s=s))
            assert zn.files == zr.files
            a = np.concatenate([zn[f].reshape(len(kn), -1) for f in zn.files], 1)
            b = np.concatenate([zr[f].reshape(len(ri_r), -1) for f in zr.files], 1)[pos]
            parts[name].append((a, b, at))
    for name, lst in parts.items():
        if lst:
            res[name] = stats(np.concatenate([x[0].reshape(len(x[0]), -1) for x in lst]), np.concatenate([x[1].reshape(len(x[1]), -1) for x in lst]),
                              np.concatenate([x[2] for x in lst]))
    return res


def main():
    """Runs the comparison described in the module docstring."""
    ap = argparse.ArgumentParser()
    for k in ('--release', '--view', '--run', '--ref', '--cut'):
        ap.add_argument(k, required=True)
    ap.add_argument('--drop-stamps', type=int, default=0)
    ap.add_argument('--probs')
    ap.add_argument('--json')
    a = ap.parse_args()
    cut = pd.Timestamp(a.cut)
    key_ = pd.read_csv(Path(a.release) / 'submission_key.csv', usecols=['submission_id', 'task', 'split', 'window_id'], dtype=str)
    key_ = key_[key_.task.eq('queue')].copy(); key_['submission_id'] = key_.submission_id.astype('int64')
    # Windows the truncated view lists: forecast origin <= τ.
    idx = pd.concat([pd.read_csv(p, dtype=str) for p in sorted(Path(a.view).glob('task2/*/*/window_index.csv'))
                     if p.parent.name != 'train'], ignore_index=True)
    idx['origin'] = pd.to_datetime(idx.forecast_origin, utc=True)
    assert (idx.origin <= cut).all()
    at_ids = set(idx.window_id[idx.origin == cut])
    res = dict(cut=str(cut), drop_stamps=a.drop_stamps, strict=a.drop_stamps > 0)
    for kind, cond in (('onset', 'queue_onset'), ('ongoing', 'queue_ongoing')):
        w = idx[idx.condition.eq(cond)]
        cells = key_[key_.window_id.isin(set(w.window_id))]
        new = pd.read_parquet(Path(a.run) / 'run/final_sources' / f'{kind}.parquet').set_index('submission_id').queue_pred
        ref = pd.read_parquet(Path(a.ref) / 'run/final_sources' / f'{kind}.parquet').set_index('submission_id').queue_pred
        got = new.reindex(cells.submission_id); want = ref.reindex(cells.submission_id)
        assert want.notna().all(), 'the full run must cover every compared cell'
        changed = got.notna() & (got != want)
        at = cells.window_id.isin(at_ids).to_numpy()
        res[kind] = dict(windows=int(len(w)), cells=int(len(cells)), changed=int(changed.sum()),
                         missing=int(got.isna().sum()), extra=int(len(set(new.index) - set(cells.submission_id))),
                         queued=int(want.sum()),
                         at_cut_windows=sorted(w.window_id[w.origin == cut]), at_cut_cells=int(at.sum()),
                         at_cut_changed=int(changed.to_numpy()[at].sum()),
                         changed_windows=sorted(set(cells.window_id[changed.to_numpy()])))
    if a.probs:
        splits = sorted(set(idx.split), key=['validation', 'private'].index)
        res['probs'] = probs(Path(a.probs), Path(a.ref) / 'run/trafficflowbench', splits, set(idx.window_id), at_ids)
    print(json.dumps(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + '\n')


def copy_probs(tree, dest, splits):
    """Copies the probability files (KEEP) of run tree `tree` (its trafficflowbench/) to `dest`, same relative paths."""
    import shutil
    for pat in KEEP:
        for rel in sorted({pat.format(s=s, t=TAG.get(s, 'real' + s)) for s in splits}):
            src = Path(tree) / rel
            if src.exists():
                (Path(dest) / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, Path(dest) / rel)


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['--copy-probs']:   # --copy-probs <run tree trafficflowbench/> <dest> <split> [<split>]
        copy_probs(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        main()
