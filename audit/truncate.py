"""Build a truncated view of the released data: every observation stamped after a cut is removed.

  python audit/truncate.py --release <kaggle_public> --out <view_dir> --cut 2031-03-05T07:10:00Z [--drop-stamps N]

The view has the folder layout of the release. Files are symbolic links to the release, except the
files the cut changes, which are written as truncated copies. Nothing in the release is written.

What the cut removes, in the validation and private splits (the training split is left as is):

- corridors/<panel>/<split>/mainline_states_masked/*/synthetic_mainline_<day>.parquet and
  corridors/<panel>/<split>/ramp_states/*/synthetic_ramp_<day>.parquet (one file per UTC day):
  days after the cut day are left out; the cut day keeps the rows stamped at or before the cut;
- task2/<panel>/<split>/window_index.csv: windows whose forecast origin is after the cut;
- task2/<panel>/<split>/window_history.parquet: rows stamped after the cut, and the rows of the
  windows removed from window_index.csv. Such a window has not started at the cut; its history rows
  stamped before the cut are copies of the masked layer, which the view keeps.

--drop-stamps N (negative control) also removes the last N stamps up to the cut: N = 1 removes the
rows stamped at the cut (the window whose origin is the cut loses its origin observation T), N = 2
also those at T-5 (the last stamp of that window's history). --strict is the same as --drop-stamps 1.

Left unchanged: the training split, the network and config files, and the files that list what to
predict rather than what was observed (submission_key.csv, sample_submission.csv, the per-task
sample_submission_*.csv). Their queue rows of later windows are never reached, because the serving
code maps key rows only to the windows listed in window_index.csv.

Writes <view_dir>/TRUNCATION.json with the cut and the count of rows removed per file kind.
"""
import argparse, json, os, re, sys
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DAY = re.compile(r'_(\d{4})_(\d{2})_(\d{2})\.parquet$')
SPLITS = ('validation', 'private')


def link(src, dst):
    """Creates dst as a symbolic link to src, with its parent folders."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


def keep_rows(src, dst, col, cut, strict, windows=None):
    """Writes the rows of parquet `src` with `col` <= cut (< `strict` when that is set) to dst; returns rows removed.

    With `windows`, only rows whose window_id is in that set are kept.
    """
    t = pq.read_table(src)
    m = pc.less(t[col], strict) if strict else pc.less_equal(t[col], cut)
    if windows is not None:
        m = pc.and_(m, pc.is_in(t['window_id'], value_set=pa.array(sorted(windows), pa.string())))
    k = t.filter(m)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(k, dst)
    return t.num_rows - k.num_rows


def main():
    """Builds the view described in the module docstring."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cut', required=True, help="UTC time, as the release writes it: 2031-03-05T07:10:00Z")
    ap.add_argument('--drop-stamps', type=int, default=0)
    ap.add_argument('--strict', action='store_true')
    a = ap.parse_args()
    rel, out = Path(a.release).resolve(), Path(a.out)
    cut = pd.Timestamp(a.cut).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    n = a.drop_stamps or (1 if a.strict else 0)
    # strict: the first removed stamp (rows stamped at or after it are removed); '' when nothing extra.
    a.strict = (pd.Timestamp(cut) - pd.Timedelta(minutes=5 * (n - 1))).strftime('%Y-%m-%dT%H:%M:%SZ') if n else ''
    cut_day = cut[:10]
    assert not a.strict or a.strict[:10] == cut_day, 'the removed stamps must lie on the cut day'
    if out.exists():
        sys.exit(f'refusing to reuse {out}')
    stats = {'day_files_linked': 0, 'day_files_cut': 0, 'day_files_dropped': 0, 'day_rows_removed': 0,
             'history_rows_removed': 0, 'windows_removed': 0, 'windows_kept': 0}
    for src in sorted(rel.rglob('*')):
        if src.name.startswith('._') or src.name == '.DS_Store' or not src.is_file():
            continue
        r = src.relative_to(rel); dst = out / r; parts = r.parts
        # corridors/<panel>/<split>/{mainline_states_masked,ramp_states}/<partition>/<day file>
        if parts[0] == 'corridors' and len(parts) >= 6 and parts[2] in SPLITS:
            m = DAY.search(src.name)
            if not m:
                sys.exit(f'unexpected file in a split folder: {r}')
            day = f'{m[1]}-{m[2]}-{m[3]}'
            if day < cut_day:
                link(src, dst); stats['day_files_linked'] += 1
            elif day == cut_day:
                stats['day_rows_removed'] += keep_rows(src, dst, 'timestamp', cut, a.strict); stats['day_files_cut'] += 1
            else:
                stats['day_files_dropped'] += 1
                stats['day_rows_removed'] += pq.read_metadata(src).num_rows
        # task2/<panel>/<split>/{window_history.parquet,window_index.csv}
        elif parts[0] == 'task2' and len(parts) == 4 and parts[2] in SPLITS and src.name == 'window_history.parquet':
            w = pd.read_csv(src.parent / 'window_index.csv', dtype=str, keep_default_na=False)
            kept = set(w.window_id[pd.to_datetime(w.forecast_origin, utc=True) <= pd.Timestamp(cut)])
            stats['history_rows_removed'] += keep_rows(src, dst, 'timestamp', cut, a.strict, kept)
        elif parts[0] == 'task2' and len(parts) == 4 and parts[2] in SPLITS and src.name == 'window_index.csv':
            w = pd.read_csv(src, dtype=str, keep_default_na=False)
            keep = pd.to_datetime(w.forecast_origin, utc=True) <= pd.Timestamp(cut)
            stats['windows_removed'] += int((~keep).sum()); stats['windows_kept'] += int(keep.sum())
            dst.parent.mkdir(parents=True, exist_ok=True)
            w[keep].to_csv(dst, index=False)
        else:
            link(src, dst)
    (out / 'TRUNCATION.json').write_text(json.dumps(dict(cut=cut, drop_stamps=n, removed_from=a.strict or None, release=str(rel), **stats), indent=1) + '\n')
    print(json.dumps(dict(cut=cut, drop_stamps=n, removed_from=a.strict or None, **stats)))


if __name__ == '__main__':
    main()
