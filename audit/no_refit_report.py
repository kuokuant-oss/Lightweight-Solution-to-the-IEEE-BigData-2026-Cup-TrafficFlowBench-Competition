"""Report of the no-refit check (audit/run_no_refit.sh).

  python audit/no_refit_report.py --out <run_no_refit out dir> --ref <full-run out dir> [--md <report.md>]

1. Code that ran: the union of the run-tree modules every Python process of onset.sh and ongoing.sh
   loaded (probe 'exit' records), plus the program files that onset.sh and ongoing.sh name.
2. Static list: every line of that code that calls a fitting or training routine (pattern FIT below).
3. Dynamic record: every fitting call the probe saw at run time, by program and entry point.
4. Outputs: the probed run's onset and ongoing sources against the reference run (SHA256).
"""
import argparse, collections, hashlib, json, re
from pathlib import Path

FIT = re.compile(r'\.fit\(|\.fit_transform\(|partial_fit\(|\blgb\.train\(|lightgbm\.train\(|\blgb\.cv\(|CatBoost\w*\(|'
                 r'LogisticRegression\(|GradientBoosting\w*\(|RandomForest\w*\(|\bRidge\(|\bLasso\(|LinearRegression\(|'
                 r'\bminimize\(|least_squares\(|\bnnls\(|lsq_linear\(|curve_fit\(|polyfit\(|lstsq\(|\.train\(|optim\.\w+\(')


def sha(p):
    """Returns the SHA256 hex digest of file p."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    """Writes the report described in the module docstring."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--ref', required=True); ap.add_argument('--md')
    a = ap.parse_args()
    out = Path(a.out); repo = out / 'run/trafficflowbench'; pkg = Path(__file__).resolve().parents[1]
    recs = [json.loads(l) for l in (out / 'probe.jsonl').read_text().splitlines() if l.strip()]
    exits = [r for r in recs if r['kind'] == 'exit']; fits = [r for r in recs if r['kind'] == 'fit']
    mods = set()
    for r in exits:
        mods |= set(r['modules'])
    # Program files named in the segment scripts (a -c program's file is not a loaded module).
    for sh in ('onset.sh', 'ongoing.sh'):
        t = (pkg / 'pipeline' / sh).read_text()
        mods |= {m for m in re.findall(r'((?:build|scripts|pipeline)/[\w./-]+\.py)', t) if (repo / m).exists()}
        mods |= {f'build/review_20260924_newworld/work/blocks/{m}' for m in re.findall(r"run_path\('(\w+\.py)'", t)}
    mods = sorted(m for m in mods if m.endswith('.py') and 'ssd_bootstrap/tfb_fit_probe' not in m)
    hits = []
    for m in mods:
        for i, line in enumerate((repo / m).read_text(errors='replace').splitlines(), 1):
            s = line.strip()
            if s.startswith('#') or not FIT.search(line):
                continue
            hits.append((m, i, s[:150]))
    byprog = collections.Counter((r['program'], r['call']) for r in fits)
    patched = sorted({p for r in exits for p in r.get('patched', [])})
    srcs = {f: (sha(out / 'run/final_sources' / f), sha(Path(a.ref) / 'run/final_sources' / f)) for f in ('onset.parquet', 'ongoing.parquet')}
    L = [f'Processes recorded: {len(exits)}. Run-tree code files that ran: {len(mods)}.', '',
         f'Fitting calls recorded at run time: {len(fits)}.']
    if byprog:
        L += ['', '| Program | Entry point | Calls |', '|---|---|---:|'] + [f'| `{p}` | `{c}` | {n} |' for (p, c), n in sorted(byprog.items())]
    L += ['', f'Entry points the probe had wrapped in at least one process: {len(patched)} '
          f"(lightgbm: {sum(p.startswith('lightgbm') for p in patched)}, sklearn: {sum(p.startswith('sklearn') for p in patched)}, "
          f"scipy.optimize: {sum(p.startswith('scipy') for p in patched)}, numpy: {sum(p.startswith('numpy') for p in patched)}).", '',
          f'Lines in that code that call a fitting routine: {len(hits)}.', '', '| File | Line | Code |', '|---|---:|---|']
    L += [f'| `{m}` | {i} | `{s.replace("|", "/")}` |' for m, i, s in hits]
    L += ['', '| Source | Probed run | Reference | Equal |', '|---|---|---|---|']
    L += [f'| `{f}` | `{x[:12]}…` | `{y[:12]}…` | {x == y} |' for f, (x, y) in srcs.items()]
    L += ['', 'Code files that ran:', ''] + [f'- `{m}`' for m in mods]
    text = '\n'.join(L) + '\n'
    print(text)
    if a.md:
        Path(a.md).write_text(text)


if __name__ == '__main__':
    main()
