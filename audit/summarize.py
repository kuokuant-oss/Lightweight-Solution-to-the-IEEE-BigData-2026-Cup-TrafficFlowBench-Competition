"""Checkpoints and summary of the truncation audit (called by run_truncation.sh).

  python audit/summarize.py --checkpoint <work>/<name>.json <name> <results>/cut_<name>.json
      writes one cut's checkpoint: τ, windows, cells, changed, rows missing or extra, the cells of the
      window(s) at τ, wall time, and the per-kind detail of audit/compare.py;
  python audit/summarize.py --line <results>/cut_<name>.json
      prints the one-line form used in progress.log;
  python audit/summarize.py <dir>
      reads every cut_<name>.json in <dir> (or, when there is none, the <name>.json files of a work
      folder), in the order of audit/cuts.csv, and writes <dir>/summary.md (the cut table and the
      negative-control table, in the form of audit/RESULTS.md) and <dir>/summary.json (the totals).
"""
import json, sys
from pathlib import Path
import pandas as pd

CUTS = pd.read_csv(Path(__file__).resolve().parent / 'cuts.csv', dtype=str).set_index('name')


def checkpoint(d, name):
    """Returns the checkpoint record of one cut from the compare.py output `d` (with wall_s added)."""
    on, og = d['onset'], d['ongoing']
    s = lambda k: on[k] + og[k]
    return dict(name=name, window_id=d['window_id'], tau=pd.Timestamp(d['cut']).strftime('%Y-%m-%dT%H:%MZ'), splits=d['splits'],
                strict=bool(d['strict']), drop_stamps=int(d.get('drop_stamps', 1 if d['strict'] else 0)), note=CUTS.note.get(name, ''), windows=s('windows'), onset_windows=on['windows'],
                ongoing_windows=og['windows'], cells=s('cells'), changed=s('changed'), missing=s('missing'), extra=s('extra'),
                at_cut_cells=s('at_cut_cells'), at_cut_changed=s('at_cut_changed'), wall_s=d['wall_s'],
                changed_windows=on['changed_windows'] + og['changed_windows'], onset=on, ongoing=og, **prob_totals(d.get('probs')))


def prob_totals(pr):
    """Totals over all probability sources: values compared, changed and max |Δp|, before τ and at τ."""
    if not pr:
        return {}
    v = list(pr.values())
    return dict(probs=pr, prob_values=sum(x['values'] for x in v), prob_earlier_changed=sum(x['earlier_changed'] for x in v),
                prob_earlier_max=max(x['earlier_max_abs_diff'] for x in v), prob_at_cut_values=sum(x['at_cut_values'] for x in v),
                prob_at_cut_changed=sum(x['at_cut_changed'] for x in v), prob_at_cut_max=max(x['at_cut_max_abs_diff'] for x in v))


def summary(folder):
    """Writes summary.md and summary.json for the cuts recorded in `folder`."""
    folder = Path(folder); recs = []
    for name in CUTS.index:
        ck, wk = folder / f'cut_{name}.json', folder / f'{name}.json'
        if ck.exists():
            recs.append(json.loads(ck.read_text()))
        elif wk.exists() and 'wall_s' in json.loads(wk.read_text()):
            recs.append(checkpoint(json.loads(wk.read_text()), name))
    r = pd.DataFrame(recs)
    cut, ctl = r[~r.strict], r[r.strict]
    L = ['| Cut | Window at τ | τ (UTC) | Splits run | Windows compared (onset + ongoing) | Cells compared | Cells changed | Rows missing / extra | Wall time |',
         '|---|---|---|---|---:|---:|---:|---:|---:|']
    for x in cut.itertuples():
        L.append(f'| {x.name} | `{x.window_id}` | {x.tau} | {x.splits} | {x.windows} ({x.onset_windows} + {x.ongoing_windows}) | {x.cells:,} | {x.changed} | {x.missing} / {x.extra} | {x.wall_s} s |')
    if len(cut):
        L.append(f'| **Total** | {len(cut)} cuts | | | {cut.windows.sum():,} | {cut.cells.sum():,} | {cut.changed.sum()} | {cut.missing.sum()} / {cut.extra.sum()} | |')
    L += ['', '| Control | Window at τ | τ (UTC) | Cells of the window at τ | Of them changed | Cells of earlier windows | Of them changed | Windows that changed |',
          '|---|---|---|---:|---:|---:|---:|---|']
    for x in ctl.itertuples():
        L.append(f'| {x.name} | `{x.window_id}` | {x.tau} | {x.at_cut_cells} | {x.at_cut_changed} | {x.cells - x.at_cut_cells:,} | {x.changed - x.at_cut_changed} | '
                 + (', '.join(f'`{w}`' for w in x.changed_windows) or 'none') + ' |')
    if 'prob_values' in r:
        pr = r[r.prob_values.notna()]
        L += ['', '| Run | Window at τ | Stamps removed at τ | Probability values compared | Windows before τ: changed | max abs Δp | Window at τ: values | changed | share | max abs Δp | Decision cells changed at τ |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for x in pr.itertuples():
            share = x.prob_at_cut_changed / x.prob_at_cut_values if x.prob_at_cut_values else 0
            L.append(f'| {x.name} | `{x.window_id}` | {int(x.drop_stamps)} | {int(x.prob_values):,} | {int(x.prob_earlier_changed)} | {x.prob_earlier_max:.3g} | '
                     f'{int(x.prob_at_cut_values):,} | {int(x.prob_at_cut_changed):,} | {share:.1%} | {x.prob_at_cut_max:.3g} | {x.at_cut_changed} of {x.at_cut_cells} |')
        L += ['', '| Run | Source | Values (window at τ) | Changed | Share | max abs Δp |', '|---|---|---:|---:|---:|---:|']
        for x in pr[pr.strict].itertuples():
            for s, v in x.probs.items():
                if v['at_cut_values']:
                    L.append(f"| {x.name} | {s} | {v['at_cut_values']:,} | {v['at_cut_changed']:,} | {v['at_cut_changed'] / v['at_cut_values']:.1%} | {v['at_cut_max_abs_diff']:.3g} |")
    (folder / 'summary.md').write_text('\n'.join(L) + '\n')
    tot = dict(cuts=int(len(cut)), windows=int(cut.windows.sum()) if len(cut) else 0, cells=int(cut.cells.sum()) if len(cut) else 0,
               changed=int(cut.changed.sum()) if len(cut) else 0, wall_s=int(r.wall_s.sum()) if len(r) else 0,
               controls=[dict(name=x.name, window=x.window_id, at_cut_cells=int(x.at_cut_cells), at_cut_changed=int(x.at_cut_changed),
                              earlier_changed=int(x.changed - x.at_cut_changed)) for x in ctl.itertuples()])
    (folder / 'summary.json').write_text(json.dumps(tot, indent=1) + '\n')
    print('\n'.join(L)); print(json.dumps(tot))


if __name__ == '__main__':
    if sys.argv[1] == '--checkpoint':
        src, name, dst = sys.argv[2:5]
        Path(dst).write_text(json.dumps(checkpoint(json.loads(Path(src).read_text()), name), indent=1) + '\n')
    elif sys.argv[1] == '--line':
        d = json.loads(Path(sys.argv[2]).read_text())
        print(f"tau {d['tau']} windows {d['windows']} cells {d['cells']} changed {d['changed']} "
              f"at_cut {d['at_cut_changed']}/{d['at_cut_cells']} wall {d['wall_s']} s"
              + (f" | probs {d['prob_values']} values, before τ changed {d['prob_earlier_changed']} max {d['prob_earlier_max']:.3g},"
                 f" at τ changed {d['prob_at_cut_changed']}/{d['prob_at_cut_values']} max {d['prob_at_cut_max']:.3g}" if 'prob_values' in d else ''))
    else:
        summary(sys.argv[1])
