"""Replicated first-onset events on train truth.
Event = time tq (queue at some link) with corridor-wide no queue in [tq-17, tq-1] (12-step history + 5 clear horizon
steps), i.e. an onset window origin T = tq-6. Also emits the replicated selector windows (360-min spacing chain).
Blocks at tq: maximal runs of queued links (allow 1-link gaps); head = max index (queues grow to lower index).
"""
# Training record for the onset event table. The reproduction does not run this program; it
# reads the shipped table. The program reads the full train caches (not shipped) next to its
# original place in the development repository, and the release under the hard-coded
# development root in `REL` (mapped by the path layer).
#
# Data: `cache/<corridor>_train.npz` (days x links x 288 five-minute slots; speed `V` in km/h
#   and score eligibility `E`; train days 2030-06-01 .. 2031-02-28, February included; the
#   shipped train caches keep only the link order) of the 4 corridors in `PANELS`, and the
#   released `links.csv` (free-flow speed).
# Queue rule here: speed <= 0.6 x free-flow speed on a score-eligible cell.
# Output: `events_train.parquet`, one row per queued block at the onset step: corridor,
#   date, day index `di`, onset slot `tq`, first and last link index `a`/`b` of the block,
#   queued links `n`, blocks at that slot `nblk`. Shipped under
#   `trafficflowbench/build/review_20260924_ctm/ruler/onset_site/` (1,471 block rows,
#   1,231 onset events). It feeds the A1 extent templates and the rule of at least 10
#   training onsets per site of the onset site models.
# Invocation: `python events.py`. This program writes the events only; the replicated
#   selector windows named in the docstring are built elsewhere.
import numpy as np, pandas as pd, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REL = Path('/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public')
PANELS = ['D7_I210_W', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S']  # ruler copy: only multi-site panels (events used for templates + n_events filter)


def load(panel, split='train'):
    """Loads one corridor's cache and its link table in cache link order.

    Returns:
        (cache arrays, link ids, `links.csv` rows indexed by link id).
    """
    z = np.load(HERE / 'cache' / f'{panel}_{split}.npz', allow_pickle=True)
    links = z['links'].astype(str)
    lk = pd.read_csv(REL / 'corridors' / panel / 'network' / 'links.csv', dtype={'link_id': str}).set_index('link_id').reindex(links)
    return z, links, lk


def blocks(q):
    """Returns the queued runs of one time slot as (first, last) link index pairs.

    A single free link between queued links stays inside the run.

    Args:
        q: Boolean queue state per link (1-D).
    """
    idx = np.where(q)[0]
    if len(idx) == 0: return []
    out = []; s = idx[0]; p = idx[0]
    for i in idx[1:]:
        if i - p > 2:
            out.append((s, p)); s = i
        p = i
    out.append((s, p)); return out


rows = []
for panel in PANELS:
    z, links, lk = load(panel)
    V = z['V'].astype(np.float32); E = z['E']; dates = z['dates']
    vc = (0.6 * lk.free_speed_kmh.to_numpy())[None, :, None]
    Q = (V <= vc) & E
    D, L, Tn = Q.shape
    anyq = Q.any(1)  # D x T
    # Onset slot tq: a queue somewhere on the corridor at tq and none in the 17 slots before
    # (12 history + 5 clear horizon slots), so T = tq - 6 is an onset window origin.
    for d in range(D):
        for tq in range(18, Tn):
            if anyq[d, tq] and not anyq[d, tq - 17:tq].any():
                bl = blocks(Q[d, :, tq])
                for (a, b) in bl:
                    rows.append(dict(panel=panel, date=str(dates[d]), di=d, tq=tq, a=int(a), b=int(b), n=int(Q[d, a:b + 1, tq].sum()), nblk=len(bl)))
ev = pd.DataFrame(rows)
ev.to_parquet(HERE / 'events_train.parquet', index=False)
print(ev.groupby('panel').agg(n_events=('tq', 'size'), days=('date', 'nunique')))
print(ev.n.describe())
