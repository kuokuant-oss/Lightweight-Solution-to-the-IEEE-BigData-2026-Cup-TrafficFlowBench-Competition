"""Package addition: generic never-empty fallback for the F2 ongoing operator (replaces the hand-made guard file).

An ongoing window always holds a queue in its horizon, so an all-free prediction scores IoU 0. The original chain
handled this with one hard-coded window (G1 guard: Q2_D12_I5_N_validation_007 took candidate D's A-block decode,
written by hand to candidates/guard_val007.parquet; serve_refit.py kept those cells via GUARD = <window id>).
This file states the rule for every window instead, the same rule the R20 operator applies (r22_serve.decision):

  never_empty(window_id, decision, fallback): a window whose decision has no queued cell takes its fallback cells.
  fallback = A-block decode: the A25 ensemble link probability (ogens_<tag>_prob.parquet) averaged within
             (window, timestamp, my_blocks block), queued if >= 0.5 -- the arithmetic of the ongoing half of
             work/blocks/build_D_inputs.py, which is where the guard's 15 cells came from.

  python make_never_empty_fallback.py   -> candidates/a_block_fallback.parquet
       (window_id, timestamp, link_id, split, queue_pred) for every official ongoing row of both splits.
serve_refit.py imports never_empty() and applies it to the F2 decode of every window. Inputs are <= T only.
"""
# Runs in the reproduction as `pipeline/ongoing.sh` step `fallback`.
import os
import numpy as np, pandas as pd
# Package edit (2026-11-10): the split list comes from TFB_SPLITS (default 'validation private', see README).
SPLITS = tuple(os.environ.get('TFB_SPLITS', 'validation private').split())
NWTAG = lambda s: {'validation': 'realval', 'private': 'realpriv'}.get(s, 'real' + s)  # newworld file tag of a split
PANEL_RX = r'Q2_(.+)_(?:' + '|'.join(SPLITS) + ')_'  # panel name from a window id Q2_<panel>_<split>_<n>
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NW = ROOT / 'build/review_20260924_newworld/work'
OUT = HERE / 'candidates/a_block_fallback.parquet'


def never_empty(window_id, decision, fallback):
    """Per window: keep `decision` if it has any queued cell, else use `fallback`. Returns (decision, empty mask)."""
    decision = np.asarray(decision).astype(int)
    empty = pd.Series(decision).groupby(np.asarray(window_id)).transform('sum').to_numpy() == 0
    return np.where(empty, np.asarray(fallback).astype(int), decision), empty


def a_block_decode():
    """Returns the A-block decode of every official ongoing row of both splits.

    Returns:
      Frame with window_id, timestamp, link_id, split and queue_pred (0/1):
      1 when the mean A-ensemble probability of the link's block, within the
      same window and timestamp, is at least 0.5.
    """
    B = np.load(NW / 'blocks/my_blocks.npy', allow_pickle=True).item(); BID = {p: dict(zip(v['links'], v['bid'])) for p, v in B.items()}
    parts = []
    for tag, split in [(NWTAG(s), s) for s in SPLITS]:
        e = pd.read_parquet(NW / f'feat/ogens_{tag}_prob.parquet'); e['panel'] = e.window_id.str.extract(PANEL_RX)[0]
        e['bid'] = [BID[p][l] for p, l in zip(e.panel, e.link_id)]
        e['bm'] = e.groupby(['window_id', 'timestamp', 'bid']).prob.transform('mean')
        parts.append(e[['window_id', 'timestamp', 'link_id']].assign(split=split, queue_pred=(e.bm >= 0.5).astype('int64')))
    return pd.concat(parts, ignore_index=True)


if __name__ == '__main__':
    out = a_block_decode()
    # 87,000 rows: every official ongoing (window, link, timestamp) cell of
    # validation and private.
    # Package edit (2026-11-10): row and window counts follow window_index.csv instead of the fixed 5-window layout.
    # serve_refit.py joins it one to one with the CTM serving rows and asserts full coverage.
    assert not out[['window_id', 'timestamp', 'link_id']].duplicated().any()
    OUT.parent.mkdir(exist_ok=True); assert not OUT.exists(), OUT
    out.to_parquet(OUT, index=False)
    w = out.groupby(['split', 'window_id']).queue_pred.sum()
    print('A-block fallback rows', len(out), 'windows', len(w), 'queued', int(out.queue_pred.sum()),
          'windows with an empty A-block decode', {s: int((w[s] == 0).sum()) for s in SPLITS})
