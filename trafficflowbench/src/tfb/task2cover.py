"""Training-window coverage: fill the (panel, condition) cells the chain drops.

``m7_task2_windows/spaced.parquet`` is a condition-AGNOSTIC earliest-first
chain with 6-hour spacing, run per (panel, date). Every model so far trains on
it. Measured on the release (2026-09-21):

    panel x condition       candidates   chain keeps   dates covered
    D7_I10_W   ongoing           2,552             2          2 / 98
    D7_I210_E  ongoing          21,957           351        211 / 217
    ...                   ongoing cells keep ~1%, onset cells ~13%

D7_I10_W x ongoing is not a corridor with two congested windows in seven
months; it has 2,552 candidates and the chain lands on onset windows almost
every time. The official scorer weights each (panel, condition) cell equally
-- one of sixteen -- so a cell the training set cannot see is a cell the model
cannot learn, and the OOF ranking of every candidate is decided partly by two
windows.

This module ADDS windows to that chain rather than replacing it. The chain
stays in the pool, so the comparison is "chain" against "chain + fill", not
two unrelated samplers, and the already-measured behaviour is not discarded.

Selection, per (panel, condition) cell that is short of ``target``:

  * dates are walked round-robin, fewest-already-selected first, so the fill
    spreads over the calendar instead of copying one heavy event;
  * at most ``max_per_date`` additions per (cell, date);
  * within a date the next origin is the candidate FARTHEST from every origin
    already chosen for that (panel, date) -- across both conditions, because
    two windows 10 minutes apart share 11 of their 12 history steps and are
    not two observations. That maximin walk is what spreads the additions over
    an event's formation, propagation and decay rather than clustering them at
    its peak;
  * a hard ``min_gap`` floor refuses any origin closer than that to an
    already-chosen one.

Nothing here touches labels: an added window is an ordinary candidate window
with its own history and its own horizon, read from the same released day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Role: no program of the reproduction pipeline imports this module; it is the
# record of the training-window fill rule.
# TARGET: windows wanted per (panel, condition) cell. MIN_GAP: smallest gap in
# 5-minute steps between two origins on one (panel, date). MAX_PER_DATE: most
# additions per (cell, date).
TARGET = 350
MIN_GAP = 12            # steps, 60 minutes
MAX_PER_DATE = 4


def fill(cand: pd.DataFrame, base: pd.DataFrame, target: int = TARGET,
         min_gap: int = MIN_GAP, max_per_date: int = MAX_PER_DATE,
         seed: int = 20260921) -> pd.DataFrame:
    """``base`` plus enough extra candidate windows to reach ``target`` per cell."""
    # `cand`: every candidate window (task2pool.enumerate_candidates columns).
    # `base`: the existing chain windows. `seed` fixes the random order of
    # dates before the stable fewest-selected-first sort, so ties are broken
    # reproducibly.
    # Returns the base rows (source="chain") plus the added rows
    # (source="fill"), sorted by panel, date and origin. The selection reads
    # only dates and origins.
    rng = np.random.default_rng(seed)
    cols = ["panel", "date", "origin", "condition", "history_coverage",
            "persistence_iou", "n_true", "n_scored"]
    base = base[cols].copy()
    base["source"] = "chain"

    # origins already committed per (panel, date), both conditions together
    taken: dict[tuple[str, str], list[int]] = {}
    for p, dt, o in zip(base.panel, base.date, base.origin):
        taken.setdefault((p, str(dt)), []).append(int(o))

    have = base.groupby(["panel", "condition"]).size()
    pool = cand[cols].copy()
    key = pd.MultiIndex.from_arrays([pool.panel, pool.date, pool.origin])
    # Candidates already in the chain are removed from the pool.
    bkey = set(zip(base.panel, base.date.astype(str), base.origin.astype(int)))
    pool = pool[[k not in bkey for k in zip(pool.panel, pool.date.astype(str),
                                            pool.origin.astype(int))]]

    added = []
    for (panel, cond), grp in pool.groupby(["panel", "condition"], sort=True):
        need = target - int(have.get((panel, cond), 0))
        if need <= 0:
            continue
        by_date = {str(dt): g.sort_values("origin") for dt, g in grp.groupby("date", sort=True)}
        per_date = {d: 0 for d in by_date}
        order = sorted(by_date)
        rng.shuffle(order)
        # fewest-already-selected first: dates the chain never used come first
        order.sort(key=lambda d: len(taken.get((panel, d), [])))
        # Round robin over dates: each pass adds at most one window per date,
        # until the cell reaches `target` or no date can add another.
        progress = True
        while need > 0 and progress:
            progress = False
            for d in order:
                if need <= 0:
                    break
                if per_date[d] >= max_per_date:
                    continue
                g = by_date[d]
                if g.empty:
                    continue
                chosen = np.asarray(taken.get((panel, d), []), dtype=float)
                o = g.origin.to_numpy(dtype=float)
                if chosen.size:
                    gap = np.abs(o[:, None] - chosen[None, :]).min(axis=1)
                else:
                    gap = np.full(o.shape, np.inf)
                ok = gap >= min_gap
                if not ok.any():
                    by_date[d] = g.iloc[:0]
                    continue
                pick = int(np.argmax(np.where(ok, gap, -np.inf)))   # maximin
                row = g.iloc[pick]
                added.append(row)
                taken.setdefault((panel, d), []).append(int(row.origin))
                by_date[d] = g.drop(g.index[pick])
                per_date[d] += 1
                need -= 1
                progress = True

    if added:
        extra = pd.DataFrame(added).reset_index(drop=True)
        extra["source"] = "fill"
        out = pd.concat([base, extra], ignore_index=True)
    else:
        out = base
    return out.sort_values(["panel", "date", "origin"]).reset_index(drop=True)


def report(win: pd.DataFrame) -> pd.DataFrame:
    """Counts windows per (panel, condition).

    Returns:
        DataFrame with n (windows), dates (distinct dates) and, when ``win``
        has a ``source`` column, the chain and fill counts.
    """
    g = win.groupby(["panel", "condition"])
    out = g.agg(n=("origin", "size"), dates=("date", "nunique")).reset_index()
    if "source" in win.columns:
        out["chain"] = g.source.apply(lambda s: int((s == "chain").sum())).to_numpy()
        out["fill"] = g.source.apply(lambda s: int((s == "fill").sum())).to_numpy()
    return out
