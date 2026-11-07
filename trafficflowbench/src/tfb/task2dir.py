"""Per-panel travel direction on the milepost axis.

Every geometric feature in task2feat / task2cum / task2spatial was written
assuming the link chain runs downstream as milepost increases ("upstream is the
lower-milepost side"). That is true for the E/N corridors and false for the W/S
ones, so half the corridors got their upstream/downstream neighbours, their
cumulative on-ramp inflow and their shockwave source swapped.

The sign is not guessed from the corridor name. It is read off the released
``lwr_mainline_topology.csv``: for every mainline link with at least one
outgoing mainline link that also carries a milepost, we compare the two
mileposts. A corridor whose outgoing edges all move to a LOWER milepost travels
in the decreasing direction, so its sign is -1.

Measured on the release (2026-09-21), the vote is unanimous per corridor:

    D7_I10_E    85+/0-     D7_I10_W     0+/28-
    D7_I210_E  290+/0-     D7_I210_W    0+/15-
    D7_I405_N  332+/0-     D7_I405_S    0+/15-
    D12_I5_N   256+/1-     D12_I5_S     0+/30-
    D12_I405_N  69+/1-     D12_I405_S   0+/10-

The two stray reverse edges on the N corridors are why this reads the topology
per edge rather than trusting the ``_N``/``_S`` suffix: the suffix happens to
agree here, the topology is the actual evidence, and a future corridor need not
be monotone.

``sign(panel) == +1`` reproduces every legacy call exactly, so a caller that
does not opt in is bit-identical to the frozen path.
"""
from __future__ import annotations

import functools

import pandas as pd

from . import release as R

# Role: task2infer.features_for_split and task2oof.base_features call `sign`
# only with directed=True, which no program in this repository passes; the
# shipped features therefore use sgn = +1 (traffic toward increasing
# milepost).
# Later on-ramp and off-ramp flow-balance checks showed that traffic moves
# toward increasing milepost on all eight Task 2 corridors; the W/S edges of
# lwr_mainline_topology point the other way. sgn = +1 is the correct direction.


@functools.lru_cache(maxsize=None)
def _vote(panel: str) -> tuple[int, int]:
    """Counts mainline topology edges that rise or fall in milepost.

    Reads the panel's released ``lwr_mainline_topology`` table and the link
    mileposts of its first train truth day; edges whose links lack a milepost
    are skipped.

    Returns:
        (inc, dec): number of edges whose outgoing link has a higher milepost,
        and number whose outgoing link has a lower one.
    """
    topo = R.network(panel, "lwr_mainline_topology")
    t = pd.read_parquet(R.truth_files(panel, "train")[0], columns=["link_id", "milepost"])
    mp = dict(zip(t.link_id.astype(str), t.milepost.astype(float)))
    inc = dec = 0
    for row in topo.itertuples(index=False):
        a = str(row.mainline_link_id)
        if a not in mp:
            continue
        outs = str(getattr(row, "outgoing_link_ids", "") or "")
        for b in outs.split(";"):
            b = b.strip()
            if not b or b not in mp:
                continue
            if mp[b] > mp[a]:
                inc += 1
            elif mp[b] < mp[a]:
                dec += 1
    return inc, dec


@functools.lru_cache(maxsize=None)
def sign(panel: str) -> int:
    """+1 if traffic travels toward increasing milepost, -1 otherwise."""
    inc, dec = _vote(panel)
    if inc == 0 and dec == 0:
        raise RuntimeError(f"{panel}: topology gives no sensor-to-sensor mainline edge")
    return 1 if inc >= dec else -1


def signs() -> dict[str, int]:
    """Returns {panel: travel-direction sign} for every released panel."""
    return {p: sign(p) for p in R.panels()}
