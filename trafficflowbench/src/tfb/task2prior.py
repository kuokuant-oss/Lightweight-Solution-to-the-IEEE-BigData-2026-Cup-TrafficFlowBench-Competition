"""The queue prior, with its SOURCE separated from its ROLE.

THE DEFECT THIS REPLACES. Every prior entry point in the repo accumulated one
rate over ``fit_dates | es_dates`` and handed that same array to the fit rows,
the early-stopping rows, the threshold-calibration rows and the held-out rows:

    task2abl.run_arm      d["queue_prior"] = arm.prior(d, set(fit_d)|set(es_d), panels)
    scripts/124, /127     the same call
    scripts/128 stage A   leave-one-day-out, but still over fit|es

Two things are wrong with that, and the second survives a leave-one-day-out fix:

  1. a fit row on day d reads a prior that contains day d's own horizon, which
     is the answer it is being trained to produce;
  2. every fit row reads a prior built partly from the EARLY-STOPPING days, so
     the features the trees are grown on carry ES labels. Dropping "day d
     itself" does not touch that: the other ES days are still in there.

The official aggregation then chooses thresholds on a block whose prior was
also built from its own labels, so the threshold is picked against a target
that has partly been shown to the estimator.

THE RULE IMPLEMENTED HERE. Exactly one set of dates -- ``fit_dates`` -- may
ever contribute a label to a prior. What differs is which row reads what:

    row on a fit date d        the rate over fit_dates MINUS d      (LOO)
    ES / calibration / held-out    the rate over all of fit_dates
    final refit, row on date d     the rate over refit_dates MINUS d
    serving                        the rate over the model's manifest dates

A scored day is never in the source set, so serving reads the plain rate --
which is exactly what the ES / calibration / held-out rows read here. That is
the point: the distribution a threshold is calibrated against is the one
serving produces.

The leave-one-out is applied to the numerator, the denominator, the sparse-cell
fallback AND the global mean the fallback uses. Subtracting only the numerator
counts would leave the day's own denominator in place and bias every sparse
cell toward it.

``PriorSpec`` is written into every run's stamp so a result can be read back
with the boundary it was produced under, rather than with the boundary someone
later assumes it had.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import release as R, score2, task2build as TB

# Role: the training programs of the shipped Task 2 models
# (scripts/145_onset_prior_forward.py, scripts/strongest_20260922_ongoing.py)
# fill their queue_prior column with build + apply; at serving,
# task2infer.queue_prior computes the full-source rate over the model's prior
# dates. Only train-split truth days are read.
# MODE_*: labels written into PriorSpec. build always applies the fit-only
# rule; MODE_LEGACY names the older rule in run records.
MODE_FIT_LOO = "fit_loo"          # corrected: source = fit_dates only
MODE_LEGACY = "legacy_fit_es"     # reproduces the pre-correction runs


@dataclass(frozen=True)
class PriorSpec:
    """Everything needed to rebuild a prior, and to audit what it saw."""
    # source_dates: YYYY-MM-DD train days whose labels may enter the prior.
    # mode: label of the rule (see MODE_*). min_den: fewest eligible
    # observations a (link, time-of-day) cell needs to use its own rate.
    source_dates: tuple[str, ...]
    mode: str = MODE_FIT_LOO
    min_den: int = TB.PRIOR_MIN_DEN

    def to_dict(self) -> dict:
        """Returns the spec as a JSON-ready dict with the sorted dates."""
        s = sorted(self.source_dates)
        return {"mode": self.mode, "min_den": self.min_den, "n_source_dates": len(s),
                "source_first": s[0] if s else None, "source_last": s[-1] if s else None,
                "source_dates": s}


# Caching every panel's every day costs about a gigabyte at eight panels and
# 217 days, which is enough on a 16 GB machine to push LightGBM's
# num_leaves=5000 histogram allocation over the edge -- observed as EXIT=139
# (SIGSEGV), not as a Python MemoryError. Only the panel currently being built
# is retained; a fold rebuilds one panel at a time and never needs two.
_COUNTS: dict[str, dict[str, tuple]] = {}
CACHE_PANELS = 1


def _day_counts(panel: str, dates: set[str]):
    """Per-day (queue, eligible) count matrices, float32 and cached per panel.

    Two memory rules, both learned the hard way -- the first version held every
    day's counts AND every day's leave-one-out rate in float64 for all eight
    panels at once, about 2.9 GB, and was OOM-killed mid-run with no traceback:

      * counts are float32. They are small non-negative integers; the sums that
        matter are accumulated in float64 by _rate.
      * they are cached per (panel, date) across folds. Which days are IN the
        source set changes per fold; the counts for a given day do not.
    """
    # Returns {date: (Q, E)} for the requested train days of `panel`: float32
    # (link x stamp) arrays of queued and eligible flags, links in milepost
    # order. Raises RuntimeError when none of `dates` has a truth file.
    cut = score2.v_cut(panel)
    from .task2pool import date_of
    while len(_COUNTS) >= CACHE_PANELS and panel not in _COUNTS:
        _COUNTS.pop(next(iter(_COUNTS)))
    have = _COUNTS.setdefault(panel, {})
    per = {}
    for path in R.truth_files(panel, "train"):
        d = date_of(path)
        if d not in dates:
            continue
        if d not in have:
            _, _, _, _, _, E, Q, _ = TB.day_arrays(panel, path, cut)
            have[d] = (Q.astype(np.float32), E.astype(np.float32))
        per[d] = have[d]
    if not per:
        raise RuntimeError(f"{panel}: no source day for the prior over {len(dates)} dates")
    return per


def clear_cache() -> None:
    """Drop the per-day count cache. Callers that interleave prior builds with
    a large learner should call this between folds; see task2abl.run_arm."""
    _COUNTS.clear()


def _rate(num: np.ndarray, den: np.ndarray, min_den: int) -> np.ndarray:
    """Returns num/den per cell, or the overall rate where den < min_den."""
    g = float(num.sum() / max(den.sum(), 1.0))     # global mean also excludes the held day
    return np.where(den >= min_den, num / np.maximum(den, 1.0), g)


@dataclass
class PanelPrior:
    """The full-source rate, plus the counts needed to form any day's LOO rate.

    The LOO arrays are built ON DEMAND rather than all at once: ``apply``
    consumes exactly one of them at a time, so materialising every source day's
    rate for every panel only raised the peak. ``per_day`` remains available as
    a mapping for callers that want one explicitly.
    """
    # full: rate over all source days. tot_q, tot_n: summed queued and
    # eligible counts. counts: per-day (Q, E) arrays. _buf, _den: scratch
    # buffers reused by loo().
    full: np.ndarray
    tot_q: np.ndarray
    tot_n: np.ndarray
    counts: dict[str, tuple]
    min_den: int
    _buf: np.ndarray | None = None
    _den: np.ndarray | None = None

    def loo(self, date: str) -> np.ndarray:
        """The rate over the source set with ``date`` removed; the full rate if
        that date is not a source day (ES, calibration, held-out, serving).

        Written into a reusable scratch buffer. Minting a fresh (links x 288)
        float64 array per (panel, date) churned well over a gigabyte per fold
        through the allocator, interleaved with LightGBM's num_leaves=5000
        arena; the process died with SIGSEGV rather than a MemoryError. Only
        the two subtraction temporaries are reused; np.where still returns a
        fresh array, so the RESULT is independent of the next loo() call.
        """
        c = self.counts.get(str(date))
        if c is None:
            return self.full
        if self._buf is None:
            self._buf = np.empty_like(self.tot_q)
            self._den = np.empty_like(self.tot_n)
        np.subtract(self.tot_q, c[0], out=self._buf)
        np.subtract(self.tot_n, c[1], out=self._den)
        g = float(self._buf.sum() / max(self._den.sum(), 1.0))
        return np.where(self._den >= self.min_den,
                        self._buf / np.maximum(self._den, 1.0), g)

    @property
    def per_day(self) -> dict[str, np.ndarray]:
        """Returns {date: leave-one-out rate} for every source day."""
        return {d: self.loo(d) for d in self.counts}


def build(panels: list[str], spec: PriorSpec) -> dict[str, PanelPrior]:
    """One PanelPrior per panel: the full-source rate plus per-day LOO on demand."""
    src = set(spec.source_dates)
    out = {}
    for panel in panels:
        per = _day_counts(panel, src)
        tot_q = np.zeros_like(next(iter(per.values()))[0], dtype=np.float64)
        tot_n = np.zeros_like(tot_q)
        for q, n in per.values():
            tot_q += q
            tot_n += n
        out[panel] = PanelPrior(full=_rate(tot_q, tot_n, spec.min_den),
                                tot_q=tot_q, tot_n=tot_n, counts=per,
                                min_den=spec.min_den)
    return out


def apply(d: pd.DataFrame, priors: dict[str, PanelPrior]) -> np.ndarray:
    """Role-aware lookup. A row whose date is a source day gets that day's LOO
    rate; every other row -- ES, calibration, held-out, and serving -- gets the
    full-source rate, which contains none of its own labels."""
    # `d` needs columns panel, date, _link_idx (row on the panel's
    # milepost-ordered link axis) and _tod_idx (time-of-day slot, 0..287).
    # Returns a float32 array aligned with the rows of `d`.
    out = np.empty(len(d), dtype=np.float32)
    li = d._link_idx.to_numpy()
    ti = d._tod_idx.to_numpy()
    da = d.date.to_numpy()
    for panel, idx in d.groupby("panel", sort=False).indices.items():
        pp = priors[panel]
        for date in np.unique(da[idx]):
            sel = idx[da[idx] == date]
            p = pp.loo(str(date))          # full rate when the date is not a source day
            out[sel] = p[li[sel], ti[sel]]
    return out


def serving_prior(panel: str, spec: PriorSpec) -> np.ndarray:
    """The array serving must use: the full-source rate, no day removed, because
    a scored day is not in the source set."""
    return build([panel], spec)[panel].full
