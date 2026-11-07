"""Task 2 window pool -- official-shape window manifests for validation_v3.

Every selector rule here is read out of ``config/synthetic_release_v1.json``
(task2) rather than restated, and the shape is the one the release actually
ships. Measured on the 80 official ``task2/<panel>/train/window_index.csv``
windows (scripts/65 re-checks all three):

  * a panel carries exactly 5 ``queue_onset`` + 5 ``queue_ongoing`` per split;
  * the 360-minute spacing holds across BOTH conditions inside one
    (panel, date) -- every same-date gap in the official 80 is >= 360 min,
    cross-condition ones included (min observed 360);
  * a condition's quota is filled over as many dates as it needs, so the two
    conditions may sit on disjoint date sets -- D7_I10_W fills onset on
    2030-06-02..06-10 and ongoing on 2030-06-12..06-20.

The selector's own tie-breaking is not released, so it is modelled and then
calibrated against what the 80 released windows measurably are. Two facts
decided the model (both re-measured by scripts/65):

  * a window's difficulty is set by WHERE IN THE DAY it sits, not by the day.
    An onset window may only start in the six steps before a queue first
    appears, so the earliest one of an event sees a single queue cell in its
    horizon and the latest sees the whole formation. Taking candidates
    chronologically reproduces the official difficulty almost exactly --
    official onset n_true mean 3.10 / median 3 against 3.27 / 3 for the
    chronological chain, and ongoing 158.3 / 88 against 156.8 / 84. Sampling
    candidates uniformly inside a day does not: it lands at onset n_true
    median 12, windows the official selector never reaches.
  * the official walks dates chronologically and stops when both quotas are
    full: 5.88 dates per panel on average, four to seven for most panels and
    ten for the sparse D7_I10_W. Its first pick of a day is the earliest
    candidate in 37/47 cases, the misses being dates where one condition was
    already full and it reached past it. Filling both quotas together in date
    order is what leaves ongoing windows deep in a day's chain, where the
    queues are large; a sampler that fills the conditions independently takes
    early-morning ongoing windows instead and lands 25% light on n_true.

What varies between replicates is therefore WHICH DATES a set is built from,
which is the variation the official set itself has -- it used 15 of the 153
dates its split offered.

One released window (D12_I5_S 2030-06-06, origin 175) carries no queue in its
horizon at all, which ``requires_queue_in_horizon`` forbids and the enumeration
here rejects. 79 of the 80 reproduce exactly, condition, coverage,
persistence IoU and n_true included.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import release as R, score2

HIST, HOR = 12, 6
STEP_MIN = 5


@dataclass(frozen=True)
class Rules:
    """The task2 selector rules, as released."""

    onset_per_panel: int
    ongoing_per_panel: int
    requires_queue_in_horizon: bool
    min_history_coverage: float
    max_persistence_iou: float
    spacing_minutes: int

    @property
    def spacing_steps(self) -> int:
        return self.spacing_minutes // STEP_MIN

    def quota(self) -> dict[str, int]:
        return {"queue_onset": self.onset_per_panel, "queue_ongoing": self.ongoing_per_panel}

    def as_series(self) -> pd.Series:
        return pd.Series({
            "onset_per_panel": self.onset_per_panel,
            "ongoing_per_panel": self.ongoing_per_panel,
            "requires_queue_in_horizon": self.requires_queue_in_horizon,
            "min_history_coverage": self.min_history_coverage,
            "max_persistence_iou": self.max_persistence_iou,
            "spacing_minutes": self.spacing_minutes,
            "hist_steps": HIST,
            "horizon_steps": HOR,
        })


def rules() -> Rules:
    cfg = json.loads((R.OFFICIAL / "config" / "synthetic_release_v1.json").read_text(encoding="utf-8"))["task2"]
    return Rules(
        onset_per_panel=int(cfg["queue_onset_windows_per_panel_split"]),
        ongoing_per_panel=int(cfg["queue_ongoing_windows_per_panel_split"]),
        requires_queue_in_horizon=bool(cfg["requires_queue_in_horizon"]),
        min_history_coverage=float(cfg["minimum_history_coverage"]),
        max_persistence_iou=float(cfg["maximum_persistence_iou"]),
        spacing_minutes=int(cfg["minimum_spacing_minutes"]),
    )


def panels() -> list[str]:
    return [p for p in R.panels() if (R.RELEASE / "task2" / p).is_dir()]


def window_id(panel: str, date: str, origin: int) -> str:
    """The id scripts/50 writes, so a manifest joins straight onto m13/m14 rows."""
    return f"{panel}|{date}|{int(origin)}"


def day_queue_matrices(panel: str, path, cut: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """(link x time) queue / eligibility for one day -- scripts/36.day_matrices."""
    f = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
    f["link_id"] = f.link_id.astype(str)
    links = sorted(f.link_id.unique())
    li = {k: i for i, k in enumerate(links)}
    times = np.sort(f.timestamp.unique())
    ti = {k: i for i, k in enumerate(times)}
    n = (len(links), len(times))
    r = f.link_id.map(li).to_numpy()
    c = f.timestamp.map(ti).to_numpy()
    E = np.zeros(n, bool)
    E[r, c] = f.is_score_eligible.astype(bool).to_numpy()
    V = np.full(n, np.nan)
    V[r, c] = pd.to_numeric(f.speed_kmh, errors="coerce").to_numpy()
    cutv = np.array([cut.get(k, 63.0) for k in links])[:, None]
    return (V <= cutv) & E, E


def date_of(path) -> str:
    return path.name[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-")


def enumerate_candidates(dates: set[str], rule: Rules | None = None,
                         panel_list: list[str] | None = None,
                         progress: bool = False) -> pd.DataFrame:
    """Every window on ``dates`` that passes the official per-window rules.

    Spacing is NOT applied here: it constrains a chosen set, not a single
    window, so it belongs to ``sample_manifest``. The per-window columns are
    the ones scripts/36 emits, which lets scripts/65 diff this enumeration
    against the frozen m7 candidate table cell by cell.
    """
    rule = rule or rules()
    out = []
    for panel in (panel_list or panels()):
        cut = score2.v_cut(panel)
        for path in R.truth_files(panel, "train"):
            date = date_of(path)
            if date not in dates:
                continue
            Q, E = day_queue_matrices(panel, path, cut)
            nT = Q.shape[1]
            for t in range(HIST, nT - HOR):
                hs, he = t - HIST, t
                fs, fe = t + 1, t + 1 + HOR
                cov = float(E[:, hs:he].mean())
                if cov < rule.min_history_coverage:
                    continue
                hz_e = E[:, fs:fe]
                truth = Q[:, fs:fe][hz_e]
                if rule.requires_queue_in_horizon and truth.sum() == 0:
                    continue
                hist_q = bool(Q[:, hs:he].any())
                pred = np.repeat(Q[:, t - 1:t], HOR, axis=1)[hz_e]
                piou = score2.iou(pred, truth)
                if piou > rule.max_persistence_iou:
                    continue
                out.append({"panel": panel, "date": date, "origin": t,
                            "condition": "queue_ongoing" if hist_q else "queue_onset",
                            "history_coverage": cov, "persistence_iou": piou,
                            "n_true": int(truth.sum()), "n_scored": int(hz_e.sum())})
        if progress:
            print(f"  {panel}: {sum(1 for r in out if r['panel'] == panel):,} candidates", flush=True)
    d = pd.DataFrame(out)
    return d.sort_values(["panel", "date", "origin"]).reset_index(drop=True)


def official_windows(split: str = "train") -> pd.DataFrame:
    """The released windows, re-measured on the truth days with our own rules.

    ``forecast_origin`` is the exclusive upper bound of the 12-step history, so
    the origin column index is the grid position of ``forecast_origin`` itself:
    for Q2_D7_I10_E_train_001 the released window_history runs 08:05..09:00,
    forecast_origin is 09:05 and the horizon 09:10..09:35, which is exactly
    ``history = times[t-12:t]``, ``horizon = times[t+1:t+7]`` at t = 09:05.
    Verified by the condition column agreeing on all 80 and history_coverage
    matching to 5e-05 (the release rounds it to four places).
    """
    idx = score2.window_index(split)
    idx = idx.assign(origin_ts=pd.to_datetime(idx.forecast_origin, utc=True))
    out = []
    for panel, g in idx.groupby("panel"):
        cut = score2.v_cut(panel)
        by_date = {date_of(p): p for p in R.truth_files(panel, "train")}
        for date, gd in g.groupby("date"):
            path = by_date.get(date)
            if path is None:
                continue
            Q, E = day_queue_matrices(panel, path, cut)
            times = np.sort(pd.read_parquet(path, columns=["timestamp"]).timestamp.unique())
            tmap = {(pd.Timestamp(t, tz="UTC") if pd.Timestamp(t).tz is None else pd.Timestamp(t)): i
                    for i, t in enumerate(times)}
            for w in gd.itertuples(index=False):
                t = tmap.get(w.origin_ts)
                if t is None:
                    continue
                hz = E[:, t + 1:t + 1 + HOR]
                truth = Q[:, t + 1:t + 1 + HOR][hz]
                pred = np.repeat(Q[:, t - 1:t], HOR, axis=1)[hz]
                out.append({"window_id": w.window_id, "panel": panel, "date": date, "origin": t,
                            "condition": w.condition,
                            "history_coverage": float(E[:, t - HIST:t].mean()),
                            "persistence_iou": score2.iou(pred, truth),
                            "n_true": int(truth.sum()), "n_scored": int(hz.sum())})
    return pd.DataFrame(out)


def difficulty(win: pd.DataFrame) -> pd.DataFrame:
    """The window-difficulty signature a set has to match to stand in for the official one."""
    g = win.groupby("condition")
    return pd.DataFrame({
        "n_windows": g.size(),
        "n_true_mean": g.n_true.mean(), "n_true_median": g.n_true.median(),
        "n_scored_mean": g.n_scored.mean(),
        "persistence_iou_mean": g.persistence_iou.mean(),
        "history_coverage_mean": g.history_coverage.mean(),
    })


def spacing_ok(origins: list[int], origin: int, spacing_steps: int) -> bool:
    return all(abs(origin - o) >= spacing_steps for o in origins)


def empty_horizon_candidates(dates: set[str], rule: Rules | None = None,
                             k_per_cell: int = 2, seed: int = 0) -> pd.DataFrame:
    """Onset windows whose horizon carries no queue at all, capped per (panel, date).

    The released set holds exactly one of these -- Q2_D12_I5_S_train_005,
    2030-06-06, origin 175 -- and it breaks two of the rules the same config
    states: ``requires_queue_in_horizon``, and, because an empty prediction
    against an empty truth takes the empty-union IoU of 1.0,
    ``maximum_persistence_iou`` as well. It is the only window of the 80 above
    that ceiling.

    That free 1.0 is not small: it alone lifts the official persistence
    reference from 0.294878 to 0.307378, +0.0125 of S_queue. Whether it is one
    bad record or something the generator does at some rate is not knowable
    from one split, so validation_v3 does not simply drop it -- it keeps a pool
    here and a replicate mode that injects such a window at the observed 1-in-80
    rate, so a candidate's delta can be read with and without them.
    """
    import dataclasses
    rule = rule or rules()
    relaxed = dataclasses.replace(rule, requires_queue_in_horizon=False, max_persistence_iou=1.01)
    cand = enumerate_candidates(dates, relaxed)
    cand = cand[(cand.n_true == 0) & (cand.condition == "queue_onset")]
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in cand.groupby(["panel", "date"], sort=True):
        keep.append(g if len(g) <= k_per_cell
                    else g.sample(n=k_per_cell, random_state=int(rng.integers(0, 2 ** 31 - 1))))
    d = pd.concat(keep, ignore_index=True)
    d["window_id"] = [window_id(p, dt, o) for p, dt, o in zip(d.panel, d.date, d.origin)]
    d["family_id"] = d.panel.map(R.families())
    return d.sort_values(["panel", "date", "origin"]).reset_index(drop=True)


def inject_empty_horizon(man: pd.DataFrame, pool: pd.DataFrame, seed: int, n: int = 1,
                         rule: Rules | None = None) -> pd.DataFrame:
    """Swap ``n`` of a manifest's onset windows for empty-horizon ones.

    The replacement is drawn from the same (panel, date) the onset window it
    replaces sat on, and has to clear spacing against everything else kept on
    that date, so shape stays 5 + 5 and no other rule is broken. That is the
    shape the released exception has: it occupies an onset slot on a date its
    panel was already using.
    """
    rule = rule or rules()
    if n <= 0 or pool.empty:
        return man
    rng = np.random.default_rng(seed)
    out = man.copy()
    onset = out[out.condition == "queue_onset"]
    swapped = 0
    for idx in rng.permutation(onset.index):
        if swapped >= n:
            break
        row = out.loc[idx]
        same = pool[(pool.panel == row.panel) & (pool.date == row.date)]
        if same.empty:
            continue
        others = out[(out.panel == row.panel) & (out.date == row.date) & (out.index != idx)]
        legal = [c for c in same.itertuples(index=False)
                 if spacing_ok(list(others.origin.astype(int)), int(c.origin), rule.spacing_steps)]
        if not legal:
            continue
        pick = legal[int(rng.integers(0, len(legal)))]
        for col in ("origin", "n_true", "n_scored", "persistence_iou", "history_coverage",
                    "window_id"):
            out.at[idx, col] = getattr(pick, col)
        swapped += 1
    return out.sort_values(["panel", "date", "origin"]).reset_index(drop=True)


def chain_windows(cand: pd.DataFrame, rule: Rules | None = None) -> pd.DataFrame:
    """The quota-free chronological chain per (panel, date) -- scripts/36's selector.

    This is what produced ``m7_task2_windows/spaced.parquet``, so it is the set
    the frozen model was trained and scored on. scripts/63 checks the two agree
    before using it as the calib set, which keeps control's thresholds exactly
    the frozen ones.
    """
    rule = rule or rules()
    keep = []
    for _, g in cand.groupby(["panel", "date"], sort=True):
        last = -10 ** 9
        for idx, origin in g.sort_values("origin").origin.items():
            if origin - last >= rule.spacing_steps:
                keep.append(idx)
                last = origin
    d = cand.loc[sorted(keep)].reset_index(drop=True)
    d["window_id"] = [window_id(p, dt, o) for p, dt, o in zip(d.panel, d.date, d.origin)]
    d["family_id"] = d.panel.map(R.families())
    return d


MODES = ("chain_reach", "scan")


def sample_manifest(cand: pd.DataFrame, seed: int, rule: Rules | None = None,
                    mode: str = "chain_reach") -> pd.DataFrame:
    """One official-shaped window set: 5 onset + 5 ongoing per panel.

    Per panel the dates are walked CHRONOLOGICALLY from a seeded random start
    and the walk stops as soon as both quotas are full. That is the official
    behaviour -- it consumed 5.88 dates per panel on average, four to seven for
    most and ten for the sparse D7_I10_W -- and the random start is what makes
    one replicate differ from the next. Going date-by-date in time order is
    what keeps the two quotas filling together, which in turn keeps ongoing
    windows on the later, heavier part of a day's chain.

    ``mode`` selects how a single day is read:

      chain_reach  the day's condition-agnostic chain first (earliest legal,
                   then every next one 360 min on), keeping those whose
                   condition is still short; then, only for a condition no
                   chain ever supplied, a second walk reaches past the chain
                   for it. This is the primary variant: the official set shows
                   both halves -- ongoing windows sitting deep in a day's chain,
                   and D7_I10_W reaching past a chain of pure onset candidates
                   to find its five ongoing windows.
      scan         earliest-first among conditions still short, no chain pass.
                   Kept as a robustness variant; it brackets chain_reach from
                   the lighter side.

    Difficulty against the released 80 (200 replicates, measured on the
    official's own June dates by scripts/63): onset n_true mean 3.34 against
    3.10, ongoing persistence IoU 0.587 against 0.590, persistence S_queue
    0.2934 with the official 0.3074 inside the replicate 95% interval. The one
    statistic that stays off is ongoing n_true, 136 against 158: the official
    reaches somewhat heavier ongoing windows than any earliest-first rule does.
    Both control and candidate are scored on the same windows, so this biases
    the absolute level, not the paired delta.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    rule = rule or rules()
    sp = rule.spacing_steps
    rng = np.random.default_rng(seed)
    quota = rule.quota()
    rows = []
    for panel in sorted(cand.panel.unique()):
        need = dict(quota)
        cp = cand[cand.panel == panel].sort_values("origin")
        by_date = {d: g for d, g in cp.groupby("date", sort=True)}
        avail = sorted(by_date)
        start = int(rng.integers(0, len(avail)))
        order = [avail[(start + k) % len(avail)] for k in range(len(avail))]
        taken: dict[str, list[int]] = {d: [] for d in avail}

        def keep(date: str, row) -> None:
            taken[date].append(int(row.origin))
            need[row.condition] -= 1
            rows.append({"panel": panel, "date": date, "origin": int(row.origin),
                         "condition": row.condition,
                         "history_coverage": float(row.history_coverage),
                         "persistence_iou": float(row.persistence_iou),
                         "n_true": int(row.n_true), "n_scored": int(row.n_scored)})

        if mode == "chain_reach":
            for date in order:
                if not any(need.values()):
                    break
                last = -10 ** 9
                for row in by_date[date].itertuples(index=False):
                    if row.origin - last < sp:
                        continue
                    last = row.origin
                    if need.get(row.condition, 0) > 0:
                        keep(date, row)
        for date in order:                      # scan mode, and chain_reach's second walk
            if not any(need.values()):
                break
            g = by_date[date]
            while any(need.values()):
                legal = g[[need.get(c, 0) > 0 for c in g.condition]
                          & np.array([spacing_ok(taken[date], int(o), sp) for o in g.origin])]
                if legal.empty:
                    break
                keep(date, legal.iloc[0])
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["window_id"] = [window_id(p, dt, o) for p, dt, o in zip(d.panel, d.date, d.origin)]
    d["family_id"] = d.panel.map(R.families())
    return d.sort_values(["panel", "date", "origin"]).reset_index(drop=True)


def check_manifest(man: pd.DataFrame, rule: Rules | None = None,
                   allow_empty_horizon: int = 0) -> list[str]:
    """Official-shape and legality problems in a manifest. Empty list = clean.

    ``allow_empty_horizon`` tolerates that many windows with no queue in the
    horizon -- and, with them, the persistence IoU of 1.0 they necessarily
    carry. Only the empty-horizon replicate mode sets it; everywhere else the
    official rules are enforced strictly.
    """
    rule = rule or rules()
    bad: list[str] = []
    n_empty = int((man.n_true <= 0).sum())
    if n_empty > allow_empty_horizon:
        bad.append(f"{n_empty} windows with no queue in the horizon, "
                   f"at most {allow_empty_horizon} allowed")
    quota = rule.quota()
    for panel, g in man.groupby("panel"):
        for cond, want in quota.items():
            got = int((g.condition == cond).sum())
            if got != want:
                bad.append(f"{panel}/{cond}: {got} windows, official shape is {want}")
        for date, gd in g.groupby("date"):
            o = np.sort(gd.origin.to_numpy())
            if len(o) > 1 and int(np.diff(o).min()) < rule.spacing_steps:
                bad.append(f"{panel}/{date}: spacing {int(np.diff(o).min()) * STEP_MIN} min "
                           f"< {rule.spacing_minutes}")
    if (man.history_coverage < rule.min_history_coverage).any():
        bad.append("history_coverage below the official minimum")
    over = man[(man.persistence_iou > rule.max_persistence_iou) & (man.n_true > 0)]
    if len(over):
        bad.append(f"{len(over)} windows with persistence_iou above the official maximum")
    if man.window_id.duplicated().any():
        bad.append("duplicate window_id")
    return bad
