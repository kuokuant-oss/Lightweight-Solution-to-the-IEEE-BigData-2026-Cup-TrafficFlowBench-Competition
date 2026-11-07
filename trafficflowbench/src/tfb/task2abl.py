"""Shared evaluation harness for retrained Task 2 OOF candidates.

Why this exists: the first five ablations (scripts/73-77) held the frozen
model's 0.45/0.525 thresholds fixed for both arms. That fixes the threshold
NUMBER, not the threshold SELECTION PROCEDURE, and doc/reference/validation.md
requires the latter -- "旋鈕只在 calib 選,結果在 eval 報". Holding an old number
penalises any candidate that moves the probability scale, which is exactly what
the day-type prior (it changes queue_prior and therefore the init_score) and
focal loss (a different objective, a different probability scale) do. Their
frozen-threshold deltas confound calibration drift with model quality.

What a fair single-factor comparison fixes is the data, the procedure around
the model, and the threshold selection rule. The two arms are allowed, and
expected, to land on different threshold NUMBERS. That is not a second factor;
it is the same deployment procedure calibrated to each model.

Split, fixed before any number was looked at
--------------------------------------------
For each held-out fold the remaining dev dates are sorted ascending and cut
into three disjoint, time-ordered blocks:

    fit      first 70%     trees are fit here
    es       next  15%     early stopping only
    thrcal   last  15%     threshold selection only

None of the three reads the held-out fold, and the held-out fold's labels are
never used to pick a threshold.

The queue prior for a fold is accumulated over ``fit + es`` only. That keeps it
clean for BOTH the threshold-calibration block and the held-out fold: neither
contributes its own day to the prior it is scored under. scripts/68 accumulated
over every source day, which was clean for the held-out fold but would have made
the thrcal block optimistic and biased threshold selection.

Consequence, stated so it is not mistaken for a defect: the control's absolute
S_queue under this split is NOT scripts/68's 0.812402. It trains on 70% of the
source dates rather than 85% and its prior comes from 85% of them rather than
100%. Both arms carry that change identically, so the paired delta is unaffected.

Two results per experiment
--------------------------
    diagnostic   both arms on the frozen 0.45/0.525. Describes probability-scale
                 drift. NOT a basis for adopting or rejecting anything.
    deployment   each arm on thresholds its own fold-local thrcal block chose,
                 using the same grid and the same selection function.

Adoption and rejection follow the deployment comparison only.

min_size stays at 1 throughout: adding a second knob would stop this being a
single-factor comparison.
"""
#
# Role in the package: a research record. It is the shared harness of the
# paired Task 2 ablations (a control arm against a candidate arm, both
# retrained out of fold); no program in this package imports it.
#
# Terms: OOF = out-of-fold. S_queue is the Task 2 score: per-window IoU
# averaged window -> (panel, condition) -> panel -> family with equal weights
# (score2.aggregate). "thrcal" is the threshold-calibration date block. The
# quoted rule in the docstring reads "knobs are chosen on calib only; results
# are reported on eval". scripts/NN, doc/... and the ledger are development
# records that are not part of this package.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import release as R, score2, task2eval as TE, task2oof as OOF, task2prior as PR

# Shares of each fold's source dates used for early stopping and for
# threshold selection; the rest (FIT_FRACTION) fits the trees.
ES_FRACTION = 0.15
THRCAL_FRACTION = 0.15
# scripts/93: threshold selection on 13 days moves a RETRAINED candidate's paired
# delta by 0.0072 -- more than the effects being measured -- while moving
# scripts/92's no-op by only 0.0009. Two near-identical models pick near-identical
# thresholds; two genuinely different score surfaces do not, so a no-op floor
# understates this. At 26 days the oblique effect reproduced across two column
# orders (+0.005666 and +0.005912). Any block a threshold is chosen on must hold
# at least this many dates.
MIN_THR_DATES = 26
FIT_FRACTION = 1.0 - ES_FRACTION - THRCAL_FRACTION
# Smallest predicted component the decision rule keeps; 1 keeps every one.
MIN_SIZE = 1
# Candidate thresholds 0.05, 0.075, ..., 0.95 for the per-condition sweep.
THR_GRID = np.round(np.arange(0.05, 0.96, 0.025), 3)
# Config of the frozen first-submission Task 2 model, with its thresholds.
FROZEN_CFG = R.REPO / "build" / "submission_v1" / "frozen" / "task2_model" / "config.csv"


def frozen_thresholds() -> dict[str, float]:
    """Reads the frozen model's per-condition thresholds from FROZEN_CFG."""
    cfg = pd.read_csv(FROZEN_CFG, index_col=0).iloc[:, 0]
    return {"queue_onset": float(cfg["thr_queue_onset"]),
            "queue_ongoing": float(cfg["thr_queue_ongoing"])}


def nested_split(source_dates: list[str]) -> tuple[list[str], list[str], list[str]]:
    """fit / es / thrcal, time-ordered and disjoint. See the module docstring."""
    # Returns three sorted date lists cut at rounded shares of the date
    # count; an empty block raises RuntimeError.
    src = sorted(source_dates)
    n = len(src)
    i_es = int(round(n * FIT_FRACTION))
    i_th = int(round(n * (FIT_FRACTION + ES_FRACTION)))
    fit, es, thr = src[:i_es], src[i_es:i_th], src[i_th:]
    if not fit or not es or not thr:
        raise RuntimeError(f"degenerate nested split: {len(fit)}/{len(es)}/{len(thr)}")
    return fit, es, thr


def default_prior(d: pd.DataFrame, src: set[str], panels: list[str]) -> np.ndarray:
    """Returns the legacy queue prior for every row of d.

    The prior is the plain per-(link, time of day) queue rate over the dates
    in src (task2oof.panel_prior), looked up at each row's cell.
    """
    return OOF.apply_prior(d, {p: OOF.panel_prior(p, src) for p in panels})


def make_fitter(objective=None, feval=None):
    """A fitter with the objective and the stopping metric chosen independently.

    Separating them is what lets an experiment isolate a custom objective from
    the shortened training an objective-matched stopping metric causes. When
    ``feval`` is supplied the built-in metric is switched off, so stopping is
    driven by that metric alone in every arm that uses it.
    """
    # objective replaces PARAMS' objective (and drops its built-in metric);
    # feval is a custom metric passed to lgb.train. Training otherwise matches
    # task2oof.fit_fold: same parameters, "cells" window weights and
    # logit(queue_prior) as init_score.
    def fit(train: pd.DataFrame, valid: pd.DataFrame, feats: list[str]):
        """Trains on train with early stopping on valid; returns (model, best_iter)."""
        import lightgbm as lgb
        P = dict(OOF.PARAMS)
        if objective is not None:
            P.pop("metric", None)
            P["objective"] = objective
        if feval is not None:
            P["metric"] = "None"
        dtrain = lgb.Dataset(train[feats], train.label, weight=OOF.window_weights(train),
                             init_score=OOF.logit(train.queue_prior.to_numpy()))
        dvalid = lgb.Dataset(valid[feats], valid.label, weight=OOF.window_weights(valid),
                             init_score=OOF.logit(valid.queue_prior.to_numpy()),
                             reference=dtrain)
        m = lgb.train(P, dtrain, num_boost_round=OOF.NUM_BOOST_ROUND, valid_sets=[dvalid],
                      feval=feval,
                      callbacks=[lgb.early_stopping(OOF.EARLY_STOPPING, verbose=False),
                                 lgb.log_evaluation(0)])
        return m, int(m.best_iteration or OOF.NUM_BOOST_ROUND)
    return fit


@dataclass
class Arm:
    """One side of a paired ablation.

    ``feats_by_cond`` lets each condition carry its own column set, which is
    what a condition-specific feature table means. When it is None both
    conditions use ``feats``.
    """
    # label: arm name in logs; feats: feature columns; fitter, prior and
    # predictor: the training, prior and prediction functions (the defaults
    # reproduce the control); meta: free-form notes carried with the arm.
    label: str
    feats: list[str]
    feats_by_cond: dict[str, list[str]] | None = None
    fitter: Callable = OOF.fit_fold                  # (train, valid, feats) -> (model, best_iter)
    prior: Callable = default_prior                  # (d, src_dates, panels) -> ndarray
    # An arm trained without init_score, or with a rescaled one, must not have
    # OOF.predict's logit(queue_prior) added back. Default keeps scripts/68.
    predictor: Callable = OOF.predict                # (model, frame, feats) -> prob
    meta: dict = field(default_factory=dict)


def run_arm(d: pd.DataFrame, arm: Arm, fold_of: dict[str, int], dates: list[str],
            panels: list[str], n_folds: int = 5, verbose: bool = True,
            extra_cols: tuple[str, ...] = (), per_fold_features=None,
            fit_mask: np.ndarray | None = None,
            eval_mask: np.ndarray | None = None,
            prior_mode: str = "legacy_fit_es",
            fit_filter: dict | None = None,
            zero_filter: dict | None = None):
    """Out-of-fold probabilities, plus probabilities on each fold's thrcal block.

    The thrcal predictions are what the deployment thresholds are chosen on. They
    come from the same model as the held-out predictions and from dates that
    model never fit, so they are an honest stand-in for deployment calibration.

    ``fit_mask`` / ``eval_mask`` separate WHICH ROWS A MODEL IS FIT ON from WHICH
    ROWS IT IS SCORED ON, which is what makes a training-coverage experiment a
    single-factor one. Both default to None = every row, reproducing every
    caller written before them exactly.

      fit_mask   rows eligible for the fit and early-stopping blocks
      eval_mask  rows eligible for the held-out and threshold-calibration blocks

    Two arms that differ only in fit_mask are scored on the identical windows
    with thresholds chosen on the identical windows, so the delta cannot be an
    artefact of one arm being evaluated on an easier population. The date-block
    split still governs everything: a row is only ever fit on if its DATE is in
    the fold's fit block, whatever the masks say.

    ``prior_mode`` selects the prior's SOURCE (src/tfb/task2prior):

      legacy_fit_es  one plain rate over fit + ES dates handed to every row.
                     What every run before 2026-09-21 used; kept so those
                     results stay reproducible under their own procedure. It
                     leaks two ways -- a fit row reads its own day's horizon,
                     and every fit row reads the ES days.
      fit_loo        the corrected rule. Only fit dates ever contribute a
                     label. A fit row on day d reads the fit-minus-d rate; ES,
                     threshold-calibration and held-out rows read the plain
                     fit-only rate, which is also what serving reads
                     (scripts/130 proves both by label perturbation).

    The resolved PriorSpec per fold is attached as ``.attrs["prior_specs"]`` so
    a score can be read back with the boundary it was produced under.

    ``fit_filter`` / ``zero_filter`` implement a CONDITIONAL head without
    touching the scored cell set. Both are {condition: boolean row mask}.

      fit_filter   rows that condition may be fit and early-stopped on
      zero_filter  rows whose prediction is forced to 0 everywhere it is used

    The organizer has confirmed that in all 120 released onset windows the
    queue appears only at T+30 and never in T+5..T+25
    (discussion/742350), so an onset head can be trained and applied on step 6
    alone. Forcing the other five steps to zero rather than DROPPING them keeps
    every arm scored over the identical cells, which is what makes the
    comparison single-factor -- dropping them would score the conditional head
    on a sixth as many cells and flatter it.
    """
    # Args not covered above: d is the feature table (task2oof.base_features
    # rows with fold and date columns); fold_of maps date -> held-out fold;
    # dates are all source dates; extra_cols are copied into the output;
    # per_fold_features(k, fit_dates, es_dates) rebuilds fold-dependent
    # features. Returns the OOF frame (prob = held-out probability, 0 where a
    # condition was not fitted) with calib, iters, importance, prior_specs and
    # prior_mode in .attrs.
    keep = ["window_id", "panel", "condition", "date", "fold", "link_id", "step", "label"]
    keep = keep + [c for c in extra_cols if c not in keep]
    n_rows = len(d)
    fit_ok = np.ones(n_rows, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    ev_ok = np.ones(n_rows, bool) if eval_mask is None else np.asarray(eval_mask, bool)
    if len(fit_ok) != n_rows or len(ev_ok) != n_rows:
        raise ValueError("fit_mask / eval_mask must be one flag per row of d")
    oof = d[keep].copy()
    oof["prob"] = np.nan
    # queue_prior is rebuilt per fold, so it is captured per fold rather than
    # taken from whatever the last fold happened to leave in d.
    if "queue_prior" in extra_cols:
        oof["queue_prior"] = np.nan
    cal_parts, iters, imp, prior_specs = [], [], [], []
    base = d
    for k in range(n_folds):
        held = sorted({dt for dt, f in fold_of.items() if f == k})
        src = sorted(set(dates) - set(held))
        fit_d, es_d, th_d = nested_split(src)
        # A feature that is itself estimated from the fold's training days has
        # to be rebuilt per fold, or it would carry the held-out fold's data.
        d = base if per_fold_features is None else per_fold_features(k, fit_d, es_d)
        if d is not base:
            # oof rows were laid out from `base` and are written positionally,
            # so a rebuilt frame must enumerate the same cells in the same order.
            same = (len(d) == len(base)
                    and (d.window_id.to_numpy() == base.window_id.to_numpy()).all()
                    and (d.link_id.to_numpy() == base.link_id.to_numpy()).all()
                    and (d.step.to_numpy() == base.step.to_numpy()).all())
            if not same:
                raise RuntimeError("per_fold_features changed the row order")
        if prior_mode == PR.MODE_LEGACY:
            d["queue_prior"] = arm.prior(d, set(fit_d) | set(es_d), panels)
            pspec = PR.PriorSpec(source_dates=tuple(sorted(set(fit_d) | set(es_d))),
                                 mode=PR.MODE_LEGACY)
        elif prior_mode == PR.MODE_FIT_LOO:
            pspec = PR.PriorSpec(source_dates=tuple(sorted(fit_d)), mode=PR.MODE_FIT_LOO)
            _pri = PR.build(panels, pspec)
            d["queue_prior"] = PR.apply(d, _pri)
            # Release the count matrices before LightGBM allocates its
            # histograms. Holding one fold's worth of them across a
            # num_leaves=5000 fit fragmented the heap badly enough to abort the
            # process with SIGSEGV (EXIT=139) rather than a MemoryError, after
            # the prior itself had been verified correct at full scale.
            del _pri
            PR.clear_cache()
            import gc; gc.collect()
        else:
            raise ValueError(f"unknown prior_mode {prior_mode!r}")
        prior_specs.append({"fold": k, "fit": [fit_d[0], fit_d[-1]],
                            "es": [es_d[0], es_d[-1]], "thrcal": [th_d[0], th_d[-1]],
                            "held": [held[0], held[-1]], **pspec.to_dict()})
        dates_arr = d.date.to_numpy()
        is_held = (d.fold.to_numpy() == k) & ev_ok
        is_fit = np.isin(dates_arr, fit_d) & fit_ok
        is_es = np.isin(dates_arr, es_d) & fit_ok
        is_th = np.isin(dates_arr, th_d) & ev_ok
        assert not (is_held & (is_fit | is_es | is_th)).any(), "held-out fold leaked into a block"
        for cond in OOF.CONDITIONS:
            sel = d.condition.to_numpy() == cond
            ff = np.ones(n_rows, bool) if not fit_filter else np.asarray(
                fit_filter.get(cond, np.ones(n_rows, bool)), bool)
            zf = np.zeros(n_rows, bool) if not zero_filter else np.asarray(
                zero_filter.get(cond, np.zeros(n_rows, bool)), bool)
            tr, va, te, ca = (d[is_fit & sel & ff], d[is_es & sel & ff],
                              d[is_held & sel], d[is_th & sel])
            if tr.empty or te.empty or ca.empty:
                continue
            fts = arm.feats if arm.feats_by_cond is None else arm.feats_by_cond[cond]
            m, best = arm.fitter(tr, va, fts)
            ph = arm.predictor(m, te, fts)
            ph = np.where(zf[(is_held & sel).nonzero()[0]], 0.0, ph)
            oof.loc[is_held & sel, "prob"] = ph
            if "queue_prior" in extra_cols:
                oof.loc[is_held & sel, "queue_prior"] = te.queue_prior.to_numpy()
            c = ca[keep].copy()
            pc = arm.predictor(m, ca, fts)
            c["prob"] = np.where(zf[(is_th & sel).nonzero()[0]], 0.0, pc)
            c["cal_fold"] = k
            cal_parts.append(c)
            iters.append({"fold": k, "condition": cond, "best_iter": best})
            g = m.feature_importance("gain")
            imp.append(pd.DataFrame({"fold": k, "condition": cond, "feature": m.feature_name(),
                                     "gain": g / max(g.sum(), 1e-12)}))
        if verbose:
            print(f"  [{arm.label}] fold {k} ({held[0]}..{held[-1]}) "
                  f"fit {len(fit_d)}d / es {len(es_d)}d / thrcal {len(th_d)}d", flush=True)
    if eval_mask is not None:            # rows outside the evaluation set never scored
        oof = oof[ev_ok].reset_index(drop=True)
    oof["prob"] = oof.prob.fillna(0.0)
    oof.attrs["calib"] = pd.concat(cal_parts, ignore_index=True)
    oof.attrs["iters"] = pd.DataFrame(iters)
    oof.attrs["importance"] = pd.concat(imp, ignore_index=True)
    oof.attrs["prior_specs"] = prior_specs
    oof.attrs["prior_mode"] = prior_mode
    return oof


def check_thr_dates(n_dates: int, where: str = "") -> None:
    """Refuse to choose a threshold on too few dates -- see MIN_THR_DATES."""
    if n_dates < MIN_THR_DATES:
        raise RuntimeError(
            f"threshold selection needs >= {MIN_THR_DATES} dates, got {n_dates}"
            f"{' in ' + where if where else ''}. scripts/93 measured that fewer moves "
            f"a retrained candidate's delta by 0.0072, more than the effects measured. "
            f"Enlarge THRCAL_FRACTION rather than lowering this.")


def calibrated_thresholds(arm_oof: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Per-fold thresholds, chosen on that fold's thrcal block only."""
    cal = arm_oof.attrs["calib"]
    rows = []
    for k in range(n_folds):
        g = cal[cal.cal_fold == k]
        if g.empty:
            continue
        check_thr_dates(g.date.nunique(), f"fold {k}")
        thr = TE.choose_thresholds(g, THR_GRID, min_size=MIN_SIZE)
        rows.append({"fold": k, **{f"thr_{c}": thr[c] for c in TE.CONDITIONS}})
    return pd.DataFrame(rows)


def frame_with(arm_oof: pd.DataFrame, thr, per_fold: bool = False) -> pd.DataFrame:
    """Decision frame. ``thr`` is one dict, or a per-fold threshold table."""
    # Returns the per-window IoU frame (task2eval.iou_frame) with date and
    # fold columns added.
    if not per_fold:
        pred = TE.decide(arm_oof, thr, MIN_SIZE)
    else:
        pred = np.zeros(len(arm_oof), dtype=int)
        fa = arm_oof.fold.to_numpy()
        for r in thr.itertuples(index=False):
            m = fa == r.fold
            if not m.any():
                continue
            t = {"queue_onset": r.thr_queue_onset, "queue_ongoing": r.thr_queue_ongoing}
            pred[m] = TE.decide(arm_oof[m], t, MIN_SIZE)
    f = TE.iou_frame(arm_oof, pred)
    m = arm_oof.drop_duplicates("window_id").set_index("window_id")
    f["date"] = f.window_id.map(m.date)
    f["fold"] = f.window_id.map(m.fold)
    return f


def subgroup_deltas(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Returns control, candidate and delta S_queue per subgroup.

    Subgroups are overall, each condition, each panel and each fold.

    Args:
        a: Per-window IoU frame of the control arm (frame_with output).
        b: The same windows for the candidate arm.
    """
    rows = []
    def add(name, fa, fb):
        """Appends one subgroup row; skips an empty subgroup."""
        if fa.empty:
            return
        rows.append({"subgroup": name, "n_windows": fa.window_id.nunique(),
                     "control": TE.s_queue(fa), "candidate": TE.s_queue(fb),
                     "delta": TE.s_queue(fb) - TE.s_queue(fa)})
    add("overall", a, b)
    for c in TE.CONDITIONS:
        add(c, a[a.condition == c], b[b.condition == c])
    for p in sorted(a.panel.unique()):
        add(f"panel::{p}", a[a.panel == p], b[b.panel == p])
    for k in sorted(a.fold.unique()):
        add(f"fold::{k}", a[a.fold == k], b[b.fold == k])
    return pd.DataFrame(rows)


def compare(a_oof: pd.DataFrame, b_oof: pd.DataFrame, seed: int, n_boot: int = 2000,
            n_folds: int = 5) -> dict:
    """Both readings: frozen-threshold diagnostic, and deployment comparison."""
    # a_oof / b_oof: run_arm outputs of the control and candidate arms on the
    # same rows. The bootstrap resamples dates (task2eval.paired_bootstrap).
    fz = frozen_thresholds()
    out = {"frozen_thresholds": fz}
    fa, fb = frame_with(a_oof, fz), frame_with(b_oof, fz)
    out["diagnostic"] = {
        "subgroups": subgroup_deltas(fa, fb),
        "bootstrap": TE.paired_bootstrap(fa, fb, n_boot=n_boot, seed=seed, block="date")}

    ta = calibrated_thresholds(a_oof, n_folds)
    tb = calibrated_thresholds(b_oof, n_folds)
    ca, cb = frame_with(a_oof, ta, per_fold=True), frame_with(b_oof, tb, per_fold=True)
    out["deployment"] = {
        "thresholds_control": ta, "thresholds_candidate": tb,
        "subgroups": subgroup_deltas(ca, cb),
        "bootstrap": TE.paired_bootstrap(ca, cb, n_boot=n_boot, seed=seed, block="date")}
    out["frames"] = {"diag_control": fa, "diag_candidate": fb,
                     "dep_control": ca, "dep_candidate": cb}
    return out


def report(res: dict, title: str) -> None:
    """Prints a compare() result: threshold shift, subgroup deltas, bootstrap CIs."""
    print(f"\n=== {title}: threshold shift ===")
    print("  control  :", res["deployment"]["thresholds_control"].to_string(index=False))
    print("  candidate:", res["deployment"]["thresholds_candidate"].to_string(index=False))
    print(f"  frozen   : {res['frozen_thresholds']}")
    for key, label in (("diagnostic", "DIAGNOSTIC (both arms on frozen thresholds)"),
                       ("deployment", "DEPLOYMENT (each arm on its own calibrated thresholds)")):
        sg, bt = res[key]["subgroups"], res[key]["bootstrap"]
        print(f"\n--- {label} ---")
        print(sg.round(6).to_string(index=False))
        pr = sg[sg.subgroup.str.startswith("panel::")]
        fo = sg[sg.subgroup.str.startswith("fold::")]
        print(f"  panels positive {int((pr.delta > 0).sum())}/{len(pr)}   "
              f"folds positive {int((fo.delta > 0).sum())}/{len(fo)}   "
              f"worst panel {pr.delta.min():+.6f}")
        print(f"  delta {bt['delta']:+.6f}  95% CI [{bt['ci_lo']:+.6f}, {bt['ci_hi']:+.6f}]"
              f"  P(delta>0) {bt['p_gt0']:.3f}")
    print("\n  Adoption or rejection follows DEPLOYMENT only; the diagnostic row "
          "describes probability-scale drift.")


# spaced.parquet holds exactly 2 queue_ongoing windows for D7_I10_W and both score
# IoU 0. score2.aggregate averages 4 corridor families at the top, so that one
# (panel, condition) cell carries 1/4 * 1/2 * 1/2 = 0.0625 of the score, and it
# halves the weight of that panel's ONSET differences. It cancels in a paired
# delta -- no candidate has ever moved those 2 windows -- but it changes the
# RANKING between candidates. See task2-failure-analysis, "D7_I10_W x ongoing".
DEAD_CELL = ("D7_I10_W", "queue_ongoing")


def drop_dead_cell(frame: pd.DataFrame) -> pd.DataFrame:
    """Returns frame without the rows of DEAD_CELL (D7_I10_W, queue_ongoing)."""
    panel, cond = DEAD_CELL
    return frame[~((frame.panel == panel) & (frame.condition == cond))]


def _dep_frames(res: dict):
    """Locate the deployment control/candidate frames in either result shape.

    compare() returns {"frames": {"dep_control": ..., "dep_candidate": ...}}.
    The FP scripts (86, 96, 101, 102) build their own per-arm results as
    {arm: {"deployment": {"frames": {"control": ..., "candidate": ...}}}} and
    are handed one arm's dict. Accepting both is what stops a reporting helper
    from taking down a three-hour run -- scripts/101 crashed here after its
    numbers were computed and before any artifact was written, which is exactly
    how scripts/91 lost its provenance.
    """
    if "frames" in res:
        f = res["frames"]
        return f["dep_control"], f["dep_candidate"]
    if "deployment" in res and "frames" in res["deployment"]:
        f = res["deployment"]["frames"]
        return f["control"], f["candidate"]
    raise KeyError(f"no deployment frames in result with keys {sorted(res)}")


def dead_cell_report(res: dict, title: str) -> dict[str, float]:
    """Deployment delta with and without the dead cell. Both are printed."""
    a, b = _dep_frames(res)
    out = {"with_dead": TE.s_queue(b) - TE.s_queue(a),
           "without_dead": TE.s_queue(drop_dead_cell(b)) - TE.s_queue(drop_dead_cell(a))}
    out["shift"] = out["without_dead"] - out["with_dead"]
    print(f"\n=== {title}: the D7_I10_W dead cell ===")
    print(f"  deployment delta, as the ledger records it : {out['with_dead']:+.6f}")
    print(f"  deployment delta, dead cell removed        : {out['without_dead']:+.6f}"
          f"   (shift {out['shift']:+.6f})")
    # Fixed reference deltas of earlier ablations (labelled by script number)
    # and the no-op floor, printed for comparison.
    print("  ranking reference, dead cell removed: "
          "103A_D +0.009095, 100 +0.008421, 73 +0.007248, 99 +0.007227, 87 +0.005863, 88 +0.005724, 83 +0.005257, 97 +0.003326, 101b +0.003288, 96b +0.002794, 98 +0.000486; no-op floor 0.000787")
    return out


def save_artifacts(out_dir, slug: str, a_oof: pd.DataFrame, b_oof: pd.DataFrame,
                   res: dict) -> dict[str, str]:
    """Raw probabilities, chosen thresholds and per-window IoU, so a later check
    does not require retraining."""
    # out_dir: a pathlib.Path; slug prefixes the three file names. Returns the
    # SHA256 of each written file, plus their paths under "_paths".
    from . import prov
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    cols = ["window_id", "panel", "condition", "date", "fold", "link_id", "step",
            "label", "prob"]
    p = out_dir / f"{slug}_probs.parquet"
    # .attrs carries DataFrames, and concat compares attrs for equality, so strip
    # them first rather than letting pandas raise on the ambiguous truth value.
    ca = a_oof[cols].copy(); ca.attrs = {}; ca["arm"] = "control"
    cb = b_oof[cols].copy(); cb.attrs = {}; cb["arm"] = "candidate"
    pd.concat([ca, cb], ignore_index=True).to_parquet(p, index=False)
    paths["probs"] = str(p)
    p = out_dir / f"{slug}_thresholds.csv"
    pd.concat([res["deployment"]["thresholds_control"].assign(arm="control"),
               res["deployment"]["thresholds_candidate"].assign(arm="candidate")],
              ignore_index=True).to_csv(p, index=False)
    paths["thresholds"] = str(p)
    p = out_dir / f"{slug}_window_iou.csv"
    pd.concat([res["frames"][k].assign(frame=k) for k in res["frames"]],
              ignore_index=True).to_csv(p, index=False)
    paths["window_iou"] = str(p)
    return {k: prov.sha256_file(v) for k, v in paths.items()} | {"_paths": paths}
