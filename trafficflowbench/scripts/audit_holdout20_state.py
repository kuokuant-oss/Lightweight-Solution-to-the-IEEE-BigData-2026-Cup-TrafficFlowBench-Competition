"""Frozen state boosters on the common chronological 20% holdout.

Density alpha is selected on development dates before any held labels are read.
No production artifacts are modified. Physics uses eligible released truth as
the conservation reference, not unavailable organizer-held boundary fluxes.
"""

# Role: research audit of the Task 1 state, kept as a record; the reproduction
# pipeline does not run it, but the later state audits import its helpers.
# It replays the frozen Task 1 stack boosters and the auxiliary density
# booster (AUX_PATH) on train days, where truth exists, with the masked layer
# as the only input. The density blend flow = (1 - alpha) * q + alpha * k_aux * v
# is scored for several alpha on development calibration dates, alpha is
# chosen there, and then every panel is scored on the 55 holdout days.
# Scores: S_state (Task 1) and S_FD, S_LWR, S_physics (Task 3, with released
# truth as the conservation reference), combined as
# .35 * S_state + .15 * S_physics (`weighted_state_physics`).
# Outputs in BASE/state: per-cell predictions, daily score sums, calibration
# and holdout results.

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tfb import release as R, grid, baselines as B, predict1 as P1, task1stack as TS, score1, physics

# AUX_PATH: the auxiliary density booster; it predicts the density k (veh/km)
# of each target cell from the stack features.
BASE = ROOT / "build/fixed_holdout20_20260922"
OUT = BASE / "state"
AUX_PATH = ROOT / "build/final_sprint_review_20260922/experiment_audit/state_recency/aux_density_weightedk/k_l1_weightedL_600.txt"
START = time.time()
threadpool_limits(limits=2)


def sha(path):
    """Returns the SHA-256 hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def date_name(date):
    """Returns the day file name of a date, e.g. synthetic_mainline_2031_01_05.parquet."""
    return "synthetic_mainline_" + date.replace("-", "_") + ".parquet"


def indexes(panel):
    """Indexes a panel's train day files by file name.

    Returns:
        (masked, truth): masked maps file name -> (mask regime, path); truth
        maps file name -> path of the unmasked file.
    """
    return ({p.name: (rg, p) for rg in R.REGIME_ORDER for p in R.masked_files(panel, "train", rg)},
            {p.name: p for p in R.truth_files(panel, "train")})


def aggregate(d):
    """Pool numerator/denominator, then apply official hierarchical means."""
    # Per (panel, regime): S_speed, S_flow and S_state from pooled squared
    # errors; S_LWR = 1 - min(1, sum|dN_pred - dN_true| / sum|dN_true|) (0
    # without transitions); S_FD = 1 - sum|q - q_FD| / sum|q| per lane (0 when
    # more than EMPTY_SHARE of the target cells have flow below
    # EMPTY_FLOOR_VPH); S_physics = (S_FD + 2 * S_LWR) / 3. S_state is
    # averaged regime -> panel -> family -> overall; the physics scores are
    # averaged over regimes, then over panels.
    g = d.groupby(["panel", "regime"], sort=True).sum(numeric_only=True)
    g["S_speed"] = np.maximum(0, 1 - np.sqrt(g.speed_sse / g.n) / 25)
    g["S_flow"] = np.maximum(0, 1 - np.sqrt(g.flow_lane_sse / g.n) / 600)
    g["S_state"] = .54 * g.S_speed + .46 * g.S_flow
    g["S_LWR"] = np.maximum(0, 1 - np.minimum(1, g.lwr_num / (g.lwr_den + 1e-9)))
    g.loc[g.n_transitions == 0, "S_LWR"] = 0
    g["S_FD"] = np.maximum(0, 1 - g.fd_num / (g.fd_den + 1e-9))
    g.loc[g.n_empty_target / g.n > physics.EMPTY_SHARE, "S_FD"] = 0
    g["S_physics"] = (g.S_FD + 2 * g.S_LWR) / 3
    pp = g.groupby("panel").mean(numeric_only=True)
    fam = R.families()
    state = pp.S_state.groupby(pp.index.map(fam)).mean().mean()
    r = {k: float(pp[k].mean()) for k in ["S_FD", "S_LWR", "S_physics"]}
    r.update(S_state=float(state), weighted_state_physics=float(.35 * state + .15 * r["S_physics"]),
             target_rows=int(g.n.sum()), eligible_rows=int(g.n_eligible.sum()),
             transitions=int(g.n_transitions.sum()), panels=len(pp), panel_regimes=len(g),
             rmse_speed=float(np.sqrt(g.speed_sse.sum() / g.n.sum())),
             rmse_flow_lane=float(np.sqrt(g.flow_lane_sse.sum() / g.n.sum())))
    return r, g.reset_index()


def fallback(raw, field, g, hist):
    """Applies the production fallback chain and bounds to one field's raw stack prediction.

    Missing or below-floor values take the weekday x time-of-day history mean,
    then the median of the day's observed values. Speed is clipped to
    [SPEED_FLOOR, SPEED_CAP]; flow is floored at FLOW_FLOOR.

    Args:
        raw: predictions at the day's target cells (g.target order).
        field: 'speed_kmh' or 'flow_vph'.
        g: the day's masked grid.
        hist: history means keyed (field, weekday).

    Returns:
        The corrected values, same shape as `raw`.
    """
    floor = P1.SPEED_FLOOR if field == "speed_kmh" else P1.FLOW_FLOOR
    z = raw.copy()
    h = hist.get((field, pd.Timestamp(g.times[0]).weekday()))
    bad = ~np.isfinite(z) | (z < floor)
    if h is not None:
        hh = h[g.target]
        z = np.where(bad & np.isfinite(hh), hh, z)
    bad = ~np.isfinite(z) | (z < floor)
    if bad.any():
        obs = g.value[field]
        median = float(np.nanmedian(obs)) if np.isfinite(obs).any() else floor
        z = np.where(bad, median, z)
    return np.clip(z, floor, P1.SPEED_CAP) if field == "speed_kmh" else np.maximum(z, floor)


def metrics(panel, rg, date, g, v, q, par, keep, rv, lanes, previous):
    """Computes one day's Task 1 and Task 3 score sums for one prediction.

    The scored field is truth at non-target cells and the prediction at target
    cells, as in the official Task 3 scorer.

    Args:
        panel: panel ID.
        rg: mask regime of the day.
        date: the day.
        g: truth grid of the day.
        v: predicted speed at g.target cells.
        q: predicted flow at g.target cells.
        par: link parameters (length, free speed, capacity, lanes, critical
            and jam density).
        keep: links that enter S_LWR (observability filter).
        rv: (link x time) ramp validity; transitions on invalid cells are dropped.
        lanes: scorer lane count per link (flow errors are per lane).
        previous: last time slice of the previous day for this prediction, or None.

    Returns:
        (row, last): row holds the sums (squared errors, FD and LWR numerators
        and denominators, counts); last holds the final slice for the next
        day's cross-midnight transition.
    """
    m = g.target
    vt, qt = g.truth["speed_kmh"], g.truth["flow_vph"]
    V, Q = vt.copy(), qt.copy()
    V[m], Q[m] = v, q
    # N: vehicles on the link = density q / v (veh/km) x length (km), with the
    # prediction (N) and with truth (Nt).
    L = par["length_km"][:, None]
    N, Nt = Q / np.clip(V, 1, None) * L, qt / np.clip(vt, 1, None) * L
    # A transition counts when both slices are finite, eligible and on a kept
    # link, the ramp cells are valid and the time step is at most DT_MAX_H.
    finite = g.eligible & np.isfinite(N) & np.isfinite(Nt) & keep[:, None]
    valid = finite[:, :-1] & finite[:, 1:] & rv[:, :-1]
    times = pd.to_datetime(g.times, utc=True)
    dt = np.diff(times.asi8) / 3.6e12
    valid &= (dt > 0)[None, :] & (dt <= physics.DT_MAX_H)[None, :]
    # LWR error: change of the predicted vehicle count minus change of the
    # true count; `signal` is the true change.
    err = np.abs(np.diff(N, axis=1) - np.diff(Nt, axis=1))
    signal = np.abs(np.diff(Nt, axis=1))
    touch = m[:, :-1] | m[:, 1:]
    ln, ld, transitions = float(err[valid].sum()), float(signal[valid].sum()), int(valid.sum())
    touching = int((valid & touch).sum())
    cross_num, cross_den, cross_n = 0., 0., 0
    # Transition from the previous day's last slice into this day's first slice
    # (same mask regime only).
    if previous is not None and previous["regime"] == rg:
        gap = (times[0] - previous["time"]).total_seconds() / 3600
        if 0 < gap <= physics.DT_MAX_H:
            cross = previous["finite"] & finite[:, 0] & previous["rv"]
            de = np.abs((N[:, 0] - previous["N"]) - (Nt[:, 0] - previous["Nt"]))
            ds = np.abs(Nt[:, 0] - previous["Nt"])
            cross_num, cross_den, cross_n = float(de[cross].sum()), float(ds[cross].sum()), int(cross.sum())
            # Attribute the transition to its start date, including bootstrap.
            prior_row = previous["row"]
            prior_row["lwr_num"] += cross_num
            prior_row["lwr_den"] += cross_den
            prior_row["n_transitions"] += cross_n
            prior_row["n_touching_transitions"] += int((cross & (previous["target"] | m[:, 0])).sum())
            prior_row["midnight_num"] += cross_num
            prior_row["midnight_den"] += cross_den
            prior_row["midnight_transitions"] += cross_n
    # Per-lane triangular fundamental diagram: free branch vf * k up to the
    # critical density, then a straight line falling to 0 at jam density.
    lane = np.maximum(par["lanes"], 1)[:, None]
    ql, kl = Q / lane, Q / np.clip(V, 1, None) / lane
    cap, kc, kj = par["capacity_vph"][:, None] / lane, par["critical_density"][:, None] / lane, par["k_jam"][:, None] / lane
    qfd = np.where(kl <= kc, par["free_speed_kmh"][:, None] * kl,
                   cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - kl, 0))
    ok = g.eligible & np.isfinite(ql) & np.isfinite(qfd)
    row = dict(panel=panel, regime=rg, date=date, n=int(m.sum()), n_target=int(m.sum()), n_eligible=int(ok.sum()),
               speed_sse=float(np.square(v - vt[m]).sum()),
               flow_lane_sse=float(np.square((q - qt[m]) / np.broadcast_to(lanes[:, None], m.shape)[m]).sum()),
               fd_num=float(np.abs(ql[ok] - qfd[ok]).sum()), fd_den=float(np.abs(ql[ok]).sum()),
               lwr_num=ln, lwr_den=ld, n_transitions=transitions, n_touching_transitions=touching,
               midnight_num=0., midnight_den=0., midnight_transitions=0,
               n_empty_target=int((q < physics.EMPTY_FLOOR_VPH).sum()))
    last = dict(regime=rg, time=times[-1], N=N[:, -1], Nt=Nt[:, -1], finite=finite[:, -1], rv=rv[:, -1], target=m[:, -1], row=row)
    return row, last


def evaluate(panel, chunks, alphas, label, models, aux, W, features, density_panels):
    """Predicts and scores one panel on chunks of train days for each alpha.

    Each chunk is one LRTC tensor (its days are completed together). Features
    and predictions come from the masked layer only; truth is read afterwards
    for scoring.

    Args:
        panel: panel ID.
        chunks: list of date lists.
        alphas: density blend weights to score (0 = champion flow).
        label: phase name used in file names ('calibration' or 'held').
        models: (speed booster, flow booster) of the Task 1 stack.
        aux: auxiliary density booster.
        W: frozen blend weights (stack input).
        features: expected stack feature order.
        density_panels: panels where the density blend is applied.

    Returns:
        Daily score sums, one row per (day, alpha). The per-cell predictions
        and the daily table are also written under OUT.
    """
    mi, tm = indexes(panel)
    hist = P1.historical_mean(panel, P1.dev_train_files(panel))
    att, ri = physics.ramp_index(panel, "train")
    prev = {a: None for a in alphas}
    rows, target_parts = [], []
    for chunk_no, dates in enumerate(chunks):
        tick = time.time()
        names = [date_name(d) for d in dates]
        # Model inputs originate only from the released masked view.
        gs = [grid.build_masked(panel, mi[n][1]) for n in names]
        lr = {f: B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f], nan=0.) for g in gs], axis=2),
                            rho=P1.LRTC_KNOBS[f][0], theta=P1.LRTC_KNOBS[f][1]) for f in P1.FIELDS}
        links = list(map(str, gs[0].links))
        par = physics.link_params(panel, links)
        lanes = score1.lane_vector(pd.Series(gs[0].station), pd.Series(links), score1.lanes_by_key(panel))
        keep = physics.scoreable_mask(panel, links, "train", gs[0].times)
        for i, (name, date, gm) in enumerate(zip(names, dates, gs)):
            X = TS.feature_table(gm, i, lr, hist, W, par)
            assert list(X) == features
            rawv, rawq = (model.predict(X, num_threads=4) for model in models)
            v, q = fallback(rawv, "speed_kmh", gm, hist), fallback(rawq, "flow_vph", gm, hist)
            khat = np.maximum(aux.predict(X, num_threads=4), 0) if panel in density_panels else None
            # Only after feature construction and prediction, load scoring truth.
            g = grid.build(panel, tm[name], mi[name][1])
            assert np.array_equal(g.target, gm.target) and g.links == gm.links
            assert np.array_equal(g.times, gm.times)
            rg = mi[name][0]
            rv = physics.ramp_valid_grid(links, g.times, att, ri)
            p = pd.DataFrame(dict(date=date, panel=panel, regime=rg, flat=np.flatnonzero(g.target),
                                  champion_speed=v, champion_flow=q, true_speed=g.truth["speed_kmh"][g.target],
                                  true_flow=g.truth["flow_vph"][g.target]))
            if khat is not None: p["density_khat"] = khat
            # Density blend: flow = (1 - alpha) * q + alpha * k_aux * v (floored); speed unchanged.
            for alpha in alphas:
                predq = np.maximum((1 - alpha) * q + alpha * khat * v, P1.FLOW_FLOOR) if khat is not None else q
                row, last = metrics(panel, rg, date, g, v, predq, par, keep, rv, lanes, prev[alpha])
                prev[alpha] = last
                row.update(alpha=alpha, phase=label)
                rows.append(row)
                if alpha: p[f"flow_alpha_{alpha:g}"] = predq
            target_parts.append(p)
        print(label, panel, "chunk", chunk_no, "dates", len(dates), "seconds", round(time.time()-tick, 1), flush=True)
    pd.concat(target_parts, ignore_index=True).to_parquet(OUT / f"{label}_{panel}_predictions.parquet", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"{label}_{panel}_daily.csv", index=False)
    return d


def main():
    """Selects alpha on development dates, then scores alpha 0, the selected alpha and .1 on the holdout."""
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)
    design = json.loads((BASE / "design.json").read_text())
    cfg = design["state"]
    dev = set(design["development_dates"])
    held = set(design["holdout_dates"])
    assert len(dev) == 218 and len(held) == 55 and not dev & held
    # History means use the dev-fold train days, all inside development.
    hist_dates = set(R.day_split().query("fold == 'dev'").date)
    assert hist_dates.issubset(dev) and max(hist_dates) == "2031-01-03"
    assert set(cfg["density_calibration_dates"]).issubset(dev)
    # Frozen stack boosters (fit labels through 2030-12-20) and the aux density
    # booster; nothing is refit here.
    models = TS.load(); features = models[2]; models = models[:2]
    aux = lgb.Booster(model_file=str(AUX_PATH)); W = P1.load_weights()
    manifest = dict(design_sha256=sha(BASE / "design.json"), fit_label_end="2030-12-20",
                    historical_masked_source_end=max(hist_dates), historical_source_count=len(hist_dates),
                    model_source_policy="Reuse frozen champion and density boosters with no held-date fit labels; select alpha on dev only",
                    density_panels=cfg["density_panels"], alpha_grid=cfg["density_alpha_grid"],
                    calibration_dates=cfg["density_calibration_dates"], held_lrtc_chunks=cfg["held_lrtc_chunks"],
                    physics_note="Eligible cells only; pooled FD/LWR numerators and denominators; actual same-regime cross-midnight transitions included. LWR is released-truth anchored proxy, hidden flux unavailable.",
                    retrospectivity="Density feature family and panel selection were previously informed by January/February research; not a fresh blind test",
                    sources={str(p.relative_to(ROOT)): sha(p) for p in [AUX_PATH, TS.DIR / "stack_speed.txt", TS.DIR / "stack_flow.txt", TS.DIR / "stack_features.csv", P1.WEIGHTS_CSV, Path(__file__)]})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Alpha selection on the development calibration dates of the density panels.
    cal = pd.concat([evaluate(p, [cfg["density_calibration_dates"]], cfg["density_alpha_grid"], "calibration", models, aux, W, features, cfg["density_panels"]) for p in cfg["density_panels"]], ignore_index=True)
    cal.to_csv(OUT / "calibration_daily.csv", index=False)
    cr = []
    for a, d in cal.groupby("alpha"):
        r, g = aggregate(d); r["alpha"] = a; cr.append(r)
        g.to_csv(OUT / f"calibration_alpha_{a:g}_by_panel_regime.csv", index=False)
    cr = pd.DataFrame(cr).sort_values(["weighted_state_physics", "alpha"], ascending=[False, True])
    cr.to_csv(OUT / "calibration_results.csv", index=False)
    selected = float(cr.iloc[0].alpha)
    selection = dict(selected_alpha=selected, criterion="Development Dec21-Jan4 .35*S_state+.15*S_physics only; no held labels read", calibration_result=cr.iloc[0].to_dict(), elapsed_seconds=time.time()-START)
    (OUT / "selection_before_holdout.json").write_text(json.dumps(selection, indent=2))
    print("CALIBRATION", cr.to_string(index=False), "SELECTED", selected, flush=True)
    # Holdout: champion (alpha 0), the selected alpha and alpha .1, on every panel.
    alphas = sorted(set([0., selected, .1]))
    held_daily = pd.concat([evaluate(p, cfg["held_lrtc_chunks"], alphas, "held", models, aux, W, features, cfg["density_panels"]) for p in cfg["panels"]], ignore_index=True)
    held_daily.to_csv(OUT / "held_daily.csv", index=False)
    summary = []
    for a, d in held_daily.groupby("alpha"):
        r, g = aggregate(d)
        r.update(alpha=a, is_primary_candidate=a == selected, is_champion=a == 0, is_literal_prepared_alpha=a == .1)
        summary.append(r); g.to_csv(OUT / f"held_alpha_{a:g}_by_panel_regime.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "held_results.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(dict(selected_alpha=selected, held=summary, elapsed_seconds=time.time()-START), indent=2))
    print("HELD RESULTS", pd.DataFrame(summary).to_string(index=False), "elapsed", round(time.time()-START, 1), flush=True)


if __name__ == "__main__":
    main()
