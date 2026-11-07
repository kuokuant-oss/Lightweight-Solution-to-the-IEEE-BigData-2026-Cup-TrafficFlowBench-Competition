"""Development-only selection of density correction geometry, then fixed holdout.

Four predeclared families: flow-only auxiliary density, joint state-cost projection,
bounded short-gap temporal density, and equal auxiliary/short-gap density blend.
The same frozen champion and auxiliary predictions are reused. No fitting or
model selection uses the final55 held dates. Research is retrospective.
"""
from pathlib import Path
import importlib.util
import json
import time
import sys
import hashlib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tfb import release as R, grid, physics, score1, predict1 as P1
SPEC = importlib.util.spec_from_file_location("state_holdout", ROOT / "scripts/audit_holdout20_state.py")
H = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(H)
SRC = ROOT / "build/fixed_holdout20_20260922/state"
OUT = ROOT / "build/strongest_candidate_20260922/state"
PANELS = ["D7_I10_E", "D12_I5_N"]
ALPHAS = [.1, .25, .5]
threadpool_limits(limits=3)
START = time.time()


def short_gap_density(g, max_gap=3):
    """Only interpolate gaps bounded by valid observed density, never extrapolate."""
    obs = g.value["flow_vph"] / np.clip(g.value["speed_kmh"], 1, None)
    out = np.full_like(obs, np.nan)
    for j, row in enumerate(obs):
        hits = np.flatnonzero(np.isfinite(row))
        for lo, hi in zip(hits[:-1], hits[1:]):
            if 1 <= hi-lo-1 <= max_gap:
                out[j, lo+1:hi] = np.linspace(row[lo], row[hi], hi-lo+1)[1:-1]
    return out[g.target]


def cost_weights():
    """Derivative of weighted RMSE risk, estimated on development only."""
    d = pd.read_csv(SRC / "calibration_daily.csv")
    d = d[d.alpha == 0]
    p = d.groupby("panel")[["n", "speed_sse", "flow_lane_sse"]].sum()
    out = {}
    for panel, z in p.iterrows():
        rv, rq = np.sqrt(z.speed_sse/z.n), np.sqrt(z.flow_lane_sse/z.n)
        out[panel] = dict(rmse_speed=float(rv), rmse_flow_lane=float(rq),
                          speed_precision=float(.54 / (25*max(rv, .1))),
                          flow_lane_precision=float(.46 / (600*max(rq, 1.))))
    return out


def apply(v, q, khat, temporal, lanes, family, alpha, weights):
    basek = q / v
    if family in ("flow_aux", "joint_aux"):
        target = khat
    elif family == "joint_shortgap":
        target = np.where(np.isfinite(temporal), temporal, basek)
    elif family == "joint_aux_shortgap":
        target = np.where(np.isfinite(temporal), .5*khat+.5*temporal, khat)
    else:
        raise ValueError(family)
    kk = np.maximum((1-alpha)*basek + alpha*target, 0)
    if family == "flow_aux":
        return v, np.maximum(kk*v, P1.FLOW_FLOOR)
    wv = weights["speed_precision"]
    wq = weights["flow_lane_precision"] / np.square(lanes)
    # Exact weighted projection to q = k*v. The direction is adaptive to k/lanes.
    vv = (wv*v + wq*kk*q) / (wv + wq*np.square(kk))
    vv = np.clip(vv, P1.SPEED_FLOOR, P1.SPEED_CAP)
    qq = np.maximum(kk*vv, P1.FLOW_FLOOR)
    return vv, qq


def evaluate(phase, recipes, weights):
    rows = []
    for panel in PANELS:
        mi, tm = H.indexes(panel)
        p = pd.read_parquet(SRC / f"{phase}_{panel}_predictions.parquet")
        att, ri = physics.ramp_index(panel, "train")
        previous = {name: None for name in recipes}
        outputs = []
        for date, z in p.groupby("date", sort=True):
            name = H.date_name(date); rg, mp = mi[name]
            gm = grid.build_masked(panel, mp)
            assert np.array_equal(z.flat.to_numpy(), np.flatnonzero(gm.target))
            # Legal observed inputs are prepared before reading labels.
            temporal = short_gap_density(gm)
            link_lanes = score1.lane_vector(pd.Series(gm.station), pd.Series(gm.links), score1.lanes_by_key(panel))
            lanes = np.broadcast_to(link_lanes[:, None], gm.target.shape)[gm.target]
            v, q, khat = z.champion_speed.to_numpy(), z.champion_flow.to_numpy(), z.density_khat.to_numpy()
            pred = {}
            for arm, (family, alpha) in recipes.items():
                pred[arm] = (v, q) if arm == "champion" else apply(v, q, khat, temporal, lanes, family, alpha, weights[panel])
            g = grid.build(panel, tm[name], mp)
            par = physics.link_params(panel, list(map(str, g.links)))
            keep = physics.scoreable_mask(panel, list(map(str,g.links)), "train", g.times)
            rv = physics.ramp_valid_grid(list(map(str,g.links)), g.times, att, ri)
            if phase == "held":
                output = z[["date", "panel", "regime", "flat"]].copy()
            for arm, (vv, qq) in pred.items():
                row, previous[arm] = H.metrics(panel, rg, date, g, vv, qq, par, keep, rv, link_lanes, previous[arm])
                row.update(arm=arm, phase=phase, shortgap_rows=int(np.isfinite(temporal).sum()))
                rows.append(row)
                if phase == "held":
                    output[arm+"_speed"] = vv
                    output[arm+"_flow"] = qq
            if phase == "held": outputs.append(output)
        if outputs: pd.concat(outputs, ignore_index=True).to_parquet(OUT / f"held_{panel}_predictions.parquet", index=False)
        print("EVALUATED", phase, panel, "elapsed", round(time.time()-START, 1), flush=True)
    return pd.DataFrame(rows)


def summarize(d, phase):
    result=[];tables=[]
    for arm, z in d.groupby("arm"):
        r, t=H.aggregate(z);r["arm"]=arm;result.append(r);t["arm"]=arm;tables.append(t)
    result=pd.DataFrame(result).sort_values("weighted_state_physics", ascending=False)
    result.to_csv(OUT / f"{phase}_results.csv", index=False)
    pd.concat(tables).to_csv(OUT / f"{phase}_by_panel_regime.csv", index=False)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    weights=cost_weights()
    recipes={"champion":("flow_aux",0.)}
    for family in ["flow_aux","joint_aux","joint_shortgap","joint_aux_shortgap"]:
        for a in ALPHAS: recipes[f"{family}_{a:g}"]=(family,a)
    protocol=dict(development_calibration=["2030-12-21","2031-01-04"], holdout=["2031-01-05","2031-02-28"],
                  recipes=recipes, panels=PANELS, shortgap_max_cells=3, weights=weights,
                  selection="Highest development .35*S_state+.15*S_physics; one global recipe and alpha for both panels; champion included; no held tuning",
                  composition="Other8panels exactly champion, aggregateall10 after selected2panelheld",
                  note="Retrospective research: January/February have been read in prior work. Fixed-held dates excluded from this round of parameter selection.",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    cal=evaluate("calibration", recipes, weights);cal.to_csv(OUT/"calibration_daily.csv",index=False)
    cr=summarize(cal,"calibration")
    chosen=str(cr.iloc[0].arm)
    decision=dict(chosen_arm=chosen, recipe=recipes[chosen], weights=weights,
                  calibration=cr.iloc[0].to_dict(), selected_before_held=True)
    (OUT/"selection_before_held.json").write_text(json.dumps(decision,indent=2))
    print("CALIBRATION",cr.to_string(index=False),"CHOSEN",chosen,flush=True)
    # Literal current submission .1 and previously development-selected .25 are fixed references.
    evalrecipes={name:recipes[name] for name in dict.fromkeys(["champion","flow_aux_0.1","flow_aux_0.25",chosen])}
    held=evaluate("held",evalrecipes,weights)
    old=pd.read_csv(SRC/"held_daily.csv");old=old[(old.alpha==0)&(~old.panel.isin(PANELS))].copy()
    others=[]
    for arm in evalrecipes:
        z=old.copy();z["arm"]=arm;others.append(z)
    full=pd.concat([held,*others],ignore_index=True)
    full.to_csv(OUT/"held_daily.csv",index=False)
    hr=summarize(full,"held")
    (OUT/"results.json").write_text(json.dumps(dict(chosen_arm=chosen,held=hr.to_dict("records"),elapsed_seconds=time.time()-START),indent=2))
    print("HELD",hr.to_string(index=False),flush=True)


if __name__=="__main__": main()
