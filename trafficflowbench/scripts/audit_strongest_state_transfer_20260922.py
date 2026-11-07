"""All-panel transfer of fixed auxiliary density with development-only scope selection."""
from pathlib import Path
import importlib.util
import json
import shutil
import time
import sys
import hashlib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.tfb import release as R,grid,baselines as B,physics,score1,predict1 as P1,task1stack as TS
sp=importlib.util.spec_from_file_location("strong_state",ROOT/"scripts/audit_strongest_state_20260922.py")
S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S);H=S.H
BASE=ROOT/"build/strongest_candidate_20260922/state"
OUT=BASE/"transfer";CACHE=OUT/"cache"
OLD=ROOT/"build/fixed_holdout20_20260922/state"
FIRST=["D7_I10_E","D12_I5_N"]
START=time.time();threadpool_limits(limits=3)


def generate(panel,phase,models,aux,W,features,cfg):
    dest=CACHE/f"{phase}_{panel}_predictions.parquet"
    if dest.exists():return
    original=OLD/f"{phase}_{panel}_predictions.parquet"
    if panel in FIRST:
        shutil.copyfile(original,dest);return
    mi,tm=H.indexes(panel)
    hist=P1.historical_mean(panel,P1.dev_train_files(panel))
    chunks=[cfg["density_calibration_dates"]] if phase=="calibration" else cfg["held_lrtc_chunks"]
    prior=pd.read_parquet(original) if phase=="held" else None
    outputs=[]
    for chunk_i,dates in enumerate(chunks):
        tick=time.time();names=[H.date_name(d) for d in dates]
        gs=[grid.build_masked(panel,mi[n][1]) for n in names]
        lr={f:B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f],nan=0.) for g in gs],axis=2),rho=P1.LRTC_KNOBS[f][0],theta=P1.LRTC_KNOBS[f][1]) for f in P1.FIELDS}
        par=physics.link_params(panel,list(map(str,gs[0].links)))
        for i,(name,date,g) in enumerate(zip(names,dates,gs)):
            X=TS.feature_table(g,i,lr,hist,W,par);assert list(X)==features
            khat=np.maximum(aux.predict(X,num_threads=3),0)
            if prior is None:
                v,q=(H.fallback(m.predict(X,num_threads=3),f,g,hist) for m,f in zip(models,P1.FIELDS))
                true=grid.build(panel,tm[name],mi[name][1]);assert np.array_equal(true.target,g.target)
                z=pd.DataFrame(dict(date=date,panel=panel,regime=mi[name][0],flat=np.flatnonzero(g.target),champion_speed=v,champion_flow=q,true_speed=true.truth["speed_kmh"][g.target],true_flow=true.truth["flow_vph"][g.target]))
            else:
                z=prior[prior.date==date].copy();assert np.array_equal(z.flat.to_numpy(),np.flatnonzero(g.target))
            z["density_khat"]=khat;outputs.append(z)
        print("CACHE",phase,panel,"chunk",chunk_i,"sec",round(time.time()-tick,1),flush=True)
    pd.concat(outputs,ignore_index=True).to_parquet(dest,index=False)


def weights(panels):
    out={}
    for panel in panels:
        p=pd.read_parquet(CACHE/f"calibration_{panel}_predictions.parquet")
        mi,_=H.indexes(panel);g=grid.build_masked(panel,mi[H.date_name(p.date.iloc[0])][1])
        lv=score1.lane_vector(pd.Series(g.station),pd.Series(g.links),score1.lanes_by_key(panel))
        lanes=lv[p.flat.to_numpy()//len(g.times)]
        rv=float(np.sqrt(np.square(p.champion_speed-p.true_speed).mean()))
        rq=float(np.sqrt(np.square((p.champion_flow-p.true_flow)/lanes).mean()))
        out[panel]=dict(rmse_speed=rv,rmse_flow_lane=rq,speed_precision=.54/(25*max(rv,.1)),flow_lane_precision=.46/(600*max(rq,1.)))
    return out


def selection(cal,panels,dates):
    """Select among original two-panel, global transfer, and 3-block consensus."""
    possibilities={"current_two_panel_joint_025":{p:(.25 if p in FIRST else 0.) for p in panels}}
    decisions=[]
    for a in [.1,.25,.5]:
        arm=f"joint_{a:g}";active=[]
        for panel in panels:
            gains=[]
            for block in [dates[:5],dates[5:10],dates[10:]]:
                z=cal[(cal.panel==panel)&cal.date.isin(block)]
                r0,_=H.aggregate(z[z.arm=="champion"]);r1,_=H.aggregate(z[z.arm==arm])
                gains.append(r1["weighted_state_physics"]-r0["weighted_state_physics"])
            if min(gains)>0:active.append(panel)
            decisions.append(dict(alpha=a,panel=panel,block_gains=gains,three_blocks_positive=min(gains)>0))
        possibilities[f"all10_joint_{a:g}"]={p:a for p in panels}
        possibilities[f"consensus_joint_{a:g}"]={p:(a if p in active else 0.) for p in panels}
    rows=[]
    for name,mapping in possibilities.items():
        sub=pd.concat([cal[(cal.panel==p)&(cal.arm==("champion" if a==0 else f"joint_{a:g}"))] for p,a in mapping.items()],ignore_index=True)
        r,_=H.aggregate(sub);r.update(rule=name,active_panels=sum(a>0 for a in mapping.values()));rows.append(r)
    t=pd.DataFrame(rows).sort_values(["weighted_state_physics","active_panels","rule"],ascending=[False,True,True])
    t.to_csv(OUT/"calibration_scope_results.csv",index=False)
    (OUT/"three_block_consensus.json").write_text(json.dumps(decisions,indent=2))
    name=t.iloc[0].rule
    return dict(rule=name,panel_alphas=possibilities[name],calibration=t.iloc[0].to_dict())


def main():
    import lightgbm as lgb
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    design=json.loads((ROOT/"build/fixed_holdout20_20260922/design.json").read_text());cfg=design["state"]
    panels=cfg["panels"]
    protocol=dict(phase="Sequential extension after observing the first two-panel joint projection holdout; retrospective research, not a fresh blind test",
                  decision_dates=cfg["density_calibration_dates"],holdout_dates=design["holdout_dates"],
                  alpha_grid=[.1,.25,.5],families=["current2paneljoint.25","all10joint","3-contiguous-calibration-block-consensusjoint"],
                  consensus="For a fixed shared alpha, activate each panel iff weighted state+physics delta positive in EACH of 3 fixed5date development blocks; selectscope+sharedalpha onfull15daycalibration",
                  state_model="Same champion boosters fit throughDec20; fixed auxiliarydensitybooster trainedonlyD7_I10_E/D12_I5_NthroughDec20; test transfer to all10panel features",
                  weights="Perpanel state risk derivative from developmentcalibrationRMSE only, fixed beforeheld",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    ms,mq,fs=TS.load();aux=lgb.Booster(model_file=str(H.AUX_PATH));W=P1.load_weights()
    for panel in panels:generate(panel,"calibration",(ms,mq),aux,W,fs,cfg)
    w=weights(panels);(OUT/"weights.json").write_text(json.dumps(w,indent=2))
    S.OUT=OUT;S.SRC=CACHE;S.PANELS=panels
    recipes={"champion":("joint_aux",0.),**{f"joint_{a:g}":("joint_aux",a) for a in [.1,.25,.5]}}
    cal=S.evaluate("calibration",recipes,w);cal.to_csv(OUT/"calibration_daily.csv",index=False)
    choice=selection(cal,panels,cfg["density_calibration_dates"])
    choice.update(weights=w,selected_before_extension_held=True)
    (OUT/"selection_before_held.json").write_text(json.dumps(choice,indent=2))
    print("TRANSFER CHOICE",json.dumps(choice["panel_alphas"]),choice["rule"],flush=True)
    pieces=[]
    old=pd.read_csv(OLD/"held_daily.csv");old=old[old.alpha==0].copy()
    for panel,a in choice["panel_alphas"].items():
        if a:
            generate(panel,"held",(ms,mq),aux,W,fs,cfg)
            S.PANELS=[panel]
            d=S.evaluate("held",{"candidate":("joint_aux",a)},w)
            pieces.append(d)
        else:
            d=old[old.panel==panel].copy();d["arm"]="candidate";pieces.append(d)
    cand=pd.concat(pieces,ignore_index=True)
    first=pd.read_csv(BASE/"held_daily.csv")
    refs=first[first.arm.isin(["champion","flow_aux_0.1","flow_aux_0.25","joint_aux_0.25"])].copy()
    d=pd.concat([cand,refs],ignore_index=True);d.to_csv(OUT/"held_daily.csv",index=False)
    results=S.summarize(d,"held")
    (OUT/"results.json").write_text(json.dumps(dict(selection=choice,held=results.to_dict("records"),elapsed_seconds=time.time()-START),indent=2))
    print("TRANSFER HELD",results.to_string(index=False),flush=True)


if __name__=="__main__":main()
