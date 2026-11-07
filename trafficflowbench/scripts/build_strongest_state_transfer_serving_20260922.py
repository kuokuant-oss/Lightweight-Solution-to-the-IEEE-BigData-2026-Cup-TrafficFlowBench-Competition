"""Create full selected-panel state updates for the frozen all-panel transfer rule."""

# Role: reproduction pipeline, pipeline/task1.sh step 2 ("kstar"). k* is the
# joint density projection of audit_strongest_state_20260922.apply (family
# joint_aux) with the per-panel alpha frozen in
# transfer/selection_before_held.json (five panels at alpha .1, the rest 0).
# For each official state target cell of a selected panel, the density moves
# to k = (1 - alpha) * q / v + alpha * k_aux, where k_aux is the auxiliary
# booster's density, and speed and flow are projected onto q = k * v.
# Inputs: the v3 stack state CSV (CHAMP; in the pipeline a state-only
# rebuild), the k_aux rows of the two original panels (OLD), the frozen stack
# and aux boosters and the released masked layer. For the other panels the
# stack features are rebuilt per split and day, and 64 sampled cells per day
# check that the stack reproduces the CSV values.
# Output: transfer/serving/stateupdates.parquet (submission_id, panel, old and
# new speed_kmh / flow_vph, aux_density_vpkm, alpha) and manifest.json.

from pathlib import Path
import importlib.util,json,hashlib,sys,time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.tfb import release as R,grid,baselines as B,physics,score1,predict1 as P1,task1stack as TS
# S: audit_strongest_state_20260922 (apply; S.H = audit_holdout20_state for AUX_PATH).
sp=importlib.util.spec_from_file_location("joint",ROOT/"scripts/audit_strongest_state_20260922.py")
S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S)
# SEL: the frozen transfer selection (panel -> alpha, cost weights). CHAMP: the
# v3 stack submission CSV. OLD: k_aux rows of the two original panels (FIRST),
# written by build_density_serving_updates.py.
BASE=ROOT/"build/strongest_candidate_20260922/state/transfer"
OUT=BASE/"serving";SEL=BASE/"selection_before_held.json"
CHAMP=ROOT/"build/onset_condition_candidate_20260922/submission_onset_endpoint_top1.csv"
OLD=ROOT/"build/final_sprint_review_20260922/experiment_audit/state_recency/aux_density_weightedk/serving_updates.parquet"
FIRST=["D7_I10_E","D12_I5_N"]
threadpool_limits(limits=3)


def sha(p):
 """Returns the SHA-256 hex digest of a file, read in 8 MiB blocks."""
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()


def correct(z,khat,actual,choice):
 """Applies the k* projection to the target cells `z` of one panel.

 Args:
     z: key rows of one panel (submission_id, panel, station_id, link_id).
     khat: auxiliary density per row (veh/km).
     actual: champion state indexed by submission_id (speed_kmh, flow_vph).
     choice: the frozen selection (panel_alphas, weights).

 Returns:
     (updates, checks): old and new speed and flow per submission_id with
     k_aux and alpha; counts of capped or floored cells and the largest
     |q / v - target density| over cells where no bound applies.
 """
 panel=str(z.panel.iloc[0]);alpha=choice["panel_alphas"][panel]
 old=actual.reindex(z.submission_id);assert old.notna().all().all()
 v,q=old.speed_kmh.to_numpy(),old.flow_vph.to_numpy()
 lv=score1.lane_vector(z.station_id,z.link_id,score1.lanes_by_key(panel))
 vv,qq=S.apply(v,q,khat,np.full(len(z),np.nan),lv,"joint_aux",alpha,choice["weights"][panel])
 assert np.isfinite(vv).all() and np.isfinite(qq).all() and (vv>=P1.SPEED_FLOOR).all() and (qq>=P1.FLOW_FLOOR).all()
 target=(1-alpha)*q/v+alpha*khat
 unbound=(vv>P1.SPEED_FLOOR)&(vv<P1.SPEED_CAP)&(qq>P1.FLOW_FLOOR)
 err=float(np.max(np.abs(qq[unbound]/vv[unbound]-target[unbound]))) if unbound.any() else 0.
 assert err<1e-9
 o=z[["submission_id","panel"]].copy()
 o["old_speed_kmh"],o["old_flow_vph"],o["new_speed_kmh"],o["new_flow_vph"]=v,q,vv,qq
 o["aux_density_vpkm"],o["alpha"]=khat,alpha
 return o,dict(rows=len(z),v_cap=int((vv==P1.SPEED_CAP).sum()),v_floor=int((vv==P1.SPEED_FLOOR).sum()),q_floor=int((qq==P1.FLOW_FLOOR).sum()),density_ratio_error=err)


def fallback_sample(raw,field,g,hist,rr,cc):
 """Applies the production fallback chain and bounds to sampled cells (rows rr, columns cc) of one field."""
 floor=P1.SPEED_FLOOR if field=="speed_kmh" else P1.FLOW_FLOOR
 h=hist.get((field,pd.Timestamp(g.times[0]).weekday()));z=raw.copy();bad=~np.isfinite(z)|(z<floor)
 if h is not None:z=np.where(bad&np.isfinite(h[rr,cc]),h[rr,cc],z)
 bad=~np.isfinite(z)|(z<floor)
 if bad.any():
  ob=g.value[field];med=float(np.nanmedian(ob)) if np.isfinite(ob).any() else floor;z=np.where(bad,med,z)
 return np.clip(z,floor,P1.SPEED_CAP) if field=="speed_kmh" else np.maximum(z,floor)


def main():
 """Builds stateupdates.parquet for every selected panel and both scored splits."""
 import lightgbm as lgb
 start=time.time();OUT.mkdir(parents=True,exist_ok=True);chunks=OUT/"keys";chunks.mkdir(exist_ok=True)
 # Active panels have alpha > 0; `new` are those outside FIRST, rebuilt here.
 choice=json.loads(SEL.read_text());active=[p for p,a in choice["panel_alphas"].items() if a>0];new=[p for p in active if p not in FIRST]
 # Champion state values (speed_kmh, flow_vph) by submission_id.
 actual=pd.concat([z[z.task=="state"][["submission_id","speed_kmh","flow_vph"]] for z in pd.read_csv(CHAMP,usecols=["submission_id","task","speed_kmh","flow_vph"],chunksize=500000)],ignore_index=True).set_index("submission_id")
 assert actual.index.is_unique
 cols=["submission_id","task","panel","split","timestamp","station_id","link_id","mask_regime"]
 # Split the official state keys of the new panels into per-panel chunk files.
 for j,z in enumerate(pd.read_csv(R.RELEASE/"submission_key.csv",usecols=cols,chunksize=400000,dtype={c:str for c in cols if c!="submission_id"})):
  z=z[(z.task=="state")&z.panel.isin(new)]
  for p,k in z.groupby("panel",sort=False):k.to_parquet(chunks/f"{p}_{j}.parquet",index=False)
 # Frozen stack boosters, aux density booster and blend weights.
 ms,mq,fs=TS.load();aux=lgb.Booster(model_file=str(S.H.AUX_PATH));W=P1.load_weights()
 receipts=[];parity=[];cached=pd.read_parquet(OLD);paths=[]
 for panel in active:
  rows=[];checks=[]
  # FIRST panels: reuse the cached k_aux rows after checking their old values against the champion.
  if panel in FIRST:
   z=cached[cached.panel==panel];got=actual.reindex(z.submission_id)
   assert np.array_equal(got.speed_kmh.to_numpy(),z.speed_kmh.to_numpy()) and np.array_equal(got.flow_vph.to_numpy(),z.old_flow_vph.to_numpy())
   o,c=correct(z,z.aux_density_vpkm.to_numpy(),actual,choice);rows.append(o);checks.append(c)
  else:
   # Other panels: per split and day, rebuild the stack features at the official
   # target cells and predict k_aux.
   key=pd.concat([pd.read_parquet(p) for p in sorted(chunks.glob(f"{panel}_*.parquet")) if not p.name.startswith("._")],ignore_index=True)
   assert key.submission_id.is_unique;key["date"]=key.timestamp.str[:10]
   hist=P1.historical_mean(panel,P1.dev_train_files(panel))
   for split in ["validation","private"]:
    tick=time.time();files=P1.split_masked_files(panel,split);gs=[grid.build_masked(panel,p) for p in files]
    lr={f:B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f],nan=0.) for g in gs],axis=2),rho=P1.LRTC_KNOBS[f][0],theta=P1.LRTC_KNOBS[f][1]) for f in P1.FIELDS}
    par=physics.link_params(panel,list(map(str,gs[0].links)));ks=key[key.split==split]
    for i,g in enumerate(gs):
     z=ks[ks.date==g.date].copy()
     if z.empty:continue
     li={str(x):j for j,x in enumerate(g.links)};ti={pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ"):j for j,t in enumerate(g.times)}
     rr=z.link_id.map(li);cc=z.timestamp.map(ti);assert rr.notna().all() and cc.notna().all()
     rr,cc=rr.to_numpy(int),cc.to_numpy(int);assert g.target[rr,cc].all()
     mask=np.zeros_like(g.target);mask[rr,cc]=True;assert mask.sum()==len(z)
     z["flat"]=rr*len(g.times)+cc;z=z.set_index("flat").reindex(np.flatnonzero(mask));assert z.submission_id.notna().all()
     X=TS.feature_table(g,i,lr,hist,W,par,mask=mask);assert list(X)==fs
     khat=np.maximum(aux.predict(X,num_threads=3),0)
     o,c=correct(z,khat,actual,choice);rows.append(o);checks.append(c)
     # Deterministic whole-period sampling includes exact original fallback chain.
     ix=np.unique(np.linspace(0,len(X)-1,min(len(X),64),dtype=int));mr,mc=np.nonzero(mask)
     rebuilt=[fallback_sample(m.predict(X.iloc[ix],num_threads=3),f,g,hist,mr[ix],mc[ix]) for m,f in zip((ms,mq),P1.FIELDS)]
     old=actual.reindex(z.submission_id.iloc[ix]);dv=np.abs(rebuilt[0]-old.speed_kmh.to_numpy());dq=np.abs(rebuilt[1]-old.flow_vph.to_numpy())
     assert max(dv.max(),dq.max())<=.000501
     parity.append(dict(panel=panel,split=split,date=g.date,n=len(ix),max_speed_abs=float(dv.max()),max_flow_abs=float(dq.max())))
    print("SERVING",panel,split,"sec",round(time.time()-tick,1),flush=True)
   assert sum(len(o) for o in rows)==len(key)
  d=pd.concat(rows,ignore_index=True);assert d.submission_id.is_unique
  p=OUT/f"updates_{panel}.parquet";d.to_parquet(p,index=False);paths.append(p)
  receipts.append(dict(panel=panel,alpha=choice["panel_alphas"][panel],rows=len(d),v_cap=sum(c["v_cap"] for c in checks),v_floor=sum(c["v_floor"] for c in checks),q_floor=sum(c["q_floor"] for c in checks),max_density_ratio_error=max(c["density_ratio_error"] for c in checks)))
 # Concatenate the per-panel update files into stateupdates.parquet.
 writer=None;ids=[]
 for p in paths:
  t=pq.read_table(p);ids.append(t.column("submission_id").to_numpy())
  if writer is None:writer=pq.ParquetWriter(OUT/"stateupdates.parquet",t.schema,compression="snappy")
  writer.write_table(t)
 if writer:writer.close()
 allids=np.concatenate(ids);assert len(np.unique(allids))==len(allids)
 pd.DataFrame(parity).to_csv(OUT/"champion_feature_parity.csv",index=False)
 manifest=dict(champion=str(CHAMP),champion_sha256=sha(CHAMP),selected_rule=choice,selection_sha256=sha(SEL),
               old_values_exact_match_champion=True,coverage="Every officialstate target ineach selectedpanel, bothvalidation/private; unchangedpanels omitted",rows=len(allids),checks=receipts,
               formula="Development-selected joint weighted projection to k=(1-alpha)*q_champ/v_champ+alpha*k_aux. Actualroundedchampionv/qinputs; speedclip[6,120.75],flowfloor50.",
               parity_sample_rows=sum(r["n"] for r in parity),parity_max_speed=max((r["max_speed_abs"] for r in parity),default=0),parity_max_flow=max((r["max_flow_abs"] for r in parity),default=0),
               updates_sha256=sha(OUT/"stateupdates.parquet"),sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"scripts/audit_strongest_state_20260922.py",ROOT/"scripts/audit_strongest_state_transfer_20260922.py",S.H.AUX_PATH,ROOT/"src/tfb/task1stack.py",ROOT/"src/tfb/predict1.py",ROOT/"src/tfb/score1.py",ROOT/"src/tfb/baselines.py"]},elapsed_seconds=time.time()-start)
 (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ["rows","checks","updates_sha256","elapsed_seconds"]},indent=2),flush=True)


if __name__=="__main__":main()
