"""Builds the auxiliary density rows (k_aux) that the k* state projection reads.

Task 1 reproduction pipeline, step 2 (`pipeline/task1.sh`, log `kaux_first`).
For every official Task 1 state target on the panels D7_I10_E and D12_I5_N, in
the validation and private months, the script rebuilds the v3 stack feature
table from the masked release (LRTC-TNN completion, interpolations, historical
mean, link parameters) and predicts a traffic density with a LightGBM model
(`k_l1_weightedL_600.txt`). k* is the joint speed/flow projection onto a
blended density; its builder
(`scripts/build_strongest_state_transfer_serving_20260922.py`) reads these rows
for the two panels (speed_kmh, old_flow_vph, aux_density_vpkm) and computes the
other k* panels itself.

Inputs:
  * `submission_key.csv` of the release: the state rows of the two panels.
  * `build/onset_condition_candidate_20260922/submission_onset_endpoint_top1.csv`:
    the v3 stack state (speed_kmh, flow_vph per submission_id). In the
    reproduction this is the state-only stand-in written by
    `pipeline/task1_state_csv.py`.
  * The density model and the stack feature list
    `build/task1_stack/stack_features.csv`.

Outputs (folder `state_recency/aux_density_weightedk/`):
  * `serving_updates.parquet`: one row per target with submission_id, panel,
    split, timestamp, station_id, link_id, mask_regime, speed_kmh (v3 speed,
    unchanged), old_flow_vph (v3 flow), new_flow_vph
    (max(0.9 * old flow + 0.1 * density * speed, FLOW_FLOOR)),
    aux_density_vpkm (predicted density in vehicles per km, clipped at 0) and
    alpha (0.1).
  * `serving_parity_samples.parquet`: 256 random targets per day on which the
    shipped stack speed and flow models are re-run and compared with the v3
    CSV values.
  * `serving_manifest.json`: SHA256 of the inputs, sources and output, row
    counts and the parity summary.

The script does its work at import and takes no arguments.
"""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.tfb import release as R,grid,physics,predict1 as P1,task1stack as TS,baselines as B
from threadpoolctl import threadpool_limits
threadpool_limits(limits=2)
# Output folder (it also holds the density model), the two panels served here,
# and ALPHA, the weight of the predicted density in new_flow_vph.
OUT=Path(__file__).parent/'state_recency/aux_density_weightedk';PANELS=['D7_I10_E','D12_I5_N'];ALPHA=.1
BASE=ROOT/'build/onset_condition_candidate_20260922/submission_onset_endpoint_top1.csv';MODEL=OUT/'k_l1_weightedL_600.txt';FEATURES=pd.read_csv(ROOT/'build/task1_stack/stack_features.csv').feature.tolist();START=time.time()
def sha(p):
 """Returns the SHA256 hex digest of file `p`, read in 8 MB chunks."""
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
# Official Task 1 state targets of the two panels, read from the submission key.
kc=['submission_id','task','split','panel','timestamp','station_id','link_id','mask_regime'];keys=[]
for c in pd.read_csv(R.RELEASE/'submission_key.csv',usecols=kc,chunksize=400000,dtype={k:str for k in kc if k!='submission_id'}):
 keys.append(c[(c.task=='state')&c.panel.isin(PANELS)])
key=pd.concat(keys,ignore_index=True);assert not key.submission_id.duplicated().any();ids=pd.Index(key.submission_id)
# Attach the v3 speed and flow of every target (each target must have both);
# `_date` is the target's calendar day, used to split the targets by masked day.
base=[]
for c in pd.read_csv(BASE,usecols=['submission_id','task','speed_kmh','flow_vph'],chunksize=400000):base.append(c[c.submission_id.isin(ids)])
base=pd.concat(base,ignore_index=True);assert not base.submission_id.duplicated().any();key=key.merge(base,on=['submission_id','task'],how='left',validate='one_to_one');assert key[['speed_kmh','flow_vph']].notna().all().all();key['_date']=key.timestamp.str[:10]
print('official targets',len(key),'champion read',round(time.time()-START,1),flush=True)
import lightgbm as lgb
# The density model must use exactly the shipped stack feature order. ms and mq
# are the v3 stack speed and flow models; they serve only the parity check.
aux=lgb.Booster(model_file=str(MODEL));ms,mq,shipfeatures=TS.load();assert FEATURES==shipfeatures==aux.feature_name()
# W: blend weights of the five classical fills; rng picks the parity rows.
W=P1.load_weights();updates=[];parity=[];counts=[];rng=np.random.default_rng(20260922)
for panel in PANELS:
 # Link x weekday x time-of-day mean over the dev-fold training days.
 hist=P1.historical_mean(panel,P1.dev_train_files(panel))
 for split in ['validation','private']:
  # All masked days of the split, and an LRTC-TNN low-rank completion of the
  # link x time x day tensor per field (blanked cells enter as 0): the same
  # inputs as the v3 stack, so the feature table matches the shipped one.
  tick=time.time();paths=P1.split_masked_files(panel,split);gs=[grid.build_masked(panel,p) for p in paths]
  lr={f:B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f],nan=0.) for g in gs],axis=2),rho=P1.LRTC_KNOBS[f][0],theta=P1.LRTC_KNOBS[f][1]) for f in P1.FIELDS}
  # Per-link length, free-flow speed, capacity and lanes; the split's targets.
  par=physics.link_params(panel,list(map(str,gs[0].links)));ks=key[(key.panel==panel)&(key.split==split)]
  print('prepared',panel,split,len(gs),'days',round(time.time()-tick,1),flush=True)
  nn=0
  for di,g in enumerate(gs):
   # Construct exactly the official target mask, which is a subset of eligible+blanked.
   kd=ks[ks._date==str(g.date)].copy()
   if kd.empty:continue
   li={str(l):i for i,l in enumerate(g.links)};ti={pd.Timestamp(t).strftime('%Y-%m-%dT%H:%M:%SZ'):i for i,t in enumerate(g.times)}
   rr=kd.link_id.map(li);cc=kd.timestamp.map(ti);assert rr.notna().all() and cc.notna().all()
   rr,cc=rr.to_numpy(int),cc.to_numpy(int);assert g.target[rr,cc].all()
   mask=np.zeros_like(g.target);mask[rr,cc]=True;assert int(mask.sum())==len(kd)
   # One feature row per target cell, in row-major order of the mask; `axes`
   # maps each row back to its submission_id through the key columns.
   X=TS.feature_table(g,di,lr,hist,W,par,mask=mask)[FEATURES];r,c=np.nonzero(mask)
   axes=pd.DataFrame({'panel':panel,'split':split,'timestamp':[pd.Timestamp(g.times[t]).strftime('%Y-%m-%dT%H:%M:%SZ') for t in c],'station_id':np.array(g.station,dtype=object)[r].astype(str),'link_id':np.array(g.links,dtype=object)[r].astype(str),'_Xrow':np.arange(len(X))})
   join=['panel','split','timestamp','station_id','link_id'];axes=axes.merge(kd,on=join,how='left',validate='one_to_one').sort_values('_Xrow');assert axes.submission_id.notna().all()
   # Predicted density (vehicles per km, at least 0). new_flow_vph moves the v3
   # flow 10% toward density x v3 speed (km/h), the flow that density implies.
   khat=np.maximum(aux.predict(X,num_threads=4),0.)
   oldv=axes.speed_kmh.to_numpy(float);oldq=axes.flow_vph.to_numpy(float);newq=np.maximum(.9*oldq+.1*khat*oldv,P1.FLOW_FLOOR)
   assert np.isfinite(newq).all()
   u=axes[['submission_id','panel','split','timestamp','station_id','link_id','mask_regime']].copy();u['speed_kmh']=oldv;u['old_flow_vph']=oldq;u['new_flow_vph']=newq;u['aux_density_vpkm']=khat;u['alpha']=ALPHA;updates.append(u);nn+=len(u)
   # Full feature implementation is shared with 112/113. Verify old model outputs
   # against the champion's actual rounded values on 256 fixed random targets/day.
   ix=np.sort(rng.choice(len(X),size=min(256,len(X)),replace=False));sv=np.clip(ms.predict(X.iloc[ix],num_threads=4),P1.SPEED_FLOOR,P1.SPEED_CAP);sq=np.maximum(mq.predict(X.iloc[ix],num_threads=4),P1.FLOW_FLOOR)
   for row,xv,xq in zip(ix,sv,sq):parity.append({'submission_id':int(axes.iloc[row].submission_id),'panel':panel,'split':split,'date':str(g.date),'speed_rebuilt':float(xv),'speed_champion':float(oldv[row]),'flow_rebuilt':float(xq),'flow_champion':float(oldq[row])})
  assert nn==len(ks);counts.append({'panel':panel,'split':split,'days':len(gs),'targets':nn,'seconds':time.time()-tick});print('completed',counts[-1],flush=True)
# Every target gets exactly one row; the parity sample must agree with the v3
# CSV within its 3-decimal rounding (0.000501) before the updates are written.
u=pd.concat(updates,ignore_index=True).sort_values('submission_id');p=pd.DataFrame(parity);p['speed_abs_diff']=np.abs(p.speed_rebuilt-p.speed_champion);p['flow_abs_diff']=np.abs(p.flow_rebuilt-p.flow_champion);p.to_parquet(OUT/'serving_parity_samples.parquet',index=False)
assert not u.submission_id.duplicated().any();missing=len(ids.difference(pd.Index(u.submission_id)));extra=len(pd.Index(u.submission_id).difference(ids));assert missing==extra==0
checks={'sample_rows':len(p),'speed_max_abs_diff':float(p.speed_abs_diff.max()),'flow_max_abs_diff':float(p.flow_abs_diff.max()),'speed_over_rounding_tolerance':int((p.speed_abs_diff>.000501).sum()),'flow_over_rounding_tolerance':int((p.flow_abs_diff>.000501).sum()),'rounding_tolerance':.000501}
manifest={'base':str(BASE),'base_sha256':sha(BASE),'model':str(MODEL),'model_sha256':sha(MODEL),'official_key':str(R.RELEASE/'submission_key.csv'),'official_key_sha256':sha(R.RELEASE/'submission_key.csv'),'alpha':ALPHA,'formula':'max(.9*champion_flow + .1*aux_density*champion_speed, FLOW_FLOOR)','speed_source':'actual champion compact values, unchanged','rows':len(u),'missing_keys':missing,'extra_keys':extra,'counts':counts,'features':FEATURES,'parity':checks,'elapsed_seconds':time.time()-START,'sources':{str(q.relative_to(ROOT)):sha(q) for q in [Path(__file__),ROOT/'src/tfb/task1stack.py',ROOT/'src/tfb/baselines.py',ROOT/'src/tfb/predict1.py',ROOT/'scripts/112_build_submission_v3.py']}}
assert checks['speed_over_rounding_tolerance']==checks['flow_over_rounding_tolerance']==0,checks
u.to_parquet(OUT/'serving_updates.parquet',index=False);manifest['updates_sha256']=sha(OUT/'serving_updates.parquet');(OUT/'serving_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['rows','missing_keys','extra_keys','parity','elapsed_seconds','updates_sha256']},indent=2),flush=True)
