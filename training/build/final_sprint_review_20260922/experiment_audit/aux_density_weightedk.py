"""Trains the k_aux link-density model and compares flow mixes on February 2031.

Training record. The reproduction does not run this program; it loads the shipped model.
The program reads intermediate tables that are not shipped, and its paths resolve from its
original place in the development repository, whose layout `trafficflowbench/` mirrors.

Trains k_aux: a LightGBM model of the density k = flow / max(speed, 1) of a blanked Task 1
cell (veh/km, all lanes). The k* projection of the Task 1 state pulls each cell's flow/speed
ratio toward it.

Data: corridors D7_I10_E and D12_I5_N, train days 2030-06-01 .. 2030-12-20 (no February),
1,000 target cells of the train masks per day: 203,000 rows per corridor, 406,000 in all. The
rows come from the feature caches `state_recency/<corridor>_old{0..6}.parquet`, written by
`state_recency.py` in this folder (not shipped). Inputs: the 71 v3 stack features
(`build/task1_stack/stack_features.csv`). Sample weight: link length (km), normalised to
mean 1, so the absolute-error loss counts vehicles on the link.

Settings: objective regression_l1, learning rate .05, 127 leaves, min_data_in_leaf 200,
feature and bagging fraction .8 (every iteration), seed 20260922, 600 rounds, 4 threads.

Output: `state_recency/aux_density_weightedk/k_l1_weightedL_600.txt`, shipped under
`trafficflowbench/build/final_sprint_review_20260922/experiment_audit/`, and `manifest.json`.
After the fit the program scores February 2031 target cells of the train masks with the
shipped Task 1 state: flow q = (1 - alpha) q_shipped + alpha N_hat v / L for each alpha of
`ALPHAS` (N_hat: predicted vehicles on the link, L: link length), speed unchanged. Alpha is
chosen on February 1-14 by .35 S_state + .15 S_physics and read once on February 15-28
(`daily_metrics.csv`, `results.csv`, `by_panel_regime.csv`, `selection.json`). The k* rule
served in the reproduction is selected by `scripts/audit_strongest_state_transfer_20260922.py`.

Usage:
    python aux_density_weightedk.py
"""
from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
# ROOT: development repository root, three folders above this file's original place.
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.tfb import release as R,grid,physics,predict1 as P1,score1
from threadpoolctl import threadpool_limits
# NumPy/BLAS use 2 threads; LightGBM sets its own thread count below.
threadpool_limits(limits=2)
# SRC: folder of the feature caches; OUT: folder of the model and the February reports.
SRC=Path(__file__).parent/'state_recency';OUT=SRC/'aux_density_weightedk';OUT.mkdir(exist_ok=True)
PANELS=['D7_I10_E','D12_I5_N'];FEATURES=pd.read_csv(ROOT/'build/task1_stack/stack_features.csv').feature.tolist()
# ALPHAS: flow-mix weights compared on February (0 keeps the shipped flow).
ALPHAS=[0.,.1,.25,.5,1.];SEED=20260922;START=time.time()
# The run plan is written before fitting.
manifest={'train':'2030-06-01..2030-12-20','panels':PANELS,'fit_rows':406000,'calibration':'2031-02-01..2031-02-14','held_eval':'2031-02-15..2031-02-28','features':FEATURES,'objective':'regression_l1','target':'k=flow_vph/max(speed_kmh,1); sample weight length_km, normalized; equivalent N absolute-error risk with known geometry applied outside tree','rounds':600,'seed':SEED,'alpha_grid':ALPHAS,'decision':'choose alpha solely by calibration .35*S_state+.15*S_physics; then read held once','composition':'speed fixed to shipped; q=(1-alpha)*q_shipped+alpha*Nhat*v_shipped/length_km','boundary':'all-Feb LRTC features use legal masked observations only; no held labels enter fit or calibration'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
idxs={};parts=[]
for panel in PANELS:
 # Masked and truth file indexes of the train split; link parameters in the grid's link order.
 mi={p.name:(r,p) for r in R.REGIME_ORDER for p in R.masked_files(panel,'train',r)};tm={p.name:p for p in R.truth_files(panel,'train')};idxs[panel]=(mi,tm)
 n='synthetic_mainline_2030_06_01.parquet';g=grid.build(panel,tm[n],mi[n][1]);par=physics.link_params(panel,list(map(str,g.links)))
 # Seven cache parts: 203 days x 1,000 cells, ending 2030-12-20.
 df=pd.concat([pd.read_parquet(SRC/f'{panel}_old{i}.parquet') for i in range(7)],ignore_index=True)
 assert len(df)==203000 and df._date.max()=='2030-12-20'
 # `_flat` is the flat index link * n_times + time, so `_flat // n_times` is the link row.
 # Target `_k`: density (veh/km) = flow / max(speed, 1).
 df['_length']=par['length_km'][df._flat.to_numpy()//len(g.times)];df['_k']=df._flow.to_numpy()/np.clip(df._speed.to_numpy(),1,None);parts.append(df)
fit=pd.concat(parts,ignore_index=True)
# Fit k_aux with link-length weights normalised to mean 1, then save the shipped model.
import lightgbm as lgb
p={'objective':'regression_l1','learning_rate':.05,'num_leaves':127,'min_data_in_leaf':200,'feature_fraction':.8,'bagging_fraction':.8,'bagging_freq':1,'verbose':-1,'seed':SEED,'num_threads':4}
m=lgb.train(p,lgb.Dataset(fit[FEATURES],fit._k,weight=fit._length.to_numpy()/fit._length.mean()),num_boost_round=600);m.save_model(str(OUT/'k_l1_weightedL_600.txt'));del fit,parts
print('fit complete',round(time.time()-START,1),flush=True)
daily=[]
# February check: the shipped Task 1 state (`*_eval_predictions.parquet`) and its stack
# features (`*_feb_eval.parquet`) on the February target cells.
for panel in PANELS:
 ep=pd.read_parquet(SRC/f'{panel}_eval_predictions.parquet');feat=pd.read_parquet(SRC/f'{panel}_feb_eval.parquet')
 assert np.array_equal(ep._flat.to_numpy(),feat._flat.to_numpy()) and ep._date.equals(feat._date)
 # N_aux: predicted vehicles on the link = max(k_hat, 0) x link length.
 mi,tm=idxs[panel];first='synthetic_mainline_2031_02_01.parquet';g0=grid.build(panel,tm[first],mi[first][1]);L0=physics.link_params(panel,list(map(str,g0.links)))['length_km'];nhat=np.maximum(m.predict(feat[FEATURES],num_threads=4),0.)*L0[feat._flat.to_numpy()//len(g0.times)];ep['N_aux']=nhat;ep[['_panel','_date','_flat','N_aux']].to_parquet(OUT/f'{panel}_aux_predictions.parquet',index=False)
 # Ramp detectors attached to each link (LWR transitions need observed ramps).
 mi,tm=idxs[panel];att,ri=physics.ramp_index(panel,'train')
 # Per February day: truth outside the target cells, shipped speed (clipped) inside.
 for date,ix in ep.groupby('_date',sort=True).indices.items():
  z=ep.iloc[ix];name='synthetic_mainline_'+date.replace('-','_')+'.parquet';rg,mp=mi[name];g=grid.build(panel,tm[name],mp);mask=g.target;links=list(map(str,g.links));par=physics.link_params(panel,links);L=par['length_km'][:,None]
  assert np.array_equal(np.flatnonzero(mask),z._flat.to_numpy())
  keep=physics.scoreable_mask(panel,links,'train',g.times);rv=physics.ramp_valid_grid(links,g.times,att,ri)
  vt,qt=g.truth['speed_kmh'],g.truth['flow_vph'];v=vt.copy();v[mask]=np.clip(z.shipped_speed,P1.SPEED_FLOOR,P1.SPEED_CAP);baseq=np.maximum(z.shipped_flow.to_numpy(),P1.FLOW_FLOOR)
  len_target=np.broadcast_to(L,mask.shape)[mask];auxq=z.N_aux.to_numpy()*v[mask]/len_target
  # True vehicle count per cell: N = q / v x L.
  Nt=qt/np.clip(vt,1,None)*L
  # Per alpha: mixed flow at the target cells. LWR term: |change of N between consecutive
  # steps - true change| over valid transitions (finite, scoreable link, ramps observed).
  # FD term: |per-lane flow - triangular fundamental-diagram flow at the per-lane density|.
  for alpha in ALPHAS:
   q=qt.copy();q[mask]=np.maximum((1-alpha)*baseq+alpha*auxq,P1.FLOW_FLOOR)
   N=q/np.clip(v,1,None)*L;finite=np.isfinite(N)&np.isfinite(Nt)&keep[:,None];valid=finite[:,:-1]&finite[:,1:]&rv[:,:-1];touch=valid&(mask[:,:-1]|mask[:,1:])
   de=np.abs(np.diff(N,axis=1)-np.diff(Nt,axis=1));dt=np.abs(np.diff(Nt,axis=1))
   lanes=np.maximum(par['lanes'],1)[:,None];ql=q/lanes;kl=q/np.clip(v,1,None)/lanes;cap=par['capacity_vph'][:,None]/lanes;kc=par['critical_density'][:,None]/lanes;kj=par['k_jam'][:,None]/lanes
   qfd=np.where(kl<=kc,par['free_speed_kmh'][:,None]*kl,cap/np.maximum(kj-kc,1e-9)*np.maximum(kj-kl,0));ok=np.isfinite(ql)&np.isfinite(qfd)
   daily.append({'alpha':alpha,'panel':panel,'regime':rg,'date':date,'split':'cal' if date<='2031-02-14' else 'held','n':int(mask.sum()),'speed_sse':float(np.square((v-vt)[mask]).sum()),'flow_lane_sse':float(np.square((q-qt)[mask]/z._lane.to_numpy()).sum()),'N_abs_err':float(np.abs(N-Nt)[mask].sum()),'lwr_num':float(de[valid].sum()),'lwr_den':float(dt[valid].sum()),'n_transitions':int(valid.sum()),'n_touching_transitions':int(touch.sum()),'fd_num':float(np.abs(ql[ok]-qfd[ok]).sum()),'fd_den':float(np.abs(ql[ok]).sum()),'n_empty_target':int((q[mask]<physics.EMPTY_FLOOR_VPH).sum())})
 print('evaluated',panel,round(time.time()-START,1),flush=True)
d=pd.DataFrame(daily);d.to_csv(OUT/'daily_metrics.csv',index=False);by=[];res=[]
# Scores per split (cal = Feb 1-14, held = Feb 15-28) and alpha, per corridor and regime:
# S_state from speed and per-lane flow RMSE; S_physics = (S_FD + 2 S_LWR) / 3; S_FD is 0
# when too many target flows are near zero. local_LB_part = .35 S_state + .15 S_physics.
for (split,alpha),dd in d.groupby(['split','alpha']):
 g=dd.groupby(['panel','regime']).sum(numeric_only=True)
 g['S_state']=.54*np.maximum(0,1-np.sqrt(g.speed_sse/g.n)/25)+.46*np.maximum(0,1-np.sqrt(g.flow_lane_sse/g.n)/600)
 g['S_LWR']=np.maximum(0,1-np.minimum(1,g.lwr_num/g.lwr_den));g['S_FD']=np.maximum(0,1-g.fd_num/(g.fd_den+1e-9));g.loc[g.n_empty_target/g.n>physics.EMPTY_SHARE,'S_FD']=0
 g['S_physics']=(g.S_FD+2*g.S_LWR)/3
 r={'split':split,'alpha':alpha,**{k:float(g.groupby('panel')[k].mean().mean()) for k in ['S_state','S_LWR','S_FD','S_physics']},'N_MAE_target':float(dd.N_abs_err.sum()/dd.n.sum()),'N_delta_MAE_valid':float(dd.lwr_num.sum()/dd.n_transitions.sum()),'N_delta_MAE_touching':float(dd.lwr_num.sum()/dd.n_touching_transitions.sum())};r['local_LB_part']=.35*r['S_state']+.15*r['S_physics'];res.append(r)
 g['split']=split;g['alpha']=alpha;by.append(g.reset_index())
t=pd.DataFrame(res);t.to_csv(OUT/'results.csv',index=False);pd.concat(by).to_csv(OUT/'by_panel_regime.csv',index=False)
# Alpha with the best calibration score (smaller alpha on ties); held-out change against alpha 0.
cal=t[t.split=='cal'].sort_values(['local_LB_part','alpha'],ascending=[False,True]);a=float(cal.iloc[0].alpha)
held=t[(t.split=='held')&(t.alpha==a)].iloc[0];base=t[(t.split=='held')&(t.alpha==0)].iloc[0]
selection={'chosen_alpha':a,'criterion':'calibration local_LB_part only','held_delta_local_LB_part':float(held.local_LB_part-base.local_LB_part),'held_delta_S_state':float(held.S_state-base.S_state),'held_delta_S_physics':float(held.S_physics-base.S_physics),'elapsed_seconds':time.time()-START}
(OUT/'selection.json').write_text(json.dumps(selection,indent=2));print(t.to_string(index=False),flush=True);print(json.dumps(selection,indent=2),flush=True)
