"""Predeclared ongoing forward experiments; does not submit or touch old artifacts."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='2'
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
from scipy.special import expit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.tfb import release as R,task2oof as O,task2prior as PR,task2eval as E,task2build as TB,task2infer as TI
OUT=ROOT/'build/strongest_candidate_20260922/ongoing'
KEY=['window_id','panel','date','condition','link_id','step','label','_eligible']
NEW_PARAMS=dict(objective='binary',metric='binary_logloss',num_leaves=127,learning_rate=.05,min_data_in_leaf=100,feature_fraction=.9,bagging_fraction=.8,bagging_freq=1,lambda_l1=0.,lambda_l2=10.,max_bin=127,seed=20260922,num_threads=2,verbose=-1)
RECIPES=['champ_refit_fixed','champ_refit_cal','champ_stageA_cal','plain50_stageA_cal','origin_net_stageA_cal','ensemble_stageA_cal']
GRID=np.round(np.arange(.10,.901,.025),3)
FOLDS=[('oct','2030-08-15','2030-08-16','2030-08-31','2030-09-01','2030-09-30','2030-10-01','2030-10-31'),('nov','2030-09-15','2030-09-16','2030-09-30','2030-10-01','2030-10-31','2030-11-01','2030-11-30'),('dec','2030-10-15','2030-10-16','2030-10-31','2030-11-01','2030-11-30','2030-12-01','2031-01-04')]
NET=['origin_on_ramp_T','origin_off_ramp_T','origin_net_ramp_T','origin_dnet_ramp']

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def windows():
 w=pd.read_parquet(ROOT/'data/interim/score_sprint_20260921/recency_chain_windows.parquet');return w[w.condition.eq('queue_ongoing')].copy()
def all_dates():return sorted({p.stem.replace('synthetic_mainline_','').replace('_','-')for p in R.truth_files('D7_I10_E')})
def identities(d):
 panels=sorted(R.panels());mapping={};i=0
 for p in panels:
  links=pd.read_parquet(R.truth_files(p)[0],columns=['link_id']).link_id.astype(str).unique()
  for link in sorted(links):mapping[p,str(link)]=i;i+=1
 d['panel_code']=d.panel.map({p:i for i,p in enumerate(panels)}).astype('int32');d['link_code']=[mapping[p,str(l)]for p,l in zip(d.panel,d.link_id)];d['link_code']=d.link_code.astype('int32')
 return d

def net_columns(d,w,split='train'):
 """Only observations at forecast origin T and T-5. Attached OR and FR separated."""
 lookup=w.set_index('window_id');out={c:np.full(len(d),np.nan,dtype='float32')for c in NET}
 for (p,date),ii in d.groupby(['panel','date'],sort=False).indices.items():
  attach=R.network(p,'ramp_attachment_map');attach=attach[['ramp_link_id','nearest_mainline_link_id','ramp_type']].copy();attach['ramp_link_id']=attach.ramp_link_id.astype(str);attach['nearest_mainline_link_id']=attach.nearest_mainline_link_id.astype(str)
  ramp=TB.ramp_day(p,'synthetic_mainline_'+str(date).replace('-','_')+'.parquet',split=split).merge(attach,on='ramp_link_id',how='inner')
  if not len(ramp):continue
  ramp['_stamp']=pd.to_datetime(ramp.timestamp,utc=True);ramp['_type']=ramp.ramp_type.astype(str).str.upper().map(lambda x:'on' if x in ('OR','ON') else 'off')
  sums=ramp.groupby(['_stamp','nearest_mainline_link_id','_type']).flow_vph.sum(min_count=1)
  attached={kind:set(attach.loc[attach.ramp_type.astype(str).str.upper().isin(['OR','ON'])==(kind=='on'),'nearest_mainline_link_id'])for kind in ('on','off')}
  sub=d.iloc[ii]
  for wid,jj in sub.groupby('window_id',sort=False).indices.items():
   idx=ii[jj];r=lookup.loc[wid]
   if 'forecast_origin' in lookup.columns:T=pd.Timestamp(r.forecast_origin);T=T.tz_localize('UTC')if T.tzinfo is None else T.tz_convert('UTC')
   else:T=pd.Timestamp(str(date),tz='UTC')+pd.Timedelta(minutes=5*int(r.origin))
   vals={}
   links=d.link_id.to_numpy()[idx]
   for kind in ('on','off'):
    for lag in (0,1):
     ts=T-pd.Timedelta(minutes=5*lag)
     vals[kind,lag]=np.asarray([sums.get((ts,str(l),kind),np.nan if str(l)in attached[kind]else 0.)for l in links],dtype='float32')
   out[NET[0]][idx]=vals['on',0];out[NET[1]][idx]=vals['off',0]
   out[NET[2]][idx]=vals['on',0]-vals['off',0]
   out[NET[3]][idx]=(vals['on',0]-vals['off',0])-(vals['on',1]-vals['off',1])
 for c,v in out.items():d[c]=v
 return d

def features(w,name):
 path=OUT/f'{name}_full_features.parquet'
 if path.exists():return pd.read_parquet(path)
 parts=[]
 for panel,wp in w.groupby('panel'):
  tick=time.time();f=O.base_features(wp[['panel','date','origin','condition']],oblique=True,dispersion=True,clock=True,origin_obs=True,include_ineligible=True,split='train')
  f=net_columns(f,wp);f.attrs={};parts.append(f);print('FEATURES',name,panel,len(f),round(time.time()-tick,1),flush=True)
 d=pd.concat(parts,ignore_index=True);identities(d);d.attrs={};d.to_parquet(path,index=False);return d

def priors(ds,dates,panels):
 for d in ds:d['queue_prior']=np.float32(0)
 for p in panels:
  pr=PR.build([p],PR.PriorSpec(tuple(dates)))
  for d in ds:
   ii=d.panel.eq(p)
   if ii.any():d.loc[ii,'queue_prior']=PR.apply(d.loc[ii],pr)
  PR.clear_cache();del pr;gc.collect()

def predict(m,d,fs,offset):
 raw=m.predict(d[fs],raw_score=True,num_threads=2)
 if offset:raw+=O.logit(d.queue_prior.to_numpy())
 return expit(raw)
def train(d,es,fs,offset,kind):
 kw={'init_score':O.logit(d.queue_prior.to_numpy())}if offset else {};ve={'init_score':O.logit(es.queue_prior.to_numpy())}if offset else {}
 cat=[x for x in ('panel_code','link_code')if x in fs]
 dt=lgb.Dataset(d[fs],d.label,weight=O.window_weights(d),categorical_feature=cat,**kw);dv=lgb.Dataset(es[fs],es.label,weight=O.window_weights(es),categorical_feature=cat,reference=dt,**ve)
 m=lgb.train(dict(O.PARAMS,num_threads=2)if kind=='champ' else NEW_PARAMS,dt,num_boost_round=O.NUM_BOOST_ROUND if kind=='champ' else 1000,valid_sets=[dv],callbacks=[lgb.early_stopping(O.EARLY_STOPPING if kind=='champ' else 80,verbose=False),lgb.log_evaluation(0)])
 del dt,dv;gc.collect();return m

def frame(d,prob,threshold):return E.iou_frame(d[d._eligible].reset_index(drop=True),(np.asarray(prob)>=threshold)[d._eligible.to_numpy()])
def score(frame,panels):
 g=frame.groupby('panel').IoU.mean();assert set(panels)<=set(g.index);return float(g.loc[panels].mean())
def calibrate(d,p):
 cov=d[['window_id','panel','date']].drop_duplicates().groupby('panel').agg(windows=('window_id','size'),dates=('date','nunique'));panels=cov.query('windows>=10 and dates>=5').index.tolist();assert len(panels)>=6
 rows=[dict(threshold=float(t),score=score(frame(d,p,t),panels))for t in GRID];best=max(rows,key=lambda x:x['score']);return best['threshold'],rows,panels

def run_case(d,case,fold,dates):
 tag,fitend,esstart,esend,calstart,calend,evalstart,evalend=case;folder=OUT/fold;folder.mkdir(exist_ok=True)
 tr=d[d.date.le(fitend)&d._eligible].copy();es=d[d.date.between(esstart,esend)&d._eligible].copy();ca=d[d.date.between(calstart,calend)].copy();ev=d[d.date.between(evalstart,evalend)].copy()
 assert max(tr.date)<min(es.date)<=max(es.date)<min(ca.date)<=max(ca.date)<min(ev.date)
 fs50=json.loads((ROOT/'data/interim/m21_task2_recency_corrected/prior_manifest.json').read_text())['features'];fsnew=[c for c in d if c not in O.DROP and not c.startswith('_')]
 assert len(fs50)==50 and len(fsnew)==76,(len(fs50),len(fsnew),fsnew)
 fitdates=[x for x in dates if x<=fitend];refitdates=[x for x in dates if x<=calend]
 panels=sorted(d.panel.unique());priors([tr,es,ca,ev],fitdates,panels)
 results=[];probs={};models={};man={'case':case,'fit_dates':fitdates,'refit_dates':refitdates,'rows':{'fit':len(tr),'es':len(es),'cal':len(ca),'eval':len(ev)},'models':{}}
 for kind,fs,offset in [('champ',fs50,True),('plain50',fs50,False),('origin_net',fsnew,False)]:
  tick=time.time();m=train(tr,es,fs,offset,kind);m.save_model(str(folder/f'{kind}_stageA.txt'));cp,ep=predict(m,ca,fs,offset),predict(m,ev,fs,offset);thr,grid,calpanels=calibrate(ca,cp)
  models[kind]=m;probs[kind]=(cp,ep);name=kind+'_stageA_cal'
  results.append((name,ep,thr));man['models'][kind]={'features':fs,'offset':offset,'trees':m.current_iteration(),'threshold':thr,'grid':grid,'calibration_panels':calpanels,'seconds':time.time()-tick}
  print('FIT',tag,kind,m.current_iteration(),thr,round(time.time()-tick,1),flush=True)
 # Refit the original offset model: evaluate fixed deployed threshold and stageA-calibrated threshold.
 ref=d[d.date.le(calend)&d._eligible].copy();evb=ev.copy();priors([ref,evb],refitdates,panels)
 dt=lgb.Dataset(ref[fs50],ref.label,weight=O.window_weights(ref),init_score=O.logit(ref.queue_prior.to_numpy()))
 mb=lgb.train(dict(O.PARAMS,num_threads=2),dt,num_boost_round=models['champ'].current_iteration());mb.save_model(str(folder/'champ_refit.txt'));bp=predict(mb,evb,fs50,True)
 results.extend([('champ_refit_fixed',bp,.675),('champ_refit_cal',bp,man['models']['champ']['threshold'])])
 mixcp=.5*probs['champ'][0]+.5*probs['origin_net'][0];mixep=.5*probs['champ'][1]+.5*probs['origin_net'][1];thr,grid,calpanels=calibrate(ca,mixcp)
 results.append(('ensemble_stageA_cal',mixep,thr));man['models']['ensemble']={'threshold':thr,'grid':grid,'calibration_panels':calpanels,'probability_weights':{'champ':.5,'origin_net':.5}}
 records=[];frames=[]
 for name,ep,threshold in results:
  f=frame(ev,ep,threshold);f['recipe']=name;frames.append(f)
  ef=ev[KEY].copy();ef['prob']=ep;ef['pred']=(ep>=threshold).astype('int8');ef.to_parquet(folder/f'{name}_eval_predictions.parquet',index=False)
  pc=f.groupby('panel').IoU.mean().to_dict();dense=[p for p in sorted(pc)if p!='D7_I10_W'];rec={'fold':tag,'recipe':name,'threshold':threshold,'dense7_score':float(np.mean([pc[p]for p in dense])),'available_panel_score':E.s_queue(f),'panel_scores':pc,'windows':len(f),'dates':f.date.nunique()};records.append(rec)
 pd.concat(frames,ignore_index=True).to_parquet(folder/'window_results.parquet',index=False);dump(folder/'manifest.json',man);dump(folder/'results.json',records)
 del tr,es,ca,ev,ref,evb,models,dt,mb;gc.collect();return records

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',default='dev',choices=['dev']);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 w=windows();dates=all_dates();dev=w[w.date.le('2031-01-04')]
 design={'created_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'role':'Ongoing independent development-only selection; immutable formerholdout artifacts','folds':FOLDS,'recipes':RECIPES,'new_model_params':NEW_PARAMS,'calibration_grid':GRID.tolist(),'calibration_panel_rule':'atleast10windowsand5independentdates; eligibilityonlyscoring','selection_panel_rule':'fixedsevenpanels excludingD7_I10_W which hasonly2developmentwindows; officialavailablepanelmetricdiagnostic','selection_rule':'againstchamp_refit_fixed, requirepositiveDeltaonatleast2of3folds andworstfoldDelta>=-.02 andmeanDelta>.002; selecthighestmean dense7score amongqualifiers, elsekeepchamp. Neverchooseonholdout.','net_features':NET,'net_feature_source':'releasedrampobservationsatTandT-5, OR/ON entering, FR/OFF exiting, nofuture','feature_source':'fullphysicalgrid withfutureeligibleonlymetadata; originTmaskedmainline','source_sha256':sha(__file__),'n_dev_ongoing_windows':len(dev)}
 if (OUT/'design.json').exists():
  old=json.loads((OUT/'design.json').read_text());assert old['source_sha256']==design['source_sha256'],'Sourcechangedafterdesign; explicitlyrecordamendment'
 else:dump(OUT/'design.json',design)
 d=features(dev,'development');allresults=[]
 for case in FOLDS:
  f=OUT/case[0]/'results.json'
  if f.exists():res=json.loads(f.read_text())
  else:res=run_case(d,case,case[0],dates)
  allresults.extend(res);dump(OUT/'development_results.json',allresults)
 t=pd.DataFrame(allresults);wide=t.pivot(index='fold',columns='recipe',values='dense7_score');deltas=wide.subtract(wide['champ_refit_fixed'],axis=0);summary=pd.DataFrame({'mean_score':wide.mean(),'mean_delta':deltas.mean(),'positive_folds':(deltas>0).sum(),'worst_fold_delta':deltas.min()});summary['qualifies']=(summary.mean_delta>.002)&(summary.positive_folds>=2)&(summary.worst_fold_delta>=-.02)
 qualifiers=summary[summary.qualifies].sort_values('mean_score',ascending=False);choice=qualifiers.index[0]if len(qualifiers)else'champ_refit_fixed';summary.to_csv(OUT/'development_recipe_comparison.csv');wide.to_csv(OUT/'development_fold_scores.csv')
 dump(OUT/'selection_before_holdout.json',{'created_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'selected_recipe':choice,'rule':design['selection_rule'],'summary':summary.reset_index().to_dict('records'),'held_dates_not_used_in_this_round_selection':True,'prior_research_already_saw_JanFeb':True});print('SELECTION',choice,summary.to_string(),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=2):main()
