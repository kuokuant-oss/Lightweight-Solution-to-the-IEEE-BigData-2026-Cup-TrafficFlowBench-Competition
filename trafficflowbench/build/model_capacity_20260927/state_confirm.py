"""Frozen January selection tested on February full DK50 background."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='3'
import sys,json,gc,time
import numpy as np,pandas as pd,lightgbm as lgb
from common import HERE,ROOT,dump,state_alpha
sys.path.insert(0,str(HERE/'packages'));sys.path.insert(0,str(ROOT/'build/review_20260927_structural'))
import density_experiment as D
OUT=HERE/'state';CONF=OUT/'confirmation';CONF.mkdir(exist_ok=True)

def predict(folder,choice,d,y,production=False):
    if 'head_sources'in choice:folder=OUT/(choice['head_sources'][y]['name']+('_final'if production else'_refit'))
    trial=json.loads((folder/'trial.json').read_text());family=trial['family'];fs=json.loads((OUT/'data.json').read_text())['features']
    if family=='lgb':return lgb.Booster(model_file=str(folder/f'{y}_chosen.txt')).predict(d[fs],num_threads=3)
    if family=='cat':
        from catboost import CatBoostRegressor
        m=CatBoostRegressor();m.load_model(str(folder/f'{y}.cbm'));return m.predict(d[fs],thread_count=3)
    import xgboost as xgb
    m=xgb.Booster();m.load_model(folder/f'{y}.ubj');return m.predict(xgb.DMatrix(d[fs].to_numpy('float32')))

def run():
    t=time.time();choice=json.loads((OUT/'selection.json').read_text());folder=OUT/(choice['name']+'_refit');parts=[];cparts=[]
    for pi,p in enumerate(D.E.PANELS10):
        d=pd.read_parquet(D.MATCH.OUT/f'{p}_features.parquet');parts.append(d)
        if pi<8:rv=predict(folder,choice,d,'y_v');rq=predict(folder,choice,d,'y_q')
        else:
            a=np.load(D.MATCH.OUT/f'{p}_parent.npz');rv=a['y_v'];rq=a['y_q']
        cparts.append((np.clip(d.p_v.to_numpy()+rv,6,120.75),np.maximum(d.p_q.to_numpy()+rq,50/d.lanes.to_numpy())))
        print('February candidate predictions',p,len(d),flush=True)
    d=pd.concat(parts,ignore_index=True);del parts;gc.collect()
    rawv,rawq=D.raw_parent(d,'2031_02');hv,hq=D.parent_kstar(d,rawv,rawq,'2031_02')
    kh=np.load(D.HERE/'density/feb_density_predictions.npz')['l2_uniform']
    pv,pq=hv.copy(),hq.copy()
    for pi,p in enumerate(D.E.PANELS10[:8]):
        ix=np.flatnonzero(d.panel_i.to_numpy()==pi);ln=d.lanes.to_numpy()[ix]
        vv,qq=D.project(hv[ix],hq[ix],kh[ix],ln,p,.5);pv[ix]=np.round(vv,3);pq[ix]=np.round(qq,3)/ln
    nv,nq=D.parent_kstar(d,np.concatenate([a[0]for a in cparts]),np.concatenate([a[1]for a in cparts]),'2031_02')
    alpha=state_alpha(choice['alpha'],d.p_r.to_numpy());cv=np.round(np.clip(pv+alpha*(nv-hv),6,120.75),3);ln=d.lanes.to_numpy();cq=np.round(np.maximum((pq+alpha*(nq-hq))*ln,50),3)/ln
    np.savez(CONF/'predictions.npz',parent_v=pv,parent_q=pq,candidate_v=cv,candidate_q=cq)
    D.OUT=CONF;a,res=D.evaluate(d,{'parent':(pv,pq),'candidate':(cv,cq)});D.uncertainty(a)
    old=json.loads((D.HERE/'density/feb_results.json').read_text());parity={}
    for mode in ['strict','serve']:
        vals=res[mode]['levels']['parent'];ref=old[mode]['levels']['l2_uniform_0.5'];dif={k:float(vals[k]-ref[k])for k in vals};assert max(abs(v)for v in dif.values())<1e-10;parity[mode]=dif
    report=dict(selection=choice,full_DK50_parent_parity=parity,results=res,uncertainty=json.loads((CONF/'feb_uncertainty.json').read_text()),seconds=time.time()-t)
    dump(CONF/'report.json',report)
    print('FEBRUARY CONFIRMATION',json.dumps(report,indent=2),flush=True)

if __name__=='__main__':run()
