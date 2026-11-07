"""Frozen January selection tested on February full DK50 background."""
# Record of the CAP February confirmation (training months only). run() needs
# the January-fit `_refit` heads and the February feature tables, which are not
# shipped, so it is not run in the reproduction. Its output
# state/confirmation/report.json is the gate that build_state_candidate.py
# checks; the reproduction imports `predict` only.
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='3'
import sys,json,gc,time
import numpy as np,pandas as pd,lightgbm as lgb
from common import HERE,ROOT,dump,state_alpha
sys.path.insert(0,str(HERE/'packages'));sys.path.insert(0,str(ROOT/'build/review_20260927_structural'))
import density_experiment as D
OUT=HERE/'state';CONF=OUT/'confirmation';CONF.mkdir(exist_ok=True)

def predict(folder,choice,d,y,production=False):
    """Predicts one residual head of the chosen state model.

    The model family (LightGBM, CatBoost or XGBoost) is read from the folder's
    trial.json and the feature list from state/data.json.

    Args:
      folder: model folder, state/<name>_final or state/<name>_refit.
      choice: the selection dict (state/selection.json); an optional
        'head_sources' entry names a separate folder per head.
      d: feature table with the columns listed in state/data.json.
      y: 'y_v' (speed residual, km/h) or 'y_q' (flow residual per lane, veh/h).
      production: with 'head_sources', picks the `_final` folder instead of
        the `_refit` one.

    Returns:
      1-D array of residual predictions, one per row of `d`.
    """
    if 'head_sources'in choice:folder=OUT/(choice['head_sources'][y]['name']+('_final'if production else'_refit'))
    trial=json.loads((folder/'trial.json').read_text());family=trial['family'];fs=json.loads((OUT/'data.json').read_text())['features']
    if family=='lgb':return lgb.Booster(model_file=str(folder/f'{y}_chosen.txt')).predict(d[fs],num_threads=3)
    if family=='cat':
        from catboost import CatBoostRegressor
        m=CatBoostRegressor();m.load_model(str(folder/f'{y}.cbm'));return m.predict(d[fs],thread_count=3)
    import xgboost as xgb
    m=xgb.Booster();m.load_model(folder/f'{y}.ubj');return m.predict(xgb.DMatrix(d[fs].to_numpy('float32')))

def run():
    """Scores the frozen January selection on February and writes report.json.

    Compares the DK50 parent with the candidate on all 10 panels and checks
    that the parent reproduces the stored DK50 February levels.
    """
    t=time.time();choice=json.loads((OUT/'selection.json').read_text());folder=OUT/(choice['name']+'_refit');parts=[];cparts=[]
    for pi,p in enumerate(D.E.PANELS10):
        d=pd.read_parquet(D.MATCH.OUT/f'{p}_features.parquet');parts.append(d)
        # Refit heads on the 8 Task 2 corridors; on D12_I405_N/S the parent's
        # own residuals, so the candidate equals the parent there.
        if pi<8:rv=predict(folder,choice,d,'y_v');rq=predict(folder,choice,d,'y_q')
        else:
            a=np.load(D.MATCH.OUT/f'{p}_parent.npz');rv=a['y_v'];rq=a['y_q']
        cparts.append((np.clip(d.p_v.to_numpy()+rv,6,120.75),np.maximum(d.p_q.to_numpy()+rq,50/d.lanes.to_numpy())))
        print('February candidate predictions',p,len(d),flush=True)
    d=pd.concat(parts,ignore_index=True);del parts;gc.collect()
    # Parent: RLB residual + k* (hv, hq), then the DK50 projection (weight 0.5)
    # onto the stored February density predictions on the 8 corridors. Flow is
    # per lane in this function.
    rawv,rawq=D.raw_parent(d,'2031_02');hv,hq=D.parent_kstar(d,rawv,rawq,'2031_02')
    kh=np.load(D.HERE/'density/feb_density_predictions.npz')['l2_uniform']
    pv,pq=hv.copy(),hq.copy()
    for pi,p in enumerate(D.E.PANELS10[:8]):
        ix=np.flatnonzero(d.panel_i.to_numpy()==pi);ln=d.lanes.to_numpy()[ix]
        vv,qq=D.project(hv[ix],hq[ix],kh[ix],ln,p,.5);pv[ix]=np.round(vv,3);pq[ix]=np.round(qq,3)/ln
    # Candidate: k* composition of the refit predictions, added to the DK50
    # parent as alpha x (candidate - k* parent).
    nv,nq=D.parent_kstar(d,np.concatenate([a[0]for a in cparts]),np.concatenate([a[1]for a in cparts]),'2031_02')
    alpha=state_alpha(choice['alpha'],d.p_r.to_numpy());cv=np.round(np.clip(pv+alpha*(nv-hv),6,120.75),3);ln=d.lanes.to_numpy();cq=np.round(np.maximum((pq+alpha*(nq-hq))*ln,50),3)/ln
    np.savez(CONF/'predictions.npz',parent_v=pv,parent_q=pq,candidate_v=cv,candidate_q=cq)
    D.OUT=CONF;a,res=D.evaluate(d,{'parent':(pv,pq),'candidate':(cv,cq)});D.uncertainty(a)
    # The parent must equal the stored DK50 February results (diff < 1e-10).
    old=json.loads((D.HERE/'density/feb_results.json').read_text());parity={}
    for mode in ['strict','serve']:
        vals=res[mode]['levels']['parent'];ref=old[mode]['levels']['l2_uniform_0.5'];dif={k:float(vals[k]-ref[k])for k in vals};assert max(abs(v)for v in dif.values())<1e-10;parity[mode]=dif
    report=dict(selection=choice,full_DK50_parent_parity=parity,results=res,uncertainty=json.loads((CONF/'feb_uncertainty.json').read_text()),seconds=time.time()-t)
    dump(CONF/'report.json',report)
    print('FEBRUARY CONFIRMATION',json.dumps(report,indent=2),flush=True)

if __name__=='__main__':run()
