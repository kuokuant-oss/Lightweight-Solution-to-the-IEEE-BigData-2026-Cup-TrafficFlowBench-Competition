"""Forward-development selection, then one retrospective fixed-holdout check.

Never selects a recipe using the 55 held dates. Native candidate and champion
holdout predictions already exist; they are comparison controls, not features.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='3'
import argparse,gc,hashlib,importlib.util,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sp=importlib.util.spec_from_file_location('holdqueue',ROOT/'scripts/audit_holdout20_queue.py')
Q=importlib.util.module_from_spec(sp);sp.loader.exec_module(Q)
M=Q.M;OOF=Q.OOF;PR=Q.PR;TE=Q.TE
OUT=ROOT/'build/strongest_candidate_20260922/onset'
HOLD=ROOT/'build/fixed_holdout20_20260922/queue'
SOURCE=ROOT/'build/final_sprint_review_20260922/prior'
KEY=Q.KEY
RULES=['reference127','capacity31','capacity63','no_unconditional_prior','no_absolute_clock','legacy_blend25']
CLOCK=['tod_min','tod_sin','tod_cos','origin_tod_min']
PARAMS=dict(M.PARAMS,num_threads=3)
FOLDS=[('oct','2030-08-15','2030-08-16','2030-08-31','2030-09-01','2030-09-30','2030-10-01','2030-10-31'),
       ('nov','2030-09-15','2030-09-16','2030-09-30','2030-10-01','2030-10-31','2030-11-01','2030-11-30'),
       ('dec','2030-10-15','2030-10-16','2030-10-31','2030-11-01','2030-11-30','2030-12-01','2031-01-04')]


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def dates(lo,hi):return pd.date_range(lo,hi).strftime('%Y-%m-%d').tolist()


def rulespec(rule,features):
    p=dict(PARAMS);fs=list(features)
    if rule=='capacity31':p['num_leaves']=31
    elif rule=='capacity63':p['num_leaves']=63
    elif rule=='no_unconditional_prior':fs.remove('queue_prior')
    elif rule=='no_absolute_clock':fs=[f for f in fs if f not in CLOCK]
    return fs,p


def load_static():
    d=pd.read_parquet(SOURCE/'onset_features.parquet')
    panels=sorted(d.panel.unique());mapping={}
    for panel in panels:
        for link in sorted(d.loc[d.panel.eq(panel),'link_id'].astype(str).unique()):mapping[(panel,link)]=len(mapping)
    M.identity(d,panels,mapping)
    fs=json.loads((SOURCE/'final_feature_hazard_manifest.json').read_text())['features']
    oldfs=json.loads((ROOT/'data/interim/m21_task2_recency_corrected/prior_manifest.json').read_text())['features']
    return d,panels,mapping,fs,oldfs


def fullgrid_development(mapping,panels):
    cache=OUT/'development_fullgrid_step6_features.parquet'
    if cache.exists():return pd.read_parquet(cache)
    win=pd.read_parquet(ROOT/'data/interim/score_sprint_20260921/recency_chain_windows.parquet')
    win=win[win.condition.eq('queue_onset') & win.date.between('2030-09-01','2031-01-04')]
    # Reuse only deterministic features, with held dates removed before fitting.
    previous=pd.read_parquet(HOLD/'candidate_onset_calibration_holdout_full_features.parquet')
    previous=previous[previous.date.between('2030-09-01','2031-01-04')].copy()
    have=set(previous.window_id)
    parts=[previous]
    for panel,w in win[~win.window_id.isin(have)].groupby('panel'):
        start=time.time()
        f=OOF.base_features(w[['panel','date','origin','condition']],oblique=True,dispersion=True,
             clock=True,origin_obs=True,include_ineligible=True,split='train')
        f=f[f.step.eq(6)].copy();f.attrs={};parts.append(f)
        print('FEATURES',panel,len(f),round(time.time()-start,1),flush=True)
    z=pd.concat(parts,ignore_index=True);z.attrs={}
    assert set(z.window_id)==set(win.window_id)
    assert z.date.le('2031-01-04').all() and not z.duplicated(Q.STABLE_KEY).any()
    M.identity(z,panels,mapping);z.to_parquet(cache,index=False)
    return z


def early_truth(d):
    return d[d.step.lt(6)].groupby('window_id').label.sum()


def fullsix(d,pred,early):
    f=Q.frame(d,pred)
    extra=f.window_id.map(early).fillna(0).to_numpy()
    f['union']=f.union.to_numpy()+extra
    f['IoU']=np.where(f.union.eq(0),1.,f.inter/f.union.clip(lower=1))
    return f


def fit_predict(rule,tr,es,queries,features,oldfeatures,out):
    out.mkdir(parents=True,exist_ok=True)
    if rule=='legacy50':
        fs=oldfeatures;params=dict(OOF.PARAMS,num_threads=3);cats=[]
        traink={'init_score':OOF.logit(tr.queue_prior.to_numpy())};esk={'init_score':OOF.logit(es.queue_prior.to_numpy())}
        rounds,stop=OOF.NUM_BOOST_ROUND,OOF.EARLY_STOPPING
    else:
        fs,params=rulespec(rule,features);cats=['panel_code','link_code'];traink={};esk={};rounds,stop=1000,100
    dt=lgb.Dataset(tr[fs],tr.label,weight=OOF.window_weights(tr),categorical_feature=cats,**traink)
    dv=lgb.Dataset(es[fs],es.label,weight=OOF.window_weights(es),categorical_feature=cats,reference=dt,**esk)
    start=time.time();model=lgb.train(params,dt,num_boost_round=rounds,valid_sets=[dv],
         callbacks=[lgb.early_stopping(stop,verbose=False),lgb.log_evaluation(0)])
    path=out/f'{rule}.txt';model.save_model(str(path))
    probabilities=[M.predictions(model,q,fs,'offset' if rule=='legacy50' else 'feature_hazard') for q in queries]
    details={'recipe':rule,'features':fs,'params':params,'trees':model.current_iteration(),'model_sha256':Q.sha(path),
             'fit_rows':len(tr),'es_rows':len(es),'seconds':time.time()-start}
    del model,dt,dv;gc.collect()
    return probabilities,details


def evaluate_recipe(rule,ca,ev,cp,ep,early,out,details):
    threshold,grid=Q.calibrate(ca,cp,'queue_onset',M.GRID)
    pred=Q.decode(ev,ep,{'queue_onset':threshold})
    f=fullsix(ev,pred,early)
    f.to_parquet(out/f'{rule}_windows.parquet',index=False)
    ev[KEY].assign(prob=ep,pred=pred.astype('int8')).to_parquet(out/f'{rule}_predictions.parquet',index=False)
    ca[KEY].assign(prob=cp).to_parquet(out/f'{rule}_calibration.parquet',index=False)
    r={'recipe':rule,'threshold':threshold,'S_onset':TE.s_queue(f),'step6_S_onset':TE.s_queue(Q.frame(ev,pred)),
       'calibration_grid':grid,'calibration_windows':ca.window_id.nunique(),'evaluation_windows':ev.window_id.nunique(),
       'panel_scores':f.groupby('panel').IoU.mean().to_dict(),'model':details}
    dump(out/f'{rule}_result.json',r);print('RESULT',out.name,rule,threshold,r['S_onset'],flush=True)
    return r


def explore():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':Q.sha(__file__),
       'selection_data_last_date':'2031-01-04','fixed_holdout_not_used_for_selection':True,
       'recipes':RULES,'params':PARAMS,'folds':FOLDS,'threshold_grid':M.GRID.tolist(),
       'threshold_objective':'full physical-grid decoded step6 eligible macro IoU, matching native145',
       'selection_metric':'equal average of three forward all-six-step onset macro IoUs',
       'selection_gate':'positive mean delta vs reference, >=2/3 positive folds, worst fold delta >= -.015',
       'near_tie':'within .001 of best eligible mean; prefer capacity31, capacity63, no_absolute_clock, no_unconditional_prior, reference127, legacy_blend25',
       'hypotheses':{'capacity':'smaller trees may reduce sparse onset overfitting',
          'prior':'unconditional queue propensity can compete with conditional-onset hazard',
          'clock':'absolute clock could shortcut relative history/hazard with temporal drift',
          'blend':'legacy all-horizon, no-origin head may have complementary ranking errors'},
       'legacy_blend25':'0.75*reference127 probability + 0.25*separately fitted legacy50 probability; own calibration',
       'historically_seen_holdout_warning':'Last55 dates were used in prior research; retrospective audit, not untouched blindtest',
       'calibration_length_note':'all 3 inner calibration blocks >=30 calendar days; corrected before fitting'}
    if (OUT/'protocol.json').exists():raise SystemExit('Refusing to overwrite an existing exploration protocol')
    dump(OUT/'protocol.json',protocol)
    cached,panels,mapping,fs,oldfs=load_static();cached=cached[cached.date.le('2031-01-04')].copy()
    full=fullgrid_development(mapping,panels)
    old=pd.read_parquet(HOLD/'champion_queue_onset_development_features.parquet')
    assert old.date.le('2031-01-04').all()
    truth=early_truth(old)
    results=[]
    for name,end,es0,es1,ca0,ca1,ev0,ev1 in FOLDS:
        fitdates=dates('2030-06-01',end)
        tr=cached[cached.date.le(end)].copy().reset_index(drop=True)
        va=cached[cached.date.between(es0,es1)].copy().reset_index(drop=True)
        ca=full[full.date.between(ca0,ca1)].copy().reset_index(drop=True)
        ev=full[full.date.between(ev0,ev1)].copy().reset_index(drop=True)
        ot=old[old.date.le(end)].copy().reset_index(drop=True)
        ov=old[old.date.between(es0,es1)].copy().reset_index(drop=True)
        Q.apply_prior([tr,va,ca,ev,ot,ov],fitdates,panels)
        hz=M.Hazard(cached,fitdates)
        for d in [tr,va,ca,ev]:hz.apply(d)
        dest=OUT/'development'/name
        dump(dest/'dates.json',{'fit':fitdates,'es':dates(es0,es1),'cal':dates(ca0,ca1),'eval':dates(ev0,ev1)})
        probs={};model_details={}
        for rule in RULES[:-1]+['legacy50']:
            pp,detail=fit_predict(rule,ot if rule=='legacy50' else tr,ov if rule=='legacy50' else va,[ca,ev],fs,oldfs,dest)
            probs[rule]=pp;model_details[rule]=detail
            if rule!='legacy50':
                r=evaluate_recipe(rule,ca,ev,*pp,truth,dest,detail);r['fold']=name;results.append(r)
        blend=[.75*x+.25*y for x,y in zip(probs['reference127'],probs['legacy50'])]
        r=evaluate_recipe('legacy_blend25',ca,ev,*blend,truth,dest,
                          {'weights':{'reference127':.75,'legacy50':.25},'components':[model_details['reference127'],model_details['legacy50']]})
        r['fold']=name;results.append(r);dump(OUT/'development_results.json',results)
        del tr,va,ca,ev,ot,ov,hz,probs;gc.collect()
    table=pd.DataFrame([{'fold':r['fold'],'recipe':r['recipe'],'S_onset':r['S_onset'],'threshold':r['threshold']} for r in results])
    pivot=table.pivot(index='fold',columns='recipe',values='S_onset').loc[['oct','nov','dec']]
    delta=pivot.subtract(pivot.reference127,axis=0)
    summary=[]
    for rule in RULES:
        v=delta[rule];eligible=bool(v.mean()>0 and (v>0).sum()>=2 and v.min()>=-.015)
        summary.append({'recipe':rule,'mean_S_onset':float(pivot[rule].mean()),'mean_delta':float(v.mean()),
                        'positive_folds':int((v>0).sum()),'worst_fold_delta':float(v.min()),'eligible':eligible,'fold_deltas':v.to_dict()})
    eligible=[r for r in summary if r['eligible']]
    if eligible:
        best=max(r['mean_S_onset'] for r in eligible);short={r['recipe'] for r in eligible if r['mean_S_onset']>=best-.001}
        order=['capacity31','capacity63','no_absolute_clock','no_unconditional_prior','reference127','legacy_blend25']
        selected=next(r for r in order if r in short)
    else:selected='reference127'
    freeze={'selected_at_utc':datetime.now(timezone.utc).isoformat(),'selected_recipe':selected,
            'protocol_sha256':Q.sha(OUT/'protocol.json'),'development_results_sha256':Q.sha(OUT/'development_results.json'),
            'held_scores_seen_during_this_selection':False,'summary':summary,'reason':'Predeclared development-only gate and near-tie preference'}
    dump(OUT/'selection.json',freeze);table.to_csv(OUT/'development_scores.csv',index=False)
    print('SELECTED',json.dumps(freeze,indent=2),flush=True)


def holdout():
    selection=json.loads((OUT/'selection.json').read_text());rule=selection['selected_recipe']
    dest=OUT/'holdout'
    if (dest/'results.json').exists():raise SystemExit('Holdout already evaluated; refusing repeat selection')
    add=json.loads((ROOT/'build/fixed_holdout20_20260922/native_recipe_addendum.json').read_text())
    fitdates=add['fit_dates'];esdates=add['es_dates'];caldates=add['calibration_dates'];evaldates=add['holdout_dates']
    cached,panels,mapping,fs,oldfs=load_static()
    tr=cached[cached.date.isin(fitdates)].copy().reset_index(drop=True)
    va=cached[cached.date.isin(esdates)].copy().reset_index(drop=True)
    query=pd.read_parquet(HOLD/'candidate_onset_calibration_holdout_full_features.parquet')
    M.identity(query,panels,mapping)
    ca=query[query.date.isin(caldates)].copy().reset_index(drop=True)
    ev=query[query.date.isin(evaldates)].copy().reset_index(drop=True)
    old=pd.read_parquet(HOLD/'champion_queue_onset_development_features.parquet')
    ot=old[old.date.isin(fitdates)].copy().reset_index(drop=True)
    ov=old[old.date.isin(esdates)].copy().reset_index(drop=True)
    Q.apply_prior([tr,va,ca,ev,ot,ov],fitdates,panels)
    hz=M.Hazard(cached,fitdates)
    for d in [tr,va,ca,ev]:hz.apply(d)
    dump(dest/'selection_receipt.json',{'selection_sha256':Q.sha(OUT/'selection.json'),'recipe':rule,
         'fit_dates':fitdates,'es_dates':esdates,'calibration_dates':caldates,'held_dates':evaldates,
         'created_before_fit_utc':datetime.now(timezone.utc).isoformat()})
    if rule=='legacy_blend25':
        p1,i1=fit_predict('reference127',tr,va,[ca,ev],fs,oldfs,dest)
        p2,i2=fit_predict('legacy50',ot,ov,[ca,ev],fs,oldfs,dest)
        pp=[.75*x+.25*y for x,y in zip(p1,p2)];info={'components':[i1,i2],'weights':[.75,.25]}
    else:pp,info=fit_predict(rule,tr,va,[ca,ev],fs,oldfs,dest)
    champion=pd.read_parquet(HOLD/'champion_holdout_predictions.parquet')
    prior_candidate=pd.read_parquet(HOLD/'native/candidate_holdout_predictions.parquet')
    truth=early_truth(champion)
    r=evaluate_recipe(rule,ca,ev,*pp,truth,dest,info)
    own=pd.read_parquet(dest/f'{rule}_predictions.parquet').set_index(Q.STABLE_KEY)
    candidate=champion.copy();on=candidate.condition.eq('queue_onset');candidate.loc[on,['pred','prob']]=0
    target=candidate.loc[on & candidate.step.eq(6),Q.STABLE_KEY];idx=pd.MultiIndex.from_frame(target)
    assert len(idx)==len(own) and set(idx)==set(own.index)
    for c in ('label','_eligible'):assert np.array_equal(candidate.loc[target.index,c],own.loc[idx,c])
    candidate.loc[target.index,'pred']=own.loc[idx,'pred'].to_numpy()
    candidate.loc[target.index,'prob']=own.loc[idx,'prob'].to_numpy()
    candidate.attrs={};candidate.to_parquet(dest/'candidate_holdout_predictions.parquet',index=False)
    frames={n:Q.frame(d,d.pred.to_numpy()) for n,d in [('champion',champion),('existing_native',prior_candidate),('candidate',candidate)]}
    merged=frames['champion'][['window_id','panel','family_id','condition','date']].copy()
    for name,f in frames.items():
        merged=merged.merge(f[['window_id','IoU','inter','union']].rename(columns={c:name+'_'+c for c in ['IoU','inter','union']}),on='window_id',validate='one_to_one')
    merged['delta_IoU']=merged.candidate_IoU-merged.champion_IoU
    merged.to_parquet(dest/'paired_main_all6_eligible_windows.parquet',index=False)
    scores={n:{'S_queue':TE.s_queue(f),'subgroups':TE.subgroups(f)} for n,f in frames.items()}
    dump(dest/'results.json',{'selected_recipe':rule,'model_and_calibration':r,'scores':scores,
         'delta_vs_champion':scores['candidate']['S_queue']-scores['champion']['S_queue'],
         'delta_vs_existing_native':scores['candidate']['S_queue']-scores['existing_native']['S_queue']})
    print('HOLDOUT',json.dumps(scores,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['explore','holdout']);args=p.parse_args()
    with threadpool_limits(limits=3):
        explore() if args.mode=='explore' else holdout()
