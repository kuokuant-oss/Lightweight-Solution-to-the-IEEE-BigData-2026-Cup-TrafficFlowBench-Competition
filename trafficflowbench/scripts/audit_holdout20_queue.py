"""Frozen chronological 20% queue holdout, paired clean-recipe comparison.

The canonical design MUST exist before any fitting. Neither previously shipped
booster is used for predictions: both saw dates in this holdout. Full query grids
are decoded before the hidden eligibility mask is applied to score either arm.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '4'
import gc, hashlib, importlib.util, json, sys, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tfb import task2oof as OOF, task2prior as PR, task2eval as TE
_spec = importlib.util.spec_from_file_location('prior145', ROOT/'scripts/145_onset_prior_forward.py')
M = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M)
BASE = ROOT/'build/fixed_holdout20_20260922'
OUT = BASE/'queue'
KEY = ['window_id','panel','date','condition','link_id','step','label','_eligible']
STABLE_KEY = ['window_id','link_id','step']
CHAMP_THR = {'queue_onset':.20, 'queue_ongoing':.675}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def features(w, name, origin=False, full=False, step6=False):
    path=OUT/f'{name}_features.parquet'
    if path.exists():return pd.read_parquet(path)
    parts=[]
    for panel, wp in w.groupby('panel'):
        start=time.time()
        f=OOF.base_features(wp[['panel','date','origin','condition']],oblique=True,dispersion=True,
                            clock=origin,origin_obs=origin,split='train',include_ineligible=full)
        if step6:f=f[f.step.eq(6)].copy()
        f.attrs={};parts.append(f)
        print('features',name,panel,len(f),round(time.time()-start,1),flush=True)
    d=pd.concat(parts,ignore_index=True);d.attrs={};d.to_parquet(path,index=False)
    return d


def apply_prior(ds, dates, panels):
    """Retain only one panel's daily count matrices at a time."""
    for d in ds:d['queue_prior']=np.float32(0)
    for panel in panels:
        p=PR.build([panel],PR.PriorSpec(tuple(dates)))
        for d in ds:
            ix=d.panel.eq(panel)
            if ix.any():d.loc[ix,'queue_prior']=PR.apply(d.loc[ix],p)
        del p;PR.clear_cache();gc.collect()


def decode(d, prob, thresholds):
    y=np.asarray(prob)>=d.condition.map(thresholds).to_numpy()
    onset=d.condition.eq('queue_onset').to_numpy()
    y[onset & d.step.ne(6).to_numpy()]=False
    indices=np.flatnonzero(onset & d.step.eq(6).to_numpy())
    for ix in d.iloc[indices].groupby('window_id',sort=False).indices.values():
        jj=indices[ix]
        if not y[jj].any():y[jj[np.argmax(np.asarray(prob)[jj])]]=True
    return y


def frame(d, pred, step6=False, official_pattern=False):
    use=d._eligible.to_numpy().copy()
    if step6:use &= d.step.eq(6).to_numpy() | d.condition.ne('queue_onset').to_numpy()
    if official_pattern:
        onset=d.condition.eq('queue_onset')
        early=d[onset & d.step.lt(6) & d._eligible].groupby('window_id').label.sum()
        end=d[onset & d.step.eq(6) & d._eligible].groupby('window_id').label.sum()
        good=set(early[early.eq(0)].index)&set(end[end.gt(0)].index)
        use &= d.condition.ne('queue_onset').to_numpy() | d.window_id.isin(good).to_numpy()
    return TE.iou_frame(d.loc[use].reset_index(drop=True),np.asarray(pred)[use])


def calibrate(d,p,cond,grid):
    rows=[]
    # For the new onset recipe, the only predicted endpoint is its fitting and
    # calibration target. Always include the complete link grid for decoding.
    for t in grid:
        y=decode(d,p,{cond:float(t)})
        f=frame(d,y,step6=cond=='queue_onset')
        rows.append({'threshold':float(t),'score':TE.s_queue(f)})
    best=max(rows,key=lambda x:x['score'])
    return best['threshold'],rows


def save_predictions(name,d,prob,pred):
    z=d[KEY].copy();z['prob']=prob;z['pred']=np.asarray(pred,dtype=np.int8)
    assert not z.duplicated(STABLE_KEY).any()
    z.attrs={};z.to_parquet(OUT/f'{name}_predictions.parquet',index=False)
    return z


def main():
    if not (BASE/'design.json').exists():raise SystemExit('Canonical design missing; no fitting permitted')
    design=json.loads((BASE/'design.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    dates={'development':design['development_dates'],'holdout':design['holdout_dates'],
           'fit':design['queue']['fit_dates'],'early_stopping':design['queue']['es_dates'],
           'calibration':design['queue']['calibration_dates']}
    dev=dates['development'];held=dates['holdout']
    fit=dates['fit'];es=dates['early_stopping'];cal=dates['calibration']
    assert len(dev)==218 and len(held)==55
    assert len(set(fit+es+cal))==len(dev) and sorted(fit+es+cal)==dev
    assert set(dev).isdisjoint(held) and max(dev)<min(held)
    source=ROOT/'data/interim/score_sprint_20260921/recency_chain_windows.parquet'
    windows=pd.read_parquet(source)
    windows=windows[windows.date.isin(dev+held)].copy()
    assert not windows.window_id.duplicated().any()
    panels=sorted(windows.panel.unique())
    windowhash=hashlib.sha256(windows.sort_values('window_id').to_csv(index=False).encode()).hexdigest()
    hw=windows[windows.date.isin(held)].copy()
    fixed=pd.read_parquet(BASE/design['queue']['window_file'])
    assert set(hw.window_id)==set(fixed.window_id)
    hw.to_csv(OUT/'held_windows.csv',index=False)
    cw=windows[windows.date.isin(cal)]
    dw=windows[windows.date.isin(dev)]
    champmanifest=json.loads((ROOT/'data/interim/m21_task2_recency_corrected/prior_manifest.json').read_text())
    champfeats=champmanifest['features'];assert len(champfeats)==50
    candmanifest=json.loads((ROOT/'build/final_sprint_review_20260922/prior/final_feature_hazard_manifest.json').read_text())
    candfeats=candmanifest['features'];assert len(candfeats)==75
    assert all(not c.startswith('_') and c not in OOF.DROP for c in champfeats+candfeats)
    manifest={'canonical_design_sha256':sha(BASE/'design.json'),'script_sha256':sha(__file__),
              'source_windows_sha256':sha(source),'filtered_windows_csv_sha256':windowhash,
              'prior145_script_sha256':sha(ROOT/'scripts/145_onset_prior_forward.py'),
              'dates':dates,'champion_thresholds_fixed':CHAMP_THR,'champion_features':champfeats,
              'candidate_features':candfeats,'no_shipped_booster_reused':True,
              'candidate_refit_after_calibration':False,'champion_refit_dates':dev,
              'fit_prior_dates':fit,'training_eligible_only':True,'decode_full_grid_before_eligibility':True,
              'historically_seen_research_dates':True,'models':{}}
    dump('manifest.json',manifest)
    champion_parts=[];diag_parts=[]
    for cond in OOF.CONDITIONS:
        start=time.time()
        d=features(dw[dw.condition.eq(cond)],f'champion_{cond}_development')
        ca=features(cw[cw.condition.eq(cond)],f'champion_{cond}_calibration_full',full=True)
        ev=features(hw[hw.condition.eq(cond)],f'champion_{cond}_holdout_full',full=True)
        apply_prior([d,ca],fit,panels)
        tr=d[d.date.isin(fit)].copy();va=d[d.date.isin(es)].copy()
        dt=lgb.Dataset(tr[champfeats],tr.label,weight=OOF.window_weights(tr),init_score=OOF.logit(tr.queue_prior.to_numpy()))
        dv=lgb.Dataset(va[champfeats],va.label,weight=OOF.window_weights(va),init_score=OOF.logit(va.queue_prior.to_numpy()),reference=dt)
        params=dict(OOF.PARAMS,num_threads=4)
        m=lgb.train(params,dt,num_boost_round=OOF.NUM_BOOST_ROUND,valid_sets=[dv],
                    callbacks=[lgb.early_stopping(OOF.EARLY_STOPPING,verbose=False),lgb.log_evaluation(0)])
        trees=int(m.best_iteration or OOF.NUM_BOOST_ROUND)
        m.save_model(str(OUT/f'champion_stageA_{cond}.txt'))
        cp=OOF.predict(m,ca,champfeats)
        threshold,grid=calibrate(ca,cp,cond,np.round(np.arange(.05,.951,.025),3))
        save_predictions(f'champion_stageA_{cond}_calibration',ca,cp,decode(ca,cp,{cond:threshold}))
        del dt,dv,tr,va,m;gc.collect()
        # Exactly the original shipping recipe's stage-B refit, restricted to
        # development dates. Its shipped thresholds are the primary control.
        apply_prior([d,ev],dev,panels)
        dt=lgb.Dataset(d[champfeats],d.label,weight=OOF.window_weights(d),init_score=OOF.logit(d.queue_prior.to_numpy()))
        m=lgb.train(params,dt,num_boost_round=trees)
        m.save_model(str(OUT/f'champion_clean_{cond}.txt'))
        ep=OOF.predict(m,ev,champfeats)
        champion_parts.append(save_predictions(f'champion_{cond}_holdout',ev,ep,decode(ev,ep,CHAMP_THR)))
        diag_parts.append(save_predictions(f'champion_recalibrated_{cond}_holdout',ev,ep,decode(ev,ep,{cond:threshold})))
        manifest['models'][f'champion_{cond}']={'trees':trees,'diagnostic_calibration_threshold':threshold,
                   'primary_fixed_threshold':CHAMP_THR[cond],'params':params,'calibration_grid':grid,
                   'fit_rows':int(d.date.isin(fit).sum()),'es_rows':int(d.date.isin(es).sum()),
                   'refit_rows':len(d),'holdout_full_rows':len(ev),'holdout_eligible_rows':int(ev._eligible.sum()),
                   'seconds':time.time()-start}
        dump('manifest.json',manifest)
        print('CHAMPION',cond,manifest['models'][f'champion_{cond}'],flush=True)
        del d,ca,ev,dt,m;gc.collect()

    champion=pd.concat(champion_parts,ignore_index=True)
    diagnostic=pd.concat(diag_parts,ignore_index=True)
    del champion_parts,diag_parts;gc.collect()
    # Candidate fit features predate the holdout design, but are deterministic
    # observation-derived values. Recompute ALL label-dependent priors below.
    src_cache=ROOT/'build/final_sprint_review_20260922/prior/onset_features.parquet'
    cached=pd.read_parquet(src_cache)
    manifest['candidate_static_feature_cache_sha256']=sha(src_cache)
    cached=cached[cached.date.isin(dev+held)].copy().reset_index(drop=True)
    linkmap={}
    for panel in panels:
        for link in sorted(cached.loc[cached.panel.eq(panel),'link_id'].astype(str).unique()):linkmap[(panel,link)]=len(linkmap)
    M.identity(cached,panels,linkmap)
    d=cached[cached.date.isin(fit+es)].copy().reset_index(drop=True)
    qwin=windows[windows.condition.eq('queue_onset') & windows.date.isin(cal+held)]
    query=features(qwin,'candidate_onset_calibration_holdout_full',origin=True,full=True,step6=True)
    M.identity(query,panels,linkmap)
    # Equality assertion over common eligible rows audits the pre-existing
    # cache against freshly generated features, without using targets to fit.
    cols=[c for c in query if c!='_eligible']
    a=query[query._eligible][cols].sort_values(STABLE_KEY).reset_index(drop=True)
    b=cached[cached.date.isin(cal+held)][cols].sort_values(STABLE_KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    del a,b;gc.collect()
    apply_prior([d,query],fit,panels)
    hz=M.Hazard(cached,fit);hz.apply(d);hz.apply(query)
    # Perturb held labels: all candidate query features must remain identical.
    altered=cached.copy();altered.loc[altered.date.isin(held),'label']=1-altered.loc[altered.date.isin(held),'label']
    hz2=M.Hazard(altered,fit);q2=query.copy();hz2.apply(q2)
    pd.testing.assert_frame_equal(query[candfeats],q2[candfeats],check_exact=True)
    del altered,hz2,q2,cached;gc.collect()
    tr=d[d.date.isin(fit)].copy();va=d[d.date.isin(es)].copy()
    ca=query[query.date.isin(cal)].copy().reset_index(drop=True)
    ev=query[query.date.isin(held)].copy().reset_index(drop=True)
    cats=['panel_code','link_code']
    dt=lgb.Dataset(tr[candfeats],tr.label,weight=OOF.window_weights(tr),categorical_feature=cats)
    dv=lgb.Dataset(va[candfeats],va.label,weight=OOF.window_weights(va),categorical_feature=cats,reference=dt)
    m=lgb.train(M.PARAMS,dt,num_boost_round=1000,valid_sets=[dv],
                callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
    m.save_model(str(OUT/'candidate_clean_onset.txt'))
    cp=M.predictions(m,ca,candfeats,'feature_hazard')
    threshold,grid=calibrate(ca,cp,'queue_onset',M.GRID)
    ep=M.predictions(m,ev,candfeats,'feature_hazard')
    final=save_predictions('candidate_onset_step6_holdout',ev,ep,decode(ev,ep,{'queue_onset':threshold}))
    save_predictions('candidate_onset_step6_calibration',ca,cp,decode(ca,cp,{'queue_onset':threshold}))
    manifest['models']['candidate_onset']={'trees':m.current_iteration(),'threshold':threshold,'calibration_grid':grid,
                'params':M.PARAMS,'fit_rows':len(tr),'es_rows':len(va),'calibration_full_rows':len(ca),
                'holdout_full_rows':len(ev),'holdout_eligible_rows':int(ev._eligible.sum()),
                'held_label_perturbation_feature_exact_match':True,'cached_eligible_feature_exact_match':True}
    dump('manifest.json',manifest)
    # Keyed replacement of endpoint predictions; all ongoing probabilities and
    # decisions are identically reused in candidate and champion.
    candidate=champion.copy()
    ons=candidate.condition.eq('queue_onset')
    candidate.loc[ons,'pred']=0;candidate.loc[ons,'prob']=0
    q=final.set_index(STABLE_KEY)
    target=candidate.loc[ons & candidate.step.eq(6),STABLE_KEY]
    ix=pd.MultiIndex.from_frame(target)
    assert len(ix)==len(q) and set(ix)==set(q.index)
    candidate.loc[target.index,'pred']=q.loc[ix,'pred'].to_numpy()
    candidate.loc[target.index,'prob']=q.loc[ix,'prob'].to_numpy()
    pd.testing.assert_frame_equal(candidate.loc[~ons],champion.loc[~ons],check_exact=True)
    for label,z in [('champion',champion),('candidate',candidate),('champion_recalibrated',diagnostic)]:
        z.attrs={};z.to_parquet(OUT/f'{label}_holdout_predictions.parquet',index=False)
    reports={}
    for metric,opts in [('main_all6_eligible',{}),('onset_step6_eligible',{'step6':True}),
                         ('official_pattern_eligible',{'official_pattern':True})]:
        fs={name:frame(z,z.pred.to_numpy(),**opts) for name,z in
                  [('champion',champion),('candidate',candidate),('champion_recalibrated',diagnostic)]}
        for name,f in fs.items():f.to_parquet(OUT/f'{name}_{metric}_windows.parquet',index=False)
        reports[metric]={name:{'S_queue':TE.s_queue(f),'subgroups':TE.subgroups(f),
                           'windows':len(f),'dates':f.date.nunique()} for name,f in fs.items()}
        paired=fs['champion'].rename(columns={'IoU':'champion_IoU','inter':'champion_inter','union':'champion_union'})
        for other,name in [('candidate','candidate'),('champion_recalibrated','recalibrated_champion')]:
            paired=paired.merge(fs[other][['window_id','IoU','inter','union']].rename(
                columns={'IoU':name+'_IoU','inter':name+'_inter','union':name+'_union'}),on='window_id',validate='one_to_one')
        paired['delta_IoU']=paired.candidate_IoU-paired.champion_IoU
        paired.to_parquet(OUT/f'paired_{metric}_windows.parquet',index=False)
        paired.groupby(['family_id','panel','condition'])[['champion_IoU','candidate_IoU','recalibrated_champion_IoU','delta_IoU']].mean().to_csv(OUT/f'paired_{metric}_groups.csv')
        reports[metric]['paired_candidate_vs_champion']={'delta':TE.s_queue(fs['candidate'])-TE.s_queue(fs['champion']),
                   'confidence_interval':'root joint fixed-weight date bootstrap; avoid renormalizing a missing ongoing group'}
        reports[metric]['paired_candidate_vs_recalibrated_champion']={'delta':TE.s_queue(fs['candidate'])-TE.s_queue(fs['champion_recalibrated'])}
    dump('results.json',reports)
    hashes={p.name:sha(p) for p in OUT.glob('*.txt')}
    manifest['model_sha256']=hashes;dump('manifest.json',manifest)
    print('FINAL',json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
