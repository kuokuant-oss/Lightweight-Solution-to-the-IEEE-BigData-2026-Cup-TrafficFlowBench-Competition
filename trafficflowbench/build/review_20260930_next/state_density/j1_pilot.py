"""Preregistered CAP-aligned density pilot; writes only beside this script."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import sys,json,time,gc,resource,hashlib
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq,lightgbm as lgb
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STRUCT=ROOT/'build/review_20260927_structural'
NOISE=ROOT/'build/review_20260929_target092/noise_floor'
sys.path.insert(0,str(STRUCT));sys.path.insert(0,str(NOISE));sys.path.insert(0,str(ROOT))
import density_experiment as D
import fb
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
PARAMS=dict(objective='regression_l2',learning_rate=.05,num_leaves=31,min_data_in_leaf=400,lambda_l2=10,
            feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,num_threads=2,seed=0,verbose=-1,
            force_col_wise=True)
def dump(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def prepare():
    start=time.time()
    if (HERE/'prepared.json').exists():return
    fs=json.loads((STRUCT/'density/features.json').read_text())
    n=sum(pq.ParquetFile(STRUCT/f'density/matched/{p}_features.parquet').metadata.num_rows for p in D.E.PANELS10)
    xx={a:np.lib.format.open_memmap(HERE/f'X_{a}.npy',mode='w+',shape=(n,len(fs)),dtype='float32') for a in ['cap','raw']}
    pred=np.load(fb.PRED);cv=pred['candidate_v'];cq=pred['candidate_q'];dv=pred['parent_v'];dq=pred['parent_q']
    parts=[];hvs=[];hqs=[];offset=0;parity=[]
    for pi,p in enumerate(D.E.PANELS10):
        d=pd.read_parquet(STRUCT/f'density/matched/{p}_features.parquet');nn=len(d);sl=slice(offset,offset+nn)
        a=np.load(STRUCT/f'density/matched/{p}_parent.npz');ln=d.lanes.to_numpy(float)
        rv=np.clip(d.p_v.to_numpy()+a['y_v'],6,120.75);rq=np.maximum(d.p_q.to_numpy()+a['y_q'],50/ln)
        assert D.augment(d,rv,rq)==fs
        xx['raw'][sl]=d[fs].to_numpy('float32')
        hv=np.round(rv,3);hq=np.round(rq*ln,3)
        if p in D.R.KSTAR:
            hv,hq=D.R.kstar(hv,hq,a['khat'],ln,D.R.SEL['panel_alphas'][p],D.R.SEL['weights'][p])
        hvs.append(np.round(hv,3));hqs.append(np.round(hq,3)/ln)
        assert D.augment(d,cv[sl],cq[sl])==fs
        xx['cap'][sl]=d[fs].to_numpy('float32')
        z,meta=D.MATCH.load(p,'2031_02')
        date=np.array(meta['dates'])[d.day.to_numpy(int)]
        r=d[['panel_i','day','li','ci','lanes','vt','qt']].copy();r['date']=date;r['odd']=pd.to_datetime(date).day%2==1
        r['raw_v']=rv;r['raw_q']=rq;r['true_k']=r.qt/np.maximum(r.vt,1)
        parts.append(r)
        parity.append(dict(panel=p,rows=nn,key_order=True,dates=sorted(np.unique(date).tolist())))
        offset+=nn
        print('PREPARED',p,nn,round(time.time()-start,1),flush=True)
        del d,z,a,r;gc.collect()
    for x in xx.values():x.flush()
    meta=pd.concat(parts,ignore_index=True);meta.to_parquet(HERE/'meta.parquet',index=False)
    np.savez(HERE/'parents.npz',cap_v=cv,cap_q=cq,dk_v=dv,dk_q=dq,hmp_v=np.concatenate(hvs),hmp_q=np.concatenate(hqs))
    assert offset==n
    dump('prepared.json',dict(features=fs,rows=n,train_rows=int((meta.panel_i<8).sum()),panels=parity,
        source_sha={str(fb.PRED.relative_to(ROOT)):sha(fb.PRED),'PREREGISTRATION.md':sha(HERE/'PREREGISTRATION.md')},
        seconds=time.time()-start,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
def train():
    start=time.time();meta=pd.read_parquet(HERE/'meta.parquet');parents=np.load(HERE/'parents.npz');fs=json.loads((HERE/'prepared.json').read_text())['features']
    true_k=meta.true_k.to_numpy();active=(meta.panel_i<8).to_numpy();odd=meta.odd.to_numpy();record=[]
    for arm in ['cap','raw']:
        x=np.load(HERE/f'X_{arm}.npy',mmap_mode='r')
        parent_k=parents['cap_q']/parents['cap_v'] if arm=='cap' else meta.raw_q.to_numpy()/meta.raw_v.to_numpy()
        y=true_k-parent_k
        for fitodd in [True,False]:
            tag=f'{arm}_fit'+('odd' if fitodd else 'even');model=HERE/f'{tag}.txt';ppath=HERE/f'{tag}_pred.npy'
            if model.exists() and ppath.exists():continue
            fit=active&(odd==fitodd);test=~(odd==fitodd);tick=time.time()
            print('TRAIN_BEGIN',tag,'rows',int(fit.sum()),'test',int(test.sum()),'time',time.strftime('%Y-%m-%dT%H:%M:%S'),flush=True)
            dataset=lgb.Dataset(np.asarray(x[fit],dtype='float32'),label=y[fit],feature_name=fs,free_raw_data=True)
            m=lgb.train(PARAMS,dataset,num_boost_round=200,callbacks=[lgb.log_evaluation(50)])
            m.save_model(str(model));del dataset;gc.collect()
            predictions=np.full(len(meta),np.nan)
            for pi in range(10):
                ix=np.flatnonzero(test&(meta.panel_i.to_numpy()==pi))
                predictions[ix]=parent_k[ix]+m.predict(x[ix],num_threads=2)
            np.save(ppath,np.maximum(predictions,0))
            info=dict(tag=tag,fit_dates=sorted(meta.loc[fit,'date'].unique().tolist()),eval_dates=sorted(meta.loc[test,'date'].unique().tolist()),fit_rows=int(fit.sum()),seconds=time.time()-tick,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            record.append(info);dump('training_progress.json',record);print('TRAIN_DONE',json.dumps(info),flush=True)
            del m,predictions;gc.collect()
        del x,y;gc.collect()
    dump('training_complete.json',dict(params=PARAMS,rounds=200,records=record,seconds=time.time()-start))
def build_predictions(arm,fitodd,meta,parents):
    odd=meta.odd.to_numpy();held=~(odd==fitodd);tag=f'{arm}_fit'+('odd' if fitodd else 'even');kk=np.load(HERE/f'{tag}_pred.npy')
    cv=parents['cap_v'].copy();cq=parents['cap_q'].copy()
    for pi,p in enumerate(D.E.PANELS10[:8]):
        ix=np.flatnonzero((meta.panel_i.to_numpy()==pi)&held);ln=meta.lanes.to_numpy(float)[ix]
        if arm=='cap':bv,bq=cv[ix],cq[ix]
        else:bv,bq=parents['hmp_v'][ix],parents['hmp_q'][ix]
        vv,qq=D.project(bv,bq,kk[ix],ln,p,.5)
        if arm=='raw':vv=np.round(vv,3)+(parents['cap_v'][ix]-parents['dk_v'][ix]);qq=np.round(qq,3)+(parents['cap_q'][ix]-parents['dk_q'][ix])*ln
        cv[ix]=np.round(np.clip(vv,6,120.75),3);cq[ix]=np.round(np.maximum(qq,50),3)/ln
    return cv,cq
def fit_final():
    start=time.time();model=HERE/'cap_full_feb.txt'
    assert not model.exists(),'Refuse to replace frozen full model'
    assert json.loads((HERE/'feb_results.json').read_text())['pass_feb']
    assert (HERE/'FROZEN_TRANSFER_FULL.md').exists()
    meta=pd.read_parquet(HERE/'meta.parquet',columns=['panel_i','true_k']);parents=np.load(HERE/'parents.npz');fs=json.loads((HERE/'prepared.json').read_text())['features']
    active=meta.panel_i.to_numpy()<8;n=int(active.sum());assert active[:n].all() and not active[n:].any()
    y=meta.true_k.to_numpy()[:n]-parents['cap_q'][:n]/parents['cap_v'][:n]
    x=np.load(HERE/'X_cap.npy',mmap_mode='r')
    print('FULL_FIT_BEGIN',n,time.strftime('%Y-%m-%dT%H:%M:%S'),flush=True)
    data=lgb.Dataset(x[:n],label=y,feature_name=fs,free_raw_data=True)
    m=lgb.train(PARAMS,data,num_boost_round=200);m.save_model(str(model))
    dump('full_fit.json',dict(params=PARAMS,rounds=200,fit_rows=n,parent_out_of_fit=True,fit_month='February',parent_fit_month='January',model_sha=sha(model),freeze_sha=sha(HERE/'FROZEN_TRANSFER_FULL.md'),seconds=time.time()-start,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('FULL_FIT_DONE',time.time()-start,flush=True)
def summarize(data,baseline='C0'):
    level,p,_=fb.summarize(data);delta=level.subtract(level.loc[baseline]);return dict(levels=level.to_dict('index'),deltas=delta.to_dict('index')),p
def evaluate():
    start=time.time();meta=pd.read_parquet(HERE/'meta.parquet');parents=np.load(HERE/'parents.npz');rows=[];combined=[]
    parity={};fkeys=fb.keys()
    assert np.array_equal(meta[['panel_i','day','li','ci']].to_numpy(),fkeys[['panel_i','day','li','ci']].to_numpy())
    arrays={};allcv=parents['cap_v'].copy();allcq=parents['cap_q'].copy()
    for fitodd in [True,False]:
        fold='eval_even' if fitodd else 'eval_odd';testdates=set(meta.loc[meta.odd!=fitodd,'date'])
        for arm,nm in [('cap','C1'),('raw','R1')]:
            arrays[nm]=build_predictions(arm,fitodd,meta,parents)
            if nm=='C1':
                sel=meta.odd.to_numpy()!=fitodd;allcv[sel]=arrays[nm][0][sel];allcq[sel]=arrays[nm][1][sel]
        arrays['C0']=(parents['cap_v'],parents['cap_q'])
        fb.preds().update(arrays)
        for pi,p in enumerate(D.E.PANELS10):
            F=fb.frame(p,'C0')
            for nm,(cv,cq) in arrays.items():
                v=F['v'].copy();q=F['q'].copy();v[F['li'],F['ti']]=cv[F['sel']];q[F['li'],F['ti']]=cq[F['sel']]*F['ln'][F['li'],0]
                scored=fb.score(F,v,q,anatomy=True)
                for r in scored:
                    r['cand']=nm;r['fold']=fold
                    if r['date'] in testdates:rows.append(r)
                    if nm=='C0' and fitodd:combined.append(r.copy())
            del F,v,q;gc.collect()
        print('FOLD_SCORE_DONE',fold,round(time.time()-start,1),flush=True)
    a=pd.DataFrame(rows);a.to_parquet(HERE/'feb_fold_days.parquet',index=False)
    ref=json.loads((ROOT/'build/model_capacity_20260927/state/confirmation/report.json').read_text())['results']['strict']['levels']['candidate']
    b=pd.DataFrame(combined);lev=fb.summarize(b)[0].loc['C0'];parity={k:float(lev[k]-ref[k])for k in lev.index};assert max(abs(x)for x in parity.values())<1e-10,parity
    report={'CAP_metric_parity':parity,'support':'Full February strict-B parent frame; held-out dates only; historical HMP50 holes; observed boundary transitions; truth-storage RHS proxy','folds':{}}
    for fold in ['eval_even','eval_odd']:
        item,_=summarize(a[a.fold==fold]);report['folds'][fold]=item
        print('FOLD_RESULT',fold,json.dumps(item['deltas']),flush=True)
    report['combined'],p=summarize(a);p.to_csv(HERE/'feb_perpanel.csv',index=False)
    assembled=[]
    for p in D.E.PANELS10:
        F=fb.frame(p,'C0');v=F['v'].copy();q=F['q'].copy()
        v[F['li'],F['ti']]=allcv[F['sel']];q[F['li'],F['ti']]=allcq[F['sel']]*F['ln'][F['li'],0]
        scored=fb.score(F,v,q,anatomy=True)
        for r in scored:r['cand']='C1_all_oof'
        assembled.extend(scored);del F,v,q;gc.collect()
    oo=pd.concat([b,pd.DataFrame(assembled)],ignore_index=True);oo.to_parquet(HERE/'feb_all_oof_days.parquet',index=False)
    report['all_oof_assembled'],_=summarize(oo)
    dates=sorted(a.date.unique());rng=np.random.default_rng(300930);boot=[];missing=0
    for _ in range(1000):
        counts=pd.Series(rng.choice(dates,len(dates),replace=True)).value_counts();x=a.copy();cols=x.select_dtypes('number').columns;x[cols]=x[cols].mul(x.date.map(counts).fillna(0),axis=0)
        # Require all panel/regime supports before treating bootstrap as official macro.
        n=x.groupby(['panel','regime']).n.sum()
        if (n==0).any():missing+=1;continue
        m=fb.summarize(x)[0];boot.append((m.loc['C1']-m.loc['C0']).to_dict())
    report['ci95']={k:np.percentile([r[k]for r in boot],[2.5,97.5]).tolist()for k in boot[0]}
    report['bootstrap']=dict(requested=1000,valid=len(boot),missing_regime=missing,conditional_on_all_regimes=True)
    report['pass_feb']=bool(all(report['folds'][f]['deltas']['C1']['total_B']>0 for f in report['folds'])and report['ci95']['total_B'][0]>0)
    report['seconds']=time.time()-start
    dump('feb_results.json',report);np.savez(HERE/'feb_all_oof_predictions.npz',v=allcv,q=allcq)
    print('RESULT',json.dumps(report,indent=2),flush=True)
    return report
if __name__=='__main__':
    phase=sys.argv[1]
    {'prepare':prepare,'train':train,'evaluate':evaluate,'fit_final':fit_final}[phase]()
