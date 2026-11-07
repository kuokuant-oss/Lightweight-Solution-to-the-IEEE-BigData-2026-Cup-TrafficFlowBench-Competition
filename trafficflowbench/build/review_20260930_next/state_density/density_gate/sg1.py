"""Preregistered SG1. No training; diagnostics/output only beside this file."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,gc,resource,hashlib
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE));import j1_pilot as J
fb=J.fb;D=J.D
def cap_gate(v,vf):
    v=np.asarray(v,dtype=np.float64);assert np.array_equal(v,np.round(v,3)), 'CAP gate parent must be round3'
    vf=np.asarray(vf,dtype=np.float32);assert np.isfinite(vf).all() and (vf>0).all()
    return v<.85*vf
def dump(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def summarize(a):
    m,p,_=fb.summarize(a);return m,p
def comparisons(a):
    m,p=summarize(a)
    return dict(levels=m.to_dict('index'),SG1_minus_C1=(m.loc['SG1']-m.loc['C1']).to_dict(),SG1_minus_C0=(m.loc['SG1']-m.loc['C0']).to_dict()),p
def bootstrap(a,n=1000):
    dates=sorted(a.date.unique());rng=np.random.default_rng(300930);values=[];missing=0
    for _ in range(n):
        counts=pd.Series(rng.choice(dates,len(dates),replace=True)).value_counts();x=a.copy();cols=x.select_dtypes('number').columns;x[cols]=x[cols].mul(x.date.map(counts).fillna(0),axis=0)
        if (x.groupby(['panel','regime']).n.sum()==0).any():missing+=1;continue
        m,_=summarize(x);values.append(dict(C1=m.loc['SG1']-m.loc['C1'],C0=m.loc['SG1']-m.loc['C0']))
    return dict(requested=n,valid=len(values),missing_regime=missing,conditional_on_all_regimes=True,
        ci95={b:{c:np.percentile([v[b][c]for v in values],[2.5,97.5]).tolist()for c in values[0][b].index}for b in ['C1','C0']})
def feb():
    start=time.time();assert not (HERE/'feb_results.json').exists()
    assert (HERE/'PREREGISTRATION.md').exists();ref=json.loads((BASE/'feb_results.json').read_text())
    meta=pd.read_parquet(BASE/'meta.parquet',columns=['panel_i','day','li','ci','lanes','date','odd']);parents=np.load(BASE/'parents.npz')
    vf=np.concatenate([pd.read_parquet(D.MATCH.OUT/f'{p}_features.parquet',columns=['vf']).vf.to_numpy()for p in D.E.PANELS10])
    assert len(vf)==len(meta) and np.isfinite(vf).all() and (vf>0).all()
    gate=cap_gate(parents['cap_v'],vf)&(meta.panel_i.to_numpy()<8)
    fk=fb.keys();assert np.array_equal(meta[['panel_i','day','li','ci']].to_numpy(),fk[['panel_i','day','li','ci']].to_numpy())
    parts=[];parent_days=[];all_c1_v=parents['cap_v'].copy();all_c1_q=parents['cap_q'].copy()
    for fitodd in [True,False]:
        fold='eval_even'if fitodd else'eval_odd';held=meta.odd.to_numpy()!=fitodd;dates=set(meta.loc[held,'date'])
        cv,cq=J.build_predictions('cap',fitodd,meta,parents);all_c1_v[held]=cv[held];all_c1_q[held]=cq[held]
        sg_v=parents['cap_v'].copy();sg_q=parents['cap_q'].copy();sg_v[gate&held]=cv[gate&held];sg_q[gate&held]=cq[gate&held]
        arrays={'C0':(parents['cap_v'],parents['cap_q']),'C1':(cv,cq),'SG1':(sg_v,sg_q)};fb.preds().update(arrays)
        for p in D.E.PANELS10:
            F=fb.frame(p,'C0')
            for nm,(vv,qq)in arrays.items():
                v=F['v'].copy();q=F['q'].copy();v[F['li'],F['ti']]=vv[F['sel']];q[F['li'],F['ti']]=qq[F['sel']]*F['ln'][F['li'],0]
                out=fb.score(F,v,q)
                for r in out:
                    r['cand']=nm;r['fold']=fold
                    if r['date']in dates:parts.append(r)
                    if fitodd and nm=='C0':parent_days.append(r.copy())
            del F,v,q;gc.collect()
    a=pd.DataFrame(parts);a.to_parquet(HERE/'feb_fold_days.parquet',index=False)
    report=dict(status='Exploratory SG1 gate; frozen .85 CAP-parent rule only',gate='unchanged CAP speed < .85 vf; eight general panels only',gate_rows=int(gate.sum()),general_rows=int((meta.panel_i<8).sum()),folds={})
    parity={}
    for fold in ['eval_even','eval_odd']:
        report['folds'][fold],_=comparisons(a[a.fold==fold]);m=report['folds'][fold]['levels']
        for name,old in [('C0','C0'),('C1','C1')]:
            dif={k:float(m[name][k]-ref['folds'][fold]['levels'][old][k])for k in m[name]};assert max(abs(v)for v in dif.values())<1e-12;parity[fold+'_'+name]=dif
    report['combined'],per=comparisons(a);per.to_csv(HERE/'feb_perpanel.csv',index=False)
    for name in ['C0','C1']:
        dif={k:float(report['combined']['levels'][name][k]-ref['combined']['levels'][name][k])for k in report['combined']['levels'][name]};assert max(abs(v)for v in dif.values())<1e-12;parity['combined_'+name]=dif
    report['C0_C1_metric_parity']=parity
    # All-OOF assembled frame exposes cross-midnight differences without changing primary.
    arrays={'C1':(all_c1_v,all_c1_q),'SG1':(np.where(gate,all_c1_v,parents['cap_v']),np.where(gate,all_c1_q,parents['cap_q']))};assembled=[]
    for p in D.E.PANELS10:
        F=fb.frame(p,'C0')
        for nm,(vv,qq)in arrays.items():
            v=F['v'].copy();q=F['q'].copy();v[F['li'],F['ti']]=vv[F['sel']];q[F['li'],F['ti']]=qq[F['sel']]*F['ln'][F['li'],0]
            out=fb.score(F,v,q)
            for r in out:r['cand']=nm
            assembled.extend(out)
        del F,v,q;gc.collect()
    full=pd.concat([pd.DataFrame(parent_days),pd.DataFrame(assembled)],ignore_index=True);full.to_parquet(HERE/'feb_full_days.parquet',index=False)
    report['full_all_oof'],_=comparisons(full)
    for k,v in report['full_all_oof']['levels']['C1'].items():assert abs(v-ref['all_oof_assembled']['levels']['C1_all_oof'][k])<1e-12
    report['bootstrap']=bootstrap(a)
    report['pass_feb']=bool(all(report['folds'][f]['SG1_minus_C1']['total_B']>0 for f in report['folds'])and report['bootstrap']['ci95']['C1']['total_B'][0]>0)
    report['seconds']=time.time()-start;report['maxrss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;report['prereg_sha']=J.sha(HERE/'PREREGISTRATION.md')
    dump('feb_results.json',report);print(json.dumps(dict(pass_feb=report['pass_feb'],folds={f:{b:report['folds'][f][b]['total_B']for b in ['SG1_minus_C1','SG1_minus_C0']}for f in report['folds']},combined={b:report['combined'][b]['total_B']for b in ['SG1_minus_C1','SG1_minus_C0']},CI=report['bootstrap']['ci95']['C1']['total_B'],seconds=report['seconds'],rss=report['maxrss_bytes']),indent=2),flush=True)
if __name__=='__main__':{'feb':feb}[sys.argv[1]]()
