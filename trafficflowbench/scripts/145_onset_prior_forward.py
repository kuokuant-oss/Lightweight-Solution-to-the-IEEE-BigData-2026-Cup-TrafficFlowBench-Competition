"""Matched onset-only prior experiment and fixed-model serving candidate.

Three predeclared arms share clock, explicit identity, origin-T observations,
step-6 labels, dates, tree recipe and calibration. Only the prior treatment
changes. January and February are forward evaluations; final fit ends in
December, early stopping uses January, calibration uses February. The SAME
calibrated model is served, avoiding full-refit probability-scale changes.
No network calls, no submission, no changes to champion artifacts.
"""
from __future__ import annotations
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.ndimage import convolve1d
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tfb import release as R, task2oof as OOF, task2prior as PR, task2eval as TE, task2infer as TI

OUT = ROOT / 'build/final_sprint_review_20260922/prior'
ARMS = ['offset', 'feature', 'feature_hazard']
PARAMS = dict(OOF.PARAMS, num_threads=4, num_leaves=127, max_bin=127, seed=20260922)
CASES = [('jan', '2030-10-31', '2030-11', '2030-12', '2031-01'),
         ('feb', '2030-11-30', '2030-12', '2031-01', '2031-02'),
         ('final', '2030-12-31', '2031-01', '2031-02', None)]
GRID = np.round(np.arange(.05, .801, .025), 3)
KEY = ['window_id', 'panel', 'date', 'condition', 'link_id', 'step', 'label']


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def identity(d, panels, linkmap):
    d['panel_code'] = d.panel.map({p:i for i,p in enumerate(panels)}).astype('int32')
    d['link_code'] = [linkmap[(p,str(l))] for p,l in zip(d.panel,d.link_id)]
    d['link_code'] = d.link_code.astype('int32')
    return d


class Hazard:
    """Fit-date, LOO-within-fit conditional onset frequency with partial pooling."""
    def __init__(self, d, dates):
        self.parts = {}
        for panel, allp in d.groupby('panel'):
            tr = allp[allp.date.isin(dates)]
            shape = (int(allp._link_idx.max())+1, 288)
            num, den = np.zeros(shape), np.zeros(shape)
            day = {}
            for date,g in tr.groupby('date'):
                n, z = np.zeros(shape), np.zeros(shape)
                ij = (g._link_idx.to_numpy(), g._tod_idx.to_numpy())
                np.add.at(n,ij,g.label.to_numpy()); np.add.at(z,ij,1)
                num += n; den += z; day[str(date)] = (n,z)
            self.parts[panel] = (num,den,day)

    @staticmethod
    def rate(num, den):
        global_p = float((num.sum()+1)/(den.sum()+2))
        ln, ld = num.sum(axis=1), den.sum(axis=1)
        link_p = (ln+20*global_p)/(ld+20)
        kernel = np.array([.25,.5,1,.5,.25])
        n = convolve1d(num,kernel,axis=1,mode='wrap')
        z = convolve1d(den,kernel,axis=1,mode='wrap')
        return (n+10*link_p[:,None])/(z+10), link_p

    def apply(self, d):
        h, link, mass = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
        for panel, ix in d.groupby('panel',sort=False).indices.items():
            num,den,days = self.parts[panel]
            for date in d.iloc[ix].date.unique():
                ii=ix[d.date.to_numpy()[ix]==date]
                a,b=days.get(str(date),(0,0))
                r,lr=self.rate(num-a,den-b)
                li=d._link_idx.to_numpy()[ii]; ti=d._tod_idx.to_numpy()[ii]
                h[ii]=r[li,ti]; link[ii]=lr[li]
                # Normalize over the complete physical link axis, regardless of
                # which future cells have training labels / evaluation masks.
                mass[ii]=r.sum(axis=0)[ti]
        d['onset_hazard']=h
        d['onset_link_rate']=link
        d['onset_hazard_share']=h/np.maximum(mass,1e-10)
        return d


def predictions(model, d, feats, arm):
    raw = model.predict(d[feats],raw_score=True)
    if arm == 'offset': raw += OOF.logit(d.queue_prior.to_numpy())
    return expit(raw)


def decode(d, prob, thr):
    y=prob>=thr
    for ix in d.groupby('window_id',sort=False).indices.values():
        if not y[ix].any(): y[ix[np.argmax(prob[ix])]]=True
    return y


def select_threshold(d, p):
    rows=[]
    for t in GRID:
        f=TE.iou_frame(d,decode(d,p,t))
        rows.append((TE.s_queue(f),float(t)))
    score,thr=max(rows,key=lambda x:x[0])
    return thr,score,rows


def diagnostics(d, p, thr):
    frame=TE.iou_frame(d,decode(d,p,thr))
    ranked=[]
    for wid,ix in d.groupby('window_id',sort=False).indices.items():
        order=np.argsort(-p[ix],kind='stable'); y=d.label.to_numpy()[ix][order]>0
        t=np.cumsum(y); k=np.arange(1,len(y)+1)
        ranked.append(dict(window_id=wid,top1_hit=bool(y[0]),oracle=float(np.max(t/np.maximum(k+y.sum()-t,1))),maxprob=float(p[ix].max()),prior_at_top=float(d.queue_prior.iloc[ix[order[0]]])))
    rank=pd.DataFrame(ranked)
    f=frame.merge(rank,on='window_id',validate='one_to_one')
    return f,dict(S_onset=TE.s_queue(f),top1_hit=float(f.top1_hit.mean()),oracle=float(f.oracle.mean()),lowconf_n=int(f.maxprob.lt(.1).sum()),lowconf_iou=float(f.loc[f.maxprob.lt(.1),'IoU'].mean()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'onset_features.parquet'
    win=pd.read_parquet(ROOT/'data/interim/score_sprint_20260921/recency_chain_windows.parquet')
    win=win[win.condition.eq('queue_onset')]
    panels=sorted(win.panel.unique())
    dates=sorted({p.stem.replace('synthetic_mainline_','').replace('_','-') for p in R.truth_files(panels[0])})
    if cache.exists(): d=pd.read_parquet(cache)
    else:
        parts=[]
        for panel,w in win.groupby('panel'):
            t=time.time()
            f=OOF.base_features(w[['panel','date','origin','condition']],oblique=True,dispersion=True,clock=True,origin_obs=True,split='train')
            f=f[f.step.eq(6)].copy();parts.append(f)
            print('features',panel,len(f),'seconds',round(time.time()-t,1),flush=True)
        d=pd.concat(parts,ignore_index=True);d.attrs={};d.to_parquet(cache,index=False)
    linkmap={}
    for panel in panels:
        for link in sorted(d.loc[d.panel.eq(panel),'link_id'].astype(str).unique()):linkmap[(panel,link)]=len(linkmap)
    identity(d,panels,linkmap)
    feats=[c for c in d if c not in OOF.DROP and not c.startswith('_')]
    # Prior is already a base column; conditional-hazard features are arm-specific.
    categorical=['panel_code','link_code']
    dump('design.json',dict(arms=ARMS,params=PARAMS,cases=CASES,features=feats,categorical=categorical,prior_smoothing={'link_pseudocount':20,'tod_pseudocount':10,'time_kernel':[.25,.5,1,.5,.25]},windows=len(win),rows=len(d)))
    all_results=[]
    for case,fit_end,esm,calm,evm in CASES:
        fitdates=[x for x in dates if x<=fit_end]
        d['queue_prior']=PR.apply(d,PR.build(panels,PR.PriorSpec(tuple(fitdates))))
        PR.clear_cache();gc.collect()
        hazard=Hazard(d,fitdates);hazard.apply(d)
        isfit=d.date.le(fit_end); ises=d.date.str.startswith(esm); iscal=d.date.str.startswith(calm)
        tr=d[isfit].reset_index(drop=True);va=d[ises].reset_index(drop=True);ca=d[iscal].reset_index(drop=True)
        ev=d[d.date.str.startswith(evm)].reset_index(drop=True) if evm else None
        assert set(tr.date).isdisjoint(va.date) and set(tr.date).isdisjoint(ca.date)
        print('case',case,'fit/es/cal/eval',len(tr),len(va),len(ca),len(ev) if ev is not None else 0,flush=True)
        for arm in ARMS:
            fs=feats+(['onset_hazard','onset_link_rate','onset_hazard_share'] if arm=='feature_hazard' else [])
            kw1={'init_score':OOF.logit(tr.queue_prior.to_numpy())} if arm=='offset' else {}
            kw2={'init_score':OOF.logit(va.queue_prior.to_numpy())} if arm=='offset' else {}
            dt=lgb.Dataset(tr[fs],tr.label,weight=OOF.window_weights(tr),categorical_feature=categorical,**kw1)
            dv=lgb.Dataset(va[fs],va.label,weight=OOF.window_weights(va),categorical_feature=categorical,reference=dt,**kw2)
            t=time.time()
            model=lgb.train(PARAMS,dt,num_boost_round=1000,valid_sets=[dv],callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
            cp=predictions(model,ca,fs,arm);thr,cscore,grid=select_threshold(ca,cp)
            model.save_model(str(OUT/f'{case}_{arm}.txt'))
            ca[KEY].assign(prob=cp).to_parquet(OUT/f'{case}_{arm}_calib.parquet',index=False)
            result=dict(case=case,arm=arm,trees=model.current_iteration(),threshold=thr,calib_S_onset=cscore,seconds=round(time.time()-t,1),fit_end=fit_end)
            if ev is not None:
                ep=predictions(model,ev,fs,arm)
                f,diag=diagnostics(ev,ep,thr);result.update(diag)
                f.to_csv(OUT/f'{case}_{arm}_windows.csv',index=False)
                ev[KEY+['queue_prior']].assign(prob=ep).to_parquet(OUT/f'{case}_{arm}_eval.parquet',index=False)
            else:
                serve=[]
                for split in ['validation','private']:
                    for panel in panels:
                        s=TI.features_for_split(panel,split,oblique=True,dispersion=True,clock=True,origin_obs=True,prior_dates=tuple(fitdates))
                        s=s[s.condition.eq('queue_onset') & s.step.eq(6)].copy().reset_index(drop=True)
                        # Serving's explicit cell identities map to training's physical link axis.
                        axis=d[d.panel.eq(panel)].drop_duplicates('link_id').set_index('link_id')._link_idx.to_dict()
                        s['_link_idx']=s.link_id.map(axis).astype(int)
                        ts=pd.to_datetime(s.timestamp,utc=True);s['_tod_idx']=ts.dt.hour*12+ts.dt.minute//5
                        s['date']=ts.dt.strftime('%Y-%m-%d');s['split']=split
                        identity(s,panels,linkmap);hazard.apply(s)
                        sp=predictions(model,s,fs,arm)
                        serve.append(s[['window_id','timestamp','link_id','panel','split','step','queue_prior']].assign(prob=sp,queue_pred=decode(s,sp,thr).astype(int)))
                pd.concat(serve,ignore_index=True).to_parquet(OUT/f'final_{arm}_serving.parquet',index=False)
                dump(f'final_{arm}_manifest.json',dict(arm=arm,features=fs,threshold=thr,model=f'{case}_{arm}.txt',prior_dates=fitdates,calibration_month=calm,es_month=esm,refit_after_calibration=False,link_codes=[dict(panel=p,link=l,code=c) for (p,l),c in linkmap.items()]))
            all_results.append(result);dump('results.json',all_results);print(result,flush=True)
            del model,dt,dv;gc.collect()
        del hazard,tr,va,ca,ev;gc.collect()


if __name__=='__main__':main()
