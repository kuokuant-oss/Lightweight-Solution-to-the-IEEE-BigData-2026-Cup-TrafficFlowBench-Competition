"""Matched onset-only prior experiment and fixed-model serving candidate.

Three predeclared arms share clock, explicit identity, origin-T observations,
step-6 labels, dates, tree recipe and calibration. Only the prior treatment
changes. January and February are forward evaluations; final fit ends in
December, early stopping uses January, calibration uses February. The SAME
calibrated model is served, avoiding full-refit probability-scale changes.
No network calls, no submission, no changes to champion artifacts.
"""

# Role: training record of the 0922 onset head ("M1" in pipeline/onset.sh).
# The reproduction does not run main(); pipeline/onset_head_serve.py imports
# Hazard, identity, predictions and decode from this file and serves the saved
# `final_feature_hazard` model.
# Arms: `offset` adds logit(queue_prior) to the trees' raw score; `feature`
# uses queue_prior as an input column; `feature_hazard` also adds the three
# conditional onset-hazard columns built by Hazard.
# Inputs: train onset windows (recency_chain_windows.parquet) and their
# features at step 6 (T+30), one row per window and link. Outputs under OUT:
# models `<case>_<arm>.txt`, calibration and evaluation predictions,
# results.json and, for case `final`, `final_<arm>_serving.parquet` with its
# manifest.

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

# CASES: (name, last fit date, early-stopping month, calibration month,
# evaluation month or None). `final` fits through 2030-12-31, stops early on
# January 2031, calibrates on February 2031 and then serves validation and
# private. GRID: decision thresholds tried on the calibration month. KEY:
# identity and label columns saved with every prediction table.
OUT = ROOT / 'build/final_sprint_review_20260922/prior'
ARMS = ['offset', 'feature', 'feature_hazard']
PARAMS = dict(OOF.PARAMS, num_threads=4, num_leaves=127, max_bin=127, seed=20260922)
CASES = [('jan', '2030-10-31', '2030-11', '2030-12', '2031-01'),
         ('feb', '2030-11-30', '2030-12', '2031-01', '2031-02'),
         ('final', '2030-12-31', '2031-01', '2031-02', None)]
GRID = np.round(np.arange(.05, .801, .025), 3)
KEY = ['window_id', 'panel', 'date', 'condition', 'link_id', 'step', 'label']


def dump(name, obj):
    """Writes `obj` as indented UTF-8 JSON to OUT/name."""
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def identity(d, panels, linkmap):
    """Adds integer category codes for panel and link (LightGBM categorical features).

    Args:
        d: rows with `panel` and `link_id`; changed in place.
        panels: sorted panel list; a panel's code is its position.
        linkmap: (panel, link_id) -> code; training and serving use the same map.

    Returns:
        `d` with int32 columns `panel_code` and `link_code`.
    """
    d['panel_code'] = d.panel.map({p:i for i,p in enumerate(panels)}).astype('int32')
    d['link_code'] = [linkmap[(p,str(l))] for p,l in zip(d.panel,d.link_id)]
    d['link_code'] = d.link_code.astype('int32')
    return d


class Hazard:
    """Fit-date, LOO-within-fit conditional onset frequency with partial pooling."""
    # Per panel it keeps counts over (link x 288 five-minute time-of-day slots):
    # queued step-6 labels (num) and rows (den) over the fit dates, plus each fit
    # day's own counts so that a row on a fit day can leave its day out.
    def __init__(self, d, dates):
        """Counts step-6 onset labels per (link, time-of-day slot) for each panel.

        Args:
            d: onset rows with panel, date, label, `_link_idx` (position on the
                panel's physical link axis) and `_tod_idx` (5-minute slot of
                the day, 0..287).
            dates: fit dates; only their labels are counted.
        """
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
        """Turns counts into a smoothed onset rate per (link, time-of-day slot).

        The link rate is shrunk toward the panel-wide rate with 20 pseudo-rows.
        The (link, slot) counts are smoothed over +-2 neighbouring slots
        (weights .25, .5, 1, .5, .25, wrapping at midnight) and shrunk toward
        the link rate with 10 pseudo-rows.

        Args:
            num: (n_links, 288) counts of queued labels.
            den: (n_links, 288) counts of rows.

        Returns:
            (rate per link and slot, rate per link).
        """
        global_p = float((num.sum()+1)/(den.sum()+2))
        ln, ld = num.sum(axis=1), den.sum(axis=1)
        link_p = (ln+20*global_p)/(ld+20)
        kernel = np.array([.25,.5,1,.5,.25])
        n = convolve1d(num,kernel,axis=1,mode='wrap')
        z = convolve1d(den,kernel,axis=1,mode='wrap')
        return (n+10*link_p[:,None])/(z+10), link_p

    def apply(self, d):
        """Adds the hazard features to `d` in place.

        A row whose date is a fit date reads rates computed without that day
        (leave-one-day-out); rows of any other date (calibration, evaluation,
        serving) read the rates of all fit dates.

        Returns:
            `d` with `onset_hazard` (rate at the row's link and slot),
            `onset_link_rate` (rate of the row's link) and `onset_hazard_share`
            (the row's rate divided by the sum of the rates of all links of the
            panel at that slot).
        """
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
    """Returns queue probabilities for rows `d`; the `offset` arm adds logit(queue_prior) to the raw score."""
    raw = model.predict(d[feats],raw_score=True)
    if arm == 'offset': raw += OOF.logit(d.queue_prior.to_numpy())
    return expit(raw)


def decode(d, prob, thr):
    """Marks cells with probability >= thr as queued; a window with none gets its single most probable cell.

    Returns:
        Boolean decision per row of `d`.
    """
    y=prob>=thr
    for ix in d.groupby('window_id',sort=False).indices.values():
        if not y[ix].any(): y[ix[np.argmax(prob[ix])]]=True
    return y


def select_threshold(d, p):
    """Picks the GRID threshold with the best official-style onset score on `d`.

    Returns:
        (threshold, best score, list of (score, threshold) for every grid value).
    """
    rows=[]
    for t in GRID:
        f=TE.iou_frame(d,decode(d,p,t))
        rows.append((TE.s_queue(f),float(t)))
    score,thr=max(rows,key=lambda x:x[0])
    return thr,score,rows


def diagnostics(d, p, thr):
    """Scores a decoded evaluation month and adds ranking diagnostics.

    Returns:
        (per-window frame, summary). The frame has each window's IoU plus
        top1_hit (the most probable cell is queued), oracle (best IoU over
        probability-ranked prefixes), maxprob and the prior at the top cell.
        The summary has S_onset, the top-1 hit rate, the mean oracle IoU and
        the count and mean IoU of low-confidence windows (max probability < .1).
    """
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
    """Fits, calibrates and evaluates (or, for case `final`, serves) every arm for each case."""
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'onset_features.parquet'
    win=pd.read_parquet(ROOT/'data/interim/score_sprint_20260921/recency_chain_windows.parquet')
    win=win[win.condition.eq('queue_onset')]
    panels=sorted(win.panel.unique())
    # Train days, from the day file names of the first panel's unmasked layer.
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
    # One integer code per (panel, link), shared by training and serving.
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
        # Queue prior and hazard from the fit dates only; rows on fit dates read leave-one-day-out values.
        d['queue_prior']=PR.apply(d,PR.build(panels,PR.PriorSpec(tuple(fitdates))))
        PR.clear_cache();gc.collect()
        hazard=Hazard(d,fitdates);hazard.apply(d)
        # Row roles: fit (<= fit_end), early stopping, calibration and evaluation months.
        isfit=d.date.le(fit_end); ises=d.date.str.startswith(esm); iscal=d.date.str.startswith(calm)
        tr=d[isfit].reset_index(drop=True);va=d[ises].reset_index(drop=True);ca=d[iscal].reset_index(drop=True)
        ev=d[d.date.str.startswith(evm)].reset_index(drop=True) if evm else None
        assert set(tr.date).isdisjoint(va.date) and set(tr.date).isdisjoint(ca.date)
        print('case',case,'fit/es/cal/eval',len(tr),len(va),len(ca),len(ev) if ev is not None else 0,flush=True)
        for arm in ARMS:
            fs=feats+(['onset_hazard','onset_link_rate','onset_hazard_share'] if arm=='feature_hazard' else [])
            # The offset arm starts boosting from logit(queue_prior) (init_score).
            kw1={'init_score':OOF.logit(tr.queue_prior.to_numpy())} if arm=='offset' else {}
            kw2={'init_score':OOF.logit(va.queue_prior.to_numpy())} if arm=='offset' else {}
            dt=lgb.Dataset(tr[fs],tr.label,weight=OOF.window_weights(tr),categorical_feature=categorical,**kw1)
            dv=lgb.Dataset(va[fs],va.label,weight=OOF.window_weights(va),categorical_feature=categorical,reference=dt,**kw2)
            t=time.time()
            model=lgb.train(PARAMS,dt,num_boost_round=1000,valid_sets=[dv],callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
            # Threshold chosen on the calibration month; the model is not refit afterwards.
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
                # Case `final`: predict and decode every released validation/private onset window at step 6.
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
