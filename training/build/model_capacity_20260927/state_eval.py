"""Select paired v/q capacity changes with full target/physics scoring."""
# Training record for the CAP selection. The reproduction does not run this program; it
# reads the shipped `state/selection.json`. The program reads the January arrays of
# `state_train.py` (not shipped), and its paths resolve from its original place in the
# development repository (`common.ROOT`).
#
# Scores every January candidate of `state_train.py` (each checkpoint, plus the per-target
#   best heads) on the select days 2031-01-25..31, all 10 corridors. The parent is L_base at
#   400 rounds (the RLB recipe fitted on 2031-01-04..17). Each candidate is served as
#   parent + alpha x (candidate - parent) for alpha in .1, .25, .5, 1 (plus speed-gated
#   weights for L_large__cap), with the k* projection on its corridors; the D12_I405 pair
#   keeps the parent. Scores: local Task 1 S_state and Task 3 physics (`rleval.evaluate`);
#   rank by total_B = .35 S_state + .15 S_phys_B (physics on masked-layer cells and targets)
#   as a change against the parent.
# Output: `state/selection_grid.csv`, `selection_days.parquet`, `stopping_head_choices.json`,
#   `fullpath_head_choices.json` and, with `--freeze`, `state/selection.json` (the top
#   candidate). Shipped: `trafficflowbench/build/model_capacity_20260927/state/selection.json`
#   (C6_long, checkpoint cap: 4,800 rounds for both targets, alpha 0.5). February is not
#   scored here.
# Usage: `python state_eval.py [--freeze]` (`state_pipeline.py` runs it with `--freeze`).
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='3'
import sys,json,gc
import numpy as np,pandas as pd
from common import HERE,ROOT,dump,state_alpha
OUT=HERE/'state';RL=ROOT/'build/review_20260924_ctm/rl'
sys.path.insert(0,str(HERE/'packages'))
sys.path.insert(0,str(RL));import rl_eval as R,rleval as E

def fullpath_heads(trials):
    """Picks, per target, the trial and round with the lowest stop-day loss over the whole curve.

    Args:
        trials: Trial folder name -> parsed `trial.json`.

    Returns:
        (heads, preds): per target the choice {name, rounds, stop_mse}, and its predictions
        on the select split (loaded from a saved checkpoint or recomputed).
    """
    heads={};preds={}
    for y in ['y_v','y_q']:
        choices=[]
        for name,t in trials.items():
            h=json.loads((OUT/name/f'{y}_curve.json').read_text());split='validation'if t['family']=='cat'else'stop';loss=np.asarray(h[split][next(iter(h[split]))]);i=int(np.argmin(loss));mse=float(loss[i]if t['family']=='lgb'else loss[i]**2)
            choices.append(dict(name=name,rounds=i+1,stop_mse=mse))
        c=min(choices,key=lambda x:x['stop_mse']);heads[y]=c;t=trials[c['name']];folder=OUT/c['name'];it=c['rounds']
        match=next((k for k,v in t['heads'][y]['checkpoints'].items()if v==it),None)
        if match is not None:preds[y]=np.load(folder/f'select_{y}_{match}.npy');continue
        x=np.load(OUT/'X_select.npy',mmap_mode='r')
        if t['family']=='cat':
            from catboost import CatBoostRegressor
            m=CatBoostRegressor();m.load_model(str(folder/f'{y}.cbm'));preds[y]=m.predict(x,ntree_end=it,thread_count=3)
        elif t['family']=='xgb':
            import xgboost as xgb
            m=xgb.Booster();m.load_model(folder/f'{y}.ubj');preds[y]=m.predict(xgb.DMatrix(x),iteration_range=(0,it))
        else:
            import lightgbm as lgb
            m=lgb.Booster(model_file=str(folder/f'{y}_cap.txt'));preds[y]=m.predict(x,num_iteration=it,num_threads=3)
        np.save(folder/f'select_{y}_fullpath_best.npy',preds[y]);del m,x;gc.collect()
    return heads,preds

def compose(r,rv,rq,p,kh=None):
    """Adds the residuals to the base state and returns speed and per-lane flow.

    Speed is clipped to [6, 120.75] km/h and all-lane flow floored at 50 veh/h; with `kh`
    (k_aux densities) the k* projection of corridor `p` is applied.

    Args:
        r: Select rows of one corridor (base `p_v`, `p_q`, `lanes`).
        rv: Speed residuals (km/h).
        rq: Per-lane flow residuals (veh/h per lane).
        p: Corridor id.
        kh: k_aux density per row, or None.

    Returns:
        (speed km/h, per-lane flow veh/h per lane), rounded to 3 decimals.
    """
    ln=r.lanes.to_numpy(float);v=np.round(np.clip(r.p_v.to_numpy()+rv,6,120.75),3);q=np.round(np.maximum((r.p_q.to_numpy()+rq)*ln,50),3)
    if kh is not None:v,q=R.kstar(v,q,kh,ln,R.SEL['panel_alphas'][p],R.SEL['weights'][p])
    return np.round(v,3),np.round(q,3)/ln

def select(freeze=False):
    """Scores all candidates on the select days; with `freeze`, writes `state/selection.json`.

    Args:
        freeze: Write the top candidate; refuses when `selection.json` already exists.
    """
    d=pd.read_parquet(OUT/'data_select.parquet');base={y:np.load(OUT/'L_base'/f'select_{y}_400.npy')for y in ['y_v','y_q']};names={};trials={}
    for folder in sorted(OUT.iterdir()):
        if not (folder/'trial.json').exists()or folder.name.endswith(('_refit','_final')):continue
        t=json.loads((folder/'trial.json').read_text());trials[folder.name]=t
        for key in t['heads']['y_v']['checkpoints']:
            name=folder.name+'__'+key;names[name]=dict(name=folder.name,checkpoint=key,rounds={y:t['heads'][y]['checkpoints'][key]for y in ['y_v','y_q']})
    preds={name:{y:np.load(OUT/c['name']/f'select_{y}_{c["checkpoint"]}.npy')for y in ['y_v','y_q']}for name,c in names.items()}
    heads={}
    for y in ['y_v','y_q']:
        def mse(n):
            """Returns trial n's stop-day MSE at its 'minimum' checkpoint (CatBoost logs RMSE)."""
            loss=trials[n]['heads'][y]['minimum_budget']['loss']
            return loss if trials[n]['family']=='lgb'else loss**2
        best=min(trials,key=mse)
        heads[y]=dict(name=best,rounds=trials[best]['heads'][y]['checkpoints']['minimum'],stop_mse=mse(best))
    if len({v['name']for v in heads.values()})>1:
        name='H_stopheads__minimum';names[name]=dict(name='H_stopheads',checkpoint='minimum',head_sources=heads,rounds={y:v['rounds']for y,v in heads.items()})
        preds[name]={y:np.load(OUT/v['name']/f'select_{y}_minimum.npy')for y,v in heads.items()}
    dump(OUT/'stopping_head_choices.json',heads)
    globalheads,globalpreds=fullpath_heads(trials)
    names['H_globalheads__global']=dict(name='H_globalheads',checkpoint='global',head_sources=globalheads,rounds={y:v['rounds']for y,v in globalheads.items()})
    preds['H_globalheads__global']=globalpreds;dump(OUT/'fullpath_head_choices.json',globalheads)
    acc=[]
    for pi,p in enumerate(E.PANELS10):
        ix=np.flatnonzero(d.panel_i.to_numpy()==pi);r=d.iloc[ix].reset_index(drop=True);days=sorted(r.day.unique().tolist());kh=None
        if p in R.KSTAR:
            z=np.load(RL/f'cache/{p}_2031_01_khat.npz');kh=np.concatenate([z[f'd{di}']for di in days]);assert len(kh)==len(r)
        pv,pq=compose(r,base['y_v'][ix],base['y_q'][ix],p,kh)
        cands={'parent':(pv,pq)}
        for name,c in names.items():
            v,q=compose(r,preds[name]['y_v'][ix],preds[name]['y_q'][ix],p,kh)
            if pi>=8:v,q=pv,pq
            policies=[.1,.25,.5,1.]+(['gate75','gate100']if name=='L_large__cap'else[])
            for policy in policies:
                alpha=state_alpha(policy,r.p_r.to_numpy())
                cands[f'{name}__{policy}']=(np.round(pv+alpha*(v-pv),3),np.round((pq+alpha*(q-pq))*r.lanes.to_numpy(),3)/r.lanes.to_numpy())
        rows=E.evaluate(p,'2031_01',days,cands,r[['day','li','ci']]);acc.extend([dict(row,panel=p)for row in rows]);print('state selection eval',p,flush=True);gc.collect()
    a=pd.DataFrame(acc);a.to_parquet(OUT/'selection_days.parquet',index=False);g,p,mean=E.summarize(acc,E.PANELS10)
    mean['total_A']=.35*mean.S_state+.15*mean.S_phys_A;mean['total_B']=.35*mean.S_state+.15*mean.S_phys_B
    delta=mean-mean.loc['parent'];delta=delta.sort_values('total_B',ascending=False);delta.to_csv(OUT/'selection_grid.csv');print(delta.head(20).to_string())
    if freeze:
        assert not(OUT/'selection.json').exists()
        key=next(k for k in delta.index if k!='parent');name,check,alpha=key.split('__');choice=names[name+'__'+check].copy();choice.update(alpha=alpha if alpha.startswith('gate')else float(alpha),selection_delta=delta.loc[key].to_dict(),selection_dates=sorted(d.date.unique()),candidate=key,role='January selection; February not scored')
        dump(OUT/'selection.json',choice)

if __name__=='__main__':select('--freeze'in sys.argv)
