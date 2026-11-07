"""RLB capacity experiments, date-isolated stopping and selection."""
# Training record for the CAP CatBoost heads. The reproduction does not run this program; it
# loads the shipped models. The program reads feature tables that are not shipped, and its
# paths resolve from its original place in the development repository (`common.ROOT`).
#
# Trains second residual learners with the RLB targets: the error of the base Task 1 state
#   C1 in speed (km/h, `y_v`) and per-lane flow (veh/h per lane, `y_q`), from the 130 RLB
#   inputs. `SPECS` holds the declared candidates: name -> (family, rounds, extra settings).
# Data: the RL feature tables (`build/review_20260924_ctm/rl/feats/`) of January 2031
#   (`prepare`) and of January + February 2031 (`prepare_final`), 8 Task 2 corridors.
#   Free-flow cells (C1 speed >= 0.85 x free-flow speed) are kept at 30% (rng seed 0) with
#   weight 1/0.3. `prepare` splits January by date: fit 2031-01-04..17 and stop 01-18..24
#   (8 corridors, sampled), select 01-25..31 (all 10 corridors, not sampled), fullfit
#   01-04..31 (8 corridors, sampled).
# Commands (first argument):
#   prepare        writes `state/X_<split>.npy`, `yw_<split>.npz`, `data_<split>.parquet` and
#                  `state/data.json` (input list and split record);
#   prepare_final  writes `state/X_finalfit.npy`, `yw_finalfit.npz` and `final_data.json`,
#                  with the rows of `rl/fit_final.py` (5,952,690 before the sample);
#   <SPECS name>   trains one candidate on fit, records the loss on the stop days and saves
#                  checkpoints with their predictions on the select days;
#   selected --refit | --final
#                  retrains the choice of `state/selection.json` with fixed rounds on fullfit
#                  (January) or finalfit (January + February).
# Shipped (under `trafficflowbench/build/model_capacity_20260927/`): `state/data.json` and
#   `state/C6_long_final/y_v.cbm`, `y_q.cbm`, `trial.json` from `selected --final`: CatBoost
#   RMSE, depth 6, 4,800 iterations, learning rate .05, l2_leaf_reg 10, border_count 128,
#   random_seed 0, 6 threads, no stopping set. `trial.json` stores the SHA256 of this program
#   (`source_sha`). `state_pipeline.py` ran the refit and final steps.
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='3'
import sys,time,gc,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'packages'))
import numpy as np,pandas as pd,lightgbm as lgb
from common import HERE,ROOT,dump,sha,stopping_point,test_stopping,MinimumBudget,lgb_prefix_predictions
RL=ROOT/'build/review_20260924_ctm/rl';sys.path.insert(0,str(RL));import rl_eval as R,rleval as E
OUT=HERE/'state';OUT.mkdir(exist_ok=True)
# Declared candidates: LightGBM (base RLB recipe, slower, larger), CatBoost depth 6 with
# 2,400 or 4,800 iterations, XGBoost depth 8.
SPECS={'L_base':('lgb',2400,{}),'L_slow':('lgb',4000,dict(learning_rate=.02)),
       'L_large':('lgb',2400,dict(num_leaves=127,min_data_in_leaf=150,lambda_l2=10.,max_bin=511)),
       'C6':('cat',2400,dict(depth=6)),'C6_long':('cat',4800,dict(depth=6)),'X8':('xgb',2400,dict(max_depth=8))}

def prepare():
    """Writes the January 2031 split arrays (fit, stop, select, fullfit) and `state/data.json`."""
    if (OUT/'data.json').exists():return
    d=R.load_month('2031_01',True);fs=R.featsets()['RLB'];date=np.empty(len(d),dtype=object)
    for pi,p in enumerate(E.PANELS10):
        m=json.loads((RL/f'cache/{p}_2031_01_meta.json').read_text());ix=np.flatnonzero(d.panel_i.to_numpy()==pi)
        date[ix]=np.array(m['dates'])[d.day.to_numpy()[ix].astype(int)]
    # Free-flow cells (C1 speed / free-flow speed >= .85) are kept at 30%.
    d['date']=date;free=d.p_r.to_numpy()>=.85;u=np.random.default_rng(0).random(len(d));sample=~free|(u<.3)
    masks={'fit':(date<='2031-01-17')&(d.panel_i.to_numpy()<8)&sample,
           'stop':(date>='2031-01-18')&(date<='2031-01-24')&(d.panel_i.to_numpy()<8)&sample,
           'select':date>='2031-01-25','fullfit':(d.panel_i.to_numpy()<8)&sample}
    info={}
    for name,m in masks.items():
        z=d[m].reset_index(drop=True);z.to_parquet(OUT/f'data_{name}.parquet',index=False)
        np.save(OUT/f'X_{name}.npy',z[fs].to_numpy('float32'))
        np.savez(OUT/f'yw_{name}.npz',y_v=z.y_v.to_numpy('float32'),y_q=z.y_q.to_numpy('float32'),w=np.where(z.p_r>=.85,1/.3,1).astype('float32'))
        info[name]=dict(rows=len(z),dates=sorted(z.date.unique()),panels=z.groupby('panel_i').size().to_dict())
        print('prepared state',name,len(z),flush=True)
    dump(OUT/'data.json',dict(features=fs,splits=info,recipe=R.P,free_fraction=.3,subsampling_seed=0,source_hashes={p.name:sha(p)for p in sorted((RL/'feats').glob('*2031_01.parquet'))if not p.name.startswith(('._','selfsup'))}))

def prepare_final():
    """Exactly the old RLB_final row order/RNG, with bounded-memory panel reads."""
    # One RNG stream over months, then corridors, as in rl/fit_final.py, so the kept rows
    # equal the RLB_final rows. Rows are written into a memory-mapped array.
    if (OUT/'final_data.json').exists():return
    fs=R.featsets()['RLB'];rng=np.random.default_rng(0);pieces=[];n=0;rawrows=0
    for month in ['2031_01','2031_02']:
        for p in E.PANELS:
            d=pd.read_parquet(RL/f'feats/{p}_{month}.parquet',columns=['p_r']);u=rng.random(len(d));mask=(d.p_r.to_numpy()<.85)|(u<.3)
            pieces.append((month,p,mask));n+=int(mask.sum());rawrows+=len(d)
    assert rawrows==5952690,rawrows
    x=np.lib.format.open_memmap(OUT/'X_finalfit.npy',mode='w+',dtype='float32',shape=(n,len(fs)))
    yy={y:np.empty(n,dtype='float32')for y in ['y_v','y_q','w']};offset=0
    for month,p,mask in pieces:
        d=pd.read_parquet(RL/f'feats/{p}_{month}.parquet');b=pd.read_parquet(RL/f'feats/blk_{p}_{month}.parquet');d=pd.concat([d,b],axis=1).loc[mask];stop=offset+len(d)
        x[offset:stop]=d[fs].to_numpy('float32')
        for y in ['y_v','y_q']:yy[y][offset:stop]=d[y].to_numpy()
        yy['w'][offset:stop]=np.where(d.p_r>=.85,1/.3,1)
        print('production training data',month,p,len(d),flush=True);offset=stop;del d,b;gc.collect()
    assert offset==n;x.flush();np.savez(OUT/'yw_finalfit.npz',**yy)
    dump(OUT/'final_data.json',dict(rows_before_subsample=rawrows,fit_rows=n,months=['2031_01','2031_02'],panels=E.PANELS,seed=0,free_sample_fraction=.3,order='month then original PANELS order; continuous RNG stream; identical to fit_final.py',features=fs))

def run(name,refit=False,rounds=None,production=False):
    """Trains one declared candidate, or refits the selected one with fixed rounds.

    A trial trains on the fit split, records the loss curve on the stop split and saves
    checkpoints ('400', '1200', 'cap' = last round trained, 'ordinary' = best round with
    patience 150, 'minimum' = the same with at least 600 rounds) with their predictions on
    the select split. A refit trains the given rounds with no stopping set. Returns at once
    when the target folder already has `trial.json`.

    Args:
        name: Key of `SPECS`.
        refit: Train with fixed rounds on fullfit or finalfit.
        rounds: For refits, rounds per target, {"y_v": n, "y_q": n}.
        production: With `refit`, train on finalfit (January + February) into
            `state/<name>_final`; otherwise fullfit (January) into `state/<name>_refit`.
    """
    folder=OUT/(name+('_final'if production else'_refit'if refit else''));folder.mkdir(exist_ok=True)
    if (folder/'trial.json').exists():print('completed state',folder.name);return
    family,nr,extra=SPECS[name];info=json.loads((OUT/'data.json').read_text());fs=info['features'];t=time.time()
    train_split='finalfit'if production else'fullfit'if refit else'fit'
    x=np.load(OUT/f'X_{train_split}.npy',mmap_mode='r');a=np.load(OUT/f'yw_{train_split}.npz')
    s=np.load(OUT/'X_stop.npy',mmap_mode='r');b=np.load(OUT/'yw_stop.npz');xx=np.load(OUT/'X_select.npy',mmap_mode='r');allmeta={}
    for y in (list(rounds) if refit else ['y_v','y_q']):
        # Threads: 3 for trials, 6 for refits; a production refit uses 9 when
        # `queue/adoption.json` exists and either declines promotion or
        # `queue/serving/report.json` exists. The shipped final records 6.
        threads=6 if refit else 3
        qa=HERE/'queue/adoption.json'
        if production and qa.exists():
            decision=json.loads(qa.read_text())
            if not decision['promote']or(HERE/'queue/serving/report.json').exists():threads=9
        params=dict(R.P,**extra);params.update(metric='l2',feature_pre_filter=False,num_threads=threads)
        nround=rounds[y]if refit else nr;hist={};tick=time.time()
        if family=='lgb':
            dt=lgb.Dataset(x,label=a[y],weight=a['w'],feature_name=fs,params={'feature_pre_filter':False})
            valid=[]if refit else[dt,lgb.Dataset(s,label=b[y],weight=b['w'],reference=dt)]
            cb=[lgb.record_evaluation(hist),lgb.log_evaluation(200)]
            if not refit and name!='L_base':cb.append(MinimumBudget(600,150))
            m=lgb.train(params,dt,num_boost_round=nround,valid_sets=valid,valid_names=[]if refit else['fit','stop'],callbacks=cb,keep_training_booster=True)
            if refit:ordinary=minimum=dict(best=nround,executed=nround,reason='frozen_refit',loss=0.);checkpoints={'chosen':nround}
            else:
                ordinary=stopping_point(hist['stop']['l2'],1,150);minimum=stopping_point(hist['stop']['l2'],600,150)
                nround=len(hist['stop']['l2']);checkpoints={'400':min(400,nround),'1200':min(1200,nround),'cap':nround,'ordinary':ordinary['best'],'minimum':minimum['best']}
            cached_predictions={}if refit else lgb_prefix_predictions(m,xx,checkpoints.values())
            for key,it in checkpoints.items():
                m.save_model(str(folder/f'{y}_{key}.txt'),num_iteration=it)
                if not refit:
                    np.save(folder/f'select_{y}_{key}.npy',cached_predictions[it])
            it=checkpoints['chosen'if refit else'minimum'];reload=lgb.Booster(model_file=str(folder/f'{y}_{"chosen"if refit else"minimum"}.txt'))
            diff=float(np.max(abs(m.predict(s[:1024],num_iteration=it,num_threads=3)-reload.predict(s[:1024],num_threads=3))))
            del dt,valid,reload,m
        elif family=='cat':
            from catboost import CatBoostRegressor,Pool
            params=dict(iterations=nround,depth=extra['depth'],learning_rate=.05,l2_leaf_reg=10,loss_function='RMSE',random_seed=0,thread_count=threads,allow_writing_files=False,border_count=128,verbose=200)
            m=CatBoostRegressor(**params);m.fit(Pool(x,a[y],weight=a['w']),eval_set=None if refit else Pool(s,b[y],weight=b['w']),use_best_model=False)
            hist=m.get_evals_result()
            if refit:ordinary=minimum=dict(best=nround,executed=nround,reason='frozen_refit',loss=0.);checkpoints={'chosen':nround}
            else:
                ordinary=stopping_point(hist['validation']['RMSE'],1,150);minimum=stopping_point(hist['validation']['RMSE'],600,150);checkpoints={'400':400,'1200':1200,'cap':nr,'ordinary':ordinary['best'],'minimum':minimum['best']}
            m.save_model(str(folder/f'{y}.cbm'))
            for key,it in checkpoints.items():
                if not refit:np.save(folder/f'select_{y}_{key}.npy',m.predict(xx,ntree_end=it,thread_count=3))
            reload=CatBoostRegressor();reload.load_model(str(folder/f'{y}.cbm'));diff=float(np.max(abs(m.predict(s[:1024])-reload.predict(s[:1024]))));del m,reload
        else:
            import xgboost as xgb
            params=dict(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',max_depth=extra['max_depth'],eta=.05,min_child_weight=150,reg_lambda=10,subsample=.8,colsample_bytree=.8,max_bin=256,nthread=threads,seed=0)
            dt=xgb.DMatrix(x,label=a[y],weight=a['w']);ds=xgb.DMatrix(s,label=b[y],weight=b['w'])
            m=xgb.train(params,dt,num_boost_round=nround,evals=[]if refit else[(ds,'stop')],evals_result=hist,verbose_eval=200)
            if refit:ordinary=minimum=dict(best=nround,executed=nround,reason='frozen_refit',loss=0.);checkpoints={'chosen':nround}
            else:
                ordinary=stopping_point(hist['stop']['rmse'],1,150);minimum=stopping_point(hist['stop']['rmse'],600,150);checkpoints={'400':400,'1200':1200,'cap':nr,'ordinary':ordinary['best'],'minimum':minimum['best']}
            m.save_model(folder/f'{y}.ubj')
            if not refit:
                dm=xgb.DMatrix(xx)
                for key,it in checkpoints.items():np.save(folder/f'select_{y}_{key}.npy',m.predict(dm,iteration_range=(0,it)))
                del dm
            reload=xgb.Booster();reload.load_model(folder/f'{y}.ubj');diff=float(np.max(abs(m.predict(xgb.DMatrix(s[:1024]))-reload.predict(xgb.DMatrix(s[:1024])))));del m,dt,ds,reload
        # The saved model must predict the first 1,024 stop rows exactly as the trained one.
        assert diff<1e-12
        dump(folder/f'{y}_curve.json',hist);allmeta[y]=dict(params=params,completed=nround,checkpoints=checkpoints,ordinary=ordinary,minimum_budget=minimum,reload_max_diff=diff,seconds=time.time()-tick)
        print('STATE HEAD',name,y,allmeta[y],flush=True);gc.collect()
    dump(folder/'trial.json',dict(family=family,name=name,refit=refit,production=production,train_split=train_split,heads=allmeta,source_sha=sha(__file__),seconds=time.time()-t,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

# Usage: state_train.py prepare | prepare_final | <SPECS name> | selected --refit | selected --final
# For --refit/--final, `selection.json` names one candidate, or one per target in `head_sources`.
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='prepare_final':prepare_final()
    elif '--refit'in sys.argv or '--final'in sys.argv:
        choice=json.loads((OUT/'selection.json').read_text())
        if 'head_sources'in choice:
            for name in sorted({v['name']for v in choice['head_sources'].values()}):
                rounds={y:v['rounds']for y,v in choice['head_sources'].items()if v['name']==name};run(name,True,rounds,production='--final'in sys.argv)
        else:run(choice['name'],True,choice['rounds'],production='--final'in sys.argv)
    else:run(sys.argv[1])
