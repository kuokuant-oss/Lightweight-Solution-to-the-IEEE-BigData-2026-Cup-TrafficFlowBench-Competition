"""Offline two-regime expected-speed pilot with matched T1 hole inputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import density_holes as D
from density_holes import np,pd,lgb,json,gc,S,R2,R3,G2,ROOT,acc,summarize,physics
DEST=Path(__file__).resolve().parent/'hole_mixture';DEST.mkdir(exist_ok=True)

def fit_predict(f,a,tag):
    queue=(f.y_v<=.6*f.f_free_speed_kmh).to_numpy();res=f.y_v-f.ch_v
    models=[]
    for kind,mask,target,obj in [('pqueue',np.ones(len(f),bool),queue,'binary'),('free',~queue,res,'l2'),('queued',queue,res,'l2')]:
        model=lgb.train(dict(S.PAR,objective=obj),lgb.Dataset(f.loc[mask,S.FEATS],np.asarray(target)[mask]),S.NR)
        model.save_model(str(DEST/f'{tag}_{kind}.txt'));models.append(model)
        print('fit',tag,kind,int(mask.sum()),flush=True)
    p,free,queued=[m.predict(a[S.FEATS],num_threads=3)for m in models]
    v=np.clip(a.ch_v.to_numpy()+(1-p)*free+p*queued,S.SPEED_FLOOR,S.SPEED_CAP)
    del models;gc.collect();return v,p

def bootstrap(k,old,new,n=1000):
    d=k.loc[k.is_target.astype(bool)&k.y_v.notna(),['panel','date','y_v',old,new]].copy()
    d['e0']=(d[old]-d.y_v)**2;d['e1']=(d[new]-d.y_v)**2
    group=d.groupby(['panel','date'])[['e0','e1']].sum();rng=np.random.default_rng(0)
    ix=rng.integers(0,len(group),(n,len(group)));a=group.e0.to_numpy();b=group.e1.to_numpy();values=b[ix].sum(1)/a[ix].sum(1)-1
    return {'target_cells':len(d),'rmse_base':float(np.sqrt(d.e0.mean())),'rmse_new':float(np.sqrt(d.e1.mean())),'relative_mse':float(d.e1.sum()/d.e0.sum()-1),'panelday_boot_ci':np.percentile(values,[2.5,97.5]).tolist(),'perpanel_relative_mse':d.groupby('panel').apply(lambda g:float(g.e1.sum()/g.e0.sum()-1),include_groups=False).to_dict()}

def run():
    A=pd.read_parquet(R3/'work/inworld_apply_rows.parquet');S.derive(A)
    F=S.load_ph([R2/'work'/f'ph_train_r2_{p}.parquet'for p in S.PANELS])
    windows=pd.concat([pd.read_csv(G2/f'eval_windows_{p}_2031_02.csv')[['panel','date','origin']]for p in S.PANELS]);holes={}
    for w in windows.itertuples():holes.setdefault((w.panel,w.date),[]).append(int(w.origin)+1)
    win=F[['panel','date','origin']].drop_duplicates();win['bad']=[any((o0+1-13<o+18)and(o<o0+1+18+12)for o in holes.get((p,d),[]))for p,d,o0 in zip(win.panel,win.date,win.origin)]
    F=F.merge(win,on=['panel','date','origin'],validate='many_to_one');F=F[(~F.bad)&F.obs].reset_index(drop=True)
    print('fit',len(F),'apply',len(A),'test_windows',len(windows),flush=True)
    base=np.load(R3/'verify/work/v5_inworld_preds.npz')
    K=A[['panel','date','origin','li','step_i','ch_ql','y_v','y_ql','is_target']].copy()
    K['serve_new'],K['serve_pqueue']=fit_predict(F,A,'serve')
    K['strict_new']=np.nan;K['strict_pqueue']=np.nan
    for odd in [True,False]:
        tr=(F.day%2==1).to_numpy()==odd;te=(A.day%2==1).to_numpy()!=odd
        v,p=fit_predict(F.loc[tr],A.loc[te],f'strict_fitodd{int(odd)}');K.loc[te,'strict_new']=v;K.loc[te,'strict_pqueue']=p
    results={'error':{}}
    for mode in ['serve','strict']:
        K[mode+'_base']=base[mode];K[mode+'_blend25']=.75*base[mode]+.25*K[mode+'_new']
        for tag in ['new','blend25']:
            result=bootstrap(K,mode+'_base',mode+'_'+tag);results['error'][mode+'_'+tag]=result;print(mode,tag,json.dumps(result),flush=True)
    K.to_parquet(DEST/'predictions.parquet',index=False);del F,A;gc.collect()
    rows=[]
    for panel in S.PANELS:
        z=dict(np.load(G2/f'c0_cache/{panel}_2031_02_v2.npz',allow_pickle=True));meta=json.loads(str(z.pop('meta')))
        par=physics.link_params(panel,meta['links']);keep=physics.scoreable_mask(panel,meta['links'],'train',meta['times'])
        lanes=np.maximum(par['lanes'],1.);ln=lanes[:,None];length=par['length_km'][:,None];vf=par['free_speed_kmh'][:,None]
        capl=par['capacity_vph'][:,None]/ln;kcl=par['critical_density'][:,None]/ln;kjl=par['k_jam'][:,None]/ln;c=K[K.panel==panel]
        for di,date in enumerate(meta['dates']):
            key=f'd{di}';a=c[c.date==date];li=a.li.to_numpy(int);ti=(a.origin+1+a.step_i).to_numpy(int)
            for name in [f'{mode}_{tag}'for mode in ['serve','strict']for tag in ['base','new','blend25']]:
                v=z[f'{key}_v0'].astype(float).copy();q=z[f'{key}_q0'].astype(float).copy();v[li,ti]=a[name];q[li,ti]=a.ch_ql.to_numpy()*lanes[li]
                rows.append(dict(panel=panel,date=date,regime=meta['regimes'][di],cand=name,**acc(v,q,z,key,length,ln,vf,capl,kcl,kjl,keep)))
        del z;gc.collect()
    days=pd.DataFrame(rows);days.to_parquet(DEST/'days.parquet',index=False)
    ref=pd.read_parquet(D.HERE/'inworld_days.parquet')
    for mode in ['serve','strict']:
        p=days[days.cand==mode+'_base'].merge(ref[ref.cand==mode+'_base'],on=['panel','date'],validate='one_to_one',suffixes=('_new','_old'))
        for col in ['sse_v_in','numB','denB','fdnB']:
            assert np.max(np.abs(p[col+'_new']-p[col+'_old'])/np.maximum(np.abs(p[col+'_old']),1e-9))<1e-5
        for tag in ['new','blend25']:
            name=mode+'_'+tag;pp={}
            for panel in S.PANELS:
                sub=days[days.panel==panel];a=summarize(sub[sub.cand==mode+'_base']);b=summarize(sub[sub.cand==name])
                row={k:float(b[k]-a[k])for k in ['S_state','S_phys_A','S_phys_B']};row['total_proxy_B']=.35*row['S_state']+.15*row['S_phys_B'];pp[panel]=row
            avg={k:.8*float(np.mean([r[k]for r in pp.values()]))for k in next(iter(pp.values()))}
            results[name]={'five_family_mean':avg,'perpanel':pp};print('total',name,json.dumps(avg),flush=True)
    (DEST/'results.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':run()
