"""January -> February density head on full RLB+k*+HMP50 reconstruction.

No February labels enter head fit. Report all prespecified strengths. LWR RHS
equals clean storage difference only as an explicit organizer-projection proxy.
"""
import os,sys,json,gc,time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='3'
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'density'
RL=ROOT/'build/review_20260924_ctm/rl';sys.path.insert(0,str(RL))
import rl_eval as R,rleval as E
import matched_background as MATCH
RECIPE=json.loads((HERE/'density_recipe.json').read_text())
def dump(path,d):path.write_text(json.dumps(d,indent=2,allow_nan=False))
def raw_parent(d,month):
    if month=='2031_01':
        a=np.load(OUT/'january_rlb_oof.npz');b=np.load(RL/'results/crossfit_2031_01_kstar_oof.npz')
        rv=np.where(d.panel_i<8,a['y_v'],b['RL_y_v']);rq=np.where(d.panel_i<8,a['y_q'],b['RL_y_q'])
    else:
        parts=[np.load(MATCH.OUT/f'{p}_parent.npz')for p in E.PANELS10]
        rv=np.concatenate([p['y_v']for p in parts]);rq=np.concatenate([p['y_q']for p in parts])
    v=np.clip(d.p_v.to_numpy()+rv,6,120.75);q=np.maximum(d.p_q.to_numpy()+rq,50/d.lanes.to_numpy())
    return v,q
def augment(d,v,q):
    # All features available at serving. No eligibility/truth/RHS in features.
    fs=list(R.featsets()['RLB']);d['parent_v']=v;d['parent_q']=q;d['parent_k']=q/v
    d['parent_dv']=v-d.p_v;d['parent_dq']=q-d.p_q
    extra=['parent_v','parent_q','parent_k','parent_dv','parent_dq']
    for axis in ['t','x']:
        for h in [-2,-1,1,2]:
            c=f'neighbor_k_{axis}{h:+d}'
            d[c]=(d.p_q+d[f'dq_{axis}{h:+d}'])/(d.p_v+d[f'dv_{axis}{h:+d}']).clip(lower=1)
            extra.append(c)
    return fs+extra
def weight(d,month):
    w=np.zeros(len(d));report=[]
    for pi,p in enumerate(E.PANELS10):
        z,meta=E.load(p,month);ln=np.maximum(z['ln'],1);L=z['par_length_km'];store={};dens={}
        for di in range(len(meta['dates'])):
            k=f'd{di}';ok=z[k+'_ok'];obs=np.isfinite(z[k+'_obsv'])&np.isfinite(z[k+'_obsq']);rt=z[k+'_rt']&ok
            mem=ok&(rt|obs)&z['keep'][:,None];m=mem[:,:-1]&mem[:,1:]&z[k+'_rv'][:,:-1]
            degree=np.zeros(ok.shape);degree[:,:-1]+=m;degree[:,1:]+=m
            n=z[k+'_qt']/np.maximum(z[k+'_vt'],1)*L[:,None]
            den=float(np.abs(np.diff(n,axis=1))[m].sum());rg=meta['regimes'][di]
            dens[rg]=dens.get(rg,0)+den;store[di]=(degree,rg)
        for di,(degree,rg) in store.items():
            ix=d.index[(d.panel_i==pi)&(d.day==di)].to_numpy();r=d.loc[ix]
            w[ix]=degree[r.li.to_numpy(int),r.ci.to_numpy(int)]*L[r.li.to_numpy(int)]*ln[r.li.to_numpy(int)]/max(dens[rg],1e-9)
        report.append(dict(panel=p,denominators=dens))
    w*=len(w)/max(w.sum(),1e-9)
    dump(OUT/'training_weight_support.json',dict(support='B within-day transition degree; macro panel/regime denominator',panels=report,zero_weight=int((w==0).sum()),max_weight=float(w.max())))
    return w
def project(v,q,k,lanes,p,alpha):
    # q and k in per-lane units; use total-unit production projection.
    lanes=np.asarray(lanes,dtype=float)
    return R.kstar(v,q*lanes,k*lanes,lanes,alpha,R.SEL['weights'][p])
def parent_kstar(d,v,q,month):
    ov=v.copy();oq=q.copy()
    for pi,p in enumerate(E.PANELS10):
        ix=np.flatnonzero(d.panel_i.to_numpy()==pi);ln=d.lanes.to_numpy(float)[ix]
        vv=np.round(v[ix],3);qq=np.round(q[ix]*ln,3)
        if p in R.KSTAR:
            if month=='2031_02':kh=np.load(MATCH.OUT/f'{p}_parent.npz')['khat']
            else:
                z,meta=E.load(p,month);khz=np.load(RL/'cache'/f'{p}_{month}_khat.npz');kh=np.concatenate([khz[f'd{di}']for di in range(len(meta['dates']))])
            assert len(kh)==len(ix)
            vv,qq=R.kstar(vv,qq,kh,ln,R.SEL['panel_alphas'][p],R.SEL['weights'][p])
        ov[ix]=np.round(vv,3);oq[ix]=np.round(qq,3)/ln
    return ov,oq
def candidates(d,pv,pq,pred):
    c={'parent':(pv,pq)}
    for nm,kh in pred.items():
        for alpha in RECIPE['strengths']:
            v=pv.copy();q=pq.copy()
            for pi,p in enumerate(E.PANELS10[:8]):
                ix=np.flatnonzero(d.panel_i.to_numpy()==pi);ln=d.lanes.to_numpy()[ix]
                vv,qq=project(pv[ix],pq[ix],kh[ix],ln,p,alpha)
                v[ix]=np.round(vv,3);q[ix]=np.round(qq,3)/ln
            c[f'{nm}_{alpha}']=(v,q)
    return c
def summarize(a):
    g=a.groupby(['panel','cand','regime']).sum(numeric_only=True)
    g['S_state']=.54*(1-np.sqrt(g.ssev/g.n)/25).clip(lower=0)+.46*(1-np.sqrt(g.sseq/g.n)/600).clip(lower=0)
    for f in 'AB':
        g['S_FD_'+f]=(1-g['fdn'+f]/g['fdd'+f]).clip(lower=0)
        g['S_LWR_'+f]=(1-g['num'+f]/g['den'+f]).clip(lower=0)
        g['S_phys_'+f]=(g['S_FD_'+f]+2*g['S_LWR_'+f])/3
        g['total_'+f]=.35*g.S_state+.15*g['S_phys_'+f]
    metrics=['S_state']+[c+'_'+f for f in 'AB'for c in ['S_FD','S_LWR','S_phys','total']]
    p=g[metrics].groupby(['panel','cand']).mean().reset_index();p['family']=p.panel.map(E.FAM)
    mean=p.groupby(['family','cand'])[metrics].mean().groupby('cand').mean()
    return mean,p,g
def evaluate(d,cands):
    holes=pd.read_parquet(ROOT/'build/review_20260927_research/feb_pool/predictions.parquet');out=[];parity=[]
    for pi,p in enumerate(E.PANELS10):
        z,meta=MATCH.load(p,'2031_02');nd=len(meta['dates']);lii=np.flatnonzero(d.panel_i.to_numpy()==pi)
        ln=np.maximum(z['ln'],1)[:,None];L=z['par_length_km'][:,None];lanes=np.maximum(z['par_lanes'],1)[:,None]
        vf=z['par_free_speed_kmh'][:,None];cap=z['par_capacity_vph'][:,None]/lanes;kc=z['par_critical_density'][:,None]/lanes;kj=z['par_k_jam'][:,None]/lanes
        stack=lambda field:np.concatenate([z[f'd{i}_{field}']for i in range(nd)],axis=1)
        ok=stack('ok');rt=stack('rt')&ok;gm=stack('gm');vt=stack('vt').astype(float);qt=stack('qt').astype(float);obs=np.isfinite(stack('obsv'))&np.isfinite(stack('obsq'));rv=stack('rv')
        ntrue=qt/np.maximum(vt,1)*L;dt=np.diff(ntrue,axis=1)
        supports={'A':ok&np.isfinite(ntrue),'B':ok&np.isfinite(ntrue)&(rt|obs)}
        masks={}
        for f,mem in supports.items():
            mm=mem[:,:-1]&mem[:,1:]&z['keep'][:,None]&rv[:,:-1]
            for i in range(nd-1):
                if meta['regimes'][i]!=meta['regimes'][i+1] or (pd.Timestamp(meta['dates'][i+1])-pd.Timestamp(meta['dates'][i])).days!=1:mm[:,(i+1)*288-1]=False
            masks[f]=mm
        rr=d.iloc[lii];li=rr.li.to_numpy(int);ti=rr.day.to_numpy(int)*288+rr.ci.to_numpy(int)
        assert rt[li,ti].all() and not gm[li,ti].any()
        h=holes[holes.panel==p].copy();hv={};hq=stack('q1').astype(float);dmap={date:i for i,date in enumerate(meta['dates'])}
        if len(h):
            hli=h.li.to_numpy(int);hti=h.date.map(dmap).to_numpy(int)*288+h.origin.to_numpy(int)+1+h.step_i.to_numpy(int)
            assert gm[hli,hti].all(),f'{p}: hole grids differ'
            assert np.array_equal(rt[hli,hti],h.is_target.to_numpy(bool)),f'{p}: target masks differ'
            ev=np.abs(vt[hli,hti]-h.y_v.to_numpy());eq=np.abs(qt[hli,hti]/ln[hli,0]-h.y_ql.to_numpy())
            assert np.nanmax(ev)<1e-4 and np.nanmax(eq)<1e-3,(p,np.nanmax(ev),np.nanmax(eq))
            hq[hli,hti]=np.round(h.ch_ql.to_numpy()*ln[hli,0],3)
            for mode in ['strict','serve']:hv[mode]=np.round(2*h[mode+'_pool']-h[mode+'_t1'],3).to_numpy()
            parity.append(dict(panel=p,rows=len(h),max_truth_v=float(np.nanmax(ev)),max_truth_q_lane=float(np.nanmax(eq))))
        for mode in ['strict','serve']:
            for name,(cv,cq) in cands.items():
                v=stack('v1').astype(float);q=hq.copy()
                v[li,ti]=cv[lii];q[li,ti]=cq[lii]*ln[li,0]
                if len(h):v[hli,hti]=hv[mode]
                v=np.where(rt,v,vt);q=np.where(rt,q,qt)
                n=q/np.maximum(v,1)*L;err=n-ntrue;de=np.diff(err,axis=1)
                assert np.nanmax(np.abs(de-(np.diff(n,axis=1)-dt)))<1e-8
                kl=n/L/lanes;ql=q/lanes;fd=np.where(kl<=kc,vf*kl,cap/np.maximum(kj-kc,1e-9)*np.maximum(kj-kl,0))
                for di,date in enumerate(meta['dates']):
                    sl=slice(di*288,(di+1)*288);es=slice(di*288,min((di+1)*288,n.shape[1]-1));target=rt[:,sl]
                    row=dict(panel=p,date=date,regime=meta['regimes'][di],cand=name,mode=mode,n=int(target.sum()),ssev=float(((v[:,sl]-vt[:,sl])[target]**2).sum()),sseq=float((((q[:,sl]-qt[:,sl])/ln)[target]**2).sum()),n_hole=int((target&gm[:,sl]).sum()))
                    for f,mem in supports.items():
                        fm=mem[:,sl];tm=masks[f][:,es]
                        row.update({f'num{f}':float(np.abs(de[:,es])[tm].sum()),f'den{f}':float(np.abs(dt[:,es])[tm].sum()),f'trans{f}':int(tm.sum()),f'fdn{f}':float(np.abs(ql[:,sl]-fd[:,sl])[fm].sum()),f'fdd{f}':float(np.abs(ql[:,sl])[fm].sum()),f'fdrows{f}':int(fm.sum())})
                        rtleft=rt[:,:-1][:,es];rtright=rt[:,1:][:,es]
                        for kind,edge in [('tt',rtleft&rtright),('to',rtleft^rtright),('oo',~rtleft&~rtright)]:
                            take=tm&edge;row[f'num{f}_{kind}']=float(np.abs(de[:,es])[take].sum());row[f'n{f}_{kind}']=int(take.sum())
                    row['mae_storage_sum']=float(np.abs(err[:,sl])[target].sum());out.append(row)
        print('evaluated full HMP50',p,flush=True);gc.collect()
    a=pd.DataFrame(out);a.to_parquet(OUT/'feb_full_hmp50_days.parquet',index=False);dump(OUT/'full_parent_parity.json',parity)
    results={}
    for mode in ['strict','serve']:
        aa=a[a['mode']==mode];mean,p,g=summarize(aa);delta=mean-mean.loc['parent'];delta.to_csv(OUT/f'feb_{mode}_delta.csv');p.to_csv(OUT/f'feb_{mode}_perpanel.csv',index=False)
        results[mode]=dict(levels=mean.to_dict('index'),deltas=delta.to_dict('index'))
        print(mode,'full-parent deltas',delta[['S_state','S_LWR_A','S_LWR_B','total_A','total_B']].to_string(),flush=True)
    dump(OUT/'feb_results.json',results)
    return a,results
def uncertainty(a):
    a=a[a['mode']=='strict'];dates=sorted(a.date.unique());rng=np.random.default_rng(9);base,_p,_g=summarize(a);metrics=['total_A','total_B'];dist={n:[]for n in base.index if n!='parent'}
    # One whole calendar day draw shared across panels; retain family/regime macro.
    for b in range(400):
        selected=rng.choice(dates,len(dates),replace=True);counts=pd.Series(selected).value_counts();x=a.copy();w=x.date.map(counts).fillna(0).to_numpy();num=x.select_dtypes(include='number').columns;x[num]=x[num].mul(w,axis=0);m,_,_=summarize(x)
        for n in dist:dist[n].append((m.loc[n,metrics]-m.loc['parent',metrics]).to_numpy())
    out={}
    for n,vals in dist.items():
        leave=[]
        for date in dates:
            m,_,_=summarize(a[a.date!=date]);leave.append((m.loc[n,metrics]-m.loc['parent',metrics]).to_numpy())
        out[n]=dict(ci95={m:np.percentile(np.array(vals)[:,j],[2.5,97.5]).tolist()for j,m in enumerate(metrics)},leave_one_day_out_min=dict(zip(metrics,np.min(leave,axis=0).tolist())))
    dump(OUT/'feb_uncertainty.json',out)
def run():
    OUT.mkdir(exist_ok=True);t=time.time()
    if '--resume' in sys.argv:
        fs=json.loads((OUT/'features.json').read_text());models={n:lgb.Booster(model_file=str(OUT/f'{n}.txt'))for n in RECIPE['variants']}
    else:
        d=R.load_month('2031_01',True);v,q=raw_parent(d,'2031_01');fs=augment(d,v,q);weights=weight(d,'2031_01');y=d.qt/np.maximum(d.vt,1)-d.parent_k
        sub=(d.panel_i<8).to_numpy();models={};params=RECIPE['head'].copy();nr=params.pop('num_boost_round');params['verbose']=-1
        for name in RECIPE['variants']:
            ww=weights if name=='l1_transition' else np.ones(len(d));mask=sub&(ww>0)
            m=lgb.train(dict(params,objective='regression_l1'if name=='l1_transition'else'regression_l2'),lgb.Dataset(d.loc[mask,fs],y[mask],weight=ww[mask]),nr)
            m.save_model(str(OUT/f'{name}.txt'));models[name]=m
            print('head fit',name,'rows',int(mask.sum()),'seconds',round(time.time()-t),flush=True)
        dump(OUT/'features.json',fs);del d,v,q,weights,y;gc.collect()
    d=pd.concat([pd.read_parquet(MATCH.OUT/f'{p}_features.parquet')for p in E.PANELS10],ignore_index=True);v,q=raw_parent(d,'2031_02');assert augment(d,v,q)==fs
    pred={n:np.maximum(d.parent_k.to_numpy()+m.predict(d[fs],num_threads=3),0)for n,m in models.items()}
    np.savez(OUT/'feb_density_predictions.npz',**pred);pv,pq=parent_kstar(d,v,q,'2031_02');c=candidates(d,pv,pq,pred)
    d[['panel_i','day','li','ci','parent_v','parent_q','parent_k']].to_parquet(OUT/'feb_prediction_keys.parquet',index=False)
    a,results=evaluate(d,c);uncertainty(a)
    dump(OUT/'runtime.json',dict(seconds=time.time()-t,fit_month='January only',evaluation_month='February',primary='l1_transition_0.25',rows=len(d)))
if __name__=='__main__':run()
