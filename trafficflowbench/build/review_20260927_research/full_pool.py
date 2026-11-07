"""Full eligible-cell feature reconstruction: confirm pooling without label-availability-dependent support."""
from common import *
import s3_build_rows as C
from structure import topology
DEST=HERE/'full_pool';DEST.mkdir(exist_ok=True)

def pool(frame,group):
    return frame.assign(_group=np.asarray(group)).groupby(['date','origin','_group','step_i'],sort=False).p.transform('mean').to_numpy()

def build(split='validation'):
    src=R2 if split=='validation' else R3
    old,st=load(split);del st
    parts=[];checks={}
    for panel in S.PANELS:
        dest=DEST/f'{split}_{panel}.parquet'
        if dest.exists():print('reuse',dest.name,flush=True);continue
        ctx=C.panel_ctx(panel,split);gs=ctx['grids'];dates=[g.date for g in gs];links=ctx['links'];nL=len(links);nT=ctx['nT'];lanes=ctx['lanes'];bid=ctx['bid']
        ms=lgb.Booster(model_file=str(C.MD/'gap_speed.txt'));mf=lgb.Booster(model_file=str(C.MD/'gap_flow.txt'));feats=json.loads((C.MD/'gap_features.json').read_text())
        op=old[old.panel==panel];windows=op[['date','origin','cond']].drop_duplicates()
        keypath=R2/'work/champ_val_state.parquet' if split=='validation' else R3/'work/parent_private_state.parquet'
        ch=pd.read_parquet(keypath);ch=ch[ch.panel==panel];chd={d:x for d,x in ch.assign(date=ch.timestamp.str[:10]).groupby('date')}
        ixlink={l:i for i,l in enumerate(links)};ixtime={pd.Timestamp(t).strftime('%H:%M:%S'):i for i,t in enumerate(gs[0].times)}
        Vs=np.stack([g.value['speed_kmh']for g in gs]);Qs=np.stack([g.value['flow_vph']for g in gs]);Es=np.stack([g.eligible for g in gs]);dayno=np.array([int(d[-2:])for d in dates]);SS={};cls={}
        for odd in [True,False]:
            ix=np.where((dayno%2==1)==odd)[0];nan=np.full((1,)+Vs.shape[1:],np.nan)
            ss=C.SplitStats(np.concatenate([Vs[ix],nan]),np.concatenate([Qs[ix],nan]),np.concatenate([Es[ix],np.zeros(nan.shape,bool)]),ctx['vcut'],lanes,bid)
            SS[odd]=ss.loo(len(ix));cls[odd]=topology(panel,split,odd)
        del Vs,Qs,Es,ch;gc.collect();pp=[]
        for di,g in enumerate(gs):
            ws=windows[windows.date==g.date]
            if not len(ws):continue
            cz=chd.get(g.date);CV=np.full((nL,nT),np.nan);CQ=CV.copy()
            if cz is not None:
                rr=cz.link_id.astype(str).map(ixlink).to_numpy();cc=cz.timestamp.str[11:19].map(ixtime).to_numpy()
                CV[rr,cc]=cz.speed_kmh.to_numpy();CQ[rr,cc]=cz.flow_vph.to_numpy()
            wd=pd.Timestamp(g.date).weekday();hv=ctx['prof'][f'speed_kmh_{wd}'];hq=ctx['prof'][f'flow_vph_{wd}']
            odd=int(g.date[-2:])%2==0 # fit other half
            for w in ws.itertuples():
                T=int(w.origin);a=T+1;seg=slice(a,a+18);hs=slice(T-12,T)
                V=g.value['speed_kmh'].copy();Q=g.value['flow_vph'].copy();O=g.value['occupancy'].copy();E=g.eligible
                for X,XC in [(V,CV),(Q,CQ)]:X[:,hs]=np.where(np.isnan(X[:,hs])&np.isfinite(XC[:,hs]),XC[:,hs],X[:,hs])
                for X in [V,Q,O]:X[:,seg]=np.nan
                # Window selector fields are constant across original query rows; reuse legal frozen values.
                ref=op[(op.date==g.date)&(op.origin==T)].iloc[0]
                # gap_matrix computes selector again from gap_min; recover it from the official pseudo selector table.
                # Direct f overrides below preserve the audited selection features, avoiding hidden label selection.
                f=C.gap_matrix(V,Q,O,E,a,nT,ctx['par'],ctx['vcut'],hv,hq,bid,ctx['tmed'],ctx['tiqr'],dict(cond=w.cond,gap_min=np.nan))
                for key in ['cond_on','prev_rej','n_rej','blk1']:
                    f[key]=np.full((nL,18),float(ref['f_'+key]),np.float32)
                pv,pq=C.g2v2_predict(ms,mf,feats,f,panel,lanes)
                li,si=np.where(E[:,seg]);cc=a+si
                row=pd.DataFrame({'panel':panel,'date':g.date,'day':int(g.date[-2:]),'origin':T,'li':li,'step_i':si,'ch_v':pv[li,si],'ch_ql':pq[li,si]/lanes[li],'group':cls[odd][li],'static_group':bid[li]})
                for k in C.KEEP_F:row['f_'+k]=np.asarray(f[k])[li,si].astype(np.float32)
                row['ch_v']=row.ch_v.astype(np.float32);row['ch_ql']=row.ch_ql.astype(np.float32)
                C.split_feats(row,SS[odd],li,cc,a);S.derive(row);pp.append(row)
        frame=pd.concat(pp,ignore_index=True);del pp,ctx,gs,SS;gc.collect()
        # Exact keys and actual feature parity with the original sparse table, after converting both to stored float32.
        # All base and derived feature dtypes match the original saved float32 table.
        chk=op[KEY+['ch_v','ch_ql']].merge(frame[KEY+['ch_v','ch_ql']],on=KEY,validate='one_to_one',suffixes=('_old','_full'))
        delta=max(float(np.max(np.abs(chk[k+'_old']-chk[k+'_full'])))for k in ['ch_v','ch_ql']);assert delta<2e-4,(panel,delta)
        for odd in [True,False]:
            te=(frame.day%2==1).to_numpy()!=odd;tag=f'{split}_strict_fitodd{int(odd)}'
            for kind,col in [('pqueue','p'),('free','fv'),('queued','qv')]:
                m=lgb.Booster(model_file=str(HM/f'{tag}_{kind}.txt'));v=m.predict(frame.loc[te,S.FEATS],num_threads=3)
                frame.loc[te,col]=v if kind=='pqueue'else frame.loc[te,'ch_v'].to_numpy()+v
        pchk=op[KEY+['old_pqueue']].merge(frame[KEY+['p']],on=KEY,validate='one_to_one');dp=float(np.max(np.abs(pchk.old_pqueue-pchk.p)));assert dp<1e-8,(panel,dp)
        keep=KEY+['group','static_group','p','fv','qv'];frame[keep].to_parquet(dest,index=False)
        checks[panel]={'full_eligible_rows':len(frame),'sparse_scored_rows':len(op),'base_max_diff':delta,'pqueue_max_diff':dp}
        (DEST/f'{split}_{panel}_check.json').write_text(json.dumps(checks[panel],indent=2));print('full',panel,checks[panel],flush=True)
        del frame;gc.collect()
    pred=pd.concat([pd.read_parquet(DEST/f'{split}_{p}.parquet')for p in S.PANELS],ignore_index=True)
    # Panel is included by processing each panel separately. Static unknowns use separate links.
    for name,col in [('adaptive','group'),('static','static_group')]:
        pred[name+'_p']=np.nan
        for p,ii in pred.groupby('panel',observed=True).indices.items():
            sub=pred.iloc[ii];groups=sub[col].to_numpy();groups=np.where(groups<0,-sub.li.to_numpy()-1000,groups)
            pred.loc[ii,name+'_p']=pool(sub,groups)
    z=old.merge(pred,on=KEY,validate='one_to_one');report={};out=z[KEY+['y_v','day','cond','hm25','t1']].copy()
    mix=np.clip((1-z.p)*z.fv+z.p*z.qv,S.SPEED_FLOOR,S.SPEED_CAP)
    assert np.max(np.abs(.75*z.t1+.25*mix-z.hm25))<1e-5
    for name in ['adaptive','static']:
        p=z[name+'_p'];new=np.clip((1-p)*z.fv+p*z.qv,S.SPEED_FLOOR,S.SPEED_CAP)
        out[name]=.75*z.t1+.25*new;report[name]=evaluate(out,out[name]);print(name,json.dumps(report[name]),flush=True)
    out.to_parquet(DEST/f'{split}_predictions.parquet',index=False);(DEST/f'{split}_results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='validation');args=ap.parse_args();build(args.split)
