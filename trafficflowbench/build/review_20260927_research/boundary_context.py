"""Observed-only, full boundary sequence features. No targets or filled values enter context."""
from common import *
sys.path.insert(0,str(R2));import build_ph as B
from src.tfb import grid,physics,predict1 as P1,release as R
DEST=HERE/'boundary';DEST.mkdir(exist_ok=True)
WK=['panel','date','origin','li']

def extract(V,Q,E,t,bid,vf,cap):
    # Enforce whole-corridor hole even if the caller passes unmasked arrays.
    V=V.copy();Q=Q.copy();V[:,t+1:t+19]=np.nan;Q[:,t+1:t+19]=np.nan
    columns=np.r_[np.arange(t-12,t),np.arange(t+19,t+31)]
    assert np.all((columns<=t)|(columns>=t+19))
    vv=V[:,columns];qq=Q[:,columns];ee=E[:,columns]
    valid=np.isfinite(vv);validq=np.isfinite(qq);queue=valid&(vv<=.6*vf[:,None])
    bq=np.full(vv.shape,np.nan);bn=np.zeros(vv.shape)
    # Unknown blocks stay singleton; never gather all bid=-1 as one unit.
    ids=np.where(bid>=0,bid,-np.arange(len(bid))-1000)
    for b in np.unique(ids):
        m=ids==b;n=valid[m].sum(0);fr=queue[m].sum(0)/np.maximum(n,1);fr[n==0]=np.nan
        bq[m]=fr;bn[m]=n/max(m.sum(),1)
    groups={'v':vv/np.maximum(vf[:,None],1),'q':qq/np.maximum(cap[:,None],1),'vm':valid.astype(float),'qm':validq.astype(float),'bq':bq,'bn':bn}
    feat={}
    for nm,x in groups.items():
        for j in range(len(columns)):feat[f'ctx_{nm}_{j:02d}']=x[:,j].astype(np.float32)
    return feat

def build(split='validation',official=False):
    src=R2 if split in ['validation','train_r2'] else R3
    of=pd.read_parquet(R3/'work'/f'off_rows_{split}.parquet') if official else None
    for panel in S.PANELS:
        path=DEST/f'context_{split}_{"official" if official else "pseudo"}_{panel}.parquet'
        if path.exists():continue
        d=of[of.panel==panel] if official else pd.read_parquet(src/'work'/f'ph_{split}_{panel}.parquet',columns=WK)
        w=d[WK].drop_duplicates();gs=B.load_panel(panel,'train','2031_02') if split=='train_r2' else [grid.build_masked(panel,p)for p in P1.split_masked_files(panel,split)]
        par=physics.link_params(panel,[str(l) for l in gs[0].links]);bid=B.block_ids(panel,[str(l)for l in gs[0].links]);parts=[]
        for g in gs:
            q=w[w.date==g.date]
            for t,z in q.groupby('origin',sort=False):
                t=int(t);V=g.value['speed_kmh'].copy();Q=g.value['flow_vph'].copy()
                if split=='train_r2':
                    hs=slice(t-12,t)
                    for x,field in [(V,'speed_kmh'),(Q,'flow_vph')]:x[:,hs]=np.where(g.eligible[:,hs]&np.isnan(x[:,hs]),g.truth[field][:,hs],x[:,hs])
                fs=extract(V,Q,g.eligible,t,bid,par['free_speed_kmh'],par['capacity_vph'])
                # Poisoned hole must produce bitwise-equivalent context, including NaN pattern.
                V[:,t+1:t+19]=123456;Q[:,t+1:t+19]=-98765
                poison=extract(V,Q,g.eligible,t,bid,par['free_speed_kmh'],par['capacity_vph'])
                assert all(np.array_equal(fs[k],poison[k],equal_nan=True)for k in fs)
                ix=z.li.to_numpy(int);o=z.reset_index(drop=True).copy()
                out=pd.DataFrame({k:v[ix] for k,v in fs.items()});parts.append(pd.concat([o,out],axis=1))
        out=pd.concat(parts,ignore_index=True);assert len(out)==len(w) and not out.duplicated(WK).any()
        out.to_parquet(path,index=False);print('context',split,official,panel,len(out),'poison checked',flush=True)
        del gs,parts,out;gc.collect()

def augment(d,split,official=False):
    c=pd.concat([pd.read_parquet(DEST/f'context_{split}_{"official" if official else "pseudo"}_{p}.parquet') for p in S.PANELS],ignore_index=True)
    return d.merge(c,on=WK,validate='many_to_one',sort=False)

def run(split='validation',mixture=False):
    build(split);d,st=load(split);d=augment(d,split)
    features=S.FEATS+[c for c in d if c.startswith('ctx_')]
    mode='mixture' if mixture else 'l2';out=d[KEY+['y_v','day','cond','hm25','t1']].copy();out['new']=np.nan
    for odd in [True,False]:
        x=strict(d,st,odd);tr=(x.day%2==1).to_numpy()==odd;te=~tr
        if mixture:
            queue=(x.y_v<=.6*x.f_free_speed_kmh).to_numpy();pred=[]
            for kind,mask,target,obj in [('pqueue',tr,queue,'binary'),('free',tr&~queue,x.r_v,'l2'),('queued',tr&queue,x.r_v,'l2')]:
                m=lgb.train(dict(S.PAR,objective=obj),lgb.Dataset(x.loc[mask,features],np.asarray(target)[mask]),S.NR)
                m.save_model(str(DEST/f'{split}_{mode}_{int(odd)}_{kind}.txt'));pred.append(m.predict(x.loc[te,features],num_threads=3));del m;gc.collect()
            p,free,queued=pred;out.loc[te,'new']=np.clip(x.loc[te,'ch_v']+(1-p)*free+p*queued,S.SPEED_FLOOR,S.SPEED_CAP)
        else:
            m=lgb.train(S.PAR,lgb.Dataset(x.loc[tr,features],x.loc[tr,'r_v']),S.NR)
            m.save_model(str(DEST/f'{split}_{mode}_{int(odd)}.txt'))
            out.loc[te,'new']=np.clip(x.loc[te,'ch_v']+m.predict(x.loc[te,features],num_threads=3),S.SPEED_FLOOR,S.SPEED_CAP)
            del m;gc.collect()
        del x;gc.collect();print('fit',split,mode,'odd',odd,flush=True)
    out['blend25']=.75*out.hm25+.25*out.new
    report={nm:evaluate(out,out[nm]) for nm in ['new','blend25']}
    report['matched_l2_control']=evaluate(out,out.new,base='t1') if not mixture else {}
    out.to_parquet(DEST/f'{split}_{mode}_predictions.parquet',index=False)
    (DEST/f'{split}_{mode}_results.json').write_text(json.dumps(report,indent=2))
    (DEST/'features.json').write_text(json.dumps(features,indent=2))
    for nm,r in report.items():print(nm,json.dumps(r),flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='validation',choices=['validation','private','train_r2']);ap.add_argument('--mixture',action='store_true');ap.add_argument('--build-only',action='store_true');args=ap.parse_args()
    if args.build_only:build(args.split)
    else:run(args.split,args.mixture)
