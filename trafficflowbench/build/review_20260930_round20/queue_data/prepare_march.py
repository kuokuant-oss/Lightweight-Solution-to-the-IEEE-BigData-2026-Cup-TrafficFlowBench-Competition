"""March input-only phase, validation sources and strict map; no model inference."""
from public_operator import *
from transform import transform_block
import gc,time,resource
def legal_frame(P,c,date,T,wid,raw,history):
    origin=pd.Timestamp(date,tz='UTC')+pd.Timedelta(minutes=5*T);day={k:v.copy()for k,v in raw.items()}
    if history is not None:
        ts=pd.to_datetime(history.timestamp,utc=True);assert(ts<origin).all()and(ts>=origin-pd.Timedelta(minutes=60)).all()
        rr=history.link_id.astype(str).map(c.li).to_numpy(int);cc=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
        for col in day:day[col][rr,cc]=history[col].to_numpy()
    day=source_day(day,T);f=c.features(date,T,day,wid,5,None,source_day(raw,T));assert len(f)==len(P.links)*6 and not f[KEY].duplicated().any()
    assert not set(FS)-set(f);return f
def source_poison(P,c,date,T,wid,raw,h):
    a=legal_frame(P,c,date,T,wid,raw,h);origin=pd.Timestamp(date,tz='UTC')+pd.Timedelta(minutes=5*T);x=P.encode(raw,T,h,origin if h is not None else None)
    changed={k:v.copy()for k,v in raw.items()}
    for col in changed:changed[col][:,T+1:]=True if col=='is_score_eligible'else 987654.
    save={t:v.copy()for t,v in c.ramp(date).items()};c.ramps[date]={t:({k:987654. for k in v}if t>T else v.copy())for t,v in save.items()}
    b=legal_frame(P,c,date,T,wid,changed,h);y=P.encode(changed,T,h,origin if h is not None else None);c.ramps[date]=save
    assert np.array_equal(a[D.FS].to_numpy(),b[D.FS].to_numpy(),equal_nan=True)and np.array_equal(x,y)
    return dict(panel=P.panel,window_id=wid,base98_and_A_feature_union_exact=True,encoder_source_exact=True,mainline_and_ramp_postT_poison=True,
        scope='sampled input-only check; no model forward/probabilities/scores',split='validation')
def run():
    assert not(ME/'MARCH_INPUT_MANIFEST.json').exists();assert json.loads((ME/'INPUT_MANIFEST.json').read_text())['PASS'];start=time.time();records={}
    blocks=np.load(BLOCKS,allow_pickle=True).item();D.L.BID={p:dict(zip(v['links'],v['bid']))for p,v in blocks.items()};spec=json.loads((ME/'TRANSFORM_base196.json').read_text())
    for rv,expected in [('v2',104004),('v1',104490)]:
        source=ROOT/'build/review_20260927_direct'/f'{rv}_query.parquet';q=pd.read_parquet(source,columns=['window_id','panel','date','link_id','step','timestamp','origin']).reset_index(drop=True)
        assert len(q)==expected and not q[KEY].duplicated().any();assert q.date.astype(str).between('2031-03-01','2031-03-12').all()
        first=q.index[~q[['window_id','link_id']].duplicated()].to_numpy();ci=q.iloc[first][['window_id','panel','date','link_id','origin']].copy().reset_index(drop=True)
        ci['center_row']=np.arange(len(ci));mapping=pd.MultiIndex.from_frame(ci[['window_id','link_id']]).get_indexer(pd.MultiIndex.from_frame(q[['window_id','link_id']]));assert(mapping>=0).all();q['center']=mapping
        basepath=CACHE/f'march_{rv}_base98.npy';xpath=CACHE/f'march_{rv}_encoder.npy';assert not basepath.exists()and not xpath.exists()
        base=np.lib.format.open_memmap(basepath,mode='w+',dtype=np.float64,shape=(len(q),98));enc=np.lib.format.open_memmap(xpath,mode='w+',dtype=np.float32,shape=(len(ci),4,9,13))
        bids=np.empty(len(q),np.int64);sourcefiles={};audits=[];profiles=[]
        for p,ii in q.groupby('panel',sort=False).indices.items():
            P=PublishedPanel(p,'validation');c=D.Context(p,'validation');assert P.split==c.split=='validation'
            for date,gdate in q.iloc[ii].groupby('date',sort=False):
                raw=P.day(P.published[date]);sourcefiles[str(P.published[date])]=sha(P.published[date])
                for wid,g in gdate.groupby('window_id',sort=False):
                    origin=pd.Timestamp(g.origin.iloc[0]);T=origin.hour*12+origin.minute//5;assert origin==pd.Timestamp(date,tz='UTC')+pd.Timedelta(minutes=5*T)
                    h=P.own_history(wid,origin);f=legal_frame(P,c,date,T,wid,raw,h).set_index(KEY).reindex(pd.MultiIndex.from_frame(g[KEY])).reset_index()
                    assert f[KEY].equals(g[KEY].reset_index(drop=True))and len(g)==len(P.links)*6 and g.link_id.nunique()==len(P.links)
                    assert np.array_equal(pd.to_datetime(f.timestamp,utc=True),pd.to_datetime(g.timestamp,utc=True));base[g.index]=f[FS].to_numpy(np.float64);bids[g.index]=f.bid.to_numpy(int)
                    x=P.encode(raw,T,h,origin if h is not None else None);one=g.loc[~g.link_id.duplicated()];enc[q.loc[one.index,'center'].to_numpy()]=x[one.link_id.astype(str).map(P.li).to_numpy(int)]
                    profiles.append(dict(window_id=wid,panel=p,date=date,own_genuine_history=h is not None,published_joint_observed_fraction=float(x[:,3,4,:].mean())))
                    if p not in{a['panel']for a in audits}:audits.append(source_poison(P,c,date,T,wid,raw,h))
            del P,c,raw;D.REC.day_q.cache_clear();gc.collect()
        q['bid']=bids;q.to_parquet(CACHE/f'march_{rv}_ROW_INDEX.parquet',index=False);ci.to_parquet(CACHE/f'march_{rv}_CENTER_INDEX.parquet',index=False)
        base.flush();base._mmap.close();enc.flush();enc._mmap.close();del base,enc
        x=np.load(basepath,mmap_mode='r');path196=CACHE/f'march_{rv}_base196.npy';out=np.lib.format.open_memmap(path196,mode='w+',dtype=np.float32,shape=(len(q),196))
        for begin in range(0,len(q),20000):b=transform_block(x[begin:begin+20000],spec);out[begin:begin+len(b)]=b
        out.flush();out._mmap.close();x._mmap.close();del out,x
        files=[basepath,xpath,path196,CACHE/f'march_{rv}_ROW_INDEX.parquet',CACHE/f'march_{rv}_CENTER_INDEX.parquet']
        records[rv]=dict(rows=len(q),centers=len(ci),windows=q.window_id.nunique(),input_operator='validation published raw≤T plus genuine ownhistory<T when ID exists;K5 same-split strictly prior-date recency; raw originT; no traintruth overlay',
            query_source_sha256=sha(source),files={str(p):sha(p)for p in files},source_hashes=sourcefiles,profiles=profiles,causal_audits=audits,
            full_physical_keys_unchanged=True,original_sparse_scorer_targets_not_modified='labels remain the original bound ruler/scorer, not regenerated by this input-only adapter',
            split='validation',min_date=q.date.min(),max_date=q.date.max(),strict_block_sha256=sha(BLOCKS))
        print('R20_MARCH_INPUT_READY',rv,len(q),len(ci),'seconds',round(time.time()-start,1),flush=True)
    dump(ME/'MARCH_INPUT_MANIFEST.json',dict(PASS=True,status='INPUT_ONLY_NO_CANDIDATE_INFERENCE_NO_OUTCOMES',records=records,input_manifest_sha256=sha(ME/'INPUT_MANIFEST.json'),
        code_sha256=sha(Path(__file__)),base_transform_sha256=sha(ME/'TRANSFORM_base196.json'),strict_block_sha256=sha(BLOCKS),
        March_parent='Literal prospective current throughFeb R11cN6+A25 under same new strictJan31map and legal validation operator; never interpreted as FebOOF',
        seconds=time.time()-start,peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9))
if __name__=='__main__':run()
