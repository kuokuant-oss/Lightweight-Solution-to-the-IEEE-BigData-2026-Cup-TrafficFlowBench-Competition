"""Genuine80 ongoing serving inputs only; no model inference or CSV writes."""
# Runs in pipeline/ongoing.sh ("r20_prepare"), first step of the R20 operator.
# Builds the serving inputs of the 80 official queue_ongoing windows (5 per
# panel and split). Per window it reads the released masked day (values at
# the forecast time T), the window's own released history (slots T-12..T-1),
# ramp counts at T and T-1, and recency from up to 5 earlier released days of
# the same split. No model runs here and no CSV is written.
# Outputs in this folder, per split: <split>_base98.npy (the 98 base features
# FS per window, link and horizon step), <split>_encoder.npy (encoder input,
# 4 x 9 x 13 per window and link, from PublishedPanel.encode),
# <split>_base196.npy (base features after TRANSFORM_base196.json),
# <split>_ROW_INDEX.parquet, <split>_CENTER_INDEX.parquet and
# <split>_SUBMISSION_MAP.parquet (row -> submission_id); plus WINDOWS.parquet
# and INPUT_MANIFEST.json (source and code hashes, look-ahead audits).
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent;Q=HERE.parent;sys.path.insert(0,str(Q))
from common import *
from public_operator import D,PublishedPanel,FIELDS
from prepare_march import legal_frame,source_poison
from transform import transform_block
import gc,time,resource,itertools
# EXPECTED_PARENT_SHA / PARENT: the R16b submission CSV; submission_binding no
# longer reads them (package edit).
EXPECTED_PARENT_SHA='05cebca599b04cba99802d2f97ab2a2145e6992cc3af12fa77c415a5770c5d64'
PARENT=ROOT/'build/deliverables/2026-09-30/submission_R16b_R11c_CAPDENS_C85.csv'
# Package edit (2026-11-10): PublishedPanel.day (e_data.py, pinned by INPUT_MANIFEST.json, so not edited)
# also asserted that a masked day file holds all 288 slots of every link. A day file whose rows after
# the forecast origin were removed then stopped the run. This copy of day() is the same except that
# assert: rows are placed in the links x 288 grid by timestamp and absent cells stay NaN, as before.
# Its readers (encode, legal_frame, source_poison) use the slots up to T only.
def _published_day(self,path,truth=False):
    cols=['timestamp','link_id']+FIELDS+['is_score_eligible']
    d=pd.read_parquet(path,columns=cols);rr=d.link_id.astype(str).map(self.li).to_numpy();ts=pd.to_datetime(d.timestamp,utc=True);cc=(ts.dt.hour*12+ts.dt.minute//5).to_numpy()
    assert np.isfinite(rr).all() and not d[['timestamp','link_id']].duplicated().any() and len(d)<=len(self.links)*288
    out={}
    for c in FIELDS+['is_score_eligible']:
        a=np.full((len(self.links),288),np.nan);a[rr,cc]=d[c].to_numpy();out[c]=a.astype(bool)if c=='is_score_eligible'else a
    return out
PublishedPanel.day=_published_day
# Package edit (2026-11-10): the split list comes from TFB_SPLITS (default 'validation private', see README).
SPLITS = tuple(os.environ.get('TFB_SPLITS', 'validation private').split())
def indexes():
    """Reads the official queue_ongoing windows of both splits.

    Returns:
      (windows, hashes): the 80 windows with window_id, panel, split, date,
      forecast_origin, condition, origin (timestamp) and T (5-minute slot of
      the forecast time), and the SHA256 of each window_index.csv read.
    """
    records=[];source={}
    for split in SPLITS:
        for panel in PANELS:
            path=D.R.RELEASE/f'task2/{panel}/{split}/window_index.csv'
            w=pd.read_csv(path,usecols=['window_id','panel','split','date','forecast_origin','condition'])
            # Package edit (2026-11-10): row and window counts follow window_index.csv instead of the fixed 5-window layout.
            w=w[w.condition.eq('queue_ongoing')].copy();assert (w.panel==panel).all()and(w.split==split).all()
            w['origin']=pd.to_datetime(w.forecast_origin,utc=True);w['T']=w.origin.dt.hour*12+w.origin.dt.minute//5
            assert(w['T'].between(12,281)).all()and w.date.equals(w.origin.dt.strftime('%Y-%m-%d'))
            assert w.window_id.is_unique;records.append(w);source[str(path)]=sha(path)
    idx=pd.concat(records,ignore_index=True);assert idx.window_id.is_unique;return idx,source
def submission_binding(windows):
    """Maps the queue rows of the official windows to their submission ids.

    Checks that sample_submission.csv and submission_key.csv list the ids
    1..N in order. Only id, task, split, panel, window, link and timestamp are
    read from the key; no prediction column is read.

    Args:
      windows: Output of `indexes`.

    Returns:
      (mapping, binding): the 87,000 queue rows of the windows with
      submission_id and submission_row_index, and a provenance dict.
    """
    rel=D.R.RELEASE;sample=rel/'sample_submission.csv';key=rel/'submission_key.csv'  # package edit: R16b CSV SHA assert removed (whole-submission file, not regenerable here)
    cols=['submission_id','task'];dtype={'submission_id':np.int64,'task':str}
    si=pd.read_csv(sample,usecols=cols,dtype=dtype,chunksize=100000)
    rows=0;h=hashlib.sha256();queue=0
    for a in si:  # package edit: sample vs R16b ID/task order comparison removed; sample ID order still asserted below
        assert np.array_equal(a.submission_id.to_numpy(),np.arange(rows+1,rows+len(a)+1))
        h.update(pd.util.hash_pandas_object(a,index=False).to_numpy().tobytes());queue+=int(a.task.eq('queue').sum());rows+=len(a)
    wanted=set(windows.window_id);parts=[];offset=0;kcols=['submission_id','task','split','panel','window_id','link_id','timestamp']
    for d in pd.read_csv(key,usecols=kcols,dtype={c:str for c in kcols if c!='submission_id'},chunksize=100000):
        take=d.task.eq('queue')&d.window_id.isin(wanted);g=d[take].copy();g['submission_row_index']=offset+np.flatnonzero(take)
        if len(g):parts.append(g)
        assert np.array_equal(d.submission_id.to_numpy(),np.arange(offset+1,offset+len(d)+1));offset+=len(d)
    assert offset==rows;mapping=pd.concat(parts,ignore_index=True);assert mapping.submission_id.is_unique
    assert set(mapping.window_id)==wanted
    # Sample and R16b task/ID rows were identical; queue targetkeys are read ONLY
    # for output mapping and are never passed to a feature/encoder function.
    files={str(p):sha(p)for p in [sample,key]}  # package edit: PARENT (R16b CSV) dropped
    return mapping,dict(files=files,total_submission_rows=rows,all_queue_rows=queue,active_ongoing_mapped_rows=len(mapping),
        sample_R16b_all_ID_task_order_exact='not checked (package edit: R16b binding removed)',ID_task_order_hash=h.hexdigest(),parent_identity='R16bref56697520public.90144',no_prediction_columns_read=True)
def run():
    """Builds and writes the serving inputs of both splits and the input manifest.

    Raises:
      AssertionError: When an output already exists, a recorded code hash
        differs, a window history is not exactly the 12 slots before T, or a
        row count or mapping check fails.
    """
    assert not(HERE/'INPUT_MANIFEST.json').exists();start=time.time();windows,indexhash=indexes()
    core=json.loads((Q/'INPUT_MANIFEST.json').read_text());assert core['PASS']
    for path,h in core['code_hashes'].items():assert sha(path)==h,path
    strict=np.load(BLOCKS,allow_pickle=True).item();D.L.BID={p:dict(zip(v['links'],v['bid']))for p,v in strict.items()}
    spec=json.loads((Q/'TRANSFORM_base196.json').read_text());assert sha(Q/'TRANSFORM_base196.json')==core['base_transform_sha256']
    sourcefiles=dict(indexhash);records={};audits=[];profiles=[];recency_sources=[];ramp_sources=[]
    for split in SPLITS:
        # n rows per split: links x 6 horizon steps x ongoing windows, summed over the 8 panels
        # (43,500 for the released 5 windows per panel). Package edit (2026-11-10): from window_index.
        win=windows[windows.split.eq(split)].copy();n=sum(len(strict[p]['links'])*6*int(win.panel.eq(p).sum()) for p in PANELS)
        basepath=HERE/f'{split}_base98.npy';encpath=HERE/f'{split}_encoder.npy';assert not basepath.exists()and not encpath.exists()
        base=np.lib.format.open_memmap(basepath,mode='w+',dtype=np.float64,shape=(n,98));enc=np.lib.format.open_memmap(encpath,mode='w+',dtype=np.float32,shape=(n//6,4,9,13))
        rows=[];centers=[];roff=coff=0
        for panel in PANELS:
            P=PublishedPanel(panel,split);c=D.Context(panel,split);assert P.split==c.split==split
            # Only link_id/milepost are read from this first training file;
            # static geometry is not a serving target or future-label source.
            for path in [D.R.truth_files(panel,'train')[0]]+[D.R.RELEASE/f'corridors/{panel}/network/{name}.csv'for name in ['links','fd_parameters','ramp_attachment_map']]:
                sourcefiles[str(path)]=sha(path)
            hp=D.R.RELEASE/f'task2/{panel}/{split}/window_history.parquet';sourcefiles[str(hp)]=sha(hp)
            # Per window: history = the released window_history (slots T-12..T-1);
            # features from direct.Context.features (legal_frame); encoder input
            # from PublishedPanel.encode.
            raw=h=None  # package edit (2026-11-10): a panel without ongoing windows in this split skips the loop
            for i,w in enumerate(win[win.panel.eq(panel)].itertuples()):
                origin=pd.Timestamp(w.origin);T=int(w.T);raw=P.day(P.published[w.date]);sourcefiles[str(P.published[w.date])]=sha(P.published[w.date])
                h=P.own_history(w.window_id,origin);assert h is not None
                ts=pd.to_datetime(h.timestamp,utc=True);rr=h.link_id.astype(str).map(P.li).to_numpy(int);tt=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
                assert len(h)==len(P.links)*12 and set(tt)==set(range(T-12,T))and(ts<origin).all()
                assert set(h.link_id.astype(str))==set(P.links)and not h[['timestamp','link_id']].duplicated().any()
                # No futurelabels or future support are read/attached in serving.
                f=legal_frame(P,c,w.date,T,w.window_id,raw,h);x=P.encode(raw,T,h,origin)
                assert len(f)==len(P.links)*6 and np.array_equal(f.timestamp,origin+pd.to_timedelta(f.step*5,unit='m'))
                li=f.link_id.astype(str).map(P.li).to_numpy(int);center=coff+li;assert np.array_equal(f.bid,f.link_id.map(D.L.BID[panel]))
                base[roff:roff+len(f)]=f[FS].to_numpy(np.float64);enc[coff:coff+len(P.links)]=x
                rows.append(pd.DataFrame(dict(window_id=w.window_id,panel=panel,split=split,date=w.date,timestamp=f.timestamp,origin=origin,
                    link_id=f.link_id.astype(str),step=f.step.to_numpy(np.int8),center=center,bid=f.bid.to_numpy(int),physical_row=np.arange(roff,roff+len(f)))))
                centers.append(pd.DataFrame(dict(window_id=w.window_id,panel=panel,split=split,date=w.date,origin=origin,link_id=P.links,center_row=np.arange(coff,coff+len(P.links)))))
                diffs={}
                for col in FIELDS+['is_score_eligible']:
                    a=h[col].to_numpy();b=raw[col][rr,tt];diffs[col]=int((~((a==b)|(pd.isna(a)&pd.isna(b)))).sum())
                profiles.append(dict(window_id=w.window_id,panel=panel,split=split,date=w.date,history_rows=len(h),history_preT_exact=True,
                    genuine_history_vs_raw_past_mismatches=diffs,encoder_joint_observed_fraction=float(x[:,3,4,:].mean()),origin_source='unaltered rawpublishedT'))
                # The first window of each panel also gets the junk-after-T check
                # (source_poison).
                if i==0:audits.append({**source_poison(P,c,w.date,T,w.window_id,raw,h),'split':split})
                roff+=len(f);coff+=len(P.links)
            # Bind the exact ramp source selected by TB.ramp_day and every
            # same-split recency file actually eligible under fixed K5.
            for date in win[win.panel.eq(panel)].date.unique():
                paths=sorted(p for p in(D.R.RELEASE/f'corridors/{panel}/{split}/ramp_states').glob(f'**/synthetic_ramp_{date.replace("-","_")}.parquet')if not p.name.startswith('._'))
                if paths:sourcefiles[str(paths[0])]=sha(paths[0])
                ramp_sources.append(dict(panel=panel,split=split,date=date,selected_path=str(paths[0])if paths else None,used_timestamps='originT and T-1 only'))
                prior=sorted(d for d in D.REC.day_index(panel,split)if d<date)[-5:]
                for previous in prior:
                    p=Path(D.REC.day_index(panel,split)[previous]);assert previous<date and f'/{split}/mainline_states_masked/'in str(p)
                    sourcefiles[str(p)]=sha(p)
                recency_sources.append(dict(panel=panel,split=split,date=date,prior_dates=prior,strict_prior_date=True,K=5))
            del P,c,raw,h;D.REC.day_q.cache_clear();gc.collect()
        assert roff==n and coff==n//6;base.flush();base._mmap.close();enc.flush();enc._mmap.close();del base,enc
        ri=pd.concat(rows,ignore_index=True);ci=pd.concat(centers,ignore_index=True)
        assert len(ri)==n and ri.window_id.nunique()==len(win) and not ri[['window_id','link_id','step']].duplicated().any()
        assert np.array_equal(ri.physical_row,np.arange(len(ri)))and np.array_equal(ci.center_row,np.arange(len(ci)))
        assert np.array_equal(pd.MultiIndex.from_frame(ci.loc[ri.center,['window_id','link_id']]),pd.MultiIndex.from_frame(ri[['window_id','link_id']]))
        ri.to_parquet(HERE/f'{split}_ROW_INDEX.parquet',index=False);ci.to_parquet(HERE/f'{split}_CENTER_INDEX.parquet',index=False)
        # base196: the 98 base features transformed in blocks of 20,000 rows.
        x=np.load(basepath,mmap_mode='r');outpath=HERE/f'{split}_base196.npy';out=np.lib.format.open_memmap(outpath,mode='w+',dtype=np.float32,shape=(n,196))
        for begin in range(0,n,20000):v=transform_block(x[begin:begin+20000],spec);out[begin:begin+len(v)]=v
        out.flush();out._mmap.close();x._mmap.close();del x,out
        files=[basepath,encpath,outpath,HERE/f'{split}_ROW_INDEX.parquet',HERE/f'{split}_CENTER_INDEX.parquet'];records[split]=dict(rows=n,centers=coff,windows=len(win),files={str(p):sha(p)for p in files})
        print('SERVING_INPUTS_PHYSICAL_READY',split,n,coff,'seconds',round(time.time()-start,1),flush=True)
    # Output membership is assigned only AFTER both physical source grids exist.
    mapping,binding=submission_binding(windows)
    for split in records:
        ri=pd.read_parquet(HERE/f'{split}_ROW_INDEX.parquet');m=mapping[mapping.split.eq(split)].copy();m['timestamp']=pd.to_datetime(m.timestamp,utc=True)
        joined=ri.merge(m,on=['window_id','panel','split','link_id','timestamp'],how='left',validate='one_to_one',sort=False)
        assert len(joined)==records[split]['rows'] and joined.submission_id.notna().all()and joined.submission_id.is_unique
        assert np.array_equal(joined.physical_row,ri.physical_row)and joined.task.eq('queue').all()
        assert len(m)==len(joined);joined['submission_id']=joined.submission_id.astype(np.int64);joined['submission_row_index']=joined.submission_row_index.astype(np.int64)
        path=HERE/f'{split}_SUBMISSION_MAP.parquet';joined.to_parquet(path,index=False);records[split]['files'][str(path)]=sha(path)
    windows.to_parquet(HERE/'WINDOWS.parquet',index=False)
    code=[Path(__file__),HERE/'PREREG.json',Q/'prepare_march.py',Q/'e_data.py',Q/'public_operator.py',Q/'transform.py',Q/'common.py']
    result=dict(PASS=True,status='GENUINE_SERVING_INPUT_ONLY; NO_INFERENCE_NO_CSV_NO_OUTCOMES',records=records,total_windows=len(windows),total_physical_rows=sum(r['rows'] for r in records.values()),
        source_hashes=sourcefiles,submission_binding=binding,code_hashes={str(p):sha(p)for p in code},input_manifest_sha256=sha(Q/'INPUT_MANIFEST.json'),
        frozen_joint_inference_manifest_sha256=sha(Q.parent/'INFERENCE_MANIFEST.json'),base_transform_sha256=sha(Q/'TRANSFORM_base196.json'),strict_blocks_sha256=sha(BLOCKS),
        window_index_sha256=sha(HERE/'WINDOWS.parquet'),causal_audits=audits,profiles=profiles,recency_sources=recency_sources,ramp_sources=ramp_sources,
        geometry_source_scope='First training parquet reads ONLY link_id/milepost; released network links/FD/attachments; no truth values used',full_physical_grid_precedes_output_mapping=True,
        no_futuretruth_or_scoringeligibility_used=True,model_forward_calls=0,candidate_probabilities_written=False,CSV_written=False,
        parent_if_qualified='CurrentthroughFebR11cselrec3+unchangedA25,newarmgeom3,same strictJan31map/Marchoperator; fixedmixture/threshold/guard',
        seconds=time.time()-start,peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9)
    dump(HERE/'INPUT_MANIFEST.json',result);print('SERVING_INPUTS_READY',result['seconds'],result['peak_RSS_GB'],flush=True)
if __name__=='__main__':run()
