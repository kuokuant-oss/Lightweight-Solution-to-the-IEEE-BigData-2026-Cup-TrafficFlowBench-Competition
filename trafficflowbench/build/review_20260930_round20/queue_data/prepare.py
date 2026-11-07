"""Build R20 complete train/held data, two causal sources, no new model fitting."""
from public_operator import *
import time,gc,resource,pyarrow as pa,pyarrow.parquet as pq
pa.set_cpu_count(2);pa.set_io_thread_count(2)
COLS=['window_id','date','panel','link_id','step','_link_idx','label','_eligible','is_rep']
def metadata():
    parts={};order=[];seen=set();sizes={'fit':0,'held':0};centers={'fit':0,'held':0};counts={};cohort={}
    for panel in PANELS:
        src=REP2/f'rep2_s0_{panel}.parquet';d=pd.read_parquet(src,columns=COLS);d=d[d.is_rep.eq(1)].reset_index(drop=True)
        assert not d[KEY].duplicated().any()
        for wid in pd.unique(d.window_id):
            if wid not in seen:order.append(wid);seen.add(wid)
        parts[panel]=d
        P=PublishedPanel(panel)
        for name,mask in [('fit',d.date.astype(str)<=FIT_END),('held',d.date.astype(str).between(HELD_START,HELD_END))]:
            q=d[mask];counts[panel,name]=q.window_id.nunique();sizes[name]+=int(q._eligible.sum())if name=='fit'else len(q)
            centers[name]+=counts[panel,name]*len(P.links)
            cohort.setdefault(name,[]).append(q[['window_id','panel','date']].drop_duplicates('window_id'))
        del P
    assert len(order)==3013 and sum(counts[p,'held']for p in PANELS)==322
    for name in cohort:
        z=pd.concat(cohort[name],ignore_index=True);z.to_parquet(CACHE/f'{name}_WINDOWS.parquet',index=False)
        assert z.window_id.is_unique
        if name=='held':assert z.date.nunique()==28 and set(z.panel)==set(PANELS)
    rng=np.random.default_rng(123);drop=set(np.asarray(order,dtype=object)[rng.random(len(order))<.25]);assert len(drop)==765
    dump(ME/'ROW_PLAN.json',dict(fit_end=FIT_END,physical_downstream_rows=sizes,center_rows=centers,
        windows={name:int(sum(counts[p,name]for p in PANELS))for name in ['fit','held']},
        windows_by_panel={p:{name:int(counts[p,name])for name in ['fit','held']}for p in PANELS},
        original_window_order=order,original_order_sha256=hashlib.sha256('\n'.join(order).encode()).hexdigest(),
        selector_dropout_windows=sorted(drop),fit_selector_dropout_windows=int(sum(w in drop and w.split('|')[1]<=FIT_END for w in order)),
        held_scoring='full physical key cache; score mask future eligibility separate',data_prereg_sha256=sha(ME/'PREREG_DATA.json')))
    return parts,order,drop,sizes,centers,counts
def run():
    assert not(ME/'PREPARED.json').exists();assert BLOCKS.exists();start=time.time()
    parts,order,drop,sizes,centers,counts=metadata();wix={w:i for i,w in enumerate(order)}
    for name in sizes:
        for source in ['matched','raw']:
            a=np.lib.format.open_memmap(CACHE/f'{name}_{source}_base98.npy',mode='w+',dtype=np.float32 if name=='fit'else np.float64,shape=(sizes[name],98));a.flush();a._mmap.close()
    metadata_parts={name:[]for name in sizes};center_parts={name:[]for name in sizes};offset={name:0 for name in sizes};center_offset={name:0 for name in sizes}
    records=[];audits=[];source_hashes={};all_fit_dates=set();panel_idx={p:i for i,p in enumerate(PANELS)}
    for panel in PANELS:
        d=parts.pop(panel);P=PublishedPanel(panel);c=strict_context(panel);L=len(P.links)
        for name in ['fit','held']:
            arrays={source:np.load(CACHE/f'{name}_{source}_base98.npy',mmap_mode='r+')for source in ['matched','raw']}
            take=d.date.astype(str)<=FIT_END if name=='fit'else d.date.astype(str).between(HELD_START,HELD_END)
            q=d[take].copy();wids=pd.unique(q.window_id);N=len(wids)*L;assert len(wids)==counts[panel,name]
            enc={source:np.lib.format.open_memmap(CACHE/f'{name}_{panel}_{source}_X.npy',mode='w+',dtype=np.float32,shape=(N,4,9,13))for source in ['matched','raw']}
            target={k:np.empty((N,6)if k in ['label','mask']else N,dtype=dt)for k,dt in [('label',np.float32),('mask',bool),('switch',np.int64),('switch_mask',bool),('cell_weight',np.float64),('switch_weight',np.float64)]}
            target['switch_weight'].fill(0);current=None;labels_bad=0;eligible_bad=0;rawfinite=truthfinite=histcells=0;finite_value_diff=0;rows=0;winmeta=[]
            for i,(wid,g)in enumerate(q.groupby('window_id',sort=False)):
                _,date,stamp=wid.split('|');T=int(stamp);assert date==str(g.date.iloc[0]);sl=slice(i*L,(i+1)*L)
                if date!=current:
                    raw=P.day(P.published[date]);truth=P.day(P.truth[date],True);current=date
                    for path in [P.published[date],P.truth[date]]:
                        if str(path)not in source_hashes:source_hashes[str(path)]=sha(path)
                past=prefix(truth,T)
                # Inputs are built with ONLY released raw data plus the pre-T truth slice.
                frames={source:frame(P,c,date,T,wid,raw,past if source=='matched'else None)for source in ['matched','raw']}
                for source in enc:enc[source][sl]=encode(P,raw,T,past if source=='matched'else None)
                # Targets and future score support are attached strictly after the input path.
                yy,mm,ss,kk=P.targets(truth,T)
                target['label'][sl]=yy;target['mask'][sl]=mm;target['switch'][sl]=ss;target['switch_mask'][sl]=kk
                assert mm.sum()>0;target['cell_weight'][sl]=1/(8*len(wids)*mm.sum())
                if kk.any():target['switch_weight'][sl]=1/(8*len(wids)*kk.sum())
                li=g._link_idx.to_numpy(int);step=g.step.to_numpy(int)-1
                assert np.array_equal(P.links[li],g.link_id.astype(str).to_numpy())
                # Original rep2 labels outside eligibility are placeholders (zero
                # in the checked panel); physical rows never turn them into targets.
                eg=g._eligible.to_numpy(bool)
                labels_bad+=int((g.label.to_numpy()[eg].astype(bool)!=yy[li[eg],step[eg]].astype(bool)).sum())
                eligible_bad+=int((g._eligible.to_numpy(bool)!=mm[li,step]).sum())
                keep=g._eligible.to_numpy(bool)if name=='fit'else np.ones(len(g),bool);r=g.iloc[np.flatnonzero(keep)].copy();nr=len(r)
                row_center=center_offset[name]+i*L+r._link_idx.to_numpy(int);row_step=r.step.to_numpy(int)-1
                for source,f in frames.items():
                    f=f.set_index(KEY).reindex(pd.MultiIndex.from_frame(r[KEY])).reset_index();assert f[KEY].equals(r[KEY].reset_index(drop=True))
                    v=f[FS].to_numpy(np.float32 if name=='fit'else np.float64)
                    if name=='fit'and wid in drop:v[:,FS.index('prev_rej')]=np.nan;v[:,FS.index('n_rej')]=np.nan
                    arrays[source][offset[name]+rows:offset[name]+rows+nr]=v
                # Same target values/weights/key order for both source tables and all learners.
                metadata_parts[name].append(pd.DataFrame(dict(window_id=wid,panel=panel,date=date,link_id=r.link_id.astype(str).to_numpy(),step=r.step.to_numpy(np.int8),
                    label=yy[r._link_idx.to_numpy(int),row_step],target_mask=mm[r._link_idx.to_numpy(int),row_step],win=np.int32(wix[wid]),pan=np.int8(panel_idx[panel]),center=row_center)))
                center_parts[name].append(pd.DataFrame(dict(window_id=wid,panel=panel,date=date,link_id=P.links,T=T,center_row=np.arange(center_offset[name]+i*L,center_offset[name]+(i+1)*L))))
                if i==0:
                    for source in ['matched','raw']:audits.append(dict(cohort=name,**causal_QA(P,c,date,T,wid,raw,past if source=='matched'else None)))
                rv=raw['speed_kmh'][:,T-12:T];tv=past['speed_kmh'];histcells+=rv.size;rawfinite+=np.isfinite(rv).sum();truthfinite+=np.isfinite(tv).sum()
                both=np.isfinite(rv)&np.isfinite(tv);finite_value_diff+=int((rv[both]!=tv[both]).sum());rows+=nr
                if name=='fit':all_fit_dates.add(date)
                winmeta.append(dict(window_id=wid,eligible_cells=int(mm.sum()),physical_cells=L*6,switch_paths=int(kk.sum())))
            assert labels_bad==eligible_bad==0,(panel,name,labels_bad,eligible_bad)
            for x in enc.values():x.flush();x._mmap.close()
            for x in arrays.values():x.flush();x._mmap.close()
            np.savez(CACHE/f'{name}_{panel}_targets.npz',**target)
            records.append(dict(panel=panel,cohort=name,downstream_rows=rows,centers=N,windows=len(wids),eligible_cells=int(target['mask'].sum()),
                legacy_label_mismatches=labels_bad,legacy_eligible_mismatches=eligible_bad,window_support=winmeta,
                prefix_support=dict(cells=int(histcells),raw_finite_fraction=float(rawfinite/histcells),matched_finite_fraction=float(truthfinite/histcells),overlap_value_difference=finite_value_diff)))
            offset[name]+=rows;center_offset[name]+=N;del q,target,enc,arrays,frames,past,raw,truth;gc.collect()
            print('R20_PREPARED_PANEL',panel,name,rows,N,'seconds',round(time.time()-start,1),flush=True)
        del d,P,c;D.REC.day_q.cache_clear();gc.collect();pa.default_memory_pool().release_unused()
    dates=sorted(all_fit_dates);fold={date:i%5 for i,date in enumerate(dates)};assert max(dates)<=FIT_END
    for name in sizes:
        assert offset[name]==sizes[name]and center_offset[name]==centers[name]
        m=pd.concat(metadata_parts[name],ignore_index=True);idx=pd.concat(center_parts[name],ignore_index=True)
        assert not m[KEY].duplicated().any()and np.array_equal(idx.center_row.to_numpy(),np.arange(len(idx)))
        idx['fold']=idx.date.map(fold).fillna(-1).astype(np.int8);m['fold']=m.date.map(fold).fillna(-1).astype(np.int8)
        assert (m.fold>=0).all()if name=='fit'else(m.fold==-1).all()
        idx.to_parquet(CACHE/f'{name}_CENTER_INDEX.parquet',index=False);m.to_parquet(CACHE/f'{name}_ROW_INDEX.parquet',index=False)
        nc=m.groupby('window_id').label.transform('size').to_numpy(float);pwin=m[['panel','window_id']].drop_duplicates().groupby('panel').size()
        weight=1/(nc*m.panel.map(pwin).to_numpy());weight/=weight.mean()
        if name=='fit':assert m.target_mask.all()
        np.savez(CACHE/f'{name}_meta.npz',label=m.label.to_numpy(np.float32),target_mask=m.target_mask.to_numpy(bool),weight=weight.astype(np.float64),
            win=m.win.to_numpy(np.int32),pan=m.pan.to_numpy(np.int8),center=m.center.to_numpy(np.int64),step=m.step.to_numpy(np.int8)-1,fold=m.fold.to_numpy(np.int8))
    result=dict(PASS=True,status='DATA_READY_NO_NEW_FITS_NO_CANDIDATE_SCORES',fit_end=FIT_END,rows=sizes,centers=centers,
        windows={name:int(sum(counts[p,name]for p in PANELS))for name in sizes},fit_dates=dates,date_folds=fold,base_features=FS,semantic_features=SEMANTICS,
        records=records,causal_audits=audits,source_hashes=source_hashes,strict_blocks_path=str(BLOCKS),strict_blocks_sha256=sha(BLOCKS),
        prereg_sha256=sha(ME/'PREREG_DATA.json'),copied_code_sha256=sha(ME/'COPIED_CODE.json'),row_plan_sha256=sha(ME/'ROW_PLAN.json'),
        same_keys_labels_between_sources=True,held_full_physical_rows_before_mask=True,new_model_parameter_updates=0,
        seconds=time.time()-start,peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9)
    dump(ME/'PREPARED.json',result);print('R20_DATA_READY',result['rows'],result['windows'],result['seconds'],result['peak_RSS_GB'],flush=True)
if __name__=='__main__':run()
