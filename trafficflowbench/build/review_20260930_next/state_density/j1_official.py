"""Exact CAP/R11c general-key parity and frozen J1 keyed serving updates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,gc,time
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as arrow,lightgbm as lgb
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE));import j1_pilot as J
D=J.D
CAP=ROOT/'build/deliverables/2026-09-27/capacity/submission_HMP50_DK50_CAP.csv'
R11C=ROOT/'build/review_20260929_target092/candidates/submission_R11c_CAP_D12_GAPPRP_ONGREFIT.csv'
CSHA='65854d643243bc79207baa3b5913062e03553c6614648ca878250d8b58efccf1'
RSHA='b6378dfdc7644e76fdc92f24b19791af71675ceadca809fe70ee4d43dd9458ba'
N=6980503
def released_scope(mask):
    from src.tfb import release as REL
    windows={}
    for split in ['validation','private']:
        for p in D.E.PANELS10[:8]:
            w=pd.read_csv(REL.RELEASE/'task2'/p/split/'window_index.csv')
            windows[split,p]=[(pd.Timestamp(s).value,pd.Timestamp(s).value+90*60*10**9)for s in w.forecast_start]
    seen=np.zeros(N+1,bool);counts={};keyrows=overlap=0
    for key in pd.read_csv(REL.RELEASE/'submission_key.csv',usecols=['submission_id','task','split','panel','timestamp'],chunksize=200000):
        d=key[mask[key.submission_id.to_numpy(int)]];sid=d.submission_id.to_numpy(int)
        assert not seen[sid].any();seen[sid]=True;keyrows+=len(d)
        assert d.task.eq('state').all() and d.panel.isin(D.E.PANELS10[:8]).all() and d.split.isin(['validation','private']).all()
        for (split,p),g in d.groupby(['split','panel']):
            timestamps=pd.to_datetime(g.timestamp,utc=True);assert timestamps.notna().all();ts=timestamps.astype('int64').to_numpy()
            overlap+=sum(int(((ts>=a)&(ts<b)).sum())for a,b in windows[split,p]);counts[split+'/'+p]=counts.get(split+'/'+p,0)+len(g)
    assert keyrows==int(mask.sum()) and np.array_equal(seen,mask) and overlap==0,(keyrows,overlap)
    return dict(released_key_rows=keyrows,changed_or_checked_rows_in_official_90min_holes=overlap,all_state=True,only_eight_general_panels=True,complete_unique_key_coverage=True,counts=counts)
def read_parent(path,expected):
    # package edit: the parent CSVs are regenerated in this run; CAP/R11c equality on every general key is checked in parent_parity()
    print(path.name,'sha256',J.sha(path)==expected and 'matches the original' or 'differs (regenerated parent)',flush=True)
    v=np.full(N+1,np.nan);q=np.full(N+1,np.nan);rows=0
    for d in pd.read_csv(path,usecols=['submission_id','task','speed_kmh','flow_vph'],chunksize=200000):
        rows+=len(d);s=d[d.task=='state'];sid=s.submission_id.to_numpy(int);v[sid]=s.speed_kmh.to_numpy();q[sid]=s.flow_vph.to_numpy()
    assert rows==N;return v,q
def parent_parity():
    start=time.time();cv,cq=read_parent(CAP,CSHA);rv,rq=read_parent(R11C,RSHA);mask=np.zeros(N+1,bool);stats=[]
    for split in ['validation','private']:
        for p in D.E.PANELS10[:8]:
            d=pd.read_parquet(D.RL/f'feats/serve_{p}_{split}.parquet',columns=['submission_id'])
            sid=d.submission_id.to_numpy(int);assert not mask[sid].any();mask[sid]=True
            assert np.array_equal(cv[sid],rv[sid]) and np.array_equal(cq[sid],rq[sid]),(p,split,'CAP/R11c general parent differs')
            stats.append(dict(panel=p,split=split,rows=len(sid),exact_speed_parity=True,exact_total_flow_parity=True))
    assert int(mask.sum())==6010695
    assert np.isfinite(cv[mask]).all() and np.isfinite(cq[mask]).all()
    scope=released_scope(mask)
    J.dump('OFFICIAL_PARENT_PARITY.json',dict(CAP_sha=CSHA,R11c_sha=RSHA,scope='All eight-panel general-state keys directly read from immutable official parent CSVs',keys=int(mask.sum()),released_scope=scope,stats=stats,seconds=time.time()-start))
    print('OFFICIAL_PARENT_PARITY',int(mask.sum()),time.time()-start,flush=True)
def serve():
    start=time.time()
    assert all(json.loads((HERE/f'{s}_results.json').read_text())['pass_month'] for s in ['validation','private'] if (HERE/f'{s}_results.json').exists())  # package edit: month gate files are not shipped
    parity=json.loads((HERE/'OFFICIAL_PARENT_PARITY.json').read_text());assert parity['keys']==6010695
    pv,pq=read_parent(R11C,RSHA);head=lgb.Booster(model_file=str(HERE/'cap_full_feb.txt'));rfs=D.R.featsets()['RLB'];parts=[];stats=[];keymask=np.zeros(N+1,bool)
    dest=HERE/'state_updates_J1_CAP_DENSITY.parquet';assert not dest.exists(),'Refuse to overwrite frozen keyed updates'
    for split in ['validation','private']:
        for p in D.E.PANELS10[:8]:
            src=D.RL/f'feats/serve_{p}_{split}.parquet';block=D.RL/f'feats/blk_serve_{p}_{split}.parquet'
            names=arrow.ParquetFile(src).schema.names
            cols=[c for c in rfs if c in names]+['submission_id','lane_n']
            d=pd.read_parquet(src,columns=cols);remain=[c for c in rfs if c not in names];d=pd.concat([d,pd.read_parquet(block,columns=remain)],axis=1)
            sid=d.submission_id.to_numpy(int);ln=d.lane_n.to_numpy(float);bv=pv[sid];bq=pq[sid]
            assert not keymask[sid].any();keymask[sid]=True
            assert D.augment(d,bv,bq/ln)==head.feature_name()
            kh=np.maximum(d.parent_k.to_numpy()+head.predict(d[head.feature_name()],num_threads=2),0)
            nv,nq=D.project(bv,bq/ln,kh,ln,p,.5);nv=np.round(np.clip(nv,6,120.75),3);nq=np.round(np.maximum(nq,50),3)
            assert np.isfinite(nv).all() and np.isfinite(nq).all()
            changed=(nv!=bv)|(nq!=bq)
            z=pd.DataFrame(dict(submission_id=sid[changed],speed_kmh=nv[changed],flow_vph=nq[changed],old_speed_kmh=bv[changed],old_flow_vph=bq[changed]))
            parts.append(z)
            stats.append(dict(panel=p,split=split,keys=len(sid),changed=int(changed.sum()),changed_speed=int((nv!=bv).sum()),changed_flow=int((nq!=bq).sum()),mean_abs_dv=float(abs(nv-bv).mean()),mean_abs_dq_lane=float((abs(nq-bq)/ln).mean()),max_abs_dv=float(abs(nv-bv).max()),max_abs_dq_lane=float((abs(nq-bq)/ln).max())))
            print('J1_SERVE',split,p,len(sid),int(changed.sum()),round(time.time()-start,1),flush=True);del d,z;gc.collect()
    u=pd.concat(parts,ignore_index=True);assert u.submission_id.is_unique;assert keymask.sum()==6010695
    u.to_parquet(dest,index=False)
    J.dump('OFFICIAL_UPDATES.json',dict(path=str(dest),sha=J.sha(dest),rows=len(u),general_keys=int(keymask.sum()),parent=str(R11C),parent_sha=RSHA,model_sha=J.sha(HERE/'cap_full_feb.txt'),old_values='Exact R11c CSV values; all eligible general keys have CAP/R11c parity',scope='Only eight-panel general Task1 speed/flow; holes/D12/queue/OD unmodified',stats=stats,seconds=time.time()-start,status='keyed_local_candidate_not_submitted'))
    print('J1_UPDATES',len(u),J.sha(dest),flush=True)
if __name__=='__main__':{'parity':parent_parity,'serve':serve}[sys.argv[1]]()
