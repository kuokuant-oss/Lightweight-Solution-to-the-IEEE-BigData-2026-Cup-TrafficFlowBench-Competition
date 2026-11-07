"""Q4: ongoing recency refit. Same recipe as rep2/train_rep2.py R_V2selrec seed0 (NEW_PARAMS, 25% selector-dropout windows rng 123,
per-window/per-panel weights), but FIXED 693 rounds (production seed0 best iteration) and fit end given on the CLI:
  train_refit.py 2030-12-31 ctrl   -> control (production data window, retrained at 2 threads)
  train_refit.py 2031-02-28 refit  -> Jan+Feb added (the months production used for ES / calibration)
Memory-lean: reads only needed columns, float32 matrix, no pandas copy of the full frame."""
# Training record for the N6 refit of selrec seed 0. The reproduction does not run this
# program; it loads the shipped model. The program reads window tables that are not shipped,
# and its paths resolve from the hard-coded development root (`ROOT`, mapped by the path layer).
#
# Data: the rep2 tables `rep2/rep2_s0_<corridor>.parquet` (eligible rows of replicated
#   ongoing windows of the 8 Task 2 corridors dated up to the fit end) with the recency inputs
#   of `ruler2/rec_train_queue_ongoing_s0.parquet`. The same 25% of windows as in
#   `rep2/train_rep2.py` (rng seed 123 over the window order of the original frame) get
#   their selector inputs blanked.
# Settings: `NEW_PARAMS` of `scripts/strongest_20260922_ongoing.py`, seed 20260922, 2 threads,
#   fixed rounds (default 693 = tree count of the production `rep2/m_R_V2selrec_s0_seed0.txt`,
#   whose input order is checked), no early stopping. Weights as in `train_rep2.py`.
# Usage: `train_refit.py <fit_end> <tag> [rounds]`. Output: `m_R_V2selrec_<tag>_seed0.txt` and
#   `train_<tag>.json` (rows, windows per period, seconds, peak memory); an existing model
#   file is never overwritten.
# Shipped: `m_R_V2selrec_refit_seed0.txt` from `train_refit.py 2031-02-28 refit`
#   (2030-06-01 .. 2031-02-28, 3,013 windows, 2,582,803 rows) under
#   `trafficflowbench/build/review_20260929_target092/contract_audit/q4_refit/`.
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[n]='2'
import sys, json, glob, importlib.util, gc, time, resource
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq
ROOT=Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
spec=importlib.util.spec_from_file_location('Q',ROOT/'scripts/strongest_20260922_ongoing.py'); Q=importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
HERE=Path(__file__).resolve().parent; REP2=ROOT/'build/review_20260924_ctm/rep2'
fit_end=sys.argv[1]; tag=sys.argv[2]; ROUNDS=int(sys.argv[3]) if len(sys.argv)>3 else 693
prod=lgb.Booster(model_file=str(REP2/'m_R_V2selrec_s0_seed0.txt')); FS=prod.feature_name(); assert prod.num_trees()==693
man=json.loads((ROOT/'build/review_20260923/ongoing_P1/manifest.json').read_text())
V2=[f for f in man['features'] if f not in ('link_code','panel_code','queue_prior')]
assert FS==V2+['prev_rej','n_rej']+['rec_s','rec_any','rec_n'], 'feature order differs from train_rep2 recipe'
t0=time.time(); rss=lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,2)
files=sorted(glob.glob(str(REP2/'rep2_s0_*.parquet'))); files=[f for f in files if '/._' not in f]
rec=pd.read_parquet(ROOT/'build/review_20260924_ctm/ruler2/rec_train_queue_ongoing_s0.parquet'); rec['link_id']=rec.link_id.astype(str)
# 1) reproduce the window order of the original concatenated frame -> same 25% selector-dropout windows
order=[]; seen=set()
for f in files:
    for w in pd.unique(pq.read_table(f,columns=['window_id']).column(0).to_numpy(zero_copy_only=False)):
        if w not in seen: seen.add(w); order.append(w)
wids=np.array(order,dtype=object); rng=np.random.default_rng(123); maskw=set(wids[rng.random(len(wids))<0.25])
print('windows',len(wids),'masked',len(maskw),flush=True)
base_cols=[c for c in FS if c not in ('rec_s','rec_any','rec_n')]
import pyarrow as pa, pyarrow.compute as pc
def filt(f):
    """Returns the row mask of one table: eligible rows of replicated windows up to `fit_end`."""
    t=pq.read_table(f,columns=['_eligible','is_rep','date'])
    return pc.and_(pc.and_(t['_eligible'],pc.equal(t['is_rep'],1)),pc.less_equal(t['date'],fit_end))
counts=[int(pc.sum(filt(f)).as_py() or 0) for f in files]; N=sum(counts); print('rows',N,counts,flush=True)
# 2) fill a preallocated float32 matrix table by table; recency inputs joined by (window, link, step)
X=np.empty((N,len(FS)),np.float32); y=np.empty(N,np.float32); win=np.empty(N,object); pan=np.empty(N,object); per=np.empty(N,object); o=0
for f,n in zip(files,counts):
    m_=filt(f)
    t=pq.read_table(f,columns=base_cols+['label','window_id','panel','date','link_id','step']).filter(m_)
    k=pd.DataFrame({'window_id':t['window_id'].to_numpy(zero_copy_only=False),'link_id':t['link_id'].to_numpy(zero_copy_only=False).astype(str),'step':t['step'].to_numpy()})
    r=k.merge(rec,on=['window_id','link_id','step'],how='left'); assert len(r)==n
    for j,c in enumerate(FS):
        X[o:o+n,j]=(r[c].to_numpy(np.float32) if c in ('rec_s','rec_any','rec_n') else t[c].to_numpy(zero_copy_only=False).astype(np.float32))
    mk=k.window_id.isin(maskw).to_numpy()
    sl=slice(o,o+n); X[sl][mk,FS.index('prev_rej')]=np.nan; X[sl][mk,FS.index('n_rej')]=np.nan
    y[sl]=t['label'].to_numpy(zero_copy_only=False).astype(np.float32); win[sl]=k.window_id.to_numpy(object); pan[sl]=t['panel'].to_numpy(zero_copy_only=False)
    dt=t['date'].to_numpy(zero_copy_only=False); per[sl]=np.where(dt<='2030-12-31','fit',np.where(dt<='2031-01-31','jan','feb'))
    o+=n; del t,k,r; gc.collect(); pa.default_memory_pool().release_unused(); print(os.path.basename(f),n,'rss',rss(),flush=True)
assert o==N
chk=np.isnan(X[:,FS.index('prev_rej')]); print('prev_rej nan frac',round(float(chk.mean()),4),flush=True)
P=dict(Q.NEW_PARAMS,num_threads=2,seed=20260922+0)
# 3) weights 1 / (rows of the window x windows of the corridor), normalised to mean 1
s=pd.DataFrame({'w':win,'p':pan}); nc=s.groupby('w').w.transform('size').to_numpy(float)
pwin=s.drop_duplicates('w').groupby('p').size(); wt=1/(nc*s.p.map(pwin).to_numpy()); wt=wt/wt.mean()
info=dict(tag=tag,fit_end=fit_end,rounds=ROUNDS,rows=int(len(y)),windows=int(s.w.nunique()),windows_by_period={k:int(len(set(win[per==k]))) for k in ('fit','jan','feb')},
          pos_rate=float(y.mean()),features=len(FS))
print(json.dumps(info),'rss',rss(),flush=True)
ds=lgb.Dataset(X,y,weight=wt,feature_name=FS,free_raw_data=True,params=P); ds.construct(); del X; gc.collect(); print('dataset built',round(time.time()-t0),'s rss',rss(),flush=True)
def cb(env):
    """Prints elapsed time and peak memory every 100 rounds."""
    if (env.iteration+1)%100==0: print('iter',env.iteration+1,round(time.time()-t0),'s rss',rss(),flush=True)
m=lgb.train(P,ds,num_boost_round=ROUNDS,callbacks=[cb])
out=HERE/f'm_R_V2selrec_{tag}_seed0.txt'; assert not out.exists(), 'refuse overwrite'
m.save_model(str(out)); info.update(seconds=round(time.time()-t0),maxrss_gb=rss(),model=str(out),trees=m.num_trees())
json.dump(info,open(HERE/f'train_{tag}.json','w'),indent=1); print('DONE',json.dumps(info),flush=True)
