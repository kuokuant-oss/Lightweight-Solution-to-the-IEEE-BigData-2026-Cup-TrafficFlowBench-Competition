"""Q4: ongoing recency refit. Same recipe as rep2/train_rep2.py R_V2selrec seed0 (NEW_PARAMS, 25% selector-dropout windows rng 123,
per-window/per-panel weights), but FIXED 693 rounds (production seed0 best iteration) and fit end given on the CLI:
  train_refit.py 2030-12-31 ctrl   -> control (production data window, retrained at 2 threads)
  train_refit.py 2031-02-28 refit  -> Jan+Feb added (the months production used for ES / calibration)
Memory-lean: reads only needed columns, float32 matrix, no pandas copy of the full frame."""
# This file is `train_refit3.py`: the docstring above is shared with `train_refit.py`; this
# version takes the seed and the variant on the command line.
#
# Training record for five of the six N6 refits. The reproduction does not run this program;
# it loads the shipped models. The program reads window tables that are not shipped, and its
# paths resolve from the hard-coded development root (`ROOT`, mapped by the path layer).
#
# Usage: `train_refit3.py <fit_end> <tag> <seed> [variant]` (variant R_V2selrec, the default,
#   or R_V2geomselrec). Rounds = tree count of the production model
#   `rep2/m_<variant>_s0_seed<seed>.txt`, whose input order is checked; seed 20260922 + seed;
#   2 threads; no early stopping. Data, selector dropout and weights as in `train_refit.py`;
#   window, link and corridor ids become integer codes, and the recency inputs are joined by
#   a sorted integer key (window, link, step) to save memory.
# Output: `m_<variant>_<tag>_seed<seed>.txt` and `train_<variant>_<tag>_seed<seed>.json`; an
#   existing model file is never overwritten.
# Shipped: `m_R_V2selrec_refit_seed{1,2}.txt` and `m_R_V2geomselrec_refit_seed{0,1,2}.txt`
#   (fit end 2031-02-28, tag refit; run by `run_rest.sh`) under
#   `trafficflowbench/build/review_20260929_target092/contract_audit/q4_refit/`.
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[n]='2'
import sys, json, glob, importlib.util, gc, time, resource
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc
ROOT=Path('/Volumes/Crucial X9/trafficflow/trafficflowbench')
spec=importlib.util.spec_from_file_location('Q',ROOT/'scripts/strongest_20260922_ongoing.py'); Q=importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
HERE=Path(__file__).resolve().parent; REP2=ROOT/'build/review_20260924_ctm/rep2'
fit_end=sys.argv[1]; tag=sys.argv[2]; SEED=int(sys.argv[3]); VAR=sys.argv[4] if len(sys.argv)>4 else 'R_V2selrec'
prod=lgb.Booster(model_file=str(REP2/f'm_{VAR}_s0_seed{SEED}.txt')); FS=prod.feature_name(); ROUNDS=prod.num_trees()
man=json.loads((ROOT/'build/review_20260923/ongoing_P1/manifest.json').read_text())
V2=[f for f in man['features'] if f not in ('link_code','panel_code','queue_prior')]
GEOM=['g_gap_km','g_gap_links','g_tail_vel','g_rh_w','g_eta_rh','g_eta_vel','g_block_len_km','g_block_links','g_head_age','g_in_block','g_pos_in_block_km','g_clear_eta_rh','g_n_blocks','g_link_age','g_q_up','g_q_in','g_tail_age','g_eta_rh_m_step','g_eta_vel_m_step','g_clear_m_step']
assert FS==V2+(GEOM if 'geom' in VAR else [])+['prev_rej','n_rej']+['rec_s','rec_any','rec_n'], 'feature order differs from train_rep2 recipe'
t0=time.time(); rss=lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,2)
files=sorted(glob.glob(str(REP2/'rep2_s0_*.parquet'))); files=[f for f in files if '/._' not in f]
# 1) reproduce the window order of the original concatenated frame -> same 25% selector-dropout windows
order=[]; seen=set()
for f in files:
    for w in pd.unique(pq.read_table(f,columns=['window_id']).column(0).to_numpy(zero_copy_only=False)):
        if w not in seen: seen.add(w); order.append(w)
WIX={w:i for i,w in enumerate(order)}; wids=np.array(order,dtype=object); rng=np.random.default_rng(123); maskw=set(wids[rng.random(len(wids))<0.25])
print('windows',len(wids),'masked',len(maskw),flush=True)
# LIX: link id -> integer code, grown on first sight.
LIX={}
def codes(col,table_map):
    """Maps a pyarrow column to integer codes through `table_map`.

    With `LIX` (link ids) unseen values get new codes; with any other map (window ids
    `WIX`) unseen values get -1.
    """
    de=col.combine_chunks().dictionary_encode(); dic=de.dictionary.to_pylist()
    lut=np.array([table_map.setdefault(x,len(table_map)) if table_map is LIX else table_map.get(x,-1) for x in dic],np.int64)
    return lut[de.indices.to_numpy(zero_copy_only=False)]
# Recency table as sorted keys RK = (window code x 100000 + link code) x 100 + step and
# values RV (rec_s, rec_any, rec_n); rows of windows outside the rep2 tables are dropped.
rt=pq.read_table(ROOT/'build/review_20260924_ctm/ruler2/rec_train_queue_ongoing_s0.parquet')
rw=codes(rt['window_id'],WIX); ok=rw>=0
rl=codes(rt['link_id'].cast(pa.string()),LIX)
RK=(rw[ok]*100000+rl[ok])*100+rt['step'].to_numpy()[ok].astype(np.int64)
RV=np.stack([rt[c].to_numpy(zero_copy_only=False).astype(np.float32)[ok] for c in ('rec_s','rec_any','rec_n')],1)
o_=np.argsort(RK); RK=RK[o_]; RV=RV[o_]; assert len(np.unique(RK))==len(RK); del rt,rw,rl,o_; gc.collect(); print('rec keys',len(RK),'rss',rss(),flush=True)
base_cols=[c for c in FS if c not in ('rec_s','rec_any','rec_n')]
import pyarrow as pa, pyarrow.compute as pc
def filt(f):
    """Returns the row mask of one table: eligible rows of replicated windows up to `fit_end`."""
    t=pq.read_table(f,columns=['_eligible','is_rep','date'])
    return pc.and_(pc.and_(t['_eligible'],pc.equal(t['is_rep'],1)),pc.less_equal(t['date'],fit_end))
counts=[int(pc.sum(filt(f)).as_py() or 0) for f in files]; N=sum(counts); print('rows',N,counts,flush=True)
# 2) fill a preallocated float32 matrix table by table; recency looked up by binary search on RK
X=np.empty((N,len(FS)),np.float32); y=np.empty(N,np.float32); win=np.empty(N,np.int32); pan=np.empty(N,np.int8); per=np.empty(N,np.int8); PIX={}; o=0
for f,n in zip(files,counts):
    m_=filt(f)
    t=pq.read_table(f,columns=base_cols+['label','window_id','panel','date','link_id','step']).filter(m_)
    wc=codes(t['window_id'],WIX); assert (wc>=0).all()
    lc=codes(t['link_id'].cast(pa.string()),LIX)
    key=(wc*100000+lc)*100+t['step'].to_numpy().astype(np.int64); pos=np.clip(np.searchsorted(RK,key),0,len(RK)-1); hit=RK[pos]==key
    hit_rate=hit.mean(); RR=np.full((n,3),np.nan,np.float32); RR[hit]=RV[pos[hit]]; RC={'rec_s':RR[:,0],'rec_any':RR[:,1],'rec_n':RR[:,2]}
    for j,c in enumerate(FS):
        X[o:o+n,j]=(RC[c] if c in RC else t[c].to_numpy(zero_copy_only=False).astype(np.float32))
    MWC=np.array(sorted(WIX[w] for w in maskw)); mk=np.isin(wc,MWC)
    sl=slice(o,o+n); X[sl][mk,FS.index('prev_rej')]=np.nan; X[sl][mk,FS.index('n_rej')]=np.nan
    # Labels, window codes and corridor codes (PIX); `pn` is always None.
    y[sl]=t['label'].to_numpy(zero_copy_only=False).astype(np.float32); win[sl]=wc.astype(np.int32); pn=codes(t['panel'],PIX) if False else None; pd_=t['panel'].combine_chunks().dictionary_encode(); lut=np.array([PIX.setdefault(x,len(PIX)) for x in pd_.dictionary.to_pylist()]); pan[sl]=lut[pd_.indices.to_numpy(zero_copy_only=False)]
    dt=t['date'].to_numpy(zero_copy_only=False); per[sl]=np.where(dt<='2030-12-31',0,np.where(dt<='2031-01-31',1,2))
    o+=n; del t,wc,lc,key,pos,hit,RR,RC,pn; gc.collect(); pa.default_memory_pool().release_unused(); print(os.path.basename(f),n,'rec_hit',round(float(hit_rate),3),'rss',rss(),flush=True)
assert o==N
chk=np.isnan(X[:,FS.index('prev_rej')]); print('prev_rej nan frac',round(float(chk.mean()),4),flush=True)
P=dict(Q.NEW_PARAMS,num_threads=2,seed=20260922+SEED)
# 3) weights 1 / (rows of the window x windows of the corridor), normalised to mean 1
s=pd.DataFrame({'w':win,'p':pan}); nc=s.groupby('w').w.transform('size').to_numpy(float)
pwin=s.drop_duplicates('w').groupby('p').size(); wt=1/(nc*s.p.map(pwin).to_numpy()); wt=wt/wt.mean()
info=dict(var=VAR,seed=SEED,tag=tag,fit_end=fit_end,rounds=ROUNDS,rows=int(len(y)),windows=int(s.w.nunique()),windows_by_period={k:int(len(np.unique(win[per==i]))) for i,k in enumerate(('fit','jan','feb'))},
          pos_rate=float(y.mean()),features=len(FS))
print(json.dumps(info),'rss',rss(),flush=True)
ds=lgb.Dataset(X,y,weight=wt,feature_name=FS,free_raw_data=True,params=P); ds.construct(); del X; gc.collect(); print('dataset built',round(time.time()-t0),'s rss',rss(),flush=True)
def cb(env):
    """Prints elapsed time and peak memory every 100 rounds."""
    if (env.iteration+1)%100==0: print('iter',env.iteration+1,round(time.time()-t0),'s rss',rss(),flush=True)
m=lgb.train(P,ds,num_boost_round=ROUNDS,callbacks=[cb])
out=HERE/f'm_{VAR}_{tag}_seed{SEED}.txt'; assert not out.exists(), 'refuse overwrite'
m.save_model(str(out)); info.update(seconds=round(time.time()-t0),maxrss_gb=rss(),model=str(out),trees=m.num_trees())
json.dump(info,open(HERE/f'train_{VAR}_{tag}_seed{SEED}.json','w'),indent=1); print('DONE',json.dumps(info),flush=True)
