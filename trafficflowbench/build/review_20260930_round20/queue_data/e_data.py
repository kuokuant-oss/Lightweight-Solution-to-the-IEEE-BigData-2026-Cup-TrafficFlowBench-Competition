"""Q3-E genuine <=T published inputs, exact train truth targets kept separate."""
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ME=Path(__file__).resolve().parent;ROOT=ME.parents[2];CT=ROOT/'build/review_20260924_ctm';ECACHE=ME/'cache'
sys.path.insert(0,str(ROOT));from src.tfb import release as R,grid,task2feat as TF
PANELS=['D12_I5_N','D12_I5_S','D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S']
FIELDS=['speed_kmh','flow_vph','occupancy'];SEMANTICS=[f'TCN_p_queue_step{s}'for s in range(1,7)]+['TCN_first_switch_capped_step']
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
class PublishedPanel:
    def __init__(self,panel,split='train'):
        self.panel=panel;self.split=split
        f=R.truth_files(panel,'train')[0];order=pd.read_parquet(f,columns=['link_id','milepost']).drop_duplicates().sort_values('milepost')
        self.links=order.link_id.astype(str).to_numpy();self.mp=order.milepost.to_numpy(float);self.li={l:i for i,l in enumerate(self.links)}
        assert np.all(np.diff(self.mp)>=0) and len(set(self.links))==len(self.links)
        static=TF.static_table(panel,list(self.links),grid,R);self.vf=static.free_speed.to_numpy(float);self.cap=static.capacity.to_numpy(float);self.vcut=static.v_cut.to_numpy(float)
        paths=[p for regime in R.REGIME_ORDER for p in R.masked_files(panel,split,regime)]
        self.published={p.stem.replace('synthetic_mainline_','').replace('_','-'):p for p in paths}
        assert len(self.published)==len(paths),'one published mainline file per calendar date'
        self.truth={p.stem.replace('synthetic_mainline_','').replace('_','-'):p for p in R.truth_files(panel,'train')}
    def day(self,path,truth=False):
        cols=['timestamp','link_id']+FIELDS+['is_score_eligible']
        d=pd.read_parquet(path,columns=cols);rr=d.link_id.astype(str).map(self.li).to_numpy();ts=pd.to_datetime(d.timestamp,utc=True);cc=(ts.dt.hour*12+ts.dt.minute//5).to_numpy()
        assert np.isfinite(rr).all() and not d[['timestamp','link_id']].duplicated().any() and len(d)==len(self.links)*288
        out={}
        for c in FIELDS+['is_score_eligible']:
            a=np.full((len(self.links),288),np.nan);a[rr,cc]=d[c].to_numpy();out[c]=a.astype(bool)if c=='is_score_eligible'else a
        return out
    def encode(self,raw,T,history=None,origin=None):
        assert 12<=T<=281
        # Truncate before any operation; neither interpolation nor offline reconstruction is used.
        hist={c:np.asarray(raw[c][:,T-12:T+1],np.float64).copy()for c in FIELDS}
        if history is not None:
            assert origin is not None;h=history.copy();ts=pd.to_datetime(h.timestamp,utc=True)
            assert (ts<origin).all(),'own published window_history must be strictly before origin'
            cc=(ts.dt.hour*12+ts.dt.minute//5).to_numpy();assert (cc>=T-12).all()and(cc<T).all()
            rr=h.link_id.astype(str).map(self.li).to_numpy();assert np.isfinite(rr).all()
            assert not h[['timestamp','link_id']].duplicated().any()
            for c in FIELDS:hist[c][rr,cc-(T-12)]=h[c].to_numpy()
        mask=np.logical_and.reduce([np.isfinite(hist[c])for c in FIELDS])
        vals=np.stack([hist['speed_kmh']/self.vf[:,None],hist['flow_vph']/self.cap[:,None],hist['occupancy'],mask.astype(float)],axis=-1)
        vals=np.nan_to_num(vals,nan=0.,posinf=0.,neginf=0.).astype(np.float32)
        neigh=np.arange(len(self.links))[:,None]+np.arange(-4,5)[None,:];valid=(neigh>=0)&(neigh<len(self.links));idx=np.clip(neigh,0,len(self.links)-1)
        x=vals[idx];x[~valid]=0.;x=x.transpose(0,3,1,2).copy();assert x.shape==(len(self.links),4,9,13)
        return x
    def targets(self,truth,T):
        v=truth['speed_kmh'][:,T+1:T+7];m=truth['is_score_eligible'][:,T+1:T+7]
        assert v.shape==(len(self.links),6) and np.isfinite(v[m]).all()
        y=(v<=self.vcut[:,None]).astype(np.float32)
        originv=truth['speed_kmh'][:,T];y0=originv<=self.vcut
        known=m.all(1)&np.isfinite(v).all(1)&np.isfinite(originv)
        changes=y.astype(bool)!=y0[:,None];sw=np.where(changes.any(1),changes.argmax(1),6).astype(np.int64)
        return y,m,sw,known
    def own_history(self,wid,origin):
        """Genuine release history only; synthetic train/pseudo windows have no such file key."""
        if not hasattr(self,'history'):
            path=R.RELEASE/'task2'/self.panel/self.split/'window_history.parquet'
            self.history=pd.read_parquet(path) if path.exists()else pd.DataFrame()
            if len(self.history):self.history['link_id']=self.history.link_id.astype(str)
        if not len(self.history):return None
        h=self.history[self.history.window_id==wid].copy()
        if not len(h):return None
        ts=pd.to_datetime(h.timestamp,utc=True)
        assert (ts<origin).all() and (ts>=origin-pd.Timedelta(minutes=60)).all()
        return h
def poison(panel,raw,T,history=None,origin=None):
    before=panel.encode(raw,T,history,origin);afterraw={k:v.copy()for k,v in raw.items()}
    for a in afterraw.values():a[:,T+1:]=987654.
    after=panel.encode(afterraw,T,history,origin);assert np.array_equal(before,after)
    return dict(T=int(T),raw_source_post_T_poison_exact=True,shape=list(before.shape),source='rawpublished<=T + ownhistory<T when supplied; no offline latent')
