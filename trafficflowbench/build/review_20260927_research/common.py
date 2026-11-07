from pathlib import Path
import sys,os,json,gc
os.environ.setdefault('OMP_NUM_THREADS','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
R2=ROOT/'build/review_20260925/round2/task1_trans';R3=ROOT/'build/review_20260925/round3/task1_trans_serving'
HM=ROOT/'build/review_20260927/iteration2/hole_mixture'
sys.path.insert(0,str(R3));import serve_lib as S
import numpy as np,pandas as pd,lightgbm as lgb
KEY=['panel','date','origin','li','step_i']
RAW_S=['s_v','s_ql','s_qf','s_n','s_vq','s_tq','s_bqf','s_v_pre','s_qf_pre','s_bqf_pre','s_v_post','s_qf_post','s_bqf_post']
def load(split='validation'):
    source=R2 if split=='validation' else R3
    d=S.load_ph([source/'work'/f'ph_{split}_{p}.parquet' for p in S.PANELS])
    old=pd.read_parquet(HM/f'{split}_strict_predictions.parquet')
    assert not old.duplicated(KEY).any()
    old=old[KEY+['base','blend25','pqueue']].rename(columns={'base':'t1','blend25':'hm25','pqueue':'old_pqueue'})
    d=d.merge(old,on=KEY,validate='one_to_one',sort=False)
    assert np.isfinite(d.hm25).all()
    st=pd.concat([pd.read_parquet(source/'verify/work'/f'strict_{split}_{p}.parquet') for p in S.PANELS],ignore_index=True)
    for k in ['panel','origin','li','step_i']:assert np.array_equal(st[k].to_numpy(),d[k].to_numpy()),k
    return d,st

def strict(d,st,odd):
    x=d.copy()
    for k in RAW_S:x[k]=st[k+('_fA' if odd else '_fB')].to_numpy()
    S.derive(x);return x

def evaluate(d,pred,base='hm25'):
    e0=(d[base].to_numpy()-d.y_v.to_numpy())**2;e1=(np.asarray(pred)-d.y_v.to_numpy())**2
    out={}
    for nm,mask in [('all',np.ones(len(d),bool)),('odd',d.day%2==1),('even',d.day%2==0),('early12',d.day<=12),('onset',d.cond=='queue_onset'),('ongoing',d.cond=='queue_ongoing')]:
        m=np.asarray(mask);out[nm]={'n':int(m.sum()),'rmse0':float(np.sqrt(e0[m].mean())),'rmse1':float(np.sqrt(e1[m].mean())),'rel_mse':float(e1[m].sum()/e0[m].sum()-1)}
    g=d[['panel','date']].copy();g['e0']=e0;g['e1']=e1
    z=g.groupby(['panel','date'])[['e0','e1']].sum();ix=np.random.default_rng(0).integers(0,len(z),(1000,len(z)))
    diff=z.e1.to_numpy()[ix].sum(1)/z.e0.to_numpy()[ix].sum(1)-1
    out['cluster_ci']=np.percentile(diff,[2.5,97.5]).tolist()
    out['perpanel']={p:float(q.e1.sum()/q.e0.sum()-1) for p,q in g.groupby('panel')}
    return out
