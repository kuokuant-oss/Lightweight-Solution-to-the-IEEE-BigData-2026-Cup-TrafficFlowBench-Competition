"""Rebuild RLB/k* on the EXACT February HMP50 window/mask world.

Old RLB g3x and HMP50 v2 caches have different holes. Never join their
predictions by panel/date/cell without rebuilding the legal feature inputs.
"""
import os,sys,json,gc,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='3'
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,lightgbm as lgb
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RL=ROOT/'build/review_20260924_ctm/rl';OUT=HERE/'density/matched'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(RL))
import rl_eval as R,rleval as E,rlfeat as RF
from src.tfb import physics,predict1 as P1,baselines as B,task1stack as TS
G2=ROOT/'build/review_20260924_ctm/g2v2';_cache={}
def load(panel,month='2031_02'):
    if month!='2031_02' or panel not in E.PANELS:return E.load(panel,month)
    if panel in _cache:return _cache[panel]
    _cache.clear();old,om=E.load(panel,month);cc=np.load(G2/f'c0_cache/{panel}_2031_02_v2.npz',allow_pickle=True);meta=json.loads(str(cc['meta']))
    assert om['links']==meta['links'];assert om['dates']==meta['dates'];assert om['regimes']==meta['regimes']
    z={c:old[c]for c in ['ln','keep','milepost','par_length_km','par_free_speed_kmh','par_capacity_vph','par_lanes','par_critical_density','par_k_jam']}
    rows=pd.read_parquet(G2/f'res/{panel}_2031_02_v2_cmp_cells.parquet',columns=['date','origin','li','step','v2F_v','v2F_ql'])
    for di,date in enumerate(meta['dates']):
        k=f'd{di}';z[k+'_v1']=cc[k+'_v0'].copy();z[k+'_q1']=cc[k+'_q0'].copy();c=rows[rows.date==date];li=c.li.to_numpy(int);ti=(c.origin+1+c.step).to_numpy(int)
        z[k+'_v1'][li,ti]=c.v2F_v;z[k+'_q1'][li,ti]=c.v2F_ql.to_numpy()*z['ln'][li]
        for dest,src in [('vt','TV'),('qt','TQ'),('ok','E'),('rt','rt'),('gm','gm'),('rv','rv'),('obsv','V'),('obsq','Q'),('obso','O')]:z[k+'_'+dest]=cc[k+'_'+src]
        assert np.array_equal(z['ln'],cc[k+'_ln'])
    cc.close();_cache[panel]=(z,meta);return z,meta
def build():
    OUT.mkdir(parents=True,exist_ok=True);blocks=np.load(ROOT/'build/review_20260924_newworld/work/blocks/my_blocks.npy',allow_pickle=True).item();t=time.time()
    aux=lgb.Booster(model_file=str(ROOT/'build/final_sprint_review_20260922/experiment_audit/state_recency/aux_density_weightedk/k_l1_weightedL_600.txt'))
    for pi,panel in enumerate(E.PANELS10):
        if (OUT/f'{panel}_parent.npz').exists() and (OUT/f'{panel}_features.parquet').exists():print('resume existing',panel,flush=True);continue
        z,meta=load(panel);ln=np.maximum(z['ln'],1);vf=z['par_free_speed_kmh'];length=z['par_length_km'];parts=[];grids=[]
        bid=blocks[panel]['bid']if panel in blocks else np.full(len(ln),-1)
        if panel in blocks:assert list(blocks[panel]['links'])==meta['links']
        for di,date in enumerate(meta['dates']):
            k=f'd{di}';tg=z[k+'_rt']&z[k+'_ok']&~z[k+'_gm'];v=z[k+'_v1'].astype(float);q=z[k+'_q1'].astype(float)/ln[:,None]
            ov=z[k+'_obsv'].astype(float);oq=z[k+'_obsq'].astype(float)/ln[:,None];oo=z[k+'_obso'].astype(float)
            c=RF.day_features(v,q,ov,oq,oo,vf,ln,length,tg);c.update(RF.block_features(v,ov,vf,tg,bid));li,ti=np.where(tg)
            df=pd.DataFrame(c);df['panel_i']=np.int8(pi);df['day']=np.int8(di);df['li']=li.astype(np.int16);df['ci']=ti.astype(np.int16);df['vt']=z[k+'_vt'][tg];df['qt']=z[k+'_qt'][tg]/ln[li];parts.append(df)
            if panel in R.KSTAR:grids.append(SimpleNamespace(times=pd.date_range(date,periods=288,freq='5min',tz='UTC'),milepost=z['milepost'],target=tg,value={'speed_kmh':ov,'flow_vph':oq*ln[:,None],'occupancy':oo}))
        d=pd.concat(parts,ignore_index=True);d.to_parquet(OUT/f'{panel}_features.parquet',index=False);model='RLB'if pi<8 else'RL';md=RL/'models'/f'{model}_fit2031_01';fs=json.loads((md/'feats.json').read_text());pred={}
        for nm,base in [('y_v','p_v'),('y_q','p_q')]:
            m=lgb.Booster(model_file=str(md/f'{nm}.txt'));pred[nm]=m.predict(d[fs],num_threads=3);del m
        print('matched RLB features',panel,len(d),round(time.time()-t),flush=True)
        if panel in R.KSTAR:
            W=P1.load_weights();hist=P1.historical_mean(panel,P1.dev_train_files(panel));par=physics.link_params(panel,meta['links'])
            lrtc={f:B.lrtc_tnn(np.stack([np.nan_to_num(g.value[f],nan=0)for g in grids],axis=2),rho=P1.LRTC_KNOBS[f][0],theta=P1.LRTC_KNOBS[f][1])for f in P1.FIELDS}
            ks=[]
            for di,g in enumerate(grids):
                x=TS.feature_table(g,di,lrtc,hist,W,par,mask=g.target);assert list(x)==aux.feature_name();ks.append(np.maximum(aux.predict(x,num_threads=3),0))
            pred['khat']=np.concatenate(ks);assert len(pred['khat'])==len(d)
            del lrtc,hist,grids
        np.savez(OUT/f'{panel}_parent.npz',**pred);print('matched parent complete',panel,round(time.time()-t),flush=True);del d,parts;gc.collect()
if __name__=='__main__':build()
