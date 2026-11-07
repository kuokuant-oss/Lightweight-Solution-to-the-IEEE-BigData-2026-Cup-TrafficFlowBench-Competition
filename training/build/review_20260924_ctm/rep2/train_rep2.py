"""Pilot: ongoing V2 (no IDs, no queue_prior) trained on replicated windows vs + dense origins, +/- selector feature.
In-world: fit <= 2030-12-31, ES Jan 2031 (replicated only), calibrate/score Feb 2031 (replicated only)."""
# Training record for the rep2 ongoing models. The reproduction does not run this program;
# it loads the shipped models. The program reads window tables that are not shipped, and its
# paths resolve from the hard-coded development root (`ROOT`, mapped by the path layer).
#
# Trains LightGBM binary models of P(link queued) at each of the 6 steps T+5 .. T+30 of an
#   ongoing Task 2 window. Base inputs V2: the 73 inputs of
#   `build/review_20260923/ongoing_P1/manifest.json` without link code, corridor code and
#   queue prior. A variant name adds groups: `geom` (20 queue-geometry inputs), `sel` (2
#   selector inputs: whether the scan origin just before T was rejected as unchanged, and how
#   many consecutive ones were), `rec` (3 recency inputs: how often the link was queued at
#   this time of day on up to 5 earlier days of the month). A variant starting with "D" also
#   fits on the dense (non-replicated) origins.
# Data: `rep2_s<stride>_<corridor>.parquet` (replicated ongoing windows of the 8 Task 2
#   corridors on train days; label = link queued at the step) joined with
#   `ruler2/rec_train_queue_ongoing_s<stride>.parquet`. Fit <= 2030-12-31; early stopping on
#   January 2031 (patience 80, at most 1,500 rounds); February 2031 only to calibrate a
#   decode threshold. 25% of windows (rng seed 123) get their selector inputs blanked.
# Settings: `NEW_PARAMS` of `scripts/strongest_20260922_ongoing.py` (binary, 127 leaves,
#   learning rate .05, min_data_in_leaf 100, feature fraction .9, bagging .8 every iteration,
#   lambda_l2 10, max_bin 127), seed 20260922 + seed, threads from the environment variable
#   NT (default 4). Row weight = 1 / (rows of its window x windows of its corridor),
#   normalised to mean 1, so every corridor carries the same total weight.
# Usage: `train_rep2.py <stride> <variant,...> [seed]`. Output: `m_<variant>_s<stride>_seed<seed>.txt`
#   and `res_<stride>_<variants>_seed<seed>.json`.
# Shipped: `m_R_V2selrec_s0_seed{0,1,2}.txt` and `m_R_V2geomselrec_s0_seed{0,1,2}.txt` under
#   `trafficflowbench/build/review_20260924_ctm/rep2/`, from stride 0 with the variants
#   R_V2sel,R_V2geomsel,R_V2,R_V2geom,R_V2selrec,R_V2geomselrec and seeds 0, 1, 2 (the result
#   file names record these arguments; the model headers record 4 threads). Fit 2,337 windows,
#   early stopping 354, calibration 322.
import os
NT=os.environ.get('NT','4')
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[n]=NT
import sys, json, glob, importlib.util, gc
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT=Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('Q',ROOT/'scripts/strongest_20260922_ongoing.py'); Q=importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
HERE=Path(__file__).resolve().parent
man=json.loads((ROOT/'build/review_20260923/ongoing_P1/manifest.json').read_text())
FS0=man['features']; V2=[f for f in FS0 if f not in ('link_code','panel_code','queue_prior')]
GEOM=['g_gap_km','g_gap_links','g_tail_vel','g_rh_w','g_eta_rh','g_eta_vel','g_block_len_km','g_block_links','g_head_age','g_in_block','g_pos_in_block_km','g_clear_eta_rh','g_n_blocks','g_link_age','g_q_up','g_q_in','g_tail_age','g_eta_rh_m_step','g_eta_vel_m_step','g_clear_m_step']
SEL=['prev_rej','n_rej']
REC=['rec_s','rec_any','rec_n']
PARAMS=dict(Q.NEW_PARAMS,num_threads=int(NT))
stride=sys.argv[1]; variants=sys.argv[2].split(','); seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
def weights(d):
    """Returns row weights 1 / (rows of the window x windows of the corridor), mean 1."""
    nc=d.groupby('window_id').label.transform('size').to_numpy(float)
    pwin=d[['panel','window_id']].drop_duplicates().groupby('panel').size()
    w=1/(nc*d.panel.map(pwin).to_numpy()); return w/w.mean()
d=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(HERE/f'rep2_s{stride}_*.parquet')))],ignore_index=True)
rp=ROOT/'build/review_20260924_ctm/ruler2'/f'rec_train_queue_ongoing_s{stride}.parquet'
if rp.exists():
    r=pd.read_parquet(rp); r['link_id']=r.link_id.astype(str); d['link_id']=d.link_id.astype(str); d=d.merge(r,on=['window_id','link_id','step'],how='left'); print('rec coverage',round(float(d.rec_s.notna().mean()),3),flush=True)
pass
d['period']=np.where(d.date<='2030-12-31','fit',np.where(d.date<='2031-01-31','es','cal'))
# Selector dropout: 25% of windows get NaN selector inputs, so the model also works when
# the selector history is unknown.
rng=np.random.default_rng(123)
wids=d.window_id.unique(); maskw=set(wids[rng.random(len(wids))<0.25])
mk=d.window_id.isin(maskw).to_numpy()
d.loc[mk,'prev_rej']=np.nan; d.loc[mk,'n_rej']=np.nan
d['prev_rej']=d.prev_rej.astype('float32'); d['n_rej']=d.n_rej.astype('float32')
print('rows',len(d),'windows',d.window_id.nunique(),d.groupby(['period','is_rep']).window_id.nunique().to_dict(),flush=True)
res=[]
for v in variants:
    fs=V2+(GEOM if 'geom' in v else [])+(SEL if 'sel' in v else [])+(REC if 'rec' in v else [])
    use_dense=v.startswith('D')
    # Fit and stopping rows: eligible cells; calibration: every row of February replicated windows.
    tr=d[d.period.eq('fit')&d._eligible&(d.is_rep.eq(1)|use_dense)]
    es=d[d.period.eq('es')&d._eligible&d.is_rep.eq(1)]; ca=d[d.period.eq('cal')&d.is_rep.eq(1)].reset_index(drop=True)
    P=dict(PARAMS,seed=20260922+seed)
    dt=lgb.Dataset(tr[fs],tr.label,weight=weights(tr)); dv=lgb.Dataset(es[fs],es.label,weight=weights(es),reference=dt)
    m=lgb.train(P,dt,num_boost_round=1500,valid_sets=[dv],callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])
    cp=m.predict(ca[fs],num_threads=int(NT)); thr,grid,pans=Q.calibrate(ca,cp)
    g={r['threshold']:r['score'] for r in grid}
    r=dict(variant=v,seed=seed,trees=m.current_iteration(),train_windows=int(tr.window_id.nunique()),thr=thr,inworld_cal=round(max(g.values()),4),inworld_thr05=round(g.get(0.5,np.nan),4),best_val_loss=round(m.best_score['valid_0']['binary_logloss'],5))
    print(json.dumps(r),flush=True); res.append(r)
    m.save_model(str(HERE/f'm_{v}_s{stride}_seed{seed}.txt'))
    del dt,dv,m; gc.collect()
json.dump(res,open(HERE/f'res_{stride}_{"_".join(variants)}_seed{seed}.json','w'),indent=1)
