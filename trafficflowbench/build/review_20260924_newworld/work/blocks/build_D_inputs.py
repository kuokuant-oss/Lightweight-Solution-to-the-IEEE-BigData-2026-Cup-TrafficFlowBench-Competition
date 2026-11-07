"""Inputs for candidate D (decisions encoded as prob in {0,1}, thresholded at 0.5 by build_cand).
Onset: C3 site choice on D12_I5_N/D12_I5_S/D7_I405_S, production on D7_I210_W, block decoding t=.7 on single-site panels.
Ongoing: champion-A ensemble probabilities averaged over train-derived blocks per (window, step), >= 0.5."""
import numpy as np, pandas as pd, json, os
# Package edit (2026-11-07): the C3 component was read from a deleted scratchpad; it is now recomputed by
# pipeline/onset_c3.py into work/onset_site/ (where the preserved copy sat).
Z='/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924_newworld/work/onset_site'
ROOT='/Volumes/Crucial X9/trafficflow/trafficflowbench'
B=np.load('my_blocks.npy',allow_pickle=True).item(); BID={p:dict(zip(v['links'],v['bid'])) for p,v in B.items()}
SINGLE=['D7_I10_E','D7_I10_W','D7_I210_E','D7_I405_N']; MULTI_SITE=['D12_I5_N','D12_I5_S','D7_I405_S']
prod=pd.read_parquet(f'{ROOT}/build/strongest_candidate_20260922/onset/production/onset_component.parquet'); prod['link_id']=prod.link_id.astype(str)
c3=pd.read_parquet(f'{Z}/onset_component_submission_C3_onset_sitehazard_minev10_umin0.01_exclD7_I210_W.parquet'); c3['link_id']=c3.link_id.astype(str)
for d in (prod,c3): d['timestamp']=pd.to_datetime(d.timestamp,utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
m=prod.merge(c3[['window_id','timestamp','link_id','queue_pred']],on=['window_id','timestamp','link_id'],suffixes=('','_c3'),validate='one_to_one')
assert len(m)==len(prod)
# sanity: C3 differs from production only on multi-site panels (excl. D7_I210_W)
dif=m[m.queue_pred!=m.queue_pred_c3]; assert dif.panel.isin(MULTI_SITE).all(), dif.panel.unique()
m['bid']=[BID[p][l] for p,l in zip(m.panel,m.link_id)]
dec=m.queue_pred.to_numpy().astype(int).copy()
for wid,ix in m.groupby('window_id',sort=False).indices.items():
    pn=m.panel.iloc[ix[0]]
    if pn in MULTI_SITE: dec[ix]=m.queue_pred_c3.to_numpy()[ix]
    elif pn in SINGLE:
        p=m.prob.to_numpy()[ix]; b=m.bid.to_numpy()[ix]
        s=pd.Series(p).groupby(b).mean(); sel=set(s[s>=0.7].index)|{s.idxmax()}; dec[ix]=np.isin(b,list(sel)).astype(int)
m['dec']=dec
for split,tag in [('validation','realval'),('private','realpriv')]:
    x=m[m.split==split]; x[['window_id','timestamp','link_id']].assign(prob=x.dec.astype(float)).to_parquet(f'D_onset_{tag}.parquet',index=False)
    ch=x[x.dec!=x.queue_pred]; print(split,'onset cells changed vs production',len(ch),'windows',ch.window_id.nunique(),ch.groupby('panel').window_id.nunique().to_dict())
# ongoing block-mean
# Package edit (2026-11-07): skipped when the ongoing A-ensemble probabilities are absent (the onset run has none).
for tag in [t for t in ['realval','realpriv'] if os.path.exists(f'../feat/ogens_{t}_prob.parquet')]:
    e=pd.read_parquet(f'../feat/ogens_{tag}_prob.parquet'); e['panel']=e.window_id.str.extract(r'Q2_(.+)_(?:validation|private)_')[0]
    e['bid']=[BID[p][l] for p,l in zip(e.panel,e.link_id)]
    e['bm']=e.groupby(['window_id','timestamp','bid']).prob.transform('mean')
    e[['window_id','timestamp','link_id']].assign(prob=e.bm).to_parquet(f'D_ongoing_{tag}.parquet',index=False)
    print(tag,'ongoing cells changed vs A',int(((e.prob>=0.5)!=(e.bm>=0.5)).sum()),'windows',e[(e.prob>=0.5)!=(e.bm>=0.5)].window_id.nunique())
