"""Serving-side selector-structure features for official windows (legal: window_index origins <= T and observations <= T-5).
prev_rej = 1 if origin T-5 was an ongoing-shaped candidate (cov>=0.7 over its 12-step history, observed queue in that history
and at its last step T-10) and T-5 >= T_prev + 360 (not spacing-blocked): it was then rejected by the persistence rule.
NaN if T-5 was spacing-blocked (unknown). n_rej = consecutive such predecessors (cap 12), counting back while not blocked.
Usage: serve_sel.py <split> [check]   (split train: also compares with the truth-candidate version)"""
import sys, glob, numpy as np, pandas as pd
REL='/Volumes/Crucial X9/trafficflow/tfb_competition_data/kaggle_public'
SP='/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924_ctm/cache'
OUT='/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924_ctm/dense'
split=sys.argv[1]
rows=[]
for p in ['D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S','D12_I5_N','D12_I5_S']:
    z=np.load(f'{SP}/{p}_{split}.npz',allow_pickle=True)
    links=z['links'].astype(str); dates=list(z['dates'].astype(str))
    lk=pd.read_csv(f'{REL}/corridors/{p}/network/links.csv',dtype={'link_id':str}).set_index('link_id').reindex(links)
    vc=0.6*lk.free_speed_kmh.to_numpy()
    V=z['V'].astype(np.float32); E=z['E']
    wi=pd.read_csv(f'{REL}/task2/{p}/{split}/window_index.csv'); wi['T']=pd.to_datetime(wi.forecast_origin,utc=True)
    wi=wi.sort_values('T').reset_index(drop=True)
    if split!='train':
        h=pd.read_parquet(f'{REL}/task2/{p}/{split}/window_history.parquet'); h['link_id']=h.link_id.astype(str)
        li={k:i for i,k in enumerate(links)}; h['ts']=pd.to_datetime(h.timestamp,utc=True)
        # overlay unmasked window history (T-60..T-5) onto the masked-layer cache
        for d,g in h.groupby(h.ts.dt.strftime('%Y-%m-%d')):
            di=dates.index(d); c=(g.ts.dt.hour*12+g.ts.dt.minute//5).to_numpy(); r=g.link_id.map(li).to_numpy(); ok=~np.isnan(r.astype(float))
            V[di,r[ok].astype(int),c[ok]]=g.speed_kmh.to_numpy()[ok]; E[di,r[ok].astype(int),c[ok]]=g.is_score_eligible.to_numpy()[ok]==1
    Qo=(V<=vc[None,:,None])&E&~np.isnan(V)
    for i,w in wi.iterrows():
        if w.condition!='queue_ongoing': continue
        T=w['T']; di=dates.index(T.strftime('%Y-%m-%d')); t=T.hour*12+T.minute//5
        Tprev=wi['T'].iloc[i-1] if i>0 else None
        def shaped(o):  # origin index o: history o-12..o-1
            if o-12<0: return False
            cov=E[di,:,o-12:o].mean(); return bool(cov>=0.7 and Qo[di,:,o-12:o].any() and Qo[di,:,o-1].any() and Qo[di,:,o-2].any())
        def blocked(o):
            To=T.normalize()+pd.Timedelta(minutes=5*o)
            return Tprev is not None and (To-Tprev).total_seconds()/60<360
        if blocked(t-1): pr=np.nan; nr=np.nan
        else:
            k=0
            while k<12 and not blocked(t-1-k) and shaped(t-1-k): k+=1
            pr=float(k>0); nr=float(k)
        rows.append(dict(window_id=w.window_id,panel=p,date=T.strftime('%Y-%m-%d'),t=t,prev_rej=pr,n_rej=nr,gap_min=(T-Tprev).total_seconds()/60 if Tprev is not None else np.nan))
d=pd.DataFrame(rows); d.to_parquet(f'{OUT}/sel2_{split}.parquet',index=False)
print(split,len(d),'prev_rej mean',round(d.prev_rej.mean(),3),'unknown share',round(d.prev_rej.isna().mean(),3))
if split=='train':
    c=pd.read_parquet('/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260924_ctm/rep2/candidates_v2.parquet')
    shp=((c['cov']>=0.7)&c.hor_any.astype(bool)&c.hist_any.astype(bool)&c.last_any.astype(bool)&(c.piou>0.9))
    key=dict(zip(zip(c.panel,c.date,c.t),shp.to_numpy()))
    tr=[float(key.get((r.panel,r.date,r.t-1),False)) for r in d.itertuples()]
    d['prev_rej_truth']=tr
    kn=d.prev_rej.notna()
    print('agreement on known:',round((d.prev_rej[kn]==d.prev_rej_truth[kn]).mean(),3),'n',int(kn.sum()),'truth mean',round(d.prev_rej_truth.mean(),3))
    print(pd.crosstab(d.prev_rej.fillna(-1),d.prev_rej_truth))
