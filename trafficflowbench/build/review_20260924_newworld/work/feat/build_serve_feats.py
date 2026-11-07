"""Serving-path features (exact production code) for a root/split; saves ongoing(all steps) + onset(all steps)."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[n]='2'
import sys, json, importlib.util
from pathlib import Path
import pandas as pd
ROOT=Path('/Volumes/Crucial X9/trafficflow/trafficflowbench'); sys.path.insert(0,str(ROOT))
from src.tfb import release as R, score2, task2infer as TI
root,split,tag=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
R.RELEASE=root
spec=importlib.util.spec_from_file_location('Q',ROOT/'scripts/strongest_20260922_ongoing.py'); Q=importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
fitdates=json.loads((ROOT/'build/review_20260923/ongoing_P1/manifest.json').read_text())['fit_dates']
wi_all=score2.window_index(split); wi_all['date']=pd.to_datetime(wi_all.forecast_origin).dt.strftime('%Y-%m-%d')
parts=[]
for panel in sorted(wi_all.panel.unique()):
    s=TI.features_for_split(panel,split,oblique=True,dispersion=True,clock=True,origin_obs=True,prior_dates=tuple(fitdates))
    wi=wi_all[wi_all.panel.eq(panel)]
    s['date']=s.window_id.map(wi.set_index('window_id').date); Q.identities(s); Q.net_columns(s,wi,split=split)
    s['timestamp']=pd.to_datetime(s.timestamp,utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    parts.append(s); print(panel,len(s),flush=True)
f=pd.concat(parts,ignore_index=True); f.attrs={}
f.to_parquet(f'serve_{tag}.parquet',index=False); print('wrote',len(f))
