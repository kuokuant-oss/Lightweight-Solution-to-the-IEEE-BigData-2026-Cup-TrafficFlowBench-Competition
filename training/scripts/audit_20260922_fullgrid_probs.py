"""Re-evaluate frozen 145 onset models on all physical links, no training.

Full-grid query features are independent of future eligibility. The retained
E-only subset must reproduce cached features and the original model predictions.
The `_eligible` field is evaluation metadata, never a model input.
"""
# Training record for the onset decoder calibration file. The reproduction does not run this
# program; it reads the shipped calibration file. The program runs from the root of the
# development repository and reads intermediate tables that are not shipped.
#
# Scores the frozen onset hazard heads of `scripts/145_onset_prior_forward.py` (loaded as
#   module `M`) on every physical link, not only the score-eligible ones, at step 6 (T+30)
#   of the replicated onset windows dated from 2030-12-01
#   (`data/interim/score_sprint_20260921/recency_chain_windows.parquet`). For each case the
#   queue prior and the onset-hazard columns are rebuilt from the train dates up to the
#   case's fit end. On eligible rows the features must equal the cached ones exactly and the
#   probabilities the original predictions within 1e-12 (asserted).
# Cases (`M.CASES`): jan and feb (forward checks) and final (fit through 2030-12-31,
#   calibration month 2031-02).
# Shipped output: `build/final_sprint_review_20260922/fullgrid/final_feature_hazard_calib.parquet`
#   (case final, arm feature_hazard): the 0922 head's probability `prob` for every link of
#   the 217 February 2031 onset windows (41,658 rows, with the `_eligible` flag). The
#   reproduction derives the onset decoder thresholds (panel_shrink25) from it.
#   Also writes the jan and feb files, the feature cache
#   `onset_all_links_dec_jan_feb_features.parquet` and `audit.json`.
# Usage: `python scripts/audit_20260922_fullgrid_probs.py [--controls-only]`;
#   `--controls-only` scores the offset and feature control arms of the jan and feb cases
#   instead (`audit_controls.json`).
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import argparse, gc, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tfb import release as R,task2oof as OOF,task2prior as PR
spec=importlib.util.spec_from_file_location('prior145',ROOT/'scripts/145_onset_prior_forward.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
OUT=ROOT/'build/final_sprint_review_20260922/fullgrid'
SOURCE=ROOT/'build/final_sprint_review_20260922/prior'
KEY=M.KEY


def main():
    """Builds the all-link features, checks them against the cache and writes probabilities per case."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cached=pd.read_parquet(SOURCE/'onset_features.parquet')
    cached.attrs={}
    windows=pd.read_parquet(ROOT/'data/interim/score_sprint_20260921/recency_chain_windows.parquet')
    windows=windows[windows.condition.eq('queue_onset') & windows.date.ge('2030-12-01')]
    panels=sorted(cached.panel.unique())
    cache=OUT/'onset_all_links_dec_jan_feb_features.parquet'
    if cache.exists():
        allgrid=pd.read_parquet(cache)
    else:
        parts=[]
        for panel,w in windows.groupby('panel'):
            start=time.time()
            f=OOF.base_features(w[['panel','date','origin','condition']],oblique=True,dispersion=True,
                                 clock=True,origin_obs=True,split='train',include_ineligible=True)
            f=f[f.step.eq(6)].copy();f.attrs={};parts.append(f)
            print('fullgrid features',panel,len(f),round(time.time()-start,2),flush=True)
        allgrid=pd.concat(parts,ignore_index=True);allgrid.attrs={};allgrid.to_parquet(cache,index=False)
    assert allgrid._eligible.dtype==bool
    original=cached[cached.date.ge('2030-12-01')].sort_values(KEY).reset_index(drop=True)
    retained=allgrid[allgrid._eligible].drop(columns='_eligible').sort_values(KEY).reset_index(drop=True)
    assert len(original)==len(retained)
    pd.testing.assert_frame_equal(original[retained.columns],retained,check_exact=True)
    linkmap={}
    for panel in panels:
        for link in sorted(cached.loc[cached.panel.eq(panel),'link_id'].astype(str).unique()):
            linkmap[(panel,link)]=len(linkmap)
    M.identity(cached,panels,linkmap);M.identity(allgrid,panels,linkmap)
    design=json.loads((SOURCE/'design.json').read_text())
    features=design['features']+['onset_hazard','onset_link_rate','onset_hazard_share']
    assert 'label' not in features and '_eligible' not in features
    all_dates=sorted({p.stem.replace('synthetic_mainline_','').replace('_','-') for p in R.truth_files(panels[0])})
    audits=[]
    with threadpool_limits(limits=2):
        cases=[case for case in M.CASES if case[0]!='final'] if args.controls_only else M.CASES
        # Case: (name, fit end date, early-stopping month, calibration month, evaluation month).
        for case,fit_end,esm,calm,evm in cases:
            fitdates=[d for d in all_dates if d<=fit_end]
            hazard=M.Hazard(cached,fitdates)
            wanted=[calm]+([evm] if evm else [])
            query=allgrid[allgrid.date.str[:7].isin(wanted)].reset_index(drop=True).copy()
            old=cached[cached.date.str[:7].isin(wanted)].reset_index(drop=True).copy()
            # Build one panel at a time to avoid retaining all daily matrices.
            query['queue_prior']=np.float32(0);old['queue_prior']=np.float32(0)
            for panel in panels:
                pp=PR.build([panel],PR.PriorSpec(tuple(fitdates)))
                qi=query.panel.eq(panel);oi=old.panel.eq(panel)
                query.loc[qi,'queue_prior']=PR.apply(query.loc[qi],pp)
                old.loc[oi,'queue_prior']=PR.apply(old.loc[oi],pp)
                PR.clear_cache();del pp;gc.collect()
            hazard.apply(query);hazard.apply(old)
            qret=query[query._eligible].sort_values(KEY).reset_index(drop=True)
            oret=old.sort_values(KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(qret[KEY+features],oret[KEY+features],check_exact=True)
            arms=['offset','feature'] if args.controls_only else ['feature_hazard']
            for arm in arms:
                model=lgb.Booster(model_file=str(SOURCE/f'{case}_{arm}.txt'))
                model_features=model.feature_name()
                assert model_features==features if arm=='feature_hazard' else model_features==design['features']
                assert '_eligible' not in model_features and 'label' not in model_features
                p=M.predictions(model,query,model_features,arm)
                query['prob']=p
                for purpose,month in [('calib',calm)]+([('eval',evm)] if evm else []):
                    data=query[query.date.str.startswith(month)][KEY+['_eligible','prob']].copy()
                    ref=pd.read_parquet(SOURCE/f'{case}_{arm}_{purpose}.parquet')
                    matched=data[data._eligible].merge(ref[KEY+['prob']],on=KEY,suffixes=('_full','_original'),validate='one_to_one')
                    diff=float(np.max(np.abs(matched.prob_full-matched.prob_original)))
                    assert len(matched)==len(ref) and diff<1e-12,(case,arm,purpose,len(matched),len(ref),diff)
                    assert not data.duplicated(KEY).any()
                    output=OUT/f'{case}_{arm}_{purpose}.parquet'
                    data.attrs={};data.to_parquet(output,index=False)
                    audits.append(dict(case=case,arm=arm,purpose=purpose,month=month,rows=len(data),eligible_rows=int(data._eligible.sum()),
                                       windows=int(data.window_id.nunique()),max_probability_difference_on_eligible=diff,
                                       retained_feature_exact_match=True,features=len(model_features),
                                       source_model_sha256=hashlib.sha256((SOURCE/f'{case}_{arm}.txt').read_bytes()).hexdigest(),path=str(output)))
                    audit_name='audit_controls.json' if args.controls_only else 'audit.json'
                    (OUT/audit_name).write_text(json.dumps(audits,indent=2)+'\n')
                    print('saved',audits[-1],flush=True)
            del query,old,hazard,model;gc.collect()


if __name__=='__main__':main()
