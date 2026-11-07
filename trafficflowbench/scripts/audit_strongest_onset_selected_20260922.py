"""Validate/freeze the selected decoder; production is a separate explicit mode."""
from __future__ import annotations
import argparse,importlib.util,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('decode_explore',ROOT/'scripts/audit_strongest_onset_decoder_20260922.py')
D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D)
X=D.X;Q=D.Q;TE=D.TE;OUT=D.OUT;HOLD=X.HOLD


def descriptors():
    selection=json.loads((OUT/'final_selection.json').read_text())
    assert selection['selected_recipe']=='panel_shrink25','This replay implements the actually selected unchanged-head decoder'
    return selection


def holdout():
    selection=descriptors();dest=OUT/'held_selected_decoder';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'results.json').exists():raise SystemExit('Selected decoder already evaluated on held set')
    native=HOLD/'native'
    receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':Q.sha(__file__),
      'final_selection_sha256':Q.sha(OUT/'final_selection.json'),'selected_recipe':selection['selected_recipe'],
      'model_sha256':Q.sha(native/'candidate_native_onset.txt'),
      'calibration_input_sha256':Q.sha(native/'candidate_onset_step6_calibration_predictions.parquet'),
      'held_input_sha256':Q.sha(native/'candidate_onset_step6_holdout_predictions.parquet'),
      'new_head_fit':False,'reason':'Selected decoder leaves the validated native75/127 head unchanged',
      'native_reference_manifest_sha256':Q.sha(native/'manifest.json')}
    X.dump(dest/'pre_calibration_receipt.json',receipt)
    ca=pd.read_parquet(native/'candidate_onset_step6_calibration_predictions.parquet')
    assert ca.date.between('2030-12-08','2031-01-04').all()
    global_t,grid=Q.calibrate(ca,ca.prob.to_numpy(),'queue_onset',D.M.GRID)
    desc=D.calibrate_decoder('panel_shrink25',ca,ca.prob.to_numpy(),global_t)
    desc.update({'global_threshold':global_t,'global_calibration_grid':grid,
                 'selected_at_utc':datetime.now(timezone.utc).isoformat()})
    X.dump(dest/'calibration.json',desc)
    ev=pd.read_parquet(native/'candidate_onset_step6_holdout_predictions.parquet')
    pred=D.decode('panel_shrink25',ev,ev.prob.to_numpy(),desc)
    ev['pred']=pred.astype('int8');ev.attrs={};ev.to_parquet(dest/'onset_step6_predictions.parquet',index=False)
    champion=pd.read_parquet(HOLD/'champion_holdout_predictions.parquet')
    prior=pd.read_parquet(native/'candidate_holdout_predictions.parquet')
    candidate=prior.copy();rows=candidate.condition.eq('queue_onset') & candidate.step.eq(6)
    own=ev.set_index(Q.STABLE_KEY);idx=pd.MultiIndex.from_frame(candidate.loc[rows,Q.STABLE_KEY])
    assert len(idx)==len(own) and set(idx)==set(own.index)
    for col in ['label','_eligible','prob']:assert np.array_equal(candidate.loc[rows,col],own.loc[idx,col])
    candidate.loc[rows,'pred']=own.loc[idx,'pred'].to_numpy()
    pd.testing.assert_frame_equal(candidate.loc[~rows],prior.loc[~rows],check_exact=True)
    candidate.attrs={};candidate.to_parquet(dest/'candidate_holdout_predictions.parquet',index=False)
    frames={n:Q.frame(d,d.pred.to_numpy()) for n,d in [('champion',champion),('existing_native',prior),('candidate',candidate)]}
    paired=frames['champion'][['window_id','panel','condition','family_id','date']].copy()
    for n,f in frames.items():
        paired=paired.merge(f[['window_id','IoU','inter','union']].rename(columns={c:n+'_'+c for c in ['IoU','inter','union']}),on='window_id',validate='one_to_one')
    paired['delta_IoU']=paired.candidate_IoU-paired.champion_IoU
    paired['delta_vs_native_IoU']=paired.candidate_IoU-paired.existing_native_IoU
    paired.to_parquet(dest/'paired_main_all6_eligible_windows.parquet',index=False)
    scores={n:{'S_queue':TE.s_queue(f),'subgroups':TE.subgroups(f)} for n,f in frames.items()}
    onset=paired[paired.condition.eq('queue_onset')]
    results={'selected_recipe':selection['selected_recipe'],'scores':scores,
      'delta_S_queue_vs_champion':scores['candidate']['S_queue']-scores['champion']['S_queue'],
      'delta_S_queue_vs_existing_native':scores['candidate']['S_queue']-scores['existing_native']['S_queue'],
      'changed_fullgrid_cells_vs_native':int((candidate.pred!=prior.pred).sum()),
      'changed_eligible_cells_vs_native':int(((candidate.pred!=prior.pred)&candidate._eligible).sum()),
      'onset_windows_improved_vs_native':int(onset.delta_vs_native_IoU.gt(1e-12).sum()),
      'onset_windows_worsened_vs_native':int(onset.delta_vs_native_IoU.lt(-1e-12).sum()),
      'per_panel_onset_delta_vs_native':onset.groupby('panel').delta_vs_native_IoU.mean().to_dict()}
    X.dump(dest/'results.json',results);print(json.dumps(results,indent=2),flush=True)


def production():
    selection=descriptors();dest=OUT/'production';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():raise SystemExit('Production component already frozen')
    source=ROOT/'build/final_sprint_review_20260922'
    model=source/'prior/final_feature_hazard.txt'
    reference_manifest=json.loads((source/'prior/final_feature_hazard_manifest.json').read_text())
    assert reference_manifest['prior_dates'][-1]=='2030-12-31' and reference_manifest['refit_after_calibration'] is False
    ca=pd.read_parquet(source/'fullgrid/final_feature_hazard_calib.parquet')
    assert ca.date.between('2031-02-01','2031-02-28').all() and ca.date.nunique()==28
    global_t,grid=Q.calibrate(ca,ca.prob.to_numpy(),'queue_onset',D.M.GRID)
    desc=D.calibrate_decoder('panel_shrink25',ca,ca.prob.to_numpy(),global_t)
    desc.update({'global_threshold':global_t,'global_calibration_grid':grid})
    X.dump(dest/'calibration.json',desc)
    s=pd.read_parquet(source/'prior/final_feature_hazard_serving.parquet')
    s['condition']='queue_onset';assert s.step.eq(6).all()
    s['queue_pred']=D.decode('panel_shrink25',s,s.prob.to_numpy(),desc).astype('int8')
    assert not s.duplicated(['window_id','timestamp','link_id']).any()
    assert s.groupby('window_id').queue_pred.sum().ge(1).all()
    s.attrs={};s.to_parquet(dest/'onset_component.parquet',index=False)
    # Freeze an exact copy of the unchanged, already-trained production head.
    (dest/'reference75_head.txt').write_bytes(model.read_bytes())
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':Q.sha(__file__),
      'selected_recipe':selection['selected_recipe'],'selection_sha256':Q.sha(OUT/'final_selection.json'),
      'model_sha256':Q.sha(model),'features':reference_manifest['features'],'model_training':reference_manifest,
      'model_policy':'Reuse exact unchanged production75/127 head; newly select decoder only on February calibration',
      'calibration_source_sha256':Q.sha(source/'fullgrid/final_feature_hazard_calib.parquet'),
      'serving_probability_source_sha256':Q.sha(source/'prior/final_feature_hazard_serving.parquet'),
      'calibration':desc,'coverage':'All official validation/private onset full physical links at step6 only; builder must set onset steps1-5 to0',
      'rows':len(s),'windows':s.window_id.nunique(),'splits':s.groupby('split').size().to_dict(),
      'output_sha256':Q.sha(dest/'onset_component.parquet'),'key':['window_id','timestamp','link_id'],
      'queue_pred_positive':int(s.queue_pred.sum())}
    X.dump(dest/'manifest.json',manifest);print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['holdout','production']);a=p.parse_args()
    holdout() if a.mode=='holdout' else production()
