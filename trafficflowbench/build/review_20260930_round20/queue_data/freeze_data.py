"""Bind complete prepared data/code/config before any actual R20 fits."""
from encoder_reader import *
import time
def run():
    assert not(ME/'INPUT_MANIFEST.json').exists();start=time.time()
    pre=json.loads((ME/'PREPARED.json').read_text());plan=json.loads((ME/'ROW_PLAN.json').read_text());spec=json.loads((ME/'TRANSFORM_base196.json').read_text());assert pre['PASS']
    assert pre['rows']=={'fit':2303064,'held':366456}and pre['windows']=={'fit':2691,'held':322}
    reader=EncoderReader();folds=[]
    for fold in range(5):
        tr,te,w,sw,counts=reader.fold(fold);pan=reader.index.iloc[tr].panel.to_numpy()
        masses={p:float((w[tr[pan==p],None]*reader.targets['mask'][tr[pan==p]]).sum())for p in PANELS};assert all(abs(v-1/8)<1e-12 for v in masses.values())
        folds.append(dict(fold=fold,train_centers=len(tr),held_centers=len(te),train_dates=sorted(reader.index.iloc[tr].date.unique()),held_dates=sorted(reader.index.iloc[te].date.unique()),
            train_windows_by_panel=counts,panel_BCE_mass=masses,eligible_training_cells=int(reader.targets['mask'][tr].sum()),switch_weight_mass=float((sw[tr]*reader.targets['switch_mask'][tr]).sum())))
    encoderfiles=[CACHE/'fit_CENTER_INDEX.parquet']+[CACHE/f'fit_{p}_{s}'for p in PANELS for s in ['matched_X.npy','targets.npz']]
    datafiles=sorted(p for p in CACHE.iterdir()if p.is_file()and not p.name.startswith('._'));codefiles=['common.py','e_data.py','public_operator.py','e_architecture.py','encoder_reader.py','prepare.py','transform.py','train_encoder.py','assemble_oof.py','train_gbdt.py','infer_encoder.py','infer_gbdt.py','freeze_data.py']
    tabm_path=ME.parent/'tabm_models/TRAIN_CONFIG.json';tabm=json.loads(tabm_path.read_text())
    root_code={path:h for path,h in tabm['source_sha256'].items()};root_code[str(tabm_path)]=sha(tabm_path)
    for path,h in root_code.items():assert sha(path)==h,path
    params=dict(objective='binary',metric='binary_logloss',num_leaves=127,learning_rate=.05,min_data_in_leaf=100,feature_fraction=.9,bagging_fraction=.8,bagging_freq=1,lambda_l1=0.,lambda_l2=10.,max_bin=127,seed=20260922,num_threads=2,verbose=-1)
    result=dict(PASS=True,fit_end=FIT_END,fit_rows=2303064,held_physical_rows=366456,held_eligible_cells=int(sum(r['eligible_cells']for r in pre['records']if r['cohort']=='held')),
        prepared_sha256=sha(ME/'PREPARED.json'),data_prereg_sha256=sha(ME/'PREREG_DATA.json'),row_plan_sha256=sha(ME/'ROW_PLAN.json'),base_transform_sha256=sha(ME/'TRANSFORM_base196.json'),
        strict_blocks_path=str(BLOCKS),strict_blocks_sha256=sha(BLOCKS),source_hashes=pre['source_hashes'],data_hashes={str(p):sha(p)for p in datafiles},
        encoder_input_hashes={str(p):sha(p)for p in encoderfiles},code_hashes={**{str(ME/n):sha(ME/n)for n in codefiles},**root_code},original_source_copy= json.loads((ME/'COPIED_CODE.json').read_text()),
        features=FS,transformed_base_features=feature_names(False),transformed_semantic_features=feature_names(True),date_folds=folds,
        encoder_config=dict(seed=20260930,epochs=10,batch_centers=1024,optimizer='Adam',learning_rate=.001,weight_decay=.0001,grad_clip_norm=5.,architecture='unchangedR19causalTCN10125params'),
        gbdt_config=dict(params=params,rounds={'0':554,'1':577,'2':681},seeds=[20260922,20260923,20260924]),tabm_config=tabm,tabm_config_sha256=sha(tabm_path),
        semantic_transform_deferred='Fit-date OOF semantics only; TRANSFORM_semantic210 and OOF_ASSEMBLED hashes must bind learner training afterencoderfits, before any held candidate outcome.',
        new_model_updates_so_far=0,causal_audits=pre['causal_audits'],seconds=time.time()-start)
    assert result['held_eligible_cells']==279739;dump(ME/'INPUT_MANIFEST.json',result);print('R20_INPUT_FROZEN',result['seconds'],flush=True)
if __name__=='__main__':run()
