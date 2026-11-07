"""R20 shared data contract. Import loads metadata, never fits or scores."""
import os,sys,json,hashlib,importlib.util
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
ME=Path(__file__).resolve().parent;ROOT=ME.parents[2];CACHE=ME/'cache';CT=ROOT/'build/review_20260924_ctm';REP2=CT/'rep2'
R19=ROOT/'build/review_20260930_round19/queue_models'
FS=json.loads((R19/'PREPARED_B.json').read_text())['features'];assert len(FS)==98
PANELS=['D12_I5_N','D12_I5_S','D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S']
SEMANTICS=[f'TCN_p_queue_step{s}'for s in range(1,7)]+['TCN_first_switch_capped_step']
KEY=['window_id','link_id','step'];FIT_END='2031-01-31';HELD_START='2031-02-01';HELD_END='2031-02-28'
BLOCKS=ME.parent/'evaluation_audit/strict_blocks_through_20310131.npy'
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def feature_names(sem=False):
    f=FS+(SEMANTICS if sem else[])
    return [f'z_{i}_{name}'for i,name in enumerate(f)]+[f'missing_{i}_{name}'for i,name in enumerate(f)]
def require_release(job):
    r=json.loads((ME/'MAIN_SLOT_RELEASED.json').read_text());assert r['root_released_main_slot']is True and job in r['authorized_jobs']
    pre=json.loads((ME/'INPUT_MANIFEST.json').read_text());assert sha(ME/'INPUT_MANIFEST.json')==r['input_manifest_sha256']
    for path,h in pre['code_hashes'].items():assert sha(path)==h,path
    return r,pre
