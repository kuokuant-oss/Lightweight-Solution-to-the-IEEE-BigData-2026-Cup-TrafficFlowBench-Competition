"""Separate prospective inference phase, never weakens frozen training contract."""
from common import *
JOINT=ME.parent/'INFERENCE_MANIFEST.json'
def outcome_release(job):
    r=json.loads((ME/'OUTCOME_RELEASED.json').read_text());assert r['root_released_outcomes']is True
    assert job in r['authorized_jobs']and r['inference_manifest_sha256']==sha(JOINT)
    joint=json.loads(JOINT.read_text());assert joint['queue_inference_phase_sha256']==sha(ME/'INFERENCE_PHASE.json')
    for path,h in joint['code_hashes'].items():assert sha(path)==h,path
    phase=json.loads((ME/'INFERENCE_PHASE.json').read_text());assert phase['input_manifest_sha256']==sha(ME/'INPUT_MANIFEST.json')
    for path,h in phase['code_hashes'].items():assert sha(path)==h,path
    for path,h in phase['parent_model_hashes'].items():assert sha(path)==h,path
    return r,phase
def parent_paths(cohort):
    assert cohort in ['held','march']
    if cohort=='held':
        n={name:[REP2/f'm_R_V2{name}_s0_seed{s}.txt'for s in range(3)]for name in ['selrec','geomselrec']}
    else:
        d=ROOT/'build/review_20260929_target092/contract_audit/q4_refit'
        n={name:[d/f'm_R_V2{name}_refit_seed{s}.txt'for s in range(3)]for name in ['selrec','geomselrec']}
    a=ROOT/'build/review_20260924_newworld/work/feat/models'
    n['a25']=[a/f'og_{name}.txt'for name in ['V2_noID_noPrior','V2s2','V2s3','V6_V2geom','V6s2','V6s3']]
    return n
