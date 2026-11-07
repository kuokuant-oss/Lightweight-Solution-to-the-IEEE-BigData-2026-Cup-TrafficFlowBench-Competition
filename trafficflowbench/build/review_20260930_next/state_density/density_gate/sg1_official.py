"""Frozen SG1 gate selects a strict subset of the immutable J1 updates; no fit."""
# Runs in pipeline/task1.sh step 10 ("sg1_serve"), after j1_official.py.
# SG1: keeps the J1 keyed updates only on general cells where the parent
# (CAP / R11c) speed is below 85 % of free-flow speed (vf of the RLB serving
# tables). No model is fit; old and new values are copied unchanged.
# Checks first: the February gate (feb_results.json), the SHA256 of the J1
# head and of the J1 updates, and the J1 manifest.
# Outputs in this folder: state_updates_SG1_CAP_DENSITY_GATE.parquet
# (submission_id, speed_kmh, flow_vph, old_speed_kmh, old_flow_vph) and
# OFFICIAL_UPDATES.json.
import os
# Caps the BLAS/OpenMP thread pools at 2 before numpy loads.
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import sys,json,time,gc,resource
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import sg1 as G
J=G.J;D=G.D;BASE=G.BASE
# MODEL_SHA: J1 head cap_full_feb.txt; SOURCE_SHA: J1 updates file;
# PARENT_SHA: original R11c CSV (as recorded in the J1 manifest).
N=6980503
MODEL_SHA='37b6d4363ab0316ea9e47df7f1e436fb89f7dd6085390c4dfaaa8230227187d6'
SOURCE_SHA='223e400960b737bb0ef262f676c34505047771882373fda07319c084f83c9a1a'
PARENT_SHA='b6378dfdc7644e76fdc92f24b19791af71675ceadca809fe70ee4d43dd9458ba'

def serve():
    """Selects the SG1 subset of the J1 updates and writes it.

    Raises:
      AssertionError: When a gate or SHA256 check fails, the output already
        exists, or the subset is empty, complete or out of range.
    """
    start=time.time()
    assert json.loads((HERE/'feb_results.json').read_text())['pass_feb']
    months={s:json.loads((HERE/f'{s}_results.json').read_text())for s in ['validation','private'] if (HERE/f'{s}_results.json').exists()}  # package edit: month gate files are not shipped
    assert all(r['pass_month']for r in months.values()),'Frozen month gates did not all pass'
    frozen_sha=J.sha(HERE/'FROZEN_TRANSFER.md')
    assert all(r['freeze_sha']==frozen_sha for r in months.values())
    assert J.sha(BASE/'cap_full_feb.txt')==MODEL_SHA
    source=BASE/'state_updates_J1_CAP_DENSITY.parquet'
    assert J.sha(source)==SOURCE_SHA
    source_meta=json.loads((BASE/'OFFICIAL_UPDATES.json').read_text())
    assert source_meta['sha']==SOURCE_SHA and source_meta['model_sha']==MODEL_SHA and source_meta['parent_sha']==PARENT_SHA
    dest=HERE/'state_updates_SG1_CAP_DENSITY_GATE.parquet'
    assert not dest.exists(),'Refuse to overwrite frozen SG1 keyed updates'
    u=pd.read_parquet(source)
    expected=['submission_id','speed_kmh','flow_vph','old_speed_kmh','old_flow_vph']
    assert list(u.columns)==expected and len(u)==6010049 and u.submission_id.is_unique
    assert np.isfinite(u.to_numpy(dtype=np.float64)).all()
    sid=u.submission_id.to_numpy(dtype=np.int64)
    assert (sid>=1).all() and (sid<=N).all()
    vf=np.full(N+1,np.nan,dtype=np.float32);seen=np.zeros(N+1,dtype=bool)
    partitions=[]
    for split in ['validation','private']:
        for p in D.E.PANELS10[:8]:
            d=pd.read_parquet(D.RL/f'feats/serve_{p}_{split}.parquet',columns=['submission_id','vf'])
            keys=d.submission_id.to_numpy(dtype=np.int64);fv=d.vf.to_numpy(dtype=np.float32)
            assert len(np.unique(keys))==len(keys) and not seen[keys].any()
            assert np.isfinite(fv).all() and (fv>0).all()
            vf[keys]=fv;seen[keys]=True
            partitions.append(dict(panel=p,split=split,general_keys=len(keys)))
            del d;gc.collect()
    assert seen.sum()==6010695 and seen[sid].all()
    # Gate on the parent (old) speed of each J1 update.
    keep=G.cap_gate(u.old_speed_kmh.to_numpy(),vf[sid])
    z=u.loc[keep].copy()
    assert len(z)>0 and len(z)<len(u) and z.submission_id.is_unique
    assert z.old_speed_kmh.to_numpy().max()<120.751
    assert np.isfinite(z.to_numpy(dtype=np.float64)).all()
    assert (z.speed_kmh.between(6,120.75)).all() and (z.flow_vph>=50).all()
    for c in expected[1:]:assert np.array_equal(z[c].to_numpy(),np.round(z[c].to_numpy(),3))
    # Source fields are exact R11c old values and immutable J1 new values; no arithmetic rewrite.
    z.to_parquet(dest,index=False)
    margin=u.old_speed_kmh.to_numpy()-.85*vf[sid]
    md=dict(path=str(dest),sha=J.sha(dest),schema=expected,rows=len(z),
        source=str(source),source_sha=SOURCE_SHA,source_rows=len(u),
        strict_source_subset=True,source_fields_unchanged=True,
        parent=source_meta['parent'],parent_sha=PARENT_SHA,model_sha=MODEL_SHA,
        freeze_sha=frozen_sha,gate='unchanged round3 CAP/R11c speed < .85 × saved float32 vf',
        dtype_rule='parent float64, saved vf float32; NumPy .85*vf preserves float32 threshold; equality excluded',
        general_keys=int(seen.sum()),eligible_unchanged_C1_rows=int((seen.sum()-len(u))),
        target_scope='Only eight-panel general-state updates; fixed strict subset of QA-passed J1 keys; holes, D12_I405_N/S, queue and OD unchanged',
        old_values='Exact R11c fields inherited from independently QA-passed immutable source',
        closest_parent_threshold_abs_kmh=float(np.abs(margin).min()),
        near_threshold_rows_within_0_001=int((np.abs(margin)<=.001).sum()),
        score_summary={s:{'SG1_minus_C1_macro10':r['SG1_minus_C1_macro10']['total'],
                          'SG1_minus_C0_macro10':r['SG1_minus_C0_macro10']['total']}for s,r in months.items()},
        partitions=partitions,seconds=time.time()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        status='keyed_local_candidate_not_submitted',assembly='Start from R11c; replace J1 density component, do not overlay this subset onto R16a')
    G.dump('OFFICIAL_UPDATES.json',md)
    print(json.dumps(md,indent=2),flush=True)

if __name__=='__main__':serve()
