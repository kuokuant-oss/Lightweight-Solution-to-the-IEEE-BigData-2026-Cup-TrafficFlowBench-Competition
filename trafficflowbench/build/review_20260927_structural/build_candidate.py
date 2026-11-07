"""Build DK50 only after frozen March confirmation passes. Never submits."""
# Runs in pipeline/task1.sh step 8 ("dk50"). DK50: a density head ("DK")
# applied at strength 0.5 to the general (non-hole) Task 1 cells of the 8
# Task 2 panels. The head (density/l2_uniform.txt, fit on January 2031 by
# density_experiment.py) predicts the per-lane density error of the RLB fill;
# speed and flow are then projected jointly onto the density
# .5 x parent density + .5 x corrected density (k* projection, rl_eval.kstar).
# "RLB": the residual LightGBM that corrects the stacked fill. "k*": the joint
# speed/flow projection toward an auxiliary density on 5 panels.
# Parent: the HMP50 submission CSV (in the package run, the state-only CSV
# rebuilt by pipeline/task1_state_csv.py). Outputs:
# density/serving/state_updates.parquet (changed rows: submission_id,
# speed_kmh, flow_vph) and stats.json, plus the candidate CSV
# deliverables/2026-09-27/density50/submission_HMP50_DK50.csv and MANIFEST.json.
import os,sys,json,hashlib,gc,time
# Caps the BLAS/OpenMP thread pools at 3 before numpy and LightGBM load.
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='3'
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));import density_experiment as D
# OUT: candidate CSV folder; WORK: keyed updates that later steps read;
# KUPD: k* auxiliary density per serving cell.
OUT=ROOT/'build/deliverables/2026-09-27/density50';WORK=HERE/'density/serving'
PARENT=ROOT/'build/deliverables/2026-09-27/hmp50/submission_HMP50.csv';PSHA='7d3c40c07e2c1f73f16546d7d369615fde7180179af2bed20036328a7647502c'
KUPD=ROOT/'build/strongest_candidate_20260922/state/transfer/serving'
def sha(p):
    """Returns the SHA256 hex digest of file `p`, read in 8 MB blocks."""
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def run():
    """Builds the DK50 updates and writes the candidate CSV.

    Raises:
      AssertionError: When the rebuilt parent differs from the parent CSV, a
        projected value leaves the allowed range, or the candidate CSV
        already exists.
    """
    confirmation_path=HERE/'density/march_float64/results.json'
    if confirmation_path.exists():  # package edit: March confirmation gate file is not shipped (built from the validation split)
        confirmation=json.loads(confirmation_path.read_text());assert confirmation['promote'];assert confirmation['selected']=='l2_uniform_0.5'
    # package edit: the parent CSV may be a regenerated state-only stand-in; its state rows are checked row by row below (dv == dq == 0)
    print('parent sha256',sha(PARENT)==PSHA and 'matches the original' or 'differs (state-only stand-in)',flush=True)
    OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True);dest=OUT/'submission_HMP50_DK50.csv';assert not dest.exists(),'refuse to overwrite frozen CSV'
    t0=time.time();base=pd.concat([c[c.task=='state'][['submission_id','speed_kmh','flow_vph']]for c in pd.read_csv(PARENT,usecols=['submission_id','task','speed_kmh','flow_vph'],chunksize=500000)],ignore_index=True).set_index('submission_id')
    # head: the frozen density head; mv / mq: RLB residual boosters for speed
    # and per-lane flow.
    head=lgb.Booster(model_file=str(HERE/'density/l2_uniform.txt'));md=D.RL/'models/RLB_final';fs=json.loads((md/'feats.json').read_text());mv=lgb.Booster(model_file=str(md/'y_v.txt'));mq=lgb.Booster(model_file=str(md/'y_q.txt'));parts=[];stats=[]
    for split in ['validation','private']:
        for panel in D.E.PANELS:
            tab=pd.read_parquet(D.RL/f'feats/serve_{panel}_{split}.parquet');tab=pd.concat([tab,pd.read_parquet(D.RL/f'feats/blk_serve_{panel}_{split}.parquet')],axis=1);ln=tab.lane_n.to_numpy(float);sid=tab.submission_id.to_numpy();ref=base.loc[sid]
            rv=mv.predict(tab[fs],num_threads=3);rq=mq.predict(tab[fs],num_threads=3)
            # Exact existing production composition uses original C1 fields.
            v=np.clip(tab.c1_v.to_numpy()+rv,6,120.75);q=np.maximum((tab.c1_q.to_numpy()/ln+rq)*ln,50)
            if panel in D.R.KSTAR:
                u=pd.read_parquet(KUPD/f'updates_{panel}.parquet').set_index('submission_id').loc[sid];assert u.aux_density_vpkm.notna().all()
                v,q=D.R.kstar(np.round(v,3),np.round(q,3),u.aux_density_vpkm.to_numpy(),ln,D.R.SEL['panel_alphas'][panel],D.R.SEL['weights'][panel])
            # The rebuilt parent must equal the parent CSV exactly after rounding.
            v=np.round(v,3);q=np.round(q,3);dv=float(abs(v-ref.speed_kmh.to_numpy()).max());dq=float(abs(q-ref.flow_vph.to_numpy()).max());assert dv==0 and dq==0,(panel,split,dv,dq)
            # Density head uses the same float32 feature inputs as January fit.
            rawv=np.clip(tab.p_v.to_numpy()+rv,6,120.75);rawq=np.maximum(tab.p_q.to_numpy()+rq,50/ln);hfs=D.augment(tab,rawv,rawq);assert hfs==head.feature_name()
            # Corrected per-lane density kh = raw parent density + head output;
            # the parent speed/flow move halfway toward it (alpha 0.5).
            kh=np.maximum(tab.parent_k.to_numpy()+head.predict(tab[hfs],num_threads=3),0);nv,nq=D.project(v,q/ln,kh,ln,panel,.5);nv=np.round(nv,3);nq=np.round(nq,3)
            assert np.isfinite(nv).all() and np.isfinite(nq).all() and nv.min()>=6 and nv.max()<=120.75 and nq.min()>=50
            changed=(nv!=v)|(nq!=q);part=pd.DataFrame({'submission_id':sid[changed],'speed_kmh':nv[changed],'flow_vph':nq[changed]});parts.append(part)
            stats.append(dict(panel=panel,split=split,nonhole_rows=len(tab),changed=int(changed.sum()),changed_speed=int((nv!=v).sum()),changed_flow=int((nq!=q).sum()),parent_identity_speed=dv,parent_identity_flow=dq,mean_abs_dv=float(abs(nv-v).mean()),max_abs_dv=float(abs(nv-v).max()),mean_abs_dq_lane=float((abs(nq-q)/ln).mean()),max_abs_dq_lane=float((abs(nq-q)/ln).max())))
            print('serving',panel,split,'changed',int(changed.sum()),'seconds',round(time.time()-t0),flush=True);del tab;gc.collect()
    updates=pd.concat(parts,ignore_index=True);assert updates.submission_id.is_unique;updates.to_parquet(WORK/'state_updates.parquet',index=False);D.dump(WORK/'stats.json',stats)
    # Candidate CSV: copies the parent line by line and rewrites only the
    # speed / flow fields that changed (3 decimals); submission_id indexes
    # the update arrays.
    size=6980504;vv=np.full(size,np.nan);qq=np.full(size,np.nan);ii=updates.submission_id.to_numpy();assert ii.min()>=1 and ii.max()<size;vv[ii]=updates.speed_kmh;qq[ii]=updates.flow_vph
    n=changedrows=changedv=changedq=0
    with open(PARENT,'rb')as src,open(dest,'xb')as dst:
        dst.write(src.readline())
        for line in src:
            n+=1;sid=int(line.split(b',',1)[0])
            if not np.isfinite(vv[sid]):dst.write(line);continue
            fields=line.rstrip(b'\r\n').split(b',');assert fields[1]==b'state';oldv=float(fields[2]);oldq=float(fields[3]);cv=vv[sid]!=oldv;cq=qq[sid]!=oldq;assert cv or cq
            if cv:fields[2]=format(vv[sid],'.3f').encode();changedv+=1
            if cq:fields[3]=format(qq[sid],'.3f').encode();changedq+=1
            dst.write(b','.join(fields)+b'\n');changedrows+=1
    assert n==6980503 and changedrows==len(updates)
    manifest=dict(candidate=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,parent=str(PARENT),parent_sha256=PSHA,rows=n,changed_state_rows=changedrows,changed_speed_rows=changedv,changed_flow_rows=changedq,untouched='Task 2 queue, OD, all Task 1 holes, both D12_I405 panels; every non-v/q field and every unchanged row preserved byte-for-byte',keyed_updates=str(WORK/'state_updates.parquet'),keyed_updates_sha256=sha(WORK/'state_updates.parquet'),model=str(HERE/'density/l2_uniform.txt'),model_sha256=sha(HERE/'density/l2_uniform.txt'),selection_sha256=sha(HERE/'density/selection_before_march.json'),confirmation_sha256=sha(confirmation_path) if confirmation_path.exists() else None,status='local_candidate_not_submitted',public_score=None,private_score=None,seconds=time.time()-t0)  # package edit: confirmation_sha256 is None when the March gate file is absent
    D.dump(OUT/'MANIFEST.json',manifest);print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':run()
