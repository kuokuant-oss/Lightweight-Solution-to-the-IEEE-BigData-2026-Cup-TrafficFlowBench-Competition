"""Build capacity delta on DK50 only after frozen confirmations pass. Never submits."""
# Task 1 reproduction pipeline, step 8 (`pipeline/task1.sh`, log `cap`).
# CAP (capacity candidate) works on the general (non-hole) Task 1 cells of the
# 8 Task 2 corridors. CatBoost residual heads (state/C6_long_final, 4,800 trees
# each) redo the RLB residual correction of the C1 fill, and the state becomes
#   DK50 state + alpha x (CatBoost state - LightGBM state),  alpha = 0.5,
# where both states are composed the same way (C1 + residual, then k* on its 5
# corridors). DK50 is the density-correction step before this one: a LightGBM
# head corrects the density q/v and speed and flow are projected onto the
# corrected density with weight 0.5. k* is the joint speed/flow projection
# onto a blend of q/v and the auxiliary density.
#
# Inputs: the DK50 parent CSV (in the reproduction, the state-only stand-in
# written by pipeline/task1_state_csv.py), the RLB serving features
# rl/feats/{serve,blk_serve}_<panel>_<split>.parquet, the RLB models, the k*
# updates and the DK50 density head (review_20260927_structural/density/).
# Outputs: state/serving/state_updates.parquet (submission_id, speed_kmh,
# flow_vph of the changed rows), state/serving/stats.json, and the candidate
# CSV deliverables/2026-09-27/capacity/submission_HMP50_DK50_CAP.csv with
# MANIFEST.json.
import os,sys,json,hashlib,gc,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='3'
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
from common import HERE,ROOT,state_alpha
sys.path.insert(0,str(ROOT/'build/review_20260927_structural'));import density_experiment as D
from state_confirm import predict
# OUT: candidate CSV folder; WORK: keyed updates. PARENT: the DK50 state CSV;
# PSHA is the SHA256 of the original file, only reported. KUPD: k* updates.
OUT=ROOT/'build/deliverables/2026-09-27/capacity';WORK=HERE/'state/serving'
PARENT=ROOT/'build/deliverables/2026-09-27/density50/submission_HMP50_DK50.csv';PSHA='b6cb92ad28b89105d1a57a83b047f556b0fd7a04ffb27fb7c8bb04ffd4c8e219'
KUPD=ROOT/'build/strongest_candidate_20260922/state/transfer/serving'
def sha(p):
    """Returns the SHA256 hex digest of a file, read in 8 MB chunks."""
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def run():
    """Builds the CAP state updates and writes the candidate CSV.

    Raises:
      AssertionError: when the February gate fails, the recomposed DK50 state
        differs from the parent CSV, or the candidate CSV already exists.
    """
    confirmation_path=HERE/'state/march/results.json'
    if confirmation_path.exists():  # package edit: March confirmation gate file is not shipped (built from the validation split)
        confirmation=json.loads(confirmation_path.read_text());assert confirmation['promote']
    # package edit: the parent CSV is the DK50 CSV written earlier in this run; its state rows are checked row by row below (dv == dq == 0)
    print('parent sha256',sha(PARENT)==PSHA and 'matches the original' or 'differs (regenerated parent)',flush=True)
    # The chosen CatBoost heads and alpha (state/selection.json), and the
    # February confirmation gate (state/confirmation/report.json, train
    # months only): the candidate must raise the February total score.
    choice=json.loads((HERE/'state/selection.json').read_text());folder=HERE/'state'/(choice['name']+'_final')
    feb=json.loads((HERE/'state/confirmation/report.json').read_text());assert feb['results']['serve']['deltas']['candidate']['total_B']>0
    OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True);dest=OUT/'submission_HMP50_DK50_CAP.csv';assert not dest.exists(),'refuse to overwrite frozen CSV'
    # Parent speed_kmh and flow_vph of every state row, by submission_id.
    # head: DK50 density head; mv/mq: RLB residual models (speed in km/h,
    # flow per lane) and their feature list.
    t0=time.time();base=pd.concat([c[c.task=='state'][['submission_id','speed_kmh','flow_vph']]for c in pd.read_csv(PARENT,usecols=['submission_id','task','speed_kmh','flow_vph'],chunksize=500000)],ignore_index=True).set_index('submission_id')
    head=lgb.Booster(model_file=str(D.HERE/'density/l2_uniform.txt'));md=D.RL/'models/RLB_final';fs=json.loads((md/'feats.json').read_text());mv=lgb.Booster(model_file=str(md/'y_v.txt'));mq=lgb.Booster(model_file=str(md/'y_q.txt'));parts=[];stats=[]
    for split in ['validation','private']:
        for panel in D.E.PANELS:
            # One row per general target cell (holes excluded); ln: lanes;
            # ref: parent values of the same cells.
            tab=pd.read_parquet(D.RL/f'feats/serve_{panel}_{split}.parquet');tab=pd.concat([tab,pd.read_parquet(D.RL/f'feats/blk_serve_{panel}_{split}.parquet')],axis=1);ln=tab.lane_n.to_numpy(float);sid=tab.submission_id.to_numpy();ref=base.loc[sid]
            # LightGBM state: C1 fill + RLB residual, clipped to [6, 120.75]
            # km/h and >= 50 veh/h, then k* on its corridors, rounded to 3
            # decimals as in the CSV.
            rv=mv.predict(tab[fs],num_threads=3);rq=mq.predict(tab[fs],num_threads=3)
            # Exact existing production composition uses original C1 fields.
            v=np.clip(tab.c1_v.to_numpy()+rv,6,120.75);q=np.maximum((tab.c1_q.to_numpy()/ln+rq)*ln,50)
            if panel in D.R.KSTAR:
                u=pd.read_parquet(KUPD/f'updates_{panel}.parquet').set_index('submission_id').loc[sid];assert u.aux_density_vpkm.notna().all()
                v,q=D.R.kstar(np.round(v,3),np.round(q,3),u.aux_density_vpkm.to_numpy(),ln,D.R.SEL['panel_alphas'][panel],D.R.SEL['weights'][panel])
            v=np.round(v,3);q=np.round(q,3)
            # DK50: corrected density = parent q/v + head prediction (>= 0);
            # speed and flow projected onto it with weight 0.5. The result
            # must equal the parent CSV exactly on every cell.
            # Density head uses the same float32 feature inputs as January fit.
            rawv=np.clip(tab.p_v.to_numpy()+rv,6,120.75);rawq=np.maximum(tab.p_q.to_numpy()+rq,50/ln);hfs=D.augment(tab,rawv,rawq);assert hfs==head.feature_name()
            kh=np.maximum(tab.parent_k.to_numpy()+head.predict(tab[hfs],num_threads=3),0);pv,pq=D.project(v,q/ln,kh,ln,panel,.5);pv=np.round(pv,3);pq=np.round(pq,3)
            dv=float(abs(pv-ref.speed_kmh.to_numpy()).max());dq=float(abs(pq-ref.flow_vph.to_numpy()).max());assert dv==0 and dq==0,(panel,split,dv,dq)
            # CatBoost state: the same composition with the CatBoost heads.
            nr_v=predict(folder,choice,tab,'y_v',production=True);nr_q=predict(folder,choice,tab,'y_q',production=True)
            nv=np.round(np.clip(tab.c1_v.to_numpy()+nr_v,6,120.75),3);nq=np.round(np.maximum((tab.c1_q.to_numpy()/ln+nr_q)*ln,50),3)
            if panel in D.R.KSTAR:nv,nq=D.R.kstar(nv,nq,u.aux_density_vpkm.to_numpy(),ln,D.R.SEL['panel_alphas'][panel],D.R.SEL['weights'][panel])
            nv=np.round(nv,3);nq=np.round(nq,3);alpha=state_alpha(choice['alpha'],tab.p_r.to_numpy())
            # Final state = DK50 state + alpha x (CatBoost - LightGBM state).
            nv=np.round(np.clip(pv+alpha*(nv-v),6,120.75),3);nq=np.round(np.maximum(pq+alpha*(nq-q),50),3)
            v,q=pv,pq
            assert np.isfinite(nv).all() and np.isfinite(nq).all() and nv.min()>=6 and nv.max()<=120.75 and nq.min()>=50
            # Keep only the cells whose rounded speed or flow changed.
            changed=(nv!=v)|(nq!=q);part=pd.DataFrame({'submission_id':sid[changed],'speed_kmh':nv[changed],'flow_vph':nq[changed]});parts.append(part)
            stats.append(dict(panel=panel,split=split,nonhole_rows=len(tab),changed=int(changed.sum()),changed_speed=int((nv!=v).sum()),changed_flow=int((nq!=q).sum()),parent_identity_speed=dv,parent_identity_flow=dq,mean_abs_dv=float(abs(nv-v).mean()),max_abs_dv=float(abs(nv-v).max()),mean_abs_dq_lane=float((abs(nq-q)/ln).mean()),max_abs_dq_lane=float((abs(nq-q)/ln).max())))
            print('serving',panel,split,'changed',int(changed.sum()),'seconds',round(time.time()-t0),flush=True);del tab;gc.collect()
    updates=pd.concat(parts,ignore_index=True);assert updates.submission_id.is_unique;updates.to_parquet(WORK/'state_updates.parquet',index=False);D.dump(WORK/'stats.json',stats)
    # Patch the parent CSV line by line. submission_id runs 1..6,980,503, so
    # it indexes the update arrays directly; unchanged rows are copied as is.
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
    # Manifest: output and input SHA256, change counts, selection and models.
    manifest=dict(candidate=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,parent=str(PARENT),parent_sha256=PSHA,rows=n,changed_state_rows=changedrows,changed_speed_rows=changedv,changed_flow_rows=changedq,untouched='Task 2 queue, OD, all Task 1 holes, both D12_I405 panels; every non-v/q field and every unchanged row preserved byte-for-byte',keyed_updates=str(WORK/'state_updates.parquet'),keyed_updates_sha256=sha(WORK/'state_updates.parquet'),selection=choice,model_sha256={str(p.relative_to(HERE)):sha(p)for p in (HERE/'state').glob('*_final/*')if p.suffix in ['.txt','.cbm','.ubj']and not p.name.startswith('._')},selection_sha256=sha(HERE/'state/selection.json'),confirmation_sha256=sha(confirmation_path) if confirmation_path.exists() else None,status='local_candidate_not_submitted',public_score=None,private_score=None,seconds=time.time()-t0)  # package edit: confirmation_sha256 is None when the March gate file is absent
    D.dump(OUT/'MANIFEST.json',manifest);print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':run()
