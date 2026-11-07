"""Freeze-gated HMP50 serving; complete eligible-cell pooling, exact HM25 parent reconstruction."""
# Runs in pipeline/task1.sh step 7 (last program of the step).
# HMP50 ("HM-Pool50"): the HM25 hole mixture with its queue probability
# pooled (uniform mean) over the links of the same learned cluster at the
# same window and step, blended 50 % with the T1 speed:
#   speed = .5 x T1 speed + .5 x clip(G2 speed + (1 - p_pool) x free residual
#           + p_pool x queued residual).
# Clusters (feb_pool.cluster): complete-linkage groups of links whose queues
# in the split's released speeds coincide (1 - queue Jaccard <= 0.05).
# Gates before serving: frozen_recipe.json (weight .5 and the SHA256 of the
# research sources), the April pseudo-hole confirmation and the exact HM25
# rebuild. Outputs: serving/official_state_updates.parquet (128,465 hole
# rows), serving/results.json, serving/<split>_topology.json and
# private_confirmation.json.
from common import *
from feb_pool import cluster
from boundary_context import grid,physics,P1
import hashlib
DEST=HERE/'serving';DEST.mkdir(exist_ok=True)
# Parent: the HM25 submission. In the package run it is the state-only CSV
# that pipeline/task1_state_csv.py rebuilds from this run's keyed files.
PARENT=ROOT/'build/deliverables/2026-09-27/submission_T1_HM25.csv'

def run():
    """Serves HMP50 on the official hole rows of both splits.

    Raises:
      AssertionError: When a gate fails, the HM25 rebuild differs from the
        parent, or the official speed change exceeds 2 times the held-out
        change (99th percentile of |change|).
    """
    f=json.loads((HERE/'frozen_recipe.json').read_text());assert f['mixture_weight']==.5
    # package edit: the parent CSV may be a regenerated state-only stand-in; its hole speeds are checked row by row below (rebuilt HM25 == parent)
    print('parent sha256',hashlib.sha256(PARENT.read_bytes()).hexdigest()==f['parent_sha256'] and 'matches the original' or 'differs (state-only stand-in)')
    for p,sha in f['source_sha256_before_april'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha
    # April confirmation on the private pseudo holes: pool50 = 2 x adaptive
    # - T1 = .5 T1 + .5 pooled mixture (adaptive is the 25 % blend). It must
    # lower hole speed MSE against HM25 overall, on each day parity and at the
    # upper end of the panel-day bootstrap interval.
    conf=pd.read_parquet(HERE/'full_pool/private_predictions.parquet');conf['pool50']=2*conf.adaptive-conf.t1
    check=evaluate(conf,conf.pool50)
    assert all(check[p]['rel_mse']<0 for p in ['all','odd','even'])and check['cluster_ci'][1]<0
    (HERE/'private_confirmation.json').write_text(json.dumps(check,indent=2));print('FROZEN APRIL',json.dumps(check),flush=True)
    if (HERE/'strength/results.json').exists():  # package edit: research gate file (validation+Feb results); not shipped
        assert all(json.loads((HERE/'strength/results.json').read_text())['feb'][mode]['pool50']['score']['total_'+p]>0 for mode in ['serve','strict']for p in 'AB')
    # hist: the HM25 serving rows (T1 speed and unrounded HM25 speed).
    hist=pd.read_parquet(HM/'official_state_updates.parquet').rename(columns={'parent_speed':'t1_speed','speed_kmh':'old_unrounded_hm25'})
    expected=set(hist.submission_id);parent=pd.concat([a[a.submission_id.isin(expected)][['submission_id','speed_kmh','flow_vph']]for a in pd.read_csv(PARENT,chunksize=500000)],ignore_index=True).rename(columns={'speed_kmh':'parent_speed','flow_vph':'parent_flow'})
    assert len(parent)==128465 and parent.submission_id.is_unique
    result=[];report={}
    for split in ['validation','private']:
        d=pd.read_parquet(R3/'work'/f'off_rows_{split}.parquet');S.derive(d)
        # The three HM25 heads fitted on this split by hole_mixture_serve.py.
        for kind,col in [('pqueue','p'),('free','rf'),('queued','rq')]:
            m=lgb.Booster(model_file=str(HM/f'official_{split}_{kind}.txt'));d[col]=m.predict(d[S.FEATS],num_threads=3)
        d['group']=-1
        clusters={}
        # Clusters from all released days of the split; the queue probability
        # is averaged within cluster, window and step.
        for panel in S.PANELS:
            gs=[grid.build_masked(panel,path)for path in P1.split_masked_files(panel,split)];par=physics.link_params(panel,[str(l)for l in gs[0].links]);v=np.concatenate([g.value['speed_kmh']for g in gs],axis=1)
            cl=cluster(v,par['free_speed_kmh']);ix=(d.panel==panel).to_numpy();d.loc[ix,'group']=cl[d.loc[ix,'li'].to_numpy(int)]
            clusters[panel]={'link_ids':[str(l)for l in gs[0].links],'clusters':cl.tolist(),'released_dates':[g.date for g in gs]}
        d['pooled_p']=d.groupby(['panel','date','origin','group','step_i'],sort=False).p.transform('mean')
        assert d.pooled_p.between(0,1).all()
        (DEST/f'{split}_topology.json').write_text(json.dumps(clusters,indent=2))
        keys=['panel','timestamp','link_id'];g2=pd.read_parquet(R3/'work'/f'parent_{split}_state.parquet');g2=g2[g2.submission_id.isin(expected)].rename(columns={'speed_kmh':'g2_speed'})
        z=g2[['submission_id']+keys+['g2_speed']].merge(d[keys+['p','rf','rq','pooled_p','group','window_id','cond']],on=keys,validate='one_to_one')
        z=z.merge(hist[['submission_id','t1_speed','old_unrounded_hm25']],on='submission_id',validate='one_to_one').merge(parent,on='submission_id',validate='one_to_one')
        # Rebuilds HM25 (.75 T1 + .25 mixture with the link's own p); it must
        # equal the parent speed after rounding to 3 decimals.
        oldmix=np.clip(z.g2_speed+(1-z.p)*z.rf+z.p*z.rq,S.SPEED_FLOOR,S.SPEED_CAP)
        rebuilt=.75*z.t1_speed+.25*oldmix
        rounded=np.array([float(f'{v:.3f}')for v in rebuilt]);assert np.max(np.abs(rounded-z.parent_speed))==0
        assert np.max(np.abs(rebuilt-z.old_unrounded_hm25))<1e-8
        newmix=np.clip(z.g2_speed+(1-z.pooled_p)*z.rf+z.pooled_p*z.rq,S.SPEED_FLOOR,S.SPEED_CAP)
        z['speed_kmh']=.5*z.t1_speed+.5*newmix;z['split']=split
        assert z.speed_kmh.between(S.SPEED_FLOOR-1e-10,S.SPEED_CAP+1e-10).all()
        z['speed_kmh']=z.speed_kmh.clip(6.,120.75)
        # Size check: the 99th percentile of |change| stays below 2 x that of
        # the held-out change (pool50 - HM25).
        held=pd.read_parquet(HERE/'full_pool'/f'{split}_predictions.parquet');delta=2*held.adaptive-held.t1-held.hm25;dv=z.speed_kmh-z.parent_speed
        ratio=float(np.quantile(np.abs(dv),.99)/np.quantile(np.abs(delta),.99));assert ratio<2,(split,ratio)
        report[split]={'all_eligible_inference_rows':len(d),'keyed_hole_rows':len(z),'HM25_rebuild_max_diff':float(np.max(np.abs(rounded-z.parent_speed))),'p99_delta_ratio_official_to_holdout':ratio,'speed_delta':S.desc(dv),'max_probability_change':float(np.max(np.abs(z.pooled_p-z.p)))}
        result.append(z[['submission_id','panel','split','timestamp','link_id','window_id','cond','parent_speed','parent_flow','speed_kmh','p','pooled_p','group']]);print(split,json.dumps(report[split]),flush=True)
    out=pd.concat(result,ignore_index=True).sort_values('submission_id');assert len(out)==128465 and out.submission_id.is_unique and set(out.submission_id)==expected
    out.to_parquet(DEST/'official_state_updates.parquet',index=False);(DEST/'results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':run()
