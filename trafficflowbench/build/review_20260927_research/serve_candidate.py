"""Freeze-gated HMP50 serving; complete eligible-cell pooling, exact HM25 parent reconstruction."""
from common import *
from feb_pool import cluster
from boundary_context import grid,physics,P1
import hashlib
DEST=HERE/'serving';DEST.mkdir(exist_ok=True)
PARENT=ROOT/'build/deliverables/2026-09-27/submission_T1_HM25.csv'

def run():
    f=json.loads((HERE/'frozen_recipe.json').read_text());assert f['mixture_weight']==.5
    # package edit: the parent CSV may be a regenerated state-only stand-in; its hole speeds are checked row by row below (rebuilt HM25 == parent)
    print('parent sha256',hashlib.sha256(PARENT.read_bytes()).hexdigest()==f['parent_sha256'] and 'matches the original' or 'differs (state-only stand-in)')
    for p,sha in f['source_sha256_before_april'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha
    conf=pd.read_parquet(HERE/'full_pool/private_predictions.parquet');conf['pool50']=2*conf.adaptive-conf.t1
    check=evaluate(conf,conf.pool50)
    assert all(check[p]['rel_mse']<0 for p in ['all','odd','even'])and check['cluster_ci'][1]<0
    (HERE/'private_confirmation.json').write_text(json.dumps(check,indent=2));print('FROZEN APRIL',json.dumps(check),flush=True)
    if (HERE/'strength/results.json').exists():  # package edit: research gate file (validation+Feb results); not shipped
        assert all(json.loads((HERE/'strength/results.json').read_text())['feb'][mode]['pool50']['score']['total_'+p]>0 for mode in ['serve','strict']for p in 'AB')
    hist=pd.read_parquet(HM/'official_state_updates.parquet').rename(columns={'parent_speed':'t1_speed','speed_kmh':'old_unrounded_hm25'})
    expected=set(hist.submission_id);parent=pd.concat([a[a.submission_id.isin(expected)][['submission_id','speed_kmh','flow_vph']]for a in pd.read_csv(PARENT,chunksize=500000)],ignore_index=True).rename(columns={'speed_kmh':'parent_speed','flow_vph':'parent_flow'})
    assert len(parent)==128465 and parent.submission_id.is_unique
    result=[];report={}
    for split in ['validation','private']:
        d=pd.read_parquet(R3/'work'/f'off_rows_{split}.parquet');S.derive(d)
        for kind,col in [('pqueue','p'),('free','rf'),('queued','rq')]:
            m=lgb.Booster(model_file=str(HM/f'official_{split}_{kind}.txt'));d[col]=m.predict(d[S.FEATS],num_threads=3)
        d['group']=-1
        clusters={}
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
        oldmix=np.clip(z.g2_speed+(1-z.p)*z.rf+z.p*z.rq,S.SPEED_FLOOR,S.SPEED_CAP)
        rebuilt=.75*z.t1_speed+.25*oldmix
        rounded=np.array([float(f'{v:.3f}')for v in rebuilt]);assert np.max(np.abs(rounded-z.parent_speed))==0
        assert np.max(np.abs(rebuilt-z.old_unrounded_hm25))<1e-8
        newmix=np.clip(z.g2_speed+(1-z.pooled_p)*z.rf+z.pooled_p*z.rq,S.SPEED_FLOOR,S.SPEED_CAP)
        z['speed_kmh']=.5*z.t1_speed+.5*newmix;z['split']=split
        assert z.speed_kmh.between(S.SPEED_FLOOR-1e-10,S.SPEED_CAP+1e-10).all()
        z['speed_kmh']=z.speed_kmh.clip(6.,120.75)
        held=pd.read_parquet(HERE/'full_pool'/f'{split}_predictions.parquet');delta=2*held.adaptive-held.t1-held.hm25;dv=z.speed_kmh-z.parent_speed
        ratio=float(np.quantile(np.abs(dv),.99)/np.quantile(np.abs(delta),.99));assert ratio<2,(split,ratio)
        report[split]={'all_eligible_inference_rows':len(d),'keyed_hole_rows':len(z),'HM25_rebuild_max_diff':float(np.max(np.abs(rounded-z.parent_speed))),'p99_delta_ratio_official_to_holdout':ratio,'speed_delta':S.desc(dv),'max_probability_change':float(np.max(np.abs(z.pooled_p-z.p)))}
        result.append(z[['submission_id','panel','split','timestamp','link_id','window_id','cond','parent_speed','parent_flow','speed_kmh','p','pooled_p','group']]);print(split,json.dumps(report[split]),flush=True)
    out=pd.concat(result,ignore_index=True).sort_values('submission_id');assert len(out)==128465 and out.submission_id.is_unique and set(out.submission_id)==expected
    out.to_parquet(DEST/'official_state_updates.parquet',index=False);(DEST/'results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':run()
