"""Structural marginal-probability pilots. Sparse query grid => diagnostic until full-grid confirmation."""
from common import *
from boundary_context import B,grid,physics,P1
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import isotonic_regression
DEST=HERE/'structure';DEST.mkdir(exist_ok=True)

def topology(panel,split,odd):
    gs=[grid.build_masked(panel,p)for p in P1.split_masked_files(panel,split)]
    gs=[g for g in gs if (int(g.date[-2:])%2==1)==odd]
    par=physics.link_params(panel,[str(l)for l in gs[0].links]);vf=par['free_speed_kmh']
    v=np.concatenate([g.value['speed_kmh']for g in gs],axis=1);o=np.isfinite(v).astype(np.float32);q=(np.isfinite(v)&(v<=.6*vf[:,None])).astype(np.float32)
    inter=q@q.T;union=q@o.T+o@q.T-inter
    dist=np.where(union>=20,1-inter/np.maximum(union,1),1);dist=np.clip((dist+dist.T)/2,0,1);np.fill_diagonal(dist,0)
    cl=fcluster(linkage(squareform(dist,checks=True),method='complete'),t=.05,criterion='distance')
    np.savez_compressed(DEST/f'topology_{split}_{panel}_{int(odd)}.npz',clusters=cl,distance=dist,union=union,fit_dates=np.array([g.date for g in gs]))
    return cl

def run(split='validation'):
    d,st=load(split);d['p']=np.nan;d['fv']=np.nan;d['qv']=np.nan;d['group']=np.nan
    for odd in [True,False]:
        x=strict(d,st,odd);te=((x.day%2==1).to_numpy()!=odd)
        tag=f'{split}_strict_fitodd{int(odd)}'
        for kind,col in [('pqueue','p'),('free','fv'),('queued','qv')]:
            m=lgb.Booster(model_file=str(HM/f'{tag}_{kind}.txt'));v=m.predict(x.loc[te,S.FEATS],num_threads=3)
            d.loc[te,col]=v if kind=='pqueue' else x.loc[te,'ch_v'].to_numpy()+v
        for panel in S.PANELS:
            cl=topology(panel,split,odd);ix=te&(d.panel==panel).to_numpy();d.loc[ix,'group']=cl[d.loc[ix,'li'].to_numpy(int)]
        del x;gc.collect()
    q=d.p.to_numpy();mix=np.clip((1-q)*d.fv+q*d.qv,S.SPEED_FLOOR,S.SPEED_CAP)
    old=pd.read_parquet(HM/f'{split}_strict_predictions.parquet');old=old[KEY+['new']]
    chk=d[KEY].merge(old,on=KEY,validate='one_to_one');assert np.max(np.abs(mix-chk.new))<1e-8
    assert np.max(np.abs(.75*d.t1+.25*mix-d.hm25))<1e-8
    iso=q.copy();changed=0;eligible=0
    for _,ii in d.groupby(['panel','date','origin','li'],sort=False,observed=True).indices.items():
        g=d.iloc[ii];pre=g.f_pv.iloc[0];post=g.f_sv.iloc[0];cut=.6*g.f_free_speed_kmh.iloc[0]
        if not np.isfinite(pre)or not np.isfinite(post)or ((pre<=cut)==(post<=cut)):continue
        jj=ii[np.argsort(g.step_i.to_numpy())];v=isotonic_regression(q[jj],increasing=pre>cut).x
        eligible+=len(jj);changed+=int(np.count_nonzero(np.abs(v-q[jj])>1e-10));iso[jj]=v
    pool=d.groupby(['panel','date','origin','group','step_i'],observed=True).p.transform('mean').to_numpy()
    oracle=(d.y_v<=.6*d.f_free_speed_kmh).to_numpy(float)
    report={'diagnostic_sparse_grid':True,'isotonic_eligible':eligible,'isotonic_changed':changed};out=d[KEY+['y_v','day','cond','t1','hm25','p','fv','qv','group']].copy()
    for name,p in [('isotonic',iso),('pool',pool),('oracle_state',oracle)]:
        new=np.clip((1-p)*d.fv+p*d.qv,S.SPEED_FLOOR,S.SPEED_CAP);out[name]=.75*d.t1+.25*new;report[name]=evaluate(out,out[name]);print(name,json.dumps(report[name]),flush=True)
    out.to_parquet(DEST/f'{split}_predictions.parquet',index=False);(DEST/f'{split}_results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':run()
