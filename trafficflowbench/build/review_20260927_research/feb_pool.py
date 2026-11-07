"""February official-like 18-step holes: unchanged HM models, legal masked topology, full score proxy."""
from common import *
from structure import B,grid,physics,P1
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
sys.path.insert(0,str(ROOT/'build/review_20260927/iteration2'));import hole_mixture as H
DEST=HERE/'feb_pool';DEST.mkdir(exist_ok=True)
G2=ROOT/'build/review_20260924_ctm/g2v2'

def cluster(V,vf):
    ob=np.isfinite(V).astype(np.float32);q=(np.isfinite(V)&(V<=.6*vf[:,None])).astype(np.float32)
    inter=q@q.T;union=q@ob.T+ob@q.T-inter;dist=np.where(union>=20,1-inter/np.maximum(union,1),1)
    dist=np.clip((dist+dist.T)/2,0,1);np.fill_diagonal(dist,0)
    return fcluster(linkage(squareform(dist),method='complete'),.05,criterion='distance')

def run():
    a=pd.read_parquet(R3/'work/inworld_apply_rows.parquet');S.derive(a)
    old=pd.read_parquet(HM/'predictions.parquet')
    for key in KEY:assert np.array_equal(a[key].to_numpy(),old[key].to_numpy())
    k=a[KEY+['day','cond','ch_ql','y_v','y_ql','is_target']].copy()
    for mode in ['serve','strict']:
        for col in ['p','fv','qv']:k[mode+'_'+col]=np.nan
        for odd in ([None] if mode=='serve'else[True,False]):
            mask=np.ones(len(a),bool) if odd is None else((a.day%2==1).to_numpy()!=odd)
            tag='serve' if odd is None else f'strict_fitodd{int(odd)}'
            for kind,col in [('pqueue','p'),('free','fv'),('queued','qv')]:
                model=lgb.Booster(model_file=str(HM/f'{tag}_{kind}.txt'));v=model.predict(a.loc[mask,S.FEATS],num_threads=3)
                k.loc[mask,mode+'_'+col]=v if kind=='pqueue'else a.loc[mask,'ch_v'].to_numpy()+v
        p=k[mode+'_p'];fv=k[mode+'_fv'];qv=k[mode+'_qv'];mix=np.clip((1-p)*fv+p*qv,S.SPEED_FLOOR,S.SPEED_CAP)
        assert np.max(np.abs(mix-old[mode+'_new']))<1e-8
        k[mode+'_hm25']=old[mode+'_blend25'];k[mode+'_t1']=old[mode+'_base'];k[mode+'_group']=np.nan
    for panel in S.PANELS:
        gs=B.load_panel(panel,'train','2031_02');dates=[g.date for g in gs];par=physics.link_params(panel,[str(l)for l in gs[0].links]);v=np.stack([g.value['speed_kmh']for g in gs])
        wins=pd.read_csv(G2/f'eval_windows_{panel}_2031_02.csv')
        for w in wins.itertuples():v[dates.index(w.date),:,int(w.origin)+1:int(w.origin)+19]=np.nan
        days=np.array([int(d[-2:])for d in dates]);audit={}
        for odd in [None,True,False]:
            fit=np.ones(len(gs),bool) if odd is None else((days%2==1)==odd)
            vv=np.concatenate(v[fit],axis=1);cl=cluster(vv,par['free_speed_kmh']);tag='serve'if odd is None else 'strict'
            ix=(k.panel==panel).to_numpy()
            if odd is not None:ix&=((k.day%2==1).to_numpy()!=odd)
            k.loc[ix,tag+'_group']=cl[k.loc[ix,'li'].to_numpy(int)]
            audit[str(odd)]={'fit_dates':np.array(dates)[fit].tolist(),'clusters':cl.tolist(),'hidden_holes':len(wins)}
        (DEST/f'{panel}_topology.json').write_text(json.dumps(audit,indent=2))
    report={}
    for mode in ['serve','strict']:
        pooled=k.groupby(['panel','date','origin',mode+'_group','step_i'])[mode+'_p'].transform('mean')
        new=np.clip((1-pooled)*k[mode+'_fv']+pooled*k[mode+'_qv'],S.SPEED_FLOOR,S.SPEED_CAP)
        k[mode+'_pool']=.75*k[mode+'_t1']+.25*new
        target=k[k.is_target.astype(bool)&k.y_v.notna()].copy().rename(columns={mode+'_hm25':'hm25'})
        report[mode]=evaluate(target,target[mode+'_pool']);print(mode,json.dumps(report[mode]),flush=True)
    k.to_parquet(DEST/'predictions.parquet',index=False)
    rows=[]
    for panel in S.PANELS:
        z=dict(np.load(G2/f'c0_cache/{panel}_2031_02_v2.npz',allow_pickle=True));meta=json.loads(str(z.pop('meta')))
        par=physics.link_params(panel,meta['links']);keep=physics.scoreable_mask(panel,meta['links'],'train',meta['times']);lanes=np.maximum(par['lanes'],1.);ln=lanes[:,None];length=par['length_km'][:,None];vf=par['free_speed_kmh'][:,None]
        capl=par['capacity_vph'][:,None]/ln;kcl=par['critical_density'][:,None]/ln;kjl=par['k_jam'][:,None]/ln;c=k[k.panel==panel]
        for di,date in enumerate(meta['dates']):
            key=f'd{di}';b=c[c.date==date];li=b.li.to_numpy(int);ti=(b.origin+1+b.step_i).to_numpy(int)
            for mode in ['serve','strict']:
                for tag in ['hm25','pool']:
                    name=mode+'_'+tag;v=z[f'{key}_v0'].astype(float).copy();q=z[f'{key}_q0'].astype(float).copy();v[li,ti]=b[name];q[li,ti]=b.ch_ql.to_numpy()*lanes[li]
                    rows.append(dict(panel=panel,date=date,regime=meta['regimes'][di],cand=name,**H.acc(v,q,z,key,length,ln,vf,capl,kcl,kjl,keep)))
        del z;gc.collect()
    days=pd.DataFrame(rows);days.to_parquet(DEST/'days.parquet',index=False)
    ref=pd.read_parquet(HM/'days.parquet')
    for mode in ['serve','strict']:
        cmp=days[days.cand==mode+'_hm25'].merge(ref[ref.cand==mode+'_blend25'],on=['panel','date'],validate='one_to_one',suffixes=('_new','_old'))
        for col in ['sse_v_in','numB','denB','fdnB']:assert np.max(np.abs(cmp[col+'_new']-cmp[col+'_old'])/np.maximum(np.abs(cmp[col+'_old']),1e-9))<1e-7
        per={}
        for panel in S.PANELS:
            sub=days[days.panel==panel];aa=H.summarize(sub[sub.cand==mode+'_hm25']);bb=H.summarize(sub[sub.cand==mode+'_pool']);row={c:float(bb[c]-aa[c])for c in ['S_state','S_phys_A','S_phys_B']};row['total_B']=.35*row['S_state']+.15*row['S_phys_B'];per[panel]=row
        report[mode+'_score']={'five_family':{c:.8*np.mean([r[c]for r in per.values()])for c in row},'perpanel':per};print(mode,'score',json.dumps(report[mode+'_score']),flush=True)
    (DEST/'results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':run()
