"""Single predeclared 50% mixture-strength comparison, including no-pooling control."""
from common import *
sys.path.insert(0,str(ROOT/'build/review_20260927/iteration2'));import hole_mixture as H
DEST=HERE/'strength';DEST.mkdir(exist_ok=True);G2=ROOT/'build/review_20260924_ctm/g2v2'
def run():
    d=pd.read_parquet(HERE/'full_pool/validation_predictions.parquet');report={'validation':{}}
    d['pool50']=2*d.adaptive-d.t1;d['hm50']=2*d.hm25-d.t1
    assert np.all(d[['pool50','hm50']].to_numpy()>=S.SPEED_FLOOR-1e-8) and np.all(d[['pool50','hm50']].to_numpy()<=S.SPEED_CAP+1e-8)
    assert np.max(np.abs(d.pool50-(.5*d.t1+.5*(4*d.adaptive-3*d.t1))))<1e-10
    for name in ['adaptive','pool50','hm50']:
        report['validation'][name]=evaluate(d,d[name]);print('validation',name,json.dumps(report['validation'][name]),flush=True)
    d.to_parquet(DEST/'validation_predictions.parquet',index=False)
    k=pd.read_parquet(HERE/'feb_pool/predictions.parquet')
    for mode in ['serve','strict']:
        k[mode+'_pool50']=2*k[mode+'_pool']-k[mode+'_t1'];k[mode+'_hm50']=2*k[mode+'_hm25']-k[mode+'_t1']
    rows=[]
    for panel in S.PANELS:
        z=dict(np.load(G2/f'c0_cache/{panel}_2031_02_v2.npz',allow_pickle=True));meta=json.loads(str(z.pop('meta')))
        par=H.physics.link_params(panel,meta['links']);keep=H.physics.scoreable_mask(panel,meta['links'],'train',meta['times']);lanes=np.maximum(par['lanes'],1.);ln=lanes[:,None];length=par['length_km'][:,None];vf=par['free_speed_kmh'][:,None];capl=par['capacity_vph'][:,None]/ln;kcl=par['critical_density'][:,None]/ln;kjl=par['k_jam'][:,None]/ln;c=k[k.panel==panel]
        for di,date in enumerate(meta['dates']):
            key=f'd{di}';a=c[c.date==date];li=a.li.to_numpy(int);ti=(a.origin+1+a.step_i).to_numpy(int)
            for mode in ['serve','strict']:
                for tag in ['pool50','hm50']:
                    name=mode+'_'+tag;v=z[f'{key}_v0'].astype(float).copy();q=z[f'{key}_q0'].astype(float).copy();v[li,ti]=a[name];q[li,ti]=a.ch_ql.to_numpy()*lanes[li]
                    rows.append(dict(panel=panel,date=date,regime=meta['regimes'][di],cand=name,**H.acc(v,q,z,key,length,ln,vf,capl,kcl,kjl,keep)))
        del z;gc.collect()
    days=pd.concat([pd.DataFrame(rows),pd.read_parquet(HERE/'feb_pool/days.parquet')],ignore_index=True);days.to_parquet(DEST/'days.parquet',index=False)
    report['feb']={}
    for mode in ['serve','strict']:
        target=k[k.is_target.astype(bool)&k.y_v.notna()].copy().rename(columns={mode+'_hm25':'hm25'})
        report['feb'][mode]={}
        for tag in ['pool','pool50','hm50']:
            per={}
            for panel in S.PANELS:
                sub=days[days.panel==panel];a=H.summarize(sub[sub.cand==mode+'_hm25']);b=H.summarize(sub[sub.cand==mode+'_'+tag]);r={c:float(b[c]-a[c])for c in ['S_state','S_phys_A','S_phys_B']}
                for ph in 'AB':r['total_'+ph]=.35*r['S_state']+.15*r['S_phys_'+ph]
                per[panel]=r
            out={'error':evaluate(target,target[mode+'_'+tag]),'score':{c:.8*np.mean([r[c]for r in per.values()])for c in r},'perpanel_score':per}
            report['feb'][mode][tag]=out;print('feb',mode,tag,json.dumps(out['score']),flush=True)
    choice=.25
    if all(report['validation']['pool50'][p]['rel_mse']<report['validation']['adaptive'][p]['rel_mse']for p in ['all','odd','even'])and all(report['feb'][mode]['pool50']['score']['total_'+ph]>0 for mode in ['serve','strict']for ph in 'AB'):choice=.5
    report['selected_weight']=choice
    (DEST/'results.json').write_text(json.dumps(report,indent=2));print('SELECTED',choice,flush=True)
if __name__=='__main__':run()
