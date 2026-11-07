"""Serve the frozen HM25 recipe on T1 official hole speed rows only."""
from hole_mixture import *
import hashlib

PARENT=ROOT/'build/deliverables/2026-09-25/submission_T1_56549247.csv'
PARENT_SHA='80fcdecd9f0d1ab09e41601b68c83801442ae4562eb3607b6286da53eca84359'


def run():
    # package edit: the parent CSV may be a regenerated state-only stand-in; its hole speeds are checked row by row below (rebuilt T1 == parent)
    print('parent sha256',hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA and 'matches the original' or 'differs (state-only stand-in)')
    for split in ['validation','private']:
        r=json.loads((DEST/f'{split}_strict_results.json').read_text())['blend25']
        for part in ['all','odd','even']:assert r[part]['relative_mse']<0
        assert r['all']['panelday_boot_ci'][1]<0
    feb=json.loads((DEST/'results.json').read_text())
    for mode in ['serve','strict']:assert feb[mode+'_blend25']['five_family_mean']['total_proxy_B']>0
    holes=pd.read_parquet(R3/'work/trans_state_rows.parquet').submission_id
    expected=set(holes);parents=[]
    for a in pd.read_csv(PARENT,chunksize=500000):
        parents.append(a[a.submission_id.isin(expected)][['submission_id','speed_kmh','flow_vph']])
    parent=pd.concat(parents,ignore_index=True).rename(columns={'speed_kmh':'parent_speed','flow_vph':'parent_flow'})
    assert len(parent)==128465 and parent.submission_id.is_unique
    output=[];report={}
    for split in ['validation','private']:
        source=R2 if split=='validation'else R3
        fit=S.load_ph([source/'work'/f'ph_{split}_{p}.parquet'for p in S.PANELS])
        official=pd.read_parquet(R3/'work'/f'off_rows_{split}.parquet');S.derive(official)
        _,p=fit_predict(fit,official,f'official_{split}')
        conditional=[lgb.Booster(model_file=str(DEST/f'official_{split}_{kind}.txt')).predict(official[S.FEATS],num_threads=3)for kind in ['free','queued']]
        official['mixture_residual']=(1-p)*conditional[0]+p*conditional[1];official['pqueue']=p
        baseline=lgb.Booster(model_file=str(R3/'work'/f'fs_speed_{split}.txt'))
        official['baseline_residual']=S.correction(baseline,official)
        keys=['panel','timestamp','link_id']
        old=pd.read_parquet(R3/'work'/f'parent_{split}_state.parquet')
        old=old[old.submission_id.isin(expected)].rename(columns={'speed_kmh':'g2_shipped_speed'})
        z=old.merge(official[keys+['window_id','cond','ch_v','mixture_residual','baseline_residual','pqueue']],on=keys,validate='one_to_one')
        z=z.merge(parent,on='submission_id',validate='one_to_one')
        assert len(z)==len(old) and np.isfinite(z.mixture_residual).all()
        rebuilt=np.clip(z.g2_shipped_speed+z.baseline_residual,S.SPEED_FLOOR,S.SPEED_CAP)
        rounded=np.array([float(f'{v:.3f}')for v in rebuilt])
        assert np.max(np.abs(rounded-z.parent_speed))<1e-9
        mixture=np.clip(z.g2_shipped_speed+z.mixture_residual,S.SPEED_FLOOR,S.SPEED_CAP)
        z['speed_kmh']=.75*z.parent_speed+.25*mixture;z['split']=split
        dv=z.speed_kmh-z.parent_speed
        held=pd.read_parquet(DEST/f'{split}_strict_predictions.parquet')
        hdelta=held.blend25-held.base
        ratio=float(np.quantile(np.abs(dv),.99)/np.quantile(np.abs(hdelta),.99))
        assert ratio<=1.5,(split,ratio)
        report[split]={'fit_cells':len(fit),'fit_windows':int(fit.wid.nunique()),'official_cells':len(official),'key_rows':len(z),
            'T1_rebuild_max_abs_after_rounding':float(np.max(np.abs(rounded-z.parent_speed))),
            'shipped_vs_recomputed_G2_max_abs':float(np.max(np.abs(z.g2_shipped_speed-z.ch_v))),
            'speed_change':S.desc(dv),'p99_change_ratio_official_to_strict_ldo':ratio,
            'perpanel_change':z.assign(delta=dv).groupby('panel').delta.agg(['size','mean','min','max']).to_dict('index')}
        output.append(z[['submission_id','panel','split','timestamp','link_id','window_id','cond','parent_speed','parent_flow','speed_kmh','pqueue']])
        print(split,json.dumps(report[split]),flush=True);del fit,official,z;gc.collect()
    out=pd.concat(output,ignore_index=True).sort_values('submission_id')
    assert len(out)==128465 and out.submission_id.is_unique and set(out.submission_id)==expected
    assert np.isfinite(out.speed_kmh).all() and out.speed_kmh.between(6,120.75).all()
    out.to_parquet(DEST/'official_state_updates.parquet',index=False)
    (DEST/'serving_results.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':run()
