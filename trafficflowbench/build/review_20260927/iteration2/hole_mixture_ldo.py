"""Validation-world odd/even confirmation of fixed hole-mixture recipe."""
# Runs in pipeline/task1.sh step 7 as `--split <validation|private> --strict`.
# Uses the split's own pseudo holes (90-minute gaps cut into observed cells of
# the released month) with odd/even calendar-day halves: fit on one half,
# predict the other. Compares the T1 speed correction ('base') with the HM25
# mixture ('new') and the 25 % blend ('blend25' = .75 base + .25 new).
# Outputs: hole_mixture/<split>[_strict]_predictions.parquet and
# hole_mixture/<split>[_strict]_results.json; hole_mixture_serve.py reads
# them as gates and as the reference size of the speed change.
# Usage: hole_mixture_ldo.py [--split validation|private] [--strict]
#
# The star import brings fit_predict, bootstrap, numpy/pandas/LightGBM,
# S (serve_lib) and the R2/R3 folders.
from hole_mixture import *

def run(split='validation',strict=False):
    """Runs the odd/even check of the HM25 recipe on one split.

    Args:
      split: 'validation' (pseudo holes built in round 2) or 'private'
        (pseudo holes built in round 3).
      strict: When True, the month statistics features (s_* columns) of the
        fit and test rows come from the fitting half's days only
        (verify/work/strict_<split>_<panel>.parquet), so no test-half
        observation enters any feature.
    """
    source=R2 if split=='validation' else R3
    d=S.load_ph([source/'work'/f'ph_{split}_{p}.parquet'for p in S.PANELS])
    assert np.isfinite(d.y_v).all();d['is_target']=True
    tag=split+('_strict' if strict else '')
    # Month statistics features that strict mode takes from one half only.
    raw_s=['s_v','s_ql','s_qf','s_n','s_vq','s_tq','s_bqf','s_v_pre','s_qf_pre','s_bqf_pre','s_v_post','s_qf_post','s_bqf_post']
    if strict:
        st=pd.concat([pd.read_parquet(source/'verify/work'/f'strict_{split}_{p}.parquet')for p in S.PANELS],ignore_index=True)
        assert len(st)==len(d)
        for key in ['panel','li','origin','step_i']:
            assert np.array_equal(st[key].to_numpy(),d[key].to_numpy()),key
    d['base']=np.nan;d['new']=np.nan;d['pqueue']=np.nan
    for odd in [True,False]:
        tr=(d.day%2==1).to_numpy()==odd;te=~tr
        x=d.copy()
        if strict:
            for key in raw_s:x[key]=st[f'{key}_f'+('A' if odd else 'B')].to_numpy()
            S.derive(x)
        # T1 baseline and the mixture are fit on the same half and predict the
        # other half.
        base=S.fit(x.loc[tr]);d.loc[te,'base']=S.apply(base,x.loc[te]);del base;gc.collect()
        v,p=fit_predict(x.loc[tr],x.loc[te],f'{tag}_fitodd{int(odd)}');d.loc[te,'new']=v;d.loc[te,'pqueue']=p
        del x;gc.collect()
    d['blend25']=.75*d.base+.25*d.new
    result={}
    for nm in ['new','blend25']:
        result[nm]={}
        # Subsets: all cells, each day parity, days 1-12, onset windows and
        # ongoing windows.
        for name,mask in [('all',np.ones(len(d),bool)),('odd',d.day%2==1),('even',d.day%2==0),('early12',d.day<=12),('onset',d.cond=='queue_onset'),('ongoing',d.cond=='queue_ongoing')]:
            r=bootstrap(d.loc[mask],'base',nm);result[nm][name]=r;print(nm,name,json.dumps(r),flush=True)
    d[['panel','date','origin','li','step_i','cond','y_v','ch_v','base','new','blend25','pqueue']].to_parquet(DEST/f'{tag}_predictions.parquet',index=False)
    (DEST/f'{tag}_results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','private'],default='validation');p.add_argument('--strict',action='store_true');args=p.parse_args()
    run(args.split,args.strict)
