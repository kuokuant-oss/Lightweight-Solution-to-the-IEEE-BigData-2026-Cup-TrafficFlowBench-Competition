"""Fullphysical L0/LE probabilities, five semantic paths then seed mean."""
from common import *
from transform import transform_block
import lightgbm as lgb,argparse,time,resource
def run(arm,source):
    release,manifest=require_release(['gbdt_infer',arm,source]);assert(ME/'OUTCOME_RELEASED.json').exists()
    assert'torch'not in sys.modules;start=time.time();sem=arm=='LE';suffix='semantic210'if sem else'base196'
    spec=json.loads((ME/f'TRANSFORM_{suffix}.json').read_text());md=np.load(CACHE/'held_meta.npz')
    base=np.load(CACHE/f'held_{source}_base98.npy',mmap_mode='r');models=[]
    for seed in range(3):
        path=ME/'models'/f'{arm}_geom_seed{seed}.txt';info=json.loads(path.with_suffix('.json').read_text());assert info['fit_end']==FIT_END and sha(path)==info['model_sha256']
        model=lgb.Booster(model_file=str(path));assert model.feature_name()==feature_names(sem);models.append(model)
    pred=np.zeros((len(base),3),np.float64)
    for fold in(range(5)if sem else[0]):
        path=CACHE/f'held_{source}_semantics_fold{fold}.npy'
        enc=np.load(path,mmap_mode='r')if sem else None
        if sem:assert json.loads(path.with_suffix('.json').read_text())['semantics_sha256']==sha(path)
        for begin in range(0,len(base),20000):
            end=min(begin+20000,len(base));v=base[begin:end]
            if sem:v=np.concatenate([v,enc[md['center'][begin:end]]],1) # preserves original mixed query values
            x=transform_block(v,spec)
            for seed,model in enumerate(models):pred[begin:end,seed]+=model.predict(x,num_threads=2)/(5 if sem else 1)
        if enc is not None:enc._mmap.close()
    assert np.isfinite(pred).all()and((pred>0)&(pred<1)).all();out=CACHE/f'held_{source}_{arm}_prob.npz';assert not out.exists()
    np.savez(out,seed_prob=pred,mean_prob=pred.mean(1))
    dump(out.with_suffix('.json'),dict(PASS=True,arm=arm,source=source,rows=len(pred),seeds=[0,1,2],encoder_paths=5 if sem else 1,
        probability_ensemble='same downstream seed model five encoder paths probability mean then3seed probability mean; no decoder yet',
        model_hashes=[sha(ME/'models'/f'{arm}_geom_seed{s}.txt')for s in range(3)],transform_sha256=sha(ME/f'TRANSFORM_{suffix}.json'),prob_sha256=sha(out),
        fit_end=FIT_END,seconds=time.time()-start,threads=2,peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9))
    print('R20_HELD_GBDT_DONE',arm,source,round(time.time()-start,2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['L0','LE']);p.add_argument('source',choices=['matched','raw']);a=p.parse_args();run(a.arm,a.source)
