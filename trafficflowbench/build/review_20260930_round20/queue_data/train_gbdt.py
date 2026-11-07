"""L0/LE shared finite tables, original geom3 recipe, no held score reads."""
from common import *
import lightgbm as lgb,time,gc,resource,argparse
def train(arm,seed):
    assert arm in ['L0','LE']and seed in range(3)and'torch'not in sys.modules
    release,manifest=require_release([arm,seed]);sem=arm=='LE';suffix='semantic210'if sem else'base196'
    spec=json.loads((ME/f'TRANSFORM_{suffix}.json').read_text());path=CACHE/f'fit_matched_{suffix}.npy';assert sha(path)==spec['tables'][str(path)]['sha256']
    if sem:assert json.loads((ME/'OOF_ASSEMBLED.json').read_text())['no_self_fit_encoder_features']
    out=ME/'models'/f'{arm}_geom_seed{seed}.txt';assert not out.exists();x=np.load(path,mmap_mode='r');md=np.load(CACHE/'fit_meta.npz')
    assert len(x)==manifest['fit_rows']and abs(md['weight'].mean()-1)<1e-12
    config=manifest['gbdt_config'];P=dict(config['params'],seed=20260922+seed,num_threads=2);nr=config['rounds'][str(seed)];features=feature_names(sem)
    start=time.time();ds=lgb.Dataset(x,label=md['label'],weight=md['weight'],feature_name=features,free_raw_data=True,params=P);ds.construct();x._mmap.close();del x,md;gc.collect()
    def cb(env):
        if(env.iteration+1)%100==0:print('R20_GBDT_ITER',arm,seed,env.iteration+1,round(time.time()-start,1),flush=True)
    m=lgb.train(P,ds,num_boost_round=nr,callbacks=[cb]);assert m.num_trees()==nr and m.feature_name()==features;m.save_model(str(out))
    dump(out.with_suffix('.json'),dict(arm=arm,seed=seed,fit_end=FIT_END,rounds=nr,params=P,features=features,rows=manifest['fit_rows'],model_sha256=sha(out),
        transform_sha256=sha(ME/f'TRANSFORM_{suffix}.json'),input_manifest_sha256=sha(ME/'INPUT_MANIFEST.json'),seconds=time.time()-start,threads=2,peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9))
    print('R20_GBDT_DONE',arm,seed,round(time.time()-start,1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['L0','LE']);p.add_argument('seed',type=int,choices=range(3));a=p.parse_args();train(a.arm,a.seed)
