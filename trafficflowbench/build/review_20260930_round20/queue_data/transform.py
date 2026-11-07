"""Fixed fit-only finite preprocessing shared byte-for-byte by LGB and TabM."""
from common import *
def fit_stats(x,names):
    median=[];mean=[];std=[];counts=[]
    for j in range(x.shape[1]):
        v=np.asarray(x[:,j],np.float64);v=v[np.isfinite(v)];counts.append(len(v))
        if len(v):
            median.append(float(np.median(v)));mean.append(float(v.mean()));s=float(v.std());std.append(s if s>0 else 1.)
        else:median.append(0.);mean.append(0.);std.append(1.)
    return dict(original_features=names,median=median,finite_mean=mean,finite_std=std,finite_counts=counts,
        scope='outer fit≤2031-01-31 only',missing_flags='EVERY original column, fixed order after scaled values',clipping=None)
def transform_block(x,spec):
    v=np.asarray(x,np.float64);missing=~np.isfinite(v);fill=np.asarray(spec['median']);mean=np.asarray(spec['finite_mean']);std=np.asarray(spec['finite_std'])
    a=(np.where(missing,fill,v)-mean)/std;out=np.concatenate([a,missing.astype(np.float64)],1).astype(np.float32)
    assert np.isfinite(out).all();return out
def run(sem=False):
    suffix='E105'if sem else'base98';out_suffix='semantic210'if sem else'base196';specpath=ME/f'TRANSFORM_{out_suffix}.json';assert not specpath.exists()
    if sem:assert(ME/'OOF_ASSEMBLED.json').exists(),'fit-date OOF semantics required, never self fit'
    x=np.load(CACHE/f'fit_matched_{suffix}.npy',mmap_mode='r');names=FS+(SEMANTICS if sem else[]);assert x.shape[1]==len(names)
    spec=fit_stats(x,names);spec['fit_source_sha256']=sha(CACHE/f'fit_matched_{suffix}.npy');spec['table_features']=feature_names(sem);spec['tables']={}
    for cohort,source in [('fit','matched'),('held','matched'),('held','raw')]:
        src=CACHE/f'{cohort}_{source}_{suffix}.npy'
        if not src.exists():continue # Semantic held tables are generated only after release.
        z=np.load(src,mmap_mode='r');dst=CACHE/f'{cohort}_{source}_{out_suffix}.npy';assert not dst.exists()
        y=np.lib.format.open_memmap(dst,mode='w+',dtype=np.float32,shape=(len(z),len(names)*2));maximum=0.
        for begin in range(0,len(z),20000):
            b=transform_block(z[begin:begin+20000],spec);y[begin:begin+len(b)]=b;maximum=max(maximum,float(np.max(np.abs(b))))
        y.flush();y._mmap.close();z._mmap.close();spec['tables'][str(dst)]=dict(sha256=sha(dst),input_sha256=sha(src),shape=[len(y),len(names)*2],dtype='float32',max_abs=maximum)
    x._mmap.close();dump(specpath,spec);print('TRANSFORM_READY',out_suffix,spec['tables'],flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--semantic',action='store_true');run(p.parse_args().semantic)
