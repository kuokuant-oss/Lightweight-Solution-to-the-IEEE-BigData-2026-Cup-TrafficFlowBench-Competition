"""All fit-date rows receive held-date semantic outputs; no heldmonth inference."""
from common import *
def run():
    release,manifest=require_release(['assemble_oof']);assert not(ME/'OOF_ASSEMBLED.json').exists()
    idx=pd.read_parquet(CACHE/'fit_CENTER_INDEX.parquet');sem=np.empty((len(idx),7),np.float32);seen=np.zeros(len(idx),np.int8);models={}
    for fold in range(5):
        path=ME/'models'/f'encoder_fold{fold}_oof.npz';info=json.loads((ME/'models'/f'encoder_fold{fold}.json').read_text());assert info['fit_end']==FIT_END
        with np.load(path)as d:
            ids=d['center'];assert np.array_equal(ids,np.flatnonzero(idx.fold.to_numpy()==fold));sem[ids]=d['semantics'];seen[ids]+=1
        models[str(fold)]=dict(model_sha256=info['model_sha256'],oof_sha256=sha(path))
    assert(seen==1).all()and np.isfinite(sem).all();np.save(CACHE/'fit_OOF_semantics.npy',sem)
    base=np.load(CACHE/'fit_matched_base98.npy',mmap_mode='r');md=np.load(CACHE/'fit_meta.npz');out=CACHE/'fit_matched_E105.npy';assert not out.exists()
    x=np.lib.format.open_memmap(out,mode='w+',dtype=np.float32,shape=(len(base),105))
    for begin in range(0,len(base),20000):
        n=min(20000,len(base)-begin);x[begin:begin+n,:98]=base[begin:begin+n];x[begin:begin+n,98:]=sem[md['center'][begin:begin+n]]
        assert np.array_equal(x[begin:begin+n,:98],base[begin:begin+n],equal_nan=True)
    x.flush();x._mmap.close();base._mmap.close()
    dump(ME/'OOF_ASSEMBLED.json',dict(PASS=True,fit_end=FIT_END,center_rows=len(idx),rows=len(md['label']),no_self_fit_encoder_features=True,models=models,
        files={str(p):sha(p)for p in [out,CACHE/'fit_OOF_semantics.npy']},input_manifest_sha256=sha(ME/'INPUT_MANIFEST.json')))
    print('R20_OOF_ASSEMBLED',flush=True)
if __name__=='__main__':run()
