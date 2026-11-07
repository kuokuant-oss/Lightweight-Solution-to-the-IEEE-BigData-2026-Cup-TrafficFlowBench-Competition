"""Unseen full-Feb encoder paths, isolated Torch-only process after release."""
from e_architecture import *
from encoder_reader import *
import argparse,time,resource
def run(fold,source,device_name='mps'):
    torch.set_num_threads(2);torch.set_num_interop_threads(2);assert'lightgbm'not in sys.modules
    release,manifest=require_release(['encoder_infer',source,fold]);assert device_name==release['encoder_device']
    assert(ME/'OUTCOME_RELEASED.json').exists(),'candidate inference release required'
    device=torch.device(device_name);path=ME/'models'/f'encoder_fold{fold}.pt';info=json.loads(path.with_suffix('.json').read_text())
    assert info['fit_end']==FIT_END and max(info['train_dates'])<=FIT_END and sha(path)==info['model_sha256']
    reader=EncoderReader('held',source);assert set(reader.index.date).isdisjoint(info['train_dates'])
    out=CACHE/f'held_{source}_semantics_fold{fold}.npy';assert not out.exists();start=time.time()
    model=SemanticTCN().to(device);saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict']);model.eval()
    sem=np.empty((reader.n,7),np.float32)
    with torch.no_grad():
        for begin in range(0,reader.n,1024):
            ids=np.arange(begin,min(begin+1024,reader.n));sem[ids]=model.semantics(torch.from_numpy(reader.X(ids)).to(device)).cpu().numpy()
    assert np.isfinite(sem).all()and((sem[:,:6]>=0)&(sem[:,:6]<=1)).all()and((sem[:,6]>=1)&(sem[:,6]<=7)).all();np.save(out,sem)
    dump(out.with_suffix('.json'),dict(PASS=True,source=source,fold=fold,centers=reader.n,model_sha256=sha(path),semantics_sha256=sha(out),
        held_dates=sorted(reader.index.date.unique()),fit_end=FIT_END,device=device_name,seconds=time.time()-start,threads=2,interop_threads=2,
        peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,stage='all five complete probability paths must be averaged downstream, never average semantics first'))
    print('R20_HELD_ENCODER_DONE',source,fold,round(time.time()-start,2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold',type=int,choices=range(5));p.add_argument('source',choices=['matched','raw']);p.add_argument('--device',default='mps',choices=['cpu','mps']);a=p.parse_args();run(a.fold,a.source,a.device)
