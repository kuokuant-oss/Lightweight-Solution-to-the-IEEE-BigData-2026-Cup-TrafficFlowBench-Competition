"""Fixed R19 semantic architecture, NEW Jan-only fivefold models, Torch-only."""
from e_architecture import *
from encoder_reader import *
import argparse,time,resource
def train(fold,device_name='mps'):
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(20260930);np.random.seed(20260930)
    assert 'lightgbm'not in sys.modules
    release,manifest=require_release(['encoder',fold]);assert device_name==release['encoder_device'];device=torch.device(device_name)
    config=manifest['encoder_config'];assert config['epochs']==10 and config['seed']==20260930
    out=ME/'models'/f'encoder_fold{fold}.pt';assert not out.exists()
    for path,h in manifest['encoder_input_hashes'].items():assert sha(path)==h,path
    reader=EncoderReader();tr,te,w,sw,counts=reader.fold(fold);assert reader.index.iloc[tr].date.max()<=FIT_END and reader.index.iloc[te].date.max()<=FIT_END
    model=SemanticTCN().to(device);initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']);rng=np.random.default_rng(20260930);B=config['batch_centers'];start=time.time();logs=[]
    for epoch in range(10):
        model.train();order=rng.permutation(tr);total=0.;steps=0
        for begin in range(0,len(order),B):
            ids=order[begin:begin+B];x=torch.from_numpy(reader.X(ids)).to(device)
            y=torch.from_numpy(reader.targets['label'][ids]).to(device);mask=torch.from_numpy(reader.targets['mask'][ids].astype(np.float32)).to(device)
            switch=torch.from_numpy(reader.targets['switch'][ids]).to(device);known=torch.from_numpy(reader.targets['switch_mask'][ids].astype(np.float32)).to(device)
            ww=torch.from_numpy(w[ids].astype(np.float32)).to(device);ws=torch.from_numpy(sw[ids].astype(np.float32)).to(device)
            q,s=model(x);loss=((F.binary_cross_entropy_with_logits(q,y,reduction='none')*mask*ww[:,None]).sum()+.2*(F.cross_entropy(s,switch,reduction='none')*known*ws).sum())*(len(tr)/len(ids))
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip_norm']);opt.step()
            total+=float(loss.detach().cpu())*len(ids);steps+=1
        rec=dict(epoch=epoch+1,steps=steps,loss=total/len(order),seconds=time.time()-start);logs.append(rec);print('R20_ENCODER_EPOCH',fold,json.dumps(rec),flush=True)
    model.eval();held=np.empty((len(te),7),np.float32)
    with torch.no_grad():
        for begin in range(0,len(te),B):
            ids=te[begin:begin+B];held[begin:begin+len(ids)]=model.semantics(torch.from_numpy(reader.X(ids)).to(device)).cpu().numpy()
    assert np.isfinite(held).all()and((held[:,:6]>=0)&(held[:,:6]<=1)).all()and((held[:,6]>=1)&(held[:,6]<=7)).all()
    state={k:v.detach().cpu()for k,v in model.state_dict().items()};changed={k:bool(not torch.equal(v,initial[k]))for k,v in state.items()};assert all(changed.values())
    torch.save(dict(state_dict=state,fold=fold,semantic_fields=SEMANTICS,input_manifest_sha256=sha(ME/'INPUT_MANIFEST.json')),out)
    np.savez(ME/'models'/f'encoder_fold{fold}_oof.npz',center=te,semantics=held)
    info=dict(fold=fold,device=device_name,seed=20260930,epochs=10,fit_end=FIT_END,train_centers=len(tr),held_centers=len(te),
        train_dates=sorted(reader.index.iloc[tr].date.unique()),held_dates=sorted(reader.index.iloc[te].date.unique()),train_windows_by_panel=counts,
        semantic_fields=SEMANTICS,optimizer_steps=sum(r['steps']for r in logs),parameter_count=sum(p.numel()for p in model.parameters()),all_parameter_tensors_updated=changed,
        model_sha256=sha(out),oof_sha256=sha(ME/'models'/f'encoder_fold{fold}_oof.npz'),input_manifest_sha256=sha(ME/'INPUT_MANIFEST.json'),
        seconds=time.time()-start,threads=2,interop_threads=2,peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,logs=logs)
    dump(out.with_suffix('.json'),info);print('R20_ENCODER_DONE',fold,info['seconds'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold',type=int,choices=range(5));p.add_argument('--device',choices=['cpu','mps'],default='mps');a=p.parse_args();train(a.fold,a.device)
