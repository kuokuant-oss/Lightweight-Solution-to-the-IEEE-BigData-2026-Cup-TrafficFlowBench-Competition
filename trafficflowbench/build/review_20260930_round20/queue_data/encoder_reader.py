"""Torch-free fit/held data reader. No February targets used for fitting."""
from common import *
class EncoderReader:
    def __init__(self,cohort='fit',source='matched'):
        assert cohort in ['fit','held']and source in ['matched','raw']
        self.pre=json.loads((ME/'PREPARED.json').read_text());assert self.pre['PASS'];self.cohort=cohort;self.source=source
        self.index=pd.read_parquet(CACHE/f'{cohort}_CENTER_INDEX.parquet');assert np.array_equal(self.index.center_row.to_numpy(),np.arange(len(self.index)))
        self.arr=[];self.ranges=[];offset=0;targets={k:[]for k in ['label','mask','switch','switch_mask','cell_weight','switch_weight']}
        for panel in PANELS:
            a=np.load(CACHE/f'{cohort}_{panel}_{source}_X.npy',mmap_mode='r');n=len(a);self.arr.append(a);self.ranges.append((offset,offset+n));offset+=n
            with np.load(CACHE/f'{cohort}_{panel}_targets.npz')as d:
                for k in targets:targets[k].append(d[k])
        self.targets={k:np.concatenate(v)for k,v in targets.items()};self.n=offset;assert offset==len(self.index)
        assert all(len(v)==self.n for v in self.targets.values())
        if cohort=='fit':assert self.index.date.max()<=FIT_END and(self.index.fold>=0).all()
        else:assert self.index.date.astype(str).between(HELD_START,HELD_END).all()and(self.index.fold==-1).all()
    def X(self,ids):
        ids=np.asarray(ids,np.int64);out=np.empty((len(ids),4,9,13),np.float32)
        for a,(lo,hi)in zip(self.arr,self.ranges):
            take=(ids>=lo)&(ids<hi)
            if take.any():out[take]=a[ids[take]-lo]
        assert np.isfinite(out).all();return out
    def fold(self,fold):
        assert self.cohort=='fit'and self.source=='matched'and fold in range(5)
        held=self.index.fold.to_numpy()==fold;tr=np.flatnonzero(~held);te=np.flatnonzero(held)
        tw=self.index.loc[~held,['window_id','panel']].drop_duplicates();counts=tw.groupby('panel').size().to_dict()
        windows={r['panel']:r['windows']for r in self.pre['records']if r['cohort']=='fit'}
        scale=self.index.panel.map({p:windows[p]/counts[p]for p in PANELS}).to_numpy(np.float64)
        w=self.targets['cell_weight']*scale;sw=self.targets['switch_weight']*scale
        assert abs((w[tr,None]*self.targets['mask'][tr]).sum()-1)<1e-12
        assert set(self.index.loc[held,'date']).isdisjoint(self.index.loc[~held,'date'])
        return tr,te,w,sw,counts
