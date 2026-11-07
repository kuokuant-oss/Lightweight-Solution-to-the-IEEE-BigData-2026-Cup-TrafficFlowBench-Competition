"""Causal full-production-feature reconstruction and direct frozen-T1 replay."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '3'
import sys, json, gc, importlib.util, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CT = ROOT / 'build/review_20260924_ctm'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CT / 'f2_analysis/bias'))
import lib as L
from src.tfb import release as R, grid, task2feat as TF, task2origin as OG, task2build as TB
sys.path.insert(0, str(CT / 'ruler/feat'))
import geom as G
spec = importlib.util.spec_from_file_location('direct_rec', CT / 'rep2/build_recency.py')
REC = importlib.util.module_from_spec(spec); spec.loader.exec_module(REC)
PAN = list(L.FAM)
FS = sorted(set().union(*(L.booster(p).feature_name() for p in L.SR6 + [L.AMOD / f'og_{n}.txt' for n in L.A_NAMES])))
KEY = ['window_id', 'link_id', 'step']
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')

class Context:
    def __init__(self, panel, split='validation'):
        self.panel, self.split = panel, split
        order = pd.read_parquet(R.truth_files(panel, 'train')[0], columns=['link_id', 'milepost']).drop_duplicates().sort_values('milepost')
        self.links = order.link_id.astype(str).tolist()
        self.mp = order.milepost.to_numpy(float)
        self.li = {l:i for i,l in enumerate(self.links)}
        self.static = TF.static_table(panel, self.links, grid, R)
        self.vc = self.static.v_cut.to_numpy(float)
        self.attach = R.network(panel, 'ramp_attachment_map')
        for c in ['ramp_link_id', 'nearest_mainline_link_id']:
            self.attach[c] = self.attach[c].astype(str)
        fd = R.network(panel, 'fd_parameters').set_index('link_id')
        fd.index = fd.index.astype(str)
        self.length = fd.length_km.reindex(self.links).fillna(.3).to_numpy(float)
        self.bid = np.array([L.BID[panel][l] for l in self.links])
        self.ramps = {}

    def day(self, date, path=None):
        path = path or REC.day_index(self.panel, self.split)[date]
        d = pd.read_parquet(path); d['link_id'] = d.link_id.astype(str)
        d['_t'] = pd.to_datetime(d.timestamp, utc=True)
        rr = d.link_id.map(self.li).to_numpy(); cc = (d._t.dt.hour * 12 + d._t.dt.minute // 5).to_numpy()
        out = {}
        for name in ['speed_kmh', 'flow_vph', 'occupancy', 'is_score_eligible']:
            x = np.full((len(self.links), 288), np.nan); x[rr,cc] = d[name].to_numpy()
            out[name] = x.astype(bool) if name == 'is_score_eligible' else x
        return out

    def ramp(self, date):
        if date not in self.ramps:
            d = TB.ramp_day(self.panel, 'synthetic_mainline_' + date.replace('-', '_') + '.parquet', split=self.split).copy()
            d['ramp_link_id'] = d.ramp_link_id.astype(str)
            ts = pd.to_datetime(d.timestamp, utc=True)
            d['_t'] = ts.dt.hour * 12 + ts.dt.minute // 5
            self.ramps[date] = {int(t):dict(zip(s.ramp_link_id, s.flow_vph)) for t,s in d.groupby('_t')}
        return self.ramps[date]

    def features(self, date, t, day, wid, K=5, selector=None, origin_day=None):
        # Truncate first. Spatial interpolation sees only the twelve history stamps.
        v,q,o,e = [day[c][:,t-12:t].copy() for c in ['speed_kmh','flow_vph','occupancy','is_score_eligible']]
        od = origin_day if origin_day is not None else day
        vt,qt,ot = [od[c][:,t].copy() for c in ['speed_kmh','flow_vph','occupancy']]
        ramp = self.ramp(date); prev, now = ramp.get(t-1, {}), ramp.get(t,{})
        rvec = TF.ramp_vectors(self.panel,self.links,self.mp,prev,self.attach,oblique=True,sgn=1)
        def rv(values, kind=None, missing=np.nan):
            ans = np.full(len(self.links), missing)
            for a in self.attach.itertuples():
                if kind is not None and (str(a.ramp_type).upper() in ('OR','ON')) != (kind == 'on'): continue
                j = self.li.get(str(a.nearest_mainline_link_id)); x = values.get(str(a.ramp_link_id))
                if j is not None and x is not None and np.isfinite(x): ans[j] = (ans[j] if np.isfinite(ans[j]) else 0.) + x
            return ans
        origin = OG.origin_columns(self.links,self.links,vt,qt,ot,self.vc,v[:,-1],q[:,-1],o[:,-1],rv(now),rv(prev),1)
        f = TF.build_window_features(v,q,o,e,self.links,self.mp,self.static,rvec,(0,12),np.zeros((len(self.links),288)),t,288,dispersion=True,sgn=1,clock=True,origin=origin)
        for kind in ('on','off'):
            attached = set(self.attach.loc[self.attach.ramp_type.astype(str).str.upper().isin(['OR','ON']) == (kind=='on'),'nearest_mainline_link_id'])
            for lag,vals in [(0,now),(1,prev)]:
                a = rv(vals,kind); a[[l not in attached for l in self.links]] = 0.
                f[f'_{kind}{lag}'] = np.tile(a,6).astype('float32')
        f['origin_on_ramp_T']=f._on0; f['origin_off_ramp_T']=f._off0
        f['origin_net_ramp_T']=f._on0-f._off0
        f['origin_dnet_ramp']=(f._on0-f._off0)-(f._on1-f._off1)
        f=f.drop(columns=['_on0','_off0','_on1','_off1'])
        geo=G.step_frame(self.links,G.geom_features(v,q,e,self.vc,self.length))
        for c in geo.select_dtypes('float64'): geo[c]=geo[c].astype('float32')
        f=f.merge(geo,on=['link_id','step'],validate='one_to_one')
        rec=REC.feats_for(self.panel,self.split,date,t,self.links,K)
        for c in ['rec_s','rec_any']: rec[c]=rec[c].astype('float32')
        f=f.merge(rec,on=['link_id','step'],validate='one_to_one')
        if selector is None:
            V=day['speed_kmh'][:,:t+1]; E=day['is_score_eligible'][:,:t+1]
            qo=(V<=self.vc[:,None])&E&np.isfinite(V)
            k=0
            for h in range(1,13):
                at=t-h
                if at<12 or E[:,at-12:at].mean()<.7 or not qo[:,at-1].any() or not qo[:,at-2].any(): break
                k+=1
            selector=(float(k>0),float(k))
        f['prev_rej']=np.float32(selector[0]);f['n_rej']=np.float32(selector[1])
        f['window_id']=wid;f['panel']=self.panel;f['date']=date
        f['timestamp']=pd.Timestamp(date,tz='UTC')+pd.to_timedelta((t+f.step)*5,unit='m')
        f['bid']=f.link_id.map(L.BID[self.panel]);f['panel_code']=PAN.index(self.panel)
        f['history_observed_fraction']=np.tile(np.isfinite(v).mean(1),6)
        assert not set(FS)-set(f)
        return f

def predict(f):
    f=f.copy();f['pA']=L.predA(f);f['pN']=L.predN(f)
    f['p_mix']=.75*f.pN+.25*f.pA
    return f

def parity():
    ref=L.load_cache('v2'); sels=pd.read_parquet(CT/'f2_analysis/bias/pvsel_v2.parquet').set_index('window_id')
    wi=pd.read_parquet(L.RD['v2']/'windows_validation.parquet').set_index('window_id')
    result=[]
    for panel in PAN:
        c=Context(panel); s=ref[ref.panel==panel]
        wids=s.window_id.unique()[:3]
        for wid in wids:
            w=wi.loc[wid];t=int(w.t);raw=c.day(w.date)
            day=c.day(w.date,CT/f'ruler/fakeval/days/{panel}_{w.date}.parquet')
            selector=tuple(sels.loc[wid,['prev_rej_srv','n_rej_srv']])
            f=c.features(w.date,t,day,wid,REC.kdraw(wid),selector,raw)
            r=s[s.window_id==wid].copy();r['prev_rej']=np.float32(selector[0]);r['n_rej']=np.float32(selector[1])
            f=f.set_index(KEY).reindex(pd.MultiIndex.from_frame(r[KEY])).reset_index()
            diff={}
            for col in FS:
                x,y=f[col].to_numpy(float),r[col].to_numpy(float)
                bad=~np.isclose(x,y,rtol=1e-6,atol=1e-6,equal_nan=True)
                if bad.any():diff[col]={'n':int(bad.sum()),'max':float(np.nanmax(np.abs(x-y)))}
            pp=predict(f);expected=.75*L.predN(r)+.25*L.predA(r)
            err=float(np.max(np.abs(pp.p_mix-expected)))
            # Every accessible input after t is poisoned; features must stay identical.
            poisoned={k:v.copy() for k,v in day.items()}
            for k in poisoned:poisoned[k][:,t+1:]=True if k=='is_score_eligible' else 123456.
            pf=c.features(w.date,t,poisoned,wid,REC.kdraw(wid),selector,raw)
            pf=pf.set_index(KEY).reindex(pd.MultiIndex.from_frame(r[KEY])).reset_index()
            assert np.array_equal(f[FS].to_numpy(),pf[FS].to_numpy(),equal_nan=True)
            result.append(dict(panel=panel,window_id=wid,diff=diff,prob_max=err,poison=True))
            print('parity',panel,wid,diff,err,flush=True)
        del c;REC.day_q.cache_clear();gc.collect()
    dump(HERE/'parity.json',result)
    assert max(r['prob_max'] for r in result)<1e-6

def build(maxday=12, schedule='fixed'):
    folder=HERE/('dense' if schedule=='fixed' else 'dense_balanced')
    folder.mkdir(exist_ok=True)
    for panel in PAN:
        dest=folder/f'{panel}.parquet'
        if dest.exists():continue
        c=Context(panel);parts=[]
        for date in sorted(REC.day_index(panel,'validation')):
            if int(date[-2:])>maxday:continue
            day=c.day(date);V=day['speed_kmh'];E=day['is_score_eligible'];obs=np.isfinite(V)&E;qo=(V<=c.vc[:,None])&obs
            rows=[]
            phase=1 if schedule=='fixed' else (int(date[-2:])+PAN.index(panel))%6
            for t in range(12+phase,282,6):
                if E[:,t-12:t].mean()<.7 or not qo[:,t-1].any() or not qo[:,t-2].any():continue
                wid=f'{panel}|{date}|{t}';f=c.features(date,t,day,wid)
                mask=obs[:,t+1:t+7].T.reshape(-1);y=qo[:,t+1:t+7].T.reshape(-1)
                persistence=np.tile(qo[:,t],6);un=((persistence|y)&mask).sum()
                selected=bool(y.any() and un>0 and (persistence&y&mask).sum()/un<=.9)
                f['label']=y.astype('int8');f['mask']=mask;f['selected']=selected
                f['origin']=pd.Timestamp(date,tz='UTC')+pd.Timedelta(minutes=5*t);f['mature']=f.origin+pd.Timedelta(minutes=30)
                rows.append(f)
            if rows:
                d=pd.concat(rows,ignore_index=True)
                # Independent link predictors: discard unlabeled replay rows before
                # inference; no spatial pooling uses this target-only subset.
                d=predict(d[d['mask']].copy()) if d['mask'].any() else d.assign(pA=np.nan,pN=np.nan,p_mix=np.nan).iloc[:0]
                keep=list(dict.fromkeys(FS+['window_id','panel','date','link_id','step','timestamp','bid','panel_code','history_observed_fraction','label','selected','origin','mature','pA','pN','p_mix']))
                parts.append(d[keep]);print('build',panel,date,len(rows),len(d),flush=True)
        d=pd.concat(parts,ignore_index=True);d.to_parquet(dest,index=False)
        dump(dest.with_suffix('.json'),dict(rows=len(d),windows=d.window_id.nunique(),selected=d[d.selected].window_id.nunique(),max_label=str(d.timestamp.max()),features=FS))
        del parts,d,c;REC.day_q.cache_clear();gc.collect()

def official_parity():
    s=pd.read_parquet(CT/'serve/serve_real_validation.parquet')
    s=s[s.condition.eq('queue_ongoing')].merge(pd.read_parquet(CT/'ruler/feat/geom_serve_real_validation.parquet'),on=KEY)
    s=s.merge(pd.read_parquet(CT/'dense/sel2_validation.parquet')[['window_id','prev_rej','n_rej']],on='window_id')
    s=s.merge(pd.read_parquet(CT/'dense/rec_realvali_queue_ongoing.parquet'),on=KEY)
    for col in ['prev_rej','n_rej']:s[col]=s[col].astype('float32')
    audits=[]
    for panel in PAN:
        c=Context(panel)
        wi=pd.read_csv(R.RELEASE/f'task2/{panel}/validation/window_index.csv').set_index('window_id')
        hist=pd.read_parquet(R.RELEASE/f'task2/{panel}/validation/window_history.parquet')
        hist['link_id']=hist.link_id.astype(str)
        for wid,r in s[s.panel==panel].groupby('window_id',sort=False):
            T=pd.Timestamp(wi.loc[wid,'forecast_origin']);date=T.strftime('%Y-%m-%d');t=T.hour*12+T.minute//5
            raw=c.day(date);day={k:v.copy() for k,v in raw.items()};h=hist[hist.window_id==wid]
            tt=pd.to_datetime(h.timestamp,utc=True);rr=h.link_id.map(c.li).to_numpy();cc=(tt.dt.hour*12+tt.dt.minute//5).to_numpy()
            assert cc.max()<t
            for col in day:day[col][rr,cc]=h[col].to_numpy()
            selector=(r.prev_rej.iloc[0],r.n_rej.iloc[0])
            f=c.features(date,t,day,wid,5,selector,raw).set_index(KEY).reindex(pd.MultiIndex.from_frame(r[KEY])).reset_index()
            diffs={col:int((~np.isclose(f[col],r[col],rtol=1e-6,atol=1e-6,equal_nan=True)).sum()) for col in FS}
            diffs={k:v for k,v in diffs.items() if v}
            rebuilt=predict(f);expected=.75*L.predN(r)+.25*L.predA(r)
            err=float(np.max(np.abs(rebuilt.p_mix-expected)))
            audits.append(dict(window_id=wid,feature_diffs=diffs,probability_max_abs=err))
            print('official parity',wid,diffs,err,flush=True)
        del c;REC.day_q.cache_clear();gc.collect()
    dump(HERE/'official_parity.json',audits)
    assert max(a['probability_max_abs'] for a in audits)<1e-6

def guarded(z,p):
    d=L.block_mean(z,np.asarray(p))>=.5
    empty=pd.Series(d).groupby(z.window_id.to_numpy()).transform('sum').to_numpy()==0
    return np.where(empty,L.block_mean(z,z.pA)>=.5,d)

def score_fast(S,z,decision):
    """Exact keyed observed-cell IoU, vectorized over the requested windows."""
    lab,meta=S.labels()
    p=z[['window_id','timestamp','link_id']].copy()
    p['timestamp']=pd.to_datetime(p.timestamp,utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    p['link_id']=p.link_id.astype(str);p['prediction']=np.asarray(decision,bool)
    q=lab[lab.window_id.isin(p.window_id)].merge(p,on=['window_id','timestamp','link_id'],how='left',validate='one_to_one')
    assert q.prediction.notna().all() and len(q)==len(p)
    a=q.prediction.to_numpy(bool);b=q.label.to_numpy(bool);mask=q['mask'].to_numpy(bool)
    q['inter']=(a&b&mask).astype(int);q['union']=((a|b)&mask).astype(int)
    q['n_true']=(b&mask).astype(int);q['n_pred']=(a&mask).astype(int)
    g=q.groupby('window_id',sort=False)[['inter','union','n_true','n_pred']].sum().reset_index()
    g['iou']=np.where(g.union>0,g.inter/g.union,1.)
    return g.merge(meta,on='window_id',validate='one_to_one')

def bootstrap(a,b,n=1000,seed=0):
    """Same paired calendar-date resampling as L.boot, without pandas per draw."""
    m=a[['window_id','date','panel','iou']].merge(b[['window_id','iou']],on='window_id',suffixes=('_a','_b'),validate='one_to_one')
    m['d']=m.iou_b-m.iou_a;dates=m.date.unique();panels=sorted(m.panel.unique())
    di=pd.Index(dates).get_indexer(m.date);pi=pd.Index(panels).get_indexer(m.panel)
    sums=np.zeros((len(panels),len(dates)));nums=sums.copy()
    np.add.at(sums,(pi,di),m.d.to_numpy());np.add.at(nums,(pi,di),1.)
    rng=np.random.default_rng(seed);draw=rng.choice(np.arange(len(dates)),size=(n,len(dates)))
    counts=np.eye(len(dates))[draw].sum(1)
    den=counts@nums.T;pm=np.where(den>0,(counts@sums.T)/np.maximum(den,1),np.nan)
    fams=sorted({L.FAM[p] for p in panels})
    vals=np.stack([np.nanmean(pm[:,[L.FAM[p]==fam for p in panels]],axis=1) for fam in fams],axis=1)
    lo,hi=np.percentile(np.nanmean(vals,axis=1),[2.5,97.5])
    return dict(delta=L.agg_w(m,'d'),lo=float(lo),hi=float(hi),n=len(m),better=int((m.d>1e-9).sum()),worse=int((m.d<-1e-9).sum()))

def fill_history_fraction(d):
    if 'history_observed_fraction' not in d:d['history_observed_fraction']=np.nan
    for panel,ii in d.groupby('panel',sort=False).groups.items():
        if d.loc[ii,'history_observed_fraction'].notna().all():continue
        c=Context(panel)
        for date,jj in d.loc[ii].groupby('date',sort=False).groups.items():
            day=c.day(date);obs=np.isfinite(day['speed_kmh'])
            cum=np.pad(np.cumsum(obs,axis=1),((0,0),(1,0)))
            origin=pd.to_datetime(d.loc[jj,'origin'],utc=True)
            tt=(origin.dt.hour*12+origin.dt.minute//5).to_numpy()
            rr=d.loc[jj,'link_id'].map(c.li).to_numpy()
            d.loc[jj,'history_observed_fraction']=(cum[rr,tt]-cum[rr,tt-12])/12
        del c
    assert d.history_observed_fraction.between(0,1).all()

def paired_selection(d):
    selected={}
    for panel,s in d.groupby('panel',sort=False):
        c=Context(panel)
        for date,g in s.groupby('date',sort=False):
            day=c.day(date);V=day['speed_kmh'];E=day['is_score_eligible']
            obs=np.isfinite(V)&E;q=(V<=c.vc[:,None])&obs
            for w in g[['window_id','origin']].drop_duplicates('window_id').itertuples():
                T=pd.Timestamp(w.origin);t=T.hour*12+T.minute//5
                y=q[:,t+1:t+7];p=np.repeat(q[:,t,None],6,axis=1)
                mm=obs[:,t+1:t+7]&obs[:,t,None];u=((p|y)&mm).sum()
                selected[w.window_id]=bool(y.any() and u>0 and ((p&y)&mm).sum()/u<=.9)
        del c
    d['paired_selected']=d.window_id.map(selected).astype(bool)

def query(rv='v2'):
    z=L.load_cache(rv);z=z[pd.to_datetime(z.date).dt.day<=12].reset_index(drop=True)
    sels=pd.read_parquet(CT/f'f2_analysis/bias/pvsel_{rv}.parquet').set_index('window_id')
    z['prev_rej']=z.window_id.map(sels.prev_rej_srv).astype('float32');z['n_rej']=z.window_id.map(sels.n_rej_srv).astype('float32')
    z=predict(z);z['p_canonical']=z.p_mix.copy()
    wi=pd.read_parquet(L.RD[rv]/'windows_validation.parquet').set_index('window_id')
    pieces=[]
    for wid,g in z.groupby('window_id',sort=False):
        w=wi.loc[wid];p=g.panel.iloc[0];r=REC.feats_for(p,'validation',w.date,int(w.t),g.link_id.unique().tolist(),5)
        r['window_id']=wid;pieces.append(r)
    z=z.drop(columns=['rec_s','rec_any','rec_n']).merge(pd.concat(pieces),on=KEY,validate='one_to_one')
    for col in ['rec_s','rec_any']:z[col]=z[col].astype('float32')
    z=predict(z);z['panel_code']=z.panel.map({p:i for i,p in enumerate(PAN)})
    z['origin']=z.window_id.map(lambda w:pd.Timestamp(wi.loc[w,'date'],tz='UTC')+pd.Timedelta(minutes=5*int(wi.loc[w,'t'])))
    z['cutoff']=z.origin.dt.floor('6h')
    return z

def run(rv='v2', schedule='fixed', immediate=False):
    out=HERE if schedule=='fixed' else HERE/'balanced'
    if immediate:out=out/'immediate'
    out.mkdir(exist_ok=True)
    folder=HERE/('dense' if schedule=='fixed' else 'dense_balanced')
    qpath=HERE/f'{rv}_query.parquet'
    z=pd.read_parquet(qpath) if qpath.exists() else query(rv)
    if immediate:z['cutoff']=z.origin
    d=pd.concat([pd.read_parquet(folder/f'{p}.parquet') for p in PAN],ignore_index=True)
    fill_history_fraction(d)
    paired_selection(d)
    dump(out/'replay_manifest.json',dict(rows=len(d),windows=int(d.window_id.nunique()),selected_windows=int(d[d.selected].window_id.nunique()),paired_windows=int(d[d.paired_selected].window_id.nunique()),complete_rows=int((d.selected&d.history_observed_fraction.eq(1)).sum())))
    quality=d.groupby('panel').agg(rows=('label','size'),windows=('window_id','nunique'),missing_history_end=('speed_origin',lambda x:x.isna().mean()),complete_history_share=('history_observed_fraction',lambda x:(x==1).mean()))
    quality.to_csv(out/'replay_quality.csv')
    for f in (z,d):
        f['logit_mix']=logit(np.clip(f.p_mix,1e-5,1-1e-5));f['logit_A']=logit(np.clip(f.pA,1e-5,1-1e-5))
    features=FS+['logit_mix','logit_A','panel_code'];audit=[]
    for arm in (['paired'] if immediate else ['selected','all','complete','paired']):
        z[arm]=z.p_mix
        for cutoff,ii in z.groupby('cutoff',sort=True).groups.items():
            tr=(d.mature<=cutoff)&(d.mature>cutoff-pd.Timedelta(days=7))
            if arm in ('selected','complete'):tr&=d.selected
            if arm=='complete':tr&=d.history_observed_fraction.eq(1)
            if arm=='paired':tr&=d.paired_selected
            f=d[tr];nw=f.window_id.nunique()
            if nw<10:continue
            assert f.timestamp.max()<=cutoff and cutoff<=z.loc[ii,'origin'].min()
            weights=1/f.groupby('window_id').window_id.transform('size').to_numpy();weights/=weights.mean()
            modeldir=out/'models'/arm;modeldir.mkdir(parents=True,exist_ok=True)
            modelpath=modeldir/f'{cutoff.strftime("%Y%m%dT%H%M")}.txt'
            if modelpath.exists():model=lgb.Booster(model_file=str(modelpath))
            else:
                model=lgb.train(dict(objective='binary',learning_rate=.03,max_depth=3,num_leaves=7,min_data_in_leaf=200,lambda_l2=50,num_threads=3,seed=7,verbosity=-1),lgb.Dataset(f[features],f.label,weight=weights,init_score=f.logit_mix),60)
                model.save_model(str(modelpath))
            delta=model.predict(z.loc[ii,features],raw_score=True,num_threads=3)
            z.loc[ii,arm]=expit(z.loc[ii,'logit_mix'].to_numpy()+delta)
            audit.append(dict(arm=arm,cutoff=str(cutoff),max_label=str(f.timestamp.max()),nwin=int(nw),nrows=len(f)))
            print('fit',rv,arm,str(cutoff),nw,len(f),flush=True)
            del model,f;gc.collect()
    S=L.score_mod(rv);pws={};valid=set(z.window_id)-{'PV_D7_I405_N_2031-03-28_273'};result={}
    z.to_parquet(out/f'{rv}_predictions.parquet',index=False)
    cases={'T1_canonical':z.p_canonical,'T1_K5':z.p_mix}
    for arm in (['paired'] if immediate else ['selected','all','complete','paired']):cases[arm+'25']=.75*z.p_mix+.25*z[arm]
    if not immediate:cases.update(selected100=z.selected,all100=z['all'])
    for name,p in cases.items():
        pw=score_fast(S,z,guarded(z,p));pw=pw[pw.window_id.isin(valid)]
        pws[name]=pw;pw.to_parquet(out/f'{rv}_{name}_perwin.parquet',index=False)
        result[name]={'score':L.agg_w(pw,'iou'),'vs_canonical':bootstrap(pws['T1_canonical'],pw)}
        if 'T1_K5' in pws:result[name]['vs_K5']=bootstrap(pws['T1_K5'],pw)
        print(name,json.dumps(result[name]),flush=True)
    ref=pd.read_parquet(ROOT/f'build/review_20260927/serving_matched/{rv}_T1_perwin.parquet')
    chk=pws['T1_canonical'][['window_id','iou']].merge(ref[['window_id','iou']],on='window_id')
    assert np.max(np.abs(chk.iou_x-chk.iou_y))==0
    z.to_parquet(out/f'{rv}_predictions.parquet',index=False);dump(out/f'{rv}_results.json',result);dump(out/f'{rv}_audit.json',audit)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='parity':parity()
    elif cmd=='official_parity':official_parity()
    elif cmd=='build':build()
    elif cmd=='build_balanced':build(schedule='balanced')
    elif cmd=='run':run(sys.argv[2] if len(sys.argv)>2 else 'v2',sys.argv[3] if len(sys.argv)>3 else 'fixed',len(sys.argv)>4 and sys.argv[4]=='immediate')
