"""Two prospective history sources. Future truth labels never enter frame()."""
from common import *
sys.path.insert(0,str(ROOT/'build/review_20260927_direct'))
import direct as D
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
from threadpoolctl import threadpool_limits
threadpool_limits(limits=2)
from e_data import PublishedPanel,FIELDS
def strict_context(panel):
    assert BLOCKS.exists(),'strict cutoff block map required; never silently use legacy map'
    b=np.load(BLOCKS,allow_pickle=True).item();D.L.BID={p:dict(zip(v['links'],v['bid']))for p,v in b.items()}
    return D.Context(panel,'train')
def prefix(day,T):return {col:np.asarray(day[col][:,T-12:T]).copy()for col in FIELDS+['is_score_eligible']}
def source_day(raw,T,past=None):
    out={k:np.asarray(v).copy()for k,v in raw.items()}
    if past is not None:
        for col in out:
            assert past[col].shape==out[col][:,T-12:T].shape
            out[col][:,T-12:T]=past[col]
    for col in out:out[col][:,T+1:]=False if col=='is_score_eligible'else np.nan
    return out
def frame(P,c,date,T,wid,raw,past=None):
    day=source_day(raw,T,past);origin=source_day(raw,T)
    f=c.features(date,T,day,wid,D.REC.kdraw(wid),None,origin)
    assert not set(FS)-set(f)and not set(D.FS)-set(f)
    assert not f[KEY].duplicated().any()
    assert len(f)==len(P.links)*6
    assert set(f.step)==set(range(1,7))
    assert np.array_equal(f.bid.to_numpy(),f.link_id.map(D.L.BID[P.panel]).to_numpy())
    return f
def encode(P,raw,T,past=None):return P.encode(source_day(raw,T,past),T)
def old_parent_prob(f):
    # Immutable pre-Feb parents only, never R19 public_parent_N / through-Feb refit.
    n=np.mean([D.L.booster(p).predict(f[D.L.booster(p).feature_name()],num_threads=2)for p in D.L.SR6],axis=0)
    a=np.mean([D.L.booster(D.L.AMOD/f'og_{name}.txt').predict(f[D.L.booster(D.L.AMOD/f'og_{name}.txt').feature_name()],num_threads=2)for name in D.L.A_NAMES],axis=0)
    return n,a
def causal_QA(P,c,date,T,wid,raw,past=None):
    a=frame(P,c,date,T,wid,raw,past);xa=encode(P,raw,T,past)
    poisoned={k:v.copy()for k,v in raw.items()}
    for col in poisoned:poisoned[col][:,T+1:]=True if col=='is_score_eligible'else 987654.
    ramps=c.ramp(date);save={t:v.copy()for t,v in ramps.items()}
    c.ramps[date]={t:({k:987654. for k in v}if t>T else v.copy())for t,v in save.items()}
    b=frame(P,c,date,T,wid,poisoned,past);xb=encode(P,poisoned,T,past);c.ramps[date]=save
    assert np.array_equal(a[D.FS].to_numpy(),b[D.FS].to_numpy(),equal_nan=True)
    assert np.array_equal(xa,xb)
    na,aa=old_parent_prob(a);nb,ab=old_parent_prob(b)
    assert np.array_equal(na,nb)and np.array_equal(aa,ab)
    a['pA']=aa;b['pA']=ab
    assert np.array_equal(D.guarded(a,.75*na+.25*aa),D.guarded(b,.75*nb+.25*ab))
    return dict(panel=P.panel,window_id=wid,source='matched'if past is not None else'raw',base98=True,A25=True,preFeb_N6=True,decoder=True,encoder_source=True,
        postT_mainline_and_ramp_exact=True,scope='one sampled first window per panel/cohort/source; input regression, not all-window model proof',blocks_sha256=sha(BLOCKS))
