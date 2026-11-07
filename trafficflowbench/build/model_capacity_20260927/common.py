"""Small experiment infrastructure. Keep all old artifacts read-only."""
import json, math, hashlib, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def state_alpha(policy,ratio):
    if policy in ['gate75','gate100']:
        top=.75 if policy=='gate75'else 1.
        return .25+(top-.25)*np.clip((.85-np.asarray(ratio))/.25,0,1)
    return float(policy)

def lgb_prefix_predictions(model,x,iterations,binary=False):
    """Add successive disjoint tree ranges; avoid re-evaluating shared prefixes."""
    from scipy.special import expit
    previous=0;raw=np.zeros(len(x));out={}
    for it in sorted(set(iterations)):
        raw+=model.predict(x,start_iteration=previous,num_iteration=it-previous,raw_score=True,num_threads=3)
        out[it]=expit(raw)if binary else raw.copy()
        direct=model.predict(x[:1024],num_iteration=it,num_threads=3)
        assert np.max(abs(direct-out[it][:1024]))<1e-11
        previous=it
    return out

def stopping_point(losses,minimum=1,patience=150):
    """Replay a fixed GBDT path; returns best and executed 1-based rounds."""
    if minimum<1 or patience<1: raise ValueError('minimum/patience must be positive')
    best=math.inf; best_round=0
    for t,loss in enumerate(losses,1):
        if not math.isfinite(loss):raise ValueError('non-finite validation loss')
        if loss<best:best,best_round=loss,t
        if t>=minimum and t-best_round>=patience:
            return dict(best=best_round,executed=t,loss=best,reason='patience')
    return dict(best=best_round,executed=len(losses),loss=best,reason='cap')

def test_stopping():
    assert stopping_point([4,1,2,3,4],4,2)==dict(best=2,executed=4,loss=1,reason='patience')
    assert stopping_point([4,3,2,1],3,2)['executed']==4
    assert stopping_point([1,2,3,4],1,2)['executed']==3
    assert stopping_point([1,2,3,.5,.6,.7],5,2)['best']==4
    assert stopping_point([1,2,3,.5,.6,.7],1,2)['best']==1
    assert stopping_point([1,2,3,.5,.6,.7],5,2)['executed']==6
    try:stopping_point([float('nan')])
    except ValueError:pass
    else:raise AssertionError('NaN accepted')
    return {'minimum_budget_tests':7,'external_pruner':False,'passed':True}

class MinimumBudget:
    """LightGBM callback: monitor stop only, retain best even before warm-up."""
    order=30
    before_iteration=False
    def __init__(self,minimum=600,patience=150):
        # LightGBM inserts an instance default via cb.__dict__.setdefault().
        # Class attributes alone would run this before record_evaluation.
        self.order=30;self.before_iteration=False
        self.minimum=minimum;self.patience=patience;self.losses=[]
        self.best=math.inf;self.best_i=0;self.best_eval=None
    def __call__(self,env):
        import lightgbm as lgb
        values=[x for x in env.evaluation_result_list if x[0]=='stop']
        assert len(values)==1 and not values[0][3]
        loss=values[0][2];t=env.iteration+1
        if not math.isfinite(loss):raise ValueError('nonfinite stop loss')
        self.losses.append(loss)
        if loss<self.best:self.best=loss;self.best_i=env.iteration;self.best_eval=list(env.evaluation_result_list)
        if t>=self.minimum and env.iteration-self.best_i>=self.patience:
            raise lgb.callback.EarlyStopException(self.best_i,self.best_eval)
