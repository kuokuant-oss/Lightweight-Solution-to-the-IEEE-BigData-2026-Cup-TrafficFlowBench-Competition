"""Small experiment infrastructure. Keep all old artifacts read-only."""
# Shared helpers of the capacity experiment (CAP). CAP is the CatBoost
# "second learner" of the Task 1 state: CatBoost residual heads redo the RLB
# correction (RLB: the residual LightGBM model on the C1 fill), and half of the
# difference from the LightGBM result is added to the state. The reproduction
# uses HERE, ROOT and state_alpha (build_state_candidate.py) and dump
# (state_confirm.py); the stopping helpers belong to model training and are not
# called in the reproduction.
import json, math, hashlib, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def dump(path, value):
    """Writes `value` to `path` as indented JSON (numpy scalars as numbers)."""
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def sha(path):
    """Returns the SHA256 hex digest of a file, read in 8 MB chunks."""
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def state_alpha(policy,ratio):
    """Returns the weight of the (CatBoost - LightGBM) state difference.

    Args:
      policy: 'gate75' or 'gate100' for a speed-ratio gate; any other value is
        a constant weight (number or numeric string). The selected policy in
        state/selection.json is 0.5.
      ratio: fill speed divided by the link's free-flow speed (feature p_r),
        one value per cell.

    Returns:
      For a gate policy, an array that is 0.25 at ratio >= 0.85 and rises
      linearly to 0.75 ('gate75') or 1.0 ('gate100') at ratio <= 0.60, so
      congested cells take more of the difference. Otherwise float(policy).
    """
    if policy in ['gate75','gate100']:
        top=.75 if policy=='gate75'else 1.
        return .25+(top-.25)*np.clip((.85-np.asarray(ratio))/.25,0,1)
    return float(policy)

def lgb_prefix_predictions(model,x,iterations,binary=False):
    """Add successive disjoint tree ranges; avoid re-evaluating shared prefixes."""
    # Returns {iteration: prediction of the first `iteration` trees} (raw
    # scores, or probabilities when `binary`). Each step is checked against a
    # direct LightGBM prediction on the first 1,024 rows.
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
    # `losses` holds one validation loss per round. Training stops once at
    # least `minimum` rounds ran and the best round is `patience` rounds old;
    # reason 'cap' means the loss path ended first.
    if minimum<1 or patience<1: raise ValueError('minimum/patience must be positive')
    best=math.inf; best_round=0
    for t,loss in enumerate(losses,1):
        if not math.isfinite(loss):raise ValueError('non-finite validation loss')
        if loss<best:best,best_round=loss,t
        if t>=minimum and t-best_round>=patience:
            return dict(best=best_round,executed=t,loss=best,reason='patience')
    return dict(best=best_round,executed=len(losses),loss=best,reason='cap')

def test_stopping():
    """Runs the unit checks of stopping_point and returns a summary dict.

    Raises:
      AssertionError: when a check fails or a NaN loss is accepted.
    """
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
    # LightGBM runs callbacks in increasing `order`; 30 runs this one after
    # record_evaluation. It reads the metric named 'stop' (lower is better)
    # and stops at the best iteration once `minimum` rounds have run and the
    # best is `patience` rounds old.
    order=30
    before_iteration=False
    def __init__(self,minimum=600,patience=150):
        """Sets the warm-up length `minimum` and the `patience`, in rounds."""
        # LightGBM inserts an instance default via cb.__dict__.setdefault().
        # Class attributes alone would run this before record_evaluation.
        self.order=30;self.before_iteration=False
        self.minimum=minimum;self.patience=patience;self.losses=[]
        self.best=math.inf;self.best_i=0;self.best_eval=None
    def __call__(self,env):
        """Records this round's 'stop' loss and stops training when due.

        Raises:
          ValueError: when the loss is not finite.
          lightgbm.callback.EarlyStopException: to end training at the best
            iteration.
        """
        import lightgbm as lgb
        values=[x for x in env.evaluation_result_list if x[0]=='stop']
        assert len(values)==1 and not values[0][3]
        loss=values[0][2];t=env.iteration+1
        if not math.isfinite(loss):raise ValueError('nonfinite stop loss')
        self.losses.append(loss)
        if loss<self.best:self.best=loss;self.best_i=env.iteration;self.best_eval=list(env.evaluation_result_list)
        if t>=self.minimum and env.iteration-self.best_i>=self.patience:
            raise lgb.callback.EarlyStopException(self.best_i,self.best_eval)
