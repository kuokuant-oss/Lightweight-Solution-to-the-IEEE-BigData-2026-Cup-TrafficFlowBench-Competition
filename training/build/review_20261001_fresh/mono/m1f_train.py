"""Addendum C: through-Feb monotone geom model M1f (Jan-31 fit rows + Feb eligible rows)."""
# Training record for M1f. The reproduction does not run this program; it loads the shipped
# models. The program reads the R20 matched tables (not shipped) through `m_train_infer.py`,
# whose paths start at the hard-coded development root (mapped by the path layer).
#
# Trains three LightGBM binary models (seeds 0, 1, 2) of P(link queued) at each of the 6
#   steps T+5 .. T+30 of an ongoing Task 2 window, with monotone constraints. Served in the
#   geometry slot of the ongoing mix, averaged 50/50 per seed with the N6 geomselrec refit.
# Data (`MT.CACHE` = `build/review_20260930_round20/queue_data/cache/`): every row of
#   `fit_matched_base196.npy` (rep2 windows dated up to 2031-01-31; 2,303,064 rows) plus the
#   eligible February rows of `held_matched_base196.npy` (`target_mask`; 279,739 rows):
#   2,582,803 rows, 3,013 windows. Inputs (`MT.NAMES`, 196): the 98 geomselrec inputs imputed
#   and z-scored with `queue_data/TRANSFORM_base196.json` (statistics from rows up to
#   2031-01-31), plus 98 missing-value flags.
# Weights: 1 / (rows of the window x windows of the corridor) over the union, normalised to
#   mean 1; February window ids are offset by 10,000,000 so they stay distinct.
# Settings: the R20 `gbdt_config` (`MT.CFG`: binary, 127 leaves, learning rate .05,
#   min_data_in_leaf 100, feature fraction .9, bagging .8 every iteration, lambda_l2 10,
#   max_bin 127), seed 20260922 + seed, 2 threads, fixed rounds 554 / 577 / 681, no early
#   stopping, and the 29 monotone signs of `MT.variant_cols("M1")` (17 inputs push the queue
#   probability down, 12 up; none is a geometry input; flags are unconstrained) with method
#   `intermediate`.
# Output: `models/M1f_seed{0,1,2}.txt`, shipped under
#   `trafficflowbench/build/review_20261001_fresh/mono/models/`; existing files are kept.
#   Afterwards `MT.infer` scores the February and March caches for research evaluation
#   (`prob/`); the saved models do not depend on that step.
# Invocation: `python m1f_train.py` in this folder (log: `log_F.txt`). It imports
#   `m_train_infer.py` (shipped next to the original); `m_train_infer.py` run with the name
#   M1f trains a different, unconstrained model.
import os, sys
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[k]="2"
sys.dont_write_bytecode=True
from pathlib import Path
import json, time, gc
import numpy as np, pandas as pd, lightgbm as lgb
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import m_train_infer as MT
C=MT.CACHE
cols,mono=MT.variant_cols("M1"); assert len(cols)==196
fm=np.load(C/"fit_meta.npz"); hm=np.load(C/"held_meta.npz")
# February rows that are Task 2 targets (eligible cells).
hsel=np.flatnonzero(hm["target_mask"])
win=np.concatenate([fm["win"].astype(np.int64), 10_000_000+hm["win"][hsel].astype(np.int64)])
pan=np.concatenate([fm["pan"], hm["pan"][hsel]]).astype(np.int64)
lab=np.concatenate([fm["label"], hm["label"][hsel]]).astype(np.float32)
# nc: rows of each row's window; pw: windows per corridor.
s=pd.Series(win); nc=s.map(s.value_counts()).to_numpy(float)
pw=pd.DataFrame({"pan":pan,"win":win}).drop_duplicates().groupby("pan").size()
w=1/(nc*pd.Series(pan).map(pw).to_numpy()); w/=w.mean()
print("rows fit",len(fm["label"]),"feb",len(hsel),"windows",len(np.unique(win)),flush=True)
X=np.concatenate([np.load(C/"fit_matched_base196.npy",mmap_mode="r"), np.load(C/"held_matched_base196.npy",mmap_mode="r")[hsel]],0)
for seed in range(3):
    out=HERE/"models"/f"M1f_seed{seed}.txt"
    if out.exists(): continue
    P=dict(MT.CFG["params"],seed=20260922+seed,num_threads=2,monotone_constraints=mono,monotone_constraints_method="intermediate")
    t0=time.time(); ds=lgb.Dataset(X,label=lab,weight=w,feature_name=MT.NAMES,free_raw_data=False,params=P)
    m=lgb.train(P,ds,num_boost_round=MT.CFG["rounds"][str(seed)]); m.save_model(str(out)); del ds; gc.collect()
    print("TRAINED M1f",seed,round(time.time()-t0,1),flush=True)
del X; gc.collect()
MT.infer("M1f", [lgb.Booster(model_file=str(HERE/"models"/f"M1f_seed{s}.txt")) for s in range(3)])
