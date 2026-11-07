"""Addendum C: through-Feb monotone geom model M1f (Jan-31 fit rows + Feb eligible rows)."""
# The docstring above is copied from `m1f_train.py`; this file trains M1sf (Addendum F), the
# same recipe without the geometry inputs.
#
# Training record for M1sf. The reproduction does not run this program; it loads the shipped
# models. The program reads the R20 matched tables (not shipped) through `m_train_infer.py`,
# whose paths start at the hard-coded development root (mapped by the path layer).
#
# Trains three LightGBM binary models (seeds 0, 1, 2) of P(link queued) at each of the 6
#   steps T+5 .. T+30 of an ongoing Task 2 window, with monotone constraints. Served in the
#   selector-recency slot of the ongoing mix, averaged 50/50 per seed with the N6 selrec refit.
# Data: as `m1f_train.py`: every row of `fit_matched_base196.npy` (up to 2031-01-31;
#   2,303,064 rows) plus the eligible February rows of `held_matched_base196.npy` (279,739):
#   2,582,803 rows, 3,013 windows. Inputs (`MT.variant_cols("M1s")`, 156): the 196 z-scored
#   inputs and flags without the 20 geometry inputs and their 20 flags.
# Weights: as `m1f_train.py`.
# Settings: the R20 `gbdt_config` (`MT.CFG`), seed 20260922 + seed, 2 threads, fixed rounds
#   554 / 577 / 681, no early stopping, and the 29 monotone signs of `MT.variant_cols("M1s")`
#   (the same signs as M1f) with method `intermediate`.
# Output: `models/M1sf_seed{0,1,2}.txt`, shipped under
#   `trafficflowbench/build/review_20261001_fresh/mono/models/`; existing files are kept.
#   Afterwards `MT.infer` scores the February and March caches for research evaluation
#   (`prob/`, tag M1sf); the saved models do not depend on that step.
# Invocation: `python m1sf_train.py` in this folder (log: `log_G.txt`). It imports
#   `m_train_infer.py` (shipped next to the original).
import os, sys
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[k]="2"
sys.dont_write_bytecode=True
from pathlib import Path
import json, time, gc
import numpy as np, pandas as pd, lightgbm as lgb
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import m_train_infer as MT
C=MT.CACHE
cols,mono=MT.variant_cols("M1s"); assert len(cols)==156
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
X=np.concatenate([np.load(C/"fit_matched_base196.npy",mmap_mode="r")[:,cols], np.load(C/"held_matched_base196.npy",mmap_mode="r")[hsel][:,cols]],0)
for seed in range(3):
    out=HERE/"models"/f"M1sf_seed{seed}.txt"
    if out.exists(): continue
    P=dict(MT.CFG["params"],seed=20260922+seed,num_threads=2,monotone_constraints=mono,monotone_constraints_method="intermediate")
    t0=time.time(); ds=lgb.Dataset(X,label=lab,weight=w,feature_name=[MT.NAMES[c] for c in cols],free_raw_data=False,params=P)
    m=lgb.train(P,ds,num_boost_round=MT.CFG["rounds"][str(seed)]); m.save_model(str(out)); del ds; gc.collect()
    print("TRAINED M1sf",seed,round(time.time()-t0,1),flush=True)
del X; gc.collect()
MT.infer("M1s", [lgb.Booster(model_file=str(HERE/"models"/f"M1sf_seed{s}.txt")) for s in range(3)], tag="M1sf")
