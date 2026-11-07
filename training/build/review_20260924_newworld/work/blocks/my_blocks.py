"""Independent re-derivation: co-switching blocks from train truth (ALL days, both neighbours), then block decoding on pv onset."""
# Training record for the co-switching block map. The reproduction does not run this
# program; it loads the shipped map. The program reads the train release through `src.tfb`
# under the hard-coded development root (mapped onto the run tree by the path layer).
#
# Builds, per Task 2 corridor, blocks of adjacent links that switch into queue together.
# Data: true speeds of all train days, 2030-06-01 .. 2031-02-28 (February included). A cell
#   is queued when it is score-eligible and its speed is at or below the link's queue cut
#   (`score2.v_cut`: released v_cut, else 0.6 x free-flow speed, km/h).
# Rule: for each adjacent pair, `co` = share of queue onsets at one link (queued at t, free
#   at t-1) for which the neighbour, observed at t-1 and t and free at t-1, is also queued at
#   t; both directions are pooled. Pairs with co >= 0.5 stay in one block; any other pair,
#   including pairs without onsets (co = NaN), starts a new block. Deterministic.
# Output: `my_blocks.npy` in the working directory: a pickled dict corridor -> {links (in
#   milepost order), co, den (onsets counted per pair), bid (block id per link)}. Shipped as
#   `trafficflowbench/build/review_20260924_newworld/work/blocks/my_blocks.npy` and as the
#   byte-identical `trafficflowbench/build/review_20260924_ctm/ruler/blocks/my_blocks.npy`.
# Invocation: `python my_blocks.py` in this folder. This program builds the blocks only; the
#   block decoding named in the docstring is done elsewhere.
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0,'/Volumes/Crucial X9/trafficflow/trafficflowbench')
from src.tfb import release as R, score2, task2build as TB
PAN=['D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S','D12_I5_N','D12_I5_S']
out={}
for p in PAN:
    cut=score2.v_cut(p); files=R.truth_files(p,'train')
    links=None
    for f in files:
        # Link x time arrays of one day; Q = queued (speed <= cut on eligible cells).
        L,mp,V,Qf,O,E,Q,times=TB.day_arrays(p,f,cut)
        # num/den per adjacent pair i = (link i, link i+1).
        if links is None: links=L; n=len(L); num=np.zeros(n-1); den=np.zeros(n-1)
        ok=E&~np.isnan(V); q=Q
        # Onset at (link, t): queued at t, free at t-1, observed at both steps.
        on=np.zeros(q.shape,bool); on[:,1:]=q[:,1:]&~q[:,:-1]&ok[:,1:]&ok[:,:-1]
        # pair (i,i+1): an onset at either member while the other is observed at t-1,t and free at t-1
        for a,b in [(slice(0,n-1),slice(1,n)),(slice(1,n),slice(0,n-1))]:
            ev=on[a][:,1:]&ok[b][:,1:]&ok[b][:,:-1]&~q[b][:,:-1]
            num+=(ev&q[b][:,1:]).sum(1); den+=ev.sum(1)
    co=np.where(den>0,num/np.maximum(den,1),np.nan)
    # A new block starts after every pair with co < 0.5 or no onset.
    bid=np.concatenate([[0],np.cumsum(~(co>=0.5))])
    out[p]=dict(links=links,co=co,den=den,bid=bid)
    print(p,'links',n,'blocks',bid.max()+1,'co bimodality: share of pairs with co in [0.2,0.8] among den>20:',round(float(((co>0.2)&(co<0.8)&(den>20)).sum()/max((den>20).sum(),1)),3),flush=True)
np.save('my_blocks.npy',out,allow_pickle=True)
