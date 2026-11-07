"""Bottleneck demand-step phase per site (train weekday profile, ongoing/build_shape.rawprof) + u-model self-assessed accuracy."""
# Training record for the site phase table. The reproduction does not run this program; it
# reads the shipped table. `build_shape` (development repository
# `build/review_20260925/ongoing/`, not shipped) reads the full train caches.
#
# For each candidate bottleneck site (`common.BN`: corridor -> site -> bottleneck link
#   indices) of the 8 Task 2 corridors, finds the 5-minute slot (0-5) of the 30-minute cycle
#   in which the flow profile jumps most. `build_shape.rawprof` takes the weekday mean flow profile of
#   each link over the train days (veh/h, per 5-minute slot), averages its absolute change
#   per slot class over 05:00-20:00 (slots 60..239) and keeps the class with the largest mean
#   as the link phase; strength = largest class mean / median class mean.
# Output: `site_phase.json`: "<corridor>|<site>" -> phase (most common phase of the site's
#   bottleneck links), strength (mean over them), their phases, and the phases of the links
#   from 4 below to 2 above the first bottleneck link. Shipped under
#   `trafficflowbench/build/review_20260925/round2/onset_joint/`; `cl.py` reads it for the
#   demand-step phase inputs of FE. This program computes the phases only; the u-model
#   accuracy named in the docstring is not computed here.
# Invocation: `python s0b_phase.py` in this folder.
import sys, json, numpy as np, pandas as pd
import common as C
sys.path.insert(0, str(C.ROOT / 'build/review_20260925/ongoing')); sys.path.insert(0, str(C.ONS))
import build_shape as BS
out = {}
for p in ['D12_I5_N', 'D12_I5_S', 'D7_I405_S', 'D7_I210_W', 'D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I405_N']:
    P, ph, stg = BS.rawprof(p)
    for s, bl in C.BN[p].items():
        out[f'{p}|{s}'] = dict(phase=int(np.bincount(ph[bl]).argmax()), strength=float(np.mean(stg[bl])), phases=[int(x) for x in ph[bl]],
                               nb=[int(x) for x in ph[max(0, bl[0]-4):bl[0]+3]])
    BS._R.clear()
    print(p, {k: v for k, v in out.items() if k.startswith(p)}, flush=True)
json.dump(out, open(C.ME / 'site_phase.json', 'w'), indent=1)
