#!/bin/sh
# Refit (fit<=2031-02-28, fixed production rounds) for the remaining 5 N-ensemble members, sequentially, 2 threads each.
#
# Training record (the reproduction does not run it). Runs `train_refit3.py` for the five N6
# members that `train_refit.py` did not train: R_V2selrec seeds 1 and 2, R_V2geomselrec seeds
# 0, 1 and 2, each with fit end 2031-02-28 and tag `refit`. The resulting
# `m_<variant>_refit_seed<seed>.txt` files ship under
# `trafficflowbench/build/review_20260929_target092/contract_audit/q4_refit/`.
# PY and the `cd` target are the development machine's Python and this folder there.
#
# Outputs: one log `log_refit_<variant>_seed<seed>.txt` per member; prints the start of the
# last log line of each member, then ALLDONE.
PY="/Volumes/Crucial X9/trafficflow/trafficflowbench/.venv/bin/python"
cd "/Volumes/Crucial X9/trafficflow/trafficflowbench/build/review_20260929_target092/contract_audit/q4_refit"
# Each spec is "<variant> <seed>"; `set --` splits it into $1 (variant) and $2 (seed).
for spec in "R_V2selrec 1" "R_V2selrec 2" "R_V2geomselrec 0" "R_V2geomselrec 1" "R_V2geomselrec 2"; do
  set -- $spec
  "$PY" train_refit3.py 2031-02-28 refit $2 $1 > log_refit_$1_seed$2.txt 2>&1
  echo "END $1 $2 $(tail -1 log_refit_$1_seed$2.txt | cut -c1-120)"
done
echo ALLDONE
