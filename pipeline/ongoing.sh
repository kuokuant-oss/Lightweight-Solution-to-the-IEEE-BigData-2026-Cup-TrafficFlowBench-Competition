#!/bin/sh
# Task 2 queue_ongoing segment: regenerate <run>/final_sources/ongoing.parquet (the R26c ongoing cells) from the
# released data, with the frozen models. Runs both operators in the original cascade order (Option A, bit-exact):
#
#   F2 operator  -> R16b bits   (refit N6 + A25 on the CTM/newworld serving tables, my_blocks, tau .5,
#                                generic never-empty rule: an empty window takes its A-block decode)
#   R20 operator -> P, C1f      (R20 serving inputs; R21a = where(C1f != P, C1f, R16b))
#   BND          -> R26a        (official k_known, PB parent re-prediction)
#   tau .45      -> R26c        (where(T45 != C1f, T45, R21a)) -> final_sources/ongoing.parquet
#
#   pipeline/ongoing.sh <run_root>          (run tree made by pipeline/stage_tree.py)
#
# Every program runs through bin/tfb-run (path layer on). Every serving input is built from released data
# timestamped <= the window's forecast origin T (window_history T-12..T-1, masked layer at T, ramps at T and T-1/T-5,
# same-split earlier dates for recency, window_index for spacing). No raw February 2031 train file is read; the
# only raw train file read is each panel's first train day (2030-06-01), for link_id/milepost order.
#
# Names: R16b, R21a, R26a and R26c are successive submissions; each operator reproduces the ongoing
# cells of one of them, and R26c is the selected final file. A (A25) is the six-member LightGBM
# ensemble served on the newworld tables, N (N6) the refitted ensemble on the CTM tables; my_blocks are
# train groups of links that congest together; tau is the threshold on a block's mean probability.
# P and C1f are the R20 operator's decisions; BND re-predicts with the selector inputs corrected for
# the official 6-hour spacing of windows (k_known); T45 is that decision decoded at tau .45.
set -eu
[ $# -eq 1 ] || { echo "usage: ongoing.sh <run_root>" >&2; exit 2; }
RUN=$(cd "$1" && pwd -P)
PKG=$(cd "$(dirname "$0")/.." && pwd -P)
TFB="$PKG/bin/tfb-run"
# TFB_PYTHON defaults to the pinned environment; run_reproduce.sh always sets it.
export TFB_PYTHON="${TFB_PYTHON:-$HOME/.venvs/tfb-final-R26c/bin/python}"
# 3 threads at most (the programs pin LightGBM/numpy to 2 themselves)
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
# REPO: the run tree's code copy; REL: the released data inside the run tree.
REPO="$RUN/trafficflowbench"
REL="$RUN/tfb_competition_data/kaggle_public"
# NW, CT: the newworld (A slot) and CTM (N slot) program folders, relative to trafficflowbench/.
NW=build/review_20260924_newworld/work/feat
CT=build/review_20260924_ctm
LOG="$RUN/logs/ongoing"; mkdir -p "$LOG"
# SPLITS: the released splits to serve, in order (TFB_SPLITS, default "validation private"; see README).
export TFB_SPLITS="${TFB_SPLITS:-validation private}"; SPLITS=$TFB_SPLITS
# Returns the newworld file tag of a split: realval / realpriv for the released splits, real<split> otherwise.
nwtag() { case $1 in validation) echo realval ;; private) echo realpriv ;; *) echo "real$1" ;; esac; }
# Returns the short step-log name of a split: val / pri for the released splits, the split name otherwise.
short() { case $1 in validation) echo val ;; private) echo pri ;; *) echo "$1" ;; esac; }

# Runs one program through bin/tfb-run and prints its wall time and peak memory.
# Arguments:
#   Step name (log file name), program path relative to trafficflowbench/, program arguments.
# Outputs:
#   Log in $LOG/<name>.log; on failure the last 40 log lines go to stderr and the segment stops.
step() {  # step <name> <program.py> [args...]
  name=$1; shift
  printf '[%s] %-16s %s\n' "$(date +%H:%M:%S)" "$name" "$*"
  if ! /usr/bin/time -l "$TFB" "$RUN" "$@" > "$LOG/$name.log" 2>&1; then
    tail -40 "$LOG/$name.log" >&2; echo "FAILED: $name" >&2; exit 1
  fi
  awk '/ real /{r=$1} /maximum resident set size/{m=$1} END{printf "           %8.1f s  peak RSS %6.0f MB\n", r, m/1048576}' "$LOG/$name.log"
}
# Moves a file written in the repo root into the folder where the original run left it.
# Arguments:
#   File name (relative to the repo root), destination folder (relative to the repo root).
here() {  # here <file> <dir>: the only command not run through tfb-run.
  # build_serve_feats.py writes f'serve_{tag}.parquet' relative to its working directory (the original runs were
  # started from ctm/serve and newworld/work/feat). tfb-run always starts programs in the repo root, so the file is
  # moved to the folder the original run left it in, where its readers (ogens_serve.py, serve_refit.py) expect it.
  # The alternative would be editing the builder; a move keeps its bytes unchanged.
  mkdir -p "$REPO/$2" && mv "$REPO/$1" "$REPO/$2/$1"
}

# --- F2 operator, A slot: newworld serve tables (originals deleted; same builders, serve side only) -------------
TAGS=""
for s in $SPLITS; do t=$(nwtag "$s"); TAGS="$TAGS $t"
  step "nw_feats_$(short "$s")" $NW/build_serve_feats.py "$REL" "$s" "$t"; here "serve_$t.parquet" $NW
done
for s in $SPLITS; do step "nw_geom_$(short "$s")" $NW/build_geom.py serve "$REL" "$s" "$(nwtag "$s")"; done
step ogens         $NW/ogens_serve.py $TAGS                             # -> ogens_{realval,realpriv}_prob.parquet

# --- F2 operator, N slot: CTM serve tables (ctm/serve/run_real.sh, serve side) ----------------------------------
for s in $SPLITS; do
  step "ctm_feats_$(short "$s")" $CT/ruler/feat/build_serve_feats.py "$REL" "$s" "real_$s"; here "serve_real_$s.parquet" $CT/serve
done
for s in $SPLITS; do step "ctm_geom_$(short "$s")" $CT/ruler/feat/build_geom.py serve "$REL" "$s" "real_$s"; done
for s in $SPLITS; do step "rec_$(short "$s")" $CT/dense/build_recency.py real "$s" queue_ongoing; done
for sp in $SPLITS; do
  for p in D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S; do
    step "cache_${p}_$sp" $CT/cache/build_cache.py $p $sp             # masked-layer day cache for the selector
  done
done
for s in $SPLITS; do step "sel2_$(short "$s")" $CT/dense/serve_sel2.py "$s"; done

# --- F2 decision: refit N6 mix .75N + .25A, my_blocks block mean >= .5, generic never-empty fallback -----------
step fallback      build/review_20260925/make_never_empty_fallback.py   # -> candidates/a_block_fallback.parquet (all windows)
step serve_refit   build/review_20260929_target092/main/serve_refit.py  # -> override, never_empty_fallback.json, ongoing_refitN6_d1.parquet

# --- R20 operator: serving inputs, R16b bits, P / C1f -> R21a ---------------------------------------------------
step r20_prepare   build/review_20260930_round20/queue_data/serving_stage/prepare.py
step r16b_bits     build/review_20261001_fresh/serve/make_r16b_bits.py
step s1_prob       build/review_20261001_fresh/serve/s1_serve_prob.py parent M1f M1sf
step s2_C1f        build/review_20261001_fresh/serve/s2_decide.py P+M1f --sel P+M1sf --tag C1f
step r21a_bits     build/review_20261001_r22/serve/make_r21a_bits.py

# --- BND (R26a), tau .45 (R26c) ---------------------------------------------------------------------------------
step bnd_prob      build/review_20261001_r26/serve/r26_serve.py prob
step bnd_decide    build/review_20261001_r26/serve/r26_serve.py decide
step t45_decide    build/review_20261001_r26/serve/r26_serve_t45.py decide
step final_source  build/review_20261001_r26/serve/make_ongoing_source.py   # -> <run>/final_sources/ongoing.parquet
echo "ongoing segment done: $RUN/final_sources/ongoing.parquet"
