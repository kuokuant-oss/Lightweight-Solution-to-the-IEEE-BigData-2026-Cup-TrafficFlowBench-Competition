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
set -eu
[ $# -eq 1 ] || { echo "usage: ongoing.sh <run_root>" >&2; exit 2; }
RUN=$(cd "$1" && pwd -P)
PKG=$(cd "$(dirname "$0")/.." && pwd -P)
TFB="$PKG/bin/tfb-run"
export TFB_PYTHON="${TFB_PYTHON:-$HOME/.venvs/tfb-final-R26c/bin/python}"
# 3 threads at most (the programs pin LightGBM/numpy to 2 themselves)
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
REPO="$RUN/trafficflowbench"
REL="$RUN/tfb_competition_data/kaggle_public"
NW=build/review_20260924_newworld/work/feat
CT=build/review_20260924_ctm
LOG="$RUN/logs/ongoing"; mkdir -p "$LOG"

step() {  # step <name> <program.py> [args...]
  name=$1; shift
  printf '[%s] %-16s %s\n' "$(date +%H:%M:%S)" "$name" "$*"
  if ! /usr/bin/time -l "$TFB" "$RUN" "$@" > "$LOG/$name.log" 2>&1; then
    tail -40 "$LOG/$name.log" >&2; echo "FAILED: $name" >&2; exit 1
  fi
  awk '/ real /{r=$1} /maximum resident set size/{m=$1} END{printf "           %8.1f s  peak RSS %6.0f MB\n", r, m/1048576}' "$LOG/$name.log"
}
here() {  # here <file> <dir>: the only command not run through tfb-run.
  # build_serve_feats.py writes f'serve_{tag}.parquet' relative to its working directory (the original runs were
  # started from ctm/serve and newworld/work/feat). tfb-run always starts programs in the repo root, so the file is
  # moved to the folder the original run left it in, where its readers (ogens_serve.py, serve_refit.py) expect it.
  # The alternative would be editing the builder; a move keeps its bytes unchanged.
  mkdir -p "$REPO/$2" && mv "$REPO/$1" "$REPO/$2/$1"
}

# --- F2 operator, A slot: newworld serve tables (originals deleted; same builders, serve side only) -------------
step nw_feats_val  $NW/build_serve_feats.py "$REL" validation realval;  here serve_realval.parquet  $NW
step nw_feats_pri  $NW/build_serve_feats.py "$REL" private realpriv;    here serve_realpriv.parquet $NW
step nw_geom_val   $NW/build_geom.py serve "$REL" validation realval
step nw_geom_pri   $NW/build_geom.py serve "$REL" private realpriv
step ogens         $NW/ogens_serve.py realval realpriv                  # -> ogens_{realval,realpriv}_prob.parquet

# --- F2 operator, N slot: CTM serve tables (ctm/serve/run_real.sh, serve side) ----------------------------------
step ctm_feats_val $CT/ruler/feat/build_serve_feats.py "$REL" validation real_validation; here serve_real_validation.parquet $CT/serve
step ctm_feats_pri $CT/ruler/feat/build_serve_feats.py "$REL" private real_private;       here serve_real_private.parquet    $CT/serve
step ctm_geom_val  $CT/ruler/feat/build_geom.py serve "$REL" validation real_validation
step ctm_geom_pri  $CT/ruler/feat/build_geom.py serve "$REL" private real_private
step rec_val       $CT/dense/build_recency.py real validation queue_ongoing
step rec_pri       $CT/dense/build_recency.py real private queue_ongoing
for sp in validation private; do
  for p in D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S; do
    step "cache_${p}_$sp" $CT/cache/build_cache.py $p $sp             # masked-layer day cache for the selector
  done
done
step sel2_val      $CT/dense/serve_sel2.py validation
step sel2_pri      $CT/dense/serve_sel2.py private

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
