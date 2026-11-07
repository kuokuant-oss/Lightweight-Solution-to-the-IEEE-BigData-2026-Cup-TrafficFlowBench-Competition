#!/bin/sh
# Task 1/3 (traffic state) segment: regenerate the six final state sources of R26c from the released data.
#
#   TFB_PYTHON=~/.venvs/tfb-final-R26c/bin/python pipeline/task1.sh <run_root> [first_step [last_step]]
#
# <run_root> is a fresh tree made by pipeline/stage_tree.py. Every program runs through bin/tfb-run, so all
# reads and writes stay inside <run_root>. Models are the frozen artifacts shipped under trafficflowbench/;
# the per-split fits that the original serving did (T1 FS, HM25 heads, PRP head, HMP50 clusters) run here.
# Output: <run_root>/final_sources/state_{1..6}_*.parquet (submission_id, speed_kmh, flow_vph).
# Steps are numbered; first_step/last_step (default 1 and 99) are for development reruns only.
set -eu
[ $# -ge 1 ] || { echo "usage: task1.sh <run_root> [first_step [last_step]]" >&2; exit 2; }
RUN=$(cd "$1" && pwd -P); FIRST=${2:-1}; LAST=${3:-99}
PKG=$(cd "$(dirname "$0")/.." && pwd -P)
PY=${TFB_PYTHON:?set TFB_PYTHON to the pinned interpreter}
export TFB_PYTHON="$PY"
DATA="$RUN/tfb_competition_data/kaggle_public"
T="$RUN/trafficflowbench"; B="$T/build"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} PYTHONWARNINGS=ignore
LOG="$RUN/logs"; mkdir -p "$LOG"
# BSD time: with -l it adds the peak memory (maximum resident set size) to each log.
TIME=/usr/bin/time

# Tells whether a numbered step lies in the requested range FIRST..LAST.
# Arguments:
#   Step number.
# Returns:
#   0 when the step runs, non-zero otherwise.
step() { [ "$1" -ge "$FIRST" ] && [ "$1" -le "$LAST" ]; }
# run <log name> <program relative to trafficflowbench/> [args]   (peak RSS is in the log, from time -l)
# Runs one project program through bin/tfb-run and writes its output to $LOG/<name>.log.
# On failure it prints the last 30 log lines and stops the segment.
run() {
  name=$1; shift; echo "$(date +%T) $name"
  $TIME -l "$PKG/bin/tfb-run" "$RUN" "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED $name (see $LOG/$name.log)"; tail -30 "$LOG/$name.log"; exit 1; }
}
# helper <log name> <pipeline helper> [args]   (package-level helpers, plain interpreter)
# Same logging and failure handling as run(), for the scripts in pipeline/.
helper() {
  name=$1; shift; echo "$(date +%T) $name"
  $TIME -l "$PY" "$PKG/pipeline/$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED $name (see $LOG/$name.log)"; tail -30 "$LOG/$name.log"; exit 1; }
}
# The eight corridors (panels) with Task 2 windows; the other two (D12_I405_N/S) have Task 1 cells only.
PANELS8="D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S D12_I5_N D12_I5_S"
# RL: folder of the residual-correction programs (RLB model, RL fallback); FEATS: their serving
# features, including the c1all_serve_* side files that the layers below read.
RL=build/review_20260924_ctm/rl
FEATS="$B/review_20260924_ctm/rl/feats"
# Parent-CSV layers (see pipeline/task1_state_csv.py). Each historical parent submission CSV that a program reads
# is rebuilt state-only from the keyed files regenerated in this run, in the number format its builder wrote.
# Layer names (each is one stage of the Task 1 chain, in chain order):
#   v3     stacked fill: LightGBM weighting of five classical fills (LRTC, interpolations, profile)
#   k*     joint density projection of speed and flow on the 5 density corridors
#   G2     gap model inside the 90-minute holes around the Task 2 windows
#   RLB    residual LightGBM correction of the stacked fill (RL: fallback model for D12_I405)
#   G2v2   gap model version 2 (model v2_F); gives the final hole flow
#   T1     month-specific hole speed correction, fitted on pseudo holes of the same split
#   HM25, HMP50   hole speed mixture heads and their pooled version (speed only)
#   DK50, CAP     density correction and CatBoost capacity blend on the general cells
#   D12    stage-2 correction on the D12_I405 pair; PRP_plus: two-state hole speeds
# Each spec is "keyed parquet (glob allowed),speed column,flow column,number format[,mask column]".
L_V3="$FEATS/c1all_serve_*.parquet,stack_v,stack_q,np3"                                    # v3 stack (step 2)
L_KSTAR="$B/strongest_candidate_20260922/state/transfer/serving/stateupdates.parquet,new_speed_kmh,new_flow_vph,fmt6"  # k* (step 3)
L_G2="$FEATS/c1all_serve_*.parquet,c1_v,c1_q,fmt3,hole"                                    # G2 holes (step 4)
L_RLB="$B/review_20260924_ctm/rl/rlb_state_rows.parquet,speed_kmh,flow_vph,fmt3"           # RLB/RL (step 6)
L_G2V2="$B/review_20260925/task1/g2v2F_state_rows.parquet,speed_kmh,flow_vph,fmt3"         # G2v2 holes (step 7)
L_T1="$B/review_20260925/round3/task1_trans_serving/work/trans_state_rows.parquet,speed_kmh,-,fmt3"   # T1 (step 9)
L_HM25="$B/review_20260927/iteration2/hole_mixture/official_state_updates.parquet,speed_kmh,-,fmt3"  # HM25 (step 10)
L_HMP50="$B/review_20260927_research/serving/official_state_updates.parquet,speed_kmh,-,fmt3"       # HMP50 (step 11)
L_DK50="$B/review_20260927_structural/density/serving/state_updates.parquet,speed_kmh,flow_vph,fmt3"  # DK50 (step 12)
L_CAP="$B/model_capacity_20260927/state/serving/state_updates.parquet,speed_kmh,flow_vph,fmt3"        # CAP (step 13)
L_D12="$B/review_20260929_target092/noise_floor/d12_state_updates_S2_100.parquet,speed_kmh,flow_vph,fmt3"  # D12 (step 14)
L_PRP="$B/review_20260929_target092/main/state_updates_gap_prp.parquet,speed_kmh,flow_vph,fmt3"       # PRP_plus (step 14)
# Rebuilds one historical parent submission CSV, state rows only, from the given layers.
# Arguments:
#   Output CSV path, then one or more layer specs (applied in order).
# Outputs:
#   The CSV; the helper log is $LOG/csv_<name>.log.
csv() {   # csv <parent csv path> <layer> [<layer> ...]   (layer globs are expanded by the helper, not the shell)
  out=$1; shift; mkdir -p "$(dirname "$out")"
  set -f; args=""; for l in "$@"; do args="$args --layer $l"; done
  helper "csv_$(basename "$out" .csv)" task1_state_csv.py --data "$DATA" --out "$out" $args
  set +f
}

# 1. C1 rebuild (LRTC + v3 stack + fallback, G2 inside the holes) and RLB serving features, both splits.
if step 1; then
  mkdir -p "$FEATS" "$B/review_20260924_ctm/rl/results"
  # the two splits are independent: run them side by side, 2 threads each (about 14 min, 3.1 GB peak RSS each)
  ( export OMP_NUM_THREADS=2; run serve_feats_validation $RL/serve_feats.py --mode serve --split validation ) & P1=$!
  ( export OMP_NUM_THREADS=2; run serve_feats_private $RL/serve_feats.py --mode serve --split private ) & P2=$!
  wait $P1; wait $P2
fi
# 2. k* on the 5 density corridors: v3 state (parent of k*), k_aux serving rows, joint projection.
if step 2; then
  csv "$B/onset_condition_candidate_20260922/submission_onset_endpoint_top1.csv" "$L_V3"
  run kaux_first build/final_sprint_review_20260922/experiment_audit/build_density_serving_updates.py
  run kstar scripts/build_strongest_state_transfer_serving_20260922.py
fi
# 3. RLB / RL apply (+ k* re-applied) -> rlb_state_rows. Its changed-row filter compares with the step-5 champion state.
if step 3; then
  csv "$B/review_20260924_newworld/candidates/submission_D_onsetsite_blocks.csv" "$L_V3" "$L_KSTAR" "$L_G2"
  run apply_rlb $RL/apply_rl.py --model models/RLB_final --fallback-model models/RL_final --fallback-panels D12_I405_N D12_I405_S --out rlb_state_rows.parquet
fi
# 4. Per-split caches of the masked layer + ramp counts, selector features, G2v2 v2_F holes (final hole flow).
if step 4; then
  for s in validation private; do for p in $PANELS8; do run "cache_${p}_$s" build/review_20260924_ctm/cache/build_cache.py $p $s; done; done
  for s in validation private; do run "sel2_$s" build/review_20260924_ctm/dense/serve_sel2.py $s; done
  mkdir -p "$B/review_20260925/task1"
  run g2v2_serve build/review_20260924_ctm/g2v2/serve_v2.py --model build/review_20260924_ctm/g2v2/models/v2_F --out build/review_20260925/task1/g2v2F_state_rows.parquet
fi
# 5. Pseudo holes per split (T1 harness) and official hole rows.
if step 5; then
  csv "$B/review_20260925/round2/synthesis/submission_G1_FE_A1.csv" "$L_V3" "$L_KSTAR" "$L_G2" "$L_RLB" "$L_G2V2"
  run champ_state build/review_20260924_ctm/ruler/q/build_champ_state.py
  run build_fake build/review_20260924_ctm/ruler/fakeval/build_fake.py validation
  run cands_v2 build/review_20260924_ctm/ruler2/cands_v2.py
  run select_write_v2 build/review_20260924_ctm/ruler2/select_write_v2.py validation
  mkdir -p "$B/review_20260925/round2/task1_trans/work" "$B/review_20260925/round3/task1_trans_serving/work" "$B/review_20260925/round3/task1_trans_serving/results"
  run prep_key build/review_20260925/round2/task1_trans/prep_key.py
  run ph_validation build/review_20260925/round2/task1_trans/build_ph.py --split validation
  run cands_private build/review_20260925/round3/task1_trans_serving/s2_cands.py private
  run ph_private build/review_20260925/round3/task1_trans_serving/s3_build_rows.py pseudo private
  for s in validation private; do run "off_rows_$s" build/review_20260925/round3/task1_trans_serving/s3_build_rows.py official $s; done
fi
# 6. T1 FS per split (fit on the split's pseudo holes) -> trans_state_rows.
if step 6; then
  run t1_ldo build/review_20260925/round3/task1_trans_serving/s1_ldo_validation.py
  run t1_serve build/review_20260925/round3/task1_trans_serving/s6_serve.py
fi
# 7. HM25 heads per split (strict odd/even heads for the HMP50 harness, then the official heads) and HMP50 clusters.
if step 7; then
  mkdir -p "$B/review_20260925/round2/task1_trans/verify/work" "$B/review_20260925/round3/task1_trans_serving/verify/work" "$B/review_20260925/round3/task1_trans_serving/verify/results"
  run strict_validation build/review_20260925/round2/task1_trans/verify/strict_stats.py validation
  run strict_private build/review_20260925/round3/task1_trans_serving/verify/v1b_strict_private.py
  for s in validation private; do run "hm25_strict_$s" build/review_20260927/iteration2/hole_mixture_ldo.py --split $s --strict; done
  csv "$B/deliverables/2026-09-25/submission_T1_56549247.csv" "$L_V3" "$L_KSTAR" "$L_G2" "$L_RLB" "$L_G2V2" "$L_T1"
  run hm25_serve build/review_20260927/iteration2/hole_mixture_serve.py
  for s in validation private; do run "full_pool_$s" build/review_20260927_research/full_pool.py --split $s; done
  helper strength_validation task1_strength_validation.py "$T"
  csv "$B/deliverables/2026-09-27/submission_T1_HM25.csv" "$L_V3" "$L_KSTAR" "$L_G2" "$L_RLB" "$L_G2V2" "$L_T1" "$L_HM25"
  run hmp50_serve build/review_20260927_research/serve_candidate.py
fi
# 8. DK50 -> CAP on the general cells (k* re-applied inside), D12_I405 stage-2.
if step 8; then
  csv "$B/deliverables/2026-09-27/hmp50/submission_HMP50.csv" "$L_V3" "$L_KSTAR" "$L_G2" "$L_RLB" "$L_G2V2" "$L_T1" "$L_HM25" "$L_HMP50"
  run dk50 build/review_20260927_structural/build_candidate.py
  run cap build/model_capacity_20260927/build_state_candidate.py
  run d12 build/review_20260929_target092/noise_floor/d12_serve.py
fi
# 9. PRP_plus hole speeds: C4 base, per-split PR head features, D1 (c_Jan), D2 (PR head fit + official rows), keyed file.
if step 9; then
  G=build/review_20260929_target092/gap_structure
  for s in validation private; do run "c4_$s" $G/c4_phead_transfer.py $s; done
  for s in validation private; do run "g2persplit_$s" $G/g2_persplit.py $s; done
  run d1_official $G/d1_official.py
  run d2_official_prp $G/d2_official_prp.py
  mkdir -p "$B/review_20260929_target092/main"
  helper prp_keyed make_state_updates_gap_prp.py "$T/$G/d2_official_updates_prp.parquet" "$B/review_20260929_target092/main/state_updates_gap_prp.parquet"
fi
# 10. J1 density head on the R11c parent, SG1 gate (parent v < .85 vf).
if step 10; then
  csv "$B/review_20260929_target092/candidates/submission_R11c_CAP_D12_GAPPRP_ONGREFIT.csv" "$L_V3" "$L_KSTAR" "$L_G2" "$L_RLB" "$L_G2V2" "$L_T1" "$L_HM25" "$L_HMP50" "$L_DK50" "$L_CAP" "$L_D12" "$L_PRP"
  run j1_parity build/review_20260930_next/state_density/j1_official.py parity
  run j1_serve build/review_20260930_next/state_density/j1_official.py serve
  run sg1_serve build/review_20260930_next/state_density/density_gate/sg1_official.py
fi
# 11. The six final sources.
if step 11; then
  helper final_sources task1_final_sources.py "$RUN"
fi
echo "$(date +%T) task1 steps $FIRST..$LAST done"
