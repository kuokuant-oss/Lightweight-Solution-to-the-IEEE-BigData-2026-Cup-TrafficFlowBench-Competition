#!/bin/sh
# Task 2 onset: regenerate <run>/final_sources/onset.parquet from the released data.
#
#   pipeline/onset.sh <run_root>          (run tree made by pipeline/stage_tree.py)
#
# Every program runs through bin/tfb-run (path layer on). Serving reads only each window's released
# history (T-60..T-5) and the masked layer at the origin stamp T; the models are frozen train-derived
# artifacts and nothing is fitted. No raw February 2031 train file is read; the only raw train file read
# is each panel's first train day (2030-06-01), for link_id/milepost order (task2infer.features_for_split).
#   1. 0922 head (M1) served on the official windows, then the panel_shrink25 decoder (M2)
#   2. D: u-model site choice (C3) on the multi-site panels, block decoding t=.7 on single-site panels
#   3. stage1: FE site choice + A1 template on D12_I5_N, D12_I5_S, D7_I405_S
#   4. key mapping to submission_id
# Per-step logs: <run>/logs/onset/<step>.log (with /usr/bin/time -l).
#
# Names: "0922" marks the frozen onset model of 2026-09-22. "D" is the candidate
# submission_D_onsetsite_blocks (site choice on the multi-site panels, block decoding elsewhere).
# The u-models score each candidate bottleneck site of a window; FE is the within-window site choice
# and A1 the train template of the queue extent at the chosen site.
set -eu
[ $# -eq 1 ] || { echo "usage: onset.sh <run_root>" >&2; exit 2; }
RUN=$(cd "$1" && pwd -P)
PKG=$(cd "$(dirname "$0")/.." && pwd -P)
TFB="$PKG/bin/tfb-run"
# TFB_PYTHON defaults to the pinned environment; run_reproduce.sh always sets it.
export TFB_PYTHON="${TFB_PYTHON:-$HOME/.venvs/tfb-final-R26c/bin/python}"
# Thread caps for OpenMP and the BLAS libraries (2 unless OMP_NUM_THREADS is set).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS" VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
LOG="$RUN/logs/onset"; mkdir -p "$LOG"
# SPLITS: the released splits to serve, in order (TFB_SPLITS, default "validation private"; see README).
export TFB_SPLITS="${TFB_SPLITS:-validation private}"; SPLITS=$TFB_SPLITS

# Runs one program through bin/tfb-run and prints its wall time and peak memory.
# Arguments:
#   Step name (log file name), then a program path relative to trafficflowbench/ or "-c <code>",
#   then the program's arguments.
# Outputs:
#   Log in $LOG/<name>.log; on failure the last 40 log lines go to stderr and the segment stops.
step() {  # step <name> <program.py | -c code> [args...]
  name=$1; shift
  printf '[%s] %-14s %s\n' "$(date +%H:%M:%S)" "$name" "$*"
  if ! /usr/bin/time -l "$TFB" "$RUN" "$@" > "$LOG/$name.log" 2>&1; then
    tail -40 "$LOG/$name.log" >&2; echo "FAILED: $name" >&2; exit 1
  fi
  awk '/ real /{r=$1} /maximum resident set size/{m=$1} END{printf "           %8.1f s  peak RSS %6.0f MB\n", r, m/1048576}' "$LOG/$name.log"
}

# The pipeline wrappers run inside the run tree (the path layer maps the workspace roots onto it).
mkdir -p "$RUN/trafficflowbench/pipeline"
for w in onset_head_serve onset_c3 onset_sources; do cp "$PKG/pipeline/$w.py" "$RUN/trafficflowbench/pipeline/"; done

# 1. 0922 onset head: 145's final-branch serving with the frozen model, then the panel_shrink25 decode
step head_serve pipeline/onset_head_serve.py                                    # -> prior/final_feature_hazard_serving.parquet
step head_decode scripts/audit_strongest_onset_selected_20260922.py production  # -> onset/production/onset_component.parquet

# 2. D: masked-layer caches (validation, private) for the multi-site panels, C3 site choice, block decoding
for p in D12_I5_N D12_I5_S D7_I210_W D7_I405_S; do
  for s in $SPLITS; do
    step "cache_${p}_$s" build/review_20260924_ctm/cache/build_cache.py "$p" "$s"
  done
done
step c3 pipeline/onset_c3.py                                                    # -> work/onset_site/onset_component_submission_C3_*.parquet
# build_D_inputs.py reads and writes relative to its own folder.
step d_inputs -c "import os, runpy; os.chdir('build/review_20260924_newworld/work/blocks'); runpy.run_path('build_D_inputs.py', run_name='__main__')"  # -> D_onset_real{val,priv}.parquet

# 3. stage1: FE site choice + A1 template on the three D multi-site panels
step stage1_fea1 build/review_20260925/round2/synthesis/sv2_official_FEA1.py   # -> synthesis/onset_FEA1_official.parquet

# 4. final per-row source
step onset_source pipeline/onset_sources.py                                     # -> <run>/final_sources/onset.parquet
echo "[$(date +%H:%M:%S)] onset done: $RUN/final_sources/onset.parquet"
