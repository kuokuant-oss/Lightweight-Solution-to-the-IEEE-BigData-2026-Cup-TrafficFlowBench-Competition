#!/bin/sh
# Task 4 OD: regenerate <run>/final_sources/odme.parquet from the released counts and weak priors.
#
#   pipeline/od.sh <run_root>          (run tree made by pipeline/stage_tree.py)
#
# Runs the original build_odme NNLS (lambda .05) through pipeline/od.py: 10 panels x 2 splits, each split
# from its own released link counts and weak prior. Single-threaded NNLS. Log: <run>/logs/od/od.log.
set -eu
[ $# -eq 1 ] || { echo "usage: od.sh <run_root>" >&2; exit 2; }
RUN=$(cd "$1" && pwd -P)
PKG=$(cd "$(dirname "$0")/.." && pwd -P)
# TFB_PYTHON defaults to the pinned environment; run_reproduce.sh always sets it.
export TFB_PYTHON="${TFB_PYTHON:-$HOME/.venvs/tfb-final-R26c/bin/python}"
# Thread caps for OpenMP and the BLAS libraries (2 unless OMP_NUM_THREADS is set).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS" VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
LOG="$RUN/logs/od"; mkdir -p "$LOG"
# The wrapper runs inside the run tree (the path layer maps the workspace roots onto it).
mkdir -p "$RUN/trafficflowbench/pipeline"
cp "$PKG/pipeline/od.py" "$RUN/trafficflowbench/pipeline/"
# Run the wrapper through bin/tfb-run; on failure print the log tail and stop.
printf '[%s] %-14s %s\n' "$(date +%H:%M:%S)" od pipeline/od.py
if ! /usr/bin/time -l "$PKG/bin/tfb-run" "$RUN" pipeline/od.py > "$LOG/od.log" 2>&1; then
  tail -40 "$LOG/od.log" >&2; echo "FAILED: od" >&2; exit 1
fi
# Print wall time and peak memory from the time -l lines of the log.
awk '/ real /{r=$1} /maximum resident set size/{m=$1} END{printf "           %8.1f s  peak RSS %6.0f MB\n", r, m/1048576}' "$LOG/od.log"
echo "[$(date +%H:%M:%S)] od done: $RUN/final_sources/odme.parquet"
