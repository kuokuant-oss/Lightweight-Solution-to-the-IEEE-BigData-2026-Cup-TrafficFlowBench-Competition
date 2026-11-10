#!/bin/sh
# Truncation audit of Task 2: for each cut in audit/cuts.csv, remove every observation stamped after the
# cut window's forecast origin τ, rerun Task 2 on that view, and compare the onset and ongoing
# predictions of every window with origin <= τ with the full run, cell by cell.
#
#   audit/run_truncation.sh --release <kaggle_public> --ref <full-run out_dir> --work <dir> \
#       [--python <python3.12>] [--jobs 3] [--only c01,c02] [--results <dir>]
#
# --ref   the --out folder of a full or Task-2-only run on the released data, with its run tree kept
#         (run/final_sources/{onset,ongoing}.parquet and the probability files under run/trafficflowbench/
#         are the reference)
# --work  a new folder; per cut it holds view_<name>/ (the truncated view), out_<name>/ (the run, with
#         the copied code tree removed afterwards), <name>.log and <name>.json (the comparison)
# --results  checkpoint folder (default audit/results): cut_<name>.json is written as soon as a cut
#         finishes, and progress.log gets one line per cut start and end. A cut whose cut_<name>.json
#         exists is skipped, so an interrupted run resumes where it stopped (a cut that finished in
#         --work but has no checkpoint yet is imported without a rerun).
# After a cut is compared, its view and the run's code copy are deleted; out_<name>/ keeps the logs,
# final sources, probability files (probs/) and the view's TRUNCATION.json.
#
# Per cut: audit/truncate.py builds the view (--drop-stamps from cuts.csv: 0 for a cut, 1 or 2 for a
# negative control); run_reproduce.sh --task2-only serves the splits that hold windows up to τ (validation
# for a March cut, both for an April cut); audit/compare.py copies the probability files to
# out_<name>/probs/ and compares decisions and probabilities. The
# assembly step stops by design on a view (queue rows of the windows after τ have no prediction), so
# the run's exit status is not used; a missing onset or ongoing source fails the cut.
# Finally audit/summarize.py writes summary.md and summary.json from the checkpoints.
set -eu
AUD=$(cd "$(dirname "$0")" && pwd -P); PKG=$(dirname "$AUD")
REL=""; REF=""; WORK=""; PY=$(command -v python3 || true); JOBS=3; ONLY=""; ONE=""; RES="$AUD/results"
while [ $# -gt 0 ]; do
  case "$1" in
    --release) REL=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    --results) RES=$2; shift 2 ;;
    --one) ONE=$2; shift 2 ;;   # internal: run one cut (used by the parallel driver)
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$REL" ] && [ -n "$REF" ] && [ -n "$WORK" ] || { echo "usage: run_truncation.sh --release <dir> --ref <dir> --work <dir> [--python <py>] [--jobs n] [--only a,b]" >&2; exit 2; }

if [ -n "$ONE" ]; then
  # One cut: look up its window, origin and splits, build the view, run, compare.
  mkdir -p "$RES"; CK="$RES/cut_$ONE.json"; PL="$RES/progress.log"
  # Writes the checkpoint from <work>/<name>.json and logs it ($1: done or imported).
  checkpoint() {
    "$PY" "$AUD/summarize.py" --checkpoint "$WORK/$ONE.json" "$ONE" "$CK"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $ONE $1 $("$PY" "$AUD/summarize.py" --line "$CK")" >> "$PL"
  }
  if [ -f "$CK" ]; then echo "[$(date +%H:%M:%S)] $ONE skipped: $CK exists"; exit 0; fi
  if [ -f "$WORK/$ONE.json" ] && grep -q wall_s "$WORK/$ONE.json"; then checkpoint imported; echo "[$(date +%H:%M:%S)] $ONE imported from $WORK"; exit 0; fi
  rm -rf "$WORK/view_$ONE" "$WORK/out_$ONE"   # leftovers of an interrupted attempt
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $ONE start" >> "$PL"
  line=$(grep "^$ONE," "$AUD/cuts.csv"); wid=$(echo "$line" | cut -d, -f2); stamps=$(echo "$line" | cut -d, -f3)
  cut=$("$PY" -c "import pandas as pd,sys,glob; w=pd.concat([pd.read_csv(f) for f in glob.glob(sys.argv[1]+'/task2/*/*/window_index.csv')]); print(w.set_index('window_id').forecast_origin[sys.argv[2]])" "$REL" "$wid")
  case "$wid" in *_validation_*) splits="validation" ;; *) splits="validation private" ;; esac
  S="--drop-stamps $stamps"
  LOG="$WORK/$ONE.log"; T0=$(date +%s)
  echo "[$(date +%H:%M:%S)] $ONE start: window $wid, cut $cut, splits '$splits' $S"
  "$PY" "$AUD/truncate.py" --release "$REL" --out "$WORK/view_$ONE" --cut "$cut" $S > "$LOG" 2>&1
  "$PKG/run_reproduce.sh" --task2-only --splits "$splits" --data "$WORK/view_$ONE" --out "$WORK/out_$ONE" --python "$PY" >> "$LOG" 2>&1 || true
  for f in onset ongoing; do
    [ -f "$WORK/out_$ONE/run/final_sources/$f.parquet" ] || { echo "$ONE: no $f source (see $LOG)" >&2; exit 1; }
  done
  # Keep the probability files (a few MB) before the run tree is deleted, then compare decisions and probabilities.
  "$PY" "$AUD/compare.py" --copy-probs "$WORK/out_$ONE/run/trafficflowbench" "$WORK/out_$ONE/probs" $splits
  "$PY" "$AUD/compare.py" --release "$REL" --view "$WORK/view_$ONE" --run "$WORK/out_$ONE" --ref "$REF" --cut "$cut" $S \
    --probs "$WORK/out_$ONE/probs" --json "$WORK/$ONE.json" >> "$LOG" 2>&1
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); d['wall_s']=int(sys.argv[2]); d['window_id']=sys.argv[3]; d['splits']=sys.argv[4]; json.dump(d,open(sys.argv[1],'w'),indent=1)" \
    "$WORK/$ONE.json" "$(( $(date +%s) - T0 ))" "$wid" "$splits"
  rm -rf "$WORK/out_$ONE/run/trafficflowbench"   # the 1.4 GB code and model copy; logs and sources stay
  # The view takes about 10 GB on a drive with 512 KB clusters (one cluster per link); keep its record only.
  cp "$WORK/view_$ONE/TRUNCATION.json" "$WORK/out_$ONE/TRUNCATION.json"; rm -rf "$WORK/view_$ONE"
  checkpoint done
  echo "[$(date +%H:%M:%S)] $ONE done in $(( $(date +%s) - T0 )) s"
  exit 0
fi

mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd -P); REL=$(cd "$REL" && pwd -P); REF=$(cd "$REF" && pwd -P)
PY=$(cd "$(dirname "$PY")" && pwd -P)/$(basename "$PY")
names=$(tail -n +2 "$AUD/cuts.csv" | cut -d, -f1)
[ -z "$ONLY" ] || names=$(echo "$ONLY" | tr ',' ' ')
T0=$(date +%s)
# Each cut runs in its own shell; JOBS cuts at a time.
for n in $names; do echo "$n"; done | xargs -n 1 -P "$JOBS" sh "$0" --release "$REL" --ref "$REF" --work "$WORK" --python "$PY" --results "$RES" --one
echo "all cuts done in $(( ($(date +%s) - T0) / 60 )) min"
"$PY" "$AUD/summarize.py" "$RES"
