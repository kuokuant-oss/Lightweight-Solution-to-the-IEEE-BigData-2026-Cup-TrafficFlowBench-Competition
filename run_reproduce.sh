#!/bin/sh
# Rebuild the selected submission from the released data.
#
#   ./run_reproduce.sh --data /path/to/kaggle_public --out /path/to/output_dir [--python /path/to/python3.12] [--no-check]
#   ./run_reproduce.sh --task2-only --data /path/to/new_month --out /path/to/output_dir [--splits "validation private"]
#
# --data    the released competition folder that contains submission_key.csv, sample_submission.csv,
#           corridors/, task1/, task2/ and task4/
# --out     a new or empty folder; the run tree, logs and submission.csv are written here
# --python  interpreter with the packages of requirements.txt (default: python3 on PATH)
# --no-check  do not compare the result with the submitted file's SHA256 (for data other than the
#           released competition data)
# --task2-only  run only Task 2 (steps 1, 3, 4 and 6; implies --no-check). submission.csv keeps the full
#           layout, with the queue rows predicted and the state and odme rows set to 0. Use it for a
#           newly generated month: Task 1 checks the SHA256 and row counts of the released data.
# --splits  the Task 2 splits to serve, space-separated, in order (default "validation private"; the
#           same as setting TFB_SPLITS). Every queue row of submission_key.csv must belong to one of them.
#
# Steps (each logged under <out>/run/logs/):
#   1. stage a fresh run tree: <out>/run/trafficflowbench (copy of this package's tree) and
#      <out>/run/tfb_competition_data/kaggle_public -> --data (read only)
#   2. Task 1/3 traffic state       pipeline/task1.sh    -> final_sources/state_{1..6}_*.parquet
#   3. Task 2 queue onset           pipeline/onset.sh    -> final_sources/onset.parquet
#   4. Task 2 queue ongoing         pipeline/ongoing.sh  -> final_sources/ongoing.parquet
#   5. Task 4 path flows            pipeline/od.sh       -> final_sources/odme.parquet
#   6. assemble <out>/submission.csv in submission_key order and print its SHA256
set -eu
PKG=$(cd "$(dirname "$0")" && pwd -P)
# The comment block at the top of this file is the --help text.
# EXPECT: SHA256 of the selected submission file (the value step 6 compares with).
EXPECT=cf5869e23d8bf063d37e6636355f66b8385be1ee7e7257a9943a16783ac00fc5
DATA=""; OUT=""; PY=$(command -v python3 || true); CHECK=1; TASK2_ONLY=0
# Parse the command-line options.
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --no-check) CHECK=0; shift ;;
    --task2-only) TASK2_ONLY=1; CHECK=0; shift ;;
    --splits) export TFB_SPLITS="$2"; shift 2 ;;
    -h|--help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
# Both folders are required, and the interpreter must exist and be executable.
[ -n "$DATA" ] && [ -n "$OUT" ] || { echo "usage: run_reproduce.sh --data <kaggle_public> --out <dir> [--python <py>] [--no-check] [--task2-only] [--splits <list>]" >&2; exit 2; }
[ -n "$PY" ] && [ -x "$PY" ] || { echo "no Python interpreter; pass --python" >&2; exit 2; }
# Absolute interpreter path, so the segment scripts run the same Python from any folder.
PY=$(cd "$(dirname "$PY")" && pwd -P)/$(basename "$PY")
# Check Python 3.12 and the exact package versions of requirements.txt before any work.
"$PY" - <<'EOF' || exit 2
import sys
if sys.version_info[:2] != (3, 12):
    sys.exit(f"Python 3.12 is required, found {sys.version.split()[0]}")
import importlib.metadata as md
want = {"numpy": "2.5.3", "pandas": "2.3.3", "scipy": "1.18.1", "pyarrow": "25.0.1",
        "scikit-learn": "1.9.1", "lightgbm": "4.7.0", "catboost": "1.2.10"}
bad = []
for p, v in want.items():
    try:
        got = md.version(p)
    except md.PackageNotFoundError:
        got = "missing"
    if got != v:
        bad.append(f"{p} {got} (need {v})")
if bad:
    sys.exit("install requirements.txt first: " + "; ".join(bad))
EOF
# Resolve --out and --data to physical absolute paths; the run tree is <out>/run.
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd -P); DATA=$(cd "$DATA" && pwd -P)
RUN="$OUT/run"
# Never reuse an earlier run tree or overwrite an earlier submission.
[ ! -e "$RUN" ] || { echo "refusing to reuse $RUN; give a new --out" >&2; exit 2; }
[ ! -e "$OUT/submission.csv" ] || { echo "refusing to overwrite $OUT/submission.csv" >&2; exit 2; }
# The segment scripts and bin/tfb-run read the interpreter from TFB_PYTHON.
export TFB_PYTHON="$PY"
T0=$(date +%s)
# Prints one progress line prefixed with the wall-clock time.
# Arguments:
#   Words of the message.
# Outputs:
#   Writes "[HH:MM:SS] <message>" to stdout.
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

stamp "1/6 staging run tree"
"$PY" "$PKG/pipeline/stage_tree.py" --run "$RUN" --data "$DATA"
mkdir -p "$RUN/logs"
# Task 2 reads no Task 1 or Task 4 output, so --task2-only skips steps 2 and 5.
if [ "$TASK2_ONLY" -eq 1 ]; then stamp "2/6 Task 1/3 state: skipped (--task2-only)"
else stamp "2/6 Task 1/3 state";  sh "$PKG/pipeline/task1.sh" "$RUN"; fi
stamp "3/6 Task 2 onset (splits: ${TFB_SPLITS:-validation private})";    sh "$PKG/pipeline/onset.sh" "$RUN"
stamp "4/6 Task 2 ongoing";  sh "$PKG/pipeline/ongoing.sh" "$RUN"
if [ "$TASK2_ONLY" -eq 1 ]; then stamp "5/6 Task 4 OD: skipped (--task2-only)"
else stamp "5/6 Task 4 OD";       sh "$PKG/pipeline/od.sh" "$RUN"; fi
stamp "6/6 assembling submission.csv"
# The final per-row sources written by the segments. assemble.py applies the six state files in
# this order (a later file overrides an earlier one on the ids it lists) and checks the SHA256.
S="$RUN/final_sources"
if [ "$TASK2_ONLY" -eq 1 ]; then
  set -- --data "$DATA" --out "$OUT/submission.csv" --queue-only --onset "$S/onset.parquet" --ongoing "$S/ongoing.parquet"
else
set -- --data "$DATA" --out "$OUT/submission.csv" \
  --state "$S/state_1_g2v2.parquet" --state "$S/state_2_dk50.parquet" --state "$S/state_3_cap.parquet" \
  --state "$S/state_4_d12.parquet" --state "$S/state_5_prp.parquet" --state "$S/state_6_sg1.parquet" \
  --onset "$S/onset.parquet" --ongoing "$S/ongoing.parquet" --odme "$S/odme.parquet"
fi
if [ "$CHECK" -eq 1 ]; then set -- "$@" --expect-sha "$EXPECT"; fi
"$PY" "$PKG/pipeline/assemble.py" "$@"
stamp "done in $(( ($(date +%s) - T0) / 60 )) min: $OUT/submission.csv"
