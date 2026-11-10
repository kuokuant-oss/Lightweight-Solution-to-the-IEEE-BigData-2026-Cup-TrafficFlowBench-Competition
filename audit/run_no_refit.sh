#!/bin/sh
# No-refit check of Task 2: run onset.sh and ongoing.sh on the released data with a probe that records
# every model-fitting call and every run-tree module each Python process loads; then list the fit/train
# calls written in that code and whether any of them ran.
#
#   audit/run_no_refit.sh --release <kaggle_public> --out <new dir> [--python <python3.12>]
#
# The probe (audit/probe/tfb_fit_probe.py) is added to the run tree's copy of
# scripts/ssd_bootstrap/sitecustomize.py only; the package is not changed. It records and passes calls
# through, so the run's outputs are the normal ones (they are compared with --ref by audit/no_refit_report.py).
set -eu
AUD=$(cd "$(dirname "$0")" && pwd -P); PKG=$(dirname "$AUD")
REL=""; OUT=""; PY=$(command -v python3 || true)
while [ $# -gt 0 ]; do
  case "$1" in
    --release) REL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$REL" ] && [ -n "$OUT" ] || { echo "usage: run_no_refit.sh --release <dir> --out <dir> [--python <py>]" >&2; exit 2; }
PY=$(cd "$(dirname "$PY")" && pwd -P)/$(basename "$PY")
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd -P); RUN="$OUT/run"
[ ! -e "$RUN" ] || { echo "refusing to reuse $RUN" >&2; exit 2; }
"$PY" "$PKG/pipeline/stage_tree.py" --run "$RUN" --data "$REL"
mkdir -p "$RUN/logs"
BOOT="$RUN/trafficflowbench/scripts/ssd_bootstrap"
cp "$AUD/probe/tfb_fit_probe.py" "$BOOT/"
printf '\n# audit/run_no_refit.sh: fit probe (run tree copy only)\nif os.environ.get("TFB_AUDIT_PROBE"):\n    import tfb_fit_probe\n' >> "$BOOT/sitecustomize.py"
export TFB_PYTHON="$PY" TFB_AUDIT_PROBE="$OUT/probe.jsonl"
T0=$(date +%s)
sh "$PKG/pipeline/onset.sh" "$RUN"
sh "$PKG/pipeline/ongoing.sh" "$RUN"
echo "Task 2 segments done in $(( $(date +%s) - T0 )) s"
