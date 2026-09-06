#!/usr/bin/env bash
#
# Launch a MiKV configuration sweep.
#
# A thin wrapper around `scripts/mikv.py`: it picks the sweep type, gives each
# type its own results directory so runs do not interleave in one CSV, shows the
# plan before spending anything, and can put a multi-hour sweep in the background
# so it survives a dropped connection.
#
#   ./scripts/run_sweep.sh greedy                    # the recommended pass, ~3 h
#   ./scripts/run_sweep.sh greedy --dry              # cost it, load no model
#   ./scripts/run_sweep.sh confirm --low-bits 2,4    # any mikv.py flag passes through
#   ./scripts/run_sweep.sh greedy --bg               # detach; tail the log it prints
#   ./scripts/run_sweep.sh list                      # what sweep types exist
#
# Everything after the sweep type is forwarded to mikv.py verbatim, so an axis
# override (--low-bits, --window-tokens, --age-lut-entries, ...) narrows or
# widens whichever preset was named. See `./scripts/run_sweep.sh greedy --help`.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PYTHON="${PYTHON:-python3}"
RESULTS="${RESULTS_DIR:-$ROOT/results}"

# Preset names are read out of the source rather than by importing it: importing
# mikv_grids pulls in mikv_config -> torch + transformers, which is ~10 s of load
# time to print a list. `mikv.py --help` remains the authoritative list.
presets() { sed -n 's/^    "\([a-z]*\)": dict(.*/\1/p' "$HERE/mikv_grids.py"; }

usage() {
    cat <<EOF
usage: run_sweep.sh <sweep-type> [--dry] [--bg] [mikv.py flags...]
       run_sweep.sh list

sweep types:
$(presets | sed 's/^/    /')

options handled here:
    --dry     print the run plan and exit without loading the model
    --bg      detach the run (nohup) and print the log path to tail
    --        stop interpreting options here; forward the rest verbatim

Every other flag is forwarded to mikv.py unchanged -- run
    ./scripts/run_sweep.sh <type> --help
for the full axis list.

environment:
    PYTHON        interpreter to use            (default: python3)
    RESULTS_DIR   where CSV/figures are written (default: <repo>/results)
EOF
}

[ $# -ge 1 ] || { usage; exit 2; }

case "$1" in
    list)             presets; exit 0 ;;
    -h|--help|help)   usage; exit 0 ;;
esac

SWEEP="$1"; shift
if ! presets | grep -qx "$SWEEP"; then
    echo "run_sweep.sh: unknown sweep type '$SWEEP'" >&2
    echo "known types: $(presets | tr '\n' ' ')" >&2
    exit 2
fi

DRY=0
BG=0
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry|--dry-run) DRY=1; shift ;;
        --bg|--background) BG=1; shift ;;
        --) shift; ARGS+=("$@"); break ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

# One directory per sweep type. The CSV is appended to across runs by design, so
# keeping types apart is what stops a greedy pass and a confirmation grid from
# landing in the same file with no way to tell them apart afterwards. (Every row
# does carry its own run_id and full configuration, so this is convenience, not
# correctness.)
OUT="$RESULTS/$SWEEP"
mkdir -p "$OUT"

cd "$ROOT"
COMMON=(--preset "$SWEEP" --csv "$OUT/sweep.csv" --plot "$OUT/sweep.png")

if [ "$DRY" -eq 1 ]; then
    exec "$PYTHON" scripts/mikv.py "${COMMON[@]}" --dry-run ${ARGS[@]+"${ARGS[@]}"}
fi

# Always cost it first. A sweep is hours of GPU time and the plan is free.
echo "=== plan ==="
"$PYTHON" scripts/mikv.py "${COMMON[@]}" --dry-run ${ARGS[@]+"${ARGS[@]}"}
echo

if [ "$BG" -eq 1 ]; then
    LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"
    nohup "$PYTHON" scripts/mikv.py "${COMMON[@]}" ${ARGS[@]+"${ARGS[@]}"} >"$LOG" 2>&1 &
    echo "started '$SWEEP' in the background (pid $!)"
    echo "  log:     tail -f $LOG"
    echo "  results: $OUT/sweep.csv  (appended as each run completes)"
    exit 0
fi

# Interactive only: a redirected/CI invocation should not block on a prompt.
if [ -t 0 ]; then
    read -r -p "run '$SWEEP'? [y/N] " reply
    case "$reply" in [yY]*) ;; *) echo "aborted"; exit 1 ;; esac
fi

echo "=== running '$SWEEP' ==="
echo "results accumulate in $OUT/sweep.csv as each run completes"
exec "$PYTHON" scripts/mikv.py "${COMMON[@]}" ${ARGS[@]+"${ARGS[@]}"}
