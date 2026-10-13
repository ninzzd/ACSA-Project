#!/bin/bash
# GSM8K sweeps of MiKV, the follow-up to the Line Retrieval sweeps
# (followup_sweeps.sh). Line Retrieval's short answers exercise the policy almost
# only at prefill; GSM8K's 100-250-token chains of thought exercise the decode
# path -- score accumulation, the age decay, sticky demotion -- on every token.
#
#   baseline  uncompressed model on the 250 screen problems: the paired reference
#   probe     1 config + baseline on 20 problems: measures s/problem, sanity check
#   S1        operating point, swept in full: LOW int3/int4 x r 0.05-0.25
#   S2-S6     HIGH tier, age LUT/decay, window, balancer, score format -- each
#             one knob at the stress point (int3, r = 0.25), 250 fixed problems
#   S7        the final combination on all 1319 problems -- edit it in
#             scripts/mikv_gsm8k_sweeps.py once S1-S6 are in
#   tier1     = baseline + S1-S5 (~17 runs)    all = baseline + S1-S6 (~20 runs)
#   S1_full.. S6_full: the Line Retrieval follow-ups value for value (group
#             `full`); only for adding points back where a lean result is
#             borderline -- runs already done are reused, not repeated
#
# Cost at the default 30 s/problem guess (the probe replaces it with a
# measurement): `all` ~43 GPU-hours, S7 ~23 more. Each array task is ~3-4 h, so
# with 4 GPUs free on gnode2-5 `all` finishes in roughly 11-12 h of wall time.
#
# The sweeps themselves are defined in scripts/mikv_gsm8k_sweeps.py (one place,
# like mikv_grids.py); this script only plans, submits and merges them.
#
# usage (from anywhere; everything runs relative to $PROJ):
#   ./gsm8k_sweeps.sh prep               # once, login node: download the GSM8K test set
#   ./gsm8k_sweeps.sh nodecheck          # 1 tiny job per gnode: which ones have working CUDA
#   ./gsm8k_sweeps.sh dry    probe       # print the plan, progress and k preview; no GPU
#   ./gsm8k_sweeps.sh submit probe       # ~15 min: measures s/problem for both paths
#   ./gsm8k_sweeps.sh merge              # -> results/gsm8k/*.csv (run any time, safe mid-sweep)
#   ./gsm8k_sweeps.sh submit all         # one job array over baseline + S1-S6
#   ./gsm8k_sweeps.sh submit S7          # after editing the final combination
#   ./gsm8k_sweeps.sh submit all         # again, if tasks timed out: submits only what is left
#   ./gsm8k_sweeps.sh submit S3_full     # optional: add points back to a borderline sweep
#
# Environment knobs:
#   MAX_PARALLEL=8        cap on concurrently running array tasks (default: no cap)
#   SEC_PER_PROBLEM=45    override the measured MiKV s/problem used to size --time
#   PROBLEMS_PER_TASK=330 largest unit (default 330: 250-problem runs are one task,
#                         1319-problem runs split into four)
#   FORCE=1               submit even if an array for the same group is still queued
#   EXCLUDE=gnode2        keep tasks off nodes (comma list) -- e.g. one nodecheck failed
#   NODES="gnode3 gnode4" which nodes nodecheck tests (default gnode2-5)
#   LR_CSV='results/followup/*/legacy/sweep.csv'   Line Retrieval CSVs merge joins on
#
# How the work is split: one array task per (run, <=330 problems), each ~2-4 h at
# the default estimate, with --time sized to the slowest unit x 1.5 + 20 min (and
# capped under the 24 h limit). Short tasks backfill into gaps on gnode2-5 far
# sooner than 24 h jobs, run in parallel on every free GPU, and a failure costs
# one task rather than a day. Every task checkpoints after each problem.

set -euo pipefail
PROJ=/home/others/23EC10068/kvcache/ACSA-Project
RESULTS_ROOT="${GSM8K_RESULTS:-$PROJ/results/gsm8k}"
DATA="$PROJ/data/gsm8k/test.jsonl"
GSM8K_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MODE="${1:?prep|nodecheck|dry|submit|merge}"

py() {
  ( cd "$PROJ" && source .venv/bin/activate \
      && HF_HOME="$PROJ/hf_cache" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
         python scripts/mikv_gsm8k.py --results-root "$RESULTS_ROOT" "$@" )
}

prep() {
  mkdir -p "$(dirname "$DATA")"
  if [[ -s "$DATA" ]]; then
    echo "already present: $DATA"
  elif curl -fsSL --retry 3 "$GSM8K_URL" -o "$DATA.tmp"; then
    mv "$DATA.tmp" "$DATA"
  else
    rm -f "$DATA.tmp"
    echo "direct download failed; trying the HF hub (openai/gsm8k)..."
    ( cd "$PROJ" && source .venv/bin/activate && HF_HOME="$PROJ/hf_cache" python - "$DATA" <<'EOF'
import json, sys
from datasets import load_dataset
rows = load_dataset("openai/gsm8k", "main", split="test")
with open(sys.argv[1], "w") as f:
    for r in rows:
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
EOF
    )
  fi
  local n
  n=$(grep -c . "$DATA")
  if [[ "$n" -ne 1319 ]]; then
    echo "ERROR: $DATA has $n problems, expected 1319" >&2; exit 1
  fi
  echo "ok: $DATA ($n problems)"
  sha256sum "$DATA"
}

submit() {
  local group="$1" name stamp units info
  name="mikv_gsm8k_${group//,/+}"
  if [[ -z "${FORCE:-}" ]] && squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "an array named $name is still queued or running (squeue -u $USER -n $name)."
    echo "resubmitting now would duplicate its pending work; wait for it, or set FORCE=1." >&2
    exit 1
  fi
  stamp=$(date +%Y%m%d_%H%M%S)
  units="$RESULTS_ROOT/plans/${group//,/+}_${stamp}.json"
  info="$RESULTS_ROOT/plans/${group//,/+}_${stamp}.env"
  mkdir -p "$RESULTS_ROOT/plans" "$PROJ/logs"

  py plan --group "$group" --write-units "$units" --submit-info "$info" \
     ${PROBLEMS_PER_TASK:+--problems-per-task "$PROBLEMS_PER_TASK"} \
     ${SEC_PER_PROBLEM:+--sec-per-problem "$SEC_PER_PROBLEM"}
  # shellcheck disable=SC1090
  source "$info"   # NUNITS, TIME, GPU_HOURS -- written by the plan above
  if (( NUNITS == 0 )); then
    echo "=== $group: nothing pending, every run is complete -- ./gsm8k_sweeps.sh merge ==="
    rm -f "$units" "$info"
    return
  fi
  echo "=== submitting $group: $NUNITS tasks, --time $TIME each, ~$GPU_HOURS GPU-hours ==="
  ( cd "$PROJ" && sbatch --job-name="$name" \
      --array="0-$((NUNITS - 1))${MAX_PARALLEL:+%$MAX_PARALLEL}" \
      --time="$TIME" \
      ${EXCLUDE:+--exclude="$EXCLUDE"} \
      --export=ALL,GSM8K_RESULTS="$RESULTS_ROOT" \
      jobs_gsm8k.sh "$units" )
}

# One tiny job pinned to each node, running only the CUDA gate. A node with a
# broken driver fails every task sent to it within seconds, so an array landing
# there drains into it; find such nodes first and keep them out with EXCLUDE.
nodecheck() {
  local n
  mkdir -p "$PROJ/logs"
  for n in ${NODES:-gnode2 gnode3 gnode4 gnode5}; do
    ( cd "$PROJ" && sbatch --job-name="mikv_nodecheck" --nodelist="$n" \
        --time=00:10:00 --mem=8gb --cpus-per-task=1 \
        --output="$PROJ/logs/nodecheck_${n}_%j.out" --error="$PROJ/logs/nodecheck_${n}_%j.err" \
        jobs_gsm8k.sh --check )
  done
  echo "when they finish (squeue -u $USER -n mikv_nodecheck is empty):"
  echo "  grep -h -E '^cuda +:' $PROJ/logs/nodecheck_*.out"
}

case "$MODE" in
  prep)   prep ;;
  nodecheck) nodecheck ;;
  dry)    py plan --group "${2:?group: probe|baseline|S1..S7|tier1|all}" \
             ${PROBLEMS_PER_TASK:+--problems-per-task "$PROBLEMS_PER_TASK"} \
             ${SEC_PER_PROBLEM:+--sec-per-problem "$SEC_PER_PROBLEM"} ;;
  submit) submit "${2:?group: probe|baseline|S1..S7|tier1|all}" ;;
  merge)  py merge --line-retrieval-csv "${LR_CSV:-$PROJ/results/followup/*/legacy/sweep.csv}" ;;
  *)      echo "unknown mode: $MODE (prep|nodecheck|dry|submit|merge)" >&2; exit 1 ;;
esac
