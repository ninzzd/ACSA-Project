#!/bin/bash
# HumanEval sweeps of MiKV, after GSM8K (gsm8k_sweeps.sh). HumanEval decodes
# code, where one perturbed token is a failing test, so it sees the precision
# of the kept cache that Line Retrieval cannot. 164 problems, greedy, pass@1.
#
#   baseline  uncompressed model (full cache): the reference line, ~11-14%
#   probe     10 problems, baseline + 1 config: s/problem, eyeball indentation
#   sanity    LOW = 16 bits -> must equal the baseline problem for problem
#   rtn       uniform int8 / int4 (no importance policy)
#   H1        MiKV curve: LOW int3/int4 x r 0.2/0.3/0.4/0.5
#   H2-H4     balancer, importance path (float vs HW), window (w=64 vs k/2)
#   tier1     = baseline + sanity + rtn + H1 (12 runs)   all = + H2-H4 (20 runs)
#
# Sweeps are defined in scripts/mikv_humaneval_sweeps.py. Cost at the default
# 35 s/problem guess: ~1.6 h per MiKV run, `all` ~31 GPU-hours (~8 h wall on 4
# GPUs); `probe` replaces the guess with a measurement.
#
# usage (on the login node, from anywhere):
#   ./humaneval_sweeps.sh dry    probe       # plan + t_p/k preview; no GPU, no code execution
#   ./humaneval_sweeps.sh submit probe       # ~15 min
#   ./humaneval_sweeps.sh merge              # -> results/humaneval/*.csv + samples/ (safe mid-sweep)
#   ./humaneval_sweeps.sh submit tier1
#   ./humaneval_sweeps.sh submit all         # again after timeouts: submits only what is left
#   ./humaneval_sweeps.sh plot               # merge + humaneval_pass_vs_kv.png
#   ./humaneval_sweeps.sh rescore <run_key>  # official evaluate_functional_correctness, as a job
#
# Environment knobs: MAX_PARALLEL, SEC_PER_PROBLEM, FORCE=1, EXCLUDE=gnode2,
#   HUMANEVAL_MODEL=<hf id already in hf_cache>  (keys runs separately)

set -euo pipefail
PROJ=/home/others/23EC10068/kvcache/ACSA-Project
RESULTS_ROOT="${HUMANEVAL_RESULTS:-$PROJ/results/humaneval}"
MODE="${1:?dry|submit|merge|plot|rescore}"

py() {
  ( cd "$PROJ" && source .venv/bin/activate \
      && HF_HOME="$PROJ/hf_cache" HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
         python scripts/mikv_humaneval.py --results-root "$RESULTS_ROOT" "$@" )
}

submit() {
  local group="$1" name stamp units info
  name="mikv_humaneval_${group//,/+}"
  if [[ -z "${FORCE:-}" ]] && squeue -h -u "$USER" -n "$name" 2>/dev/null | grep -q .; then
    echo "an array named $name is still queued or running (squeue -u $USER -n $name)." >&2
    echo "resubmitting now would duplicate its pending work; wait for it, or set FORCE=1." >&2
    exit 1
  fi
  stamp=$(date +%Y%m%d_%H%M%S)
  units="$RESULTS_ROOT/plans/${group//,/+}_${stamp}.json"
  info="$RESULTS_ROOT/plans/${group//,/+}_${stamp}.env"
  mkdir -p "$RESULTS_ROOT/plans" "$PROJ/logs"

  py plan --group "$group" --write-units "$units" --submit-info "$info" \
     ${SEC_PER_PROBLEM:+--sec-per-problem "$SEC_PER_PROBLEM"}
  # shellcheck disable=SC1090
  source "$info"   # NUNITS, TIME, GPU_HOURS
  if (( NUNITS == 0 )); then
    echo "=== $group: nothing pending -- ./humaneval_sweeps.sh merge ==="
    rm -f "$units" "$info"
    return
  fi
  echo "=== submitting $group: $NUNITS tasks, --time $TIME each, ~$GPU_HOURS GPU-hours ==="
  ( cd "$PROJ" && sbatch --job-name="$name" \
      --array="0-$((NUNITS - 1))${MAX_PARALLEL:+%$MAX_PARALLEL}" \
      --time="$TIME" \
      ${EXCLUDE:+--exclude="$EXCLUDE"} \
      --export=ALL,HUMANEVAL_RESULTS="$RESULTS_ROOT" \
      jobs_humaneval.sh "$units" )
}

rescore() {
  local key="${1:?run_key (see humaneval_summary.csv)}" file
  file="$RESULTS_ROOT/samples/$key.jsonl"
  [[ -s "$file" ]] || { echo "no $file -- run ./humaneval_sweeps.sh merge first" >&2; exit 1; }
  ( cd "$PROJ" && sbatch --job-name="mikv_humaneval_rescore" --time=00:30:00 \
      ${EXCLUDE:+--exclude="$EXCLUDE"} \
      --output="$PROJ/logs/humaneval_rescore_${key}_%j.out" \
      --error="$PROJ/logs/humaneval_rescore_${key}_%j.err" \
      jobs_humaneval.sh --rescore "$file" )
}

case "$MODE" in
  dry)     py plan --group "${2:?group: probe|baseline|sanity|rtn|H1..H4|tier1|all}" \
              ${SEC_PER_PROBLEM:+--sec-per-problem "$SEC_PER_PROBLEM"} ;;
  submit)  submit "${2:?group: probe|baseline|sanity|rtn|H1..H4|tier1|all}" ;;
  merge)   py merge ;;
  plot)    py merge --plots ;;
  rescore) rescore "${2:-}" ;;
  *)       echo "unknown mode: $MODE (dry|submit|merge|plot|rescore)" >&2; exit 1 ;;
esac
