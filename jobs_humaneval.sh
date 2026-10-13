#!/bin/bash
#SBATCH --job-name=mikv_humaneval
#SBATCH --output=/home/others/23EC10068/kvcache/ACSA-Project/logs/humaneval_%A_%a.out
#SBATCH --error=/home/others/23EC10068/kvcache/ACSA-Project/logs/humaneval_%A_%a.err
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#
# One array task of a HumanEval sweep: runs unit $SLURM_ARRAY_TASK_ID of a frozen
# units file (see scripts/mikv_humaneval.py). Submit through humaneval_sweeps.sh,
# which plans the units, sizes --time and passes the --array range:
#
#   ./humaneval_sweeps.sh submit probe
#
# By hand, one unit of an existing plan:
#   sbatch --array=3 --time=03:00:00 jobs_humaneval.sh results/humaneval/plans/tier1_<stamp>.json
#
# Other modes:
#   jobs_humaneval.sh --check                      CUDA gate only (see gsm8k_sweeps.sh nodecheck)
#   jobs_humaneval.sh --rescore <samples.jsonl>    official evaluate_functional_correctness
#
# THIS TASK EXECUTES MODEL-GENERATED CODE (human_eval.execution, per problem).
# That is why scoring lives here and never in `merge` on the login node. The
# harness runs each program in a subprocess with resource limits inside a
# tempdir; TMPDIR is pointed at node-local scratch below so those tempdirs never
# land in the project tree, and the scratch is removed on exit.
#
# Do NOT add --nodelist=gnode[2-5] (see jobs_gsm8k.sh); use --exclude.

set -euo pipefail

PROJ=/home/others/23EC10068/kvcache/ACSA-Project
export PYTORCH_NVML_BASED_CUDA_CHECK=0

MODE="${1:?usage: sbatch --array=0-N jobs_humaneval.sh <units.json> | --check | --rescore <samples.jsonl>}"
CHECK_ONLY=0
RESCORE=""
case "$MODE" in
  --check)   CHECK_ONLY=1; UNIT="-" ;;
  --rescore) RESCORE="$(readlink -f "${2:?--rescore needs a samples .jsonl}")"; UNIT="-" ;;
  *)         UNIT="${SLURM_ARRAY_TASK_ID:?submit as an array job (--array=...), one task per unit}" ;;
esac
RESULTS_ROOT="${HUMANEVAL_RESULTS:-$PROJ/results/humaneval}"

# --- environment ---
source "$PROJ/.venv/bin/activate"
export HF_HOME="$PROJ/hf_cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Deterministic cuBLAS workspaces; must be set before CUDA initializes.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MIKV_COMMIT="$(git -C "$PROJ" -c safe.directory='*' rev-parse --short HEAD 2>/dev/null || echo 'n/a')"

SCRATCH="${TMPDIR:-/tmp}/${USER}_humaneval_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

mkdir -p "$PROJ/logs"

# --- provenance ---
echo "host      : $(hostname)"
echo "jobid     : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-?}}_${UNIT}"
echo "gpu       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "mode      : $MODE ${RESCORE}"
echo "results   : $RESULTS_ROOT"
echo "scratch   : $SCRATCH"
echo "started   : $(date -Is)"
echo "commit    : $MIKV_COMMIT"

# --- rescore: CPU only, executes code, no model ---
if [[ -n "$RESCORE" ]]; then
  cd "$SCRATCH"
  cp "$RESCORE" samples.jsonl
  evaluate_functional_correctness samples.jsonl
  cp samples.jsonl_results.jsonl "${RESCORE%.jsonl}.official_results.jsonl"
  echo "per-problem results -> ${RESCORE%.jsonl}.official_results.jsonl"
  echo "finished  : $(date -Is)"
  exit 0
fi

nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" \
  || { rc=$?; echo "WARNING   : nvidia-smi failed on $(hostname) (exit $rc); testing CUDA directly"; }

# --- CUDA gate: fail in seconds on a broken node, never fall back to the CPU ---
if ! python - <<'EOF'
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    p = torch.cuda.get_device_properties(0)
    x = torch.ones(1024, 1024, device="cuda", dtype=torch.float16)
    (x @ x).sum().item()
    print(f"cuda      : OK | torch {torch.__version__} cuda {torch.version.cuda} | "
          f"{p.name} {p.total_memory / 1e9:.1f}GB SM{p.major}.{p.minor}")
except Exception as exc:
    print(f"cuda      : FAILED | {type(exc).__name__}: {exc}")
    sys.exit(1)
EOF
then
  msg="CUDA is unusable on $(hostname) -- keep jobs off it:
    EXCLUDE=$(hostname) ./humaneval_sweeps.sh submit <group>
(nothing was written for this unit; resubmitting runs it elsewhere)"
  echo "ERROR     : $msg"
  echo "ERROR: $msg" >&2
  exit 3
fi
if (( CHECK_ONLY )); then
  echo "node $(hostname): CUDA OK"
  exit 0
fi

# --- stage the model onto node-local disk ---
# Loading straight from hf_cache on NFS pages the 13 GB of safetensors in by
# random faults, and when fileserver2 is slow that stalled probe 27665 for 30+
# minutes in model.to(cuda). One sequential copy per node (~2 min at a healthy
# 115 MB/s) avoids it; later tasks on the same node reuse the copy. flock makes
# the second task on a node wait for the first task's copy instead of racing it.
MODEL_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_name"])' "$MODE")"
MODEL_DIR="models--${MODEL_ID//\//--}"
LOCAL_HF="/tmp/${USER}_hf_cache"
mkdir -p "$LOCAL_HF/hub"
echo "staging   : $MODEL_ID -> $LOCAL_HF ($(date -Is))"
flock -w 3600 "$LOCAL_HF/.lock" bash -c '
  set -euo pipefail
  src="$1/hub/$3"; dst="$2/hub/$3"
  if [[ -f "$dst/.staged" ]]; then echo "staging   : already on this node"; exit 0; fi
  rm -rf "$dst.partial"
  cp -a "$src" "$dst.partial"
  rm -rf "$dst"
  mv "$dst.partial" "$dst"
  touch "$dst/.staged"
' _ "$PROJ/hf_cache" "$LOCAL_HF" "$MODEL_DIR"
export HF_HOME="$LOCAL_HF"
echo "staged    : $(du -shL "$LOCAL_HF/hub/$MODEL_DIR" | cut -f1) at $(date -Is)"

# --- run ---
cd "$PROJ"
echo "==== Running HumanEval unit $UNIT ===="
python scripts/mikv_humaneval.py --results-root "$RESULTS_ROOT" run --units-file "$MODE" --unit "$UNIT"

echo "finished  : $(date -Is)"
