#!/bin/bash
#SBATCH --job-name=mikv_gsm8k
#SBATCH --output=/home/others/23EC10068/kvcache/ACSA-Project/logs/gsm8k_%A_%a.out
#SBATCH --error=/home/others/23EC10068/kvcache/ACSA-Project/logs/gsm8k_%A_%a.err
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#
# One array task of a GSM8K sweep: runs unit $SLURM_ARRAY_TASK_ID of a frozen
# units file (see scripts/mikv_gsm8k.py). Do not sbatch this by hand -- submit
# through gsm8k_sweeps.sh, which plans the units, writes the units file, sizes
# --time from the measured s/problem and passes the right --array range:
#
#   ./gsm8k_sweeps.sh submit tier1
#
# By hand, for one unit (e.g. to rerun unit 7 of an existing plan):
#   sbatch --array=7 --time=04:00:00 jobs_gsm8k.sh results/gsm8k/plans/tier1_<stamp>.json
#
# Every array task takes one GPU on one node, so the array spreads over every
# free P100 in the partition (gnode2-5) on its own. Do NOT add
# --nodelist=gnode[2-5] here: for a single-node job SLURM reads a node list as
# "all of these nodes", and each task would wait for all four at once. If the
# partition holds nodes you must avoid, use --exclude instead.
#
# Resume: the task skips every problem its run already has on disk (from any
# earlier task), so a task killed at the walltime limit loses at most the problem
# it was on; `./gsm8k_sweeps.sh submit <group>` again submits what is left.
#
# `jobs_gsm8k.sh --check` runs only the CUDA gate below and exits -- what
# `./gsm8k_sweeps.sh nodecheck` submits once per node to find the healthy ones.

set -euo pipefail

PROJ=/home/others/23EC10068/kvcache/ACSA-Project
export PYTORCH_NVML_BASED_CUDA_CHECK=0

UNITS="${1:?usage: sbatch --array=0-N jobs_gsm8k.sh <units.json>   |   sbatch jobs_gsm8k.sh --check}"
CHECK_ONLY=0
if [[ "$UNITS" == "--check" ]]; then
  CHECK_ONLY=1
  UNIT="-"
else
  UNIT="${SLURM_ARRAY_TASK_ID:?submit as an array job (--array=...), one task per unit}"
fi
RESULTS_ROOT="${GSM8K_RESULTS:-$PROJ/results/gsm8k}"

# --- environment ---
source "$PROJ/.venv/bin/activate"

export HF_HOME="$PROJ/hf_cache"
export HF_HUB_DISABLE_XET=1
# The weights and tokenizer are already in hf_cache; offline mode stops every
# task from making hub requests (and from hanging if a node has no internet).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$PROJ/logs"

# --- provenance ---
echo "host      : $(hostname)"
echo "jobid     : ${SLURM_ARRAY_JOB_ID:-?}_${UNIT}"
echo "gpu       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "units     : $UNITS"
echo "unit      : $UNIT"
echo "results   : $RESULTS_ROOT"
echo "started   : $(date -Is)"
echo "commit    : $(git -C "$PROJ" rev-parse --short HEAD 2>/dev/null || echo 'n/a')"

# nvidia-smi is informational only. It goes through NVML, which can be broken on
# a node whose CUDA still works, and under `set -e` its failure used to kill the
# task before anything useful ran. Whether the GPU is usable is decided by the
# CUDA gate below, which actually runs a kernel.
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
    (x @ x).sum().item()  # a real kernel launch, not just a device query
    print(f"cuda      : OK | torch {torch.__version__} cuda {torch.version.cuda} | "
          f"{p.name} {p.total_memory / 1e9:.1f}GB SM{p.major}.{p.minor}")
except Exception as exc:
    print(f"cuda      : FAILED | {type(exc).__name__}: {exc}")
    sys.exit(1)
EOF
then
  msg="CUDA is unusable on $(hostname). 'Driver/library version mismatch' means this node's
NVIDIA kernel module and user-space driver differ (usually a driver update without a
reboot) -- only an admin can fix it; report the node. Meanwhile keep jobs off it:
    EXCLUDE=$(hostname) ./gsm8k_sweeps.sh submit <group>
(nothing was written for this unit; resubmitting runs it elsewhere)"
  echo "ERROR     : $msg"
  echo "ERROR: $msg" >&2
  exit 3
fi
if (( CHECK_ONLY )); then
  echo "node $(hostname): CUDA OK"
  exit 0
fi

# --- run ---
cd "$PROJ"
echo "==== Running GSM8K unit $UNIT ===="
python scripts/mikv_gsm8k.py --results-root "$RESULTS_ROOT" run --units-file "$UNITS" --unit "$UNIT"

echo "finished  : $(date -Is)"
