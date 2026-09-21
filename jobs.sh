#!/bin/bash
#SBATCH --job-name=mikv_repro
#SBATCH --output=/home/others/23EC10068/kvcache/ACSA-Project/logs/mikv_%j.out
#SBATCH --error=/home/others/23EC10068/kvcache/ACSA-Project/logs/mikv_%j.err
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=08:00:00

set -euo pipefail

PROJ=/home/others/23EC10068/kvcache/ACSA-Project

# --- environment ---
source "$PROJ/.venv/bin/activate"

export HF_HOME="$PROJ/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# One row per job in a machine-readable index, so a bare job id can be mapped back
# to what it ran without opening (or keeping) its log. Written at start, not at
# finish, so a job that dies early is still recorded.
mkdir -p "$PROJ/logs"
IDX="$PROJ/logs/index.tsv"
[ -f "$IDX" ] || printf '#jobid\tstarted\tnode\tkind\tpreset\tflags\n' > "$IDX"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${SLURM_JOB_ID:-interactive}" "$(date -Is)" "$(hostname)" single - "${*:--}" >> "$IDX"

# --- provenance ---
echo "host      : $(hostname)"
echo "jobid     : ${SLURM_JOB_ID:-interactive}"
echo "gpu       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "started   : $(date -Is)"
echo "commit    : $(git -C "$PROJ" rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
# Guarded because of `set -e`: nvidia-smi exits non-zero whenever the node's NVML
# userspace and kernel driver disagree (gnode2/gnode3 have), and an unguarded call
# then aborts the whole job here -- silently, since the failure is on stdout and
# Slurm records no error. CUDA itself goes through libcuda and is unaffected, so
# losing this provenance line is not a reason to lose the run.
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" || echo "nvidia-smi unavailable (NVML mismatch); continuing"

python -c "import torch; p=torch.cuda.get_device_properties(0); \
print(f'torch {torch.__version__} cuda {torch.version.cuda} | {p.name} {p.total_memory/1e9:.1f}GB SM{p.major}.{p.minor}')"

# --- run ---
cd "$PROJ"

echo "==== Running MiKV ===="
python -u scripts/mikv.py

echo "finished  : $(date -Is)"
