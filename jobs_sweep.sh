#!/bin/bash
#SBATCH --job-name=mikv_sweep
#SBATCH --output=/home/others/23EC10068/kvcache/ACSA-Project/logs/sweep_%j.out
#SBATCH --error=/home/others/23EC10068/kvcache/ACSA-Project/logs/sweep_%j.err
#SBATCH --partition=gpupart_p100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#
# Batch launcher for a MiKV sweep (see jobs.sh for a single unswept run).
# Wraps scripts/run_sweep.sh rather than mikv.py directly, so the sweep gets its
# own results directory and checkpointed CSV the same way an interactive launch
# does.
#
# usage:
#   sbatch jobs_sweep.sh <preset> [run_sweep.sh / mikv.py flags...]
#
# examples:
#   sbatch jobs_sweep.sh greedy
#   sbatch jobs_sweep.sh confirm --low-bits 2,4 --window-tokens 32,64
#   sbatch --time=02:00:00 jobs_sweep.sh ofat --age-lut-entries 128,256,512
#
# --time on the sbatch command line overrides the #SBATCH default above; size it
# to the preset (see the cost table in docs/ninaad-doc.md -- greedy/ofat ~3.1 h,
# score ~8.4 h, budget ~9.4 h, confirm ~12.6 h). Never submit `exhaustive` (it is
# ~16.8 days); cost it with `--dry` from a login node instead.
#
# --dry-run's model-free plan check happens here too, before the GPU hours are
# spent: run_sweep.sh always prints the plan first regardless.

set -euo pipefail

PROJ=/home/others/23EC10068/kvcache/ACSA-Project
# No env var protects against the broken NVML on these nodes: PYTORCH_NVML_BASED_CUDA_CHECK
# is read as `== "1"` (to opt *in*) and only gates Python-level device counting, while the
# nvmlInit_v2_ that asserts lives in the C++ allocator's out-of-memory reporter. The real
# defence is not running out of memory; see quantize_fixed_point in scripts/mikv_quant.py.

# --- sweep type and any flags to forward ---
SWEEP="${1:?usage: sbatch jobs_sweep.sh <preset> [run_sweep.sh flags...]}"
shift

# --- environment ---
source "$PROJ/.venv/bin/activate"

export HF_HOME="$PROJ/hf_cache"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$PROJ/logs"

# One row per job in a machine-readable index, so a bare job id can be mapped back
# to what it ran without opening (or keeping) its log. Written at start, not at
# finish, so a job that dies early is still recorded -- which matters here, since
# the failure modes seen so far kill jobs in their first seconds.
IDX="$PROJ/logs/index.tsv"
[ -f "$IDX" ] || printf '#jobid\tstarted\tnode\tkind\tpreset\tflags\n' > "$IDX"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${SLURM_JOB_ID:-interactive}" "$(date -Is)" "$(hostname)" sweep "$SWEEP" "$*" >> "$IDX"

# --- provenance ---
echo "host      : $(hostname)"
echo "jobid     : ${SLURM_JOB_ID:-interactive}"
echo "gpu       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "preset    : $SWEEP"
echo "flags     : $*"
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

echo "==== Running MiKV sweep: $SWEEP ===="
# run_sweep.sh's interactive confirm is guarded by `[ -t 0 ]`, so it proceeds
# straight through under sbatch (no tty) without blocking on a prompt.
./scripts/run_sweep.sh "$SWEEP" "$@"

echo "finished  : $(date -Is)"
