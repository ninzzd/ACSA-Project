#!/bin/bash
# Three follow-up runs, ranked by expected information per GPU-hour.
#
#   E1   — eviction baseline. REQUIRES eviction_baseline.patch (see below).
#   S6   — longer context (~1.9k tokens), same grid as final_sweeps.sh's F2.
#          No code change needed; this is that job under its planned name.
#   L200 — Line Retrieval re-run at n=200, nothing else changed vs. F1.
#          Lowest value of the three: tightens a CI that's already ~0-width.
#
# GSM8k / HumanEval are NOT in this script — deliberately deferred.
#
# usage (from the repo root, ACSA-Project/):
#   git apply eviction_baseline.patch      # one-time, before E1 can run at all
#   ./priority_tests.sh dry all            # print every plan, no GPU (login node)
#   ./priority_tests.sh submit E1          # one job
#   ./priority_tests.sh submit all         # E1 + S6 + L200
#
# Results: results/priority/<job>/legacy/sweep.csv, one CSV per job.
#
# Before submitting: `git apply --check eviction_baseline.patch` on the repo,
# then `./priority_tests.sh dry E1` and confirm the plan shows 4 configurations
# (2 high_tiers x 2 ratios) before it touches a GPU.

set -euo pipefail
PROJ=/home/others/23EC10068/kvcache/ACSA-Project
MODE="${1:?dry|submit}"; WHICH="${2:?E1|S6|L200|all}"

# Pinned in every job: the settled score path (same as final_sweeps.sh's CORE).
CORE="--sweep-mode grid --budget-modes fixed_length --score-decay-applications ranking \
  --score-schemes fixed --score-length-bits 30 --score-frac-bits 16 --score-signed 0"

declare -A TIME FLAGS

# --------------------------------------------------------------------------
# E1 — eviction baseline. This is the comparison the paper's headline numbers
#      actually measure (MiKV vs. H2O eviction), which nothing run so far has
#      tested: every prior sweep compared importance-based *tiering* against
#      *uniform quantization*, never against dropping the token outright.
#
#      REQUIRES eviction_baseline.patch applied first (adds `--evict` to
#      mikv.py — see the patch for what it changes: demoted positions get
#      masked out of attention entirely instead of being quantized to
#      --low-bits). --low-bits is passed only to satisfy the existing
#      high_tier-vs-low_bits validity check; it has no effect on the result
#      under --evict, which is why it isn't crossed as an axis here.
#
#      2 HIGH formats x 2 budgets = 4 runs. Same axes as final_sweeps.sh's F1
#      so each row here has a direct, paired counterpart there:
#        int8 HIGH, r 0.25 -> D's cache retained, but evicted instead of int-N
#        int8 HIGH, r 0.05 -> recency-only + eviction (the harshest control)
#        int4 HIGH, r 0.25 -> uniform-int4-style budget, but evicted not kept
#      Expect this to look nothing like F1: eviction should craft a real,
#      large gap under the important tokens' precision, where F1 found ~none.
TIME[E1]=02:00:00
FLAGS[E1]="$CORE --evict --num-samples 100 --num-records 40 --seed 1 \
  --balancers pow2 --score-decay-schemes pow2 --window-tokens 64 \
  --high-tiers int8,int4 --low-bits 4 --budget-ratios 0.05,0.25"

# --------------------------------------------------------------------------
# S6 — same grid as F1, at ~1.9k tokens instead of ~1.1k. No code change
#      needed: this is final_sweeps.sh's F2, carried over under its planned
#      name. The lever the literature says actually separates selection
#      methods on retrieval tasks is haystack size / context length, not
#      prompt count -- this is that lever, not L200's.
#
#      12 runs at ~15 min, ~3 h. 70 records is near the P100's ceiling for
#      eager attention on a 7B model; if it OOMs, drop to --num-records 55.
TIME[S6]=06:00:00
FLAGS[S6]="$CORE --num-samples 40 --num-records 70 --seed 1 \
  --balancers pow2 --score-decay-schemes pow2 --window-tokens 64 \
  --high-tiers int8,int4 --low-bits 2,3,4 --budget-ratios 0.05,0.25"

# --------------------------------------------------------------------------
# L200 — F1's exact grid, --num-samples doubled from 100 to 200. Lowest value
#        of the three: F1 already showed int3/int4 tied with pure recency to
#        within noise, and more samples narrows the interval around that
#        estimate -- it cannot manufacture a gap that isn't there. Included
#        because it's cheap to state precisely and was asked for directly.
#
#      12 runs. Cost scales ~linearly with --num-samples (per final_sweeps.sh's
#      own timing note), so budget roughly 2x F1's 5 h.
TIME[L200]=11:00:00
FLAGS[L200]="$CORE --num-samples 200 --num-records 40 --seed 1 \
  --balancers pow2 --score-decay-schemes pow2 --window-tokens 64 \
  --high-tiers int8,int4 --low-bits 2,3,4 --budget-ratios 0.05,0.25"

run() {
  local s="$1"
  if [[ -z "${FLAGS[$s]:-}" ]]; then echo "unknown job: $s" >&2; exit 1; fi
  echo "=== $s ==="
  if [[ "$MODE" == dry ]]; then
    ( cd "$PROJ" && source .venv/bin/activate && \
      RESULTS_DIR="$PROJ/results/priority/$s" ./scripts/run_sweep.sh legacy ${FLAGS[$s]} --dry )
  else
    ( cd "$PROJ" && sbatch --job-name="mikv_$s" --time="${TIME[$s]}" \
        --export=ALL,RESULTS_DIR="$PROJ/results/priority/$s" \
        jobs_sweep.sh legacy ${FLAGS[$s]} )
  fi
}

case "$WHICH" in
  all) for s in E1 S6 L200; do run "$s"; done ;;
  *)   run "$WHICH" ;;
esac

# --------------------------------------------------------------------------
# Expected plan sizes (check these against `dry` before submitting):
#   E1:   4 configs ->   4 runs
#   S6:  12 configs ->  12 runs
#   L200: 12 configs -> 12 runs
#
# Notes:
#   - E1 will fail with an "unrecognized arguments: --evict" error until
#     eviction_baseline.patch is applied (`git apply eviction_baseline.patch`
#     from the repo root). `git apply --check` first to confirm it applies
#     cleanly against your current tree.
#   - The patch touches mikv_policy.py, mikv_bench.py, mikv_sweep.py and
#     mikv.py. It does NOT touch mikv_report.py (not part of the uploaded
#     files, so not inspected here) -- if that file hardcodes the CSV's
#     column list rather than writing whatever keys a row dict has, add
#     `evict` to it by hand or the new column will be silently dropped.
#   - E1's eviction is implemented as an attention mask (evicted positions
#     get -inf added to their attention score every subsequent step), not as
#     literally shrinking the KV tensors -- same accuracy effect as H2O's
#     real eviction, cheaper to implement, no change to memory *measured
#     during the run*. The CSV's kv_size_after column is corrected for this
#     separately (low_bits treated as 0 for the footprint calculation only),
#     so E1's memory numbers are still the honest "evicted tokens cost
#     nothing" accounting the comparison needs.
#   - GSM8k / HumanEval are intentionally not here; bring those in as a
#     separate script once E1/S6/L200 are back.
