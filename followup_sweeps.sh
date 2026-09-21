#!/bin/bash
# Follow-up sweeps after the confirm run (20260910_164327).
# Each sweep pins every axis explicitly and varies only what it is testing, so
# no preset defaults leak in. Uses the `legacy` preset + --sweep-mode grid, the
# README's "one-off configuration, no preset semantics" pattern.
#
# usage (from the repo root, ACSA-Project/):
#   ./followup_sweeps.sh dry  S1         # print the plan for one sweep, no GPU (login node)
#   ./followup_sweeps.sh dry  all        # print every plan; check run counts match the table
#   ./followup_sweeps.sh submit S1       # sbatch one sweep
#   ./followup_sweeps.sh submit tier1    # sbatch S1-S5 (independent; can run concurrently)
#   ./followup_sweeps.sh submit S6probe  # one long-context run to check P100 memory
#   ./followup_sweeps.sh submit S6       # only after S6probe finished without OOM
#
# Results go to results/followup/<sweep>/legacy/sweep.csv via RESULTS_DIR,
# so each sweep keeps its own CSV.
#
# Timing assumes the confirm run's observed 5.8 min/run on gpupart_p100
# (40 prompts, t_p = 1106). --time below is runs x 5.8 min x ~1.5 margin.

set -euo pipefail
PROJ=/home/others/23EC10068/kvcache/ACSA-Project
MODE="${1:?dry|submit}"; WHICH="${2:?S1..S7|tier1|all}"

# Axes that stay fixed in every sweep below.
COMMON="--sweep-mode grid --budget-modes fixed_length --score-signed 0 --score-decay-applications ranking"
# The confirm run's benchmark settings (keep identical so rows stay comparable).
BENCH="--num-samples 40 --num-records 40 --seed 0"
# The recommended score path from the confirm analysis.
SCORE_Q1416="--score-schemes fixed --score-length-bits 30 --score-frac-bits 16"

declare -A TIME FLAGS

# ---------------------------------------------------------------- tier 1 --
# S1  Operating point: int3 vs int4, and how small B_hi can go.
#     r = 0.05 gives k = 55 < w = 64, so the policy degenerates to pure recency:
#     that run is the "does the IPU matter at all" control.
#     8 runs, ~0.8 h
TIME[S1]=01:30:00
FLAGS[S1]="$COMMON $BENCH $SCORE_Q1416 --balancers pow2 --high-tiers fp16 \
  --low-bits 3,4 --window-tokens 64 --score-decay-schemes lut --age-lut-entries 512 \
  --budget-ratios 0.05,0.1,0.15,0.25"

# S2  HIGH tier precision: fp16 vs int8 vs int4 (int4/int4 = uniform int4, a
#     second no-policy control). Decides HIGH-tier storage and whether the
#     append path needs a quantizer.
#     3 HIGH x 3 LOW x 2 r = 18 runs, ~1.7 h
TIME[S2]=03:00:00
FLAGS[S2]="$COMMON $BENCH $SCORE_Q1416 --balancers pow2 --high-tiers fp16,int8,int4 \
  --low-bits 2,3,4 --window-tokens 64 --score-decay-schemes lut --age-lut-entries 512 \
  --budget-ratios 0.25,0.5"

# S3  Comparator path (ipu_age_lut): exact divide vs 512/256/128-entry ROM vs
#     pow2 (shift only: no ROM, no 16 x (30x16) multipliers).
#     Run at the stress point (int2/int3), where selection quality shows up.
#     5 decay configs x 2 LOW x 2 r = 20 runs, ~1.9 h
TIME[S3]=03:00:00
FLAGS[S3]="$COMMON $BENCH $SCORE_Q1416 --balancers pow2 --high-tiers fp16 \
  --low-bits 2,3 --window-tokens 64 --score-decay-schemes exact,pow2,lut \
  --age-lut-entries 128,256,512 --budget-ratios 0.25,0.5"

# S4  Recency-window CSR range (ipu_mask_gen / cfg_w). W < 64 isn't legal with
#     the ROM as built (n_min = 65), so this tests 64 and above.
#     3 w x 2 LOW x 2 r = 12 runs, ~1.2 h
TIME[S4]=02:00:00
FLAGS[S4]="$COMMON $BENCH $SCORE_Q1416 --balancers pow2 --high-tiers fp16 \
  --low-bits 2,3 --window-tokens 64,128,256 --score-decay-schemes lut --age-lut-entries 512 \
  --budget-ratios 0.25,0.5"

# S5  Channel balancer: paper (sqrt, real multiply) vs pow2 (exponent add).
#     2 x 2 LOW x 2 r = 8 runs, ~0.8 h
TIME[S5]=01:30:00
FLAGS[S5]="$COMMON $BENCH $SCORE_Q1416 --balancers paper,pow2 --high-tiers fp16 \
  --low-bits 2,3 --window-tokens 64 --score-decay-schemes lut --age-lut-entries 512 \
  --budget-ratios 0.25,0.5"

# ---------------------------------------------------------------- tier 2 --
# S6  Longer context (~1.9k tokens): score-format margins, the operating point
#     and int3. 70 records is about the P100 16 GB ceiling for eager attention
#     on a 7B model. Submit S6probe first; if it OOMs, drop to 55 records.
TIME[S6probe]=01:00:00
FLAGS[S6probe]="$COMMON --num-samples 40 --num-records 70 --seed 0 $SCORE_Q1416 \
  --balancers pow2 --high-tiers fp16 --low-bits 4 --window-tokens 64 \
  --score-decay-schemes lut --age-lut-entries 512 --budget-ratios 0.25"
#     4 score formats x LOW {2,3,4} x r {0.1,0.25} = 24 runs at ~2-3x the time
TIME[S6]=12:00:00
FLAGS[S6]="$COMMON --num-samples 40 --num-records 70 --seed 0 \
  --score-schemes fp32,fixed --score-length-bits 30 --score-frac-bits 12,16,20 \
  --balancers pow2 --high-tiers fp16 --low-bits 2,3,4 --window-tokens 64 \
  --score-decay-schemes lut --age-lut-entries 512 --budget-ratios 0.1,0.25"

# S7  Tight-CI sign-off on the finalists, on fresh prompts (seed 1).
#     Edit --low-bits / --budget-ratios to the finalists S1/S2 point to.
#     2 LOW x 2 r = 4 runs at 120 samples (~3x time each), ~1.2 h
TIME[S7]=03:00:00
FLAGS[S7]="$COMMON --num-samples 120 --num-records 40 --seed 1 $SCORE_Q1416 \
  --balancers pow2 --high-tiers fp16 --low-bits 3,4 --window-tokens 64 \
  --score-decay-schemes lut --age-lut-entries 512 --budget-ratios 0.1,0.25"

run() {
  local s="$1"
  if [[ -z "${FLAGS[$s]:-}" ]]; then echo "unknown sweep: $s" >&2; exit 1; fi
  echo "=== $s ==="
  if [[ "$MODE" == dry ]]; then
    ( cd "$PROJ" && RESULTS_DIR="$PROJ/results/followup/$s" ./scripts/run_sweep.sh legacy ${FLAGS[$s]} --dry )
  else
    ( cd "$PROJ" && sbatch --job-name="mikv_$s" --time="${TIME[$s]}" \
        --export=ALL,RESULTS_DIR="$PROJ/results/followup/$s" \
        jobs_sweep.sh legacy ${FLAGS[$s]} )
  fi
}

case "$WHICH" in
  tier1) for s in S1 S2 S3 S4 S5; do run "$s"; done ;;
  all)   for s in S1 S2 S3 S4 S5 S6probe; do run "$s"; done
         echo "S6 waits for S6probe; S7 waits for your finalists from S1/S2." ;;
  *)     run "$WHICH" ;;
esac
