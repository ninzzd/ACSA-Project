#!/bin/bash
# Open-source regression: Verilator (default) or Icarus.
#   sim/run_oss.sh [vl|iv] [top <scenario>]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"; TB="$ROOT/tb"; VEC="$ROOT/vectors"
SIM=${1:-vl}; WORK="$ROOT/sim/oss_work"; mkdir -p "$WORK"
RTL=$(ls "$SRC"/ipu_*.sv)

run() {   # name top "files" "defines/includes"
  local name=$1 top=$2 files=$3 extra=$4
  if [ "$SIM" = vl ]; then
    verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -I"$SRC" $extra --top-module "$top" \
      -Mdir "$WORK/vl_$name" $files > "$WORK/vl_$name.log" 2>&1 || { tail "$WORK/vl_$name.log"; exit 1; }
    "$WORK/vl_$name/V$top" | grep -E "PASS|FAIL|ERROR|perf" | head -20
  else
    iverilog -g2012 -I"$SRC" $extra -o "$WORK/iv_$name" $files 2>&1 | grep -v "sorry: constant selects" | head
    vvp -n "$WORK/iv_$name" | grep -E "PASS|FAIL|ERROR|perf" | head -20
  fi
}

if [ "$2" != "top" ]; then
  run ingress  tb_ingress  "$SRC/ipu_ingress.sv $TB/tb_ingress.sv" "-DVEC_DIR=\"$VEC/\""
  run datapath tb_datapath "$SRC/ipu_lane_ctl.sv $SRC/ipu_ingress.sv $SRC/ipu_accum.sv $SRC/ipu_cmp_shift.sv $TB/tb_datapath.sv" "-DVEC_DIR=\"$VEC/\""
  run tree     tb_tree     "$SRC/ipu_min_tree.sv $SRC/ipu_min_track.sv $TB/tb_tree.sv" "-DVEC_DIR=\"$VEC/\""
fi
for s in ${3:-edge raw main big}; do
  run "top_$s" tb_ipu_top "$RTL $TB/tb_ipu_top.sv" "-I$VEC/top_$s -DSCN_DIR=\"$VEC/top_$s/\""
done
