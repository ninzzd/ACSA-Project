#!/bin/bash
# Vivado xsim regression (run from anywhere; needs xvlog/xelab/xsim on PATH, Vivado 2020+).
#   sim/run_xsim.sh            all unit tests + all top scenarios
#   sim/run_xsim.sh top main   one top scenario
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"; TB="$ROOT/tb"; VEC="$ROOT/vectors"
WORK="$ROOT/sim/xsim_work"; mkdir -p "$WORK"; cd "$WORK"
RTL=$(ls "$SRC"/ipu_*.sv)

run() {   # name top "files" "extra xvlog args"
  local name=$1 top=$2 files=$3 extra=$4
  xvlog -sv -i "$SRC" $extra $files > "xvlog_$name.log"
  xelab -debug off -s "snap_$name" "$top" > "xelab_$name.log"
  xsim "snap_$name" -R | tee "xsim_$name.log" | grep -E "PASS|FAIL|ERROR" | head -20
}

if [ "$1" != "top" ]; then
  run ingress  tb_ingress  "$SRC/ipu_ingress.sv $TB/tb_ingress.sv" "-d VEC_DIR=\"$VEC/\""
  run datapath tb_datapath "$SRC/ipu_lane_ctl.sv $SRC/ipu_ingress.sv $SRC/ipu_accum.sv $SRC/ipu_cmp_shift.sv $TB/tb_datapath.sv" "-d VEC_DIR=\"$VEC/\""
  run tree     tb_tree     "$SRC/ipu_min_tree.sv $SRC/ipu_min_track.sv $TB/tb_tree.sv" "-d VEC_DIR=\"$VEC/\""
fi
for s in ${2:-edge raw main big}; do
  run "top_$s" tb_ipu_top "$RTL $TB/tb_ipu_top.sv" "-i $VEC/top_$s -d SCN_DIR=\"$VEC/top_$s/\""
done
