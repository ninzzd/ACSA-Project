#!/bin/bash
# Yosys synthesis runs (yosys >= 0.33).
#   generic                       : syn/run_synth.sh
#   Nangate45 + FakeRAM45 (v2)    : LIB=<NangateOpenCellLibrary_typical.lib> syn/run_synth.sh
# The Nangate45 liberty is in OpenROAD-flow-scripts/flow/platforms/nangate45/lib/.
set -e
cd "$(dirname "$0")/.."
export OUT=${OUT:-syn/out}; mkdir -p "$OUT"
yosys -q -l "$OUT/generic.log" syn/synth_generic.ys
if [ -n "$LIB" ]; then
  export PERIOD_PS=${PERIOD_PS:-1500}
  python3 syn/fakeram/gen_fakeram45_1r1w.py syn/fakeram > /dev/null
  HIER=0 yosys -l "$OUT/nangate45.log" -c syn/synth_nangate45.tcl > /dev/null
  HIER=1 yosys -l "$OUT/nangate45_hier.log" -c syn/synth_nangate45.tcl > /dev/null
  grep -E "Chip area|Delay =" "$OUT/nangate45.log" | tail -6
fi
