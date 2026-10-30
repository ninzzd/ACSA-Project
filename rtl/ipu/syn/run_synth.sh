#!/bin/bash
# Yosys synthesis runs. Needs yosys >= 0.33 and the Sky130 HD liberty
# (e.g. OpenROAD-flow-scripts/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib).
set -e
cd "$(dirname "$0")/.."
export OUT=${OUT:-syn/out}; mkdir -p "$OUT" syn/out
yosys -q -l "$OUT/generic.log" syn/synth_generic.ys
if [ -n "$LIB" ]; then
  export PERIOD_PS=${PERIOD_PS:-5000}
  yosys -l "$OUT/sky130.log" -c syn/synth_sky130.tcl > /dev/null
  grep -E "Chip area|Delay =" "$OUT/sky130.log" "$OUT/sky130_stat.txt" | tail -20
fi
