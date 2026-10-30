# Critical-path estimate (ABC stime) for the flattened Nangate45 netlist.
# ABC sees flop outputs / macro outputs / ports as arrival 0 and flop inputs as endpoints:
# add the launching element's clk->q and the capture setup by hand (see README).
yosys -import
set srcs [lsort [glob src/ipu_*.sv]]
yosys read_verilog -sv -DIPU_SRAM_MACRO -Isrc {*}$srcs
yosys read_verilog -lib syn/fakeram/fakeram45_256x512_1r1w.v
yosys hierarchy -check -top ipu_top
yosys synth -top ipu_top -flatten
yosys dfflibmap -liberty $::env(LIB)
yosys abc -liberty $::env(LIB) -D $::env(PERIOD_PS) -script syn/abc_timing.abc -showtmp
