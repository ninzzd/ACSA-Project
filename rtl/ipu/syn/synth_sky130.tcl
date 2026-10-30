yosys -import
yosys read_verilog -sv -Isrc src/ipu_ingress.sv src/ipu_lane_ctl.sv src/ipu_accum.sv src/ipu_cmp_shift.sv src/ipu_min_tree.sv src/ipu_min_track.sv src/ipu_fifo.sv src/ipu_pipe.sv src/ipu_rd_eng.sv src/ipu_wr_eng.sv src/ipu_issue.sv src/ipu_csr.sv src/ipu_seq_fsm.sv src/ipu_top.sv
yosys read_verilog -sv -lib src/ipu_sram_1r1w.sv
yosys hierarchy -check -top ipu_top
yosys synth -top ipu_top
yosys dfflibmap -liberty $::env(LIB)
yosys abc -liberty $::env(LIB) -D $::env(PERIOD_PS)
yosys opt_clean -purge
yosys tee -o $::env(OUT)/sky130_stat.txt stat -liberty $::env(LIB)
