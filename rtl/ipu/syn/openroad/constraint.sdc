# IPU timing constraints (Sky130 HD). Start at 5 ns (200 MHz); tighten after the first run.
# Yosys/ABC pre-layout estimate at the typical corner: ~2.75 ns logic depth.
set clk_period 5.0
create_clock -name clk -period $clk_period [get_ports clk]
set_clock_uncertainty 0.25 [get_clocks clk]
set non_clk_inputs [lsearch -inline -all -not -exact [all_inputs] [get_ports clk]]
set_input_delay  [expr 0.2 * $clk_period] -clock clk $non_clk_inputs
set_output_delay [expr 0.2 * $clk_period] -clock clk [all_outputs]
set_false_path -from [get_ports rst_n]
