# OpenROAD-flow-scripts (ORFS) config for the IPU on Sky130 HD.
# Copy this directory to ORFS/flow/designs/sky130hd/ipu/ and run:
#   make DESIGN_CONFIG=./designs/sky130hd/ipu/config.mk
#
# NOTE: the score bank (ipu_sram_1r1w, 16 KB at T_MAX = 4096) must become an SRAM macro
# before a full-size run (macro choice: to be discussed). Until then this config builds a
# logic-only trial with T_MAX = 256 (bank = 16 rows x 512 b in flops) to get real placed
# timing and area for everything except the SRAM.
export PLATFORM        = sky130hd
export DESIGN_NAME     = ipu_top
export DESIGN_NICKNAME = ipu
export IPU_RTL         = $(DESIGN_HOME)/src/ipu/rtl
export VERILOG_FILES   = $(sort $(wildcard $(IPU_RTL)/ipu_*.sv))
export VERILOG_INCLUDE_DIRS = $(IPU_RTL)
export VERILOG_TOP_PARAMS   = T_MAX 256
export SDC_FILE        = $(DESIGN_HOME)/$(PLATFORM)/ipu/constraint.sdc
export CORE_UTILIZATION = 45
export PLACE_DENSITY    = 0.60
