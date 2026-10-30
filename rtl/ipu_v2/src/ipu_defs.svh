// ipu_defs.svh -- shared constants and functions for the IPU (spec v0.2).
// Included inside module bodies (no SV packages: Yosys 0.33 compatibility).
/* verilator lint_off UNUSEDPARAM */

localparam logic [1:0] TIER_EMPTY = 2'd0;
localparam logic [1:0] TIER_HIGH  = 2'd1;
localparam logic [1:0] TIER_RSVD  = 2'd2;
localparam logic [1:0] TIER_LOW   = 2'd3;

localparam logic [1:0] PH_PREFILL = 2'd1;
localparam logic [1:0] PH_DECODE  = 2'd2;

localparam int S_W   = 30;          // score: unsigned Q14.16
localparam int D_W   = 17;          // delta: unsigned Q1.16
localparam int CMP_W = 42;          // decayed compare value: S << (12 - sh)
localparam int SH_W  = 4;           // sh = round(log2(age)), 0..12
localparam int NTH   = 12;          // number of sh thresholds

// Ages at which round(log2(age)) steps up: ceil(2^(j + 1/2)).
function automatic logic [12:0] sh_thresh(input int j);
  case (j)
    0:  sh_thresh = 13'd2;
    1:  sh_thresh = 13'd3;
    2:  sh_thresh = 13'd6;
    3:  sh_thresh = 13'd12;
    4:  sh_thresh = 13'd23;
    5:  sh_thresh = 13'd46;
    6:  sh_thresh = 13'd91;
    7:  sh_thresh = 13'd182;
    8:  sh_thresh = 13'd363;
    9:  sh_thresh = 13'd725;
    10: sh_thresh = 13'd1449;
    default: sh_thresh = 13'd2897;
  endcase
endfunction
/* verilator lint_on UNUSEDPARAM */
