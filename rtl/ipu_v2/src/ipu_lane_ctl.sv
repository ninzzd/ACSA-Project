// ipu_lane_ctl -- per-lane control derived from one per-beat value.
//
// a0 = (t - 1) - P*beat is the age of lane 0; lane i has age a0 - i.
// Every "age >= X" test becomes "(a0 - X) >= i": one subtractor per beat (shared by
// all lanes) plus a log2(P)-bit compare per lane, instead of a subtractor per lane.
//   valid[i] : keep[i] and idx < t          (a0 >= i)
//   isnew[i] : idx == t - 1                  (a0 == i)
//   agew[i]  : age >= W_EFF                  (a0 - W_EFF >= i)
//   sh[i]    : round(log2(age)) = #{j : age >= sh_thresh(j)}
// Combinational; registered by the caller (stage S2).
module ipu_lane_ctl #(
  parameter int P     = 16,
  parameter int IDX_W = 12
) (
  input  logic [IDX_W-1:0]  a0,
  input  logic [6:0]        w_eff,
  input  logic [P-1:0]      keep,
  output logic [P-1:0]      valid,
  output logic [P-1:0]      isnew,
  output logic [P-1:0]      agew,
  output logic [4*P-1:0]    sh
);
  `include "ipu_defs.svh"
  localparam int LV = $clog2(P);
  localparam int DW = ((IDX_W > 12) ? IDX_W : 12) + 2;  // signed difference width (thresholds need 12 b)

  // d >= i for a small non-negative constant i (0 <= i < P)
  function automatic logic ge_lane(input logic [DW-1:0] d, input logic [LV-1:0] i);
    logic hi;
    hi = |d[DW-2:LV];
    ge_lane = !d[DW-1] && (hi || (d[LV-1:0] >= i));
  endfunction

  logic [DW-1:0] a0x, dw;
  logic [DW-1:0] dth [0:NTH-1];

  assign a0x = DW'(a0);
  assign dw  = a0x - DW'(w_eff);

  genvar j, i;
  for (j = 0; j < NTH; j++) begin : g_th
    assign dth[j] = a0x - DW'(sh_thresh(j));
  end

  for (i = 0; i < P; i++) begin : g_lane
    logic [NTH-1:0] th_hit;               // thermometer: th_hit[j] -> age >= thresh(j)
    for (j = 0; j < NTH; j++) begin : g_hit
      assign th_hit[j] = ge_lane(dth[j], LV'(i));
    end
    always_comb begin
      sh[4*i +: 4] = 4'd0;
      for (int k = 0; k < NTH; k++)
        if (th_hit[k]) sh[4*i +: 4] = 4'(k + 1);
    end
    assign valid[i] = keep[i] && ge_lane(a0x, LV'(i));
    assign isnew[i] = (a0 == IDX_W'(i));
    assign agew[i]  = ge_lane(dw, LV'(i));
  end
endmodule
