// ipu_accum -- P x score update (combinational; registered by the caller, stage S3).
//
//   new token  : tier <- HIGH, S <- delta
//   tier HIGH  : S <- min(S + delta, 2^30 - 1)       (sat flag on clamp)
//   otherwise  : S unchanged (LOW scores frozen: the adder sees +0)
//   lane invalid (idx >= t): word written as 0 (EMPTY)
//   elig = decode && valid && !new && tier == HIGH && age >= W_EFF
module ipu_accum #(
  parameter int P = 16
) (
  input  logic [32*P-1:0] word_in,
  input  logic [17*P-1:0] delta,
  input  logic [P-1:0]    valid,
  input  logic [P-1:0]    isnew,
  input  logic [P-1:0]    agew,
  input  logic            decode,
  output logic [32*P-1:0] word_out,
  output logic [P-1:0]    elig,
  output logic [P-1:0]    sat
);
  `include "ipu_defs.svh"
  genvar i;
  for (i = 0; i < P; i++) begin : g_lane
    logic [1:0]  tier;
    logic [29:0] s_in, base;
    logic [16:0] d_eff;
    logic [30:0] sum;
    logic        is_high, add_en;
    logic [29:0] s_new;
    logic [1:0]  tier_new;

    assign tier     = word_in[32*i+31 -: 2];
    assign s_in     = word_in[32*i+29 -: 30];
    assign is_high  = (tier == TIER_HIGH);
    assign add_en   = isnew[i] | is_high;
    assign base     = isnew[i] ? 30'd0 : s_in;
    assign d_eff    = add_en ? delta[17*i +: 17] : 17'd0;   // operand isolation for LOW
    assign sum      = {1'b0, base} + {14'd0, d_eff};
    assign s_new    = sum[30] ? '1 : sum[29:0];
    assign tier_new = isnew[i] ? TIER_HIGH : tier;

    assign word_out[32*i +: 32] = valid[i] ? {tier_new, s_new} : 32'd0;
    assign sat[i]  = valid[i] & sum[30];
    assign elig[i] = decode & valid[i] & ~isnew[i] & is_high & agew[i];
  end
endmodule
