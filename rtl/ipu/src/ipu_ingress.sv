// ipu_ingress -- P x (fp16 attention probability -> unsigned Q1.16 delta).
//
// delta = clamp(round_half_even(p * 2^16), 0, 2^17 - 1). Exact for subnormals.
// NaN, +/-Inf and negative non-zero inputs give 0 and raise `bad`; -0 gives 0.
// Combinational; the caller registers the outputs (pipeline stage S2).
module ipu_ingress #(
  parameter int P = 16
) (
  input  logic [16*P-1:0] fp16,
  output logic [17*P-1:0] delta,
  output logic [P-1:0]    bad
);
  genvar i;
  for (i = 0; i < P; i++) begin : g_lane
    logic        s;
    logic [4:0]  e;
    logic [9:0]  m;
    logic [10:0] mant;      // significand with hidden bit
    logic [3:0]  shl;       // E - 1, E = max(e, 1), only used for E <= 15
    logic [24:0] y;         // value * 2^24
    logic [16:0] q0;
    logic        rnd;
    logic [16:0] q;

    assign s    = fp16[16*i+15];
    assign e    = fp16[16*i+14 -: 5];
    assign m    = fp16[16*i+9  -: 10];
    assign mant = {(e != 5'd0), m};
    assign shl  = (e == 5'd0) ? 4'd0 : 4'(e - 5'd1);
    assign y    = {14'd0, mant} << shl;
    assign q0   = y[24:8];
    assign rnd  = y[7] & ((|y[6:0]) | q0[0]);   // round half to even
    assign q    = q0 + {16'd0, rnd};            // cannot overflow for E <= 15

    // Inf/NaN -> 0 (bad); negative -> 0 (bad unless -0); E >= 16 (value >= 2) saturates
    assign delta[17*i +: 17] = (e == 5'd31 || s) ? 17'd0 : (e[4] ? '1 : q);
    assign bad[i]            = (e == 5'd31) || (s && ((e != 5'd0) || (m != 10'd0)));
  end
endmodule
