// ipu_cmp_shift -- P x decayed compare value (combinational; stage S4 input).
//
//   pow2 : cmp = S << (12 - sh) = {S, 12'b0} >> sh   (exact, 42 bits)
//   RAW  : cmp = S
//   ineligible lanes: cmp = all-ones. An eligible cmp is at most 2^42 - 2^12, so the
//   all-ones code never collides with a real value and no valid bit is needed.
module ipu_cmp_shift #(
  parameter int P = 16
) (
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [32*P-1:0] word,        // tier bits unused here
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic [4*P-1:0]  sh,
  input  logic [P-1:0]    elig,
  input  logic            raw,
  output logic [42*P-1:0] cmp
);
  `include "ipu_defs.svh"
  genvar i;
  for (i = 0; i < P; i++) begin : g_lane
    logic [29:0] s;
    logic [41:0] shifted;
    assign s       = word[32*i+29 -: 30];
    assign shifted = {s, 12'd0} >> sh[4*i +: 4];
    assign cmp[42*i +: 42] = !elig[i] ? '1 : (raw ? {12'd0, s} : shifted);
  end
endmodule
