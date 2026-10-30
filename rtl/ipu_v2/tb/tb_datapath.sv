// tb_datapath -- one-beat datapath: ipu_lane_ctl + ipu_ingress + ipu_accum + ipu_cmp_shift
// against random vectors from the golden model (gen_vectors.py unit).
`timescale 1ns/1ps
`ifndef VEC_DIR
`define VEC_DIR "../vectors/"
`endif
module tb_datapath;
  localparam int P = 16, IDX_W = 12, N = 4000;
  localparam int INW = IDX_W + 7 + P + 2 + 16*P + 32*P;     // 805
  localparam int EXW = 32*P + 3*P + 42*P;                   // 1232
  logic [INW-1:0] in_mem  [0:N-1];
  logic [EXW-1:0] exp_mem [0:N-1];

  logic [IDX_W-1:0] a0;
  logic [6:0]       w_eff;
  logic [P-1:0]     keep, valid, isnew, agew, elig, sat, bad;
  logic             decode, raw;
  logic [16*P-1:0]  fp16;
  logic [32*P-1:0]  word_in, word_out;
  logic [17*P-1:0]  delta;
  logic [4*P-1:0]   sh;
  logic [42*P-1:0]  cmp;
  int errors = 0;

  ipu_lane_ctl  #(.P(P), .IDX_W(IDX_W)) u_lc (.a0(a0), .w_eff(w_eff), .keep(keep),
                  .valid(valid), .isnew(isnew), .agew(agew), .sh(sh));
  ipu_ingress   #(.P(P)) u_in (.fp16(fp16), .delta(delta), .bad(bad));
  ipu_accum     #(.P(P)) u_ac (.word_in(word_in), .delta(delta), .valid(valid), .isnew(isnew),
                  .agew(agew), .decode(decode), .word_out(word_out), .elig(elig), .sat(sat));
  ipu_cmp_shift #(.P(P)) u_cs (.word(word_out), .sh(sh), .elig(elig), .raw(raw), .cmp(cmp));

  initial begin
    $readmemh({`VEC_DIR, "unit/dp_in.hex"},  in_mem);
    $readmemh({`VEC_DIR, "unit/dp_exp.hex"}, exp_mem);
    if ($isunknown(exp_mem[0])) $fatal(1, "vector file missing");
    for (int v = 0; v < N; v++) begin
      {a0, w_eff, keep, decode, raw, fp16, word_in} = in_mem[v];
      #1;
      if ({word_out, elig, sat, bad, cmp} !== exp_mem[v]) begin
        if (errors < 5) begin
          $display("MISMATCH vector %0d a0=%0d w=%0d dec=%0d raw=%0d", v, a0, w_eff, decode, raw);
          $display("  got elig=%04h sat=%04h bad=%04h exp elig=%04h sat=%04h bad=%04h",
                   elig, sat, bad, exp_mem[v][42*P+3*P-1 -: P], exp_mem[v][42*P+2*P-1 -: P],
                   exp_mem[v][42*P+P-1 -: P]);
        end
        errors++;
      end
    end
    if (errors == 0) $display("PASS tb_datapath: %0d beats", N);
    else             $display("FAIL tb_datapath: %0d mismatches", errors);
    $finish;
  end
endmodule
