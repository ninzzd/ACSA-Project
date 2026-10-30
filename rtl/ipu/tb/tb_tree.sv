// tb_tree -- ipu_min_tree + ipu_min_track: champion (cmp, idx) per packet, ties -> lowest idx.
`timescale 1ns/1ps
`ifndef VEC_DIR
`define VEC_DIR "../vectors/"
`endif
module tb_tree;
  localparam int P = 16, CW = 42, RW = 8, NPKT = 3000, MAXB = 20000;
  localparam int INW = 2 + RW + CW*P;
  localparam int EXW = 1 + CW + 12;
  logic [INW-1:0] in_mem  [0:MAXB-1];
  logic [EXW-1:0] exp_mem [0:NPKT-1];

  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  logic              in_v, in_first, in_last;
  logic [RW-1:0]     in_row;
  logic [CW*P-1:0]   in_cmp;
  logic              t_v, t_first, t_last;
  logic [RW-1:0]     t_row;
  logic [CW-1:0]     t_cmp;
  logic [3:0]        t_lane;
  logic              done, found;
  logic [CW-1:0]     c_cmp;
  logic [RW+3:0]     c_idx;
  int errors = 0, got = 0, b = 0, pk = 0;

  ipu_min_tree  #(.P(P), .CW(CW), .RW(RW)) u_t (.clk(clk), .rst_n(rst_n), .in_v(in_v),
    .in_first(in_first), .in_last(in_last), .in_row(in_row), .in_cmp(in_cmp), .out_v(t_v),
    .out_first(t_first), .out_last(t_last), .out_row(t_row), .out_cmp(t_cmp), .out_lane(t_lane));
  ipu_min_track #(.P(P), .CW(CW), .RW(RW)) u_m (.clk(clk), .rst_n(rst_n), .in_v(t_v),
    .in_first(t_first), .in_last(t_last), .in_row(t_row), .in_cmp(t_cmp), .in_lane(t_lane),
    .done(done), .found(found), .champ_cmp(c_cmp), .champ_idx(c_idx));

  always @(posedge clk) begin
    if (done) begin
      logic [EXW-1:0] e;
      e = exp_mem[got];
      if (found !== e[EXW-1] || (found && (c_cmp !== e[CW+11:12] || c_idx[11:0] !== e[11:0]))) begin
        if (errors < 5) $display("MISMATCH pkt %0d got f=%0d c=%0h i=%0d exp %0h", got, found, c_cmp, c_idx, e);
        errors++;
      end
      got++;
    end
  end

  initial begin
    $readmemh({`VEC_DIR, "unit/tree_in.hex"},  in_mem);
    $readmemh({`VEC_DIR, "unit/tree_exp.hex"}, exp_mem);
    if ($isunknown(exp_mem[0])) $fatal(1, "vector file missing");
    in_v = 0; in_first = 0; in_last = 0; in_row = 0; in_cmp = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    begin
      while (pk < NPKT) begin
        @(negedge clk);
        if ($urandom_range(0, 3) == 0) begin
          in_v = 0;                              // random bubbles
        end else begin
          {in_first, in_last, in_row, in_cmp} = in_mem[b];
          in_v = 1;
          if (in_last) pk++;
          b++;
        end
      end
      @(negedge clk) in_v = 0;
    end
    repeat (20) @(posedge clk);
    if (errors == 0 && got == NPKT) $display("PASS tb_tree: %0d packets", got);
    else                            $display("FAIL tb_tree: %0d mismatches, %0d/%0d packets", errors, got, NPKT);
    $finish;
  end
endmodule
