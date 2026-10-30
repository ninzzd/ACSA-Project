// tb_ipu_top -- full-IPU testbench (spec v0.2 Section 8).
//
// Models: off-chip DRAM behind an AXI4 slave (random latency / back-pressure), host on
// AXI4-Lite (config, FLUSH, end-of-prefill LOW selection), attention stream driver with
// random gaps, migration engine with random ready and delayed acks.
// Pass = exact DRAM image after the first flush and at the end, identical victim
// sequence, expected sticky error bits. Vectors: model/gen_top.py.
//
// Compile with +incdir+<scenario dir> (for tb_cfg.svh) and -DSCN_DIR="<scenario dir>/".
`timescale 1ns/1ps
`include "tb_cfg.svh"
`ifndef SCN_DIR
`define SCN_DIR "../vectors/top_main/"
`endif
module tb_ipu_top;
  localparam int P = 16, T_MAX = 256, B_W = 3, L_W = 5, H_W = 5, NSEQ = 8, ADDR_W = 40;
  localparam int IDX_W = 8, T_W = 9, ROW_W = 4, HID_W = 13, TU_W = 2 + HID_W + T_W;
  localparam int DW = 32 * P, ROWS = T_MAX / P;
  localparam int NHEADS = `NB * `NL * `NH, NROWS = NHEADS * ROWS;
  localparam int MAXQ = 64;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  // ---------------- DUT signals
  logic [16*P-1:0] s_tdata;  logic [P-1:0] s_tkeep;  logic [TU_W-1:0] s_tuser;
  logic s_tlast, s_tvalid, s_tready;
  logic m_dem_valid, m_dem_ready, s_ack_valid;
  logic [HID_W+IDX_W-1:0] m_dem_data;
  logic [ADDR_W-1:0] awaddr, araddr;  logic [7:0] awlen, arlen;  logic [2:0] awsize, arsize;
  logic [1:0] awburst, arburst;  logic awvalid, awready, wvalid, wready, wlast, bvalid, bready;
  logic arvalid, arready, rvalid, rready, rlast;
  logic [DW-1:0] wdata, rdata;  logic [DW/8-1:0] wstrb;  logic [1:0] bresp, rresp;
  logic [7:0] l_awaddr, l_araddr;  logic l_awvalid, l_awready, l_wvalid, l_wready, l_bvalid, l_bready;
  logic l_arvalid, l_arready, l_rvalid, l_rready;  logic [31:0] l_wdata, l_rdata;
  logic [1:0] l_bresp, l_rresp;  logic irq;

  ipu_top #(.P(P), .T_MAX(T_MAX), .B_W(B_W), .L_W(L_W), .H_W(H_W), .NSEQ(NSEQ), .ADDR_W(ADDR_W)) dut (
    .clk(clk), .rst_n(rst_n),
    .s_attn_tdata(s_tdata), .s_attn_tkeep(s_tkeep), .s_attn_tuser(s_tuser), .s_attn_tlast(s_tlast),
    .s_attn_tvalid(s_tvalid), .s_attn_tready(s_tready),
    .m_dem_valid(m_dem_valid), .m_dem_ready(m_dem_ready), .m_dem_data(m_dem_data), .s_ack_valid(s_ack_valid),
    .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize), .m_axi_awburst(awburst),
    .m_axi_awvalid(awvalid), .m_axi_awready(awready), .m_axi_wdata(wdata), .m_axi_wstrb(wstrb),
    .m_axi_wlast(wlast), .m_axi_wvalid(wvalid), .m_axi_wready(wready), .m_axi_bresp(bresp),
    .m_axi_bvalid(bvalid), .m_axi_bready(bready), .m_axi_araddr(araddr), .m_axi_arlen(arlen),
    .m_axi_arsize(arsize), .m_axi_arburst(arburst), .m_axi_arvalid(arvalid), .m_axi_arready(arready),
    .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rlast(rlast), .m_axi_rvalid(rvalid),
    .m_axi_rready(rready),
    .s_axil_awaddr(l_awaddr), .s_axil_awvalid(l_awvalid), .s_axil_awready(l_awready),
    .s_axil_wdata(l_wdata), .s_axil_wvalid(l_wvalid), .s_axil_wready(l_wready),
    .s_axil_bresp(l_bresp), .s_axil_bvalid(l_bvalid), .s_axil_bready(l_bready),
    .s_axil_araddr(l_araddr), .s_axil_arvalid(l_arvalid), .s_axil_arready(l_arready),
    .s_axil_rdata(l_rdata), .s_axil_rresp(l_rresp), .s_axil_rvalid(l_rvalid), .s_axil_rready(l_rready),
    .irq(irq));

  // ---------------- vectors
  logic [DW-1:0]      mem     [0:NROWS-1];
  logic [DW-1:0]      mem_sel [0:NROWS-1];
  logic [DW-1:0]      mem_exp [0:NROWS-1];
  logic [31:0]        cmds    [0:`N_CMDS-1];
  logic [16*P+P-1:0]  beats   [0:`N_BEATS-1];
  logic [31:0]        lowsel  [0:`N_LOWSEL-1];
  logic [HID_W+IDX_W-1:0] victims [0:(`N_VICTIMS > 0 ? `N_VICTIMS : 1)-1];
  logic [39:0]        csrv    [0:63];

  int errors = 0;
  int cycle = 0;
  always @(posedge clk) cycle <= cycle + 1;

  // ---------------- AXI4 slave DRAM model
  function automatic logic addr_ok(input logic [ADDR_W-1:0] a);
    return a[39:23] == 17'(`BASE_HI) && a[5:0] == 6'd0 && int'(a[22:20]) < `NB &&
           int'(a[19:15]) < `NL && int'(a[14:10]) < `NH;
  endfunction

  function automatic int row_index(input logic [ADDR_W-1:0] a);
    int b, l, h, r;
    b = int'(a[22:20]); l = int'(a[19:15]); h = int'(a[14:10]); r = int'(a[9:6]);
    if (b >= `NB || l >= `NL || h >= `NH) return 0;
    return ((b * `NL + l) * `NH + h) * ROWS + r;
  endfunction

  // write side: AW queue, W beats only while an AW is available
  logic [ADDR_W-1:0] awq_a [0:MAXQ-1];  logic [7:0] awq_l [0:MAXQ-1];
  int awq_wp = 0, awq_rp = 0, w_beat = 0, b_pend = 0;
  assign awready = ((awq_wp - awq_rp) < MAXQ - 1) && rnd_aw;
  assign wready  = (awq_wp != awq_rp) && rnd_w;
  assign bvalid  = (b_pend > 0) && rnd_b;
  assign bresp   = 2'b00;
  logic rnd_aw, rnd_w, rnd_b, rnd_ar, rnd_r, rnd_dem;

  // read side: AR queue, R bursts after a random latency
  logic [ADDR_W-1:0] arq_a [0:MAXQ-1];  logic [7:0] arq_l [0:MAXQ-1];
  int arq_wp = 0, arq_rp = 0, r_beat = 0, r_wait = 0;
  assign arready = ((arq_wp - arq_rp) < MAXQ - 1) && rnd_ar;
  assign rvalid  = (arq_wp != arq_rp) && (r_wait == 0) && rnd_r;
  assign rdata   = mem[row_index(arq_a[arq_rp % MAXQ]) + r_beat];
  assign rlast   = (r_beat == int'(arq_l[arq_rp % MAXQ]));
  assign rresp   = 2'b00;

  always @(posedge clk) begin
    rnd_aw  <= ($urandom_range(0, 3) != 0);
    rnd_w   <= ($urandom_range(0, 4) != 0);
    rnd_b   <= ($urandom_range(0, 2) != 0);
    rnd_ar  <= ($urandom_range(0, 3) != 0);
    rnd_r   <= ($urandom_range(0, 5) != 0);
    rnd_dem <= ($urandom_range(0, 2) != 0);
    if (awvalid && awready) begin
      awq_a[awq_wp % MAXQ] <= awaddr; awq_l[awq_wp % MAXQ] <= awlen; awq_wp <= awq_wp + 1;
      if (awsize != 3'd6 || awburst != 2'b01 || !addr_ok(awaddr) || (int'(awaddr[9:6]) + int'(awlen)) >= ROWS) begin
        $display("ERROR AW addr %h len %0d", awaddr, awlen); errors++;
      end
    end
    if (wvalid && wready) begin
      int ri;
      ri = row_index(awq_a[awq_rp % MAXQ]) + w_beat;
      for (int by = 0; by < DW / 8; by++)
        if (wstrb[by]) mem[ri][8*by +: 8] <= wdata[8*by +: 8];
      if (wlast != (w_beat == int'(awq_l[awq_rp % MAXQ]))) begin
        $display("ERROR wlast mismatch at beat %0d len %0d", w_beat, awq_l[awq_rp % MAXQ]); errors++;
      end
      if (wlast) begin w_beat <= 0; awq_rp <= awq_rp + 1; end
      else       w_beat <= w_beat + 1;
    end
    if (bvalid && bready) b_pend <= b_pend - 1 + ((wvalid && wready && wlast) ? 1 : 0);
    else if (wvalid && wready && wlast) b_pend <= b_pend + 1;

    if (arvalid && arready) begin
      arq_a[arq_wp % MAXQ] <= araddr; arq_l[arq_wp % MAXQ] <= arlen; arq_wp <= arq_wp + 1;
      if (arsize != 3'd6 || arburst != 2'b01 || !addr_ok(araddr) || (int'(araddr[9:6]) + int'(arlen)) >= ROWS) begin
        $display("ERROR AR addr %h len %0d", araddr, arlen); errors++;
      end
      if (arq_wp == arq_rp) r_wait <= $urandom_range(2, 24);
    end
    if (r_wait > 0) r_wait <= r_wait - 1;
    if (rvalid && rready) begin
      if (rlast) begin
        r_beat <= 0; arq_rp <= arq_rp + 1;
        r_wait <= $urandom_range(0, 12);
      end else r_beat <= r_beat + 1;
    end
  end

  // ---------------- migration engine model
  int vptr = 0, ack_pend = 0, ack_tmr = 0;
  assign m_dem_ready = rnd_dem;
  assign s_ack_valid = (ack_pend > 0) && (ack_tmr == 0);
  always @(posedge clk) begin
    if (m_dem_valid && m_dem_ready) begin
      if (vptr >= `N_VICTIMS || m_dem_data !== victims[vptr]) begin
        if (errors < 20) $display("ERROR victim %0d: got %h exp %h", vptr, m_dem_data,
                                  (vptr < `N_VICTIMS) ? victims[vptr] : '0);
        errors++;
      end
      vptr <= vptr + 1;
    end
    if (s_ack_valid) begin
      ack_pend <= ack_pend - 1 + ((m_dem_valid && m_dem_ready) ? 1 : 0);
      ack_tmr  <= $urandom_range(0, 30);
    end else begin
      if (m_dem_valid && m_dem_ready) begin
        if (ack_pend == 0) ack_tmr <= $urandom_range(0, 30);
        ack_pend <= ack_pend + 1;
      end
      if (ack_tmr > 0) ack_tmr <= ack_tmr - 1;
    end
  end

  // ---------------- AXI4-Lite host
  task automatic axil_write(input logic [7:0] a, input logic [31:0] d);
    @(negedge clk);
    l_awaddr = a; l_wdata = d; l_awvalid = 1'b1; l_wvalid = 1'b1;
    do @(posedge clk); while (!l_awready);
    @(negedge clk);
    l_awvalid = 1'b0; l_wvalid = 1'b0; l_bready = 1'b1;
    while (!l_bvalid) @(posedge clk);
    @(negedge clk);
    l_bready = 1'b0;
  endtask

  task automatic axil_read(input logic [7:0] a, output logic [31:0] d);
    @(negedge clk);
    l_araddr = a; l_arvalid = 1'b1;
    do @(posedge clk); while (!l_arready);
    @(negedge clk);
    l_arvalid = 1'b0; l_rready = 1'b1;
    while (!l_rvalid) @(posedge clk);
    d = l_rdata;
    @(negedge clk);
    l_rready = 1'b0;
  endtask

  logic [31:0] ctrl_val = 32'd0;

  task automatic do_flush();
    logic [31:0] st;
    axil_write(8'h00, ctrl_val | 32'h100);
    st = 32'h3;
    while (st[1:0] != 2'b00) axil_read(8'h04, st);
  endtask

  task automatic compare_mem(input string what, input int sel);
    int bad;
    bad = 0;
    for (int r = 0; r < NROWS; r++) begin
      logic [DW-1:0] e;
      e = sel ? mem_sel[r] : mem_exp[r];
      if (mem[r] !== e) begin
        if (bad < 5) begin
          $display("ERROR %s: row %0d (head %0d row %0d) differs", what, r, r / ROWS, r % ROWS);
          for (int i = 0; i < P; i++)
            if (mem[r][32*i +: 32] !== e[32*i +: 32])
              $display("    lane %2d idx %3d got %08h exp %08h", i, (r % ROWS) * P + i,
                       mem[r][32*i +: 32], e[32*i +: 32]);
        end
        bad++;
      end
    end
    errors += bad;
    $display("[%0d] %s: %0d / %0d rows differ", cycle, what, bad, NROWS);
  endtask

  // ---------------- attention driver / command sequencer
  int bptr = 0, lptr = 0, n_drop = 0;
  initial begin
    logic [31:0] st, sweeps, vics, stalls, mis;
    s_tvalid = 0; s_tlast = 0; s_tdata = '0; s_tkeep = '0; s_tuser = '0;
    l_awvalid = 0; l_wvalid = 0; l_bready = 0; l_arvalid = 0; l_rready = 0; l_awaddr = 0;
    l_wdata = 0; l_araddr = 0;
    $readmemh({`SCN_DIR, "mem_init.hex"}, mem);
    $readmemh({`SCN_DIR, "mem_sel.hex"},  mem_sel);
    $readmemh({`SCN_DIR, "mem_exp.hex"},  mem_exp);
    $readmemh({`SCN_DIR, "cmds.hex"},     cmds);
    $readmemh({`SCN_DIR, "beats.hex"},    beats);
    $readmemh({`SCN_DIR, "lowsel.hex"},   lowsel);
    $readmemh({`SCN_DIR, "victims.hex"},  victims);
    $readmemh({`SCN_DIR, "csr.hex"},      csrv);
    if ($isunknown(mem_exp[NROWS-1]) || $isunknown(cmds[0])) $fatal(1, "vector files missing");
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);
    for (int i = 0; csrv[i][39:32] != 8'hFF; i++) begin
      axil_write(csrv[i][39:32], csrv[i][31:0]);
      if (csrv[i][39:32] == 8'h00) ctrl_val = csrv[i][31:0];
    end

    for (int c = 0; c < `N_CMDS; c++) begin
      logic [3:0] op;
      op = cmds[c][31:28];
      if (op == 4'd1) begin
        int t, nb;
        t  = int'(cmds[c][8:0]);
        nb = (t == 0) ? 1 : (t + P - 1) / P;
        if (t == 0) n_drop++;
        for (int k = 0; k < nb; k++) begin
          @(negedge clk);
          while ($urandom_range(0, 7) == 0) begin s_tvalid = 0; @(negedge clk); end
          s_tvalid = 1; s_tuser = cmds[c][TU_W-1:0];
          {s_tkeep, s_tdata} = beats[bptr];
          s_tlast = (k == nb - 1);
          do @(posedge clk); while (!s_tready);
          bptr++;
        end
        @(negedge clk) s_tvalid = 0;
      end else if (op == 4'd2) begin
        do_flush();
        compare_mem("after prefill flush", 1);
        while (lowsel[lptr] != 32'hFFFFFFFF) begin          // host: end-of-prefill LOW set
          int w;
          w = int'(lowsel[lptr]);
          mem[w / P][32*(w % P) + 30 +: 2] = 2'b11;
          lptr++;
        end
        lptr++;
      end else if (op == 4'd3) begin
        do_flush();
        repeat (200) @(posedge clk);                          // let acks drain
        compare_mem("final", 0);
        axil_read(8'h04, st);
        axil_read(8'h60, sweeps);  axil_read(8'h64, vics);
        axil_read(8'h68, stalls);  axil_read(8'h6C, mis);
        if (st[13:8] !== 6'(`EXP_ERR)) begin
          $display("ERROR sticky errors %02h, expected %02h", st[13:8], `EXP_ERR); errors++;
        end
        if (vptr != `N_VICTIMS || vics != 32'(`N_VICTIMS)) begin
          $display("ERROR victims seen %0d counted %0d expected %0d", vptr, vics, `N_VICTIMS); errors++;
        end
        if (sweeps != 32'(`N_PKTS - n_drop)) begin
          $display("ERROR SWEEPS %0d expected %0d", sweeps, `N_PKTS - n_drop); errors++;
        end
        $display("perf: %0d cycles, %0d beats, %0d sweeps, %0d stall cycles, %0d prefetch misses",
                 cycle, bptr, sweeps, stalls, mis);
      end
    end
    if (errors == 0) $display("PASS tb_ipu_top [%s]", `SCN_DIR);
    else             $display("FAIL tb_ipu_top [%s]: %0d errors", `SCN_DIR, errors);
    $finish;
  end

  initial begin
    #(200_000_000);
    $display("FAIL tb_ipu_top: timeout");
    $finish;
  end
endmodule
