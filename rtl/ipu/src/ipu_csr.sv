// ipu_csr -- AXI4-Lite register block (spec v0.2 Section 5.4).
//
// 32-bit registers, full-word writes (WSTRB ignored). One transaction at a time.
module ipu_csr #(
  parameter int NSEQ   = 8,
  parameter int T_W    = 13,
  parameter int BASE_W = 13,
  parameter int P      = 16,
  parameter int IDX_W  = 12,
  parameter int HAS_PERF = 1
) (
  input  logic              clk,
  input  logic              rst_n,
  // AXI4-Lite slave
  input  logic [7:0]        s_awaddr,
  input  logic              s_awvalid,
  output logic              s_awready,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [31:0]       s_wdata,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic              s_wvalid,
  output logic              s_wready,
  output logic [1:0]        s_bresp,
  output logic              s_bvalid,
  input  logic              s_bready,
  input  logic [7:0]        s_araddr,
  input  logic              s_arvalid,
  output logic              s_arready,
  output logic [31:0]       s_rdata,
  output logic [1:0]        s_rresp,
  output logic              s_rvalid,
  input  logic              s_rready,
  // configuration out
  output logic              enable,
  output logic              raw,
  output logic              flush_req,      // pulse
  output logic [BASE_W-1:0] base,
  output logic [4:0]        l_max,
  output logic [4:0]        h_max,
  output logic [2:0]        b_max,
  output logic [4:0]        credit_limit,
  output logic [NSEQ*T_W-1:0] k_flat,
  // status in
  input  logic              busy,
  input  logic              flush_pending,
  input  logic [5:0]        err_set,        // pulses: BAD_PROB, SAT, NO_ELIG, T_RANGE, SEQ, AXI
  input  logic              inc_sweep,
  input  logic              inc_victim,
  input  logic              inc_stall,
  input  logic              inc_mispred,
  output logic              irq
);
  logic        irq_en;
  logic [5:0]  err;
  logic [31:0] c_sweep, c_vic, c_stall, c_mis;
  logic        wr_hs, rd_hs;

  assign s_awready = s_awvalid && s_wvalid && !s_bvalid;
  assign s_wready  = s_awready;
  assign wr_hs     = s_awready;
  assign s_bresp   = 2'b00;
  assign s_arready = s_arvalid && !s_rvalid;
  assign rd_hs     = s_arready;
  assign s_rresp   = 2'b00;

  function automatic logic [6:0] weff_of(input logic [T_W-1:0] k);
    logic [T_W-1:0] h;
    h = k >> 1;
    weff_of = (h > 64) ? 7'd64 : ((h == 0) ? 7'd1 : 7'(h));
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      enable <= 1'b0; raw <= 1'b0; irq_en <= 1'b0; flush_req <= 1'b0;
      base <= '0; l_max <= 5'd31; h_max <= 5'd31; b_max <= 3'd7; credit_limit <= 5'd8;
      k_flat <= '0; err <= '0; s_bvalid <= 1'b0; s_rvalid <= 1'b0; s_rdata <= '0;
    end else begin
      flush_req <= 1'b0;
      err <= err | err_set;
      if (wr_hs) begin
        s_bvalid <= 1'b1;
        case (s_awaddr)
          8'h00: begin
            enable <= s_wdata[0]; raw <= s_wdata[1]; irq_en <= s_wdata[2];
            flush_req <= s_wdata[8];
          end
          8'h04: err <= (err | err_set) & ~s_wdata[13:8];          // W1C
          8'h08: base <= s_wdata[BASE_W-1:0];
          8'h0C: begin l_max <= s_wdata[4:0]; h_max <= s_wdata[9:5]; b_max <= s_wdata[12:10]; end
          8'h10: credit_limit <= s_wdata[4:0];
          default: begin
            for (int b = 0; b < NSEQ; b++)
              if (s_awaddr == 8'h20 + 8'(4*b)) k_flat[T_W*b +: T_W] <= s_wdata[T_W-1:0];
          end
        endcase
      end else if (s_bvalid && s_bready) begin
        s_bvalid <= 1'b0;
      end

      if (rd_hs) begin
        s_rvalid <= 1'b1;
        s_rdata  <= 32'd0;
        case (s_araddr)
          8'h00: s_rdata <= {29'd0, irq_en, raw, enable};
          8'h04: s_rdata <= {18'd0, err, 6'd0, flush_pending, busy};
          8'h08: s_rdata <= 32'(base);
          8'h0C: s_rdata <= {19'd0, b_max, h_max, l_max};
          8'h10: s_rdata <= {27'd0, credit_limit};
          8'h14: s_rdata <= {16'h0002, 8'(P), 8'(IDX_W)};
          8'h60: s_rdata <= c_sweep;
          8'h64: s_rdata <= c_vic;
          8'h68: s_rdata <= c_stall;
          8'h6C: s_rdata <= c_mis;
          default: begin
            for (int b = 0; b < NSEQ; b++)
              if (s_araddr == 8'h20 + 8'(4*b))
                s_rdata <= {9'd0, weff_of(k_flat[T_W*b +: T_W]), 16'(k_flat[T_W*b +: T_W])};
          end
        endcase
      end else if (s_rvalid && s_rready) begin
        s_rvalid <= 1'b0;
      end
    end
  end

  if (HAS_PERF != 0) begin : g_perf
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        c_sweep <= '0; c_vic <= '0; c_stall <= '0; c_mis <= '0;
      end else begin
        if (inc_sweep)   c_sweep <= c_sweep + 1'b1;
        if (inc_victim)  c_vic   <= c_vic + 1'b1;
        if (inc_stall)   c_stall <= c_stall + 1'b1;
        if (inc_mispred) c_mis   <= c_mis + 1'b1;
      end
    end
  end else begin : g_noperf
    assign c_sweep = '0; assign c_vic = '0; assign c_stall = '0; assign c_mis = '0;
  end

  assign irq = irq_en && (|err);
endmodule
