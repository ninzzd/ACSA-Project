// ipu_top -- Importance Policy Unit (spec v0.2).
//
// Ports: attention AXI4-Stream in, demote packets out + acks in, AXI4 master to the
// off-chip score region, AXI4-Lite CSR slave, interrupt.
module ipu_top #(
  parameter int P           = 16,
  parameter int T_MAX       = 4096,
  parameter int B_W         = 3,
  parameter int L_W         = 5,
  parameter int H_W         = 5,
  parameter int NSEQ        = 8,
  parameter int ADDR_W      = 40,
  parameter int ISSUE_DEPTH = 4,
  parameter int MAX_RD_OUT  = 4,
  parameter int HAS_PERF    = 1,
  // derived (do not override)
  parameter int IDX_W  = $clog2(T_MAX),
  parameter int T_W    = IDX_W + 1,
  parameter int ROW_W  = $clog2(T_MAX / P),
  parameter int HID_W  = B_W + L_W + H_W,
  parameter int TU_W   = 2 + HID_W + T_W,
  parameter int DW     = 32 * P,
  parameter int BASE_W = ADDR_W - HID_W - ROW_W - $clog2(4 * P)
) (
  input  logic                 clk,
  input  logic                 rst_n,
  // attention stream
  input  logic [16*P-1:0]      s_attn_tdata,
  input  logic [P-1:0]         s_attn_tkeep,
  input  logic [TU_W-1:0]      s_attn_tuser,
  input  logic                 s_attn_tlast,
  input  logic                 s_attn_tvalid,
  output logic                 s_attn_tready,
  // demote packets / acks
  output logic                 m_dem_valid,
  input  logic                 m_dem_ready,
  output logic [HID_W+IDX_W-1:0] m_dem_data,
  input  logic                 s_ack_valid,
  // AXI4 master (score region)
  output logic [ADDR_W-1:0]    m_axi_awaddr,
  output logic [7:0]           m_axi_awlen,
  output logic [2:0]           m_axi_awsize,
  output logic [1:0]           m_axi_awburst,
  output logic                 m_axi_awvalid,
  input  logic                 m_axi_awready,
  output logic [DW-1:0]        m_axi_wdata,
  output logic [DW/8-1:0]      m_axi_wstrb,
  output logic                 m_axi_wlast,
  output logic                 m_axi_wvalid,
  input  logic                 m_axi_wready,
  input  logic [1:0]           m_axi_bresp,
  input  logic                 m_axi_bvalid,
  output logic                 m_axi_bready,
  output logic [ADDR_W-1:0]    m_axi_araddr,
  output logic [7:0]           m_axi_arlen,
  output logic [2:0]           m_axi_arsize,
  output logic [1:0]           m_axi_arburst,
  output logic                 m_axi_arvalid,
  input  logic                 m_axi_arready,
  input  logic [DW-1:0]        m_axi_rdata,
  input  logic [1:0]           m_axi_rresp,
  input  logic                 m_axi_rlast,
  input  logic                 m_axi_rvalid,
  output logic                 m_axi_rready,
  // AXI4-Lite CSR slave
  input  logic [7:0]           s_axil_awaddr,
  input  logic                 s_axil_awvalid,
  output logic                 s_axil_awready,
  input  logic [31:0]          s_axil_wdata,
  input  logic                 s_axil_wvalid,
  output logic                 s_axil_wready,
  output logic [1:0]           s_axil_bresp,
  output logic                 s_axil_bvalid,
  input  logic                 s_axil_bready,
  input  logic [7:0]           s_axil_araddr,
  input  logic                 s_axil_arvalid,
  output logic                 s_axil_arready,
  output logic [31:0]          s_axil_rdata,
  output logic [1:0]           s_axil_rresp,
  output logic                 s_axil_rvalid,
  input  logic                 s_axil_rready,
  output logic                 irq
);
  localparam int LV = $clog2(P);

  // ---------------- CSR
  logic              enable, raw, flush_req, flush_pending, busy;
  logic [BASE_W-1:0] base;
  logic [4:0]        l_max, h_max, credit_limit;
  logic [2:0]        b_max;
  logic [NSEQ*T_W-1:0] k_flat;
  logic              ev_sweep, ev_victim, ev_stall, ev_mispred;
  logic              e_bad, e_sat, e_noel, e_trange, e_seq, e_rd_axi, e_wr_axi;

  ipu_csr #(.NSEQ(NSEQ), .T_W(T_W), .BASE_W(BASE_W), .P(P), .IDX_W(IDX_W), .HAS_PERF(HAS_PERF)) u_csr (
    .clk(clk), .rst_n(rst_n),
    .s_awaddr(s_axil_awaddr), .s_awvalid(s_axil_awvalid), .s_awready(s_axil_awready),
    .s_wdata(s_axil_wdata), .s_wvalid(s_axil_wvalid), .s_wready(s_axil_wready),
    .s_bresp(s_axil_bresp), .s_bvalid(s_axil_bvalid), .s_bready(s_axil_bready),
    .s_araddr(s_axil_araddr), .s_arvalid(s_axil_arvalid), .s_arready(s_axil_arready),
    .s_rdata(s_axil_rdata), .s_rresp(s_axil_rresp), .s_rvalid(s_axil_rvalid), .s_rready(s_axil_rready),
    .enable(enable), .raw(raw), .flush_req(flush_req), .base(base), .l_max(l_max), .h_max(h_max),
    .b_max(b_max), .credit_limit(credit_limit), .k_flat(k_flat), .busy(busy),
    .flush_pending(flush_pending),
    .err_set({e_rd_axi | e_wr_axi, e_seq, e_trange, e_noel, e_sat, e_bad}),
    .inc_sweep(ev_sweep), .inc_victim(ev_victim), .inc_stall(ev_stall), .inc_mispred(ev_mispred),
    .irq(irq));

  // ---------------- pipeline
  logic               acc, beat_first, beat_last, beat_dec, stall, s123_busy, done, found;
  logic [IDX_W-1:0]   beat_a0, champ_idx;
  logic [ROW_W-1:0]   beat_row;
  logic [6:0]         w_eff;
  logic [41:0]        champ_cmp;
  logic [DW-1:0]      bank_rdata, s3_word;
  logic               p_wb_push, p_bank_we;
  logic [ROW_W-1:0]   p_bank_waddr;
  logic               wb_ready;

  ipu_pipe #(.P(P), .IDX_W(IDX_W), .ROW_W(ROW_W)) u_pipe (
    .clk(clk), .rst_n(rst_n), .acc(acc), .fp16(s_attn_tdata), .keep(s_attn_tkeep),
    .a0(beat_a0), .row(beat_row), .first(beat_first), .last(beat_last), .dec(beat_dec),
    .w_eff(w_eff), .raw(raw), .bank_rdata(bank_rdata), .wb_ready(wb_ready),
    .wb_push(p_wb_push), .s3_word(s3_word), .bank_we(p_bank_we), .bank_waddr(p_bank_waddr),
    .stall(stall), .s123_busy(s123_busy), .err_bad(e_bad), .err_sat(e_sat),
    .done(done), .found(found), .champ_cmp(champ_cmp), .champ_idx(champ_idx));

  // ---------------- control
  logic               dr_re, dr_push;
  logic [ROW_W-1:0]   dr_raddr;
  logic               rd_load, rd_cur, rd_extend, rd_set_cur, rd_stop, rd_inval, sweep_active;
  logic [HID_W-1:0]   rd_hid, rdj_hid, wr_hid;
  logic [ROW_W:0]     rd_target, sweep_rows, rdj_target, rdj_bank, wr_n;
  logic               rdj_valid, rdj_cur, rd_idle;
  logic               wr_start_rows, wr_start_vic, wr_job_idle, wr_all_idle, wr_prev_idle;
  logic [ROW_W-1:0]   wr_row;
  logic               vic_push, iss_can_push, iss_push;
  logic [31:0]        vic_word;
  logic [LV-1:0]      vic_lane;
  logic [HID_W+IDX_W-1:0] iss_data;

  ipu_seq_fsm #(.P(P), .T_MAX(T_MAX), .B_W(B_W), .L_W(L_W), .H_W(H_W), .NSEQ(NSEQ)) u_fsm (
    .clk(clk), .rst_n(rst_n), .enable(enable), .raw(raw), .flush_req(flush_req),
    .l_max(l_max), .h_max(h_max), .b_max(b_max), .k_flat(k_flat),
    .s_tvalid(s_attn_tvalid), .s_tlast(s_attn_tlast), .s_tuser(s_attn_tuser), .s_tready(s_attn_tready),
    .acc(acc), .beat_a0(beat_a0), .beat_row(beat_row), .beat_first(beat_first),
    .beat_last(beat_last), .beat_dec(beat_dec), .w_eff(w_eff), .stall(stall),
    .s123_busy(s123_busy), .done(done), .found(found), .champ_cmp(champ_cmp), .champ_idx(champ_idx),
    .dr_re(dr_re), .dr_raddr(dr_raddr), .dr_push(dr_push),
    .rd_load(rd_load), .rd_hid(rd_hid), .rd_target(rd_target), .rd_cur(rd_cur),
    .rd_extend(rd_extend), .rd_set_cur(rd_set_cur), .rd_stop(rd_stop), .rd_inval(rd_inval),
    .sweep_active(sweep_active), .sweep_rows(sweep_rows),
    .rdj_valid(rdj_valid), .rdj_cur(rdj_cur), .rdj_hid(rdj_hid), .rdj_target(rdj_target),
    .rdj_bank(rdj_bank), .rd_idle(rd_idle),
    .wr_start_rows(wr_start_rows), .wr_start_vic(wr_start_vic), .wr_hid(wr_hid), .wr_n(wr_n),
    .wr_row(wr_row), .wr_job_idle(wr_job_idle), .wr_all_idle(wr_all_idle),
    .wr_prev_idle(wr_prev_idle), .wb_ready(wb_ready),
    .vic_push(vic_push), .vic_word(vic_word), .vic_lane(vic_lane),
    .iss_can_push(iss_can_push), .iss_push(iss_push), .iss_data(iss_data),
    .busy(busy), .flush_pending(flush_pending), .ev_sweep(ev_sweep), .ev_victim(ev_victim),
    .ev_stall(ev_stall), .ev_mispred(ev_mispred), .err_no_elig(e_noel), .err_t_range(e_trange),
    .err_seq(e_seq));

  // ---------------- score bank (single 1R1W bank)
  logic               rd_bank_we;
  logic [ROW_W-1:0]   rd_bank_waddr;
  logic [DW-1:0]      rd_bank_wdata;

  ipu_sram_1r1w #(.W(DW), .DEPTH(T_MAX / P)) u_bank (
    .clk(clk),
    .re(acc | dr_re), .raddr(dr_re ? dr_raddr : beat_row), .rdata(bank_rdata),
    .we(p_bank_we | rd_bank_we), .waddr(p_bank_we ? p_bank_waddr : rd_bank_waddr),
    .wdata(p_bank_we ? s3_word : rd_bank_wdata));

  // ---------------- read engine
  ipu_rd_eng #(.P(P), .ROW_W(ROW_W), .HID_W(HID_W), .ADDR_W(ADDR_W), .MAX_OUT(MAX_RD_OUT)) u_rd (
    .clk(clk), .rst_n(rst_n), .base(base),
    .cmd_load(rd_load), .cmd_hid(rd_hid), .cmd_target(rd_target), .cmd_cur(rd_cur),
    .cmd_extend(rd_extend), .cmd_set_cur(rd_set_cur), .cmd_stop(rd_stop), .cmd_inval(rd_inval),
    .sweep_active(sweep_active), .sweep_rows(sweep_rows),
    .job_valid(rdj_valid), .job_cur(rdj_cur), .job_hid(rdj_hid), .job_target(rdj_target),
    .job_bank(rdj_bank), .idle(rd_idle), .err_axi(e_rd_axi),
    .bank_we(rd_bank_we), .bank_waddr(rd_bank_waddr), .bank_wdata(rd_bank_wdata),
    .m_araddr(m_axi_araddr), .m_arlen(m_axi_arlen), .m_arvalid(m_axi_arvalid),
    .m_arready(m_axi_arready), .m_rdata(m_axi_rdata), .m_rresp(m_axi_rresp),
    .m_rlast(m_axi_rlast), .m_rvalid(m_axi_rvalid), .m_rready(m_axi_rready));

  // ---------------- write engine + W buffer
  logic         w_push_rows, w_last_tag;
  logic         wb_in_valid, wb_out_vic;
  logic [DW-1:0] wb_in_data;
  logic [LV-1:0] wb_out_lane;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]   wb_cnt;
  /* verilator lint_on UNUSEDSIGNAL */

  assign w_push_rows = p_wb_push | dr_push;
  assign wb_in_valid = w_push_rows | vic_push;
  assign wb_in_data  = vic_push ? {P{vic_word}} : (dr_push ? bank_rdata : s3_word);

  ipu_wr_eng #(.P(P), .ROW_W(ROW_W), .HID_W(HID_W), .ADDR_W(ADDR_W)) u_wr (
    .clk(clk), .rst_n(rst_n), .base(base),
    .start_rows(wr_start_rows), .start_vic(wr_start_vic), .start_hid(wr_hid), .start_n(wr_n),
    .start_row(wr_row), .w_push(w_push_rows), .w_last(w_last_tag),
    .job_idle(wr_job_idle), .all_idle(wr_all_idle), .prev_idle(wr_prev_idle), .err_axi(e_wr_axi),
    .m_awaddr(m_axi_awaddr), .m_awlen(m_axi_awlen), .m_awvalid(m_axi_awvalid),
    .m_awready(m_axi_awready), .m_bresp(m_axi_bresp), .m_bvalid(m_axi_bvalid),
    .m_bready(m_axi_bready));

  ipu_fifo #(.W(DW + 2 + LV), .DEPTH(2)) u_wbuf (
    .clk(clk), .rst_n(rst_n),
    .in_valid(wb_in_valid), .in_ready(wb_ready),
    .in_data({wb_in_data, (vic_push | w_last_tag), vic_push, vic_lane}),
    .out_valid(m_axi_wvalid), .out_ready(m_axi_wready),
    .out_data({m_axi_wdata, m_axi_wlast, wb_out_vic, wb_out_lane}),
    .count(wb_cnt));

  always_comb begin
    m_axi_wstrb = '1;
    if (wb_out_vic) begin
      m_axi_wstrb = '0;
      m_axi_wstrb[4*wb_out_lane +: 4] = 4'hF;
    end
  end

  assign m_axi_awsize  = 3'($clog2(4 * P));
  assign m_axi_arsize  = 3'($clog2(4 * P));
  assign m_axi_awburst = 2'b01;
  assign m_axi_arburst = 2'b01;

  // ---------------- demote issue
  ipu_issue #(.W(HID_W + IDX_W), .DEPTH(ISSUE_DEPTH)) u_iss (
    .clk(clk), .rst_n(rst_n), .credit_limit(credit_limit),
    .push(iss_push), .push_data(iss_data), .can_push(iss_can_push),
    .m_dem_valid(m_dem_valid), .m_dem_ready(m_dem_ready), .m_dem_data(m_dem_data),
    .s_ack_valid(s_ack_valid));
endmodule
