// ipu_rd_eng -- AXI4 read engine: fills the score bank with one head's rows.
//
// One job at a time: {hid, target rows}. Rows are requested in bursts that never
// leave a 16-row (1 KB) aligned block, so no burst crosses a 4 KB boundary.
// R data is registered once (rq) and written to the bank on the next cycle.
// Chase rule for a prefetch job (cur = 0): row r is accepted from R only after the
// current decode sweep has already read bank row r (r < sweep_rows), or when no
// decode sweep is active.
module ipu_rd_eng #(
  parameter int P       = 16,
  parameter int ROW_W   = 8,
  parameter int HID_W   = 13,
  parameter int ADDR_W  = 40,
  parameter int MAX_OUT = 4
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic [ADDR_W-HID_W-ROW_W-$clog2(4*P)-1:0] base,
  // commands (single-cycle)
  input  logic                 cmd_load,
  input  logic [HID_W-1:0]     cmd_hid,
  input  logic [ROW_W:0]       cmd_target,
  input  logic                 cmd_cur,
  input  logic                 cmd_extend,      // target <- cmd_target (larger)
  input  logic                 cmd_set_cur,
  input  logic                 cmd_stop,        // stop requesting (target <- requested)
  input  logic                 cmd_inval,
  // chase rule inputs
  input  logic                 sweep_active,
  input  logic [ROW_W:0]       sweep_rows,
  // status
  output logic                 job_valid,
  output logic                 job_cur,
  output logic [HID_W-1:0]     job_hid,
  output logic [ROW_W:0]       job_target,
  output logic [ROW_W:0]       job_bank,        // rows written into the bank
  output logic                 idle,            // nothing outstanding
  output logic                 err_axi,
  // bank write
  output logic                 bank_we,
  output logic [ROW_W-1:0]     bank_waddr,
  output logic [32*P-1:0]      bank_wdata,
  // AXI4 AR / R
  output logic [ADDR_W-1:0]    m_araddr,
  output logic [7:0]           m_arlen,
  output logic                 m_arvalid,
  input  logic                 m_arready,
  input  logic [32*P-1:0]      m_rdata,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [1:0]           m_rresp,        // only SLVERR/DECERR (bit 1) checked
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                 m_rlast,
  input  logic                 m_rvalid,
  output logic                 m_rready
);
  localparam int OFF_W = $clog2(4*P);
  localparam int OW    = $clog2(MAX_OUT + 1);

  logic [ROW_W:0] job_req, job_land;
  logic [OW-1:0]  out_b;
  logic [4:0]     blen;                         // rows in the next burst, 1..16
  logic [ROW_W:0] room, left;
  logic           ar_hs, r_hs, rq_v;
  logic [ROW_W-1:0] rq_row;

  assign left  = job_target - job_req;
  assign room  = (ROW_W+1)'(16) - (ROW_W+1)'(job_req[3:0]);
  assign blen  = (left < room) ? 5'(left) : 5'(room);

  assign m_arvalid = job_valid && (job_req < job_target) && (out_b < OW'(MAX_OUT));
  assign m_araddr  = {base, job_hid, job_req[ROW_W-1:0], {OFF_W{1'b0}}};
  assign m_arlen   = 8'(blen) - 8'd1;
  assign ar_hs     = m_arvalid && m_arready;

  assign m_rready  = job_valid && (job_cur || !sweep_active || (job_land < sweep_rows));
  assign r_hs      = m_rvalid && m_rready;

  assign idle      = (job_req == job_land) && !rq_v;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      job_valid <= 1'b0; job_cur <= 1'b0;
      job_target <= '0; job_req <= '0; job_land <= '0; job_bank <= '0;
      out_b <= '0; rq_v <= 1'b0; err_axi <= 1'b0;
    end else begin
      err_axi <= r_hs && m_rresp[1];
      rq_v    <= r_hs;
      if (rq_v) job_bank <= job_bank + 1'b1;
      if (ar_hs) job_req <= job_req + (ROW_W+1)'(blen);
      if (r_hs)  job_land <= job_land + 1'b1;
      if (ar_hs && !(r_hs && m_rlast)) out_b <= out_b + 1'b1;
      else if (!ar_hs && r_hs && m_rlast) out_b <= out_b - 1'b1;

      if (cmd_load) begin
        job_valid <= 1'b1; job_cur <= cmd_cur;
        job_target <= cmd_target; job_req <= '0; job_land <= '0; job_bank <= '0;
      end else begin
        if (cmd_extend)  job_target <= cmd_target;
        if (cmd_set_cur) job_cur <= 1'b1;
        if (cmd_stop)    job_target <= job_req + (ar_hs ? (ROW_W+1)'(blen) : '0);
        if (cmd_inval)   job_valid <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (cmd_load) job_hid <= cmd_hid;
    if (r_hs) begin
      bank_wdata <= m_rdata;
      rq_row     <= job_land[ROW_W-1:0];
    end
  end
  assign bank_we    = rq_v;
  assign bank_waddr = rq_row;
endmodule
