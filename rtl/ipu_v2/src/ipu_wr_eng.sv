// ipu_wr_eng -- AXI4 write engine: address/burst generation, W `last` tagging, B tracking.
//
// Jobs (one at a time):
//   rows job : rows 0..n-1 of one head (decode write-back or prefill drain)
//   vic job  : one row-aligned beat holding the victim word (partial WSTRB)
// W data is pushed by the data source (pipe S3, bank drain, or the FSM's victim beat)
// into the W buffer; this block only counts pushes to tag `last` and to know when the
// job is complete. Bursts never leave a 16-row aligned block (no 4 KB crossing).
module ipu_wr_eng #(
  parameter int P      = 16,
  parameter int ROW_W  = 8,
  parameter int HID_W  = 13,
  parameter int ADDR_W = 40
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic [ADDR_W-HID_W-ROW_W-$clog2(4*P)-1:0] base,
  input  logic                 start_rows,
  input  logic                 start_vic,       // the victim beat is pushed in the same cycle
  input  logic [HID_W-1:0]     start_hid,
  input  logic [ROW_W:0]       start_n,         // rows (rows job)
  input  logic [ROW_W-1:0]     start_row,       // row (vic job)
  input  logic                 w_push,          // a row beat entered the W buffer
  output logic                 w_last,          // `last` tag for the beat pushed now
  output logic                 job_idle,
  output logic                 all_idle,
  output logic                 prev_idle,       // all writes of earlier jobs acknowledged
  output logic                 err_axi,
  output logic [ADDR_W-1:0]    m_awaddr,
  output logic [7:0]           m_awlen,
  output logic                 m_awvalid,
  input  logic                 m_awready,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [1:0]           m_bresp,        // only SLVERR/DECERR (bit 1) checked
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                 m_bvalid,
  output logic                 m_bready
);
  localparam int OFF_W = $clog2(4*P);

  logic             act, vic;
  logic [HID_W-1:0] hid;
  logic [ROW_W:0]   n, aw_sent, w_cnt;
  logic [ROW_W-1:0] vrow;
  logic [7:0]       bcnt, older;
  logic [ROW_W:0]   left, room;
  logic [4:0]       blen;
  logic             aw_hs, done_now;

  assign left  = n - aw_sent;
  assign room  = (ROW_W+1)'(16) - (ROW_W+1)'(aw_sent[3:0]);
  assign blen  = vic ? 5'd1 : ((left < room) ? 5'(left) : 5'(room));

  assign m_awvalid = act && (aw_sent < n);
  assign m_awaddr  = {base, hid, (vic ? vrow : aw_sent[ROW_W-1:0]), {OFF_W{1'b0}}};
  assign m_awlen   = 8'(blen) - 8'd1;
  assign aw_hs     = m_awvalid && m_awready;
  assign m_bready  = 1'b1;

  // last beat of a burst: end of a 16-row block or end of the job
  assign w_last    = (w_cnt[3:0] == 4'hF) || (w_cnt == n - 1'b1);

  assign done_now  = act && (aw_sent + (aw_hs ? (ROW_W+1)'(blen) : '0) == n)
                         && (w_cnt + (ROW_W+1)'(w_push) == n);
  assign job_idle  = !act;
  assign all_idle  = !act && (bcnt == 8'd0);
  assign prev_idle = (older == 8'd0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act <= 1'b0; vic <= 1'b0; n <= '0; aw_sent <= '0; w_cnt <= '0;
      bcnt <= '0; older <= '0; err_axi <= 1'b0;
    end else begin
      err_axi <= m_bvalid && m_bresp[1];
      if (aw_hs && !m_bvalid)      bcnt <= bcnt + 1'b1;
      else if (!aw_hs && m_bvalid) bcnt <= bcnt - 1'b1;

      // B responses arrive in order (single ID): count down the earlier jobs' share
      if (start_rows || start_vic) older <= bcnt - ((m_bvalid && bcnt != 8'd0) ? 8'd1 : 8'd0);
      else if (m_bvalid && older != 8'd0) older <= older - 1'b1;
      if (start_rows || start_vic) begin
        act     <= 1'b1;
        vic     <= start_vic;
        n       <= start_vic ? (ROW_W+1)'(1) : start_n;
        aw_sent <= '0;
        w_cnt   <= start_vic ? (ROW_W+1)'(1) : '0;
      end else if (act) begin
        if (aw_hs)  aw_sent <= aw_sent + (ROW_W+1)'(blen);
        if (w_push) w_cnt   <= w_cnt + 1'b1;
        if (done_now) act <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (start_rows || start_vic) begin
      hid  <= start_hid;
      vrow <= start_row;
    end
  end
endmodule
