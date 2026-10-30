// ipu_issue -- demote packet FIFO + migration credit counter.
//
// A packet leaves only while fewer than `credit_limit` packets are un-acknowledged.
// The migration engine returns one `ack` pulse per completed migration.
module ipu_issue #(
  parameter int W     = 25,
  parameter int DEPTH = 4
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic [4:0]   credit_limit,
  input  logic         push,
  input  logic [W-1:0] push_data,
  output logic         can_push,
  output logic         m_dem_valid,
  input  logic         m_dem_ready,
  output logic [W-1:0] m_dem_data,
  input  logic         s_ack_valid
);
  logic       f_valid, pop;
  logic [5:0] outst;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [$clog2(DEPTH+1)-1:0] cnt;
  /* verilator lint_on UNUSEDSIGNAL */

  ipu_fifo #(.W(W), .DEPTH(DEPTH)) u_f (
    .clk(clk), .rst_n(rst_n), .in_valid(push), .in_ready(can_push), .in_data(push_data),
    .out_valid(f_valid), .out_ready(pop), .out_data(m_dem_data), .count(cnt));

  assign m_dem_valid = f_valid && (outst < {1'b0, credit_limit});
  assign pop         = m_dem_valid && m_dem_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)                   outst <= '0;
    else if (pop && !s_ack_valid) outst <= outst + 1'b1;
    else if (!pop && s_ack_valid && outst != 6'd0) outst <= outst - 1'b1;
  end
endmodule
