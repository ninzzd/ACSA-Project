// ipu_fifo -- small synchronous FIFO with registered full/empty flags.
//
// in_ready = !full is a register output, so the producer's stall logic never sees a
// combinational path from the consumer. DEPTH = 2 gives a full-throughput skid buffer.
module ipu_fifo #(
  parameter int W     = 8,
  parameter int DEPTH = 2
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         in_valid,
  output logic         in_ready,
  input  logic [W-1:0] in_data,
  output logic         out_valid,
  input  logic         out_ready,
  output logic [W-1:0] out_data,
  output logic [$clog2(DEPTH+1)-1:0] count
);
  localparam int AW = (DEPTH > 1) ? $clog2(DEPTH) : 1;

  logic [W-1:0]  mem [0:DEPTH-1];
  logic [AW-1:0] wp, rp;
  logic          full, empty;
  logic          push, pop;

  assign push      = in_valid && !full;
  assign pop       = out_ready && !empty;
  assign in_ready  = !full;
  assign out_valid = !empty;
  assign out_data  = mem[rp];

  function automatic logic [AW-1:0] inc(input logic [AW-1:0] p);
    inc = (p == AW'(DEPTH - 1)) ? '0 : p + 1'b1;
  endfunction

  always_ff @(posedge clk) begin
    if (push) mem[wp] <= in_data;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wp    <= '0;
      rp    <= '0;
      count <= '0;
      full  <= 1'b0;
      empty <= 1'b1;
    end else begin
      if (push) wp <= inc(wp);
      if (pop)  rp <= inc(rp);
      if (push && !pop) begin
        count <= count + 1'b1;
        empty <= 1'b0;
        full  <= (count == ($clog2(DEPTH+1))'(DEPTH - 1));
      end else if (pop && !push) begin
        count <= count - 1'b1;
        full  <= 1'b0;
        empty <= (count == ($clog2(DEPTH+1))'(1));
      end
    end
  end
endmodule
