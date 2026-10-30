// ipu_sram_1r1w -- behavioural 1-read 1-write synchronous SRAM (score bank).
//
// Read data appears the cycle after the read enable (macro-style output register).
// Replace the body with a technology macro wrapper later (spec: SRAM macros TBD).
// Read and write of the same address in the same cycle never happen in the IPU.
module ipu_sram_1r1w #(
  parameter int W     = 512,
  parameter int DEPTH = 256
) (
  input  logic                     clk,
  input  logic                     re,
  input  logic [$clog2(DEPTH)-1:0] raddr,
  output logic [W-1:0]             rdata,
  input  logic                     we,
  input  logic [$clog2(DEPTH)-1:0] waddr,
  input  logic [W-1:0]             wdata
);
  logic [W-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    if (re) rdata <= mem[raddr];
  end
endmodule
