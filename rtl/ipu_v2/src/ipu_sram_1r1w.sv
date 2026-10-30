// ipu_sram_1r1w -- 1-read 1-write synchronous SRAM (score bank), bit write mask.
//
// Read data appears the cycle after the read enable (macro-style output register).
// wmask follows the FakeRAM45 w_mask_in convention: a 1 bit is written.
// Read and write of the same address in the same cycle never happen in the IPU.
// `define IPU_SRAM_MACRO to instantiate the FakeRAM45-style 1R1W macro
// (fakeram45_256x512_1r1w, syn/fakeram/) instead of the behavioural array.
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
  input  logic [W-1:0]             wdata,
  input  logic [W-1:0]             wmask
);
`ifdef IPU_SRAM_MACRO
  fakeram45_256x512_1r1w u_macro (
    .clk(clk), .r_ce_in(re), .r_addr_in(raddr), .rd_out(rdata),
    .w_ce_in(we), .w_addr_in(waddr), .wd_in(wdata), .w_mask_in(wmask));
`else
  logic [W-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= (mem[waddr] & ~wmask) | (wdata & wmask);
    if (re) rdata <= mem[raddr];
  end
`endif
endmodule
