// ipu_min_track -- running champion over one sweep (stage CH).
//
// The first beat of a packet loads unconditionally; later beats replace the champion
// only on a strictly smaller cmp, so the earlier (lower idx) winner survives ties.
// `done` pulses the cycle after the last beat, with the final champion on the outputs.
// `found` is false when every lane of the sweep was ineligible (cmp all-ones).
module ipu_min_track #(
  parameter int P  = 16,
  parameter int CW = 42,
  parameter int RW = 8
) (
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  in_v,
  input  logic                  in_first,
  input  logic                  in_last,
  input  logic [RW-1:0]         in_row,
  input  logic [CW-1:0]         in_cmp,
  input  logic [$clog2(P)-1:0]  in_lane,
  output logic                  done,
  output logic                  found,
  output logic [CW-1:0]         champ_cmp,
  output logic [RW+$clog2(P)-1:0] champ_idx
);
  logic take;
  assign take = in_v && (in_first || (in_cmp < champ_cmp));

  always_ff @(posedge clk) begin
    if (take) begin
      champ_cmp <= in_cmp;
      champ_idx <= {in_row, in_lane};
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) done <= 1'b0;
    else        done <= in_v && in_last;
  end

  assign found = ~&champ_cmp;
endmodule
