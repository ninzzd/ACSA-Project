// ipu_min_tree -- pipelined argmin over P lanes, one comparator level per stage.
//
// P-1 two-input comparators in log2(P) register stages. Left (lower lane) wins on
// equal, which gives "lowest index wins" because lanes are in ascending idx order.
// Carries the beat's row number and first/last flags alongside. Never stalls.
module ipu_min_tree #(
  parameter int P     = 16,
  parameter int CW    = 42,
  parameter int RW    = 8
) (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              in_v,
  input  logic              in_first,
  input  logic              in_last,
  input  logic [RW-1:0]     in_row,
  input  logic [CW*P-1:0]   in_cmp,
  output logic              out_v,
  output logic              out_first,
  output logic              out_last,
  output logic [RW-1:0]     out_row,
  output logic [CW-1:0]     out_cmp,
  output logic [$clog2(P)-1:0] out_lane
);
  localparam int LV = $clog2(P);

  // Node storage for levels 1..LV, level k at node offset P - (P >> (k-1)).
  logic [CW*(P-1)-1:0]  nc;
  logic [LV*(P-1)-1:0]  nid;
  logic [LV:1]          lv_v, lv_first, lv_last;
  logic [RW*LV-1:0]     lv_row;         // level k at bits RW*(k-1)

  genvar k, j;
  for (k = 1; k <= LV; k++) begin : g_lvl
    localparam int N    = P >> k;                          // nodes at this level
    localparam int OFF  = P - (P >> (k-1));                // node offset, this level
    localparam int POFF = (k >= 2) ? P - (P >> (k-2)) : 0; // node offset, level k-1
    logic          pv, pfirst, plast;
    logic [RW-1:0] prow;

    if (k == 1) begin : g_src0
      assign pv     = in_v;
      assign pfirst = in_first;
      assign plast  = in_last;
      assign prow   = in_row;
    end else begin : g_srck
      assign pv     = lv_v[k-1];
      assign pfirst = lv_first[k-1];
      assign plast  = lv_last[k-1];
      assign prow   = lv_row[RW*(k-2) +: RW];
    end

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) lv_v[k] <= 1'b0;
      else        lv_v[k] <= pv;
    end
    always_ff @(posedge clk) begin
      lv_first[k]            <= pfirst;
      lv_last[k]             <= plast;
      lv_row[RW*(k-1) +: RW] <= prow;
    end

    for (j = 0; j < N; j++) begin : g_node
      logic [CW-1:0] ca, cb;
      logic [LV-1:0] ia, ib, iw;
      logic          a_wins;
      if (k == 1) begin : g_leaf
        assign ca = in_cmp[CW*(2*j)   +: CW];
        assign cb = in_cmp[CW*(2*j+1) +: CW];
        assign ia = '0;
        assign ib = '0;
      end else begin : g_inner
        assign ca = nc[CW*(POFF+2*j)   +: CW];
        assign cb = nc[CW*(POFF+2*j+1) +: CW];
        assign ia = nid[LV*(POFF+2*j)   +: LV];
        assign ib = nid[LV*(POFF+2*j+1) +: LV];
      end
      assign a_wins = (ca <= cb);
      // lane id: bit k-1 = 1 when the right child (higher lanes) wins
      always_comb begin
        iw = a_wins ? ia : ib;
        iw[k-1] = ~a_wins;
      end
      always_ff @(posedge clk) begin
        nc[CW*(OFF+j) +: CW]  <= a_wins ? ca : cb;
        nid[LV*(OFF+j) +: LV] <= iw;
      end
    end
  end

  assign out_v     = lv_v[LV];
  assign out_first = lv_first[LV];
  assign out_last  = lv_last[LV];
  assign out_row   = lv_row[RW*(LV-1) +: RW];
  assign out_cmp   = nc[CW*(P-2) +: CW];
  assign out_lane  = nid[LV*(P-2) +: LV];
endmodule
