// ipu_psel -- end-of-prefill bulk selection (v2, spec Section 4.3).
//
// Runs on the resident (prefill) head before it is drained. With t_p tokens and budget
// K, it demotes D = t_p - K tokens to LOW in one go:
//   candidates : age >= W_EFF            (idx < t_p - W_EFF; the window stays HIGH)
//   order      : decayed cmp ascending, ties -> older (lower idx) first
//   demote     : the first D candidates in that order
// which is exactly the host selection of v1 (ipu_golden.host_select).
//
// Method: radix select on the 42-bit cmp value, one bit per pass, MSB first.
//   pass at bit b : c0 = #{candidates whose cmp matches the decided prefix above b and
//                   has bit b = 0}.  r <= c0 -> T[b] = 0, else T[b] = 1 and r -= c0.
//   after bit 0   : T is the D-th smallest cmp; r = how many cmp == T to demote.
//   final pass    : demote cmp < T, plus the first r (lowest idx) with cmp == T.
// Ineligible lanes carry the all-ones cmp code: they never count as a 0 bit, never
// match T and are never below T, so no separate candidate mask is needed.
// Passes stream one row per cycle (42 x ceil(cand/P) cycles; 30 passes in RAW mode).
// The final pass is row-serial: read, evaluate, emit one index per cycle, then write
// the row's tier bits to LOW with a lane write mask.
module ipu_psel #(
  parameter int P     = 16,
  parameter int T_MAX = 4096
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         start,       // pulse; bank holds the head
  input  logic [$clog2(T_MAX):0]       tp,
  input  logic [$clog2(T_MAX):0]       k,
  input  logic                         raw,
  output logic                         busy,
  // score bank
  output logic                         re,
  output logic [$clog2(T_MAX/P)-1:0]   raddr,
  input  logic [32*P-1:0]              rdata,
  output logic                         we,
  output logic [$clog2(T_MAX/P)-1:0]   waddr,
  output logic [P-1:0]                 wlanes,      // lanes whose tier becomes LOW
  // demote indices (to the issue queue)
  input  logic                         can_push,
  output logic                         push,
  output logic [$clog2(T_MAX)-1:0]     idx
);
  `include "ipu_defs.svh"
  localparam int IDX_W = $clog2(T_MAX);
  localparam int T_W   = IDX_W + 1;
  localparam int ROW_W = $clog2(T_MAX / P);
  localparam int NR_W  = ROW_W + 1;
  localparam int LV    = $clog2(P);

  typedef enum logic [2:0] {S_IDLE, S_PASS, S_DRN, S_DEC, S_FRD, S_FEV, S_FEM} st_t;
  st_t st;

  logic [T_W-1:0]   r, c0, eq_seen;
  logic [NR_W-1:0]  nrows, rcnt;
  logic [ROW_W-1:0] frow;
  logic [CMP_W-1:0] pfx, mhi;            // decided prefix; mask of decided bits
  logic [IDX_W-1:0] a0;                  // age of lane 0 for the row being read
  logic [6:0]       w_q;
  logic             v1, v2;              // pass pipeline valids
  logic [P-1:0]     zq;                  // lanes counted this row (registered)
  logic [P-1:0]     eqq, pend, dmask;
  logic             last_bit;

  // ---------------- per-lane compare values for the returned row
  // Lane control is computed from the row being read and registered alongside the SRAM
  // read, so the SRAM -> shift -> compare path does not also carry the age logic.
  logic [P-1:0]       valid, agew, isnew, cand_q;
  logic [4*P-1:0]     sh, sh_q;
  logic [CMP_W*P-1:0] cmp;

  ipu_lane_ctl #(.P(P), .IDX_W(IDX_W)) u_lc (
    .a0(a0), .w_eff(w_q), .keep({P{1'b1}}), .valid(valid), .isnew(isnew), .agew(agew), .sh(sh));
  ipu_cmp_shift #(.P(P)) u_cs (
    .word(rdata), .sh(sh_q), .elig(cand_q), .raw(raw), .cmp(cmp));

  wire [CMP_W-1:0] bsel = ~mhi & ~(~mhi >> 1);     // the bit being decided
  logic [P-1:0] z_now, lt_now, eq_now;
  for (genvar i = 0; i < P; i++) begin : g_lane
    wire [CMP_W-1:0] c = cmp[CMP_W*i +: CMP_W];
    assign z_now[i]  = ~|((c ^ pfx) & mhi) & ~|(c & bsel);
    assign eq_now[i] = (c == pfx);
    assign lt_now[i] = (c < pfx);
  end

  function automatic logic [LV:0] popcnt(input logic [P-1:0] x);
    popcnt = '0;
    for (int i = 0; i < P; i++) popcnt = popcnt + (LV+1)'(x[i]);
  endfunction

  // lowest pending lane
  logic [LV-1:0] lane;
  always_comb begin
    lane = '0;
    for (int i = P - 1; i >= 0; i--) if (pend[i]) lane = LV'(i);
  end
  wire lane_eq   = eqq[lane];
  wire lane_take = !lane_eq || (eq_seen < r);

  assign busy   = (st != S_IDLE);
  assign re     = (st == S_PASS && rcnt < nrows) || (st == S_FRD);
  assign raddr  = (st == S_FRD) ? frow : rcnt[ROW_W-1:0];
  assign push   = (st == S_FEM) && (|pend) && lane_take && can_push;
  assign idx    = {frow, lane};
  assign we     = (st == S_FEM) && !(|pend) && (|dmask);
  assign waddr  = frow;
  assign wlanes = dmask;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; v1 <= 1'b0; v2 <= 1'b0;
    end else begin
      v1 <= (st == S_PASS) && re;
      v2 <= v1;
      case (st)
        S_IDLE: if (start && (tp > k)) st <= S_PASS;
        S_PASS: if (rcnt == nrows) st <= S_DRN;
        S_DRN:  if (!v1 && !v2) st <= S_DEC;
        S_DEC:  st <= last_bit ? S_FRD : S_PASS;
        S_FRD:  st <= S_FEV;
        S_FEV:  st <= S_FEM;
        S_FEM:  if (!(|pend)) st <= (NR_W'(frow) + 1'b1 == nrows) ? S_IDLE : S_FRD;
        default: st <= S_IDLE;
      endcase
    end
  end

  // ---------------- datapath registers (no reset)
  logic [T_W-1:0] ncand;
  always_comb ncand = tp - T_W'(w_eff_of(k));

  function automatic logic [6:0] w_eff_of(input logic [T_W-1:0] kk);
    w_eff_of = ((kk >> 1) > T_W'(64)) ? 7'd64 : (((kk >> 1) == '0) ? 7'd1 : 7'(kk >> 1));
  endfunction

  always_ff @(posedge clk) begin
    sh_q   <= sh;
    cand_q <= valid & agew;
    zq     <= z_now;
    case (st)
      S_IDLE: begin
        w_q      <= w_eff_of(k);
        r        <= (k == '0) ? ncand : tp - k;          // D (K = 0: all candidates)
        nrows    <= NR_W'((ncand + T_W'(P - 1)) >> LV);
        pfx      <= '0;
        mhi      <= raw ? {{(CMP_W-S_W){1'b1}}, {S_W{1'b0}}} : '0;
        rcnt     <= '0;
        c0       <= '0;
        a0       <= IDX_W'(tp - 1'b1);
        eq_seen  <= '0;
        frow     <= '0;
        last_bit <= 1'b0;
      end
      S_PASS: if (re) begin
        rcnt <= rcnt + 1'b1;
        a0   <= a0 - IDX_W'(P);
      end
      S_DEC: begin
        if (r > c0) begin
          pfx <= pfx | bsel;
          r   <= r - c0;
        end
        mhi      <= mhi | bsel;
        last_bit <= bsel[1];                             // next decision is bit 0
        rcnt     <= '0;
        c0       <= '0;
        a0       <= IDX_W'(tp - 1'b1);
      end
      S_FEV: begin
        eqq   <= eq_now;
        pend  <= lt_now | ((eq_seen < r) ? eq_now : '0);
        dmask <= '0;
      end
      S_FEM: begin
        if (|pend) begin
          if (!lane_take) pend[lane] <= 1'b0;
          else if (can_push) begin
            pend[lane]  <= 1'b0;
            dmask[lane] <= 1'b1;
            if (lane_eq) eq_seen <= eq_seen + 1'b1;
          end
        end else begin
          frow <= frow + 1'b1;
          a0   <= a0 - IDX_W'(P);
        end
      end
      default: ;
    endcase
    if (v2) c0 <= c0 + T_W'(popcnt(zq));
  end

  /* verilator lint_off UNUSEDSIGNAL */
  wire unused = |isnew;
  /* verilator lint_on UNUSEDSIGNAL */
endmodule
