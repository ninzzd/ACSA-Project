// ipu_pipe -- per-beat datapath pipeline S1..S4 + argmin tree + champion.
//
//   S1  beat registered (bank read issued by the FSM in the same cycle)
//   S2  delta, lane control and bank row registered
//   S3  updated words registered -> bank write (prefill) or W buffer push (decode)
//   T1  decayed compare (S << (12 - sh)) + first comparator level (decode beats only;
//       the shift is folded into this stage: shift + compare fits inside the S3 budget)
//   T2..T(log2 P), CH  rest of the argmin tree and running champion
//
// S1..S3 stall together only when S3 holds a decode beat and the W buffer is full
// (wb_ready is a register output, so the stall is a function of flops only).
module ipu_pipe #(
  parameter int P     = 16,
  parameter int IDX_W = 12,
  parameter int ROW_W = 8
) (
  input  logic               clk,
  input  logic               rst_n,
  // beat in (from the FSM, qualified by acc)
  input  logic               acc,
  input  logic [16*P-1:0]    fp16,
  input  logic [P-1:0]       keep,
  input  logic [IDX_W-1:0]   a0,
  input  logic [ROW_W-1:0]   row,
  input  logic               first,
  input  logic               last,
  input  logic               dec,
  input  logic [6:0]         w_eff,
  input  logic               raw,
  // score bank read data (valid the cycle after acc)
  input  logic [32*P-1:0]    bank_rdata,
  // W buffer
  input  logic               wb_ready,
  output logic               wb_push,
  output logic [32*P-1:0]    s3_word,
  // bank write (prefill)
  output logic               bank_we,
  output logic [ROW_W-1:0]   bank_waddr,
  // status
  output logic               stall,
  output logic               s123_busy,
  output logic               err_bad,
  output logic               err_sat,
  // champion
  output logic               done,
  output logic               found,
  output logic [41:0]        champ_cmp,
  output logic [ROW_W+$clog2(P)-1:0] champ_idx
);
  localparam int LV = $clog2(P);

  // ---------------- S1
  logic             v1, first1, last1, dec1;
  logic [16*P-1:0]  fp16_1;
  logic [P-1:0]     keep1;
  logic [IDX_W-1:0] a0_1;
  logic [ROW_W-1:0] row1;
  // ---------------- S2
  logic             v2, first2, last2, dec2;
  logic [17*P-1:0]  delta2;
  logic [P-1:0]     bad2, valid2, new2, agew2;
  logic [4*P-1:0]   sh2;
  logic [32*P-1:0]  word2;
  logic [ROW_W-1:0] row2;
  // ---------------- S3
  logic             v3, first3, last3, dec3;
  logic [32*P-1:0]  word3;
  logic [P-1:0]     elig3, sat3;
  logic [4*P-1:0]   sh3;
  logic [ROW_W-1:0] row3;

  logic             adv;                       // S1..S3 advance
  assign stall = v3 & dec3 & ~wb_ready;
  assign adv   = ~stall;

  // combinational stage logic
  logic [17*P-1:0] delta_c;
  logic [P-1:0]    bad_c, valid_c, new_c, agew_c, elig_c, sat_c;
  logic [4*P-1:0]  sh_c;
  logic [32*P-1:0] word_c;
  logic [42*P-1:0] cmp_c;

  ipu_ingress  #(.P(P)) u_ing (.fp16(fp16_1), .delta(delta_c), .bad(bad_c));
  ipu_lane_ctl #(.P(P), .IDX_W(IDX_W)) u_lc (.a0(a0_1), .w_eff(w_eff), .keep(keep1),
                 .valid(valid_c), .isnew(new_c), .agew(agew_c), .sh(sh_c));
  ipu_accum    #(.P(P)) u_acc (.word_in(word2), .delta(delta2), .valid(valid2), .isnew(new2),
                 .agew(agew2), .decode(dec2), .word_out(word_c), .elig(elig_c), .sat(sat_c));
  ipu_cmp_shift #(.P(P)) u_cs (.word(word3), .sh(sh3), .elig(elig3), .raw(raw), .cmp(cmp_c));

  // valid bits (reset)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0;
    end else begin
      if (adv) begin
        v1 <= acc;
        v2 <= v1;
        v3 <= v2;
      end
    end
  end

  // datapath registers (no reset)
  always_ff @(posedge clk) begin
    if (adv) begin
      if (acc) begin
        fp16_1 <= fp16;  keep1 <= keep;  a0_1 <= a0;  row1 <= row;
        first1 <= first; last1 <= last;  dec1 <= dec;
      end
      delta2 <= delta_c; bad2 <= bad_c; valid2 <= valid_c; new2 <= new_c; agew2 <= agew_c;
      sh2 <= sh_c; word2 <= bank_rdata; row2 <= row1;
      first2 <= first1; last2 <= last1; dec2 <= dec1;
      word3 <= word_c; elig3 <= elig_c; sat3 <= sat_c; sh3 <= sh2; row3 <= row2;
      first3 <= first2; last3 <= last2; dec3 <= dec2;
    end
  end

  assign s3_word    = word3;
  assign wb_push    = v3 & dec3 & wb_ready;
  assign bank_we    = v3 & ~dec3;
  assign bank_waddr = row3;
  assign s123_busy  = v1 | v2 | v3;
  assign err_bad    = v2 & adv & (|(bad2 & valid2));
  assign err_sat    = v3 & adv & (|sat3);

  // ---------------- argmin tree + champion
  logic              t_v, t_first, t_last;
  logic [ROW_W-1:0]  t_row;
  logic [41:0]       t_cmp;
  logic [LV-1:0]     t_lane;

  ipu_min_tree #(.P(P), .CW(42), .RW(ROW_W)) u_tree (
    .clk(clk), .rst_n(rst_n), .in_v(v3 & dec3 & adv), .in_first(first3), .in_last(last3), .in_row(row3),
    .in_cmp(cmp_c),          // only decode beats enter the tree
    .out_v(t_v), .out_first(t_first), .out_last(t_last), .out_row(t_row),
    .out_cmp(t_cmp), .out_lane(t_lane));

  ipu_min_track #(.P(P), .CW(42), .RW(ROW_W)) u_track (
    .clk(clk), .rst_n(rst_n), .in_v(t_v), .in_first(t_first), .in_last(t_last), .in_row(t_row),
    .in_cmp(t_cmp), .in_lane(t_lane), .done(done), .found(found), .champ_cmp(champ_cmp),
    .champ_idx(champ_idx));
endmodule
