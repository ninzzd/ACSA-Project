// ipu_seq_fsm -- packet control: header check, prefill residency / drain, decode fill
// and prefetch, sweep beat issue, victim decision, flush (spec v0.2 Section 6).
module ipu_seq_fsm #(
  parameter int P      = 16,
  parameter int T_MAX  = 4096,
  parameter int B_W    = 3,
  parameter int L_W    = 5,
  parameter int H_W    = 5,
  parameter int NSEQ   = 8
) (
  input  logic                 clk,
  input  logic                 rst_n,
  // configuration
  input  logic                 enable,
  input  logic                 raw,
  input  logic                 flush_req,
  input  logic [4:0]           l_max,
  input  logic [4:0]           h_max,
  input  logic [2:0]           b_max,
  input  logic [NSEQ*($clog2(T_MAX)+1)-1:0] k_flat,
  // attention stream (control part)
  input  logic                 s_tvalid,
  input  logic                 s_tlast,
  input  logic [2+B_W+L_W+H_W+$clog2(T_MAX):0] s_tuser,
  output logic                 s_tready,
  // pipeline
  output logic                 acc,
  output logic [$clog2(T_MAX)-1:0] beat_a0,
  output logic [$clog2(T_MAX/P)-1:0] beat_row,
  output logic                 beat_first,
  output logic                 beat_last,
  output logic                 beat_dec,
  output logic [6:0]           w_eff,
  input  logic                 stall,
  input  logic                 s123_busy,
  input  logic                 done,
  input  logic                 found,
  input  logic [41:0]          champ_cmp,
  input  logic [$clog2(T_MAX)-1:0] champ_idx,
  // score bank read port (sweep uses acc + beat_row; drain uses these)
  output logic                 dr_re,
  output logic [$clog2(T_MAX/P)-1:0] dr_raddr,
  output logic                 dr_push,      // bank rdata -> W buffer this cycle
  // read engine
  output logic                 rd_load,
  output logic [B_W+L_W+H_W-1:0] rd_hid,
  output logic [$clog2(T_MAX/P):0] rd_target,
  output logic                 rd_cur,
  output logic                 rd_extend,
  output logic                 rd_set_cur,
  output logic                 rd_stop,
  output logic                 rd_inval,
  output logic                 sweep_active,
  output logic [$clog2(T_MAX/P):0] sweep_rows,
  input  logic                 rdj_valid,
  input  logic                 rdj_cur,
  input  logic [B_W+L_W+H_W-1:0] rdj_hid,
  input  logic [$clog2(T_MAX/P):0] rdj_target,
  input  logic [$clog2(T_MAX/P):0] rdj_bank,
  input  logic                 rd_idle,
  // write engine / W buffer
  output logic                 wr_start_rows,
  output logic                 wr_start_vic,
  output logic [B_W+L_W+H_W-1:0] wr_hid,
  output logic [$clog2(T_MAX/P):0] wr_n,
  output logic [$clog2(T_MAX/P)-1:0] wr_row,
  input  logic                 wr_job_idle,
  input  logic                 wr_all_idle,
  input  logic                 wr_prev_idle,
  input  logic                 wb_ready,
  output logic                 vic_push,
  output logic [31:0]          vic_word,
  output logic [$clog2(P)-1:0] vic_lane,
  // demote issue
  input  logic                 iss_can_push,
  output logic                 iss_push,
  output logic [B_W+L_W+H_W+$clog2(T_MAX)-1:0] iss_data,
  // status / events
  output logic                 busy,
  output logic                 flush_pending,
  output logic                 ev_sweep,
  output logic                 ev_victim,
  output logic                 ev_stall,
  output logic                 ev_mispred,
  output logic                 err_no_elig,
  output logic                 err_t_range,
  output logic                 err_seq
);
  `include "ipu_defs.svh"
  localparam int IDX_W = $clog2(T_MAX);
  localparam int T_W   = IDX_W + 1;
  localparam int ROW_W = $clog2(T_MAX / P);
  localparam int NR_W  = ROW_W + 1;
  localparam int LV    = $clog2(P);
  localparam int HID_W = B_W + L_W + H_W;

  typedef enum logic [3:0] {
    ST_IDLE, ST_CHECK, ST_DRAIN, ST_SWEEP, ST_PFWAIT, ST_DCWAIT,
    ST_V1, ST_V2, ST_DECIDE, ST_FLUSH, ST_DROP
  } st_t;
  st_t st, dr_ret;

  // ---------------- header / packet context
  logic [1:0]       ph;
  logic [B_W-1:0]   hb;
  logic [L_W-1:0]   hl;
  logic [H_W-1:0]   hh;
  logic [T_W-1:0]   ht;
  logic [NR_W-1:0]  nbeats, nfill;
  logic [T_W-1:0]   kcur;
  logic [NR_W-1:0]  beat_cnt;
  logic [IDX_W-1:0] a0;
  logic             fill_ok, pf_issued;

  wire [HID_W-1:0]  hid = {hb, hl, hh};
  wire              is_dec = (ph == PH_DECODE);

  // ---------------- bank residency (prefill)
  logic             bank_dirty;
  logic [HID_W-1:0] bank_hid;
  logic [T_W-1:0]   bank_t;

  // ---------------- drain
  logic             dr_started, dr_inflight;
  logic [NR_W-1:0]  dr_cnt, dr_pushed, dr_n;

  // ---------------- victim
  logic             v_found;
  logic [41:0]      v_cmp;
  logic [IDX_W-1:0] v_idx, v_age;
  logic [29:0]      v_s;

  // ---------------- per-sequence last decode t (prefetch prediction)
  logic [T_W-1:0]   t_last [0:NSEQ-1];

  function automatic logic [NR_W-1:0] ceil_rows(input logic [T_W-1:0] x);
    ceil_rows = NR_W'((x + T_W'(P - 1)) >> LV);
  endfunction

  function automatic logic [3:0] sh_of(input logic [IDX_W-1:0] age);
    sh_of = 4'd0;
    for (int j = 0; j < NTH; j++)
      if ({1'b0, age} >= T_W'(sh_thresh(j))) sh_of = 4'(j + 1);
  endfunction

  // ---------------- header fields from tuser (sampled in IDLE)
  logic [1:0]     u_ph;
  logic [B_W-1:0] u_b;
  logic [L_W-1:0] u_l;
  logic [H_W-1:0] u_h;
  logic [T_W-1:0] u_t;
  assign {u_ph, u_b, u_l, u_h, u_t} = s_tuser;

  // ---------------- prefetch prediction (successor in b -> l -> h order)
  logic [B_W-1:0] sb;
  logic [L_W-1:0] sl;
  logic [H_W-1:0] sh_n;
  logic           s_start;
  logic [T_W-1:0] tp;
  logic           tp_ok;
  always_comb begin
    sb = hb; sl = hl; sh_n = hh + 1'b1;
    if (hh == H_W'(h_max)) begin
      sh_n = '0; sl = hl + 1'b1;
      if (hl == L_W'(l_max)) begin
        sl = '0; sb = (hb == B_W'(b_max)) ? '0 : hb + 1'b1;
      end
    end
    s_start = (sl == '0) && (sh_n == '0);
    tp      = s_start ? t_last[sb] + 1'b1 : ht;
    tp_ok   = (!s_start || (t_last[sb] != '0)) && (tp <= T_W'(T_MAX)) && (tp > T_W'(1));
  end

  // prediction is registered (off the critical path); valid from the 2nd cycle after CHECK
  logic [HID_W-1:0] pf_hid_q;
  logic [NR_W-1:0]  pf_target_q;
  logic             pf_ok_q, pf_arm;
  always_ff @(posedge clk) begin
    pf_hid_q    <= {sb, sl, sh_n};
    pf_target_q <= NR_W'((tp - 1'b1 + T_W'(P - 1)) >> LV);
    pf_ok_q     <= tp_ok && ({sb, sl, sh_n} != hid);
  end

  // ---------------- derived per-state signals
  logic hdr_bad, pf_cont, rd_match, row_ok, fire_ok;
  assign hdr_bad  = (ht == '0) || (ht > T_W'(T_MAX)) || !(ph == PH_PREFILL || ph == PH_DECODE);
  assign pf_cont  = bank_dirty && (bank_hid == hid) && (ht > T_W'(1));
  // only an unconsumed prefetch may be reused (a consumed job holds pre-update rows)
  assign rd_match = rdj_valid && !rdj_cur && (rdj_hid == hid);
  assign row_ok   = !is_dec || (beat_cnt >= nfill) || fill_ok || (beat_cnt < rdj_bank);

  assign s_tready = (st == ST_SWEEP && !stall && row_ok) || (st == ST_DROP);
  assign acc      = (st == ST_SWEEP) && s_tvalid && s_tready;

  assign beat_a0    = a0;
  assign beat_row   = beat_cnt[ROW_W-1:0];
  assign beat_first = (beat_cnt == '0);
  assign beat_last  = (beat_cnt == nbeats - 1'b1);
  assign beat_dec   = is_dec;

  assign sweep_active = (st == ST_SWEEP) && is_dec;
  assign sweep_rows   = beat_cnt;

  assign fire_ok  = iss_can_push && wr_job_idle && wb_ready;

  assign busy     = (st != ST_IDLE) || flush_pending || !wr_all_idle;
  assign ev_stall = s_tvalid && !s_tready && (st != ST_IDLE);

  // drain read: one row in flight, issued only while the W buffer has room
  assign dr_re    = (st == ST_DRAIN) && dr_started && !dr_inflight && (dr_cnt < dr_n) && wb_ready;
  assign dr_raddr = dr_cnt[ROW_W-1:0];
  assign dr_push  = dr_inflight;

  // victim word replicated into every lane; WSTRB selects the lane
  assign vic_word = {TIER_LOW, v_s};
  assign vic_lane = v_idx[LV-1:0];
  assign iss_data = {hid, v_idx};

  // ---------------- combinational command outputs
  always_comb begin
    rd_load = 1'b0; rd_hid = hid; rd_target = nfill; rd_cur = 1'b1;
    rd_extend = 1'b0; rd_set_cur = 1'b0; rd_stop = 1'b0; rd_inval = 1'b0;
    wr_start_rows = 1'b0; wr_start_vic = 1'b0; wr_hid = hid; wr_n = nbeats;
    wr_row = v_idx[IDX_W-1:LV];
    vic_push = 1'b0; iss_push = 1'b0;
    ev_sweep = 1'b0; ev_victim = 1'b0; ev_mispred = 1'b0;
    err_no_elig = 1'b0; err_t_range = 1'b0; err_seq = 1'b0;

    case (st)
      ST_CHECK: begin
        if (hdr_bad) begin
          err_t_range = 1'b1;
        end else if (!is_dec) begin
          if (!(bank_dirty && !pf_cont)) begin
            if (!rd_idle) rd_stop = 1'b1;
            else begin
              rd_inval = 1'b1;
              ev_sweep = 1'b1;
              err_seq  = pf_cont ? (ht != bank_t + 1'b1) : (ht != T_W'(1));
            end
          end
        end else if (!bank_dirty && wr_job_idle) begin
          if (rd_match) begin
            rd_set_cur = 1'b1;
            if (rdj_target < nfill) rd_extend = 1'b1;
            wr_start_rows = 1'b1;
            ev_sweep = 1'b1;
          end else if (!rd_idle) begin
            rd_stop = 1'b1;
          end else if (wr_all_idle) begin           // fill only after all writes landed
            rd_load = 1'b1;
            ev_mispred = rdj_valid && !rdj_cur;     // an unconsumed prefetch was wrong
            wr_start_rows = 1'b1;
            ev_sweep = 1'b1;
          end
        end
      end
      ST_DRAIN: begin
        if (!dr_started && wr_job_idle) begin
          wr_start_rows = 1'b1;
          wr_hid = bank_hid;
          wr_n   = ceil_rows(bank_t);
        end
      end
      ST_DECIDE: begin
        if (ht > kcur) begin
          if (!v_found) err_no_elig = 1'b1;
          else if (fire_ok) begin
            iss_push = 1'b1; wr_start_vic = 1'b1; vic_push = 1'b1; ev_victim = 1'b1;
          end
        end
      end
      ST_FLUSH: begin
        if (!bank_dirty) begin
          if (!rd_idle) rd_stop = 1'b1;
          else if (wr_all_idle) rd_inval = 1'b1;
        end
      end
      default: ;
    endcase

    // prefetch the predicted next head once this head's fill is complete
    // (never the head being written now; only once earlier writes are acknowledged)
    if (is_dec && fill_ok && !pf_issued && rd_idle && pf_ok_q && pf_arm && wr_prev_idle &&
        (st == ST_SWEEP || st == ST_DCWAIT || st == ST_V1 || st == ST_V2 || st == ST_DECIDE)) begin
      rd_load   = 1'b1;
      rd_hid    = pf_hid_q;
      rd_target = pf_target_q;
      rd_cur    = 1'b0;
    end
  end

  // ---------------- state register
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= ST_IDLE; dr_ret <= ST_IDLE;
      bank_dirty <= 1'b0; dr_started <= 1'b0; dr_inflight <= 1'b0;
      flush_pending <= 1'b0; pf_issued <= 1'b0; fill_ok <= 1'b0; pf_arm <= 1'b0;
      for (int b = 0; b < NSEQ; b++) t_last[b] <= '0;
    end else begin
      if (flush_req) flush_pending <= 1'b1;
      if (is_dec && rdj_valid && rdj_cur && (rdj_hid == hid) && (rdj_bank >= nfill)) fill_ok <= 1'b1;
      if (rd_load && !rd_cur) pf_issued <= 1'b1;
      pf_arm <= (st != ST_CHECK);

      case (st)
        ST_IDLE: begin
          if (flush_pending)               st <= ST_FLUSH;
          else if (enable && s_tvalid)     st <= ST_CHECK;
        end
        ST_CHECK: begin
          if (hdr_bad) st <= ST_DROP;
          else if (!is_dec) begin
            if (bank_dirty && !pf_cont) begin
              dr_ret <= ST_CHECK; st <= ST_DRAIN;
            end else if (rd_idle) begin
              bank_dirty <= 1'b1; bank_hid <= hid; bank_t <= ht;
              st <= ST_SWEEP;
            end
          end else if (bank_dirty) begin
            dr_ret <= ST_CHECK; st <= ST_DRAIN;
          end else if (wr_job_idle && (rd_match || (rd_idle && wr_all_idle))) begin
            t_last[hb] <= ht;
            pf_issued  <= 1'b0;
            fill_ok    <= (nfill == '0);
            st <= ST_SWEEP;
          end
        end
        ST_SWEEP: begin
          if (acc && beat_last) st <= is_dec ? ST_DCWAIT : ST_PFWAIT;
        end
        ST_PFWAIT: if (!s123_busy) st <= ST_IDLE;
        ST_DCWAIT: if (done) st <= ST_V1;
        ST_V1:     st <= ST_V2;
        ST_V2:     st <= ST_DECIDE;
        ST_DECIDE: if (!(ht > kcur) || !v_found || fire_ok) st <= ST_IDLE;
        ST_DRAIN: begin
          if (!dr_started) begin
            if (wr_job_idle) begin
              dr_started <= 1'b1; dr_n <= ceil_rows(bank_t);
            end
          end else if (dr_pushed == dr_n) begin
            dr_started <= 1'b0; bank_dirty <= 1'b0; st <= dr_ret;
          end
          dr_inflight <= dr_re;
        end
        ST_FLUSH: begin
          if (bank_dirty) begin
            dr_ret <= ST_FLUSH; st <= ST_DRAIN;
          end else if (rd_idle && wr_all_idle) begin
            flush_pending <= 1'b0; st <= ST_IDLE;
          end
        end
        ST_DROP: if (s_tvalid && s_tlast) st <= ST_IDLE;
        default: st <= ST_IDLE;
      endcase
    end
  end

  // ---------------- datapath-ish registers (no reset)
  always_ff @(posedge clk) begin
    if (st == ST_IDLE) begin
      ph <= u_ph; hb <= u_b; hl <= u_l; hh <= u_h; ht <= u_t;
      nbeats <= ceil_rows(u_t);
      nfill  <= ceil_rows(u_t - 1'b1);
      kcur   <= k_flat[T_W*u_b +: T_W];
      w_eff  <= 7'd1;
    end
    if (st == ST_CHECK) begin
      w_eff    <= ((kcur >> 1) > T_W'(64)) ? 7'd64 : (((kcur >> 1) == '0) ? 7'd1 : 7'(kcur >> 1));
      beat_cnt <= '0;
      a0       <= IDX_W'(ht - 1'b1);
    end else if (acc) begin
      beat_cnt <= beat_cnt + 1'b1;
      a0       <= a0 - IDX_W'(P);
    end
    // drain counters
    if (st == ST_DRAIN && !dr_started) begin
      dr_cnt <= '0; dr_pushed <= '0;
    end else begin
      if (dr_re)   dr_cnt    <= dr_cnt + 1'b1;
      if (dr_push) dr_pushed <= dr_pushed + 1'b1;
    end
    // victim
    if (st == ST_DCWAIT && done) begin
      v_found <= found; v_cmp <= champ_cmp; v_idx <= champ_idx;
    end
    if (st == ST_V1) v_age <= IDX_W'(ht - 1'b1) - v_idx;
    if (st == ST_V2) v_s   <= raw ? v_cmp[29:0] : 30'(v_cmp >> (4'd12 - sh_of(v_age)));
  end
endmodule
