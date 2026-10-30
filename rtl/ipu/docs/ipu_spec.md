# IPU (Importance Policy Unit) — RTL specification v0.2

Status: frozen for RTL v0.1 (30 Oct 2026). Supersedes spec v0.1.
Decisions come from the team's answers to *IPU: open questions before RTL* (v1.0).
Section F (channel balancer) is out of scope here: the balancer is not part of the IPU.

Design priorities, in order: **bit-exact policy**, **minimum area**, **maximum fmax**,
then throughput.

---

## 1. What the IPU does

The IPU decides which KV-cache tokens leave the HIGH-precision tier. It keeps one
32-bit score word per (sequence b, layer l, head h, token idx) in DRAM, updates the
scores from the attention probabilities, and demotes at most one token per head per
decode step. It never touches K/V data.

| Phase | Behaviour |
|---|---|
| Prefill | The attention engine streams every prompt row of one head, row i = i+1 probabilities, rows in order, **all rows of a head before the next head**. The IPU accumulates `S[j] += delta(p[i][j])` in its on-chip bank. No decisions. |
| End of prefill | Host issues FLUSH, reads the scores, demotes t_p − k tokens by writing tier = LOW into their DRAM words, and sends that list to the migration engine (v0.1: the testbench does this). |
| Decode | One sweep per head per step: `S += delta` for HIGH tokens, new token initialised, decayed compare, argmin over eligible tokens; if t > K, the victim's tier becomes LOW and a 25-bit packet goes to the migration engine. |

## 2. Policy (bit-exact definition; `model/ipu_golden.py` is the executable form)

Notation: `t` = number of tokens including the token generated this step (decode) or
the row length (prefill row i has t = i+1). `idx` in 0…t−1, 0 = oldest.
`age = (t−1) − idx`.

1. **Ingress.** `delta = clamp(RNE(p · 2^16), 0, 2^17−1)`, p is fp16. Unsigned Q1.16.
   Round half to even. NaN, ±Inf and negative non-zero values give delta = 0 and set
   `ERR_BAD_PROB`. −0 gives 0 without error. Subnormals are converted exactly.
2. **Accumulate.** Word = {tier[1:0], S[29:0]}, S unsigned Q14.16.
   - `idx == t−1` (new token): tier ← HIGH, S ← delta.
   - tier == HIGH: S ← min(S + delta, 2^30−1); clamping sets `ERR_SAT`.
   - any other tier (LOW, reserved, EMPTY): S unchanged (**LOW scores frozen**, A6).
3. **Tier codes.** 00 EMPTY, 01 HIGH, 10 reserved, 11 LOW.
4. **Window.** `W_EFF = min(64, max(1, ⌊K/2⌋))`, computed in hardware from K (A4).
5. **Eligible** (decode only): tier == HIGH and idx ≠ t−1 and age ≥ W_EFF.
6. **Decay compare** (pow2): `sh = round(log2(age))` = number of thresholds
   {2, 3, 6, 12, 23, 46, 91, 182, 363, 725, 1449, 2897} that age is ≥ to (0…12).
   `cmp = S << (12 − sh)`, 42 bits, exact. RAW mode: `cmp = S`.
7. **Victim.** Smallest cmp among eligible tokens; ties → lowest idx (oldest).
8. **Fire.** Decode and `t > K[b]` and a victim exists → victim word ← {LOW, S_victim},
   packet {b, l, h, idx} issued. `t > K` with no eligible token sets `ERR_NO_ELIG`.
9. **Host bulk selection at the end of prefill** (not in RTL v0.1): with t = t_p and the
   same decay, keep the W_EFF newest tokens plus the k_H = K − W_EFF largest cmp among
   the rest (ties: keep the newer token); write LOW into all others. If t_p ≤ K nothing
   is demoted.

Known deviation from the PyTorch model (accepted, A1/A2/A6): Python can demote 0, 1 or
several tokens per step, lets LOW tokens compete for top-k, leaves topk tie order
unspecified, keeps accumulating LOW scores, and evaluates the pow2 product in fp32
(rounding scores ≥ 256 at fp32 precision). The RTL implements rules 1–8 exactly.
Accuracy of these rules is validated separately (Section 5 check 1 of the questions
document).

## 3. Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `P` | 16 | lanes = tokens per beat; AXI data width = 32·P |
| `T_MAX` | 4096 | max tokens per sequence (bank rows = T_MAX/P) |
| `B_W, L_W, H_W` | 3, 5, 5 | sequence / layer / head id widths (Llama-2-7B: 32 × 32) |
| `NSEQ` | 8 | number of sequences with a K register |
| `ADDR_W` | 40 | AXI address width |
| `ISSUE_DEPTH` | 4 | demote FIFO depth |
| `HAS_PERF` | 1 | performance counters |

Derived: `IDX_W = log2(T_MAX)`, `T_W = IDX_W+1`, `CMP_W = 42`, `ROW_W = log2(T_MAX/P)`.
LOW/HIGH quantization formats (int2/int4 now) live in the migration engine; the IPU and
its packet format do not depend on them, so a v2 with other formats needs no IPU change.

## 4. Micro-architecture

### 4.1 Single score bank (area decision)

One bank of T_MAX/P rows × 32·P bits (16 KB at defaults), 1 read + 1 write port.

- **Prefill:** the bank holds the current head's scores; each row-packet reads and
  writes the bank. When a packet for a different head (or a decode packet, or FLUSH)
  arrives, the bank is drained to DRAM (rows 0…⌈t/P⌉−1). Prefill never reads DRAM:
  every word is created (as the new token) before it is used.
- **Decode:** the bank is only a prefetch buffer. The read engine fills rows
  0…⌈(t−1)/P⌉−1 (words of the old tokens) from DRAM; the sweep reads them; updated rows
  go **straight to DRAM** (rows 0…⌈t/P⌉−1, lanes ≥ t written as 0). The next head is
  prefetched into the same bank behind the sweep (a row is overwritten only after the
  current sweep has read it).
- **Victim write:** a single-beat AXI write with WSTRB covering the victim's 4 bytes.
  S_victim is reconstructed exactly: `S = cmp >> (12 − sh(age))` (RAW: `S = cmp`), so no
  bank read is needed.

Compared with the three-bank buffer of v0.1 this removes 32 KB of SRAM with no loss of
decode throughput (DRAM read and write streams still overlap the sweep).

### 4.2 Pipeline (one beat = P tokens per cycle)

| Stage | Work |
|---|---|
| S1 | beat registered; bank read issued (row = beat number) |
| S2 | delta (fp16 → Q1.16), bank row registered, per-lane valid / new / age ≥ W_EFF, sh |
| S3 | tier logic, saturating add, eligibility; word written to bank (prefill) or W buffer (decode) |
| T1 | `cmp = S << (12 − sh)` (ineligible lanes forced to all-ones) + first comparator level (8) |
| T2…T(log2 P) | one comparator level per stage (4, 2, 1 comparators) |
| CH | running champion (update only on strictly smaller) |
| DECIDE | fire test, victim word write, demote packet |

- An ineligible lane is encoded as cmp = all-ones; an eligible cmp is at most
  2^42 − 2^12, so no separate valid bit is carried through the tree.
- Comparator: left (lower lane) wins on equal. Champion: replaced only on strictly less.
- S1–S3 stall together when the W buffer is full (registered ready). T-stages never stall.
- The shift is folded into T1 (shift + one compare ≈ 2 ns, inside the 2.75 ns limit set by
  the ingress stage), which removed a 672-flop register stage.
- Between prefill packets of the same head the FSM waits for S1–S3 to empty (bank
  read-after-write). Between decode packets it waits for DECIDE (≈ 10 cycles).
- Per-beat shared logic: `a0 = t−1−P·beat` (age of lane 0). Lane i: age = a0 − i;
  `age ≥ X ⇔ (a0 − X) ≥ i`, so each threshold costs one per-beat subtractor plus a
  log2(P)-bit compare per lane.

### 4.3 Modules

| Module | Function |
|---|---|
| `ipu_defs.svh` | constants (tier/phase codes, widths) and the sh threshold function |
| `ipu_ingress` | P × fp16 → Q1.16 converters, bad-input flags |
| `ipu_lane_ctl` | per-beat a0 logic → per-lane valid, new, age ≥ W_EFF, sh |
| `ipu_accum` | P × tier logic + saturating add, LOW freeze, invalid lanes → 0 |
| `ipu_cmp_shift` | P × `S << (12−sh)` / RAW, ineligible → all-ones |
| `ipu_min_tree` | pipelined argmin tree, one level per stage |
| `ipu_min_track` | running champion over the sweep |
| `ipu_pipe` | S1–S3 registers, stall, tree + champion |
| `ipu_fifo` | FIFO with registered flags (W buffer: depth 2; demote FIFO) |
| `ipu_rd_eng` | AXI read engine: fill jobs, prefetch, chase rule, R register |
| `ipu_wr_eng` | AXI write engine: row jobs (decode / drain), victim beat, B tracking |
| `ipu_issue` | demote FIFO + credit counter |
| `ipu_csr` | AXI4-Lite registers |
| `ipu_seq_fsm` | packet control, prediction, drain/flush, victim reconstruction, decide |
| `ipu_sram_1r1w` | behavioural 1R1W SRAM (replace with a macro wrapper later) |
| `ipu_top` | integration, bank port muxes, WSTRB generation |

## 5. Interfaces

### 5.1 Attention stream (slave, AXI4-Stream)

| Signal | Width | Meaning |
|---|---|---|
| `s_attn_tdata` | 16·P | lane i = bits [16i+15:16i] = fp16 p for idx = P·beat + i |
| `s_attn_tkeep` | P | lane valid (ANDed with idx < t) |
| `s_attn_tuser` | 2+B_W+L_W+H_W+T_W | {phase, b, l, h, t}, sampled on the first beat; phase 01 = PREFILL, 10 = DECODE |
| `s_attn_tlast` | 1 | informational; the IPU counts ⌈t/P⌉ beats |
| `s_attn_tvalid / tready` | 1 | tready low while the IPU must wait (fill not ready, drain, W buffer full, decide) |

Ordering requirements: prefill rows of a head consecutive and in order (t = 1, 2, …);
decode heads in the order b → l → h (any order works, but only this order is prefetched).

### 5.2 Demote packets / acks (migration engine)

`m_dem_valid/ready`, `m_dem_data[B_W+L_W+H_W+IDX_W-1:0] = {b, l, h, idx}` (25 b);
`s_ack_valid` (1 pulse per completed migration). At most `CREDIT_LIMIT` (CSR, default 8)
packets un-acked. The migration engine never writes score words. It also stores each
new token's K/V (int4 HIGH) — no IPU involvement.

### 5.3 Memory (AXI4 master)

Data 32·P bits, address `ADDR_W`, INCR bursts ≤ 16 beats, one ID (0), all transfers
row-aligned. Row address: `{SCORE_BASE[ADDR_W-1:27], b, l, h, row, 6'b0}` (P = 16) —
SCORE_BASE must be 128 MiB aligned (no adder in the address path).
Reads: fill jobs. Writes: decode rows, prefill drain rows, victim beats (partial WSTRB).

### 5.4 CSR (AXI4-Lite slave, 32-bit, 8-bit address)

| Offset | Name | Fields |
|---|---|---|
| 0x00 | CTRL | [0] ENABLE, [1] RAW (0 = pow2 decay), [2] IRQ_EN, [8] FLUSH (write 1, self-clearing) |
| 0x04 | STATUS | [0] BUSY (RO), [1] FLUSH_PENDING (RO); W1C sticky: [8] BAD_PROB, [9] SAT, [10] NO_ELIG, [11] T_RANGE, [12] SEQ, [13] AXI |
| 0x08 | BASE | SCORE_BASE[ADDR_W-1:27] in bits [ADDR_W-28:0] |
| 0x0C | SHAPE | [4:0] L_MAX, [9:5] H_MAX, [12:10] B_MAX (last index; used for prefetch only) |
| 0x10 | CREDIT | [4:0] credit limit (default 8) |
| 0x14 | ID | version 0x0002, P, log2 T_MAX (RO) |
| 0x20+4b | K[b] | [T_W-1:0] budget K for sequence b; W_EFF readable at [31:16] |
| 0x60 | SWEEPS | packets processed (RO) |
| 0x64 | VICTIMS | demote packets issued (RO) |
| 0x68 | STALLS | cycles with tvalid && !tready (RO) |
| 0x6C | MISPRED | prefetch misses (RO) |

`irq = IRQ_EN && (any sticky error)`.

## 6. Control flow (ipu_seq_fsm)

```
IDLE --header--> CHECK
CHECK: if bank holds dirty prefill data that this packet cannot extend → DRAIN
       prefill: stop/abort any fill job → SWEEP
       decode : make sure the fill job is this head (else abort + refill) → SWEEP
SWEEP: accept ⌈t/P⌉ beats (decode beat r waits until row r is filled)
       → PF_WAIT (prefill: S1–S3 empty) → IDLE
       → DC_WAIT (decode: pipeline empty, rows pushed) → DECIDE → IDLE
FLUSH (CSR) from IDLE: drain if dirty, wait all writes acknowledged.
```

Prefetch: once the current decode head's fill is complete, the read engine loads the
predicted next head: successor of (b, l, h) under SHAPE; t_pred = t, or
T_LAST[b'] + 1 when the successor starts a sequence (T_LAST[b] = last decode t of b).

## 7. Edge cases (each has a test)

| Case | Behaviour |
|---|---|
| t ≤ K | no demotion |
| t not a multiple of P | partial last beat; lanes ≥ t written as 0 |
| (t−1) mod P = 0 | new token alone in a new row; that row is not fetched |
| first decode step / sequence switch | prefetch miss or no prediction → fill on demand |
| out-of-order head | prefetch abort, refill, MISPRED++ |
| all eligible scores equal | oldest demoted |
| K = 1 | W_EFF = 1, k_H = 0 |
| W buffer full, migration FIFO full, no credits | stall, nothing lost |
| NaN / Inf / negative input | delta 0, BAD_PROB |
| t = 0 or t > T_MAX | packet ignored, T_RANGE |
| prefill row out of order | SEQ flag, processed anyway |
| saturation | clamp, SAT |

## 8. Verification

- `model/ipu_golden.py`: pure-integer reference of Section 2 plus the memory effects of
  Section 4.1 (exact DRAM image).
- `model/gen_vectors.py`: unit vectors (ingress exhaustive, cmp/sh, min tree) and top
  scenarios (prefill → flush → host selection → decode, random DRAM init to prove no
  clearing is needed, random AXI back-pressure and latency).
- `tb/`: self-checking SystemVerilog testbenches, run on Vivado xsim (team), Verilator
  and Icarus (CI here).
- Pass: bit-exact DRAM image and identical victim sequence; Verilator `-Wall` clean;
  Yosys synthesis.

## 9. Implementation results (RTL v0.1, 30 Oct 2026)

Defaults P = 16, T_MAX = 4096, 8 sequences. Score SRAM black-boxed (macro TBD), DRAM off-chip.

| Item | Result |
|---|---|
| Sky130 HD area (Yosys + ABC, typical corner, logic only) | 0.272 mm², ≈ 5,080 flip-flops |
| Largest blocks | pipeline registers 58k µm², W buffer 39k, argmin tree 35k, cmp shift 26k, FSM 20k, read engine 19k, accumulate 19k |
| Critical path estimate (ABC `stime`, typical, no wire load) | 2.75 ns (fp16 → Q1.16 ingress, S1→S2) ≈ 360 MHz pre-layout |
| Verification | ingress exhaustive (65,536), 4,000 random beats, 3,000 argmin packets, 3 top scenarios bit-exact on Verilator 5.020 and Icarus 12 |
| Lint | Verilator `-Wall` clean |

Area/fmax decisions taken while writing the RTL: single score bank (−32 KB SRAM); ineligible
encoded as all-ones (no valid bit in the tree); shift folded into the first tree stage
(−672 flops); per-beat shared age logic; prediction registered (removed a 3.5 ns path);
no reset on datapath registers; operand isolation for frozen LOW lanes.

## 10. Not in v0.1

In-IPU prefill selection, overlap of DECIDE with the next head, SRAM macros (to be
discussed), clock gating, grouped-query attention, chunked prefill.
