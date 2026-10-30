# IPU v2 — addendum to spec v0.2 (end-of-prefill selection)

Status: RTL v2, 30 Oct 2026. v1 (`rtl/ipu/`) is unchanged; v2 lives in `rtl/ipu_v2/`.
Figures: [block diagram and area](figures/ipu_v2_blocks.svg), [test and compute flow](figures/ipu_v2_test_flow.svg).

## 1. What changes

In v1 the host made the bulk HIGH -> LOW decision at the end of prefill (FLUSH, then host
selection in DRAM). In v2 the IPU does it itself, per head, while the head is still
resident in the score bank, and sends the chosen token indices to the migration engine
on the existing demote-packet channel. The migration engine's own handling of a burst
is out of scope.

## 2. Policy (unchanged from v1 host selection, now bit-exact in RTL)

For a head that finishes prefill with t_p tokens and budget K (CSR `K[b]`):

- nothing happens if t_p <= K;
- W_EFF = min(64, max(1, K/2)); the newest W_EFF tokens stay HIGH;
- candidates: idx < t_p - W_EFF; compare value cmp = S << (12 - sh(age)), age = t_p-1-idx
  (RAW mode: cmp = S);
- demote D = t_p - K candidates (all candidates if K = 0): the D smallest cmp, ties ->
  older (lower idx) first. Equivalently, keep the K - W_EFF largest, ties keep newer.
  This is the same tie rule as the decode victim (oldest wins a tie).

Executable form: `select_lows()` in `model/ipu_golden.py`; `select_lows_radix()` is the
bit-serial form the RTL uses and is checked equal to it on 3,000 random heads.

## 3. When it runs

When the resident prefill head is about to be drained: a prefill packet of another head,
the first decode packet, or FLUSH. The FSM enters `ST_SEL` (only if CTRL.SEL_EN = 1)
before `ST_DRAIN`, so the DRAM image written by the drain already holds the LOW tiers.
Assumption (E1 = b): a head's prefill is complete before the next head starts.

## 4. `ipu_psel` — radix select

Per bit b (MSB first; 42 bits, 30 in RAW mode), one streaming pass over the candidate
rows (one 512-bit row per cycle from the bank):

    c0 = #{candidates : cmp matches the decided prefix above b, and cmp[b] = 0}
    if r <= c0 : T[b] = 0      else : T[b] = 1, r -= c0          (r starts at D)

After bit 0, T is the D-th smallest cmp and r is the number of tokens with cmp == T to
demote. A final row-serial pass demotes every cmp < T and the first r with cmp == T (by
idx), emits one index per cycle to the issue queue (credit flow control as for decode
victims), and writes the tier bits of the demoted lanes to LOW with a bit write mask
(S is kept: LOW scores are frozen).

Ineligible lanes carry the all-ones cmp code, which is never counted as a 0 bit, never
equals T and is never below T, so no candidate mask is needed in the passes.

Hardware: 16 lane compare units (shift + prefix match + 42-bit compare), prefix and mask
registers, three 13-bit counters, a 16-way priority encoder. Lane control (age, sh) is
registered alongside the SRAM read to keep the SRAM -> compare path short.

Cycles per head ≈ 42 x (ceil((t_p - W_EFF)/16) + 4) + 3 x rows + D.
Example (Llama-2-7B, t_p = 1,116, K = 279): ≈ 4,000 cycles per head, ≈ 4.1 M cycles for
32 x 32 heads (≈ 5 ms at 800 MHz). The attention stream stalls while a head is selected:
about 10 % of the ≈ 39 k prefill beats of such a head.

## 5. Interface / register changes

| Item | v2 |
|---|---|
| CTRL (0x00) bit 3 | SEL_EN: 1 = IPU selects at end of prefill, 0 = v1 behaviour (host) |
| 0x70 BULK | count of bulk demote packets (HAS_PERF) |
| 0x74 SEL_CYC | cycles spent in selection (HAS_PERF) |
| 0x14 VERSION | upper half 0x0003 |
| demote packets | unchanged 25-bit {b, l, h, idx}; bulk packets come head by head, idx ascending, before that head's drain |
| score bank | adds a bit write mask (`wmask`, FakeRAM45 `w_mask_in` convention) |

## 6. Verification

All v1 tests plus a new top scenario `big` (seq 0: t_p 250, K 62 -> 188 bulk demotes per
head; seq 1: K = 0 and a head of constant scores, so every candidate ties). The `main`,
`raw` and `edge` scenarios now run with SEL_EN = 1 and check the bulk packets in order
and the DRAM image right after the flush (with the LOW tiers set by the IPU).

| Test | Result (Verilator 5.020 and Icarus 12) |
|---|---|
| ingress, datapath, tree (unit) | PASS |
| top edge / raw / main / big | PASS: 15 / 176 / 468 / 554 bulk demotes, exact DRAM images |
| Verilator -Wall lint | clean |

## 7. Synthesis: Nangate45 + FakeRAM45 (Yosys 0.33 + ABC, typical corner)

Score bank = `fakeram45_256x512_1r1w` (`syn/fakeram/`): FakeRAM45 conventions, timing of
the official `fakeram45_128x256` (clk->q 0.302 ns, setup 0.050 ns), area = 4 x
`fakeram45_128x256` (the official 16 KB single-port equivalent). The official FakeRAM45
set has no 1R1W macro and the generator (bsg_fakeram + CACTI) was not run, so a real
1R1W bank will be larger (typically 1.3–2x for 8T cells and a second decoder).

| | v1 (same flow) | v2 |
|---|---|---|
| logic area | 55,092 µm² | 68,484 µm² (+24 %) |
| flip-flops | 5,079 | 5,463 |
| SRAM macro | (black box) | 135,961 µm² |
| total | — | 204,445 µm² = 0.204 mm² |
| ABC worst logic path | 1.02 ns (FSM prefetch prediction) | 1.09 ns (pipe, unchanged v1 logic) |

v2 logic breakdown: `ipu_psel` ≈ 11,900 µm² (own logic 6,705 + its lane-compare shifter
4,533 + lane control 690); the rest of the increase is the CSR counters, FSM state and
bank muxes.

Timing (pre-layout, no wire load; ABC path + DFF clk->q ≈ 0.10 ns + setup ≈ 0.04 ns):

| Path | Estimate |
|---|---|
| worst logic path (v1 pipe) | 1.09 + 0.14 ≈ 1.23 ns |
| worst path inside `ipu_psel` | 0.83 + 0.14 ≈ 0.97 ns |
| SRAM -> `ipu_psel` compare -> flop | 0.302 + 0.665 + 0.04 ≈ 1.01 ns |
| SRAM min period | 0.293 ns |

So v2 keeps the v1 critical path; fmax ≈ 800 MHz before layout (expect noticeably less
after placement and routing; ABC's path delay varies by a few % run to run, so the
v1/v2 1.02 vs 1.09 ns difference is noise, not a regression). Before the lane-control
register was added, `ipu_psel` was the critical path at 1.28 ns.

Run: `LIB=<OpenROAD-flow-scripts>/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib syn/run_synth.sh`
(area: `syn/out/nangate45_stat*.txt`), and `yosys -c syn/timing_nangate45.tcl` for the
ABC critical path.

## 8. Not in v2

Overlap of selection with the next head's prefill (would need a second bank); a
narrower (4-lane) selection unit trading ~4x selection time for ~8,000 µm² less area;
a real 1R1W SRAM macro; layout.
