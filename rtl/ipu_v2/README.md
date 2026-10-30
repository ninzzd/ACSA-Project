# IPU — Importance Policy Unit (RTL v2)

Hardware implementation of the MiKV importance policy studied in this repository: it keeps a
cumulative-attention score per KV-cache token, applies the pow2 age decay, and demotes at most
one token per (sequence, layer, head) per decode step from the HIGH to the LOW precision tier.
v2 adds **end-of-prefill selection in hardware** (`src/ipu_psel.sv`): at the end of each
head's prefill the IPU itself picks the t_p − K tokens to demote and sends their indices to
the migration engine. v1 is unchanged in `rtl/ipu/`.

Specification: [`docs/ipu_spec.md`](docs/ipu_spec.md) (v0.2) + [`docs/ipu_v2_addendum.md`](docs/ipu_v2_addendum.md).

## Layout

| Path | Contents |
|---|---|
| `src/` | synthesizable SystemVerilog (Yosys 0.33 / Verilator / Icarus / Vivado subset) |
| `tb/` | self-checking testbenches: `tb_ingress`, `tb_datapath`, `tb_tree` (unit), `tb_ipu_top` (full IPU) |
| `model/` | `ipu_golden.py` (pure-integer reference), `gen_vectors.py` (unit vectors), `gen_top.py` (top scenarios) |
| `vectors/` | generated test vectors (committed, so simulation needs no Python) |
| `sim/` | `run_xsim.sh` (Vivado), `run_oss.sh` (Verilator / Icarus), `lint.sh` |
| `syn/` | `synth_generic.ys`, `synth_nangate45.tcl`, `timing_nangate45.tcl`, `run_synth.sh`, `fakeram/` (FakeRAM45-style 1R1W macro), `synth_sky130.tcl`, `openroad/` (v1 ORFS config, untested) |

## Run

```bash
# regenerate vectors (needs numpy)
cd model && python3 gen_vectors.py unit && python3 gen_top.py && cd ..

# simulate
sim/run_xsim.sh                 # Vivado xsim: 3 unit tests + 3 top scenarios
sim/run_oss.sh vl               # Verilator 5.x   (sim/run_oss.sh iv for Icarus 12)
sim/lint.sh                     # Verilator -Wall, must print "lint clean"

# synthesize (generic; add LIB=<NangateOpenCellLibrary_typical.lib> for Nangate45 + FakeRAM45)
syn/run_synth.sh
```

Each testbench prints `PASS ...` or `FAIL ...`.

## Status (v2, 30 Oct 2026)

- All tests pass on Verilator 5.020 and Icarus 12; Verilator `-Wall` lint is clean.
  Top scenarios edge / raw / main / big check 15 / 176 / 468 / 554 bulk demotes and the
  exact DRAM images.
- Nangate45 typical + FakeRAM45-style 1R1W bank (Yosys + ABC, before layout):
  - logic 68,484 µm² (v1: 55,092), 5,463 flip-flops; SRAM 135,961 µm²; total 0.204 mm²
  - worst path ≈ 1.23 ns (~800 MHz), unchanged from v1; selection is not on it
- Not in v2: overlap of selection with the next head, a real 1R1W SRAM macro, layout.
- Details: `docs/ipu_v2_addendum.md`.
