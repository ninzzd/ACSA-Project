# IPU — Importance Policy Unit (RTL v0.1)

Hardware implementation of the MiKV importance policy studied in this repository: it keeps a
cumulative-attention score per KV-cache token, applies the pow2 age decay, and demotes at most
one token per (sequence, layer, head) per decode step from the HIGH to the LOW precision tier.
Specification: [`docs/ipu_spec.md`](docs/ipu_spec.md) (v0.2).

## Layout

| Path | Contents |
|---|---|
| `src/` | synthesizable SystemVerilog (Yosys 0.33 / Verilator / Icarus / Vivado subset) |
| `tb/` | self-checking testbenches: `tb_ingress`, `tb_datapath`, `tb_tree` (unit), `tb_ipu_top` (full IPU) |
| `model/` | `ipu_golden.py` (pure-integer reference), `gen_vectors.py` (unit vectors), `gen_top.py` (top scenarios) |
| `vectors/` | generated test vectors (committed, so simulation needs no Python) |
| `sim/` | `run_xsim.sh` (Vivado), `run_oss.sh` (Verilator / Icarus), `lint.sh` |
| `syn/` | `synth_generic.ys`, `synth_sky130.tcl`, `run_synth.sh`, `openroad/` (ORFS config + SDC) |

## Run

```bash
# regenerate vectors (needs numpy)
cd model && python3 gen_vectors.py unit && python3 gen_top.py && cd ..

# simulate
sim/run_xsim.sh                 # Vivado xsim: 3 unit tests + 3 top scenarios
sim/run_oss.sh vl               # Verilator 5.x   (sim/run_oss.sh iv for Icarus 12)
sim/lint.sh                     # Verilator -Wall, must print "lint clean"

# synthesize (generic; add LIB=<sky130_fd_sc_hd__tt_025C_1v80.lib> for Sky130 area)
syn/run_synth.sh
```

Each testbench prints `PASS ...` or `FAIL ...`.

## Status (30 Oct 2026)

- All tests pass on Verilator 5.020 and Icarus 12. Vivado xsim scripts are provided but were not run here.
  - ingress: exhaustive over all 65,536 fp16 inputs
  - datapath: 4,000 random beats
  - argmin tree: 3,000 packets with ties
  - top level: three scenarios, `main` (1,248 packets, 468 victims), `raw` and `edge`. Each checks the DRAM image after prefill and at the end bit-exact, the identical victim sequence, and the sticky error flags.
- Verilator `-Wall` lint is clean.
- Sky130 HD, typical corner, score SRAM excluded:
  - 0.272 mm² and about 5,080 flip-flops
  - estimated critical path 2.75 ns before layout (~360 MHz)
- Not in v0.1:
  - the SRAM macro (to be discussed)
  - the in-IPU prefill selection unit (the host does it)
  - overlap of the decision step with the next head
