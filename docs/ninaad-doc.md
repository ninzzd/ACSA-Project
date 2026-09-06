# Ninaad's Project Documentation

## Software Implementation of MiKV

**Paper Link**: [No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization](https://arxiv.org/abs/2402.18096)

### Model Details

| | |
|---|---|
| Model | `meta-llama/Llama-2-7b-chat-hf`, `LlamaForCausalLM` |
| Parameters | ~6.74B (~13 GB at fp16) |
| Layers | 32 |
| Attention heads (Q) | 32 |
| KV heads | 32 — plain MHA, **no GQA**, so `num_kv_groups == 1` |
| Hidden size | 4096 |
| Head dim | 128 |
| MLP intermediate size | 11,008 |
| Vocab size | 32,000 |
| Tied embeddings | no |
| Max context | 4096 tokens |
| Dtype (this setup) | float16 on CUDA, float32 CPU fallback |
| Attention impl | `eager` (only backend returning real softmax attention weights, needed for scoring) |
| Prompt format | Llama-2 `[INST]`/`<<SYS>>` via `tokenizer.apply_chat_template` |
| Access | gated — needs an accepted license + `huggingface-cli login` / `HF_TOKEN` |

> **Switched from `Qwen/Qwen2.5-0.5B-Instruct`** (24 layers, 14 Q heads, 2 KV heads, ~494M). The 0.5B run showed MiKV's importance score to be near-uninformative on Line Retrieval (see *Experimental Findings*), which the paper's own curves suggest is a scale artifact — they use 7B+ models. Llama-2-7b-chat is the paper's own model family, so it tests that hypothesis directly.

The **-chat** checkpoint is required, not a preference: base `Llama-2-7b-hf` ships no `chat_template`, so `apply_chat_template()` raises on it outright. -chat is fine-tuned on Llama-2's own `[INST]`/`<<SYS>>` format; architecture and config are otherwise identical.

Because there is no GQA here, the group-aware code paths (`repeat_interleave` on $b$, the `num_kv_groups` reshape in scoring) all collapse to identity — kept as the general case rather than special-branched, so the same code still runs on GQA models.

### Channel Balancer

Computed once at prefill, per (layer, kv head, channel $c$), frozen for the rest of the sequence:

$$
b_c = \sqrt{\dfrac{\max_i |Q_{i,c}|}{\max_i |K_{i,c}|}}
$$

($\max_i$ over batch and all prompt positions $i$, and over the query heads sharing the kv head under GQA.) Applied post-RoPE, pre-cache:

$$
Q' = Q \,/\, b, \qquad K' = K \cdot b
$$

Two variants, selected by `balance_scheme` (`BALANCE_SCHEMES`), swept against each other by default:

| scheme | $b_c$ | rationale |
|---|---|---|
| `paper` | $\sqrt{q_{\max}/k_{\max}}$ | the balancer exactly as MiKV specifies it |
| `pow2` | $2^{\,\mathrm{round}((e_q - e_k)/2)}$ | $b$ snapped to the nearest power of two, exponent-only (`frexp`/`ldexp`, no `pow`). Applying it is then an exact exponent add/subtract in fp16 rather than a real multiply — what a hardware implementation wants. The cost is that $b$ is quantized to the $2^n$ grid. |

Both are computed in fp32 (the ratio can overflow fp16 before the `sqrt` pulls it back) and cast to the model dtype at a single chokepoint before meeting $Q$/$K$ — leaving $b$ wider silently promotes them and blows up at `o_proj`'s fp16 weights.

### Importance Policy

**Parameters:** budget $k$, recency window $w \le k$, top-score slots $k_H = k - w$, with $w = \lfloor r_w \cdot k \rfloor$ in both modes.

**Budget modes** (`BUDGET_MODES`, both swept by default). $r_k$ is constant either way; what differs is what it multiplies:

| mode | $k$ | meaning |
|---|---|---|
| `fixed_length` | $\lfloor r_k \cdot t_p \rfloor$, frozen at prefill | a fixed *number* of tokens stays HIGH, so as decoding extends the cache the high-precision *fraction* keeps shrinking |
| `fixed_ratio` | $\lfloor r_k \cdot t \rfloor$, re-derived every step from the current total $t$ | a fixed *fraction* stays HIGH, so $k$ grows with the sequence |

The two are **not equal-footprint at the same $r_k$**, so every result row is sized against its own $k$ (`resolve_budget_k`) rather than a shared one. They agree only at $t = t_p$.

> **Both conventions appear in H2O's own reference implementation** ([FMInference/H2O](https://github.com/FMInference/H2O)), in two different code paths: the masking/simulation path (`utils_lm_eval/modify_llama.py`) computes `heavy_budget = int(heavy_ratio * attn_weights.shape[-1])` per forward — a ratio of the *current* length, i.e. `fixed_ratio` — while the real-KV-drop path (`utils_real_drop/modify_llama.py`) takes absolute `hh_size`/`recent_size` integers, i.e. `fixed_length`. The paper's numbers come from the ratio path (`run_lm_eval_harness.py` imports `utils_lm_eval`), but under lm-eval-harness scoring the sequence is consumed in essentially one forward pass, so the two collapse into each other there and H2O's headline results don't actually distinguish them. Neither the paper nor the repo reports a head-to-head. The one direct comparison found is Appendix F of [arXiv 2510.08525](https://arxiv.org/pdf/2510.08525), which reports that **H2O performs similarly under both** (the large effect was on R-KV), and prefers a dynamic budget for *fairness of measurement* rather than accuracy — under a global fixed budget, samples shorter than the budget go uncompressed and the reported compression ratio is optimistic.

**Score:**

- *Prefill* — raw cumulative sum of attention received, H2O's heavy-hitter criterion (Zhang et al. 2023, Algorithm 1: $F_{\text{score}} := \sum_s o_s$), no normalization by token age or step count:

$$
a_j \mathrel{+}= \textstyle\sum_i P_{i,j}
$$

- *Decode* — each step simply adds the new query row's attention. **The stored score is never decayed:**

$$
s^{(i+1)} = s^{(i)} + a
$$

**Age decay — stateless, and only for finding the victim.** The decay is not part of the scoreboard state. It is a static transform formed on the comparison path, used to pick what to demote, and then discarded:

$$
s'^{(i+1)} = \dfrac{s^{(i+1)}}{\text{age}}, \qquad \text{age}_i = \max(1,\ (t-1) - i), \qquad \text{victim} = \arg\min_i\ s'^{(i+1)}
$$

where $t-1$ is the index of the newest token. `clamp(min=1)` covers the newest token, whose age is 0. Because nothing is carried between steps, the discount **cannot compound**: $s'$ is recomputed from scratch from the raw sum every time, and $a$ itself remains exactly H2O's $F_{\text{score}}$.

Selected by `score_decay_application` (`SCORE_DECAY_APPLICATIONS`):

| setting | behaviour |
|---|---|
| `ranking` (**default**) | as above — raw sum stored, $s'$ on the compare path only |
| `compounding` | *legacy*: $s'$ written back into the scoreboard, so the discount re-applies to an already-discounted score every step and decays geometrically. Kept only to reproduce sweeps run before this changed. |

These are **different policies, not variants of one**, and the choice sets the accumulator width requirement: a raw sum grows monotonically toward $t$, a compounded one never does (measured over the same inputs: max 7.27 vs 0.93).

**Importance set** $S$ (recomputed every step, size always $k$ once $t > k$):

```
importance_set(a, t, k, w, k_H):
    if t <= k: return {0, ..., t-1}            # cache not yet full: everything important
    S = {t-w, ..., t-1}                        # recency window, unconditionally in
    s_prime = a / age                          # stateless, formed here and nowhere else
    if k_H > 0:
        S += top_kH(s_prime[j] for j not in S)  # best (k - w) by s', among the rest
    return S                                    # |S| == k
```

Taking the top-$k_H$ by $s'$ and demoting what falls out is the same decision as $\arg\min s'$ over the eligible positions — verified directly, the argmin is always among those excluded from $S$.

> **Deviation from H2O/MiKV spec — the ranking criterion only.** H2O's and MiKV's criterion is the *raw, undecayed* cumulative sum (MiKV explicitly delegates its importance criterion to H2O rather than defining its own), and the **stored score now matches that exactly**. The deviation is confined to one division on the read path: selection ranks on $s' = s/\text{age}$ rather than on $s$. The motivation is unchanged — at small scale the raw sum lets prefill-era mass dominate selection, since the earliest positions accrue attention over the entire prompt before the query token is even seen, a structural head start unrelated to current relevance.
>
> **History.** An early version ranked on `a/age` at selection time; that was reverted to match spec; the decay was then re-introduced as a *compounding, stateful* mutation of `a`; it has now been reverted again to the stateless one-shot form described above, which is both the original idea and what the hardware implements. Results predating this change are not comparable — see the note in *Experimental Findings*.

### Scoreboard Precision

All of this is **opt-in and off by default** — a plain run uses `native`/`exact`, which is bit-identical to the pre-existing behaviour, and the sweep does not cross these axes (they would multiply an already-4× sweep). Set them explicitly to study them.

**What the scoreboard actually is by default.** Not fp32: `eager_attention_forward` softmaxes in fp32 but casts straight back with `.to(query.dtype)`, so the attention weights arrive **fp16**, the sums are fp16, and the running add is fp16. The only fp32 island is the decay divisor, which upcasts, divides and casts back on one line.

**`score_scheme` / `score_bits`** (`SCORE_SCHEMES`) — the precision the scoreboard accumulates at:

| scheme | accumulator |
|---|---|
| `native` (**default**) | the model dtype, fp16 on CUDA |
| `fp32` | fp32, *including the prefill reduction itself* — summing $t_p$ attention rows is where fp16 error is largest, so casting only afterwards would hide what this variant measures. A reference point, not a hardware proposal. |
| `quant` | affine min/max fake-quantization to `score_bits`, per (batch, kv head) across all positions, re-derived at every update |
| `fixed` | the IPU's real score path — see below |

`quant` pays **two roundings per decode step** (after the add and after the decay), because an N-bit register is N bits at every stage, not only at the end.

> **Decay and scoreboard quantization compound.** The decay divides old positions by up to $t-1$ while the newest keeps its full value, and an affine quantizer sets its bins from the head's min/max — so the max is pinned by recent tokens and the decayed old scores collapse into the bottom bin. Measured on a $t=2048$ scoreboard: at **8 bits, 83.5% of positions land in the bottom bin** (99.0% at 4 bits, 99.7% at 2 bits). Those are exactly the positions the top-$k_H$ search has to rank, so `topk` falls back to arbitrary tie-breaking and the heavy-hitter half of the budget stops tracking attention. Read any `quant` result against that, not as "N bits is enough".

**`score_decay_scheme`** (`SCORE_DECAY_SCHEMES`) — the precision of the decay factor itself, which scoreboard quantization alone leaves untouched:

| scheme | factor |
|---|---|
| `exact` (**default**) | the fp32 divide. Deliberately a divide, not a reciprocal-multiply — they round differently, and this branch must reproduce the original behaviour bit-for-bit so the others are measured against it. |
| `quant` | $1/\text{age}$ fake-quantized to `score_bits` — a finite-width decay LUT |
| `pow2` | $1/\text{age}$ snapped to an exact power of two, so decay is an exponent subtract. Same hardware argument as the `pow2` balancer. |
| `lut` | the IPU's actual ROM — see below |

### Hardware Model: the IPU Score Path

`score_scheme="fixed"` and `score_decay_scheme="lut"` model the microarchitecture rather than a generic quantizer, so the software sweep predicts the RTL.

**Fixed-point format.** Standard DSP `fixdt(sign, length, fraction)` — MATLAB/Simulink convention, where the middle term is the **word length, not a count of integer bits**:

$$
(1,\ l,\ f): \quad l \text{ bits total, 1 sign, } f \text{ fractional} \implies l-1-f \text{ integer bits (derived)}
$$

$l$ is the numeric field, i.e. the score word minus its tier bits — the 2 bits encoding HIGH/LOW/MIGRATING carry no numeric weight:

| score word | $l$ | $f$ | range | resolution |
|---|---|---|---|---|
| 32-bit | 30 | 16 | $[-8192,\ 8191.99998]$ | $1.5\times10^{-5}$ |
| 32-bit | 30 | 8 | $[-2097152,\ 2097151.996]$ | $3.9\times10^{-3}$ |
| 16-bit | 14 | 8 | $[-32,\ 31.996]$ | $3.9\times10^{-3}$ |
| 16-bit | 14 | 0 | $[-8192,\ 8191]$ | $1$ |

Both are parameters (`score_length_bits`, `score_frac_bits`, plus `score_signed`); an $(l,f)$ pair leaving no integer bits is rejected at policy construction, not thousands of tokens into a run.

**Saturation headroom.** The largest score physically reachable in a 4096-token context is 4096 (a token attended with weight 1.0 at every step):

- `fixdt(1,30,16)` → max 8191.99998, **fits with 2× margin**
- `fixdt(0,30,16)` → max 16383.99998, 4× margin. The score path is provably non-negative, so the sign bit buys nothing; free to drop at $l=30$, and worth a full integer bit at $l=14$.
- `fixdt(1,29,16)` (the `Q12.16` annotation on p.2 of the microarchitecture notes, plus a sign bit) → max 4095.99998, **saturates exactly at the worst case, zero margin**. The 30-bit field drawn on the pipeline page is the correct one; the `Q12.16` note should read `Q14.16`.
- `fixdt(1,14,8)` → max 31.996, **saturates**. A 16-bit word needs $f \le 1$ to hold a raw cumulative sum — so on 16 bits, $f$ is not a free precision knob, it directly trades resolution against whether the accumulator saturates at all.

**Ingress (`ipu_ingress`), fp16 → fixed point.** Calling it a cast rather than a quantization is right for the values that decide anything — above about $2^{-6}$ the fp16 grid is *coarser* than $Q_.16$, so the conversion is exact (0.5, 0.1, 0.03125 all round-trip bit-exact). It is lossy in the tail: anything below half an LSB ($2^{-17} \approx 7.6\times10^{-6}$) becomes **exactly zero**. Against a uniform-attention share of $2.4\times10^{-4}$ at $t=4096$, a token receiving under ~3% of uniform contributes nothing, its score stays 0 forever, and the victim search among such tokens falls through to `ipu_min_tree`'s lowest-index tie-break — i.e. oldest-first, a FIFO. Emergent, not designed.

**Age LUT (`ipu_age_lut`).** A 512-entry $Q0.16$ ROM, ages beyond it clamped to the last entry. At $t=4096$ that means **88% of ages share the single $1/512$ entry** (age 1000 → 0.001953 vs 0.001000 exact; age 4095 → 0.001953 vs 0.000244, an 8× error). This is far more benign than it looks: within that group the clamp is a **uniform scale**, so relative order is untouched and the argmin over old tokens is decided purely by raw score. Only comparisons *across* the 512 boundary are distorted — and if $w \ge 512$, eligibility excludes everything below it and the boundary never matters. ($Q0.16$ also cannot represent 1.0, so the age-1 entry saturates to 0.999985 — harmless while the window excludes it.)

**Where the decay is applied — the microarchitecture and the software now agree.** `ipu_accum` writes `{tier, accum.score_out}` back to the score SRAM *undecayed*, while `ipu_age_lut` sits on the branch to `ipu_min_tree` feeding `cmp_val` only; p.3 states $s^{(i)} = S^{(i)}/\text{age}^{(i)}$, a one-shot quotient. That is exactly the `ranking` policy now used by default.

### KV Quantization

**Quantization:**
- prefill: $j \in S \to$ HIGH, $j \notin S \to$ LOW bits ($N$)
- decode (steady state $t = k+1$): recompute $S$; the one position that falls out of $S$ is demoted to LOW bits — sticky, never promoted back.
- HIGH means either the model's native fp16 (untouched) or quantized to `high_bits`, toggled by the global `high_precision_native` flag (default: native).

`quantize_kv` is a **fake** quantizer: it round-trips through an affine $N$-bit grid and dequantizes straight back to fp16, so it measures quantization's effect on *accuracy* without shrinking the real `DynamicCache`. Two changes from the first version:

- **Group-wise scales.** Min/max are now taken per group of `group_size` consecutive channels (default `head_dim / 2`, i.e. 2 groups of 64) instead of once per whole 128-d vector. One outlier channel no longer stretches the scale for the entire head — the dominant source of error at $N = 2$.
- **`bits >= 16` short-circuits** to the identity, so a 16-bit "LOW" bucket is a true no-op rather than a lossy round-trip through a 65535-level grid.

### Code Layout

`scripts/mikv.py` used to be one ~3600-line file. It is now the command line and nothing else, over eight modules — split so the policy can be read without the sweep and vice versa. Entry point and all commands are unchanged: `python scripts/mikv.py`.

| module | lines | what it holds |
|---|---|---|
| `mikv.py` | ~350 | argparse, preset/flag merge, `--dry-run`, the `main` block |
| `mikv_runtime.py` | ~40 | **import-order bootstrap** (see below) |
| `mikv_config.py` | ~215 | every constant the policy is parameterized by, plus the name↔knob helpers (`high_precision_knobs`, `format_score_tag`) |
| `mikv_quant.py` | ~170 | the quantizers only: affine (K/V, scoreboard) and fixed-point with a static scale |
| `mikv_policy.py` | ~725 | `load_model`, `MiKVPolicy`, the attention monkey-patch, the generation loop |
| `mikv_bench.py` | ~410 | Line Retrieval, the no-quant baseline, KV footprint bytes |
| `mikv_sweep.py` | ~985 | `SweepPoint`, axis enumeration, the three walks, the driver |
| `mikv_grids.py` | ~235 | **the knobs you edit**: coarse value lists + presets |
| `mikv_report.py` | ~495 | grouping, markdown tables, the results CSV |
| `mikv_plots.py` | ~370 | figures; not imported at all unless `--plots` |

The import graph is a clean DAG — `runtime → config → {quant, grids} → policy → bench → sweep`, with `report` off `config` and `plots` off `report` — verified acyclic.

**`mikv_runtime.py` exists because two import orderings are load-bearing**, and both fail as a bare segfault with no Python traceback. `HF_HUB_DISABLE_XET=1` must be set before `transformers`/`huggingface_hub` import (it is read once at import time), and `torch`/`transformers` must import before `matplotlib.pyplot` (a native-library symbol conflict; whichever loads first wins). In a single file that was two comments and a fixed line order; across modules it would have been a latent trap, so it is enforced in one place that everything else imports. `mikv_plots.py` imports it first with a `# noqa` and a note.

**Two consequences of the split worth knowing.** Names that were private but are now crossed between modules got a public spelling — `_score_tag` → `format_score_tag`, `_configs_in` → `configs_in`, `_pareto_front` → `pareto_front`, and so on; the genuinely module-internal ones (`_axis_values`, `_greedy_sweep`, `_build_arg_parser`) kept their underscore. And `mikv_plots` is now a **lazy import** inside the `--plots` branch, so a sweep run never pays matplotlib's import cost or its failure modes.

### Running

`scripts/run_sweep.sh` is the way in — it names a preset, gives that preset its own results directory, prints the plan before spending anything, and can detach a multi-hour run. It forwards every unrecognised flag to `mikv.py` verbatim, so nothing is hidden behind it.

```bash
./scripts/run_sweep.sh list                     # the sweep types (read out of mikv_grids.py)
./scripts/run_sweep.sh greedy --dry             # cost it; loads no model, touches no GPU
./scripts/run_sweep.sh greedy                   # the recommended pass: 54 runs, ~3 h
./scripts/run_sweep.sh greedy --bg              # same, detached; prints the log to tail
./scripts/run_sweep.sh ofat                     # controlled marginals off a fixed baseline
./scripts/run_sweep.sh confirm                  # the follow-up grid, once you know what matters
./scripts/run_sweep.sh greedy --help            # every mikv.py flag
```

**Always `--dry` first.** A sweep is GPU-hours and the plan is free; the plain form prints the plan and asks for confirmation anyway, and every axis flag below changes the count. Results land in `results/<type>/sweep.csv`, appended as each run completes. `PYTHON` and `RESULTS_DIR` override the interpreter and the output root.

Narrowing a preset is the normal way to make a grid affordable — a flag replaces one axis and leaves the rest:

```bash
# the confirmation grid, cut to the two axes a greedy pass said mattered
./scripts/run_sweep.sh confirm --low-bits 2,4 --window-tokens 32,64 --dry

# does the ROM depth matter at all, holding everything else at the reference design?
./scripts/run_sweep.sh ofat --age-lut-entries 128,256,512 --dry

# the two footprint-free axes crossed properly: same KV size at every point
./scripts/run_sweep.sh score --dry

# a one-off configuration, no preset semantics
./scripts/run_sweep.sh legacy --sweep-mode grid \
    --high-tiers fp16 --low-bits 2 --window-tokens 64 \
    --score-schemes fixed --score-length-bits 14 --score-frac-bits 4 \
    --score-decay-schemes lut --age-lut-entries 512 --dry
```

`mikv.py` can also be called directly when you do not want the per-type results directory:

```bash
python scripts/mikv.py --dry-run                            # default preset (greedy)
python scripts/mikv.py --preset confirm --csv out.csv
python scripts/mikv.py --low-bits 2,4 --high-tiers fp16,int8 --sweep-mode grid
python scripts/mikv.py --preset greedy --plots              # render the figures too
```

What each preset costs, at the observed 3.5 min/run on a V100:

| preset | configurations | runs | wall time | what it is for |
|---|---|---|---|---|
| `greedy` *(default)* | 18 | 54 | ~3.1 h | the recommended first pass — coordinate descent |
| `ofat` | 18 | 54 | ~3.1 h | the same budget as controlled marginals |
| `score` | 48 | 144 | ~8.4 h | balancer × scoreboard × ROM depth, all at one footprint |
| `budget` | 54 | 162 | ~9.4 h | mode × r × w × HIGH × LOW — read off the Pareto front |
| `confirm` | 72 | 216 | ~12.6 h | the second pass; narrow it with the flags first |
| `legacy` | 4 | 12 | ~0.7 h | the historical sweep, kept reproducible by name |
| `exhaustive` | 2304 | 6912 | **~16.8 d** | to be costed with `--dry`, not launched |

Logs to `mikv_run_<timestamp>.log` (stdout+stderr tee'd), or to `results/<type>/run_<timestamp>.log` under `--bg`. **Plotting is off by default** — the CSV and the tables are the sweep's product and the figures regenerate from them, so a matplotlib error must not be able to take down a finished sweep; pass `--plots` to render them.

### The coarse sweep grids

All value lists live together near the top of the file, so a preset, a CLI run and this doc cannot disagree about what "the scoreboard sweep" means.

| grid | constant | values |
|---|---|---|
| scoreboard | `SCOREBOARD_SWEEP` | 8: `fp32`, `fp16` (native), and fixed-point at (l=14, f=2/4/6) and (l=30, f=12/16/20), all unsigned |
| budget mode | `BUDGET_MODE_SWEEP` | `fixed_length`, `fixed_ratio` |
| budget ratio r | `BUDGET_RATIO_SWEEP` | 0.25, 0.5, 0.75 |
| LOW tier | `LOW_TIER_SWEEP` | int2, int3, int4 |
| HIGH tier | `HIGH_TIER_SWEEP` | int8, int16, fp16 |
| window w | `WINDOW_SWEEP` | 32, 64, 128 **tokens** |
| balancer | `BALANCER_SWEEP` | `paper` (sqrt), `pow2` |
| age LUT depth | `AGE_LUT_SWEEP` | 128, 256, 512 entries (256 B / 512 B / 1 KB) |

Four things about these are load-bearing:

**The scoreboard is ONE composite axis, not four crossed.** Its four fields (scheme, word length l, fraction f, signedness) only make sense together, and the legal settings are a list, not a product — crossing l and f independently enumerates `l=14, f=16`, which has no integer bits and gets dropped, alongside `l=30, f=2`, which nobody would build. `enumerate_sweep_points` accepts an axis whose values are *dicts of field assignments* for exactly this; `SCOREBOARD_SWEEP` is that list. The 2 MSBs of the stored word are tier flags (HIGH / LOW / MIGRATING), which is why a 16-bit word leaves l = 14 and a 32-bit one leaves l = 30, and why all of them are unsigned: the score is a sum of softmax weights, provably non-negative, so a sign bit would be a wasted bit of range.

**Choosing f is a range-vs-resolution trade against a running sum.** The integer field has to hold the largest score a sink token accumulates (up to ~t, so ~12 integer bits at a 4096-token context) while the fraction has to resolve one step's attention delta or that token's score never moves at all. At l = 14 the word is genuinely *tight* — f = 6 leaves 8 integer bits and will saturate on sinks, f = 2 leaves 12 but flushes small deltas — which is the point: that is the row that halves the score SRAM. At l = 30 only f = 20 is anywhere near the edge, and it is in the list to bracket where saturation starts to bite.

**w is now an absolute token count, not a fraction of k.** `window_tokens` (32/64/128) is what `MiKVPolicy.budget_for` reads when set; `window_ratio` remains as the legacy alternative and is used only when it is not. The window protects the tail of the context, and how many tokens that takes has nothing to do with how large k happens to be. It is still carved *out of* the budget and clamped to k — `k_H = k − w`, never `k + w` — so on a short prompt at r = 0.25 the widest window can meet or exceed k and degenerate to pure recency. That is a real result, not a misconfiguration, and the axis-effects table will show it as such.

**int16 and fp16 are genuinely different.** `quantize_kv` now short-circuits at `bits > 16` rather than `>= 16`, so `bits=16` is a real affine round-trip: a uniform grid over the group's [min, max] against fp16's logarithmic one. The two agree near the group maximum — where fp16's spacing, `max·2⁻¹¹`, is the *coarser* of the two — and diverge on small-magnitude channels, where a uniform `(max−min)/65535` step is far coarser than fp16 resolves. Native fp16 is reached by not calling the quantizer at all, not by passing bits=16. (`quantize_scores` keeps its own `>= 16` guard, so nothing about the scoreboard changed.)

### The age LUT (`ipu_age_lut`)

The decay factor multiplied into the scoreboard before the argmin is `1/age`. A hardware divider is iterative and multi-cycle, and the score path needs P = 16 of them retiring one beat per cycle, so the divide is replaced by a reciprocal ROM and a multiply: `R[a] = round(2^F / n)`, `cmp_val = (S * R) >> F`. Depth is now a sweep axis (`--age-lut-entries`), read only when `score_decay_scheme="lut"` and collapsed to a single point otherwise.

**The previous model was wrong, and not by a little.** It indexed the table directly by age and **clamped** everything past the end, so every token older than the table shared one decay factor. The design in the brief **folds** instead, exploiting the self-similarity of `1/n` under powers of two — there is no entry for 1000 because 1000 is 500 doubled and `1/1000` is `1/500` halved — so the table covers one octave and larger ages are halved until they land back inside, with the halving absorbed into the output shift. Measured worst-case error over ages 65–4096:

| depth | clamp (old model) | fold (implemented) |
|---|---|---|
| 128 | 1462% | **0.53%** |
| 512 | 290% | **0.19%** |

**`F` is derived, not chosen.** It is bounded by the largest stored entry, which sits at the *smallest* age the ROM can see — and that is not age 1. The recency window guarantees no token younger than `W` reaches the comparator, so `n_min = W + 1` and `F = floor(log2((2^B − 1) · n_min))`. At B = 16, W = 64 that is **F = 22** rather than the 15 a table starting at age 1 would allow: seven extra fractional bits out of a masking rule that already existed. Since `w` is itself a swept axis, the policy passes its own configured window, and F moves with it (21 / 22 / 23 at w = 32 / 64 / 128).

**The implementation is validated against the brief's own numbers.** ROM spot values (`R[0]=64528`, `R[63]=32768`, `R[511]=7282`), the monotonicity and no-adjacent-duplicates assertions the RTL testbench makes, the age-3000 worked example, and all five rows of the error table — 0.516/0.309/0.175/0.096/0.058% worst case at depths 128–2048 — reproduce exactly.

Two things the sweep has to know about this axis:

**It models the whole comparator path, not just the ROM.** `age_lut_cmp_val` shifts the *product* (`(S·R) >> (F+e)`, never `S · (R >> e)`, which would discard up to 4 of the ROM word's 16 bits) and rounds onto the comparator's own fixed-point output word. That second rounding is the larger error source — the brief measures ~1.2% victim disagreement at *every* depth from 128 to 2048 from output quantization alone, against ~0.07% from the reciprocal at depth 512. Modelling the ROM in isolation would misattribute the error and make depth look far more decisive than it is. Since the point of sweeping depth is to find where it stops mattering, the floor has to be in the model.

**Depth and window interact, and the sweep enforces it.** The fold halves an age until it is at or below the table top `W + D`, so the smallest address it can generate sits just above `(W + D)/2`; for that to remain inside the table needs **`D ≥ W + 2`**. `w = 128` with a 128-deep ROM violates this — the fold undershoots and the address clamps, silently flattening the decay for the oldest tokens. `SweepPoint.invalid_reason` drops that combination from the plan with a printed reason rather than producing quiet garbage. It is a real constraint on the design (deepen the ROM or narrow the window), not a modelling artifact.

There is **no knee to find** in the depth curve: accuracy-per-byte is a straight line on log-log, so doubling the ROM halves the error forever and a depth is chosen against a budget rather than by locating a bend. The budget here is the output-quantization floor above.

### Profiling scheme

An exhaustive pass over these grids — now including the 3 LUT depths — is **2304 configurations × 3 ratios = 6912 runs ≈ 16.8 days** of continuous V100 time at 3.5 min/run (288 of the 2592 raw combinations are dropped by the `D ≥ W + 2` rule). The `exhaustive` preset exists to be costed with `--dry-run`, not launched.

The split that makes a cheaper pass sound is structural, and the code encodes it as `FOOTPRINT_AXES`:

- **Footprint-free axes** — balancer, scoreboard, decay. They change only *which* tokens are kept and how faithfully they are ranked, never how many, so every candidate sits at an identical KV size and **accuracy alone is the right criterion**. A cheaper value that holds accuracy is free in area terms. These decisions are unambiguous.
- **Footprint axes** — budget mode, r, w, HIGH, LOW. More cache is always at least as accurate, so maximizing accuracy here just walks to the largest configuration. There is no single "best": the answer is the **Pareto front**.

The three schemes, cheapest first:

| `--sweep-mode` | preset | cost | what it gives you |
|---|---|---|---|
| `ofat` | `ofat` | 18 cfg → **54 runs (~3 h)** | Each axis moved one value off a fixed baseline. A clean, *controlled* marginal per knob — blind to interactions. |
| `greedy` | `greedy` | ≤18 cfg → **≤54 runs (~3 h)** | Coordinate descent: sweep an axis, keep its winner, move on. Same cost as OFAT, but each axis is decided against the winners of the ones before it, so it recovers some interaction structure for free. |
| `grid` | `exhaustive` | 2304 cfg → **6912 runs (~16.8 d)** | Everything, including interactions. Not affordable. |

**Recommended: `greedy` (the default), then a targeted `grid` to confirm.** `GREEDY_AXIS_ORDER` settles the footprint-free axes **first** — that ordering is the whole design. Those decisions are unambiguous and cheap to trust, and they carry into every axis afterwards; only then does the walk spend anything on axes that trade accuracy for cache, scoring those on accuracy-per-byte (`_greedy_objective_value`, `greedy_objective="auto"`).

Two honest limits, stated because greedy's answer looks more authoritative than it is:

1. It finds a **local** optimum along the path it walked. It cannot see an interaction that only pays off in a direction it already walked away from, and a different axis order can land somewhere else. Treat the result as a strong candidate, not the optimum — hence the `confirm` preset (216 runs, ~12.6 h), narrowed with the CLI flags to whichever axes actually moved accuracy.
2. `accuracy_per_byte` is a **crude scalarization** of a genuine two-objective problem. It exists so the walk can proceed. For the footprint axes, read the Pareto-front table, not the value greedy picked.

So: `greedy` (~3 h) → read the axis-effects table for which knobs moved → `confirm` narrowed to those (~half a day) → pick off the Pareto front. Under 24 h of V100 time total, against 16.8 days for the exhaustive grid.

### The runner

`scripts/run_sweep.sh` (commands under [Running](#running)) adds three things to a bare `mikv.py` invocation, all of them about not losing a long run:

**One results directory per sweep type.** `results/<type>/sweep.csv`. The CSV is *appended* to by design, so without this a greedy pass and a confirmation grid land in one file with nothing but `run_id` to tell them apart. Every row still carries its own `run_id` and full configuration, so this is convenience rather than correctness.

**The plan is always printed first**, and an interactive invocation confirms before starting. `--dry` stops there. The check is guarded by `[ -t 0 ]`, so a redirected or CI invocation runs without blocking on a prompt.

**`--bg` detaches** with `nohup` and prints the log path, so a multi-hour sweep survives a dropped SSH connection.

Rows are appended to the CSV **as each run completes** (`checkpoint_csv`, on by default; `--no-checkpoint` to disable), so a sweep that dies at run 60 of 63 keeps the first 59.

The preset list the script validates against is read out of `mikv_grids.py` with `sed`, not by importing it — importing pulls in `mikv_config` → torch + transformers, which is ~10 s of load time to print a list of seven names. `mikv.py --help` remains the authoritative list.

### Sweep mechanics

Every axis takes a comma-separated list and has a CLI flag. A flag overrides one axis of the chosen preset and leaves the rest (and un-pins that axis from a greedy/OFAT baseline, so the values asked for are actually swept). `--dry-run` prints the resulting plan without loading the model.

| flag | axis | accepted values |
|---|---|---|
| `--budget-modes` | budget mode | `fixed_length`, `fixed_ratio` |
| `--budget-ratios` | r | floats in (0, 1] |
| `--balancers` | channel balancer | `paper`, `pow2` |
| `--window-tokens` | w, absolute | integers (32, 64, 128) |
| `--window-ratios` | w as a fraction of k | floats in [0, 1] — legacy, read only when `--window-tokens` is unset |
| `--high-tiers` | HIGH storage | `fp16`, `int16`, `int8`, `int4` |
| `--low-bits` | LOW storage | integers (2, 3, 4) |
| `--score-schemes` | scoreboard | `fp32`, `native`, `quant`, `fixed` |
| `--score-bits` | scoreboard affine width | integers — read only under `quant` |
| `--score-length-bits` | scoreboard word length l | integers (14, 30) — read only under `fixed` |
| `--score-frac-bits` | scoreboard fraction f | integers — read only under `fixed`, must leave integer bits |
| `--score-signed` | scoreboard signedness | `0`/`1` — read only under `fixed` |
| `--score-decay-schemes` | age decay | `exact`, `quant`, `pow2`, `lut` |
| `--score-decay-applications` | how decay is applied | `ranking`, `compounding` |
| `--age-lut-entries` | ROM depth D | integers (128, 256, 512) — read only under `--score-decay-schemes lut`, and constrained by `D >= w + 2` |

Plus `--sweep-mode {grid,ofat,greedy}` and `--greedy-objective` to override the walk a preset chose, `--num-samples`/`--num-records`/`--max-tokens`/`--seed`/`--model` for the benchmark itself, and `--csv`/`--plot`/`--plots`/`--no-checkpoint` for output.

`high_tiers` folds the old `high_bits` + `high_precision_native` pair into **one** named axis (`HIGH_TIER_MODES`, mapped back by `high_precision_knobs`): the two are not independent — `high_bits` is unread while the native flag is set — so crossing them would run identical configurations twice.

Two guards keep an over-broad sweep from wasting GPU time. `SweepPoint.canonical` collapses configurations that are **the same experiment**: an unread `score_bits` under a non-quantized scheme, the (l, f, signed) triple under a non-fixed one, `window_ratio` once `window_tokens` is set, and the *entire* score path when w = k, since k_H = 0 makes `_importance_set` return the pure recency mask without ever consulting the scoreboard. `SweepPoint.invalid_reason` drops impossible points from the plan **with a printed reason** rather than raising hours in — an (l, f) pair with no integer bits, a HIGH tier coarser than the LOW one, a ratio out of range. `extra_points` covers what neither walk reaches: combinations only valid *together*.

**Results CSV** (`write_results_csv`, default `docs/kv_compression_sweep.csv`) — **appends**, so results accumulate across invocations. 35 columns, one row per (configuration, ratio), each row carrying the full configuration that produced it:

| group | columns |
|---|---|
| identity | `timestamp`, `run_id` (shared by every row of one sweep), `model_name` |
| policy | `budget_mode`, `scheme`, `ratio`, `window_tokens`, `window_ratio`, `high_bits`, `low_bits`, `high_precision_native` |
| scoreboard | `score_scheme`, `score_bits`, `score_decay_scheme`, `score_decay_application`, `score_length_bits`, `score_frac_bits`, `score_signed` |
| hardware model | `hw_delta_bits` (recorded, not parameterized), `hw_age_lut_entries` (now a **swept axis** — the ROM depth that produced the row) |
| run | `sweep_mode`, `num_samples`, `num_records`, `max_tokens`, `seed`, `t_p` |
| results | `k`, `seq_len`, `accuracy`, `baseline_accuracy`, `kv_size_before`, `kv_size_after`, `compression_pct`, `avg_seq_len`, `kv_bytes_occupied` |

Columns irrelevant to a row are left empty (a `native` row has no `score_length_bits`). Two guards: keys outside the fixed column list are dropped rather than shifting the alignment, and a file whose header does not match is **left untouched** while the rows divert to `<stem>_<run_id>.csv` with a warning — a sweep is hours of GPU time and must not lose results to a header mismatch. A row with no `score_decay_application` column predates that column and is read back as `compounding`, so legacy rows are not silently relabelled as today's policy.

**Tables** (`format_results_table`) — one section per configuration, then three cross-cutting summaries that only appear once something varies. *Axis effects*: per axis value, the run count, mean/best accuracy and **mean KV size** — the last column is what says whether that axis was compared at equal cost (balancer, scoreboard) or not (r, w, the bit widths). *Balancer head-to-head*: accuracy per balancer keyed on every *other* varying axis plus r, so only rows at an identical footprint under an identical policy are put side by side. *Pareto front*: the rows nothing else beats on both footprint and accuracy — the shortlist a hardware configuration should be picked from, since everything off it is strictly dominated by something in the same sweep. Only axes that actually varied are named anywhere, so a single-configuration sweep renders as compactly as it always did.

**Plots** — three figures, all of which only name the axes that varied:

- `plot_accuracy_vs_compression` → one figure per configuration plus a combined overlay, named `kv_compression_sweep_<mode>_<balancer>[_<axis slugs>].png`. A default sweep keeps the filenames it always had. Past `MAX_PER_CONFIG_FIGURES` (12) configurations the per-configuration figures are skipped by default (`per_config=True` forces them) and the overlay facets into several numbered figures of ≤ 8 series each, sorted best-accuracy-first — hues are never cycled, so a repeated colour never means two different configurations.
- `plot_axis_effects` → `..._axes.png`, one panel per varying axis: mean accuracy per value with the individual runs as dots behind it. The flat panel is a knob the hardware can choose freely; the tall step is where accuracy is being spent.
- `plot_pareto` → `..._pareto.png`, every run in (footprint, accuracy) with the front drawn and labelled.

Scoreboard tags identify a configuration compactly: `native`, `quant8+decaypow2`, `fixed30.16+decaylut`, `fixed14.8u`.

### torch / transformers Integration

Llama-2-7b-chat-hf loaded via `AutoModelForCausalLM` with `attn_implementation="eager"` (only backend that returns real softmax attention weights, needed for scoring). Each layer's `LlamaAttention.forward` is monkey-patched (`types.MethodType`) to wrap the stock `attention_interface` call: balance $Q,K$ by $b$ post-RoPE/pre-cache (a query is never cached, so this can't be done after the fact), write through to `DynamicCache.layers[i].keys/values` with a fake N-bit round-trip quantizer, then score and re-demote after attention runs. All state ($b$, $a$, $S$) is shaped `[batch, num_kv_heads, t]` per layer, so masking/`topk` apply per kv-head automatically via broadcasting — no explicit per-head loop.

The rotary/eager-attention helpers (`apply_rotary_pos_emb`, `eager_attention_forward`, `ALL_ATTENTION_FUNCTIONS`) are still imported from `transformers.models.qwen2` — they are byte-identical to Llama's, both copied from the same upstream implementation, so the import survived the model switch unchanged.

Two ordering constraints in the imports are load-bearing, not stylistic (both cause bare segfaults with no traceback): `HF_HUB_DISABLE_XET=1` must be set before `transformers`/`huggingface_hub` import, and `torch`/`transformers` must import before `matplotlib.pyplot`.

### Measurement

**Answer extraction** (`_extract_number`) prefers the paper's own `<12345>` bracketed form, falling back to the first 4–6 digit run. The original "first integer anywhere in the continuation" rule scored preamble digits ("line 3 says…") as the answer, undercounting correct responses.

**Compression accounting — fixed at a 4096-token context.** The hardware target provisions a full 4096-token KV cache up front, so that allocation is what compression acts on. Both sides of the ratio are therefore computed at `seq_len = max_tokens = 4096`:

$$
\text{compression} = \frac{\min(L,k)\cdot 16 + \max(0, L-k)\cdot N}{L \cdot 16}, \qquad L = 4096
$$

> **Fixed — mixed-basis ratio.** An earlier version sized the numerator at `max_tokens` (4096, *provisioned*) while the denominator came from the baseline's measured per-sample `output.shape[-1]` (~$t_p$ + a short generation, *occupied*). Those are different quantities, so the quotient was not a compression ratio; it overstated the retained fraction substantially — at $t_p = 1709$, $N = 2$ and ~1750 tokens occupied, $k = 0.25\,t_p$ reads 50.6% on the mixed basis versus 21.6% correctly. Both sides now share the fixed-4096 basis.

The length generation actually reaches (`avg_seq_len`, and the `kv_bytes_occupied` it implies) is still measured and reported, but only as a **diagnostic** — decoding stops at EOS well short of 4096, so it describes occupancy, not the provisioned allocation. It is carried in the per-ratio result dicts and must not be mixed into the ratio.

### Experimental Findings: Line Retrieval vs. KV Compression

> ⚠️ **These results are from the previous Qwen2.5-0.5B-Instruct configuration and have not been re-run on Llama-2-7b-chat-hf.** They are what motivated the model switch; treat the numbers as historical until the 7B sweep completes.
>
> ⚠️ **They also predate the current importance policy.** They were run with a single fixed-length budget and the *compounding* decay, before `fixed_ratio` existed and before the decay became stateless. Both changed what is selected, so these numbers are not comparable to anything produced now — they are kept as the record of why the model was switched, not as a baseline.

Setup: Qwen2.5-0.5B-Instruct, 100 planted `line <name>: REGISTER_CONTENT is <value>` facts/prompt ($t_p = 1709$), $r_k \in \{0.25, 0.5, 0.75\}$, $r_w = 0.5$, `low_bits`(N)$=2$, `high_precision_native`$=$True (the "important" bucket kept bit-exact in native precision, never quantized), $n = 50$ prompts/point, paired across the sweep by a fixed seed.

| condition | KV size (% of uncompressed) | accuracy |
|---|---|---|
| uncompressed baseline | 100% | **30.0%** (15/50) |
| $k = 0.25\, t_p$ | 34.4% | 0.0% |
| $k = 0.50\, t_p$ | 56.1% | 0.0% |
| $k = 0.75\, t_p$ | 77.8% | 2.0% |

![KV compression sweep](../kv_compression_sweep.png)

**MiKV collapses accuracy to ~0% at every budget tested, even retaining 78% of the cache.** Since `high_precision_native=True`, the important bucket is *never quantized* — isolating the failure to **selection**, not representation: a perfectly lossless bucket doesn't help if it holds the wrong tokens. Confirmed by measuring needle survival directly: the queried fact's tokens stayed in the important set only ~0–40% of the time per (layer, kv head) — no better than the ~50% from *random* eviction — with several (layer, head) pairs evicting them entirely (one such pair suffices to poison the answer through the residual stream).

**Interpretation:** MiKV's premise — that cumulative attention concentrates on salient content, so an H2O-style score flags what's worth protecting — holds poorly at 0.5B scale. The query arrives only at the very end of the prompt, so during prefill nothing distinguishes the eventually-queried fact from 99 distractors; the importance score is close to uninformative for this task, and compression degrades it far more sharply than the paper's curves (which used 7B+ instruction-tuned models).

**Not a memory artifact.** An earlier run showed CUDA OOM-retry warnings and peaked near 3.9 GB in `nvtop`, raising the question of whether PyTorch was silently evicting cache and confounding the results. It was not: those `[W…]` warnings are the caching allocator defragmenting and retrying, which only reclaims *already-freed* blocks and never touches live tensors — a genuine exhaustion raises `OutOfMemoryError` and crashes instead. Instrumenting the run confirms it (`peak_allocated` ≈ 1.5 GB, `reserved` ≈ 1.07 GB, of 4 GB — ~38% utilization); the `nvtop` figure was the allocator's cached pool plus other GPU processes, not live data. Calling `torch.cuda.empty_cache()` per sample eliminated the warnings entirely (0 vs. dozens) with **no change to the conclusion**.

## Available Compute (My System)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM (driver 595.80, CUDA 13.0) |
| CPU | Intel Core i5-12500H, 16 threads |
| RAM | 15 GB |
| Disk | 154 GB total, ~25 GB free |
| OS | Ubuntu 20.04.6 LTS, kernel 5.15.0-139-generic |
| Python | 3.12.2 |
| torch | 2.13.0+cu130 |
| transformers | 5.15.1 |

4 GB of VRAM is the binding constraint on experiment scale: `attn_implementation="eager"` (required so MiKV can read real softmax attention weights for scoring) materializes the full $t_p \times t_p$ attention-weight matrix per layer, so its memory is $O(t_p^2)$. This capped `num_records` at ~100–150 for the 0.5B model.

**Llama-2-7b-chat does not fit this GPU.** Weights alone are ~13 GB at fp16 versus 4 GB of VRAM — before the KV cache or the $O(t_p^2)$ eager attention buffers. The `__main__` sweep parameters were sized against the 0.5B model and have been reduced (`num_samples=40`, `num_records=40`) but **not re-validated** at 7B; the run needs a larger GPU or offloading, and current settings should be treated as untested rather than tuned.

