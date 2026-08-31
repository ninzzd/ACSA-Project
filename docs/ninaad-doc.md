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

### Importance Policy

**Parameters:** budget $k$, recency window $w \le k$, top-score slots $k_H = k - w$.
Resolved once from the prompt length $t_p$: $k = \lfloor r_k \cdot t_p \rfloor,\ w = \lfloor r_w \cdot k \rfloor$.

**Score:**

- *Prefill* — raw cumulative sum of attention received, H2O's heavy-hitter criterion (Zhang et al. 2023, Algorithm 1: $F_{\text{score}} := \sum_s o_s$), no normalization by token age or step count:

$$
a_j \mathrel{+}= \textstyle\sum_i P_{i,j}
$$

- *Decode* — each step adds the new query row's attention, then **age-decays** every position's running score by dividing it by that position's distance to the current token:

$$
a_i \leftarrow \dfrac{a_i + P_i}{\,t - i\,}, \qquad i \in \{0, \dots, t-2\}
$$

where $t$ is the index of the newest token ($=$ seq_len $-\,1$). The newest token ($i = t-1$, divisor $0$) is left as the plain running sum. This runs every decode step, so it **compounds**: a position's prefill-era mass decays geometrically once decoding starts, and $S$'s top-$k_H$ slots track *recently*-attended tokens rather than all-time heavy hitters. Confined to decode — prefill still accumulates the raw undecayed sum.

**Importance set** $S$ (recomputed every step, size always $k$ once $t > k$):

```
importance_set(a, t, k, w, k_H):
    if t <= k: return {0, ..., t-1}            # cache not yet full: everything important
    S = {t-w, ..., t-1}                        # recency window, unconditionally in
    if k_H > 0:
        S += top_kH(a[j] for j not in S)        # best (k - w) raw scorers among the rest
    return S                                    # |S| == k
```

> **Deliberate deviation from H2O/MiKV spec — decode-time age decay.** H2O's and MiKV's criterion is the *raw, undecayed* cumulative sum (MiKV explicitly delegates its importance criterion to H2O rather than defining its own). An earlier version of this code deviated by *ranking* on `a[j] / age[j]` at selection time; that was reverted to match spec. The decay is now re-introduced, but differently: at small scale the raw sum lets prefill-era mass dominate selection — the earliest positions accrue attention over the entire prompt before the query token is even seen, a structural head start unrelated to current relevance. Dividing the running score by distance-to-now ($t - i$) discounts that. Two differences from the reverted version: (1) it mutates the scoreboard `a` itself, recursively, every decode step (so the discount compounds), rather than being a one-shot `a / age` quotient computed only at ranking time; (2) it applies only during decode — `score_prefill` still accumulates the raw sum. It remains a deviation from the source papers, not a spec-conformant change.

**Quantization:**
- prefill: $j \in S \to$ HIGH, $j \notin S \to$ LOW bits ($N$)
- decode (steady state $t = k+1$): recompute $S$; the one position that falls out of $S$ is demoted to LOW bits — sticky, never promoted back.
- HIGH means either the model's native fp16 (untouched) or quantized to `high_bits`, toggled by the global `high_precision_native` flag (default: native).

`quantize_kv` is a **fake** quantizer: it round-trips through an affine $N$-bit grid and dequantizes straight back to fp16, so it measures quantization's effect on *accuracy* without shrinking the real `DynamicCache`. Two changes from the first version:

- **Group-wise scales.** Min/max are now taken per group of `group_size` consecutive channels (default `head_dim / 2`, i.e. 2 groups of 64) instead of once per whole 128-d vector. One outlier channel no longer stretches the scale for the entire head — the dominant source of error at $N = 2$.
- **`bits >= 16` short-circuits** to the identity, so a 16-bit "LOW" bucket is a true no-op rather than a lossy round-trip through a 65535-level grid.

### Running

```bash
python scripts/mikv.py
```

No CLI args — sweep parameters (`budget_ratios`, `num_samples`, `num_records`) are set in the `__main__` block ([mikv.py:961-963](../scripts/mikv.py#L961-L963)). Logs to `mikv_run_<timestamp>.log` (stdout+stderr tee'd) and saves the plot to `kv_compression_sweep.png` in the working directory.

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

