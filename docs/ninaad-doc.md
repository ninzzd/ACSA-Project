# Ninaad's Project Documentation

## Software Implementation of MiKV

**Paper Link**: [No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization](https://arxiv.org/abs/2402.18096)

### Model Details

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen2ForCausalLM` |
| Parameters | ~494M |
| Layers | 24 |
| Attention heads (Q) | 14 |
| KV heads | 2 (GQA, 7 query heads share each kv head) |
| Hidden size | 896 |
| Head dim | 64 |
| MLP intermediate size | 4864 |
| Vocab size | 151,936 |
| Tied embeddings | yes |
| Max context | 32,768 tokens |
| Dtype (this setup) | bfloat16 on CUDA, float32 CPU fallback |
| Attention impl | `eager` (only backend returning real softmax attention weights, needed for scoring) |
| Prompt format | Qwen ChatML via `tokenizer.apply_chat_template` |

The **-Instruct** checkpoint is required, not a preference: the base model doesn't recognize any chat/turn structure. Given the paper's literal Llama-2-chat `[INST] <<SYS>>` prompt it emits EOS immediately (verified directly), treating `[/INST]` as arbitrary text — so it can't be elicited into answering at all, and the benchmark returns 0% for reasons unrelated to compression. Architecture and config are otherwise identical to the base model.

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

**Score:** raw cumulative sum of attention received — H2O's heavy-hitter criterion (Zhang et al. 2023, Algorithm 1: $F_{\text{score}} := \sum_s o_s$), no normalization by token age or step count:

$$
a_j \mathrel{+}= \textstyle\sum_i P_{i,j}
$$

**Importance set** $S$ (recomputed every step, size always $k$ once $t > k$):

```
importance_set(a, t, k, w, k_H):
    if t <= k: return {0, ..., t-1}            # cache not yet full: everything important
    S = {t-w, ..., t-1}                        # recency window, unconditionally in
    if k_H > 0:
        S += top_kH(a[j] for j not in S)        # best (k - w) raw scorers among the rest
    return S                                    # |S| == k
```

> **Fixed:** an earlier version ranked on `a[j] / age[j]` (age-decayed) instead of raw `a[j]`, reasoned as avoiding a bias toward tokens that had simply been in the cache longer. Checked directly against H2O (Zhang et al. 2023, Algorithm 1) and MiKV (which explicitly delegates its importance criterion to H2O rather than defining its own): H2O's score is the raw cumulative sum, undecayed. The decay wasn't a deviation with a documented justification from the source papers — it also isn't neutral, since it systematically discounts earlier positions regardless of actual attention received, which is a confound the papers' own criterion doesn't have. Removed to match spec.

**Quantization:**
- prefill: $j \in S \to$ HIGH, $j \notin S \to$ LOW bits ($N$)
- decode (steady state $t = k+1$): recompute $S$; the one position that falls out of $S$ is demoted to LOW bits — sticky, never promoted back.
- HIGH means either the model's native bf16/fp16 (untouched) or quantized to `high_bits`, toggled by the global `high_precision_native` flag (default: native).

### torch / transformers Integration

Qwen2.5-0.5B-Instruct loaded via `AutoModelForCausalLM` with `attn_implementation="eager"` (only backend that returns real softmax attention weights, needed for scoring). Each layer's `Qwen2Attention.forward` is monkey-patched (`types.MethodType`) to wrap the stock `attention_interface` call: balance $Q,K$ by $b$ post-RoPE/pre-cache (a query is never cached, so this can't be done after the fact), write through to `DynamicCache.layers[i].keys/values` with a fake N-bit round-trip quantizer, then score and re-demote after attention runs. All state ($b$, $a$, $S$) is shaped `[batch, num_kv_heads, t]` per layer, so masking/`topk` apply per kv-head automatically via broadcasting — no explicit per-head loop.

### Experimental Findings: Line Retrieval vs. KV Compression

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

4 GB of VRAM is the binding constraint on experiment scale: `attn_implementation="eager"` (required so MiKV can read real softmax attention weights for scoring) materializes the full $t_p \times t_p$ attention-weight matrix per layer, so its memory is $O(t_p^2)$. This caps `num_records` in the Line Retrieval sweep (`scripts/mikv.py`, `__main__`) at ~100–150 in practice — 1000 records ($t_p \sim 15$–$20\text{k}$) would need well over 8 GB just for one layer's attention weights.

