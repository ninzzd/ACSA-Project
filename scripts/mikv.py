"""
KV cache quantization policy for autoregressive inference on
Llama-2-7b-chat-hf, implementing MiKV's channel-balanced,
budget-constrained mixed-precision cache:

- Prefill: derive a per-(layer, kv head, channel) balancer b from the
  prompt's Q/K statistics, run full causal attention to accumulate an
  importance score per prompt position, then split the budget k into a
  recency window of size w and a top-scoring set of size k - w, and write
  each position's K/V at HIGH or LOW precision accordingly.
- Decode: attend over the entire mixed-precision cache with the frozen
  balancer, accumulate importance scores for every position (including
  the one just written), and re-derive S each step; whatever falls out of
  S is demoted to LOW precision -- sticky, never restored.

The scoreboard is a plain running sum, s(i+1) = s(i) + a. The age decay is
stateless and exists only to pick the victim: s'(i+1) = s(i+1) / age is formed
on the comparison path, the demoted token is argmin s', and s' is never stored.
See SCORE_DECAY_APPLICATIONS.

The budget k comes from a ratio r under one of two modes (BUDGET_MODES):
"fixed_length" resolves k = r * t_p once from the prompt length and freezes
it, so the high-precision set is a constant *number* of tokens; the newer
"fixed_ratio" re-derives k = r * t every step from the current total token
count (prefill + decode), so the constant is instead the *fraction* of the
cache held at high precision. r itself never changes in either mode. The
sweep in `sweep_kv_compression` runs both by default -- at equal r they are
not equal-footprint configurations, so each row is sized against its own k.

Q/K must be intercepted post-RoPE, pre-cache, inside each layer's
attention forward: the balancer has to divide the *query actually used to
produce this step's output*, and a query is never cached, so unlike the
old single-lowest-score decode policy this can't be done by mutating the
DynamicCache after the fact. `LlamaAttention.forward` is therefore
monkey-patched per layer to inject balancing, scoring and quantize-on-write
around the same `attention_interface` call the stock implementation uses
(the rotary/eager-attention helper functions used here happen to be
imported from transformers' Qwen2 module, but are byte-identical to
Llama's own -- both are copied from the same upstream implementation).

Uses the -chat checkpoint, not the base model: the base model doesn't
recognize any chat/turn structure and ships no `chat_template`, so
`apply_chat_template()` can't even be called on it, let alone elicit a
system+user-style response. -chat is fine-tuned on Llama-2's own
[INST]/<<SYS>> template and responds properly to one via
`tokenizer.apply_chat_template`, same architecture and config as the base
model otherwise (`_kv_cache_shape`, head counts, etc. all still apply
unchanged). Unlike Qwen2.5, Llama-2-7b uses plain multi-head attention
(no GQA: num_key_value_heads == num_attention_heads), so `num_kv_groups`
is always 1 here -- the code handles that as a special case of the
general GQA path, not a separate branch.
"""

import csv
import datetime
import itertools
import math
import os
import random
import re
import sys
import types
from dataclasses import dataclass

# hf-xet (huggingface_hub's accelerated download backend) has been observed
# to segfault -- no Python traceback, just a raw crash -- partway through a
# first-time model weight download on this machine. Falling back to the
# plain HTTP downloader is slower but doesn't crash. Must be set before
# `transformers`/`huggingface_hub` are imported, since it's read once at
# import time. `setdefault` so an explicit shell-level override still wins.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# torch/transformers MUST be imported before matplotlib: importing
# matplotlib.pyplot first and transformers second segfaults on this
# machine (a native-library symbol conflict between the two, order-
# dependent -- whichever loads first wins the conflicting symbol). No
# Python traceback, just a raw crash, so this ordering is load-bearing,
# not stylistic -- don't reorder these two blocks.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

import matplotlib

matplotlib.use("Agg")  # headless-safe: write figures to disk, never open a GUI window
import matplotlib.pyplot as plt

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # chat-tuned checkpoint -- the base
# Llama-2-7b-hf ships no chat_template, so apply_chat_template() below would raise;
# gated -- requires an accepted license + `huggingface-cli login` (or HF_TOKEN) with
# access approved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

DEFAULT_BUDGET_RATIO = 0.5  # r: importance budget ratio. k = floor(r * t_p) under the
# "fixed_length" budget mode, k = floor(r * t) under "fixed_ratio" -- see BUDGET_MODES.
DEFAULT_WINDOW_RATIO = 0.5  # w: recency window, w = floor(WINDOW_RATIO * k)  (default: w = k/2)
DEFAULT_HIGH_BITS = 8        # bit-width the "important" bucket is quantized to when not left native
DEFAULT_LOW_BITS = 2         # N: bit-width for the low-precision ("evicted") bucket
DEFAULT_HIGH_PRECISION_NATIVE = True  # important tokens stay at the model's native fp16
# (untouched) when True; quantized to DEFAULT_HIGH_BITS instead when False.

# Channel-balancer variants (see MiKVPolicy.balance_prefill_* for the math):
#   "paper" -- b = sqrt(q_max / k_max), the balancer as specified in MiKV.
#   "pow2"  -- b snapped to the nearest power of two, exponent-only. Applying it
#              is then an exact exponent add/subtract in fp16 rather than a real
#              multiply, which is what a hardware implementation wants; the cost
#              is that b is quantized to the 2^n grid instead of exact.
BALANCE_SCHEMES = ("paper", "pow2")
DEFAULT_BALANCE_SCHEME = "paper"
_BALANCE_SCHEME_LABELS = {"paper": "MiKV (sqrt)", "pow2": "hardware (pow-2)"}

# How the steady-state importance budget k is resolved from the budget ratio r:
#   "fixed_length" -- k = floor(r * t_p), resolved once from the prompt length at
#                     prefill and frozen for the whole sequence. The high-precision
#                     set is a fixed *number* of tokens, so as decoding extends the
#                     cache the effective compression keeps improving and the
#                     fraction of the context kept at high precision keeps shrinking.
#   "fixed_ratio"  -- k = floor(r * t), re-derived every step from the *current*
#                     total token count (prefill + decode so far). r is still
#                     constant; what stays constant with it is the *fraction* of the
#                     cache held at high precision, so k grows as the sequence does.
# w (the recency window) and k_H = k - w follow k in both modes, via `budget_for`.
# Precision of the importance scoreboard `a` -- the per-position running sum of
# attention received, which is what `_importance_set` ranks on. The attention
# weights feeding it are fp16 (transformers' eager attention softmaxes in fp32 but
# casts back to the query dtype), so by default the accumulator is fp16 too:
#   "native" -- accumulate at the model's dtype (fp16 on CUDA). The original
#               behaviour, and the default: no extra error beyond fp16's own.
#   "fp32"   -- accumulate in fp32. A reference point, not a hardware proposal:
#               it says how much the fp16 accumulator was already costing, which
#               is the baseline any quantized variant has to be read against.
#   "quant"  -- fake-quantize the scoreboard to `score_bits` after every update,
#               modelling an N-bit scoreboard register file. See `quantize_scores`
#               for the granularity and for the stagnation effect this introduces.
#   "fixed"  -- the IPU's actual score path: the softmax delta is cast from fp16 to
#               a HW_DELTA_BITS fixed-point word at ingress and accumulated in a
#               saturating HW_SCORE_WORD_BITS fixed-point register. Unlike "quant"
#               this has a *static* scale, so it is exact for large values and
#               flushes small ones to zero rather than rescaling to fit them.
SCORE_SCHEMES = ("native", "fp32", "quant", "fixed")
DEFAULT_SCORE_SCHEME = "native"
DEFAULT_SCORE_BITS = 8  # width of the scoreboard under score_scheme="quant"
_SCORE_SCHEME_LABELS = {
    "native": "score fp16",
    "fp32": "score fp32",
    "quant": "score int",
}

# Precision of the decode-time age-decay factor 1/(t - i) applied to the
# scoreboard (see MiKVPolicy.score_decode). Quantizing the *scoreboard* leaves
# this multiplier untouched, so a "quantized" datapath was still getting an exact
# fp32 reciprocal for free; these variants price it:
#   "exact" -- the fp32 divide the code has always done. Default: preserves
#              existing behaviour bit-for-bit (a divide, not a reciprocal-multiply,
#              which would round differently).
#   "quant" -- the reciprocal 1/(t - i) fake-quantized to `score_bits`, modelling
#              a finite-width decay LUT.
#   "pow2"  -- the reciprocal snapped to a power of two, so decay is an exponent
#              subtract rather than a multiply. Same hardware argument as the
#              "pow2" channel balancer.
SCORE_DECAY_SCHEMES = ("exact", "quant", "pow2", "lut")
DEFAULT_SCORE_DECAY_SCHEME = "exact"

# WHERE the age decay is applied -- an algorithmic choice, not a precision one:
#   "ranking"     -- the default and the real policy. The scoreboard holds the raw
#                    cumulative sum (H2O/MiKV's own criterion) and the decay is a
#                    static, stateless transform applied only to find the token to
#                    demote:
#                        s(i+1)  = s(i) + a          <- stored, never decayed
#                        s'(i+1) = s(i+1) / age      <- comparison value only
#                        victim  = argmin s'(i+1)    over the eligible positions
#                    Nothing about the decay is carried between steps, so it cannot
#                    compound. This is also what the IPU microarchitecture does --
#                    ipu_accum writes {tier, accum.score_out} back to the score SRAM
#                    undecayed, while ipu_age_lut sits on the branch to ipu_min_tree
#                    feeding cmp_val only.
#   "compounding" -- legacy: the decayed value is written back into the scoreboard,
#                    so the discount re-applies to an already-discounted score every
#                    step and decays geometrically. Kept only to reproduce sweeps run
#                    before this changed; it is a different policy, not a variant of
#                    the one above, and the accumulator width requirement differs
#                    (a raw sum grows monotonically toward saturation, a compounded
#                    one never does).
SCORE_DECAY_APPLICATIONS = ("ranking", "compounding")
DEFAULT_SCORE_DECAY_APPLICATION = "ranking"

# --- IPU score-path parameters, mirroring the microarchitecture ---
# Used only by score_scheme="fixed" and score_decay_scheme="lut", so the software
# sweep can predict the RTL rather than a generic affine quantizer. Note the score
# path is FIXED point with a static scale, not the data-dependent min/max affine
# quantizer `quantize_scores` implements -- the two are different error models.
# The score word is WORD_BITS wide, of which TIER_BITS encode the 3-state tier
# (HIGH / LOW / MIGRATING); what is left is the numeric field, l. So l = 30 for a
# 32-bit word and l = 14 for a 16-bit one.
HW_SCORE_WORD_BITS = 32
HW_SCORE_TIER_BITS = 2                                        # HIGH / LOW / MIGRATING
HW_SCORE_LENGTH_BITS = HW_SCORE_WORD_BITS - HW_SCORE_TIER_BITS  # l = 30
HW_SCORE_FRAC_BITS = 16                                       # f
HW_SCORE_SIGNED = True     # signed: the format is (1, l, f)
HW_DELTA_BITS = 17         # DELTA_WIDTH: the ingress softmax delta, fp16 -> fixed point
HW_DELTA_FRAC_BITS = 16
HW_AGE_LUT_ENTRIES = 512   # ipu_age_lut ROM depth; ages past it clamp to the last entry
HW_AGE_LUT_FRAC_BITS = 16  # 512 x 16b ROM holding 1/age as Q0.16


def _score_tag(
    score_scheme: str,
    score_bits: int = DEFAULT_SCORE_BITS,
    score_decay_scheme: str = DEFAULT_SCORE_DECAY_SCHEME,
    score_decay_application: str = DEFAULT_SCORE_DECAY_APPLICATION,
    score_length_bits: int = HW_SCORE_LENGTH_BITS,
    score_frac_bits: int = HW_SCORE_FRAC_BITS,
    score_signed: bool = HW_SCORE_SIGNED,
) -> str:
    """Compact identifier for a scoreboard-precision setting: "native", "fp32",
    "quant8", "quant8+decaypow2". Used in filenames, labels and log lines. The bit
    width only disambiguates anything under "quant", and the decay scheme only when
    it is not the exact default, so each is folded in only when it distinguishes
    something -- which keeps existing filenames stable."""
    if score_scheme == "quant":
        tag = f"quant{score_bits}"
    elif score_scheme == "fixed":
        # l and f decide what this configuration actually is, so they belong in the
        # identifier -- "fixed" alone would collide across incomparable runs.
        tag = f"fixed{score_length_bits}.{score_frac_bits}" + ("" if score_signed else "u")
    else:
        tag = score_scheme
    if score_decay_scheme != DEFAULT_SCORE_DECAY_SCHEME:
        tag += f"+decay{score_decay_scheme}"
    if score_decay_application != DEFAULT_SCORE_DECAY_APPLICATION:
        tag += f"+{score_decay_application}"
    return tag


BUDGET_MODES = ("fixed_length", "fixed_ratio")
DEFAULT_BUDGET_MODE = "fixed_length"
_BUDGET_MODE_LABELS = {
    "fixed_length": "fixed length (k = r*t_p)",
    "fixed_ratio": "fixed ratio (k = r*t)",
}


def load_model(model_name: str = MODEL_NAME):
    print(f"[load_model] device={DEVICE} dtype={DTYPE}", flush=True)

    print(f"[load_model] fetching tokenizer for {model_name} (downloads on first run)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("[load_model] tokenizer ready", flush=True)

    print(f"[load_model] fetching model weights for {model_name} (downloads ~13GB on first run)...", flush=True)
    # eager attention is required to get real softmaxed attention weights
    # back out of the forward pass (output_attentions=True is not
    # supported by the sdpa/flash-attention backends).
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, attn_implementation="eager"
    )
    print(f"[load_model] weights loaded, moving model to {DEVICE}...", flush=True)
    model.to(DEVICE)
    model.eval()  # disables dropout etc.; inference-only forward passes
    print("[load_model] model ready", flush=True)
    return model, tokenizer


def quantize_kv(tensor: torch.Tensor, bits: int = 2, group_size: int | None = None) -> torch.Tensor:
    """
    Fake-quantize K/V vectors: round-trip them through a `bits`-wide affine
    quantizer and dequantize back to the input dtype. Quantization is
    per-group along the last dim -- `group_size` consecutive channels share
    one (min, max) pair; `group_size=None` splits `head_dim` into two
    groups. Does not include the channel biasing factor b.

    tensor: (..., head_dim)
    """
    if bits >= 16:
        return tensor
    d = tensor.shape[-1]
    g = group_size or d // 2
    shp = tensor.shape
    t = tensor.reshape(*shp[:-1], d // g, g)
    qmax = 2**bits - 1
    t_min, t_max = t.amin(-1, keepdim=True), t.amax(-1, keepdim=True)
    scale = (t_max - t_min).clamp(min=1e-8) / qmax
    return (torch.round((t - t_min) / scale) * scale + t_min).reshape(shp).to(tensor.dtype)


def quantize_scores(a: torch.Tensor, bits: int = DEFAULT_SCORE_BITS) -> torch.Tensor:
    """
    Fake-quantize the importance scoreboard: round-trip `a` through a
    `bits`-wide affine quantizer and back to its input dtype.

    a: [batch, num_kv_heads, t] -- one (min, max) pair per (batch, kv head)
    across all t positions, i.e. one scale per head's whole scoreboard,
    re-derived at every update. That granularity is the point: the scoreboard
    is ranked *within* a head (`_importance_set` topk's along t), so the
    dynamic range that matters is the head's own spread of scores, and a
    coarser or finer grouping would measure something else.

    Two consequences worth expecting when reading results from this:
    (1) it is lossy on *rank*, not just on value -- positions whose scores fall
        in the same bin become ties, and `topk` breaks ties arbitrarily, so
        which token survives becomes partly a quantization artifact;
    (2) at small `bits` the running sum can stagnate: once one position's score
        dominates the head's max, an increment smaller than half a bin rounds
        away and that position's score stops growing at all. Combined with
        sticky demotion, an early ranking can freeze in place. This is a real
        property of an N-bit accumulator, not a bug in the model -- it's the
        thing this variant exists to expose.
    """
    if bits >= 16:
        return a
    return quantize_kv(a, bits=bits, group_size=a.shape[-1])


def fixed_point_range(length_bits: int, frac_bits: int, signed: bool = True) -> tuple[float, float]:
    """
    (min, max) representable by `(sign, length_bits, frac_bits)`, so a caller (or
    a log line) can see the actual headroom rather than infer it. Integer bits are
    derived, l - sign - f, never passed in -- see `quantize_fixed_point`.
    """
    int_bits = length_bits - (1 if signed else 0) - frac_bits
    if int_bits < 0:
        raise ValueError(
            f"frac_bits={frac_bits} leaves no room in length_bits={length_bits} "
            f"({'signed' if signed else 'unsigned'} needs frac_bits <= "
            f"{length_bits - (1 if signed else 0)})"
        )
    step = 2.0**-frac_bits
    hi = 2.0**int_bits - step
    return (-(2.0**int_bits), hi) if signed else (0.0, hi)


def quantize_fixed_point(
    x: torch.Tensor, length_bits: int, frac_bits: int, signed: bool = True
) -> torch.Tensor:
    """
    Round `x` onto a fixed-point grid in the standard DSP format
    `(sign, length, fraction)` -- signedness, total word length, fraction length,
    as in MATLAB/Simulink `fixdt(1, l, f)` -- saturating at both rails. Returns
    float64.

        (1, l, f):  l bits total, of which 1 is the sign and f are fractional,
                    leaving l - 1 - f integer bits as a *derived* quantity, not a
                    parameter. Resolution 2^-f, range [-2^(l-1-f), 2^(l-1-f) - 2^-f].

    The middle term is the length of the whole word, NOT a count of integer bits.

    l is the numeric field, i.e. the score word minus its tier bits:

        32-bit score word -> l = 30       16-bit score word -> l = 14

    The 2 tier bits encoding HIGH/LOW/MIGRATING (3 states) are not part of l and
    are not modelled here -- they carry no numeric weight.

    `signed` is offered because the score path is provably non-negative -- sums of
    softmax weights -- so an unsigned format buys a full extra bit of range for
    free. That is worth nothing at l = 30 and quite a lot at l = 14; see
    `fixed_point_range`.

    Saturating rather than wrapping, matching the `saturated?` flag
    ipu_accumulator carries out.

    float64, not float32, is the container on purpose: a 30-bit fixed-point value
    needs 30 bits of mantissa to round-trip, and fp32 has 24. Holding the model of
    a 30-bit register inside fp32 would quietly add a second, un-modelled rounding
    exactly at the LSB this scheme exists to study. Costs memory and speed, which
    is part of why this scheme is opt-in.
    """
    lo, hi = fixed_point_range(length_bits, frac_bits, signed)
    scale = float(2**frac_bits)
    return torch.clamp(torch.round(x.double() * scale), lo * scale, hi * scale) / scale


def age_lut_reciprocal(
    distance: torch.Tensor,
    entries: int = HW_AGE_LUT_ENTRIES,
    frac_bits: int = HW_AGE_LUT_FRAC_BITS,
) -> torch.Tensor:
    """
    ipu_age_lut's reciprocal ROM: 1/age read from an `entries`-deep table of
    Q0.`frac_bits` words, with ages past the table clamped to its last entry.

    Two effects that follow from the depth, both worth reading results against:
    (1) every position older than `entries` shares one decay factor (1/entries).
        Within that group the decay is a uniform scale, so their relative order is
        untouched -- an argmin over old tokens is decided purely by raw score. The
        clamp distorts only comparisons *across* the boundary, which makes it far
        more benign for a victim search than it would be for a stored value. At a
        4096-token context with a 512-entry table that group is 87% of positions.
    (2) Q0.16 cannot represent 1.0, so age=1 saturates to 65535/65536. Irrelevant
        while the recency window keeps age-1 tokens out of the eligible set.
    """
    idx = distance.clamp(min=1.0, max=float(entries))
    scale = float(2**frac_bits)
    # round-to-nearest into the ROM word, then saturate: Q0.frac holds [0, 1).
    return torch.clamp(torch.round(scale / idx.double()), 0.0, scale - 1.0) / scale


@dataclass
class _LayerState:
    b: torch.Tensor | None = None  # [num_kv_heads, head_dim], frozen after prefill
    a: torch.Tensor | None = None  # [batch, num_kv_heads, t] accumulated importance score
    demoted: torch.Tensor | None = None  # [batch, num_kv_heads, t] bool, sticky
    in_s: torch.Tensor | None = None  # [batch, num_kv_heads, t] bool, current importance set S


class MiKVPolicy:
    """
    Per-generation state machine for the MiKV cache policy, shared by every
    layer's patched attention forward. `phase` ("prefill" | "decode") and
    the frozen budget (k, w, k_H) live here since they're identical across
    layers/heads; the channel balancer b and per-position bookkeeping (a,
    demoted, S) are kept per layer in `_LayerState`.

    At steady state the cache holds t > k tokens; the importance set S is
    always exactly k tokens, split between the latest w tokens (recency
    window, unconditionally important) and the top-(k - w) scoring tokens
    among the remaining t - w. Each new decode step, S is recomputed over
    the grown cache and whatever falls out gets evicted (demoted to low
    precision).

    `budget_mode` decides whether k is the one frozen at prefill
    ("fixed_length") or re-derived from the current t each step
    ("fixed_ratio", where k grows with the cache so a constant fraction
    stays high-precision) -- see BUDGET_MODES. `self.k/w/k_H` always hold
    the prefill-time budget; `budget_for(t)` is the general form, and
    `_importance_set` picks between them.
    """

    def __init__(
        self,
        num_kv_heads: int,
        budget_ratio: float = DEFAULT_BUDGET_RATIO,
        window_ratio: float = DEFAULT_WINDOW_RATIO,
        high_bits: int = DEFAULT_HIGH_BITS,
        low_bits: int = DEFAULT_LOW_BITS,
        high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
        balance_scheme: str = DEFAULT_BALANCE_SCHEME,
        budget_mode: str = DEFAULT_BUDGET_MODE,
        score_scheme: str = DEFAULT_SCORE_SCHEME,
        score_bits: int = DEFAULT_SCORE_BITS,
        score_decay_scheme: str = DEFAULT_SCORE_DECAY_SCHEME,
        score_decay_application: str = DEFAULT_SCORE_DECAY_APPLICATION,
        score_length_bits: int = HW_SCORE_LENGTH_BITS,
        score_frac_bits: int = HW_SCORE_FRAC_BITS,
        score_signed: bool = HW_SCORE_SIGNED,
    ):
        if balance_scheme not in BALANCE_SCHEMES:
            raise ValueError(f"balance_scheme must be one of {BALANCE_SCHEMES}, got {balance_scheme!r}")
        if budget_mode not in BUDGET_MODES:
            raise ValueError(f"budget_mode must be one of {BUDGET_MODES}, got {budget_mode!r}")
        if score_scheme not in SCORE_SCHEMES:
            raise ValueError(f"score_scheme must be one of {SCORE_SCHEMES}, got {score_scheme!r}")
        if score_decay_scheme not in SCORE_DECAY_SCHEMES:
            raise ValueError(
                f"score_decay_scheme must be one of {SCORE_DECAY_SCHEMES}, got {score_decay_scheme!r}"
            )
        if score_decay_application not in SCORE_DECAY_APPLICATIONS:
            raise ValueError(
                f"score_decay_application must be one of {SCORE_DECAY_APPLICATIONS}, "
                f"got {score_decay_application!r}"
            )
        self.num_kv_heads = num_kv_heads
        self.budget_ratio = budget_ratio
        self.window_ratio = window_ratio
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.high_precision_native = high_precision_native
        self.balance_scheme = balance_scheme
        self.budget_mode = budget_mode
        self.score_scheme = score_scheme
        self.score_bits = score_bits
        self.score_decay_scheme = score_decay_scheme
        self.score_decay_application = score_decay_application
        # l and f for score_scheme="fixed". Validated eagerly even when the scheme
        # is not "fixed": an (l, f) pair with f too large to leave any integer bits
        # error worth catching at construction, not thousands of tokens into a run.
        fixed_point_range(score_length_bits, score_frac_bits, score_signed)
        self.score_length_bits = score_length_bits
        self.score_frac_bits = score_frac_bits
        self.score_signed = score_signed

        self.phase = "prefill"
        self.k = self.w = self.k_H = None
        self.layers: dict[int, _LayerState] = {}

    def budget_for(self, t: int) -> tuple[int, int, int]:
        """(k, w, k_H) for a cache holding `t` tokens -- the budget split itself,
        independent of which mode decides what `t` to pass in."""
        k = max(1, math.floor(self.budget_ratio * t))
        w = max(0, min(k, math.floor(self.window_ratio * k)))
        return k, w, k - w

    def start_prefill(self, prompt_len: int) -> None:
        self.phase = "prefill"
        # Both modes agree at prefill (t == t_p there); they diverge during decode,
        # where "fixed_length" keeps these frozen and "fixed_ratio" re-derives them
        # from the grown t. These stay set either way so the frozen budget is
        # available for reporting/sizing.
        self.k, self.w, self.k_H = self.budget_for(prompt_len)
        self.layers = {}

    def start_decode(self) -> None:
        self.phase = "decode"

    def _state(self, layer_idx: int) -> _LayerState:
        return self.layers.setdefault(layer_idx, _LayerState())

    # ---- channel balancing (called post-RoPE, pre-cache) ----

    def balance_prefill(self, layer_idx, query_states, key_states, num_kv_groups):
        """Dispatch to the configured balancer variant -- see BALANCE_SCHEMES."""
        if self.balance_scheme == "pow2":
            return self.balance_prefill_pow2(layer_idx, query_states, key_states, num_kv_groups)
        return self.balance_prefill_paper(layer_idx, query_states, key_states, num_kv_groups)

    def balance_prefill_paper(self, layer_idx, query_states, key_states, num_kv_groups):
        """b[c] = sqrt(max_i|Q[i,c]| / max_i|K[i,c]|), per (layer, kv head, channel)."""
        state = self._state(layer_idx)
        batch, _, t_p, head_dim = query_states.shape

        q_by_group = query_states.view(batch, self.num_kv_heads, num_kv_groups, t_p, head_dim)
        # amax/divide/sqrt in fp32: the ratio can be large enough to overflow fp16
        # before the sqrt pulls it back. Cast b back to the model dtype afterwards --
        # leaving it fp32 silently promotes Q/K in _apply_balance and blows up at
        # o_proj (fp16 weights) with a "mat1 and mat2 have the same dtype" error.
        q_max = q_by_group.abs().amax(dim=(0, 2, 3)).float().clamp(min=1e-8)
        k_max = key_states.abs().amax(dim=(0, 2)).float().clamp(min=1e-8)
        b = torch.sqrt(q_max / k_max).clamp(min=1e-4).to(query_states.dtype)
        state.b = b

        return self._apply_balance(state.b, query_states, key_states, num_kv_groups)

    def balance_prefill_pow2(self, layer_idx, query_states, key_states, num_kv_groups):
        """b[c] = 2^round((exp(max_i|Q[i,c]|) - exp(max_i|K[i,c]|)) / 2),
        per (layer, kv head, channel). Exponent-only: mantissas are discarded,
        so b is an exact power of two and applying it is lossless in fp16."""
        state = self._state(layer_idx)
        batch, _, t_p, head_dim = query_states.shape
        q_by_group = query_states.view(batch, self.num_kv_heads, num_kv_groups, t_p, head_dim)
        q_max = q_by_group.abs().amax(dim=(0, 2, 3))        # [num_kv_heads, head_dim]
        k_max = key_states.abs().amax(dim=(0, 2))           # [num_kv_heads, head_dim]

        # Guard against exact zeros before taking exponents.
        q_max = q_max.clamp(min=1e-8)
        k_max = k_max.clamp(min=1e-8)

        # frexp: x = mantissa * 2**exponent, mantissa in [0.5, 1)
        # => IEEE exponent = exponent - 1; the -1 cancels in the difference.
        _, q_exp = torch.frexp(q_max.float())               # int32
        _, k_exp = torch.frexp(k_max.float())
        d = q_exp - k_exp                                   # int32

        # round(d / 2) with ties going away from zero
        s = torch.where(d >= 0, (d + 1).div(2, rounding_mode='floor'),
                            -((-d + 1).div(2, rounding_mode='floor')))

        # b = 2**s, built exactly via ldexp (no pow, no rounding error)
        b = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s)
        b = b.to(query_states.dtype)

        state.b = b
        return self._apply_balance(state.b, query_states, key_states, num_kv_groups)

    def balance_decode(self, layer_idx, query_states, key_states, num_kv_groups):
        state = self._state(layer_idx)  # b is frozen; reused as-is
        return self._apply_balance(state.b, query_states, key_states, num_kv_groups)

    def _apply_balance(self, b, query_states, key_states, num_kv_groups):
        # Chokepoint cast: every balancer variant's b meets Q/K here, and a b left in
        # a wider dtype would promote them, propagating fp32 through attention until
        # o_proj rejects it against its fp16 weights. Cheap, and it keeps a new
        # scheme from reintroducing that failure.
        b = b.to(query_states.dtype)
        head_dim = b.shape[-1]
        num_q_heads = query_states.shape[1]
        b_k = b.view(1, self.num_kv_heads, 1, head_dim)
        b_q = b.repeat_interleave(num_kv_groups, dim=0).view(1, num_q_heads, 1, head_dim)
        return query_states / b_q, key_states * b_k

    # ---- quantize on write ----

    def _quantize_high(self, tensor: torch.Tensor) -> torch.Tensor:
        """The "important" bucket: native fp16 (untouched) if high_precision_native,
        else quantized to high_bits -- see DEFAULT_HIGH_PRECISION_NATIVE."""
        if self.high_precision_native:
            return tensor
        return quantize_kv(tensor, bits=self.high_bits)

    def write_new_token(self, layer_idx, cache, k_bal_new, v_new):
        """
        The token just appended to the cache is always the most recent
        position, hence always inside the recency window (size w), hence
        never demoted at the moment of creation: write it at HIGH precision.
        """
        state = self._state(layer_idx)
        layer = cache.layers[layer_idx]
        layer.keys[:, :, -1:, :] = self._quantize_high(k_bal_new)
        layer.values[:, :, -1:, :] = self._quantize_high(v_new)

        batch = k_bal_new.shape[0]
        false = torch.zeros(batch, self.num_kv_heads, 1, dtype=torch.bool, device=k_bal_new.device)
        # Match the scoreboard's own dtype, not the K/V dtype: under
        # score_scheme="fp32" `state.a` is fp32 while k_bal_new is fp16, and
        # torch.cat refuses the mismatch.
        score_dtype = state.a.dtype if state.a is not None else k_bal_new.dtype
        zero = torch.zeros(batch, self.num_kv_heads, 1, dtype=score_dtype, device=k_bal_new.device)
        state.a = zero if state.a is None else torch.cat([state.a, zero], dim=-1)
        state.demoted = false if state.demoted is None else torch.cat([state.demoted, false], dim=-1)
        state.in_s = false if state.in_s is None else torch.cat([state.in_s, false], dim=-1)

    def quantize_prefill(self, layer_idx, cache, k_bal, v):
        state = self._state(layer_idx)
        s = self._importance_set(state.a)  # [batch, num_kv_heads, t_p]
        state.in_s = s
        state.demoted = ~s

        high_mask = s.unsqueeze(-1)
        layer = cache.layers[layer_idx]
        layer.keys[:] = torch.where(high_mask, self._quantize_high(k_bal), quantize_kv(k_bal, bits=self.low_bits))
        layer.values[:] = torch.where(high_mask, self._quantize_high(v), quantize_kv(v, bits=self.low_bits))

    def demote_decode(self, layer_idx, cache):
        state = self._state(layer_idx)
        s = self._importance_set(state.a)  # [batch, num_kv_heads, t]

        # sticky: only demote, and only positions not already demoted
        fell_out = state.in_s & ~s & ~state.demoted
        if fell_out.any():
            layer = cache.layers[layer_idx]
            mask = fell_out.unsqueeze(-1)
            layer.keys[:] = torch.where(mask, quantize_kv(layer.keys, bits=self.low_bits), layer.keys)
            layer.values[:] = torch.where(mask, quantize_kv(layer.values, bits=self.low_bits), layer.values)
            state.demoted = state.demoted | fell_out
        state.in_s = s

    def _importance_set(self, a: torch.Tensor) -> torch.Tensor:
        """a: [batch, num_kv_heads, t] accumulated scores -> boolean membership mask, same shape."""
        batch, num_kv_heads, t = a.shape
        # "fixed_length": the budget frozen at prefill from t_p. "fixed_ratio": the
        # same ratio re-applied to the cache's *current* length, so k (and with it w
        # and k_H) grows as the sequence does.
        if self.budget_mode == "fixed_ratio":
            k, w, k_H = self.budget_for(t)
        else:
            k, w, k_H = self.k, self.w, self.k_H

        if t <= k:
            return torch.ones_like(a, dtype=torch.bool)

        s = torch.zeros_like(a, dtype=torch.bool)
        s[:, :, t - w :] = True  # recency window: the w most recent positions
        # s'(i+1) = s(i+1) / age -- computed here and nowhere else, fresh from the
        # stored raw sum every time, carrying nothing between steps. This is the
        # ipu_age_lut -> cmp_val -> ipu_min_tree path. Under the legacy
        # "compounding" setting `a` was already decayed on write, so it is ranked
        # as-is and this is a no-op.
        cmp_val = a if self.score_decay_application == "compounding" else (
            self._apply_decay(a, t).to(a.dtype)
        )
        if k_H > 0:
            # Ranked on s' = s / age, not on the raw sum: H2O's own criterion
            # (Zhang et al. 2023, Alg. 1: F_score := sum_s o_s) is the undecayed
            # cumulative sum, so normalizing by age is a deliberate deviation from
            # it -- earlier positions otherwise carry a structural head start from
            # having accrued attention over the whole prompt before the query token
            # was even seen. Keeping the division out of the stored score is what
            # makes it a deviation in the *criterion* only, with no state and no
            # compounding: `a` itself remains exactly H2O's F_score.
            # Selecting the top-k_H by s' and demoting what falls out is the same
            # decision as taking argmin s' over the eligible positions.
            candidates = cmp_val.masked_fill(s, float("-inf"))
            top = candidates.topk(k_H, dim=-1).indices  # top-(k - w) among the rest
            s.scatter_(-1, top, torch.ones_like(top, dtype=torch.bool))
        return s

    # ---- score accumulation ----

    def _store_score(self, a: torch.Tensor) -> torch.Tensor:
        """
        Chokepoint every scoreboard write goes through, applying the configured
        accumulator precision -- see SCORE_SCHEMES. Keeping it in one place means
        the scoreboard cannot be updated anywhere without paying the configured
        precision, which is the whole point of the knob.
        """
        if self.score_scheme == "fp32":
            return a.float()
        if self.score_scheme == "quant":
            return quantize_scores(a, bits=self.score_bits)
        if self.score_scheme == "fixed":
            return quantize_fixed_point(
                a, self.score_length_bits, self.score_frac_bits, self.score_signed
            )
        return a

    def _ingress_delta(self, p: torch.Tensor) -> torch.Tensor:
        """
        ipu_ingress: the fp16 softmax value cast to a HW_DELTA_BITS fixed-point
        word before it ever reaches the adder. Only "fixed" models this.

        Calling it a cast rather than a quantization is right for the values that
        decide anything -- above about 2^-6 the fp16 grid is coarser than Q_.16, so
        the conversion is exact -- but it is lossy in the tail: fp16 resolves far
        below 2^-16, and anything under half an LSB (2^-17 ~ 7.6e-6) becomes
        exactly zero. A token whose per-step attention never clears that threshold
        accumulates a score of exactly 0 forever and is indistinguishable from
        every other such token, so the victim search among them falls through to
        ipu_min_tree's lowest-index tie-break, i.e. oldest-first.
        """
        if self.score_scheme != "fixed":
            return p
        # The delta is a softmax value in [0, 1], never negative, so it uses an
        # unsigned field regardless of how the accumulator is configured.
        return quantize_fixed_point(p, HW_DELTA_BITS, HW_DELTA_FRAC_BITS, signed=False)

    def score_prefill(self, layer_idx, attn_weights, num_kv_groups):
        """a[j] = sum over all query positions i (and, for GQA, over the group's query heads)."""
        state = self._state(layer_idx)
        batch, _, t_p, _ = attn_weights.shape
        w = attn_weights.view(batch, self.num_kv_heads, num_kv_groups, t_p, t_p)
        # Under "fp32" the prefill reduction itself runs in fp32, not just its
        # result: summing t_p attention rows is where fp16 accumulation error is
        # largest, so casting only afterwards would hide exactly what this variant
        # is meant to measure.
        if self.score_scheme == "fixed":
            # Each softmax value passes ingress individually, so quantize elementwise
            # and only then reduce. Simplification: the reduction runs at full width
            # and saturates once at the end, where the hardware saturates per add --
            # they differ only for a score that actually reaches the rail.
            summed = self._ingress_delta(w).sum(dim=(2, 3))
        elif self.score_scheme == "fp32":
            summed = w.float().sum(dim=(2, 3))
        else:
            summed = w.sum(dim=(2, 3))
        state.a = self._store_score(summed)  # -> [batch, num_kv_heads, t_p]

    def score_decode(self, layer_idx, attn_weights, num_kv_groups):
        state = self._state(layer_idx)
        batch, _, _, t = attn_weights.shape
        p = attn_weights[:, :, 0, :].view(batch, self.num_kv_heads, num_kv_groups, t).sum(dim=2)

        # Age decay: a[i] <- (a[i] + attn[i]) / (t - i), where t = index of the
        # current (newest) token = seq_len - 1 and i is a position index. Raw
        # cumulative attention accrues from prefill onward, so without this the
        # earliest positions carry a structural head start unrelated to how
        # relevant they still are; dividing by distance-to-now discounts that.
        # The newest token has t - i = 0, which would blow up, so decay is
        # applied only to positions [0, t-1] and the last position is left as
        # the plain running sum. `_apply_decay` owns the arithmetic and its
        # precision.
        if self.score_scheme == "fp32":
            p = p.float()
        p = self._ingress_delta(p)
        # The post-add, pre-decay intermediate is stored at the accumulator's own
        # precision too, not just the final result: an N-bit scoreboard register is
        # N bits at every stage of the step, so quantizing only after the decay
        # would let the add carry full-precision information the hardware never
        # holds. This means "quant" pays two roundings per decode step (add, decay),
        # which is what the datapath actually does.
        # s(i+1) = s(i) + a. That is the whole decode-time state update: the
        # scoreboard is a plain running sum and the decay never touches it. The
        # decay is stateless and lives entirely on the compare path in
        # `_importance_set`, which is the only place it is needed.
        updated = self._store_score(state.a + p)
        if self.score_decay_application == "compounding":
            updated = self._store_score(self._apply_decay(updated, t).to(state.a.dtype))
        state.a = updated

    def _apply_decay(self, updated: torch.Tensor, t: int) -> torch.Tensor:
        """
        Apply the age decay 1/(t - i) at the precision set by
        `score_decay_scheme` -- see SCORE_DECAY_SCHEMES. Returns fp32.

        Build the distances in fp32, not the score dtype: fp16 only represents
        integers exactly up to 2048, above which spacing is 2 (4095 rounds to
        4096), so at a 4096-token context the distances for older positions come
        out wrong. clamp(min=1) covers the newest token, whose distance is 0.
        """
        last = t - 1
        distance = (last - torch.arange(t, device=updated.device, dtype=torch.float32)).clamp(min=1.0)

        if self.score_decay_scheme == "exact":
            # A true divide, deliberately not a multiply by the reciprocal: the two
            # round differently, and this branch has to reproduce the original
            # behaviour exactly so the other variants are measured against it.
            return updated.float() / distance

        if self.score_decay_scheme == "lut":
            # ipu_age_lut: a real ROM read, not a divide -- see age_lut_reciprocal.
            return updated.double() * age_lut_reciprocal(distance)

        factor = 1.0 / distance
        if self.score_decay_scheme == "pow2":
            # Nearest power of two, built exactly via ldexp -- no pow, no rounding
            # error in the factor itself, and applying it is an exponent subtract.
            exponent = torch.round(torch.log2(factor)).to(torch.int32)
            factor = torch.ldexp(torch.ones_like(factor), exponent)
        else:  # "quant": a finite-width decay LUT
            factor = quantize_scores(factor.view(1, 1, -1), bits=self.score_bits).view(-1)
        return updated.float() * factor


def _mikv_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Drop-in replacement for `Qwen2Attention.forward` that applies the MiKV
    policy around the same attention computation the original performs.
    """
    policy: MiKVPolicy = self._mikv_policy
    layer_idx = self.layer_idx
    num_kv_groups = self.num_key_value_groups

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if policy.phase == "prefill":
        query_states, key_states = policy.balance_prefill(layer_idx, query_states, key_states, num_kv_groups)
    else:
        query_states, key_states = policy.balance_decode(layer_idx, query_states, key_states, num_kv_groups)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)

    if policy.phase == "decode":
        # quantize-on-write the freshly appended position before it's used below,
        # so this step's attention already sees the mixed-precision cache
        policy.write_new_token(layer_idx, past_key_values, key_states[:, :, -1:, :], value_states[:, :, -1:, :])

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        # not all attention modules define this (e.g. Llama's doesn't); harmless to omit
        # under eager attention regardless, since eager_attention_forward never reads it
        # -- windowing there is baked into the attention_mask upstream, not applied here.
        sliding_window=getattr(self, "sliding_window", None),
        **kwargs,
    )

    if policy.phase == "prefill":
        policy.score_prefill(layer_idx, attn_weights, num_kv_groups)
        policy.quantize_prefill(layer_idx, past_key_values, key_states, value_states)
    else:
        policy.score_decode(layer_idx, attn_weights, num_kv_groups)
        policy.demote_decode(layer_idx, past_key_values)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _install_mikv_policy(model, policy: MiKVPolicy) -> None:
    for layer in model.model.layers:
        attn = layer.self_attn
        attn._mikv_policy = policy
        attn.forward = types.MethodType(_mikv_attention_forward, attn)


@torch.no_grad()  # disables autograd tracking; no backward pass needed for inference
def _generate_tokens_with_mikv(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    budget_ratio: float,
    window_ratio: float,
    high_bits: int,
    low_bits: int,
    high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
    balance_scheme: str = DEFAULT_BALANCE_SCHEME,
    budget_mode: str = DEFAULT_BUDGET_MODE,
    score_scheme: str = DEFAULT_SCORE_SCHEME,
    score_bits: int = DEFAULT_SCORE_BITS,
    score_decay_scheme: str = DEFAULT_SCORE_DECAY_SCHEME,
    score_decay_application: str = DEFAULT_SCORE_DECAY_APPLICATION,
    score_length_bits: int = HW_SCORE_LENGTH_BITS,
    score_frac_bits: int = HW_SCORE_FRAC_BITS,
    score_signed: bool = HW_SCORE_SIGNED,
) -> tuple[torch.Tensor, int]:
    """
    Manual autoregressive generation loop applying the MiKV KV cache
    policy: channel balancing and mixed-precision quantize-on-write during
    prefill, sticky demotion against a frozen budget during decode.

    Returns (generated token ids, prompt length) so callers can split the
    prompt from the newly generated continuation without re-tokenizing.
    """
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    policy = MiKVPolicy(
        num_kv_heads=num_kv_heads,
        budget_ratio=budget_ratio,
        window_ratio=window_ratio,
        high_bits=high_bits,
        low_bits=low_bits,
        high_precision_native=high_precision_native,
        balance_scheme=balance_scheme,
        budget_mode=budget_mode,
        score_scheme=score_scheme,
        score_bits=score_bits,
        score_decay_scheme=score_decay_scheme,
        score_decay_application=score_decay_application,
        score_length_bits=score_length_bits,
        score_frac_bits=score_frac_bits,
        score_signed=score_signed,
    )
    _install_mikv_policy(model, policy)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = input_ids
    cache = DynamicCache()

    # prefill: derive b and the frozen budget from the prompt, quantize on write
    policy.start_prefill(input_ids.shape[-1])
    outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    cache = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=-1)
    next_input_ids = next_token

    policy.start_decode()
    for _ in range(max_tokens - input_ids.shape[-1] - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break

        outputs = model(input_ids=next_input_ids, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        next_input_ids = next_token

    return generated, input_ids.shape[-1]


def generate_with_mikv(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 4096,
    budget_ratio: float = DEFAULT_BUDGET_RATIO,
    window_ratio: float = DEFAULT_WINDOW_RATIO,
    high_bits: int = DEFAULT_HIGH_BITS,
    low_bits: int = DEFAULT_LOW_BITS,
    high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
    balance_scheme: str = DEFAULT_BALANCE_SCHEME,
    budget_mode: str = DEFAULT_BUDGET_MODE,
    score_scheme: str = DEFAULT_SCORE_SCHEME,
    score_bits: int = DEFAULT_SCORE_BITS,
    score_decay_scheme: str = DEFAULT_SCORE_DECAY_SCHEME,
    score_decay_application: str = DEFAULT_SCORE_DECAY_APPLICATION,
    score_length_bits: int = HW_SCORE_LENGTH_BITS,
    score_frac_bits: int = HW_SCORE_FRAC_BITS,
    score_signed: bool = HW_SCORE_SIGNED,
) -> str:
    generated, _ = _generate_tokens_with_mikv(
        model,
        tokenizer,
        prompt,
        max_tokens,
        budget_ratio,
        window_ratio,
        high_bits,
        low_bits,
        high_precision_native,
        balance_scheme,
        budget_mode,
        score_scheme,
        score_bits,
        score_decay_scheme,
        score_decay_application,
        score_length_bits,
        score_frac_bits,
        score_signed,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# ---- Line Retrieval benchmark (Li et al., 2023a; MiKV Figure 3 / Appendix D.3) ----
#
# The paper measures MiKV's ability to preserve context under compression by
# planting random "line <name>: REGISTER_CONTENT is <value>" facts in the
# prompt and asking the model to retrieve one by name (their Figure 15). We
# reproduce the same system instruction + user message content, but deliver
# it via the tokenizer's own chat template (`apply_chat_template`) rather
# than hand-writing the literal Llama-2-chat [INST]/<<SYS>> syntax directly:
# that syntax is specific to Llama-2-chat's training format, and on a model
# that wasn't trained on it (verified directly against a base model: it
# treats "[/INST]" as arbitrary text and emits EOS immediately) it fails to
# elicit any real response at all. The chat template achieves the same
# semantic structure -- system instruction, then a user turn primed for a
# reply -- in whatever format the target model actually understands, so
# this same code path works unchanged across chat-tuned models.

_LINE_ADJECTIVES = [
    "billowy", "psychotic", "daffy", "exclusive", "enthusiastic", "handsome",
    "enchanting", "sour", "faithful", "picayune", "wee", "forgetful", "cagey",
    "childlike", "inconclusive", "delightful", "courageous", "scandalous",
    "mere", "annoyed", "brave", "curious", "distant", "eager", "fancy",
    "gentle", "hollow", "icy", "jolly", "keen",
]
_LINE_NOUNS = [
    "schizophrenic", "cement", "pancake", "bough", "navigation", "variability",
    "thrust", "hippopotamus", "tabernacle", "cookie", "basics", "struggle",
    "cargo", "polyp", "flesh", "location", "viability", "laboratory", "affect",
    "armrest", "canyon", "domino", "engine", "feather", "garden", "harbor",
    "island", "jacket", "kettle", "ladder",
]


@dataclass
class LineRetrievalSample:
    prompt: str
    target_line: str
    expected_value: int


def make_line_retrieval_sample(
    tokenizer, num_records: int = 20, value_digits: int = 5, rng: random.Random | None = None
) -> LineRetrievalSample:
    """Build one synthetic Line Retrieval prompt: `num_records` random facts, then a retrieval question."""
    rng = rng or random
    names = set()
    while len(names) < num_records:
        names.add(f"{rng.choice(_LINE_ADJECTIVES)}-{rng.choice(_LINE_NOUNS)}")
    names = list(names)
    rng.shuffle(names)

    low, high = 10 ** (value_digits - 1), 10**value_digits - 1
    values = {name: rng.randint(low, high) for name in names}
    target_line = rng.choice(names)

    lines = "\n".join(f"line {name}: REGISTER_CONTENT is <{values[name]}>" for name in names)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a record processing computer. Given a list of records, and a "
                "target <line index>, you retrieve the '<REGISTER_CONTENT>' number."
            ),
        },
        {
            "role": "user",
            "content": (
                "Below is a record of lines I want you to remember. Each line begins with "
                "'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the "
                "line as a numerical value. For each line index, memorize its corresponding "
                "<REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve "
                "the corresponding <REGISTER_CONTENT> of a certain line index. Now the "
                f"record start:\n\n{lines}\n\n"
                "Now the record is over. Tell me what is the <REGISTER_CONTENT> in line "
                f"{target_line}? I need the number."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return LineRetrievalSample(prompt=prompt, target_line=target_line, expected_value=values[target_line])


def _extract_number(text: str) -> int | None:
    m = re.search(r"<(-?\d+)>", text)          # prefer the paper's <...> format
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d{4,6}", text)      # else: first plausible register value
    return int(nums[0]) if nums else None


def run_line_retrieval_benchmark(
    model,
    tokenizer,
    num_samples: int = 20,
    num_records: int = 20,
    budget_ratio: float = DEFAULT_BUDGET_RATIO,
    window_ratio: float = DEFAULT_WINDOW_RATIO,
    high_bits: int = DEFAULT_HIGH_BITS,
    low_bits: int = DEFAULT_LOW_BITS,
    high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
    balance_scheme: str = DEFAULT_BALANCE_SCHEME,
    budget_mode: str = DEFAULT_BUDGET_MODE,
    score_scheme: str = DEFAULT_SCORE_SCHEME,
    score_bits: int = DEFAULT_SCORE_BITS,
    score_decay_scheme: str = DEFAULT_SCORE_DECAY_SCHEME,
    score_decay_application: str = DEFAULT_SCORE_DECAY_APPLICATION,
    score_length_bits: int = HW_SCORE_LENGTH_BITS,
    score_frac_bits: int = HW_SCORE_FRAC_BITS,
    score_signed: bool = HW_SCORE_SIGNED,
    max_tokens: int = 4096,
    seed: int = 0,
) -> float:
    """
    Line Retrieval accuracy under the MiKV cache policy (paper Figure 3b /
    Table 1): generate `num_samples` synthetic line-retrieval prompts,
    greedily decode each under the mixed-precision policy, and score
    whether the retrieved value matches the planted one.

    `max_tokens` is the total context budget (prompt + generation), not
    just the new-token count -- e.g. 4096 to simulate a 4K-context run,
    however long the planted-record prompt happens to be.

    `balance_scheme` selects the channel balancer -- see BALANCE_SCHEMES.
    `budget_mode` selects how k is resolved from `budget_ratio` -- see
    BUDGET_MODES. `score_scheme`/`score_bits` select the precision the
    importance scoreboard accumulates at -- see SCORE_SCHEMES -- and
    `score_decay_scheme` the precision of the age-decay factor applied to it
    -- see SCORE_DECAY_SCHEMES.
    """
    print(
        f"[mikv] starting {num_samples} samples: budget_mode={budget_mode} "
        f"balance_scheme={balance_scheme} score_scheme={score_scheme} "
        f"score_bits={score_bits if score_scheme == 'quant' else '-'} "
        f"score_decay={score_decay_scheme}/{score_decay_application} "
        f"fixed_point=({int(score_signed)},{score_length_bits},{score_frac_bits}) "
        f"budget_ratio={budget_ratio} "
        f"window_ratio={window_ratio} high_bits={high_bits} low_bits={low_bits} "
        f"high_precision_native={high_precision_native}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed)
    correct = 0
    for i in range(num_samples):
        sample = make_line_retrieval_sample(tokenizer, num_records=num_records, rng=rng)
        prompt_len = tokenizer(sample.prompt, return_tensors="pt").input_ids.shape[-1]
        generated, _ = _generate_tokens_with_mikv(
            model,
            tokenizer,
            sample.prompt,
            max_tokens=max_tokens,
            budget_ratio=budget_ratio,
            window_ratio=window_ratio,
            high_bits=high_bits,
            low_bits=low_bits,
            high_precision_native=high_precision_native,
            balance_scheme=balance_scheme,
            budget_mode=budget_mode,
            score_scheme=score_scheme,
            score_bits=score_bits,
            score_decay_scheme=score_decay_scheme,
            score_decay_application=score_decay_application,
            score_length_bits=score_length_bits,
            score_frac_bits=score_frac_bits,
            score_signed=score_signed,
        )
        continuation = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        predicted = _extract_number(continuation)
        is_correct = predicted == sample.expected_value
        correct += int(is_correct)
        print(
            f"[mikv {i + 1}/{num_samples}] target={sample.target_line} expected={sample.expected_value} "
            f"predicted={predicted} {'OK' if is_correct else 'WRONG'}",
            flush=True,
        )
        # Each sample re-installs a fresh MiKVPolicy and grows a new DynamicCache to a
        # slightly different length (num_records draws a random count of unique names),
        # so the CUDA caching allocator accumulates many differently-sized cached blocks
        # over hundreds of samples. On a ~3.3GB-free GPU that's enough fragmentation to
        # trigger (recoverable) OOM retries -- releasing the cache each sample keeps the
        # allocator's pool from fragmenting across runs of varying shape.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / num_samples
    print(
        f"[mikv] Line Retrieval accuracy "
        f"({budget_mode}/{balance_scheme}/"
        f"{_score_tag(score_scheme, score_bits, score_decay_scheme, score_decay_application, score_length_bits, score_frac_bits, score_signed)}): "
        f"{accuracy * 100:.1f}% "
        f"({correct}/{num_samples}) {_cuda_mem_str()}",
        flush=True,
    )
    return accuracy


# ---- KV cache size estimation & compression sweep ----
#
# The fake-quantizer in `quantize_kv` round-trips values through an N-bit
# affine quantizer but dequantizes back to the model's native dtype, so it
# never actually shrinks the DynamicCache's real memory footprint -- it's
# built to measure quantization's effect on accuracy, not to save memory.
# The functions below instead *compute* what the footprint would be if the
# high/low precision buckets were actually stored at their respective bit
# widths, so accuracy can be traded off against real compression.


def _kv_cache_shape(model) -> tuple[int, int, int]:
    """(num_layers, num_kv_heads, head_dim) sizing one cached token's K/V."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return num_layers, num_kv_heads, head_dim


def _cuda_mem_str() -> str:
    """
    Real (not estimated) CUDA memory in MB, for diagnosing GPU pressure:
    `allocated` is memory backing tensors actually live right now; `reserved`
    is the caching allocator's total pool held from the driver (allocated +
    cached-but-freed blocks kept around for reuse -- this is what nvidia-smi
    /nvtop report as this process's VRAM usage, so it reads higher than
    `allocated`); `peak_allocated` is the high-water mark for `allocated`
    since the last `torch.cuda.reset_peak_memory_stats()` call.
    """
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return f"cuda_mem: allocated={allocated:.0f}MB reserved={reserved:.0f}MB peak_allocated={peak:.0f}MB"


def kv_cache_size_bytes(
    model,
    seq_len: int,
    k: int | None = None,
    high_bits: int = DEFAULT_HIGH_BITS,
    low_bits: int = DEFAULT_LOW_BITS,
    high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
) -> float:
    """
    Estimate the KV cache footprint (bytes, K+V across all layers/heads)
    for a sequence of length `seq_len`.

    k=None: uncompressed baseline -- every position at the model's native
    16-bit dtype.
    k=<int>: MiKV steady state -- per `MiKVPolicy._importance_set`, the
    importance set holds exactly k positions once the cache is longer than
    k, so exactly min(seq_len, k) positions stay HIGH precision (native 16-bit if
    `high_precision_native`, else quantized to `high_bits` -- must match
    whatever the accuracy run actually used) and the remaining
    max(0, seq_len - k) are compressed to `low_bits` (N).

    k is the budget the run *ends up at* by `seq_len`, which differs by
    budget mode -- frozen at r*t_p under "fixed_length", grown to r*seq_len
    under "fixed_ratio". Use `resolve_budget_k` to get it rather than
    computing r*t_p by hand.
    """
    num_layers, num_kv_heads, head_dim = _kv_cache_shape(model)
    elements_per_token = num_layers * num_kv_heads * head_dim * 2  # *2 for K and V

    if k is None:
        return elements_per_token * seq_len * 16 / 8

    high_bit_width = 16 if high_precision_native else high_bits
    num_high = min(seq_len, k)
    num_low = max(0, seq_len - k)
    return elements_per_token * (num_high * high_bit_width / 8 + num_low * low_bits / 8)


def resolve_budget_k(budget_mode: str, budget_ratio: float, t_p: int, seq_len: int) -> int:
    """
    The steady-state budget k a run under `budget_mode` ends up holding once
    the cache has grown to `seq_len` tokens -- the number `kv_cache_size_bytes`
    must be sized against.

    "fixed_length": k was frozen at prefill from the prompt length, so it is
    floor(r * t_p) no matter how far decoding ran. "fixed_ratio": k tracks the
    cache, so at `seq_len` tokens it is floor(r * seq_len). Mirrors
    `MiKVPolicy.budget_for` (same floor, same max(1, ...) guard) -- the two must
    agree or the reported compression won't describe the run that produced the
    accuracy next to it.
    """
    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode must be one of {BUDGET_MODES}, got {budget_mode!r}")
    t = t_p if budget_mode == "fixed_length" else seq_len
    return max(1, math.floor(budget_ratio * t))


def run_line_retrieval_no_quant(
    model,
    tokenizer,
    num_samples: int = 20,
    num_records: int = 20,
    max_tokens: int = 4096,
    seed: int = 0,
) -> tuple[float, int, float, float]:
    """
    Baseline Line Retrieval pass with the stock HF generation loop -- no
    MiKV balancing, scoring, or quantization, so the KV cache stays at the
    model's native 16-bit dtype throughout. Establishes (a) t_p, the
    prompt length at the first prefill stage (constant across samples
    since every prompt plants the same `num_records` facts), and (b) the
    "before compression" KV cache size at the end of generation -- both
    needed to evaluate the MiKV sweep in `sweep_kv_compression`.

    Must run before any MiKV call (`generate_with_mikv`,
    `run_line_retrieval_benchmark`, ...): those monkey-patch each layer's
    attention forward in place and never restore the original, so once
    installed there is no unpatched model left to baseline against.

    `max_tokens` is the total context budget (prompt + generation), passed
    straight to `model.generate(..., max_length=max_tokens)` so it means
    the same thing here as it does in `run_line_retrieval_benchmark`.

    Returns (accuracy, t_p, avg_kv_cache_bytes_occupied, avg_seq_len).
    The last two describe what generation actually *occupied* (decoding
    stops at EOS, typically well short of `max_tokens`) and are reported
    as diagnostics only -- `sweep_kv_compression` sizes its compression
    ratio at the fixed provisioned context instead, since mixing an
    occupancy figure with a provisioned one overstates compression.
    """
    print(f"[no-quant] starting {num_samples} baseline samples (num_records={num_records})...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    correct = 0
    t_p = None
    total_bytes = 0.0
    total_seq_len = 0
    for i in range(num_samples):
        sample = make_line_retrieval_sample(tokenizer, num_records=num_records, rng=rng)
        input_ids = tokenizer(sample.prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_len = input_ids.shape[-1]
        if t_p is None:
            t_p = prompt_len
            print(f"[no-quant] t_p (prompt length) = {t_p} tokens", flush=True)

        output = model.generate(input_ids, max_length=max_tokens, do_sample=False, pad_token_id=pad_token_id)
        seq_len = output.shape[-1]
        total_seq_len += seq_len
        total_bytes += kv_cache_size_bytes(model, seq_len, k=None)

        continuation = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
        predicted = _extract_number(continuation)
        is_correct = predicted == sample.expected_value
        correct += int(is_correct)
        print(
            f"[no-quant {i + 1}/{num_samples}] target={sample.target_line} expected={sample.expected_value} "
            f"predicted={predicted} {'OK' if is_correct else 'WRONG'}",
            flush=True,
        )
        if torch.cuda.is_available():  # see the matching note in run_line_retrieval_benchmark
            torch.cuda.empty_cache()

    accuracy = correct / num_samples
    avg_bytes = total_bytes / num_samples
    print(
        f"[no-quant] baseline done: accuracy={accuracy * 100:.1f}% ({correct}/{num_samples}) "
        f"t_p={t_p} avg KV cache={avg_bytes / 1e6:.2f} MB {_cuda_mem_str()}",
        flush=True,
    )

    avg_seq_len = total_seq_len / num_samples
    return accuracy, t_p, avg_bytes, avg_seq_len


def sweep_kv_compression(
    model,
    tokenizer,
    budget_ratios: tuple[float, ...] = (0.25, 0.5, 0.75),
    num_samples: int = 20,
    num_records: int = 20,
    max_tokens: int = 4096,
    seed: int = 0,
    window_ratio: float = DEFAULT_WINDOW_RATIO,
    high_bits: int = DEFAULT_HIGH_BITS,
    low_bits: int = DEFAULT_LOW_BITS,
    high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
    balance_schemes: tuple[str, ...] = BALANCE_SCHEMES,
    budget_modes: tuple[str, ...] = BUDGET_MODES,
    # Defaults to the single native-fp16 scoreboard rather than all of
    # SCORE_SCHEMES: unlike the budget modes and balancers, sweeping this
    # multiplies an already-4x sweep again (x3 for native/fp32/quant), which is
    # hours of extra GPU time. Pass SCORE_SCHEMES explicitly to sweep it.
    score_schemes: tuple[str, ...] = (DEFAULT_SCORE_SCHEME,),
    score_bits: int = DEFAULT_SCORE_BITS,
    # Also opt-in, same reasoning: another multiplier on the run count.
    score_decay_schemes: tuple[str, ...] = (DEFAULT_SCORE_DECAY_SCHEME,),
    # Opt-in like the rest. Note this axis is not a precision knob: it changes the
    # policy itself (see SCORE_DECAY_APPLICATIONS), so points from the two settings
    # answer different questions rather than bracketing one.
    score_decay_applications: tuple[str, ...] = (DEFAULT_SCORE_DECAY_APPLICATION,),
    # l and f for score_scheme="fixed"; ignored by every other scheme. Scalars, not
    # tuples: a width sweep is a separate experiment from a policy sweep, so run one
    # sweep per (l, f) rather than crossing them into this product.
    score_length_bits: int = HW_SCORE_LENGTH_BITS,
    score_frac_bits: int = HW_SCORE_FRAC_BITS,
    score_signed: bool = HW_SCORE_SIGNED,
) -> list[dict]:
    """
    Sweep the importance budget ratio r (`budget_ratios`) for each budget
    mode in `budget_modes`, each channel-balancer variant in
    `balance_schemes`, and each scoreboard precision in `score_schemes`,
    running the Line Retrieval benchmark under MiKV at every
    (mode, balancer, score precision, ratio) point.

    `score_schemes` (and `score_decay_schemes`, the precision of the age-decay
    factor) change only how the importance scoreboard is accumulated, never how
    much cache is kept, so like the balancers they move accuracy at an identical
    footprint.

    The mode decides what r multiplies: "fixed_length" freezes
    k = floor(r * t_p) at prefill (t_p from the no-quant baseline's prompt
    length), "fixed_ratio" re-derives k = floor(r * t) every step from the
    current cache length -- so at the same r the two modes land at
    *different* footprints, and each row is sized against its own k (see
    `resolve_budget_k`). For every point, pairs the resulting
    accuracy with the KV cache footprint at a **fixed** context length of
    `max_tokens` -- the context the hardware target provisions, so that
    full allocation is what compression acts on. Both the uncompressed
    baseline (`kv_size_before`) and the MiKV estimate (`kv_size_after`)
    are computed at that same `seq_len`, so the ratio between them is
    meaningful; min(seq_len, k) positions stay HIGH precision (native
    16-bit, or `high_bits` if `high_precision_native` is False), the rest
    are compressed to `low_bits` (N) -- see `kv_cache_size_bytes`.

    The length generation actually reached is reported separately
    (`avg_seq_len` / `kv_bytes_occupied`) as a diagnostic: it is usually
    well short of `max_tokens` because decoding stops at EOS, so it
    measures occupancy rather than the provisioned allocation. Do not mix
    the two bases in one ratio.

    Runs the no-quant baseline itself, first, before installing any MiKV
    patch -- see `run_line_retrieval_no_quant`.

    The footprint depends only on (seq_len, k, bit widths), so it is the
    same for every scheme at a given (mode, ratio) -- the schemes are
    compared on accuracy at equal compression, not on compression. It does
    *not* match across modes, since k differs there by construction.

    Returns a flat list of dicts, one per (mode, scheme, score scheme,
    ratio): budget_mode, scheme, score_scheme, score_bits, ratio, k,
    accuracy, baseline_accuracy, seq_len,
    kv_size_before, kv_size_after, compression_pct, the run configuration
    (window_ratio, high/low bits, sample counts, seed), plus the
    avg_seq_len / kv_bytes_occupied diagnostics -- ready to hand to
    `format_results_table`, `write_results_csv` and
    `plot_accuracy_vs_compression`.
    """
    print(f"[sweep] === stage 1/2: no-quant baseline (ratios to sweep: {list(budget_ratios)}) ===", flush=True)
    baseline_accuracy, t_p, kv_bytes_occupied, avg_seq_len = run_line_retrieval_no_quant(
        model, tokenizer, num_samples=num_samples, num_records=num_records, max_tokens=max_tokens, seed=seed
    )

    # Both sides of the compression ratio are sized at the fixed context the
    # target provisions (`max_tokens`), not at the length generation happened to
    # reach: the hardware allocates a full max_tokens-token KV cache up front, so
    # that allocation is what compression acts on. The two sides MUST share a
    # basis -- an earlier version sized kv_size_after at max_tokens while taking
    # kv_size_before from the baseline's measured per-sample length, putting a
    # provisioned figure over an occupancy one and overstating compression.
    seq_len = max_tokens
    kv_size_before = kv_cache_size_bytes(model, seq_len, k=None)
    print(
        f"[sweep] sizing at fixed context seq_len={seq_len}: uncompressed KV={kv_size_before / 1e6:.2f} MB "
        f"(baseline generation actually reached avg {avg_seq_len:.0f} tokens = "
        f"{kv_bytes_occupied / 1e6:.2f} MB occupied)",
        flush=True,
    )

    # The compression estimate depends only on (seq_len, k, bit widths), so it is
    # identical across balancer schemes -- the schemes differ in *accuracy* at the
    # same footprint, which is the whole point of comparing them. Only the
    # benchmark is re-run per scheme.
    total_runs = (
        len(budget_modes) * len(balance_schemes) * len(score_schemes)
        * len(score_decay_schemes) * len(score_decay_applications) * len(budget_ratios)
    )
    print(
        f"[sweep] === stage 2/2: {total_runs} MiKV runs ({len(budget_modes)} budget modes x "
        f"{len(balance_schemes)} balancers x {len(score_schemes)} score schemes x "
        f"{len(score_decay_schemes)} decay schemes x {len(score_decay_applications)} decay sites x "
        f"{len(budget_ratios)} ratios), t_p={t_p}, fixed context seq_len={seq_len} ===",
        flush=True,
    )
    results = []
    run = 0
    # One flat product rather than four nested loops: the sweep is a full
    # cross-product over four independent axes, and nesting them buries the body
    # four levels deep for no gain.
    for (
        budget_mode, score_scheme, score_decay_scheme, score_decay_application, scheme, ratio
    ) in itertools.product(
        budget_modes, score_schemes, score_decay_schemes, score_decay_applications,
        balance_schemes, budget_ratios,
    ):
        run += 1
        k = resolve_budget_k(budget_mode, ratio, t_p, seq_len)
        print(
            f"[sweep] ({run}/{total_runs}) mode={budget_mode} scheme={scheme} "
            f"score={_score_tag(score_scheme, score_bits, score_decay_scheme, score_decay_application, score_length_bits, score_frac_bits, score_signed)} "
            f"ratio={ratio} -> k={k}",
            flush=True,
        )
        accuracy = run_line_retrieval_benchmark(
            model,
            tokenizer,
            num_samples=num_samples,
            num_records=num_records,
            budget_ratio=ratio,
            window_ratio=window_ratio,
            high_bits=high_bits,
            low_bits=low_bits,
            high_precision_native=high_precision_native,
            balance_scheme=scheme,
            budget_mode=budget_mode,
            score_scheme=score_scheme,
            score_bits=score_bits,
            score_decay_scheme=score_decay_scheme,
            score_decay_application=score_decay_application,
            score_length_bits=score_length_bits,
            score_frac_bits=score_frac_bits,
            score_signed=score_signed,
            max_tokens=max_tokens,
            seed=seed,
        )
        kv_size_after = kv_cache_size_bytes(
            model, seq_len, k=k, high_bits=high_bits, low_bits=low_bits,
            high_precision_native=high_precision_native,
        )
        compression_pct = 100 * kv_size_after / kv_size_before
        results.append(
            dict(
                budget_mode=budget_mode,
                scheme=scheme,
                score_scheme=score_scheme,
                score_bits=score_bits if score_scheme == "quant" else "",
                score_decay_scheme=score_decay_scheme,
                score_decay_application=score_decay_application,
                score_length_bits=score_length_bits if score_scheme == "fixed" else "",
                score_frac_bits=score_frac_bits if score_scheme == "fixed" else "",
                score_signed=score_signed if score_scheme == "fixed" else "",
                # Recorded, not parameterized: these define what "fixed" / "lut" mean,
                # so a row stays interpretable if the constants are ever retuned.
                hw_delta_bits=HW_DELTA_BITS if score_scheme == "fixed" else "",
                hw_age_lut_entries=HW_AGE_LUT_ENTRIES if score_decay_scheme == "lut" else "",
                ratio=ratio,
                k=k,
                accuracy=accuracy,
                baseline_accuracy=baseline_accuracy,
                seq_len=seq_len,  # the fixed context both sizes are computed at
                kv_size_before=kv_size_before,
                kv_size_after=kv_size_after,
                compression_pct=compression_pct,
                # configuration, carried on every row so a CSV appended to
                # across runs stays self-describing
                t_p=t_p,
                window_ratio=window_ratio,
                high_bits=high_bits,
                low_bits=low_bits,
                high_precision_native=high_precision_native,
                num_samples=num_samples,
                num_records=num_records,
                max_tokens=max_tokens,
                seed=seed,
                model_name=getattr(model.config, "_name_or_path", MODEL_NAME),
                # diagnostics: what generation actually occupied, for comparison
                # against the provisioned figure above
                avg_seq_len=avg_seq_len,
                kv_bytes_occupied=kv_bytes_occupied,
            )
        )
        print(
            f"[sweep] mode={budget_mode} scheme={scheme} "
            f"score={_score_tag(score_scheme, score_bits, score_decay_scheme, score_decay_application, score_length_bits, score_frac_bits, score_signed)} "
            f"ratio={ratio} k={k} "
            f"accuracy={accuracy * 100:.1f}% KV compression={compression_pct:.1f}% "
            f"(baseline accuracy={baseline_accuracy * 100:.1f}%)",
            flush=True,
        )
    print("[sweep] done", flush=True)
    return results


def _schemes_in(results: list[dict]) -> list[str]:
    """Distinct schemes present in `results`, in first-seen order."""
    seen = []
    for r in results:
        scheme = r.get("scheme", DEFAULT_BALANCE_SCHEME)
        if scheme not in seen:
            seen.append(scheme)
    return seen


def _modes_in(results: list[dict]) -> list[str]:
    """Distinct budget modes present in `results`, in first-seen order."""
    seen = []
    for r in results:
        mode = r.get("budget_mode", DEFAULT_BUDGET_MODE)
        if mode not in seen:
            seen.append(mode)
    return seen


def _row_score_tag(r: dict) -> str:
    """The scoreboard-precision tag of one result row, tolerating rows written
    before `score_scheme` existed (they were all native fp16)."""
    scheme = r.get("score_scheme") or DEFAULT_SCORE_SCHEME
    bits = r.get("score_bits") or DEFAULT_SCORE_BITS
    decay = r.get("score_decay_scheme") or DEFAULT_SCORE_DECAY_SCHEME
    # Rows written before this column existed were all compounding, so they must
    # NOT inherit today's default -- that would silently relabel them.
    site = r.get("score_decay_application") or "compounding"
    length = r.get("score_length_bits") or HW_SCORE_LENGTH_BITS
    frac = r.get("score_frac_bits") or HW_SCORE_FRAC_BITS
    signed = r.get("score_signed")
    signed = HW_SCORE_SIGNED if signed in (None, "") else str(signed).lower() in ("true", "1")
    return _score_tag(scheme, int(bits), decay, site, int(length), int(frac), signed)


def _score_tags_in(results: list[dict]) -> list[str]:
    """Distinct scoreboard precisions present in `results`, in first-seen order."""
    seen = []
    for r in results:
        tag = _row_score_tag(r)
        if tag not in seen:
            seen.append(tag)
    return seen


def _rows_for(results: list[dict], mode: str, scheme: str, score_tag: str | None = None) -> list[dict]:
    return [
        r
        for r in results
        if r.get("budget_mode", DEFAULT_BUDGET_MODE) == mode
        and r.get("scheme", DEFAULT_BALANCE_SCHEME) == scheme
        and (score_tag is None or _row_score_tag(r) == score_tag)
    ]


def _configs_in(results: list[dict]) -> list[tuple[str, str, str]]:
    """The (budget mode, balancer, scoreboard precision) configurations actually
    present, as a list of keys to group tables and figures by."""
    return [
        (m, s, st)
        for m in _modes_in(results)
        for st in _score_tags_in(results)
        for s in _schemes_in(results)
        if _rows_for(results, m, s, st)
    ]


def _config_label(mode: str, scheme: str, score_tag: str | None = None) -> str:
    """Human-readable name for one configuration. `score_tag` is folded in only
    when the caller passes one -- with a single scoreboard precision in play
    (the default sweep) naming it in every label and title is just noise."""
    label = f"{_BUDGET_MODE_LABELS.get(mode, mode)} / {_BALANCE_SCHEME_LABELS.get(scheme, scheme)}"
    if score_tag is not None:
        label += f" / {score_tag}"
    return label


def format_results_table(results: list[dict]) -> str:
    """
    Render the sweep as a markdown table, one section per (budget mode,
    balancer scheme) configuration, plus a head-to-head accuracy comparison
    when more than one scheme ran. Returned as a string so the caller can
    both print it (into the tee'd log) and paste it into the docs.

    The head-to-head keys on (mode, ratio), not ratio alone: at the same r
    the two budget modes sit at different k, hence different footprints, so
    only rows sharing a mode are comparable at equal compression.
    """
    schemes = _schemes_in(results)
    modes = _modes_in(results)
    score_tags = _score_tags_in(results)
    # Only name the scoreboard precision once there is more than one to tell apart.
    show_score = len(score_tags) > 1
    lines = []

    baseline = results[0]["baseline_accuracy"] * 100 if results else float("nan")
    seq_len = results[0]["seq_len"] if results else 0
    kv_before_mb = results[0]["kv_size_before"] / 1e6 if results else 0.0
    lines.append(
        f"Sizing basis: fixed context seq_len={seq_len} tokens, "
        f"uncompressed KV = {kv_before_mb:.1f} MB. "
        f"Uncompressed baseline accuracy = {baseline:.1f}%."
    )
    lines.append("")

    for mode, scheme, score_tag in _configs_in(results):
        rows = _rows_for(results, mode, scheme, score_tag)
        lines.append(f"### Budget: {mode} -- {_BUDGET_MODE_LABELS.get(mode, mode)} | "
                     f"Balancer: {scheme} -- {_BALANCE_SCHEME_LABELS.get(scheme, scheme)}"
                     + (f" | Scoreboard: {score_tag}" if show_score else ""))
        lines.append("")
        lines.append("| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| - (uncompressed) | - | {kv_before_mb:.1f} | 100.0% | {baseline:.1f}% |")
        for r in rows:
            lines.append(
                f"| {r['ratio']} | {r['k']} | {r['kv_size_after'] / 1e6:.1f} | "
                f"{r['compression_pct']:.1f}% | {r['accuracy'] * 100:.1f}% |"
            )
        lines.append("")

    if len(schemes) > 1:
        lines.append("### Head-to-head (accuracy at equal compression, within a budget mode)")
        lines.append("")
        score_col = "scoreboard | " if show_score else ""
        lines.append(
            f"| budget mode | {score_col}ratio | k | KV size (%) | " + " | ".join(schemes) + " |"
        )
        lines.append("|---|---|---|---|" + ("---|" if show_score else "") + "---|" * len(schemes))
        # Keyed on (mode, scoreboard precision, ratio): only rows agreeing on all
        # three sit at the same footprint under the same scoring, so only those are
        # a fair balancer-vs-balancer comparison.
        by_point: dict[tuple[str, str, float], dict[str, dict]] = {}
        for r in results:
            key = (r.get("budget_mode", DEFAULT_BUDGET_MODE), _row_score_tag(r), r["ratio"])
            by_point.setdefault(key, {})[r.get("scheme", DEFAULT_BALANCE_SCHEME)] = r
        for key in sorted(by_point, key=lambda k: (modes.index(k[0]), score_tags.index(k[1]), k[2])):
            per_scheme = by_point[key]
            any_row = next(iter(per_scheme.values()))
            cells = [
                f"{per_scheme[s]['accuracy'] * 100:.1f}%" if s in per_scheme else "-" for s in schemes
            ]
            lines.append(
                f"| {key[0]} | " + (f"{key[1]} | " if show_score else "")
                + f"{key[2]} | {any_row['k']} | {any_row['compression_pct']:.1f}% | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


# Column order for the results CSV. Fixed (rather than taken from the first
# row's keys) because the file is *appended* to across runs: every run has to
# write the same columns in the same order under one header, and a row missing
# a key -- an older result dict, say -- must land as an empty cell rather than
# silently shifting every subsequent column. Leading config columns identify
# what produced each row (budget mode, balancer, bit widths, ...), so rows from
# different runs stay distinguishable once they're interleaved in one file.
RESULTS_CSV_FIELDS = (
    "timestamp",
    "run_id",
    "model_name",
    "budget_mode",
    "scheme",
    "score_scheme",
    "score_bits",
    "score_decay_scheme",
    "score_decay_application",
    "score_length_bits",
    "score_frac_bits",
    "score_signed",
    "hw_delta_bits",
    "hw_age_lut_entries",
    "ratio",
    "window_ratio",
    "high_bits",
    "low_bits",
    "high_precision_native",
    "num_samples",
    "num_records",
    "max_tokens",
    "seed",
    "t_p",
    "k",
    "seq_len",
    "accuracy",
    "baseline_accuracy",
    "kv_size_before",
    "kv_size_after",
    "compression_pct",
    "avg_seq_len",
    "kv_bytes_occupied",
)


def write_results_csv(
    results: list[dict],
    save_path: str = "docs/kv_compression_sweep.csv",
    run_id: str | None = None,
) -> str:
    """
    Append the sweep rows to `save_path` (one row per (budget mode, scheme,
    ratio)), creating the file with a header if it doesn't exist yet and
    reusing the existing header otherwise. Appending, not overwriting, so
    results accumulate across invocations -- each row carries the full
    configuration that produced it (see RESULTS_CSV_FIELDS) plus a
    `timestamp` and a `run_id` shared by every row of one sweep, so a
    single run can be pulled back out of the accumulated file.

    Rows are restricted to RESULTS_CSV_FIELDS; any extra key on a result
    dict is dropped rather than corrupting the column alignment of a file
    whose header was written by an earlier version.
    """
    if not results:
        return save_path
    stamp = datetime.datetime.now()
    run_id = run_id or f"{stamp:%Y%m%d_%H%M%S}"
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_header = not os.path.exists(save_path) or os.path.getsize(save_path) == 0
    if not write_header:
        # An existing file was written under some header; appending rows ordered by
        # RESULTS_CSV_FIELDS under a *different* one would put every value in the
        # wrong column, silently. If the headers don't match (a file from an older
        # version of this script, say), divert to a fresh timestamped sibling rather
        # than either corrupting it or raising -- a sweep is hours of GPU time and
        # must not lose its results to a bookkeeping mismatch.
        with open(save_path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != list(RESULTS_CSV_FIELDS):
            stem, ext = os.path.splitext(save_path)
            diverted = f"{stem}_{run_id}{ext}"
            print(
                f"[csv] WARNING: {save_path} has a different header "
                f"({len(existing)} columns vs. {len(RESULTS_CSV_FIELDS)} expected) -- "
                f"writing to {diverted} instead of appending",
                flush=True,
            )
            save_path, write_header = diverted, True
    with open(save_path, "a", newline="") as f:
        # extrasaction="ignore": keys outside RESULTS_CSV_FIELDS are skipped
        # instead of raising, so a caller-added diagnostic can't break the append.
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in RESULTS_CSV_FIELDS}
            row["timestamp"] = f"{stamp:%Y-%m-%d %H:%M:%S}"
            row["run_id"] = run_id
            writer.writerow(row)
    print(
        f"[csv] appended {len(results)} rows (run_id={run_id}) to {save_path}"
        f"{' (new file)' if write_header else ''}",
        flush=True,
    )
    return save_path


def plot_accuracy_vs_compression(
    results: list[dict], save_path: str = "docs/kv_compression_sweep.png"
) -> list[str]:
    """
    Plot Line Retrieval accuracy against KV cache compression
    (100 * size_after(k) / size_before) across the budget sweep in
    `results` (as returned by `sweep_kv_compression`).

    Writes one figure per (budget mode, balancer scheme) configuration --
    `save_path` with mode and scheme appended before the extension (e.g.
    `..._fixed_length_paper.png`, `..._fixed_ratio_pow2.png`) -- plus, when
    more than one configuration ran, a combined overlay at `save_path`
    itself for direct comparison. Saves to disk since matplotlib runs
    headless (Agg backend) here. Returns the list of paths written.

    Note that across budget modes the x axis is not a shared grid: at the
    same r, "fixed_ratio" and "fixed_length" land at different k, hence
    different footprints. That's exactly what the overlay is for -- it puts
    both modes' accuracy/compression trade-off curves on one axis, where a
    point sitting up and to the left is the better deal regardless of which
    r produced it.
    """
    configs = _configs_in(results)
    show_score = len(_score_tags_in(results)) > 1
    stem, ext = os.path.splitext(save_path)
    written = []

    def _draw(ax, rows, label=None, annotate=True):
        compressions = [r["compression_pct"] for r in rows]
        accuracies = [r["accuracy"] * 100 for r in rows]
        order = sorted(range(len(rows)), key=lambda i: compressions[i])
        xs = [compressions[i] for i in order]
        ys = [accuracies[i] for i in order]
        ax.plot(xs, ys, marker="o", label=label)
        if not annotate:
            return
        for i in order:
            ax.annotate(
                f"k={rows[i]['k']} (r={rows[i]['ratio']})",
                (compressions[i], accuracies[i]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    # one figure per (budget mode, balancer, scoreboard precision)
    for mode, scheme, score_tag in configs:
        rows = _rows_for(results, mode, scheme, score_tag)
        # The score tag enters the filename only when more than one is present, so
        # a default sweep keeps writing the same paths it did before.
        path = f"{stem}_{mode}_{scheme}" + (f"_{score_tag}" if show_score else "") + ext
        print(f"[plot] rendering {mode}/{scheme}/{score_tag} plot ({len(rows)} points)...", flush=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        _draw(ax, rows)
        if rows:
            ax.axhline(
                rows[0]["baseline_accuracy"] * 100,
                linestyle="--",
                linewidth=1,
                color="gray",
                label="uncompressed baseline",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("KV cache size after compression (% of uncompressed)")
        ax.set_ylabel("Line Retrieval accuracy (%)")
        ax.set_title(
            f"MiKV: accuracy vs. KV compression\n"
            f"{_config_label(mode, scheme, score_tag if show_score else None)}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved to {path}", flush=True)
        written.append(path)

    # combined overlay
    if len(configs) > 1:
        print("[plot] rendering combined overlay...", flush=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (mode, scheme, score_tag) in enumerate(configs):
            rows = _rows_for(results, mode, scheme, score_tag)
            # annotate once per budget mode: within a mode every scheme's series sits
            # at the same k/compression points, so repeating the labels there just
            # overplots them -- but the modes sit at *different* points, so each one
            # needs its own set.
            first_of_mode = i == next(j for j, c in enumerate(configs) if c[0] == mode)
            _draw(
                ax,
                rows,
                label=_config_label(mode, scheme, score_tag if show_score else None),
                annotate=first_of_mode,
            )
        ax.axhline(
            results[0]["baseline_accuracy"] * 100,
            linestyle="--",
            linewidth=1,
            color="gray",
            label="uncompressed baseline",
        )
        ax.set_xlabel("KV cache size after compression (% of uncompressed)")
        ax.set_ylabel("Line Retrieval accuracy (%)")
        ax.set_title(
            "MiKV: accuracy vs. KV compression -- configuration comparison", fontsize=10
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved to {save_path}", flush=True)
        written.append(save_path)

    return written


class _Tee:
    """Duplicates writes to multiple streams (e.g. the real stdout and a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":
    log_path = f"mikv_run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w")
    # stderr too, not just stdout: tqdm's progress bars and PyTorch's own [W...]
    # warnings (e.g. the CUDA OOM-retry allocator warnings) are both written there,
    # not to stdout, and interleaving them with our own prints in one file is the
    # whole point -- correlating a memory warning with the sample it happened on.
    _real_stdout, _real_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_real_stdout, log_file)
    sys.stderr = _Tee(_real_stderr, log_file)
    try:
        print(f"=== MiKV: logging this run to {log_path} ===", flush=True)

        print("=== MiKV: loading model ===", flush=True)
        model, tokenizer = load_model()

        print("=== MiKV: starting KV-compression sweep ===", flush=True)
        # num_samples=40: at n=20, one flipped sample swings accuracy by 5 points, drowning
        # out real signal. 40 roughly halves that per-sample noise.
        # num_records: eager attention materializes the full [t_p, t_p] attention-weight
        # matrix per layer (needed for scoring), so its memory is O(t_p^2). These values
        # were sized against Qwen2.5-0.5B-Instruct (24 layers, 14 heads, 2 kv heads,
        # ~1GB weights) on a 4GB GPU -- Llama-2-7b-chat-hf (32 layers, 32 heads, no GQA,
        # ~13GB weights) has a much larger footprint per token, so these were NOT
        # re-verified against it; check available VRAM before scaling num_records up.
        # balance_schemes: both channel balancers are swept over the same ratios, so
        # each ratio yields an accuracy for the paper's sqrt(q_max/k_max) balancer and
        # for the hardware-favouring power-of-two one at an identical KV footprint.
        # budget_modes: likewise both budget modes -- k frozen at r*t_p vs. k tracking
        # r*t as the cache grows. Together these multiply the number of benchmark runs
        # by 4 (2 modes x 2 schemes) over a single-configuration sweep; drop either
        # tuple to a single entry to cut that back.
        results = sweep_kv_compression(
            model,
            tokenizer,
            budget_ratios=(0.25, 0.5, 0.75),
            num_samples=40,
            num_records=40,
            max_tokens=4096,
            balance_schemes=BALANCE_SCHEMES,
            budget_modes=BUDGET_MODES,
            # score_schemes defaults to ("native",) -- the fp16 scoreboard this has
            # always used. Pass SCORE_SCHEMES to also run the fp32 reference and the
            # score_bits-wide quantized accumulator; that triples the run count, so
            # it is opt-in rather than on by default.
            # score_schemes=SCORE_SCHEMES,
            # score_bits=8,
            # score_decay_schemes=SCORE_DECAY_SCHEMES,
            # score_decay_applications=SCORE_DECAY_APPLICATIONS,
            # For the IPU score path specifically:
            #   score_schemes=("native", "fixed"), score_decay_schemes=("exact", "lut"),
            #   score_decay_applications=("compounding", "ranking")
        )

        print("=== MiKV: results ===", flush=True)
        print(format_results_table(results), flush=True)
        write_results_csv(results)

        print("=== MiKV: plotting results ===", flush=True)
        plot_accuracy_vs_compression(results)

        print(f"=== MiKV: done (log saved to {log_path}) ===", flush=True)
    finally:
        # restore the real streams *before* closing log_file -- otherwise sys.stdout/
        # stderr are left pointing at a Tee wrapping a closed file, which breaks
        # whatever Python itself tries to print during interpreter shutdown.
        sys.stdout, sys.stderr = _real_stdout, _real_stderr
        log_file.close()
