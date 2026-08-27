"""
KV cache quantization policy for autoregressive inference on
Llama-2-7b-chat-hf, implementing MiKV's channel-balanced,
budget-constrained mixed-precision cache:

- Prefill: derive a per-(layer, kv head, channel) balancer b from the
  prompt's Q/K statistics, run full causal attention to accumulate an
  importance score per prompt position, then split the budget k (resolved
  once from the prompt length and frozen for the rest of the sequence)
  into a recency window of size w and a top-scoring set of size k - w, and
  write each position's K/V at HIGH or LOW precision accordingly.
- Decode: attend over the entire mixed-precision cache with the frozen
  balancer, accumulate importance scores for every position (including
  the one just written), and re-derive S each step; whatever falls out of
  S is demoted to LOW precision -- sticky, never restored.

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

import datetime
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
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

DEFAULT_BUDGET_RATIO = 0.5  # k: importance budget, k = floor(BUDGET_RATIO * t_p)
DEFAULT_WINDOW_RATIO = 0.5  # w: recency window, w = floor(WINDOW_RATIO * k)  (default: w = k/2)
DEFAULT_HIGH_BITS = 8        # bit-width the "important" bucket is quantized to when not left native
DEFAULT_LOW_BITS = 2         # N: bit-width for the low-precision ("evicted") bucket
DEFAULT_HIGH_PRECISION_NATIVE = True  # important tokens stay at the model's native bf16/fp16
# (untouched) when True; quantized to DEFAULT_HIGH_BITS instead when False.


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


def quantize_kv(tensor: torch.Tensor, bits: int = 2) -> torch.Tensor:  # does not include channel biasing factor b
    """
    Fake-quantize K/V vectors: round-trip them through a `bits`-wide affine
    quantizer (per vector, i.e. per last dim) and dequantize back to floats.

    tensor: (..., head_dim)
    """
    qmax = 2**bits - 1
    t_min = tensor.amin(dim=-1, keepdim=True)
    t_max = tensor.amax(dim=-1, keepdim=True)
    scale = (t_max - t_min).clamp(min=1e-8) / qmax
    quantized = torch.round((tensor - t_min) / scale)
    dequantized = quantized * scale + t_min
    return dequantized.to(tensor.dtype)


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
    the (t=k+1)-token cache and the one token that falls out gets evicted
    (demoted to low precision).
    """

    def __init__(
        self,
        num_kv_heads: int,
        budget_ratio: float = DEFAULT_BUDGET_RATIO,
        window_ratio: float = DEFAULT_WINDOW_RATIO,
        high_bits: int = DEFAULT_HIGH_BITS,
        low_bits: int = DEFAULT_LOW_BITS,
        high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
    ):
        self.num_kv_heads = num_kv_heads
        self.budget_ratio = budget_ratio
        self.window_ratio = window_ratio
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.high_precision_native = high_precision_native

        self.phase = "prefill"
        self.k = self.w = self.k_H = None
        self.layers: dict[int, _LayerState] = {}

    def start_prefill(self, prompt_len: int) -> None:
        self.phase = "prefill"
        self.k = max(1, math.floor(self.budget_ratio * prompt_len))
        self.w = max(0, min(self.k, math.floor(self.window_ratio * self.k)))
        self.k_H = self.k - self.w
        self.layers = {}

    def start_decode(self) -> None:
        self.phase = "decode"

    def _state(self, layer_idx: int) -> _LayerState:
        return self.layers.setdefault(layer_idx, _LayerState())

    # ---- channel balancing (called post-RoPE, pre-cache) ----

    def balance_prefill(self, layer_idx, query_states, key_states, num_kv_groups):
        """b[c] = sqrt(max_i|Q[i,c]| / max_i|K[i,c]|), per (layer, kv head, channel)."""
        state = self._state(layer_idx)
        batch, num_q_heads, t_p, head_dim = query_states.shape

        q_by_group = query_states.view(batch, self.num_kv_heads, num_kv_groups, t_p, head_dim)
        q_max = q_by_group.abs().amax(dim=(0, 2, 3))  # [num_kv_heads, head_dim]
        k_max = key_states.abs().amax(dim=(0, 2))  # [num_kv_heads, head_dim]
        b = torch.sqrt(q_max / k_max.clamp(min=1e-8)).clamp(min=1e-4)
        state.b = b

        return self._apply_balance(state.b, query_states, key_states, num_kv_groups)

    def balance_decode(self, layer_idx, query_states, key_states, num_kv_groups):
        state = self._state(layer_idx)  # b is frozen; reused as-is
        return self._apply_balance(state.b, query_states, key_states, num_kv_groups)

    def _apply_balance(self, b, query_states, key_states, num_kv_groups):
        head_dim = b.shape[-1]
        num_q_heads = query_states.shape[1]
        b_k = b.view(1, self.num_kv_heads, 1, head_dim)
        b_q = b.repeat_interleave(num_kv_groups, dim=0).view(1, num_q_heads, 1, head_dim)
        return query_states / b_q, key_states * b_k

    # ---- quantize on write ----

    def _quantize_high(self, tensor: torch.Tensor) -> torch.Tensor:
        """The "important" bucket: native bf16/fp16 (untouched) if high_precision_native,
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
        zero = torch.zeros(batch, self.num_kv_heads, 1, dtype=k_bal_new.dtype, device=k_bal_new.device)
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
        if t <= self.k:
            return torch.ones_like(a, dtype=torch.bool)

        s = torch.zeros_like(a, dtype=torch.bool)
        s[:, :, t - self.w :] = True  # recency window: the w most recent positions
        if self.k_H > 0:
            # H2O's heavy-hitter score (Zhang et al. 2023, Alg. 1: F_score := sum_s o_s) is
            # the raw cumulative sum of attention received, with no normalization by token
            # age or step count -- rank directly on `a`, not on a derived per-step average.
            # (A previous version of this code divided by age here; that's not in H2O or
            # MiKV's spec, and it systematically discounts earlier positions in favor of
            # later ones regardless of actual attention received, which is a confound the
            # papers' own criterion doesn't have.)
            candidates = a.masked_fill(s, float("-inf"))
            top = candidates.topk(self.k_H, dim=-1).indices  # top-(k - w) among the rest
            s.scatter_(-1, top, torch.ones_like(top, dtype=torch.bool))
        return s

    # ---- score accumulation ----

    def score_prefill(self, layer_idx, attn_weights, num_kv_groups):
        """a[j] = sum over all query positions i (and, for GQA, over the group's query heads)."""
        state = self._state(layer_idx)
        batch, num_q_heads, t_p, _ = attn_weights.shape
        w = attn_weights.view(batch, self.num_kv_heads, num_kv_groups, t_p, t_p)
        state.a = w.sum(dim=(2, 3))  # -> [batch, num_kv_heads, t_p]

    def score_decode(self, layer_idx, attn_weights, num_kv_groups):
        state = self._state(layer_idx)
        batch, num_q_heads, _, t = attn_weights.shape
        p = attn_weights[:, :, 0, :].view(batch, self.num_kv_heads, num_kv_groups, t).sum(dim=2)
        state.a = state.a + p


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
) -> str:
    generated, _ = _generate_tokens_with_mikv(
        model, tokenizer, prompt, max_tokens, budget_ratio, window_ratio, high_bits, low_bits, high_precision_native
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
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


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
    """
    print(
        f"[mikv] starting {num_samples} samples: budget_ratio={budget_ratio} window_ratio={window_ratio} "
        f"high_bits={high_bits} low_bits={low_bits} high_precision_native={high_precision_native}",
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
        f"[mikv] Line Retrieval accuracy: {accuracy * 100:.1f}% ({correct}/{num_samples}) {_cuda_mem_str()}",
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
    importance budget saturates at k positions and never grows past it, so
    exactly min(seq_len, k) positions stay HIGH precision (native 16-bit if
    `high_precision_native`, else quantized to `high_bits` -- must match
    whatever the accuracy run actually used) and the remaining
    max(0, seq_len - k) are compressed to `low_bits` (N).
    """
    num_layers, num_kv_heads, head_dim = _kv_cache_shape(model)
    elements_per_token = num_layers * num_kv_heads * head_dim * 2  # *2 for K and V

    if k is None:
        return elements_per_token * seq_len * 16 / 8

    high_bit_width = 16 if high_precision_native else high_bits
    num_high = min(seq_len, k)
    num_low = max(0, seq_len - k)
    return elements_per_token * (num_high * high_bit_width / 8 + num_low * low_bits / 8)


def run_line_retrieval_no_quant(
    model,
    tokenizer,
    num_samples: int = 20,
    num_records: int = 20,
    max_tokens: int = 4096,
    seed: int = 0,
) -> tuple[float, int, float]:
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

    Returns (accuracy, t_p, avg_kv_cache_bytes_before_compression).
    """
    print(f"[no-quant] starting {num_samples} baseline samples (num_records={num_records})...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    correct = 0
    t_p = None
    total_bytes = 0.0
    for i in range(num_samples):
        sample = make_line_retrieval_sample(tokenizer, num_records=num_records, rng=rng)
        input_ids = tokenizer(sample.prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_len = input_ids.shape[-1]
        if t_p is None:
            t_p = prompt_len
            print(f"[no-quant] t_p (prompt length) = {t_p} tokens", flush=True)

        output = model.generate(input_ids, max_length=max_tokens, do_sample=False, pad_token_id=pad_token_id)
        total_bytes += kv_cache_size_bytes(model, output.shape[-1], k=None)

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
    return accuracy, t_p, avg_bytes


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
) -> list[dict]:
    """
    Sweep the importance budget k as a ratio of t_p (`budget_ratios`,
    applied against the no-quant baseline's prompt length -- 0.25*t_p,
    0.5*t_p, 0.75*t_p by default) and run the Line Retrieval benchmark
    under MiKV at each point. For every ratio, pairs the resulting
    accuracy with the estimated KV cache size at the end of generation
    (`max_tokens`, the total context budget passed through to both the
    baseline and MiKV runs; early EOS isn't tracked, so this is an upper
    bound on the true final length): min(seq_len, k) positions stay HIGH
    precision (native 16-bit, or `high_bits` if `high_precision_native` is
    False), the rest are compressed to `low_bits` (N) -- see
    `kv_cache_size_bytes`.

    Runs the no-quant baseline itself, first, before installing any MiKV
    patch -- see `run_line_retrieval_no_quant`.

    Returns a list of dicts (one per ratio): ratio, k, accuracy,
    kv_size_before, kv_size_after, compression_pct -- ready to hand to
    `plot_accuracy_vs_compression`.
    """
    print(f"[sweep] === stage 1/2: no-quant baseline (ratios to sweep: {list(budget_ratios)}) ===", flush=True)
    baseline_accuracy, t_p, kv_size_before = run_line_retrieval_no_quant(
        model, tokenizer, num_samples=num_samples, num_records=num_records, max_tokens=max_tokens, seed=seed
    )

    seq_len = max_tokens
    print(f"[sweep] === stage 2/2: MiKV runs at k = ratio * t_p ({t_p}) ===", flush=True)
    results = []
    for idx, ratio in enumerate(budget_ratios, start=1):
        k = max(1, math.floor(ratio * t_p))
        print(f"[sweep] ({idx}/{len(budget_ratios)}) ratio={ratio} -> k={k}", flush=True)
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
            max_tokens=max_tokens,
            seed=seed,
        )
        kv_size_after = kv_cache_size_bytes(
            model, seq_len, k=k, high_bits=high_bits, low_bits=low_bits, high_precision_native=high_precision_native
        )
        compression_pct = 100 * kv_size_after / kv_size_before
        results.append(
            dict(
                ratio=ratio,
                k=k,
                accuracy=accuracy,
                kv_size_before=kv_size_before,
                kv_size_after=kv_size_after,
                compression_pct=compression_pct,
            )
        )
        print(
            f"[sweep] ratio={ratio} k={k} accuracy={accuracy * 100:.1f}% KV compression={compression_pct:.1f}% "
            f"(baseline accuracy={baseline_accuracy * 100:.1f}%)",
            flush=True,
        )
    print("[sweep] done", flush=True)
    return results


def plot_accuracy_vs_compression(results: list[dict], save_path: str = "docs/kv_compression_sweep.png"):
    """
    Plot Line Retrieval accuracy against KV cache compression
    (100 * size_after(k) / size_before) across the budget sweep in
    `results` (as returned by `sweep_kv_compression`). Saves to
    `save_path` since matplotlib runs headless (Agg backend) here.
    """
    compressions = [r["compression_pct"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    labels = [f"k={r['k']} (r={r['ratio']})" for r in results]

    print(f"[plot] rendering accuracy-vs-compression plot ({len(results)} points)...", flush=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(compressions, accuracies, marker="o")
    for x, y, label in zip(compressions, accuracies, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("KV cache size after compression (% of uncompressed)")
    ax.set_ylabel("Line Retrieval accuracy (%)")
    ax.set_title("MiKV: accuracy vs. KV cache compression")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[plot] saved to {save_path}", flush=True)
    return fig


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
        # num_samples=50: at n=20, one flipped sample swings accuracy by 5 points, drowning
        # out real signal. 50 halves that per-sample noise to 2 points.
        # num_records: eager attention materializes the full [t_p, t_p] attention-weight
        # matrix per layer (needed for scoring), so its memory is O(t_p^2). These values
        # were sized against Qwen2.5-0.5B-Instruct (24 layers, 14 heads, 2 kv heads,
        # ~1GB weights) on a 4GB GPU -- Llama-2-7b-chat-hf (32 layers, 32 heads, no GQA,
        # ~13GB weights) has a much larger footprint per token, so these were NOT
        # re-verified against it; check available VRAM before scaling num_records up.
        results = sweep_kv_compression(
            model, tokenizer, budget_ratios=(0.25, 0.5, 0.75), num_samples=40, num_records=40, max_tokens=4096
        )

        print("=== MiKV: plotting results ===", flush=True)
        plot_accuracy_vs_compression(results)

        print(f"=== MiKV: done (log saved to {log_path}) ===", flush=True)
    finally:
        # restore the real streams *before* closing log_file -- otherwise sys.stdout/
        # stderr are left pointing at a Tee wrapping a closed file, which breaks
        # whatever Python itself tries to print during interpreter shutdown.
        sys.stdout, sys.stderr = _real_stdout, _real_stderr
        log_file.close()
