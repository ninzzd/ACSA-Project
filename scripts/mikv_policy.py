"""
The MiKV cache policy itself: the per-generation state machine, the attention
monkey-patch that installs it, and the generation loop that drives it.

- Prefill: derive a per-(layer, kv head, channel) balancer b from the prompt's
  Q/K statistics, run full causal attention to accumulate an importance score per
  prompt position, then split the budget k into a recency window of size w and a
  top-scoring set of size k - w, and write each position's K/V at HIGH or LOW
  precision accordingly.
- Decode: attend over the entire mixed-precision cache with the frozen balancer,
  accumulate importance scores for every position (including the one just
  written), and re-derive S each step; whatever falls out of S is demoted to LOW
  precision -- sticky, never restored.

The scoreboard is a plain running sum, s(i+1) = s(i) + a. The age decay is
stateless and exists only to pick the victim: s'(i+1) = s(i+1) / age is formed on
the comparison path, the demoted token is argmin s', and s' is never stored. See
SCORE_DECAY_APPLICATIONS.

Q/K must be intercepted post-RoPE, pre-cache, inside each layer's attention
forward: the balancer has to divide the *query actually used to produce this
step's output*, and a query is never cached, so this cannot be done by mutating
the DynamicCache after the fact. `LlamaAttention.forward` is therefore
monkey-patched per layer to inject balancing, scoring and quantize-on-write
around the same `attention_interface` call the stock implementation uses. (The
rotary/eager-attention helpers are imported from transformers' Qwen2 module but
are byte-identical to Llama's -- both copied from the same upstream source.)

Uses the -chat checkpoint, not the base model: the base model ships no
`chat_template`, so `apply_chat_template()` cannot even be called on it. -chat is
fine-tuned on Llama-2's own [INST]/<<SYS>> template, same architecture and config
otherwise. Llama-2-7b uses plain multi-head attention (no GQA:
num_key_value_heads == num_attention_heads), so `num_kv_groups` is always 1 here
-- handled as a special case of the general GQA path, not a separate branch.
"""

import math
import types
from dataclasses import dataclass

from mikv_runtime import (
    ALL_ATTENTION_FUNCTIONS,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    apply_rotary_pos_emb,
    eager_attention_forward,
    torch,
)
from mikv_config import (
    BALANCE_SCHEMES,
    BUDGET_MODES,
    DEFAULT_BALANCE_SCHEME,
    DEFAULT_BUDGET_MODE,
    DEFAULT_BUDGET_RATIO,
    DEFAULT_HIGH_BITS,
    DEFAULT_HIGH_PRECISION_NATIVE,
    DEFAULT_LOW_BITS,
    DEFAULT_SCORE_BITS,
    DEFAULT_SCORE_DECAY_APPLICATION,
    DEFAULT_SCORE_DECAY_SCHEME,
    DEFAULT_SCORE_SCHEME,
    DEFAULT_WINDOW_RATIO,
    DEFAULT_WINDOW_TOKENS,
    DEVICE,
    DTYPE,
    HW_AGE_LUT_ENTRIES,
    HW_AGE_LUT_RECENT_WINDOW,
    HW_DELTA_BITS,
    HW_DELTA_FRAC_BITS,
    HW_SCORE_FRAC_BITS,
    HW_SCORE_LENGTH_BITS,
    HW_SCORE_SIGNED,
    MODEL_NAME,
    SCORE_DECAY_APPLICATIONS,
    SCORE_DECAY_SCHEMES,
    SCORE_SCHEMES,
)
from mikv_quant import (
    age_lut_cmp_val,
    fixed_point_range,
    quantize_fixed_point,
    quantize_kv,
    quantize_scores,
)

# Query positions per block in the prefill score reduction. A host memory knob,
# not a hardware parameter: the reduction is mathematically one sum over all
# queries either way. It exists because the fixed-point path quantizes every
# softmax value individually, so a whole [batch, kv_heads, groups, t_p, t_p]
# attention tensor would otherwise be converted in one allocation -- ~0.8 GB of
# fp64 per layer at t_p = 1800, on a card that has under 2 GB free once the model
# and its KV cache are resident.
PREFILL_SCORE_CHUNK = 256


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
        window_tokens: int | None = DEFAULT_WINDOW_TOKENS,
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
        age_lut_entries: int = HW_AGE_LUT_ENTRIES,
        evict: bool = False,
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
        self.window_tokens = window_tokens
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
        self.age_lut_entries = age_lut_entries
        self.score_signed = score_signed
        # H2O-style hard eviction, as an alternative to demoting the LOW set to
        # `low_bits`: positions outside the importance set S are masked out of
        # every future attention row (an additive -inf bias, applied in
        # `_mikv_attention_forward`) instead of being kept around at reduced
        # precision. `low_bits` is accepted but never read once this is set --
        # see `quantize_prefill`/`demote_decode` -- so a sweep can still carry a
        # nominal low_bits value through its grid without it doing anything.
        self.evict = evict

        self.phase = "prefill"
        self.k = self.w = self.k_H = None
        # Overwritten by start_prefill; set here so the attribute always exists.
        self._lut_recent_window = (
            int(window_tokens) if window_tokens is not None else HW_AGE_LUT_RECENT_WINDOW
        )
        self.layers: dict[int, _LayerState] = {}

    def budget_for(self, t: int) -> tuple[int, int, int]:
        """
        (k, w, k_H) for a cache holding `t` tokens -- the budget split itself,
        independent of which mode decides what `t` to pass in.

        w comes from `window_tokens` when that is set (an absolute count of
        recent positions) and from `window_ratio` otherwise (a fraction of k).
        Either way it is clamped into [0, k]: the window is carved *out of* the
        budget, never added to it, so k_H = k - w and the importance set is
        exactly k positions in both cases. A window wider than k therefore
        degenerates to pure recency (k_H = 0) rather than growing the budget --
        worth knowing when k is small, e.g. w = 128 against k = floor(0.25 * t_p)
        on a short prompt.
        """
        k = max(1, math.floor(self.budget_ratio * t))
        if self.window_tokens is not None:
            w = max(0, min(k, int(self.window_tokens)))
        else:
            w = max(0, min(k, math.floor(self.window_ratio * k)))
        return k, w, k - w

    def start_prefill(self, prompt_len: int) -> None:
        self.phase = "prefill"
        # Both modes agree at prefill (t == t_p there); they diverge during decode,
        # where "fixed_length" keeps these frozen and "fixed_ratio" re-derives them
        # from the grown t. These stay set either way so the frozen budget is
        # available for reporting/sizing.
        self.k, self.w, self.k_H = self.budget_for(prompt_len)
        # The age LUT's smallest addressable age is n_min = W + 1, where W is the
        # recency window: no token younger than that ever reaches the comparator,
        # so the table need not cover those ages and its entries gain fractional
        # bits accordingly (see `age_lut_frac_bits`). W has to be a constant of the
        # run -- a synthesized ROM cannot re-derive F per step -- so take the
        # *configured* absolute window when there is one, and fall back to the
        # RTL's own W otherwise, since a ratio-derived w moves with k.
        self._lut_recent_window = (
            int(self.window_tokens) if self.window_tokens is not None
            else HW_AGE_LUT_RECENT_WINDOW
        )
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
        if self.evict:
            # Nothing to quantize: positions outside S are about to be masked
            # out of every future attention row (see `_mikv_attention_forward`),
            # so whatever is stored for them is never read. Leave them at
            # whatever precision they arrived at rather than paying for a
            # quantize call whose result cannot affect the output.
            layer.keys[:] = torch.where(high_mask, self._quantize_high(k_bal), k_bal)
            layer.values[:] = torch.where(high_mask, self._quantize_high(v), v)
        else:
            layer.keys[:] = torch.where(high_mask, self._quantize_high(k_bal), quantize_kv(k_bal, bits=self.low_bits))
            layer.values[:] = torch.where(high_mask, self._quantize_high(v), quantize_kv(v, bits=self.low_bits))

    def demote_decode(self, layer_idx, cache):
        state = self._state(layer_idx)
        s = self._importance_set(state.a)  # [batch, num_kv_heads, t]

        # sticky: only demote, and only positions not already demoted
        fell_out = state.in_s & ~s & ~state.demoted
        if fell_out.any():
            if not self.evict:
                layer = cache.layers[layer_idx]
                mask = fell_out.unsqueeze(-1)
                layer.keys[:] = torch.where(mask, quantize_kv(layer.keys, bits=self.low_bits), layer.keys)
                layer.values[:] = torch.where(mask, quantize_kv(layer.values, bits=self.low_bits), layer.values)
            # Under eviction, `state.demoted` alone is what matters from here:
            # `_mikv_attention_forward` turns it into an attention mask before
            # the next step, so no write to the K/V tensors is needed at all.
            state.demoted = state.demoted | fell_out
        state.in_s = s

    # ---- hard eviction (evict=True): mask instead of demote ----

    def eviction_bias(self, layer_idx, seq_len: int, num_kv_groups: int, dtype) -> torch.Tensor | None:
        """
        Additive attention bias that removes every evicted position from every
        future attention row -- the actual effect of H2O-style eviction (the
        KV pair is gone), reproduced without resizing any tensor: a demoted
        position gets -inf added to its score before the softmax, so its
        attention weight comes out at exactly 0 regardless of what is stored
        in `layer.keys`/`layer.values` there.

        Returns None when nothing has been demoted yet (prefill's own
        attention, and every step before the first demotion), so the caller
        can skip touching `attention_mask` at all in the common case.
        """
        state = self._state(layer_idx)
        if state.demoted is None or not bool(state.demoted.any()):
            return None
        batch, num_kv_heads, t = state.demoted.shape
        demoted = state.demoted
        if t < seq_len:
            # The position(s) just appended this step haven't been through
            # `write_new_token`'s bookkeeping yet when this is read (see
            # `_mikv_attention_forward`), so pad with "not demoted" -- a
            # freshly written token is always inside the recency window and
            # can never be evicted at the moment of its own creation.
            pad = torch.zeros(
                batch, num_kv_heads, seq_len - t, dtype=torch.bool, device=demoted.device
            )
            demoted = torch.cat([demoted, pad], dim=-1)
        neg_inf = torch.finfo(dtype).min
        bias = torch.zeros(demoted.shape, dtype=dtype, device=demoted.device)
        bias = bias.masked_fill(demoted, neg_inf)              # [batch, num_kv_heads, seq_len]
        bias = bias.repeat_interleave(num_kv_groups, dim=1)     # [batch, num_heads, seq_len]
        return bias.unsqueeze(2)                                 # [batch, num_heads, 1, seq_len]

    def apply_eviction_mask(self, layer_idx, attention_mask, seq_len: int, num_kv_groups: int, dtype):
        """`attention_mask` with evicted positions added in, or `attention_mask`
        unchanged if nothing is evicted yet. Broadcasts against the causal mask
        the model already built (both are additive), so this composes with it
        rather than replacing it."""
        bias = self.eviction_bias(layer_idx, seq_len, num_kv_groups, dtype)
        if bias is None:
            return attention_mask
        return bias if attention_mask is None else attention_mask + bias

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
            # Blocked over query positions, and the accumulation dtype pinned.
            # Both are memory, not model: the sum over (groups, queries) is the
            # same sum in any order, and fp64 is pinned because ingress is a
            # 17-bit word that quantize_fixed_point returns in fp32, while
            # summing up to t_p such terms in fp32 would drift past the l = 30
            # accumulator's own LSB. A reduction with an explicit dtype promotes
            # its input first rather than accumulating on the fly, so the cast is
            # a real allocation and only the blocking keeps it small.
            summed = torch.zeros(
                batch, self.num_kv_heads, t_p, dtype=torch.float64, device=w.device
            )
            for start in range(0, t_p, PREFILL_SCORE_CHUNK):
                block = w[:, :, :, start : start + PREFILL_SCORE_CHUNK, :]
                summed += self._ingress_delta(block).sum(dim=(2, 3), dtype=torch.float64)
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
            # ipu_age_lut: a ROM read and a multiply, not a divide -- and the
            # whole comparator path, output word included, not just the ROM. See
            # `age_lut_cmp_val`. The recency window is passed because it is what
            # fixes the table's smallest age (n_min = w + 1) and therefore how
            # many fractional bits the ROM entries can carry; `_lut_recent_window`
            # is resolved once at prefill so it is a constant of the run, the way
            # a synthesized ROM is.
            return age_lut_cmp_val(
                updated,
                distance,
                entries=self.age_lut_entries,
                recent_window=self._lut_recent_window,
                out_frac_bits=self.score_frac_bits,
            )

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

    if policy.evict:
        # Fold in the eviction mask *after* write_new_token, so `state.demoted`
        # already covers the position just appended (never evicted at the
        # moment of its own creation) and matches key_states' current length.
        # A no-op during prefill and up until the first demotion happens.
        attention_mask = policy.apply_eviction_mask(
            layer_idx, attention_mask, key_states.shape[-2], num_kv_groups, query_states.dtype
        )

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
    window_tokens: int | None,
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
    age_lut_entries: int = HW_AGE_LUT_ENTRIES,
    evict: bool = False,
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
        window_tokens=window_tokens,
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
        age_lut_entries=age_lut_entries,
        evict=evict,
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
    window_tokens: int | None = DEFAULT_WINDOW_TOKENS,
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
    age_lut_entries: int = HW_AGE_LUT_ENTRIES,
    evict: bool = False,
) -> str:
    generated, _ = _generate_tokens_with_mikv(
        model,
        tokenizer,
        prompt,
        max_tokens,
        budget_ratio,
        window_ratio,
        window_tokens,
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
        age_lut_entries,
        evict,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)
