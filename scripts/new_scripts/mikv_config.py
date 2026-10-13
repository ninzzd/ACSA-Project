"""
Every constant the MiKV policy is parameterized by, plus the small helpers that
translate between how a knob is *named* in a sweep and how the policy actually
takes it (`high_precision_knobs`, `format_score_tag`).

Nothing here computes anything about a model -- it is the vocabulary the rest of
the package is written in, so it sits at the bottom of the import graph and
imports only the runtime bootstrap.
"""

from mikv_runtime import torch

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
BALANCE_SCHEME_LABELS = {"paper": "MiKV (sqrt)", "pow2": "hardware (pow-2)"}

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
# --- ipu_age_lut, the reciprocal ROM that replaces the age divide ---
# A hardware divider is iterative and multi-cycle, and the score path needs P = 16
# of them retiring one beat per cycle. The standard fixed-point answer is to store
# scaled reciprocals and multiply: R[a] = round(2^F / n), cmp_val = (S * R) >> F.
#
# The table does NOT start at age 1. `ipu_mask_gen` already guarantees no token
# younger than the recency window W ever reaches the comparator, so the smallest
# age the ROM can see is n_min = W + 1, and F is set by the LARGEST stored entry,
# which sits at that smallest age:
#
#     F = floor(log2((2^B - 1) * n_min))
#
# At B = 16, W = 64 that is F = 22 rather than the F = 15 a table starting at
# age 1 would allow -- seven extra fractional bits, for free, out of a masking
# rule that already existed.
#
# Ages above the table top are folded, not clamped, by the self-similarity of
# 1/n under powers of two: 1/n = (1/n') * 2^-e for n ~ n' * 2^e. The table only
# has to cover one octave; anything larger is halved until it lands back inside,
# and the halving is absorbed into the output shift. See `age_lut_reciprocal`.
HW_AGE_LUT_ENTRIES = 512        # D: ROM depth (512 x 16 b = 1 KB)
HW_AGE_LUT_WORD_BITS = 16       # B: bits per ROM entry
HW_AGE_LUT_RECENT_WINDOW = 64   # W: the RTL's recency window, which fixes n_min = W + 1


def format_score_tag(
    score_scheme: str,
    score_bits: int = DEFAULT_SCORE_BITS,
    score_decay_scheme: str = DEFAULT_SCORE_DECAY_SCHEME,
    score_decay_application: str = DEFAULT_SCORE_DECAY_APPLICATION,
    score_length_bits: int = HW_SCORE_LENGTH_BITS,
    score_frac_bits: int = HW_SCORE_FRAC_BITS,
    score_signed: bool = HW_SCORE_SIGNED,
    age_lut_entries: int = HW_AGE_LUT_ENTRIES,
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
        if score_decay_scheme == "lut":
            # The ROM depth is the whole design question for this scheme, so two
            # depths must not share a tag -- they would collide in every table,
            # legend and filename.
            tag += str(age_lut_entries)
    if score_decay_application != DEFAULT_SCORE_DECAY_APPLICATION:
        tag += f"+{score_decay_application}"
    return tag


BUDGET_MODES = ("fixed_length", "fixed_ratio")
DEFAULT_BUDGET_MODE = "fixed_length"
BUDGET_MODE_LABELS = {
    "fixed_length": "fixed length (k = r*t_p)",
    "fixed_ratio": "fixed ratio (k = r*t)",
}


# The width of the "important" (HIGH) bucket, as a single sweepable axis. The
# policy carries two knobs for it -- `high_precision_native` (leave K/V at the
# model's native fp16, untouched) and `high_bits` (the width to fake-quantize to
# otherwise) -- but they are not independent: `high_bits` means nothing while the
# native flag is set, so crossing them as two axes would run identical
# configurations under two names and pay for both. The sweep therefore exposes
# ONE integer axis, `high_precisions`, and these helpers map it onto the pair:
#
#     16 -> native fp16, untouched   (high_precision_native=True)
#     N  -> fake-quantized to N bits (high_precision_native=False, high_bits=N)
#
# 16 is spelled as "native" rather than as a 16-bit fake-quantize because
# `quantize_kv` short-circuits at bits >= 16 anyway -- the two are the same
# operation, so collapsing them keeps the axis at one value per distinct run.
# How the "important" (HIGH) bucket is stored, as a single named axis:
#   "fp16"  -- the model's native dtype, untouched. `quantize_kv` is never called.
#   "int16" -- a real 16-bit affine round-trip. NOT the same as fp16: uniform grid
#              vs. logarithmic, see `quantize_kv`.
#   "int8"  -- 8-bit affine, the width MiKV's own experiments use for the kept set.
#   "int4"  -- 4-bit affine, offered for completeness; below the LOW tier's own
#              widths it stops being an "important" bucket at all.
# The policy carries two knobs for this -- `high_precision_native` (leave K/V at
# fp16) and `high_bits` (the width to quantize to otherwise) -- but they are not
# independent: `high_bits` is unread while the native flag is set, so crossing
# them as two axes would run identical configurations under two names. One named
# axis, mapped onto the pair by these helpers, keeps it at one value per run.
HIGH_TIER_MODES = ("fp16", "int16", "int8", "int4")
DEFAULT_HIGH_TIER = "fp16" if DEFAULT_HIGH_PRECISION_NATIVE else f"int{DEFAULT_HIGH_BITS}"


def high_precision_knobs(high_tier: str) -> tuple[int, bool]:
    """(high_bits, high_precision_native) for one value of the `high_tiers` axis."""
    if high_tier == "fp16":
        return DEFAULT_HIGH_BITS, True
    return int(str(high_tier).removeprefix("int")), False


def high_precision_of(high_bits: int, high_precision_native: bool) -> str:
    """Inverse of `high_precision_knobs`: the axis value a (high_bits, native) pair
    represents. Used to read the axis back out of a result row or a CSV, including
    rows written before the axis existed."""
    return "fp16" if high_precision_native else f"int{int(high_bits)}"


# The recency window w, in absolute tokens. w is a *count* of the most recent
# positions held unconditionally, not a fraction of the budget: it protects the
# tail of the context, and how many tokens that takes has nothing to do with how
# large k happens to be. `window_ratio` (w = ratio * k) remains as the legacy
# alternative and is what a point uses when `window_tokens` is None -- both are
# resolved in `MiKVPolicy.budget_for`, and w is clamped to k either way.
DEFAULT_WINDOW_TOKENS: int | None = None
