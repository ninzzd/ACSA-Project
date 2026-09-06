"""
The quantizers, and only the quantizers.

Three different error models live here and they are not interchangeable:

- `quantize_kv` -- a group-wise *affine* quantizer for K/V, with a data-dependent
  (min, max) per group. This is the KV cache's HIGH and LOW tiers.
- `quantize_scores` -- the same affine machinery applied to the importance
  scoreboard, one scale per head.
- `quantize_fixed_point` / `age_lut_reciprocal` -- *fixed point* with a static
  scale, plus a reciprocal ROM. These model the IPU's real score datapath, which
  is exact for large values and flushes small ones to zero, rather than
  rescaling to fit them the way an affine quantizer does.

All of them are fake quantizers: they round-trip through a grid and dequantize
back to the input dtype, so they measure a width's effect on *accuracy* without
shrinking any real allocation. `kv_cache_size_bytes` (mikv_bench) computes what
the footprint would be if the tiers were genuinely stored at their widths.
"""

import math

from mikv_runtime import torch
from mikv_config import (
    DEFAULT_SCORE_BITS,
    HW_AGE_LUT_ENTRIES,
    HW_AGE_LUT_RECENT_WINDOW,
    HW_AGE_LUT_WORD_BITS,
    HW_SCORE_FRAC_BITS,
)

def quantize_kv(tensor: torch.Tensor, bits: int = 2, group_size: int | None = None) -> torch.Tensor:
    """
    Fake-quantize K/V vectors: round-trip them through a `bits`-wide affine
    quantizer and dequantize back to the input dtype. Quantization is
    per-group along the last dim -- `group_size` consecutive channels share
    one (min, max) pair; `group_size=None` splits `head_dim` into two
    groups. Does not include the channel biasing factor b.

    tensor: (..., head_dim)

    The short-circuit is at bits > 16, NOT >= 16, so `bits=16` is a genuine
    16-bit affine round-trip rather than the identity. That matters because the
    HIGH tier offers int16 and fp16 as *different* options (HIGH_TIER_MODES) and
    they have to be measurably different, which they are: an affine grid is
    uniform over the group's [min, max] while fp16's is logarithmic, so the two
    agree near the group maximum -- where fp16's spacing, max * 2^-11, is the
    coarser of the two -- and diverge for the small-magnitude channels, where a
    uniform (max - min) / 65535 step is far coarser than fp16 resolves. Native
    fp16 is reached by not calling this at all (`high_precision_native`), not by
    passing bits=16.
    """
    if bits > 16:
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


def age_lut_frac_bits(
    word_bits: int = HW_AGE_LUT_WORD_BITS,
    recent_window: int = HW_AGE_LUT_RECENT_WINDOW,
) -> int:
    """
    F, the number of fractional bits in the stored reciprocals:

        F = floor(log2((2^B - 1) * n_min)),   n_min = recent_window + 1

    F is bounded by the LARGEST entry in the ROM, which sits at the SMALLEST age
    it can be asked about -- and that is not age 1. The recency window means no
    token younger than `recent_window` ever reaches the comparator, so the table
    never needs those ages and F rises accordingly: at B = 16, W = 64 it is 22
    rather than the 15 a table starting at age 1 would permit. Seven extra
    fractional bits out of a masking rule that already existed.
    """
    n_min = recent_window + 1
    return int(math.floor(math.log2((2**word_bits - 1) * n_min)))


def age_lut_fold(
    distance: torch.Tensor,
    entries: int = HW_AGE_LUT_ENTRIES,
    recent_window: int = HW_AGE_LUT_RECENT_WINDOW,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fold an age into the octave the ROM covers: return (n', e) with
    n' = round(age / 2^e) landing in [n_min, n_min + entries - 1].

    The table stops at n_min + entries - 1 but ages run to the full context
    length. The gap is closed by the self-similarity of 1/n under powers of two
    -- there is no entry for 1000 because 1000 is 500 doubled and 1/1000 is 1/500
    halved -- so the table covers one octave and everything larger is halved until
    it lands back inside. The halving is not lost: `e` is added to the output
    shift by the caller.

    e is found by halving until n' fits rather than from a bit-length rule
    (`e = max(0, bit_length(age) - log2(entries))`). The bit-length rule is one
    comparator cheaper and is what the RTL note proposes, but it only ever
    generates addresses in the lower 7/8 of the table, leaving the top entries
    unread and costing ~0.02 percentage points of worst-case error; more
    importantly it is derived for one specific depth and does not generalize
    across the depths this is swept over. Halving-until-it-fits is the tighter
    rule the RTL note's own measured error figures use.

    Round n' rather than truncating, for the same reason the table itself rounds:
    truncation biases every fold the same direction, and the bias grows with age,
    which systematically favours old tokens -- a policy change disguised as a
    rounding choice.
    """
    n_min = recent_window + 1
    top = n_min + entries - 1
    age = distance.double().clamp(min=1.0)
    e = torch.zeros_like(age)
    n_prime = age.clone()
    # age <= T_MAX and top >= 2 * n_min, so this converges in a handful of passes;
    # the bound is a guard, not the mechanism.
    for _ in range(32):
        over = n_prime > top
        if not bool(over.any()):
            break
        e = e + over.double()
        # Recomputed from the ORIGINAL age at the new shift, not halved
        # iteratively: n' = round(age / 2^e), which is what the RTL computes as
        # (age + (1 << (e-1))) >> e. Iterated halving would round repeatedly.
        n_prime = torch.floor(age / torch.pow(2.0, e) + 0.5)
    return n_prime, e


def age_lut_table(
    entries: int = HW_AGE_LUT_ENTRIES,
    word_bits: int = HW_AGE_LUT_WORD_BITS,
    recent_window: int = HW_AGE_LUT_RECENT_WINDOW,
) -> torch.Tensor:
    """
    The ROM contents: R[a] = round(2^F / (a + n_min)) for a in [0, entries).

    Exposed separately from the lookup so a test can assert the two properties
    the RTL testbench asserts -- entries monotonically non-increasing (otherwise
    an older token could get a LARGER scale factor than a younger one, corrupting
    the ranking) and exact at powers of two.
    """
    frac_bits = age_lut_frac_bits(word_bits, recent_window)
    n_min = recent_window + 1
    n = torch.arange(n_min, n_min + entries, dtype=torch.float64)
    rom = torch.round(2.0**frac_bits / n)
    return rom.clamp(0.0, float(2**word_bits - 1))


def age_lut_reciprocal(
    distance: torch.Tensor,
    entries: int = HW_AGE_LUT_ENTRIES,
    word_bits: int = HW_AGE_LUT_WORD_BITS,
    recent_window: int = HW_AGE_LUT_RECENT_WINDOW,
) -> torch.Tensor:
    """
    The approximation of 1/age that `ipu_age_lut` actually produces: R / 2^(F+e),
    with R read from the ROM at the folded address and e the fold's shift.

    Depth buys exactly one thing. There are two error sources and they do not
    interact: rounding the stored entry, which is bounded by n / 2^(F+1) and is
    ~0.007% at the top of a 512-deep table, i.e. negligible; and rounding `age`
    to n' during the fold, which dominates and is ~1/(recent_window + entries).
    Only the second depends on depth. Note the error does NOT grow with age --
    each further octave halves the age and halves the rounding error with it --
    so accuracy per byte is a straight line on log-log with no knee. Doubling the
    ROM halves the error forever, which means a depth is chosen against an error
    budget, never by finding where the curve bends.

    What actually bounds the useful depth is downstream: the comparator output is
    a finite fixed-point word, and its rounding is roughly an order of magnitude
    noisier than the reciprocal's. That is why `age_lut_cmp_val` models the whole
    datapath rather than just this ROM -- see its docstring.
    """
    frac_bits = age_lut_frac_bits(word_bits, recent_window)
    n_prime, e = age_lut_fold(distance, entries, recent_window)
    rom = age_lut_table(entries, word_bits, recent_window).to(distance.device)
    address = (n_prime - (recent_window + 1)).clamp(0, entries - 1).long()
    return rom[address] / torch.pow(2.0, frac_bits + e)


def age_lut_cmp_val(
    scores: torch.Tensor,
    distance: torch.Tensor,
    entries: int = HW_AGE_LUT_ENTRIES,
    word_bits: int = HW_AGE_LUT_WORD_BITS,
    recent_window: int = HW_AGE_LUT_RECENT_WINDOW,
    out_frac_bits: int = HW_SCORE_FRAC_BITS,
) -> torch.Tensor:
    """
    The full comparator path: cmp_val = (S * R) >> (F + e), with the result
    landing on the 2^-`out_frac_bits` grid of the comparator's own word.

    Two things here are deliberate and both come straight from the RTL note.

    **Shift the product, not the reciprocal.** `(S * R) >> (F + e)`, never
    `S * (R >> e)`. Pre-shifting the ROM word discards up to 4 of its 16 bits.

    **Model the output word, not just the ROM.** The comparison value is a
    fixed-point register, and rounding it is a second error source *downstream*
    of the table. It is the larger one: a Monte Carlo over the full integer
    datapath puts victim disagreement at ~1.2% for every depth from 128 to 2048,
    while the reciprocal's own contribution at depth 512 is ~0.07%. Modelling the
    ROM alone would therefore misattribute the error and make depth look far more
    decisive than it is -- the whole point of sweeping depth is to find where it
    stops mattering, and it stops mattering once it disappears under this floor.

    Truncating (floor) rather than rounding on the output shift, because that is
    what a `>>` does.
    """
    frac_bits = age_lut_frac_bits(word_bits, recent_window)
    n_prime, e = age_lut_fold(distance, entries, recent_window)
    rom = age_lut_table(entries, word_bits, recent_window).to(scores.device)
    address = (n_prime - (recent_window + 1)).clamp(0, entries - 1).long()
    reciprocal = rom[address]

    out_scale = float(2**out_frac_bits)
    # S as it sits in the score register, an integer count of 2^-out_frac_bits.
    s_fixed = torch.round(scores.double() * out_scale)
    shifted = torch.floor(s_fixed * reciprocal / torch.pow(2.0, frac_bits + e))
    return shifted / out_scale
