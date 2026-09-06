"""
The knobs you edit.

Two kinds of thing live here and nothing else does: the coarse value lists every
sweep draws from, and the named presets that combine them into a runnable pass.
Keeping them in one file is the point -- a preset, a CLI run and the docs cannot
disagree about what "the scoreboard sweep" means if there is only one place it is
written down.

They are deliberately coarse. Each benchmark run is ~3-4 min on a V100, so the
number of values on an axis is a budget rather than a preference: the exhaustive
cross product of what is below is 2304 configurations x 3 ratios = 6912 runs,
about 16.8 days of continuous GPU time. See SWEEP_PRESETS for the cheaper passes
and mikv_sweep.SWEEP_MODES for why they work.
"""

from mikv_config import (
    BALANCE_SCHEMES,
    BUDGET_MODES,
    DEFAULT_BALANCE_SCHEME,
    DEFAULT_BUDGET_MODE,
    DEFAULT_SCORE_DECAY_APPLICATION,
)

# ---------------------------------------------------------------------------
# The coarse sweep grids.
#
# These are the value lists every sweep draws from -- edit them here rather than
# at a call site, so a preset, a CLI run and the docs cannot disagree about what
# "the scoreboard sweep" means. They are deliberately coarse: each benchmark run
# is ~3-4 min on a V100, so the count of values is a budget, not a preference.
# ---------------------------------------------------------------------------

# Scoreboard: ONE composite axis, not four crossed. Its four fields (scheme,
# word length l, fraction f, signedness) only make sense in combination, so the
# legal settings are a list of 8:
#
#   fp32   -- the reference. Not a hardware proposal; it says what the fp16
#             accumulator was already costing, which is the baseline every
#             quantized variant has to be read against.
#   fp16   -- the model's own dtype ("native"), what this script has always used.
#   fixed  -- the IPU's real score path: a fixed-point word with a *static*
#             scale. 2 MSBs of the stored word are tier flags (HIGH / LOW /
#             MIGRATING), so a 16-bit word leaves l = 14 and a 32-bit word
#             leaves l = 30 for the number. Unsigned throughout -- the score is a
#             sum of softmax weights and provably non-negative, so a sign bit
#             would be a wasted bit of range.
#
# Choosing f is the whole design question for the fixed rows, and it is a
# straight range-vs-resolution trade against a score that is a running sum of
# softmax weights: the integer field must hold the largest score a sink token
# accumulates (up to ~t in the limit, so ~12 integer bits at a 4096-token
# context) while the fraction must resolve a single step's attention delta or
# that token's score never moves at all.
#
#   l = 14: int bits = 14 - f. f = 2/4/6 gives max 4096/1024/256 with resolution
#           2^-2/2^-4/2^-6. This word is *tight* -- f = 6 will saturate on sink
#           tokens and f = 2 flushes small deltas -- which is exactly why it is
#           worth measuring: it is the row that halves the score SRAM.
#   l = 30: int bits = 30 - f. f = 12/16/20 gives max 262144/16384/1024, all
#           comfortable except f = 20, which is included precisely to bracket
#           where saturation starts to bite.
SCOREBOARD_SWEEP = (
    dict(score_scheme="fp32"),
    dict(score_scheme="native"),
    dict(score_scheme="fixed", score_length_bits=14, score_frac_bits=2, score_signed=False),
    dict(score_scheme="fixed", score_length_bits=14, score_frac_bits=4, score_signed=False),
    dict(score_scheme="fixed", score_length_bits=14, score_frac_bits=6, score_signed=False),
    dict(score_scheme="fixed", score_length_bits=30, score_frac_bits=12, score_signed=False),
    dict(score_scheme="fixed", score_length_bits=30, score_frac_bits=16, score_signed=False),
    dict(score_scheme="fixed", score_length_bits=30, score_frac_bits=20, score_signed=False),
)

# The importance-set budget. Both modes take the same three ratios: under
# "fixed_length" they multiply t_p (the prompt length, frozen at prefill), under
# "fixed_ratio" they multiply the current t. The sweep crosses r with every
# configuration rather than treating it as an axis, so these two lists together
# are the 6 budget settings -- 2 modes x 3 ratios.
BUDGET_MODE_SWEEP = BUDGET_MODES              # ("fixed_length", "fixed_ratio")
BUDGET_RATIO_SWEEP = (0.25, 0.5, 0.75)

# LOW tier: the width evicted tokens are compressed to.
LOW_TIER_SWEEP = (2, 3, 4)

# HIGH tier: how the important tokens are stored. int16 and fp16 are genuinely
# different here -- uniform affine grid vs. fp16's logarithmic one; see
# `quantize_kv`.
HIGH_TIER_SWEEP = ("int8", "int16", "fp16")

# Recency window w, in ABSOLUTE tokens (not a fraction of k). Clamped to k, so
# on a short prompt at r = 0.25 the widest of these can meet or exceed k and
# degenerate to pure recency -- `budget_for` says so, and it is a real result,
# not a misconfiguration.
WINDOW_SWEEP = (32, 64, 128)

# Channel balancer.
BALANCER_SWEEP = BALANCE_SCHEMES              # ("paper", "pow2")

# ipu_age_lut ROM depth, in entries (each 16 bits, so 256 B / 512 B / 1 KB). Only
# read when score_decay_scheme="lut". The LUT replaces the age divide -- a
# hardware divider is iterative and multi-cycle, and the score path needs 16 of
# them retiring one beat per cycle -- at the cost of a small approximation error.
#
# Depth buys one thing only: the accuracy of folding an age into the table's
# octave, ~1/(W + D), which is 0.52% / 0.31% / 0.18% worst case at these three
# depths. There is no knee to find -- doubling the ROM halves the error forever --
# so the depth is chosen against a budget, and the budget is set downstream: the
# comparator's own output word rounds at roughly ten times that magnitude, so past
# some depth the ROM stops being what limits the decision. Finding that point is
# what sweeping this axis is for.
AGE_LUT_SWEEP = (128, 256, 512)


# ---- presets ----
#
# A preset is just a dict of `sweep_kv_compression` keyword arguments -- the
# named starting points worth running, so the common sweeps don't have to be
# retyped as a dozen --flags. Any individual axis flag on the command line
# overrides the preset's value for that axis and leaves the rest.
#
# Run counts below are configurations x ratios, i.e. benchmark runs; multiply by
# num_samples for generations. Check them with --dry-run before committing.
SWEEP_PRESETS: dict[str, dict] = {
    # What this script ran before the coarse grids existed: 2 budget modes x
    # 2 balancers x 3 ratios. Kept so the historical sweep stays reproducible by
    # name; it is not the default any more (DEFAULT_PRESET is "greedy").
    "legacy": dict(
        sweep_mode="grid",
        budget_modes=BUDGET_MODES,
        balance_schemes=BALANCE_SCHEMES,
        budget_ratios=BUDGET_RATIO_SWEEP,
    ),
    # --- the three ways of profiling the coarse grids, cheapest first ---
    #
    # (i) OFAT. Every axis moved one value at a time off a fixed baseline. Each
    # run answers "what does this one knob cost, holding everything else at the
    # reference design?" -- a clean, controlled marginal, blind to interactions.
    # 1 + (8-1)+(3-1)+(3-1)+(3-1)+(3-1)+(2-1)+(2-1) = 18 configurations x 3
    # ratios = 54 runs, ~3.1 h at 3.5 min/run. This is the right first pass.
    "ofat": dict(
        sweep_mode="ofat",
        budget_ratios=BUDGET_RATIO_SWEEP,
        # The baseline is the reference design every marginal is measured
        # against: the configuration the IPU implements.
        ofat_baseline=dict(
            balance_scheme="pow2",
            score_decay_scheme="lut",
            score_decay_application="ranking",
            # The RTL's own depth. Pinned rather than left to the axis's first
            # value, because the baseline has to be a configuration every other
            # axis's variations remain valid against: at D = 128 the w = 128 arm
            # violates D >= W + 2 and would be dropped from the sweep entirely.
            age_lut_entries=512,
        ),
        # The decay is held at the hardware's own setting rather than swept: it is
        # not one of the axes being profiled, and a single-valued axis is pinned by
        # the baseline above.
        score_decay_schemes=("lut",),
        score_decay_applications=("ranking",),
        budget_modes=BUDGET_MODE_SWEEP,
        balance_schemes=("pow2", "paper"),
        window_tokens_options=WINDOW_SWEEP,
        high_tiers=HIGH_TIER_SWEEP,
        low_bits_options=LOW_TIER_SWEEP,
        scoreboard_configs=SCOREBOARD_SWEEP,
        age_lut_entries_options=AGE_LUT_SWEEP,
    ),
    # (iii) Greedy coordinate descent. Same 54-run budget as OFAT, but each axis
    # is decided against the winners of the axes before it instead of against a
    # fixed baseline, so it recovers some interaction structure for free. The
    # axis order (GREEDY_AXIS_ORDER) settles the footprint-free axes first,
    # where "best" is unambiguous, before spending anything on the axes that
    # trade accuracy for cache. This is the recommended default.
    "greedy": dict(
        sweep_mode="greedy",
        greedy_objective="auto",
        budget_ratios=BUDGET_RATIO_SWEEP,
        ofat_baseline=dict(
            balance_scheme="pow2",
            score_decay_scheme="lut",
            score_decay_application="ranking",
            # The RTL's own depth. Pinned rather than left to the axis's first
            # value, because the baseline has to be a configuration every other
            # axis's variations remain valid against: at D = 128 the w = 128 arm
            # violates D >= W + 2 and would be dropped from the sweep entirely.
            age_lut_entries=512,
        ),
        score_decay_schemes=("lut",),
        score_decay_applications=("ranking",),
        budget_modes=BUDGET_MODE_SWEEP,
        balance_schemes=BALANCER_SWEEP,
        window_tokens_options=WINDOW_SWEEP,
        high_tiers=HIGH_TIER_SWEEP,
        low_bits_options=LOW_TIER_SWEEP,
        scoreboard_configs=SCOREBOARD_SWEEP,
        age_lut_entries_options=AGE_LUT_SWEEP,
    ),
    # (ii) Exhaustive. Every coarse grid crossed: 2 modes x 2 balancers x 8
    # scoreboards x 3 windows x 3 HIGH x 3 LOW x 3 LUT depths = 2304 valid
    # configurations x 3 ratios = 6912 runs (288 of the 2592 raw combinations are
    # dropped by D >= W + 2). At 3.5 min/run that is ~16.8 days of V100 time, and
    # that is the *coarse* grid -- it is here to be costed with --dry-run, not to
    # be launched. Confirm a greedy result with `confirm` instead.
    "exhaustive": dict(
        sweep_mode="grid",
        budget_ratios=BUDGET_RATIO_SWEEP,
        # The LUT *is* the decay in the reference design, so pin it: left at the
        # default "exact" this cross would never read the ROM and the depth axis
        # would collapse to a single point.
        score_decay_schemes=("lut",),
        score_decay_applications=("ranking",),
        budget_modes=BUDGET_MODE_SWEEP,
        balance_schemes=BALANCER_SWEEP,
        window_tokens_options=WINDOW_SWEEP,
        high_tiers=HIGH_TIER_SWEEP,
        low_bits_options=LOW_TIER_SWEEP,
        scoreboard_configs=SCOREBOARD_SWEEP,
        age_lut_entries_options=AGE_LUT_SWEEP,
    ),
    # --- targeted follow-ups, to be run after a first pass says what matters ---
    #
    # The two footprint-FREE axes crossed properly. Every point here sits at an
    # identical KV size, so accuracy differences are differences in selection
    # quality alone -- and "does the pow-2 balancer interact with a narrow
    # scoreboard?" is exactly the interaction neither OFAT nor greedy can see.
    # 2 x 8 x 3 LUT depths = 48 configurations x 3 ratios = 144 runs (~8.4 h).
    "score": dict(
        sweep_mode="grid",
        budget_ratios=BUDGET_RATIO_SWEEP,
        budget_modes=(DEFAULT_BUDGET_MODE,),
        balance_schemes=BALANCER_SWEEP,
        scoreboard_configs=SCOREBOARD_SWEEP,
        score_decay_schemes=("lut",),
        age_lut_entries_options=AGE_LUT_SWEEP,
        score_decay_applications=(DEFAULT_SCORE_DECAY_APPLICATION,),
    ),
    # The footprint axes crossed: how the budget is spent (mode, r, w) against
    # how the two tiers are stored. Nothing here is free -- every point is a
    # different KV size -- so read it off the Pareto front, not the mean.
    # 2 x 3 x 3 x 3 = 54 configurations x 3 ratios = 162 runs (~9.4 h).
    "budget": dict(
        sweep_mode="grid",
        budget_ratios=BUDGET_RATIO_SWEEP,
        budget_modes=BUDGET_MODE_SWEEP,
        window_tokens_options=WINDOW_SWEEP,
        high_tiers=HIGH_TIER_SWEEP,
        low_bits_options=LOW_TIER_SWEEP,
        balance_schemes=(DEFAULT_BALANCE_SCHEME,),
    ),
    # The confirmation grid: the axes a first pass most often finds decisive,
    # crossed exhaustively to check that greedy's path did not hide an
    # interaction. Narrow it to whichever axes actually moved -- via the CLI
    # flags -- before spending it. 8 x 3 x 3 = 72 configurations x 3 ratios
    # = 216 runs (~12.6 h), so it is a deliberate second pass, not a default.
    "confirm": dict(
        sweep_mode="grid",
        budget_ratios=BUDGET_RATIO_SWEEP,
        budget_modes=(DEFAULT_BUDGET_MODE,),
        balance_schemes=("pow2",),
        scoreboard_configs=SCOREBOARD_SWEEP,
        window_tokens_options=WINDOW_SWEEP,
        low_bits_options=LOW_TIER_SWEEP,
        high_tiers=("fp16",),
        score_decay_schemes=("lut",),
    ),
}
DEFAULT_PRESET = "greedy"
