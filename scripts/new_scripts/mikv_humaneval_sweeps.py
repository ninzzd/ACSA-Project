"""
The HumanEval sweeps: the knobs you edit, the way mikv_gsm8k_sweeps is for GSM8K.

HumanEval is 164 problems, so every run is the whole set (MiKV's protocol) --
there is no screen subset, and every pair of runs is a paired comparison on
identical inputs (McNemar p in the summary). At a baseline near 12% pass@1
(~20/164) differences are a few problems wide: read the lost/gained counts and
the pass matrix, not just the aggregate.

The spec's four policies map onto this codebase as:

  full   the uncompressed baseline                                  (baseline)
  rtn    uniform round-to-nearest: HIGH and LOW at the same width,
         so the importance policy decides nothing                   (rtn)
  mikv   the reference configuration below, r in 0.2-0.5           (H1)
  h2o    NOT available: the H2O hard-eviction flag exists only as
         scripts/line_retrieval_scripts/eviction_baseline.patch and
         was never ported to scripts/mikv_policy.py

--- the reference ---

The GSM8K reference (pow2 balancer, fp16 HIGH, Q30.16 score, LUT-512 decay) with
one change: the window is w = k/2 instead of w = 64. HumanEval prompts are ~130
tokens, so at r <= 0.5 k is under 64 almost everywhere and w = 64 would make
every run pure recency (k_H = 0). H4 keeps w = 64 as that control.
`humaneval_sweeps.sh dry` prints, per (sweep, r), how many problems degenerate.
"""

FULL = dict(problem_limit=None)

COMMON = dict(
    budget_mode=("fixed_length",),  # k = floor(r * t_p), frozen at prefill
    score_signed=(False,),
    score_decay_application=("ranking",),
)
SCORE_Q1416 = dict(score_scheme=("fixed",), score_length_bits=(30,), score_frac_bits=(16,))
REFERENCE = dict(
    **SCORE_Q1416,
    balance_scheme=("pow2",),
    high_tier=("fp16",),
    window_tokens=(None,),
    window_ratio=(0.5,),
    score_decay_scheme=("lut",),
    age_lut_entries=(512,),
)
RATIOS = (0.2, 0.3, 0.4, 0.5)   # MiKV Figure 6's x-axis range
ABLATION_LOW = (3,)
ABLATION_RATIOS = (0.2, 0.5)


def _axes(**axes) -> dict:
    return {**COMMON, **axes}


def _ref(**overrides) -> dict:
    return _axes(**{**REFERENCE, **overrides})


# `axes` is one axis dict (crossed as a grid) or a tuple of them (their union --
# for pairs that must move together and not be crossed).
HUMANEVAL_SWEEPS: dict[str, dict] = {
    # The uncompressed model: reference line for every plot and paired statistic.
    # Llama-2-7b(-chat) should land around 11-14% pass@1.
    "baseline": dict(FULL, baseline=True),

    # 10 problems, baseline + one MiKV point: measures s/problem for both paths
    # and shows completions to eyeball (indentation!) before spending the sweep.
    # A prefix of the full runs, so its rows are reused.
    "probe": dict(problem_limit=10, baseline=True,
                  axes=_ref(low_bits=(4,)), budget_ratios=(0.3,)),

    # Acceptance check: LOW = 16 bits is the identity quantizer and the pow2
    # balancer is an exact exponent shift, so this must match the baseline
    # problem for problem. Any lost/gained here means the pipeline, not the cache.
    "sanity": dict(FULL, axes=_ref(low_bits=(16,)), budget_ratios=(0.2,)),

    # RTN: uniform int8 and int4 (HIGH width == LOW width). Independent of r.
    "rtn": dict(FULL, axes=(_ref(high_tier=("int8",), low_bits=(8,)),
                            _ref(high_tier=("int4",), low_bits=(4,))),
                budget_ratios=(0.2,)),

    # H1  MiKV operating curve: LOW int3/int4 x r 0.2-0.5.          8 runs
    "H1": dict(FULL, axes=_ref(low_bits=(3, 4)), budget_ratios=RATIOS),

    # H2  Channel balancer: paper (sqrt) vs pow2.                    2 new
    "H2": dict(FULL, axes=_ref(low_bits=ABLATION_LOW, balance_scheme=("paper", "pow2")),
               budget_ratios=ABLATION_RATIOS),

    # H3  Importance policy: float reference (fp32 score, exact 1/age) vs the
    #     hardware path (Q30.16 score, LUT-512). Paired, not crossed.  2 new
    "H3": dict(FULL, axes=(_ref(low_bits=ABLATION_LOW, score_scheme=("fp32",),
                                score_decay_scheme=("exact",)),
                           _ref(low_bits=ABLATION_LOW)),
               budget_ratios=ABLATION_RATIOS),

    # H4  Window: the RTL's w = 64 (pure recency on nearly every problem here)
    #     against the w = k/2 reference.                              4 new
    "H4": dict(FULL, axes=(_ref(low_bits=ABLATION_LOW, window_tokens=(64,)),
                           _ref(low_bits=ABLATION_LOW)),
               budget_ratios=RATIOS),

    # H5  Is tiering needed at all? r = 0.05 quantizes nearly the whole cache
    #     (k = 6, w = 3, k_H = 3 at the median t_p = 136; k = max(1, .) floors it).
    #     Crossed with r 0.2/0.5 so every LOW width has a tiered point to compare
    #     against; int2 is the width where uniform quantization should finally
    #     break. int3/int4 at 0.2/0.5 are shared with H1.              5 new
    #     int2 may run to the token cap: submit with SEC_PER_PROBLEM=60.
    "H5": dict(FULL, axes=_ref(low_bits=(2, 3, 4)), budget_ratios=(0.05, 0.2, 0.5)),
}

HUMANEVAL_GROUPS: dict[str, tuple[str, ...]] = {
    "tier1": ("baseline", "sanity", "rtn", "H1"),
    "all": ("baseline", "sanity", "rtn", "H1", "H2", "H3", "H4"),
}


def expand_group(group: str) -> tuple[str, ...]:
    names: list[str] = []
    for part in (p.strip() for p in group.split(",") if p.strip()):
        for name in HUMANEVAL_GROUPS.get(part, (part,)):
            if name not in HUMANEVAL_SWEEPS:
                raise KeyError(f"unknown sweep or group {name!r}; sweeps: {', '.join(HUMANEVAL_SWEEPS)}; "
                               f"groups: {', '.join(HUMANEVAL_GROUPS)}")
            if name not in names:
                names.append(name)
    return tuple(names)
