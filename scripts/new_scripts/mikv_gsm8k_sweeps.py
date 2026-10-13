"""
The GSM8K sweeps: the knobs you edit, the way mikv_grids is for Line Retrieval.

Two versions of each follow-up live here:

  S1-S6       the LEAN screen, the default. ~20 MiKV runs, ~40 GPU-hours.
  S1_full...  the Line Retrieval follow-ups (followup_sweeps.sh, after the
  S6_full     confirm run 20260910_164327) value for value. ~80 runs. Kept for
              adding points back where a lean result is borderline.

Every lean run is a configuration the full sweep also contains, so the lean rows
still line up with Line Retrieval rows (`mikv_report.row_config` is identical for
both; `mikv_gsm8k.py merge --line-retrieval-csv` joins them), and anything
submitted later from a _full sweep reuses the lean runs rather than repeating them.

--- why the lean screen is enough for the RTL ---

The axes split the way mikv_sweep.FOOTPRINT_AXES says. What sizes the SRAM --
LOW width and r -- is swept properly (S1: 2 widths x 4 ratios). The rest are
hardware *cost* knobs that never change the footprint: the balancer, the age-LUT
depth, the decay, the score word, the window CSR. For those the RTL question is
one yes/no per knob -- does the cheap option hold at the operating point we will
build? -- and one well-chosen point answers it:

  LOW int3, r = 0.25, the stress point. It is the harsher of the two LOW
  candidates S1 decides between, so a cheap option that holds here holds at int4
  too (less quantization pressure means selection quality matters less); and
  at r = 0.25 there are ~160 top-k slots beyond the w = 64 window, so selection
  is actually doing work.

What the full sweeps add is the same yes/no repeated at int2 and at r = 0.5 --
robustness, not a different answer -- plus in-between values (LUT 256) that only
matter if the endpoints disagree. The number of problems is NOT cut: 250 per run
is what makes a paired comparison between two runs meaningful at all.

S7 is the safeguard: the final combination -- operating point plus the cheap
options chosen from S2-S6 -- is itself measured on all 1319 problems, so the
configuration that goes into the RTL is verified directly, not inferred.

If a lean result is borderline (|delta_accuracy| of a few points with
mcnemar_p < ~0.3), submit the matching _full sweep: only its extra runs execute.

--- the problem sets ---

SCREEN is one fixed set of 250 test problems (seed 0), shared by S1-S6: every
configuration is measured on the *same* problems, so a difference between two
rows is a paired comparison (McNemar p in the summary) rather than two noisy
samples. At a baseline accuracy near 25%, 250 problems give a 95% CI of about
+/- 5 points on one run -- too wide to rank close configurations on accuracy
alone, but the paired test sees much smaller differences, because most problems
are answered the same way by both configurations and only the discordant ones
carry information. FULL is all 1319, for S7 and its baseline.

A run's result is keyed on its configuration, not on the sweep, so a
configuration shared by two sweeps is computed once: S2-S6 all contain the
reference point (pow2 / fp16 / w=64 / lut512 / Q30.16 / int3 / r=0.25) that S1
already runs, and S7 reuses the 250 screen problems S1 ran and only computes the
other 1069.

--- t_p ---

The 8-shot prompt is ~0.8-0.9k tokens plus the question (`gsm8k_sweeps.sh dry`
prints the measured distribution and each ratio's k). That is shorter than Line
Retrieval's t_p = 1106, so k = floor(r * t_p) is smaller at the same r -- which
is why the plan flags any (r, w) where k <= w.
"""

SCREEN = dict(num_problems=250, problem_seed=0)
FULL = dict(num_problems=0, problem_seed=0)  # 0 = the whole test set

# Axes that stay fixed in every sweep below (followup_sweeps.sh's COMMON).
COMMON = dict(
    budget_mode=("fixed_length",),
    score_signed=(False,),
    score_decay_application=("ranking",),
)
# The recommended score path from the confirm analysis (SCORE_Q1416).
SCORE_Q1416 = dict(score_scheme=("fixed",), score_length_bits=(30,), score_frac_bits=(16,))
# The reference hardware configuration every sweep varies one knob away from.
REFERENCE = dict(
    **SCORE_Q1416,
    balance_scheme=("pow2",),
    high_tier=("fp16",),
    window_tokens=(64,),
    score_decay_scheme=("lut",),
    age_lut_entries=(512,),
)
# The lean screen's single point for the footprint-free knobs (see docstring).
STRESS_LOW = (3,)
STRESS_RATIOS = (0.25,)


def _axes(**axes) -> dict:
    return {**COMMON, **axes}


def _ref(**overrides) -> dict:
    """The reference configuration with `overrides` swept in place of its values."""
    return _axes(**{**REFERENCE, **overrides})


GSM8K_SWEEPS: dict[str, dict] = {
    # ------------------------------------------------------------ reference --
    # The uncompressed model on the screen set: the reference every screen run's
    # paired statistics are computed against, and a sanity check of the prompt
    # and the extraction (Llama-2-7b-chat 8-shot CoT should land in the low-to-mid
    # 20s). S7 extends it to the full test set itself.
    "baseline": dict(SCREEN, baseline=True),

    # One configuration on 20 screen problems, plus the baseline on the same 20:
    # measures s/problem for both paths (the submit step reads it back to size
    # --time) and shows a few completions to eyeball before spending the sweep.
    # Its rows are a prefix of S1's and are reused, not recomputed.
    "probe": dict(
        SCREEN, problem_limit=20, baseline=True,
        axes=_ref(low_bits=(4,)),
        budget_ratios=(0.25,),
    ),

    # ----------------------------------------------------- lean tier 1 (S1-S5) --
    # S1  Operating point -- the footprint decision, so swept in full: int3 vs
    #     int4, and how small B_hi can go. r = 0.05 gives k ~ 40-45 < w = 64, so
    #     the policy degenerates to pure recency: the "does the IPU matter at
    #     all" control.
    #     2 LOW x 4 r = 8 runs
    "S1": dict(
        SCREEN,
        axes=_ref(low_bits=(3, 4)),
        budget_ratios=(0.05, 0.1, 0.15, 0.25),
    ),

    # S2  HIGH tier precision: fp16 vs int8 vs int4, at both LOW candidates.
    #     int4/int4 is uniform int4 -- the no-policy control; it gives the same
    #     result at every r, so one ratio is all it needs. Decides HIGH-tier
    #     storage and whether the append path needs a quantizer.
    #     3 HIGH x 2 LOW x r 0.25 = 6 runs, 2 shared with S1 -> 4 new
    "S2": dict(
        SCREEN,
        axes=_ref(high_tier=("fp16", "int8", "int4"), low_bits=(3, 4)),
        budget_ratios=STRESS_RATIOS,
    ),

    # S3  Comparator path (ipu_age_lut): exact divide vs 512- and 128-entry ROM vs
    #     pow2 (shift only: no ROM, no 16 x (30x16) multipliers). 256 is dropped:
    #     if 128 matches 512 it is moot, and if not, S3_full measures it.
    #     4 decay configs at the stress point, 1 shared -> 3 new
    "S3": dict(
        SCREEN,
        axes=_ref(low_bits=STRESS_LOW, score_decay_scheme=("exact", "pow2", "lut"),
                  age_lut_entries=(128, 512)),
        budget_ratios=STRESS_RATIOS,
    ),

    # S4  Recency-window CSR range (ipu_mask_gen / cfg_w). W < 64 isn't legal with
    #     the ROM as built (n_min = 65). w = 256 is left to S4_full: at r = 0.25 on
    #     GSM8K's ~0.9k-token prompts k < 256, so it is pure recency and says
    #     nothing about the window (the plan's k preview shows it).
    #     2 w at the stress point, 1 shared -> 1 new
    "S4": dict(
        SCREEN,
        axes=_ref(low_bits=STRESS_LOW, window_tokens=(64, 128)),
        budget_ratios=STRESS_RATIOS,
    ),

    # S5  Channel balancer: paper (sqrt, real multiply) vs pow2 (exponent add).
    #     2 at the stress point, 1 shared -> 1 new
    "S5": dict(
        SCREEN,
        axes=_ref(low_bits=STRESS_LOW, balance_scheme=("paper", "pow2")),
        budget_ratios=STRESS_RATIOS,
    ),

    # --------------------------------------------------------- lean tier 2 --
    # S6  Score format: fp32 reference vs the 30-bit fixed-point word at
    #     f = 12 / 16 / 20. On Line Retrieval the scoreboard saw a prefill and a
    #     few decode steps; here it accumulates over 100-250 decode steps on top
    #     of a ~0.9k-token prefill, so a sink token's score approaches the
    #     f = 20 rail (max 1024) and small per-step deltas test the f = 12 LSB.
    #     4 formats at the stress point, 1 shared -> 3 new
    "S6": dict(
        SCREEN,
        axes=_ref(low_bits=STRESS_LOW, score_scheme=("fp32", "fixed"),
                  score_frac_bits=(12, 16, 20)),
        budget_ratios=STRESS_RATIOS,
    ),

    # S7  Sign-off on the full test set (1319), with the full-set baseline.
    #     EDIT to the final combination before submitting: the LOW / r S1 picks,
    #     AND the cheap options S2-S6 accepted (e.g. decay pow2, lut128, paper vs
    #     pow2), so the exact configuration going into the RTL is what is
    #     measured. Runs S1 already covers resume from its 250 problems.
    #     2 runs x 1319, plus the baseline's other 1069 problems
    "S7": dict(
        FULL, baseline=True,
        axes=_ref(low_bits=(3, 4)),
        budget_ratios=(0.25,),
    ),

    # ------------------------------------------------ the full follow-ups --
    # followup_sweeps.sh value for value. Submit one only to add points back where
    # the lean screen is borderline; runs the lean screen already did are reused.
    "S1_full": dict(SCREEN, axes=_ref(low_bits=(3, 4)), budget_ratios=(0.05, 0.1, 0.15, 0.25)),
    "S2_full": dict(SCREEN, axes=_ref(high_tier=("fp16", "int8", "int4"), low_bits=(2, 3, 4)),
                    budget_ratios=(0.25, 0.5)),
    "S3_full": dict(SCREEN, axes=_ref(low_bits=(2, 3), score_decay_scheme=("exact", "pow2", "lut"),
                                      age_lut_entries=(128, 256, 512)),
                    budget_ratios=(0.25, 0.5)),
    "S4_full": dict(SCREEN, axes=_ref(low_bits=(2, 3), window_tokens=(64, 128, 256)),
                    budget_ratios=(0.25, 0.5)),
    "S5_full": dict(SCREEN, axes=_ref(low_bits=(2, 3), balance_scheme=("paper", "pow2")),
                    budget_ratios=(0.25, 0.5)),
    "S6_full": dict(SCREEN, axes=_ref(low_bits=(2, 3, 4), score_scheme=("fp32", "fixed"),
                                      score_frac_bits=(12, 16, 20)),
                    budget_ratios=(0.1, 0.25)),
}

# Named groups for submission. A group is planned as ONE job array whose units are
# deduplicated across its sweeps, so shared configurations run once.
GSM8K_GROUPS: dict[str, tuple[str, ...]] = {
    "tier1": ("baseline", "S1", "S2", "S3", "S4", "S5"),
    "all": ("baseline", "S1", "S2", "S3", "S4", "S5", "S6"),
    "full": ("baseline", "S1_full", "S2_full", "S3_full", "S4_full", "S5_full", "S6_full"),
}


def expand_group(group: str) -> tuple[str, ...]:
    """A group name, a sweep name, or a comma-separated mix of both, as the list of
    sweeps it covers -- deduplicated, in the order given."""
    names: list[str] = []
    for part in (p.strip() for p in group.split(",") if p.strip()):
        members = GSM8K_GROUPS.get(part, (part,))
        for name in members:
            if name not in GSM8K_SWEEPS:
                raise KeyError(
                    f"unknown sweep or group {name!r}; sweeps: {', '.join(GSM8K_SWEEPS)}; "
                    f"groups: {', '.join(GSM8K_GROUPS)}"
                )
            if name not in names:
                names.append(name)
    return tuple(names)
