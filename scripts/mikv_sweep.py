"""
The sweep engine: what a configuration *is* (`SweepPoint`), how a set of them is
enumerated from a set of axes (`enumerate_sweep_points`), the three ways of
walking that space (SWEEP_MODES), and the driver that benchmarks each point and
turns it into a result row (`sweep_kv_compression`).

Nothing here knows anything about attention, quantization or the model -- it
calls `run_line_retrieval_benchmark` and `kv_cache_size_bytes` and is otherwise
pure bookkeeping over configurations. The value lists it is usually fed live in
mikv_grids; the reports it feeds live in mikv_report.

The axes divide sharply, and the split is what makes a sweep worth running:
FOOTPRINT_AXES change how much cache is kept, so they trade accuracy against KV
size; everything else -- the balancer, the scoreboard, the decay -- changes only
*which* tokens are kept and how faithfully they are ranked, never how many, so it
moves accuracy at an identical footprint. A cheaper value on one of those is free
in area terms exactly when its accuracy holds.
"""

import datetime
import inspect
import itertools
from dataclasses import dataclass, replace

from mikv_config import (
    BALANCE_SCHEMES,
    BUDGET_MODES,
    DEFAULT_BALANCE_SCHEME,
    DEFAULT_BUDGET_MODE,
    DEFAULT_HIGH_TIER,
    DEFAULT_LOW_BITS,
    DEFAULT_SCORE_BITS,
    DEFAULT_SCORE_DECAY_APPLICATION,
    DEFAULT_SCORE_DECAY_SCHEME,
    DEFAULT_SCORE_SCHEME,
    DEFAULT_WINDOW_RATIO,
    DEFAULT_WINDOW_TOKENS,
    HIGH_TIER_MODES,
    HW_AGE_LUT_ENTRIES,
    HW_AGE_LUT_RECENT_WINDOW,
    HW_DELTA_BITS,
    HW_SCORE_FRAC_BITS,
    HW_SCORE_LENGTH_BITS,
    HW_SCORE_SIGNED,
    MODEL_NAME,
    SCORE_DECAY_APPLICATIONS,
    SCORE_DECAY_SCHEMES,
    SCORE_SCHEMES,
    format_score_tag,
    high_precision_knobs,
)
from mikv_quant import fixed_point_range
from mikv_bench import (
    kv_cache_size_bytes,
    resolve_budget_k,
    run_line_retrieval_benchmark,
    run_line_retrieval_no_quant,
)
from mikv_report import write_results_csv

# How `sweep_kv_compression` walks the axes it is given:
#   "grid" -- the full cartesian product of every axis. Exhaustive, and the only
#             thing that can show an *interaction* between two axes, but the run
#             count multiplies: eight axes at three values each is 3^8 = 6561
#             benchmark runs, which at a few minutes apiece is not a sweep anyone
#             finishes. Use it on two or three axes at a time.
#   "ofat"  -- one factor at a time: start from a baseline configuration and vary
#             exactly one axis away from it per run, so the cost is
#             1 + sum(len(axis) - 1) rather than the product -- the same eight
#             axes at three values each is 17 runs, not 6561. It measures each
#             axis's effect *at the baseline* and cannot see interactions, which
#             is the trade; it is the right first pass for "which knob costs
#             accuracy and which is free", and a grid over the two or three knobs
#             that turned out to matter is the right second pass.
# Both modes cross their configurations with the full `budget_ratios` list rather
# than treating r as just another axis: r is the x axis of every accuracy-vs-
# compression curve, so each configuration needs the whole curve, not one point.
SWEEP_MODES = ("grid", "ofat", "greedy")
DEFAULT_SWEEP_MODE = "grid"


# The axes a sweep can vary, in the order a point is printed and a grid is
# walked. The budget ratio r is deliberately NOT here: every other axis picks a
# *configuration*, and r is the x axis swept within each one (see SWEEP_MODES),
# so it is crossed with these rather than being one of them.
SWEEP_AXES = (
    "budget_mode",
    "balance_scheme",
    "window_tokens",
    "window_ratio",
    "high_tier",
    "low_bits",
    "score_scheme",
    "score_bits",
    "score_decay_scheme",
    "score_decay_application",
    "score_length_bits",
    "score_frac_bits",
    "score_signed",
    "age_lut_entries",
)


# `sweep_kv_compression` keyword -> SWEEP_AXES name. The two name spaces differ
# (the keywords are plural, and three carry an `_options` suffix to avoid
# colliding with the scalar policy arguments of the same name), so the mapping is
# spelled out rather than derived. One table, used to build the axis dict for
# both the real sweep and the --dry-run plan, so an axis cannot be wired into one
# and forgotten in the other -- which would make --dry-run under-report the very
# run count it exists to report.
_SWEEP_KWARG_TO_AXIS = {
    "budget_modes": "budget_mode",
    "balance_schemes": "balance_scheme",
    "window_tokens_options": "window_tokens",
    "window_ratios": "window_ratio",
    "high_tiers": "high_tier",
    "low_bits_options": "low_bits",
    "score_schemes": "score_scheme",
    "score_bits_options": "score_bits",
    "score_decay_schemes": "score_decay_scheme",
    "score_decay_applications": "score_decay_application",
    "score_length_bits_options": "score_length_bits",
    "score_frac_bits_options": "score_frac_bits",
    "score_signed_options": "score_signed",
    "age_lut_entries_options": "age_lut_entries",
}


def axes_from_sweep_kwargs(kwargs: dict) -> dict[str, tuple]:
    """
    The axis dict `enumerate_sweep_points` takes, from a dict of
    `sweep_kv_compression` keyword arguments.

    Anything absent falls back to that function's *own* signature default rather
    than to a second copy of the defaults kept here -- so `--dry-run` plans the
    same sweep the run would perform, by construction rather than by agreement.
    """
    defaults = inspect.signature(sweep_kv_compression).parameters
    axes = {
        axis: tuple(
            kwargs[kwarg] if kwargs.get(kwarg) else defaults[kwarg].default
        )
        for kwarg, axis in _SWEEP_KWARG_TO_AXIS.items()
    }
    if kwargs.get("scoreboard_configs"):
        # Mirrors what `sweep_kv_compression` does with the composite axis: it
        # replaces the per-field score axes rather than joining them.
        for field in ("score_scheme", "score_bits", "score_length_bits",
                      "score_frac_bits", "score_signed"):
            axes.pop(field, None)
        axes["scoreboard"] = tuple(kwargs["scoreboard_configs"])
    return axes


@dataclass(frozen=True)
class SweepPoint:
    """
    One configuration of the MiKV policy: every knob except the budget ratio.

    Frozen and hashable on purpose -- `canonical()` maps configurations that are
    the *same experiment* onto one another, and the sweep then dedupes on
    equality. Without that, a grid crossing (say) score_scheme with score_bits
    runs the 4-, 8- and 12-bit variants of `native` as three separate hours of
    GPU time and reports three identical numbers.
    """

    budget_mode: str = DEFAULT_BUDGET_MODE
    balance_scheme: str = DEFAULT_BALANCE_SCHEME
    window_tokens: int | None = DEFAULT_WINDOW_TOKENS
    window_ratio: float = DEFAULT_WINDOW_RATIO
    high_tier: str = DEFAULT_HIGH_TIER
    low_bits: int = DEFAULT_LOW_BITS
    score_scheme: str = DEFAULT_SCORE_SCHEME
    score_bits: int = DEFAULT_SCORE_BITS
    score_decay_scheme: str = DEFAULT_SCORE_DECAY_SCHEME
    score_decay_application: str = DEFAULT_SCORE_DECAY_APPLICATION
    score_length_bits: int = HW_SCORE_LENGTH_BITS
    score_frac_bits: int = HW_SCORE_FRAC_BITS
    score_signed: bool = HW_SCORE_SIGNED
    age_lut_entries: int = HW_AGE_LUT_ENTRIES

    def canonical(self) -> "SweepPoint":
        """
        This point with every axis that cannot affect the run reset to its
        default, so configurations that are the same experiment compare equal.

        Three collapses, each because a knob is genuinely unreachable:

        (1) `score_bits` is read only by score_scheme="quant" (the accumulator
            width) and score_decay_scheme="quant" (the decay LUT width). Under
            any other pair of schemes it is never looked at.
        (2) `score_length_bits` / `score_frac_bits` / `score_signed` are the
            fixed-point format, read only by score_scheme="fixed".
        (3) window_ratio >= 1 puts w = k, hence k_H = k - w = 0, and
            `_importance_set` then returns the pure recency mask without ever
            consulting the scoreboard -- so the whole score path (scheme, width,
            decay, decay site) decides nothing. This one is worth having: a grid
            that sweeps w together with the scoreboard would otherwise re-run
            every scoreboard variant at w = k and get one number back each time.

        Collapsing to the *default* rather than to a sentinel keeps the recorded
        row honest: the reset value is what the run actually executed with.
        """
        fields = dict(
            budget_mode=self.budget_mode,
            balance_scheme=self.balance_scheme,
            window_tokens=None if self.window_tokens is None else int(self.window_tokens),
            window_ratio=float(self.window_ratio),
            high_tier=str(self.high_tier),
            low_bits=int(self.low_bits),
            score_scheme=self.score_scheme,
            score_bits=int(self.score_bits),
            score_decay_scheme=self.score_decay_scheme,
            score_decay_application=self.score_decay_application,
            score_length_bits=int(self.score_length_bits),
            score_frac_bits=int(self.score_frac_bits),
            score_signed=bool(self.score_signed),
            age_lut_entries=int(self.age_lut_entries),
        )
        if fields["window_tokens"] is not None:
            # `budget_for` reads window_tokens and ignores window_ratio entirely,
            # so a grid crossing the two would run each absolute window once per
            # ratio value and report identical numbers.
            fields["window_ratio"] = DEFAULT_WINDOW_RATIO
        if fields["window_ratio"] >= 1.0:
            fields["score_scheme"] = DEFAULT_SCORE_SCHEME
            fields["score_decay_scheme"] = DEFAULT_SCORE_DECAY_SCHEME
            fields["score_decay_application"] = DEFAULT_SCORE_DECAY_APPLICATION
        if fields["score_scheme"] != "quant" and fields["score_decay_scheme"] != "quant":
            fields["score_bits"] = DEFAULT_SCORE_BITS
        if fields["score_decay_scheme"] != "lut":
            # The ROM depth is read only by the LUT decay. Under any other decay
            # scheme it is never looked at, so the three depths are one experiment.
            fields["age_lut_entries"] = HW_AGE_LUT_ENTRIES
        if fields["score_scheme"] != "fixed":
            fields["score_length_bits"] = HW_SCORE_LENGTH_BITS
            fields["score_frac_bits"] = HW_SCORE_FRAC_BITS
            fields["score_signed"] = HW_SCORE_SIGNED
        return SweepPoint(**fields)

    def invalid_reason(self) -> str | None:
        """Why this point cannot be run, or None if it can. Checked up front so a
        malformed corner of a grid is dropped from the plan with a warning rather
        than raising thousands of tokens into an hours-long sweep."""
        if self.budget_mode not in BUDGET_MODES:
            return f"budget_mode={self.budget_mode!r} not in {BUDGET_MODES}"
        if self.balance_scheme not in BALANCE_SCHEMES:
            return f"balance_scheme={self.balance_scheme!r} not in {BALANCE_SCHEMES}"
        if self.score_scheme not in SCORE_SCHEMES:
            return f"score_scheme={self.score_scheme!r} not in {SCORE_SCHEMES}"
        if self.score_decay_scheme not in SCORE_DECAY_SCHEMES:
            return f"score_decay_scheme={self.score_decay_scheme!r} not in {SCORE_DECAY_SCHEMES}"
        if self.score_decay_application not in SCORE_DECAY_APPLICATIONS:
            return (
                f"score_decay_application={self.score_decay_application!r} "
                f"not in {SCORE_DECAY_APPLICATIONS}"
            )
        if not 0.0 <= self.window_ratio <= 1.0:
            return f"window_ratio={self.window_ratio} outside [0, 1]"
        if self.window_tokens is not None and self.window_tokens < 0:
            return f"window_tokens={self.window_tokens} negative"
        if self.high_tier not in HIGH_TIER_MODES:
            return f"high_tier={self.high_tier!r} not in {HIGH_TIER_MODES}"
        if not 1 <= self.low_bits <= 16:
            return f"low_bits={self.low_bits} outside [1, 16]"
        high_bits, high_native = high_precision_knobs(self.high_tier)
        if not high_native and high_bits < self.low_bits:
            # Not an error the policy would raise -- it would run, with the
            # "important" bucket stored *coarser* than the evicted one, which is
            # not a configuration anyone means to test.
            return f"high_tier={self.high_tier} below low_bits={self.low_bits}"
        if self.score_decay_scheme == "lut":
            # The fold halves an age until it lands at or below the table top,
            # n_min + D - 1, so the smallest address it can generate sits just
            # above (n_min + D - 1) / 2. For that to still be inside the table:
            #
            #     (W + D) / 2 >= W + 1   <=>   D >= W + 2
            #
            # Below that the fold undershoots the table and the address clamps,
            # silently flattening the decay for the oldest tokens. w = 128 with a
            # 128-deep ROM is exactly this case, and it is a real constraint on
            # the design rather than a modelling artifact -- a deeper table or a
            # narrower window is the fix.
            window = (
                int(self.window_tokens) if self.window_tokens is not None
                else HW_AGE_LUT_RECENT_WINDOW
            )
            if self.age_lut_entries < window + 2:
                return (
                    f"age_lut_entries={self.age_lut_entries} too shallow for a "
                    f"recency window of {window} (the fold needs D >= W + 2 = {window + 2})"
                )
        if not 1 <= self.score_bits <= 16:
            return f"score_bits={self.score_bits} outside [1, 16]"
        try:
            # The same eager check MiKVPolicy.__init__ performs: an (l, f) pair
            # with f too wide to leave any integer bits has no representable range.
            fixed_point_range(self.score_length_bits, self.score_frac_bits, self.score_signed)
        except ValueError as exc:
            return str(exc)
        return None

    def policy_kwargs(self) -> dict:
        """This point as the keyword arguments `run_line_retrieval_benchmark` (and
        through it `MiKVPolicy`) takes -- the one place the single
        `high_tier` axis is expanded back into the policy's two knobs."""
        high_bits, high_precision_native = high_precision_knobs(self.high_tier)
        return dict(
            budget_mode=self.budget_mode,
            balance_scheme=self.balance_scheme,
            window_ratio=self.window_ratio,
            window_tokens=self.window_tokens,
            high_bits=high_bits,
            high_precision_native=high_precision_native,
            low_bits=self.low_bits,
            score_scheme=self.score_scheme,
            score_bits=self.score_bits,
            score_decay_scheme=self.score_decay_scheme,
            score_decay_application=self.score_decay_application,
            score_length_bits=self.score_length_bits,
            score_frac_bits=self.score_frac_bits,
            score_signed=self.score_signed,
            age_lut_entries=self.age_lut_entries,
        )

    @property
    def score_tag(self) -> str:
        return format_score_tag(
            self.score_scheme,
            self.score_bits,
            self.score_decay_scheme,
            self.score_decay_application,
            self.score_length_bits,
            self.score_frac_bits,
            self.score_signed,
            self.age_lut_entries,
        )

    def describe(self, axes: tuple[str, ...] | None = None) -> str:
        """Compact `axis=value` line for logs. `axes` restricts it to the axes a
        sweep is actually varying, which is what makes a 12-axis point readable."""
        shown = axes or SWEEP_AXES
        parts = []
        for axis in SWEEP_AXES:
            if axis not in shown:
                continue
            if axis in ("score_bits", "score_decay_scheme", "score_decay_application",
                        "score_length_bits", "score_frac_bits", "score_signed",
                        "age_lut_entries"):
                continue  # folded into score= below
            parts.append(f"{axis}={getattr(self, axis)}")
        parts.append(f"score={self.score_tag}")
        return " ".join(parts)


def _axis_values(axes: dict[str, tuple], name: str) -> tuple:
    """The values requested for one axis, falling back to the SweepPoint default."""
    values = axes.get(name)
    if not values:
        return (getattr(SweepPoint(), name),)
    return tuple(values)


def _assignments(name: str, value) -> dict:
    """
    The SweepPoint fields one axis value sets.

    A plain value sets the axis's own field. A *dict* value sets several at once,
    which makes an axis whose settings only make sense together expressible as one
    list rather than a cross product. The scoreboard is the case that forces this:
    it is one design decision spanning four fields (scheme, word length l,
    fraction f, signedness), and its legal combinations are a short list, not a
    product -- crossing l and f independently would enumerate `l=14, f=16`, which
    has no integer bits and is dropped, alongside `l=30, f=2`, which nobody would
    build. See SCOREBOARD_SWEEP.
    """
    if isinstance(value, dict):
        return dict(value)
    return {name: value}


def _axis_field_names(name: str, values: tuple) -> set[str]:
    """Which SweepPoint fields an axis writes -- its own, or the union of the keys
    its dict values set."""
    fields = set()
    for value in values:
        fields |= set(_assignments(name, value))
    return fields


def enumerate_sweep_points(
    axes: dict[str, tuple],
    sweep_mode: str = DEFAULT_SWEEP_MODE,
    ofat_baseline: dict | None = None,
    extra_points: tuple[dict, ...] = (),
) -> tuple[list[SweepPoint], list[tuple[SweepPoint, str]]]:
    """
    The configurations a sweep will run, and the ones it dropped.

    `axes` maps a name in SWEEP_AXES to the tuple of values to try; an axis left
    out (or given an empty tuple) is held at its SweepPoint default. Returns
    (points, skipped), where `skipped` pairs each dropped point with the reason
    -- an invalid combination -- so the caller can print them rather than lose
    them silently. Points are canonicalized and deduped, so the returned list is
    the set of *distinct experiments*, which is usually shorter than the product
    of the axis lengths (see `SweepPoint.canonical`).

    "grid" walks the full cartesian product. "ofat" starts from a baseline point
    -- the first value of every axis, unless `ofat_baseline` overrides specific
    axes -- and emits it plus, for each axis, one point per remaining value with
    only that axis moved. See SWEEP_MODES for when each is the right choice.

    `extra_points` appends configurations neither walk would reach, as dicts of
    axis overrides on the baseline. It exists because some axes are only valid in
    combination: a 16-bit score word (l = 14) cannot carry f = 16, so an OFAT
    that moves l alone off an f = 16 baseline produces an invalid point and drops
    it, and the narrow word never gets measured at all. `extra_points` states the
    valid pairing -- `dict(score_length_bits=14, score_frac_bits=8)` -- directly.
    Points landing on something already enumerated are deduped like any other.
    """
    if sweep_mode not in SWEEP_MODES:
        raise ValueError(f"sweep_mode must be one of {SWEEP_MODES}, got {sweep_mode!r}")
    # An axis name outside SWEEP_AXES is legal only if every one of its values is
    # a dict naming real fields -- that is a composite axis (see `_assignments`).
    for name, values in axes.items():
        fields = _axis_field_names(name, tuple(values))
        unknown = fields - set(SWEEP_AXES)
        if unknown:
            raise ValueError(
                f"axis {name!r} sets unknown fields {sorted(unknown)}; "
                f"expected a subset of {SWEEP_AXES}"
            )
        if name not in SWEEP_AXES and any(not isinstance(v, dict) for v in values):
            raise ValueError(
                f"axis {name!r} is not one of SWEEP_AXES, so every one of its values "
                f"must be a dict of field assignments (a composite axis)"
            )

    # The baseline is defined in both modes -- the first value of every axis, with
    # `ofat_baseline` applied -- because `extra_points` is expressed relative to
    # it whichever walk is in use.
    # Axes are walked in a stable order: the declared ones in SWEEP_AXES order,
    # then any composite axis in the order the caller gave it.
    names = [n for n in SWEEP_AXES if n in axes] + [n for n in axes if n not in SWEEP_AXES]

    # The baseline: SweepPoint's own defaults, then the first value of every axis
    # applied in order, then `ofat_baseline`. Composite axes participate, so a
    # baseline can be "the scoreboard the hardware implements" in one entry.
    base_values = {name: getattr(SweepPoint(), name) for name in SWEEP_AXES}
    for name in names:
        base_values.update(_assignments(name, _axis_values(axes, name)[0]))
    base_values.update(ofat_baseline or {})

    # "greedy" is not enumerable up front -- which point it visits next depends
    # on how the previous axis came out -- so it walks the ofat set here, which is
    # exactly the set of configurations it can visit, hence an upper bound on its
    # cost. The real walk is `_greedy_sweep`.
    if sweep_mode == "grid":
        raw = []
        for combo in itertools.product(*(_axis_values(axes, n) for n in names)):
            fields = dict(base_values)
            for name, value in zip(names, combo):
                fields.update(_assignments(name, value))
            raw.append(SweepPoint(**fields))
    else:
        raw = [SweepPoint(**base_values)]
        for name in names:
            values = _axis_values(axes, name)
            if len(values) < 2:
                # A single-valued axis is not being swept -- it just contributes to
                # the baseline. Skipping it matters when `ofat_baseline` pins that
                # axis to something else: without this, the lone value contradicts
                # the pin and gets emitted as a variation nobody asked for. The pin
                # wins, which is what pinning means.
                continue
            # What this axis holds at the baseline, so the baseline's own value is
            # not re-run. For a composite axis that is the whole assignment dict.
            base_here = {k: base_values[k] for k in _axis_field_names(name, values)}
            for value in values:
                assignments = _assignments(name, value)
                if all(base_here.get(k) == v for k, v in assignments.items()):
                    continue
                raw.append(SweepPoint(**{**base_values, **assignments}))

    for overrides in extra_points:
        unknown_override = set(overrides) - set(SWEEP_AXES)
        if unknown_override:
            raise ValueError(
                f"extra_points entry names unknown axes {sorted(unknown_override)}; "
                f"expected a subset of {SWEEP_AXES}"
            )
        raw.append(SweepPoint(**{**base_values, **overrides}))

    points: list[SweepPoint] = []
    skipped: list[tuple[SweepPoint, str]] = []
    seen: set[SweepPoint] = set()
    for point in raw:
        reason = point.invalid_reason()
        if reason is not None:
            skipped.append((point, reason))
            continue
        canonical = point.canonical()
        if canonical in seen:
            continue
        seen.add(canonical)
        points.append(canonical)
    return points, skipped


def varying_axes(points: list[SweepPoint]) -> tuple[str, ...]:
    """The axes that actually take more than one value across `points` -- what a
    log line, a legend or a filename has to name to tell two runs apart, and
    nothing more."""
    return tuple(
        name for name in SWEEP_AXES if len({getattr(p, name) for p in points}) > 1
    )


def format_sweep_plan(
    points: list[SweepPoint],
    budget_ratios: tuple[float, ...],
    skipped: list[tuple[SweepPoint, str]] | None = None,
    num_samples: int = 0,
) -> str:
    """
    The run plan as text: every configuration, the ratios each is swept over, and
    the resulting benchmark-run and generation counts. Printed before a sweep
    starts and by `--dry-run`, which exists precisely so the size of a sweep can
    be checked *before* committing the GPU hours it costs.
    """
    varying = varying_axes(points)
    total_runs = len(points) * len(budget_ratios)
    lines = [
        f"[plan] {len(points)} configurations x {len(budget_ratios)} budget ratios "
        f"= {total_runs} benchmark runs"
        + (f" x {num_samples} samples = {total_runs * num_samples} generations" if num_samples else ""),
        f"[plan] varying axes: {', '.join(varying) if varying else '(none -- a single configuration)'}",
        f"[plan] budget ratios: {list(budget_ratios)}",
    ]
    for i, point in enumerate(points, 1):
        lines.append(f"[plan]   {i:>3}. {point.describe(varying or None)}")
    for point, reason in skipped or []:
        lines.append(f"[plan]   SKIPPED ({reason}): {point.describe()}")
    return "\n".join(lines)


# Axes that change the KV footprint. The rest -- the balancer, the scoreboard,
# the decay -- change only *which* tokens are kept and how faithfully they are
# ranked, never how many, so their values are comparable at equal cost. The
# greedy walk uses this split to pick an objective per axis, and the reports use
# it to label which comparisons are like-for-like.
FOOTPRINT_AXES = ("budget_mode", "window_tokens", "window_ratio", "high_tier", "low_bits")

# The order sweep_mode="greedy" decides axes in, and the order matters: greedy
# fixes each axis against the winners of the ones before it, so an axis decided
# early is decided on less information but constrains everything after it.
#
# Footprint-FREE axes go first, and that is the whole design of this order. They
# can be judged on accuracy alone -- nothing else moves -- so their decisions are
# unambiguous, cheap to trust, and pure wins that carry into every later axis.
# The footprint axes follow, because "best" there is a trade rather than a
# maximum: more cache is always at least as accurate, so maximizing accuracy
# would simply walk to the largest configuration. They are scored on accuracy per
# byte instead (see `_greedy_objective_value`), and the honest reading of them is
# still the Pareto front over all the rows, not the single value greedy picked.
GREEDY_AXIS_ORDER = (
    "balance_scheme",
    "scoreboard",
    "score_scheme",
    "score_decay_scheme",
    "age_lut_entries",
    "score_decay_application",
    "score_bits",
    "score_length_bits",
    "score_frac_bits",
    "score_signed",
    "window_tokens",
    "window_ratio",
    "low_bits",
    "high_tier",
    "budget_mode",
)
GREEDY_OBJECTIVES = ("auto", "accuracy", "accuracy_per_byte")


def _greedy_objective_value(rows: list[dict], objective: str) -> float:
    """
    The scalar greedy maximizes for one configuration, over its rows (one per
    budget ratio).

    "accuracy" -- mean accuracy across the ratios. Correct for a footprint-free
        axis, where every candidate sits at the same KV size and accuracy is the
        only thing that moved.
    "accuracy_per_byte" -- mean of accuracy / (KV size as a fraction of
        uncompressed). For a footprint axis, where maximizing accuracy alone
        would always choose the largest configuration on offer, this at least
        prices the cache it spends. It is a crude scalarization of a genuine
        two-objective problem and it is not a substitute for reading the Pareto
        front -- it exists so the walk can proceed, not so the answer can be
        taken on faith.
    """
    if not rows:
        return float("-inf")
    if objective == "accuracy":
        return sum(r["accuracy"] for r in rows) / len(rows)
    return sum(r["accuracy"] / max(r["compression_pct"] / 100, 1e-9) for r in rows) / len(rows)


def _greedy_sweep(
    axes: dict[str, tuple],
    budget_ratios: tuple[float, ...],
    evaluate,
    ofat_baseline: dict | None = None,
    axis_order: tuple[str, ...] | None = None,
    objective: str = "auto",
) -> SweepPoint:
    """
    Coordinate descent over the axes: sweep one axis, keep its winner, move to
    the next with that winner fixed. Returns the configuration it settled on.

    It visits the same number of configurations as "ofat" -- 1 + sum(len - 1) --
    but each axis is decided against the winners of the axes before it rather
    than against a fixed baseline, so it recovers *some* of what a grid would
    show about interactions at no extra cost. What it cannot do is see an
    interaction that only pays off in a direction it has already walked away
    from; it finds a local optimum along the path it took, and a different axis
    order can land somewhere else. Treat the result as a strong candidate to
    confirm with a small grid over the two or three axes that moved accuracy
    most, not as the optimum.

    `objective="auto"` scores footprint-free axes on accuracy and footprint axes
    on accuracy per byte -- see `_greedy_objective_value` and FOOTPRINT_AXES.
    """
    if objective not in GREEDY_OBJECTIVES:
        raise ValueError(f"greedy_objective must be one of {GREEDY_OBJECTIVES}, got {objective!r}")

    ordered = axis_order or GREEDY_AXIS_ORDER
    # Axes the caller actually gave values for, in the decision order, then any
    # the order does not mention (so a new axis is swept rather than ignored).
    names = [n for n in ordered if n in axes] + [n for n in axes if n not in ordered]

    base_values = {name: getattr(SweepPoint(), name) for name in SWEEP_AXES}
    for name in names:
        base_values.update(_assignments(name, _axis_values(axes, name)[0]))
    base_values.update(ofat_baseline or {})
    current = SweepPoint(**base_values).canonical()

    measured: dict[SweepPoint, list[dict]] = {}

    def measure(point: SweepPoint) -> list[dict]:
        # Memoized on the canonical point, so an axis whose winner a later axis
        # happens to revisit is not benchmarked twice -- and, more importantly,
        # so the incumbent is never re-run at each of the dozen decisions.
        if point not in measured:
            measured[point] = [evaluate(point, ratio) for ratio in budget_ratios]
        return measured[point]

    print(f"[greedy] baseline: {current.describe()}", flush=True)
    measure(current)

    for name in names:
        values = _axis_values(axes, name)
        if len(values) < 2:
            continue
        fields = _axis_field_names(name, values)
        axis_objective = objective
        if objective == "auto":
            axis_objective = (
                "accuracy_per_byte" if fields & set(FOOTPRINT_AXES) else "accuracy"
            )
        best, best_score = current, _greedy_objective_value(measured[current], axis_objective)
        print(
            f"[greedy] --- axis {name!r}: {len(values)} values, "
            f"objective={axis_objective}, incumbent score={best_score:.4f} ---",
            flush=True,
        )
        for value in values:
            candidate = replace(current, **_assignments(name, value)).canonical()
            reason = candidate.invalid_reason()
            if reason is not None:
                print(f"[greedy]   skipping {name}={value}: {reason}", flush=True)
                continue
            if candidate == current:
                continue
            score = _greedy_objective_value(measure(candidate), axis_objective)
            print(f"[greedy]   {name}={value} -> score {score:.4f}", flush=True)
            if score > best_score:
                best, best_score = candidate, score
        if best != current:
            changed = {f: getattr(best, f) for f in fields if getattr(best, f) != getattr(current, f)}
            print(f"[greedy] axis {name!r} -> {changed} (score {best_score:.4f})", flush=True)
        else:
            print(f"[greedy] axis {name!r} -> keeping the incumbent", flush=True)
        current = best

    print(f"[greedy] settled on: {current.describe()}", flush=True)
    return current


def sweep_kv_compression(
    model,
    tokenizer,
    budget_ratios: tuple[float, ...] = (0.25, 0.5, 0.75),
    num_samples: int = 20,
    num_records: int = 20,
    max_tokens: int = 4096,
    seed: int = 0,
    # H2O-style hard eviction for the whole sweep, as an alternative to
    # demoting the LOW set to `low_bits_options`. Deliberately NOT one of the
    # crossed axes below: a single job either models eviction or it doesn't,
    # and `low_bits` is meaningless once positions are masked out of attention
    # entirely rather than kept at reduced precision, so crossing the two would
    # just re-run the same eviction experiment once per low_bits value. See
    # `MiKVPolicy.evict` / `apply_eviction_mask`.
    evict: bool = False,
    # --- the configuration axes. Every one is a tuple; a single-element tuple
    # holds that axis fixed, which is what the defaults below do for everything
    # the original sweep did not vary, so the default call still runs exactly the
    # 2 modes x 2 balancers x len(budget_ratios) sweep it always did. ---
    budget_modes: tuple[str, ...] = BUDGET_MODES,
    balance_schemes: tuple[str, ...] = BALANCE_SCHEMES,
    # w in absolute tokens (WINDOW_SWEEP); None falls back to `window_ratios`.
    window_tokens_options: tuple[int | None, ...] = (DEFAULT_WINDOW_TOKENS,),
    window_ratios: tuple[float, ...] = (DEFAULT_WINDOW_RATIO,),
    # How the important bucket is stored -- see HIGH_TIER_MODES.
    high_tiers: tuple[str, ...] = (DEFAULT_HIGH_TIER,),
    low_bits_options: tuple[int, ...] = (DEFAULT_LOW_BITS,),
    # Defaults to the single native-fp16 scoreboard rather than all of
    # SCORE_SCHEMES: under "grid" every extra axis multiplies the run count, and
    # this one is hours of GPU time per value. Pass SCORE_SCHEMES (or switch to
    # sweep_mode="ofat") to sweep it.
    score_schemes: tuple[str, ...] = (DEFAULT_SCORE_SCHEME,),
    score_bits_options: tuple[int, ...] = (DEFAULT_SCORE_BITS,),
    score_decay_schemes: tuple[str, ...] = (DEFAULT_SCORE_DECAY_SCHEME,),
    # Not a precision knob: it changes the policy itself (see
    # SCORE_DECAY_APPLICATIONS), so points from the two settings answer different
    # questions rather than bracketing one.
    score_decay_applications: tuple[str, ...] = (DEFAULT_SCORE_DECAY_APPLICATION,),
    # The fixed-point score word (l, f, signedness) for score_scheme="fixed";
    # ignored -- and collapsed to one point by `SweepPoint.canonical` -- under
    # every other scheme, so crossing them costs nothing when "fixed" is not in
    # `score_schemes`.
    score_length_bits_options: tuple[int, ...] = (HW_SCORE_LENGTH_BITS,),
    score_frac_bits_options: tuple[int, ...] = (HW_SCORE_FRAC_BITS,),
    score_signed_options: tuple[bool, ...] = (HW_SCORE_SIGNED,),
    # ipu_age_lut ROM depth, read only by score_decay_scheme="lut" -- and
    # collapsed to one point by `SweepPoint.canonical` under every other decay,
    # so crossing it costs nothing when the LUT is not in play.
    age_lut_entries_options: tuple[int, ...] = (HW_AGE_LUT_ENTRIES,),
    # The scoreboard as ONE composite axis: a tuple of dicts, each setting the
    # score fields that only make sense together (see SCOREBOARD_SWEEP). When
    # given it supersedes the four `score_*` tuples above, which exist for
    # sweeping one score field in isolation.
    scoreboard_configs: tuple[dict, ...] = (),
    sweep_mode: str = DEFAULT_SWEEP_MODE,
    ofat_baseline: dict | None = None,
    # Configurations to run on top of whatever the walk enumerates, as axis
    # overrides on the baseline -- for combinations only valid together, which
    # a one-axis-at-a-time walk cannot reach. See `enumerate_sweep_points`.
    extra_points: tuple[dict, ...] = (),
    # Append each row to this CSV as soon as it completes, on top of whatever the
    # caller does with the returned list. A full-axis sweep is many hours; without
    # this, a crash or an OOM at run 40 of 50 loses all of it.
    checkpoint_csv: str | None = None,
    # --- sweep_mode="greedy" only ---
    # The order axes are decided in. Defaults to GREEDY_AXIS_ORDER, which puts the
    # footprint-free axes first on purpose -- see `_greedy_sweep`.
    greedy_axis_order: tuple[str, ...] | None = None,
    greedy_objective: str = "auto",
) -> list[dict]:
    """
    Run the Line Retrieval benchmark under MiKV at every point of a
    multi-axis configuration sweep, pairing each point's accuracy with the
    KV cache footprint it implies.

    Every knob of the policy is a sweepable axis here:

    | axis                        | what it changes                              |
    |-----------------------------|----------------------------------------------|
    | `budget_modes`              | k frozen at r*t_p vs. k tracking r*t          |
    | `budget_ratios`             | r, the steady-state importance-set ratio      |
    | `window_ratios`             | w = window_ratio * k, the recency window      |
    | `high_precisions`           | HIGH bucket: 16 = native fp16, else N bits    |
    | `low_bits_options`          | LOW (evicted) bucket width N                  |
    | `balance_schemes`           | channel balancer: paper sqrt vs. pow-2        |
    | `score_schemes`             | scoreboard accumulator precision              |
    | `score_bits_options`        | its width under "quant"                       |
    | `score_length_bits_options` | fixed-point word length l under "fixed"       |
    | `score_frac_bits_options`   | fixed-point fraction length f under "fixed"   |
    | `score_signed_options`      | signedness of that word                       |
    | `score_decay_schemes`       | precision of the 1/(t - i) age decay          |
    | `age_lut_entries_options`   | ipu_age_lut ROM depth, under decay="lut"      |
    | `score_decay_applications`  | where the decay is applied (ranking policy)   |

    `sweep_mode` decides how they are walked -- the full cartesian product
    ("grid") or one factor at a time from a baseline ("ofat"); see
    SWEEP_MODES, and `format_sweep_plan`, which prints the resulting run
    count before any of it is spent. Points that are the same experiment
    are collapsed to one run (`SweepPoint.canonical`) and invalid
    combinations are dropped from the plan with a reason, so an over-broad
    grid degrades into a shorter sweep rather than into wasted or crashed
    runs.

    Only `budget_ratios` is swept *within* a configuration: it is the x
    axis of the accuracy-vs-compression curve every other axis is compared
    on. Note that the axes divide sharply in what they move:

    - `budget_modes`, `budget_ratios`, `window_ratios` (through k, and w
      only in that it caps how much of k the recency window takes),
      `high_precisions` and `low_bits_options` change the **footprint**, so
      two points differing on them sit at different x.
    - `balance_schemes`, the score axes and `score_decay_*` change only
      *which* tokens are kept and how faithfully they are ranked -- never
      how many -- so they move accuracy at an **identical footprint**. That
      is what makes them the interesting ones for a hardware trade-off:
      a cheaper balancer or a narrower scoreboard is free in area terms if
      the accuracy holds.

    The mode decides what r multiplies: "fixed_length" freezes
    k = floor(r * t_p) at prefill (t_p from the no-quant baseline's prompt
    length), "fixed_ratio" re-derives k = floor(r * t) every step from the
    current cache length -- so at the same r the two modes land at
    *different* footprints, and each row is sized against its own k (see
    `resolve_budget_k`). For every point, pairs the resulting accuracy with
    the KV cache footprint at a **fixed** context length of `max_tokens` --
    the context the hardware target provisions, so that full allocation is
    what compression acts on. Both the uncompressed baseline
    (`kv_size_before`) and the MiKV estimate (`kv_size_after`) are computed
    at that same `seq_len`, so the ratio between them is meaningful.

    The length generation actually reached is reported separately
    (`avg_seq_len` / `kv_bytes_occupied`) as a diagnostic: it is usually
    well short of `max_tokens` because decoding stops at EOS, so it
    measures occupancy rather than the provisioned allocation. Do not mix
    the two bases in one ratio.

    Runs the no-quant baseline itself, first, before installing any MiKV
    patch -- see `run_line_retrieval_no_quant`.

    Returns a flat list of dicts, one per (configuration, ratio), each
    carrying the full configuration that produced it plus accuracy, k,
    kv_size_before/after and compression_pct -- ready to hand to
    `format_results_table`, `write_results_csv`,
    `plot_accuracy_vs_compression`, `plot_axis_effects` and `plot_pareto`.
    """
    axes = axes_from_sweep_kwargs(
        dict(
            budget_modes=budget_modes,
            balance_schemes=balance_schemes,
            window_tokens_options=window_tokens_options,
            window_ratios=window_ratios,
            high_tiers=high_tiers,
            low_bits_options=low_bits_options,
            score_schemes=score_schemes,
            score_bits_options=score_bits_options,
            score_decay_schemes=score_decay_schemes,
            score_decay_applications=score_decay_applications,
            score_length_bits_options=score_length_bits_options,
            score_frac_bits_options=score_frac_bits_options,
            score_signed_options=score_signed_options,
            age_lut_entries_options=age_lut_entries_options,
        )
    )
    if scoreboard_configs:
        # The composite axis replaces the four per-field score axes rather than
        # joining them: crossing "the scoreboard is one of these 8" with "and
        # also try these l values" enumerates combinations the list deliberately
        # excludes.
        for field in ("score_scheme", "score_bits", "score_length_bits",
                      "score_frac_bits", "score_signed"):
            axes.pop(field, None)
        axes["scoreboard"] = tuple(scoreboard_configs)
    budget_ratios = tuple(budget_ratios)
    if sweep_mode == "greedy":
        # Greedy cannot be enumerated up front -- which point it runs next depends
        # on how the previous axis came out -- so the plan is a bound, not a list.
        points, skipped = enumerate_sweep_points(
            axes, sweep_mode="ofat", ofat_baseline=ofat_baseline, extra_points=tuple(extra_points)
        )
        varying = varying_axes(points)
        print(f"[sweep] === sweep_mode=greedy ===", flush=True)
        print(
            f"[plan] greedy visits at most {len(points)} configurations x "
            f"{len(budget_ratios)} ratios = {len(points) * len(budget_ratios)} benchmark runs"
            + (f" x {num_samples} samples" if num_samples else "")
            + " -- the same count as ofat, but each axis is decided against the "
            "winners of the axes before it rather than against a fixed baseline",
            flush=True,
        )
    else:
        points, skipped = enumerate_sweep_points(
            axes, sweep_mode=sweep_mode, ofat_baseline=ofat_baseline, extra_points=tuple(extra_points)
        )
        varying = varying_axes(points)
        print(f"[sweep] === sweep_mode={sweep_mode} ===", flush=True)
        print(format_sweep_plan(points, budget_ratios, skipped, num_samples=num_samples), flush=True)
    if not points:
        raise ValueError("sweep produced no runnable configurations -- see the skipped list above")

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

    total_runs = len(points) * len(budget_ratios)
    print(
        f"[sweep] === stage 2/2: {'up to ' if sweep_mode == 'greedy' else ''}{total_runs} MiKV runs "
        f"({len(points)} configurations x {len(budget_ratios)} ratios), "
        f"t_p={t_p}, fixed context seq_len={seq_len} ===",
        flush=True,
    )
    results = []
    run_id = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    progress = {"run": 0}

    def evaluate(point: SweepPoint, ratio: float) -> dict:
        """Benchmark one (configuration, ratio) and return its result row, also
        appending it to `results` and to the checkpoint CSV. Factored out because
        the greedy walk has to call it point by point as it decides, while the
        static walks just iterate the enumerated product."""
        progress["run"] += 1
        run = progress["run"]
        k = resolve_budget_k(point.budget_mode, ratio, t_p, seq_len)
        print(
            f"[sweep] ({run}/{total_runs}) {point.describe(varying or None)} ratio={ratio} -> k={k}",
            flush=True,
        )
        kwargs = point.policy_kwargs()
        accuracy = run_line_retrieval_benchmark(
            model,
            tokenizer,
            num_samples=num_samples,
            num_records=num_records,
            budget_ratio=ratio,
            max_tokens=max_tokens,
            seed=seed,
            evict=evict,
            **kwargs,
        )
        # The footprint depends only on (seq_len, k, bit widths), so every
        # configuration sharing those sits at the same x -- which is the point:
        # the balancer and scoreboard axes are compared on accuracy at equal
        # compression, not on compression.
        # Under eviction, the LOW set costs nothing (it isn't stored at all,
        # not even at low_bits) -- pass low_bits=0 for this call only so the
        # footprint reflects that, rather than reusing kwargs["low_bits"],
        # which under evict=True is a nominal value the policy never reads.
        kv_size_after = kv_cache_size_bytes(
            model,
            seq_len,
            k=k,
            high_bits=kwargs["high_bits"],
            low_bits=0 if evict else kwargs["low_bits"],
            high_precision_native=kwargs["high_precision_native"],
        )
        compression_pct = 100 * kv_size_after / kv_size_before
        row = dict(
            budget_mode=point.budget_mode,
            scheme=point.balance_scheme,
            score_scheme=point.score_scheme,
            score_bits=point.score_bits if point.score_scheme == "quant" else "",
            score_decay_scheme=point.score_decay_scheme,
            score_decay_application=point.score_decay_application,
            score_length_bits=point.score_length_bits if point.score_scheme == "fixed" else "",
            score_frac_bits=point.score_frac_bits if point.score_scheme == "fixed" else "",
            score_signed=point.score_signed if point.score_scheme == "fixed" else "",
            # Recorded, not parameterized: these define what "fixed" / "lut" mean,
            # so a row stays interpretable if the constants are ever retuned.
            hw_delta_bits=HW_DELTA_BITS if point.score_scheme == "fixed" else "",
            hw_age_lut_entries=point.age_lut_entries if point.score_decay_scheme == "lut" else "",
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
            window_tokens="" if point.window_tokens is None else point.window_tokens,
            window_ratio=point.window_ratio,
            high_bits=kwargs["high_bits"],
            low_bits=0 if evict else point.low_bits,
            evict=evict,
            high_precision_native=kwargs["high_precision_native"],
            sweep_mode=sweep_mode,
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
        results.append(row)
        if checkpoint_csv:
            # Same run_id on every row of this sweep, so the whole sweep can be
            # pulled back out of an accumulated file even if it died halfway.
            write_results_csv([row], save_path=checkpoint_csv, run_id=run_id)
        print(
            f"[sweep] {point.describe(varying or None)} ratio={ratio} k={k} "
            f"accuracy={accuracy * 100:.1f}% KV compression={compression_pct:.1f}% "
            f"(baseline accuracy={baseline_accuracy * 100:.1f}%)",
            flush=True,
        )
        return row

    if sweep_mode == "greedy":
        _greedy_sweep(
            axes,
            budget_ratios,
            evaluate,
            ofat_baseline=ofat_baseline,
            axis_order=greedy_axis_order,
            objective=greedy_objective,
        )
    else:
        for point, ratio in itertools.product(points, budget_ratios):
            evaluate(point, ratio)
    print("[sweep] done", flush=True)
    return results
