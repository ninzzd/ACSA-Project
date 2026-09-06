"""
MiKV: channel-balanced, budget-constrained mixed-precision KV cache for
autoregressive inference on Llama-2-7b-chat-hf -- and the sweep machinery for
choosing a hardware configuration of it by measurement rather than argument.

This file is the command line and nothing else. The system is split so the policy
can be read without the sweep and vice versa:

    mikv_runtime.py  import-order bootstrap. Two orderings here are load-bearing
                     (HF_HUB_DISABLE_XET before transformers; torch/transformers
                     before matplotlib) and both segfault silently if violated.
    mikv_config.py   every constant the policy is parameterized by, plus the
                     helpers translating a sweep's names into the policy's knobs.
    mikv_quant.py    the quantizers: affine (K/V, scoreboard) and fixed-point
                     with a static scale (the IPU's real score datapath).
    mikv_policy.py   MiKVPolicy, the attention monkey-patch, the generation loop.
    mikv_bench.py    Line Retrieval, the no-quant baseline, KV footprint bytes.
    mikv_sweep.py    SweepPoint, axis enumeration, the three walks, the driver.
    mikv_grids.py    THE KNOBS YOU EDIT: coarse value lists and named presets.
    mikv_report.py   grouping, markdown tables, the results CSV.
    mikv_plots.py    figures (optional; not imported unless --plots).

--- the policy ---

Prefill derives a per-(layer, kv head, channel) balancer b from the prompt's Q/K
statistics, accumulates an importance score per prompt position, splits the
budget k into a recency window of size w and a top-scoring set of size k - w, and
writes each position's K/V at HIGH or LOW precision. Decode attends over the
mixed-precision cache with the frozen balancer, re-derives S each step, and
demotes whatever falls out -- sticky, never restored. See mikv_policy.

The budget k comes from a ratio r under one of two modes (BUDGET_MODES):
"fixed_length" resolves k = r * t_p once from the prompt length and freezes it,
so the high-precision set is a constant *number* of tokens; "fixed_ratio"
re-derives k = r * t every step from the current total token count, so the
constant is instead the *fraction* of the cache held at high precision. r never
changes in either mode. At equal r the two are not equal-footprint
configurations, so each result row is sized against its own k.

--- the sweep ---

Every knob is an axis: budget mode, r, the recency window w (in tokens), the HIGH
and LOW tier widths, the channel balancer, and the whole scoreboard datapath. The
axes split into two kinds and the split is the point -- FOOTPRINT_AXES trade
accuracy against KV size, while the balancer, scoreboard and decay change only
*which* tokens are kept, never how many, so they move accuracy at an identical
footprint. A cheaper value on one of those is free in area terms exactly when its
accuracy holds.

SWEEP_MODES picks how the space is walked: "greedy" (coordinate descent, the
default and the recommended pass), "ofat" (controlled marginals off a fixed
baseline), or "grid" (the full product -- 6.3 days over the coarse grids, so it
is a targeted follow-up, not a first pass). Run with --dry-run to cost a sweep
without loading the model, and --help for the axis flags and the presets.

Usage:
    python scripts/mikv.py --dry-run
    python scripts/mikv.py                      # --preset greedy, ~48 runs
    python scripts/mikv.py --preset confirm --low-bits 2,4
"""

import argparse
import datetime
import os
import sys

from mikv_config import (
    BALANCE_SCHEMES,
    BUDGET_MODES,
    HIGH_TIER_MODES,
    MODEL_NAME,
    SCORE_DECAY_APPLICATIONS,
    SCORE_DECAY_SCHEMES,
    SCORE_SCHEMES,
)
from mikv_grids import AGE_LUT_SWEEP, DEFAULT_PRESET, SWEEP_PRESETS, WINDOW_SWEEP
from mikv_policy import load_model
from mikv_sweep import (
    DEFAULT_SWEEP_MODE,
    GREEDY_OBJECTIVES,
    SWEEP_MODES,
    axes_from_sweep_kwargs,
    enumerate_sweep_points,
    format_sweep_plan,
    sweep_kv_compression,
)
from mikv_report import format_results_table, write_results_csv


def _parse_list(text: str, cast):
    """A comma-separated CLI value as a tuple, e.g. `--budget-ratios 0.25,0.5`."""
    return tuple(cast(part.strip()) for part in text.split(",") if part.strip())


def _parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in ("1", "true", "t", "yes", "y"):
        return True
    if lowered in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mikv.py",
        description=(
            "Sweep the MiKV KV-cache policy's configuration space against the Line "
            "Retrieval benchmark. Every axis takes a comma-separated list and "
            "overrides the chosen preset for that axis alone. --dry-run prints the "
            "resulting run plan without loading the model, which is how to check "
            "the size of a sweep before spending the GPU hours it costs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(SWEEP_PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--sweep-mode", choices=SWEEP_MODES, default=None,
                        help="grid (cartesian product), ofat (one factor at a time from a "
                             "fixed baseline), or greedy (coordinate descent: keep each "
                             "axis's winner and move on)")
    parser.add_argument("--greedy-objective", choices=GREEDY_OBJECTIVES, default=None,
                        help="what greedy maximizes; 'auto' scores footprint-free axes on "
                             "accuracy and footprint axes on accuracy per byte")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run plan and exit without loading the model")

    axes = parser.add_argument_group("configuration axes (comma-separated)")
    axes.add_argument("--budget-ratios", type=lambda s: _parse_list(s, float), default=None,
                      help="r, the importance-set ratio -- the x axis of every curve")
    axes.add_argument("--budget-modes", type=lambda s: _parse_list(s, str), default=None,
                      help=f"one or more of {BUDGET_MODES}")
    axes.add_argument("--balancers", type=lambda s: _parse_list(s, str), default=None,
                      help=f"channel balancer, one or more of {BALANCE_SCHEMES}")
    axes.add_argument("--window-tokens", type=lambda s: _parse_list(s, int), default=None,
                      help=f"w in absolute tokens, clamped to k (sweep grid: {WINDOW_SWEEP})")
    axes.add_argument("--window-ratios", type=lambda s: _parse_list(s, float), default=None,
                      help="w as a fraction of k -- the legacy alternative, used only "
                           "when --window-tokens is not given")
    axes.add_argument("--high-tiers", type=lambda s: _parse_list(s, str), default=None,
                      help=f"HIGH bucket storage, one or more of {HIGH_TIER_MODES}")
    axes.add_argument("--low-bits", type=lambda s: _parse_list(s, int), default=None,
                      help="LOW (evicted) bucket width N")
    axes.add_argument("--score-schemes", type=lambda s: _parse_list(s, str), default=None,
                      help=f"scoreboard accumulator, one or more of {SCORE_SCHEMES}")
    axes.add_argument("--score-bits", type=lambda s: _parse_list(s, int), default=None,
                      help='scoreboard width under score_scheme="quant"')
    axes.add_argument("--score-decay-schemes", type=lambda s: _parse_list(s, str), default=None,
                      help=f"age-decay precision, one or more of {SCORE_DECAY_SCHEMES}")
    axes.add_argument("--score-decay-applications", type=lambda s: _parse_list(s, str), default=None,
                      help=f"where the decay is applied, one or more of {SCORE_DECAY_APPLICATIONS}")
    axes.add_argument("--score-length-bits", type=lambda s: _parse_list(s, int), default=None,
                      help='fixed-point word length l under score_scheme="fixed"')
    axes.add_argument("--score-frac-bits", type=lambda s: _parse_list(s, int), default=None,
                      help='fixed-point fraction length f under score_scheme="fixed"')
    axes.add_argument("--score-signed", type=lambda s: _parse_list(s, _parse_bool), default=None,
                      help="signedness of the fixed-point score word")
    axes.add_argument("--age-lut-entries", type=lambda s: _parse_list(s, int), default=None,
                      help=f"ipu_age_lut ROM depth, under decay=lut (sweep grid: {AGE_LUT_SWEEP})")

    run = parser.add_argument_group("run configuration")
    # num_samples=40: at n=20, one flipped sample swings accuracy by 5 points,
    # drowning out real signal. 40 roughly halves that per-sample noise.
    run.add_argument("--num-samples", type=int, default=40)
    # num_records: eager attention materializes the full [t_p, t_p] attention-weight
    # matrix per layer (needed for scoring), so its memory is O(t_p^2). These values
    # were sized against Qwen2.5-0.5B-Instruct (24 layers, 14 heads, 2 kv heads,
    # ~1GB weights) on a 4GB GPU -- Llama-2-7b-chat-hf (32 layers, 32 heads, no GQA,
    # ~13GB weights) has a much larger footprint per token, so these were NOT
    # re-verified against it; check available VRAM before scaling num_records up.
    run.add_argument("--num-records", type=int, default=40)
    run.add_argument("--max-tokens", type=int, default=4096,
                     help="total context (prompt + generation), and the size both "
                          "sides of the compression ratio are computed at")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--model", default=MODEL_NAME)

    out = parser.add_argument_group("output")
    out.add_argument("--csv", default="docs/kv_compression_sweep.csv")
    # Plotting is OFF by default. A sweep is hours of GPU time whose product is
    # the CSV and the tables; the figures are a rendering of those and can be
    # regenerated from the CSV at any time, so they are not worth failing a
    # finished sweep over (a matplotlib error after the last run would otherwise
    # take the run down inside the try block).
    out.add_argument("--plots", action="store_true",
                     help="also render the figures (off by default; the CSV and tables "
                          "are the sweep's product and the plots regenerate from them)")
    out.add_argument("--plot", default="docs/kv_compression_sweep.png",
                     help="base path for the figures, used only with --plots")
    out.add_argument("--no-checkpoint", action="store_true",
                     help="do not append rows to the CSV as they complete (a long "
                          "sweep that dies then loses everything before the crash)")
    return parser


# CLI flag -> (sweep_kv_compression keyword, SWEEP_AXES name). Kept as data so a
# new axis is one row here rather than another branch in the merge below, and
# spelled out rather than derived from the flag name: the three name spaces are
# only loosely related ("--low-bits" -> `low_bits_options` -> `low_bits`), and
# guessing between them by stripping a plural silently mismaps the `_options`
# ones -- which would leave an OFAT baseline pinned on an axis the user just
# asked to sweep. `budget_ratios` has no axis entry: r is crossed with every
# configuration rather than being one of the axes.
_AXIS_FLAGS = {
    "budget_ratios": ("budget_ratios", None),
    "budget_modes": ("budget_modes", "budget_mode"),
    "balancers": ("balance_schemes", "balance_scheme"),
    "window_tokens": ("window_tokens_options", "window_tokens"),
    "window_ratios": ("window_ratios", "window_ratio"),
    "high_tiers": ("high_tiers", "high_tier"),
    "low_bits": ("low_bits_options", "low_bits"),
    "score_schemes": ("score_schemes", "score_scheme"),
    "score_bits": ("score_bits_options", "score_bits"),
    "score_decay_schemes": ("score_decay_schemes", "score_decay_scheme"),
    "score_decay_applications": ("score_decay_applications", "score_decay_application"),
    "score_length_bits": ("score_length_bits_options", "score_length_bits"),
    "score_frac_bits": ("score_frac_bits_options", "score_frac_bits"),
    "score_signed": ("score_signed_options", "score_signed"),
    "age_lut_entries": ("age_lut_entries_options", "age_lut_entries"),
}


def sweep_kwargs_from_args(args) -> dict:
    """The preset's arguments with any explicitly-given CLI axis overriding it."""
    kwargs = dict(SWEEP_PRESETS[args.preset])
    if args.sweep_mode is not None:
        kwargs["sweep_mode"] = args.sweep_mode
    if args.greedy_objective is not None:
        kwargs["greedy_objective"] = args.greedy_objective
    for flag, (kwarg, axis) in _AXIS_FLAGS.items():
        value = getattr(args, flag)
        if not value:
            continue
        kwargs[kwarg] = value
        # An overridden axis invalidates the preset's baseline pin for it:
        # keeping the pin would silently ignore the values just asked for, since
        # OFAT varies *away* from the baseline and a pinned axis would contribute
        # only the pinned value.
        if axis and axis in kwargs.get("ofat_baseline", {}):
            kwargs["ofat_baseline"] = {
                k: v for k, v in kwargs["ofat_baseline"].items() if k != axis
            }
    kwargs.update(
        num_samples=args.num_samples,
        num_records=args.num_records,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    if not args.no_checkpoint:
        kwargs["checkpoint_csv"] = args.csv
    return kwargs


def _dry_run_plan(kwargs: dict) -> str:
    """The run plan for `kwargs`, built without touching the GPU -- the point of
    --dry-run is to size a sweep before paying for it."""
    points, skipped = enumerate_sweep_points(
        axes_from_sweep_kwargs(kwargs),
        sweep_mode=kwargs.get("sweep_mode", DEFAULT_SWEEP_MODE),
        ofat_baseline=kwargs.get("ofat_baseline"),
        extra_points=tuple(kwargs.get("extra_points", ())),
    )
    plan = format_sweep_plan(
        points, tuple(kwargs.get("budget_ratios", ())), skipped,
        num_samples=kwargs.get("num_samples", 0),
    )
    if kwargs.get("sweep_mode") == "greedy":
        plan = (
            "[plan] greedy: the configurations below are what the walk MAY visit -- "
            "an upper bound. It decides each axis against the winners of the "
            "previous ones, so which of these it actually reaches is not known "
            "until it runs.\n" + plan
        )
    return plan


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
    args = _build_arg_parser().parse_args()
    sweep_kwargs = sweep_kwargs_from_args(args)

    if args.dry_run:
        # Deliberately before any model load: this path must stay runnable on a
        # machine that cannot hold the weights at all.
        print(f"=== MiKV: dry run, preset={args.preset} (no model loaded) ===", flush=True)
        print(_dry_run_plan(sweep_kwargs), flush=True)
        sys.exit(0)

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
        model, tokenizer = load_model(args.model)

        print(f"=== MiKV: starting KV-compression sweep (preset={args.preset}) ===", flush=True)
        results = sweep_kv_compression(model, tokenizer, **sweep_kwargs)

        print("=== MiKV: results ===", flush=True)
        print(format_results_table(results), flush=True)
        # Written again at the end even when checkpointing was on: the checkpoint
        # rows carry a per-sweep run_id, so re-appending would duplicate them.
        if args.no_checkpoint:
            write_results_csv(results, save_path=args.csv)

        if args.plots:
            print("=== MiKV: plotting results ===", flush=True)
            # Imported here, not at module scope: plotting is off by default and
            # mikv_plots pulls in matplotlib, which a sweep run should not pay for.
            from mikv_plots import (
                plot_accuracy_vs_compression,
                plot_axis_effects,
                plot_pareto,
            )

            plot_accuracy_vs_compression(results, save_path=args.plot)
            stem, ext = os.path.splitext(args.plot)
            plot_axis_effects(results, save_path=f"{stem}_axes{ext}")
            plot_pareto(results, save_path=f"{stem}_pareto{ext}")
        else:
            print(
                "=== MiKV: plotting disabled (pass --plots to render figures; "
                "they can also be regenerated from the CSV later) ===",
                flush=True,
            )

        print(f"=== MiKV: done (log saved to {log_path}) ===", flush=True)
    finally:
        # restore the real streams *before* closing log_file -- otherwise sys.stdout/
        # stderr are left pointing at a Tee wrapping a closed file, which breaks
        # whatever Python itself tries to print during interpreter shutdown.
        sys.stdout, sys.stderr = _real_stdout, _real_stderr
        log_file.close()
