"""
The figures. Optional: plotting is off by default (`--plots`), because the CSV
and the tables are a sweep's product and every figure here regenerates from them
-- a matplotlib error must not be able to take down hours of finished GPU work.

Three views, in the order they answer questions: the accuracy-vs-compression
curves per configuration and overlaid, the marginal effect of each swept axis,
and the Pareto front over every run.

`import mikv_runtime` below is LOAD-BEARING and must stay first: torch and
transformers have to be imported before matplotlib or the process segfaults with
no traceback. See mikv_runtime.
"""

import math
import os

import mikv_runtime  # noqa: F401  -- must precede matplotlib; see the module docstring

import matplotlib

matplotlib.use("Agg")  # headless-safe: write figures to disk, never open a GUI window
import matplotlib.pyplot as plt

from mikv_report import (
    axis_effect_summary,
    config_get,
    config_label,
    config_slug,
    configs_in,
    pareto_front,
    row_config,
    rows_for,
    varying_config_axes,
)

# Categorical series colors, in fixed order. Taken from a palette validated for
# colour-vision deficiency on its *adjacent* pairlist (the one that governs
# lines), so the order is load-bearing -- slot 3 next to slot 4 is what was
# checked, an arbitrary reordering is not. Hues are never cycled: past
# MAX_SERIES_PER_FIGURE series a pair would repeat and identity would be lost, so
# the overlay facets into several figures instead of reusing a colour.
_SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
# Marker shape carries the same identity as colour, so a CVD reader, a greyscale
# print and a screenshot all stay readable without the legend colour.
_SERIES_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
MAX_SERIES_PER_FIGURE = len(_SERIES_COLORS)
_GRID_COLOR = "#c9c9c4"
_INK_MUTED = "#52514e"


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    """The recessive frame every figure here shares: light grid behind the data,
    no top/right spines, muted axis ink."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.35, color=_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID_COLOR)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)


def _baseline_line(ax, results: list[dict]) -> None:
    ax.axhline(
        results[0]["baseline_accuracy"] * 100,
        linestyle="--",
        linewidth=1,
        color=_INK_MUTED,
        label="uncompressed baseline",
    )


# Above this many configurations, the per-configuration figures are skipped by
# default: a full grid can hold hundreds, and writing a PNG per point buries the
# three figures anyone actually reads (the overlay, the axis effects, the Pareto
# front) in a directory listing. Pass per_config=True to force them anyway.
MAX_PER_CONFIG_FIGURES = 12


def plot_accuracy_vs_compression(
    results: list[dict],
    save_path: str = "docs/kv_compression_sweep.png",
    per_config: bool | None = None,
) -> list[str]:
    """
    Plot Line Retrieval accuracy against KV cache compression
    (100 * size_after(k) / size_before) across the sweep in `results` (as
    returned by `sweep_kv_compression`).

    Writes one figure per configuration -- `save_path` with the
    configuration's slug appended before the extension (e.g.
    `..._fixed_length_paper.png`, `..._fixed_ratio_pow2_lo4.png`; only axes
    that varied appear in the name, so a default sweep keeps the filenames
    it always had) -- plus, when more than one configuration ran, a
    combined overlay at `save_path` itself for direct comparison. Saves to
    disk since matplotlib runs headless (Agg backend) here. Returns the
    list of paths written.

    `per_config` gates those individual figures; left as None it draws them
    only while there are at most MAX_PER_CONFIG_FIGURES configurations,
    since a large grid would otherwise write hundreds of PNGs nobody opens.

    Past MAX_SERIES_PER_FIGURE configurations the overlay is split across
    several numbered figures rather than cycling colours: a repeated hue in
    a legend of twenty is not a legend. Configurations are sorted by best
    accuracy first, so the interesting curves land together on the first
    facet.

    Note that across budget modes -- and across any other axis that changes
    the footprint -- the x axis is not a shared grid: at the same r,
    "fixed_ratio" and "fixed_length" land at different k. That's exactly
    what the overlay is for: it puts every configuration's accuracy/
    compression trade-off on one axis, where a point sitting up and to the
    left is the better deal regardless of which configuration produced it.
    """
    if not results:
        return []
    configs = configs_in(results)
    varying = varying_config_axes(results)
    stem, ext = os.path.splitext(save_path)
    written = []

    def _draw(ax, rows, label=None, annotate=True, color=None, marker="o"):
        ordered = sorted(rows, key=lambda r: r["compression_pct"])
        xs = [r["compression_pct"] for r in ordered]
        ys = [r["accuracy"] * 100 for r in ordered]
        ax.plot(
            xs, ys, marker=marker, markersize=6, linewidth=2, color=color,
            label=label, markeredgecolor="white", markeredgewidth=0.8,
        )
        if not annotate:
            return
        for r, x, y in zip(ordered, xs, ys):
            ax.annotate(
                f"k={r['k']} (r={r['ratio']})",
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                color=_INK_MUTED,
            )

    # one figure per configuration -- see MAX_PER_CONFIG_FIGURES
    if per_config is None:
        per_config = len(configs) <= MAX_PER_CONFIG_FIGURES
    if not per_config and len(configs) > 1:
        print(
            f"[plot] {len(configs)} configurations > MAX_PER_CONFIG_FIGURES="
            f"{MAX_PER_CONFIG_FIGURES}: skipping the per-configuration figures, "
            f"writing the overlay only (pass per_config=True to force them)",
            flush=True,
        )
    for config in configs if per_config else []:
        rows = rows_for(results, config)
        path = f"{stem}_{config_slug(config, varying)}{ext}"
        print(f"[plot] rendering {config_label(config, varying)} ({len(rows)} points)...", flush=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        # A single series needs no legend box -- the title names it -- but the
        # baseline rule does, so the legend appears only to identify that.
        _draw(ax, rows, color=_SERIES_COLORS[0])
        _baseline_line(ax, rows)
        ax.legend(fontsize=8, frameon=False)
        _style_axes(
            ax,
            "KV cache size after compression (% of uncompressed)",
            "Line Retrieval accuracy (%)",
            f"MiKV: accuracy vs. KV compression\n{config_label(config, varying)}",
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved to {path}", flush=True)
        written.append(path)

    if len(configs) <= 1:
        return written

    # combined overlay, faceted at MAX_SERIES_PER_FIGURE series apiece
    def _best(config):
        return max(r["accuracy"] for r in rows_for(results, config))

    ordered_configs = sorted(configs, key=lambda c: -_best(c))
    facets = [
        ordered_configs[i : i + MAX_SERIES_PER_FIGURE]
        for i in range(0, len(ordered_configs), MAX_SERIES_PER_FIGURE)
    ]
    for facet_idx, facet in enumerate(facets):
        # The first facet keeps `save_path` itself, so the canonical overlay path
        # is stable whether or not the sweep was large enough to need facets.
        path = save_path if facet_idx == 0 else f"{stem}_overlay{facet_idx + 1}{ext}"
        print(f"[plot] rendering overlay {facet_idx + 1}/{len(facets)} ({len(facet)} series)...", flush=True)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for i, config in enumerate(facet):
            _draw(
                ax,
                rows_for(results, config),
                label=config_label(config, varying),
                # Annotate only the leading series: within a facet several
                # configurations often sit at the identical k/compression points,
                # and repeating the labels there just overplots them.
                annotate=i == 0,
                color=_SERIES_COLORS[i],
                marker=_SERIES_MARKERS[i],
            )
        _baseline_line(ax, results)
        _style_axes(
            ax,
            "KV cache size after compression (% of uncompressed)",
            "Line Retrieval accuracy (%)",
            "MiKV: accuracy vs. KV compression -- configuration comparison"
            + (f" ({facet_idx + 1}/{len(facets)}, best first)" if len(facets) > 1 else ""),
        )
        ax.legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved to {path}", flush=True)
        written.append(path)

    return written


def plot_axis_effects(
    results: list[dict], save_path: str = "docs/kv_compression_axes.png"
) -> str | None:
    """
    One small-multiple panel per varying axis: mean accuracy for each value that
    axis took, with the individual runs behind it as dots so the spread is
    visible rather than hidden by the mean.

    This is the figure that answers "which knob matters" at a glance -- the axis
    whose panel is flat is one the hardware can choose freely on cost, and the
    one with a tall step is where the accuracy is being spent. Read it with the
    same caveat as the axis-effects table (`axis_effect_summary`): under "grid"
    each bar averages over whatever else the grid crossed, so it ranks the knobs
    rather than pricing them exactly; under "ofat" the other axes are pinned and
    the comparison is controlled but local to the baseline.

    Returns the path written, or None when nothing varied.
    """
    summary = axis_effect_summary(results)
    if not summary:
        print("[plot] axis effects: nothing varied, skipping", flush=True)
        return None

    n = len(summary)
    cols = min(3, n)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.4 * rows_n), squeeze=False)
    baseline = results[0]["baseline_accuracy"] * 100

    for idx, (axis, entries) in enumerate(summary.items()):
        ax = axes[idx // cols][idx % cols]
        entries = sorted(entries, key=lambda e: str(e["label"]))
        labels = [e["label"] for e in entries]
        means = [e["mean_accuracy"] * 100 for e in entries]
        positions = range(len(entries))
        # Magnitude by category, one series -> one hue, not a rainbow.
        ax.bar(positions, means, color=_SERIES_COLORS[0], width=0.6, zorder=2)
        for pos, entry in zip(positions, entries):
            value = entry["value"]
            runs = [
                r["accuracy"] * 100
                for r in results
                if config_get(row_config(r), axis) == value
            ]
            # The individual runs, jittered horizontally so equal accuracies (very
            # common here -- accuracy is a count out of num_samples) stay countable.
            jitter = [pos + 0.22 * ((i % 5) - 2) / 4 for i in range(len(runs))]
            ax.scatter(jitter, runs, s=14, color=_INK_MUTED, alpha=0.55, zorder=3, linewidths=0)
        ax.axhline(baseline, linestyle="--", linewidth=1, color=_INK_MUTED)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        for pos, mean in zip(positions, means):
            ax.annotate(
                f"{mean:.1f}%", (pos, mean), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8, color=_INK_MUTED,
            )
        _style_axes(ax, "", "accuracy (%)", axis)
        ax.grid(False, axis="x")

    for idx in range(n, rows_n * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        "MiKV: marginal effect of each swept axis on Line Retrieval accuracy\n"
        f"(bars = mean over runs, dots = individual runs, dashed = uncompressed baseline {baseline:.1f}%)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved to {save_path}", flush=True)
    return save_path


def plot_pareto(
    results: list[dict], save_path: str = "docs/kv_compression_pareto.png"
) -> str:
    """
    Every run as one point in (KV footprint, accuracy), with the Pareto front
    drawn through the ones nothing else beats on both axes -- see
    `pareto_front`. The front is the shortlist a hardware configuration should
    be picked from; everything below it is strictly dominated by something else
    in the same sweep.

    Two colours only, and the front is labelled directly rather than by hue
    alone: a scatter of this many points cannot carry categorical identity in
    colour, so the configuration names go on the front's points as text.
    """
    if not results:
        return save_path
    varying = varying_config_axes(results)
    front = pareto_front(results)
    front_ids = {id(r) for r in front}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    dominated = [r for r in results if id(r) not in front_ids]
    if dominated:
        ax.scatter(
            [r["compression_pct"] for r in dominated],
            [r["accuracy"] * 100 for r in dominated],
            s=34, color=_GRID_COLOR, edgecolors="white", linewidths=0.6, zorder=2,
            label=f"dominated ({len(dominated)} runs)",
        )
    ax.plot(
        [r["compression_pct"] for r in front],
        [r["accuracy"] * 100 for r in front],
        marker="o", markersize=8, linewidth=2, color=_SERIES_COLORS[0],
        markeredgecolor="white", markeredgewidth=0.8, zorder=4,
        label=f"Pareto front ({len(front)} runs)",
    )
    for r in front:
        ax.annotate(
            f"{config_label(row_config(r), varying)}\nr={r['ratio']}, k={r['k']}",
            (r["compression_pct"], r["accuracy"] * 100),
            textcoords="offset points", xytext=(8, 6), fontsize=7, color=_INK_MUTED,
        )
    _baseline_line(ax, results)
    _style_axes(
        ax,
        "KV cache size after compression (% of uncompressed)",
        "Line Retrieval accuracy (%)",
        "MiKV: accuracy vs. KV footprint across the whole sweep\n"
        "(up and to the left is better; the front is the shortlist)",
    )
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved to {save_path}", flush=True)
    return save_path
