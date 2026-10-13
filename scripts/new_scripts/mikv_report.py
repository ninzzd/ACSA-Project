"""
Reading a sweep back out of its result rows: grouping them by configuration,
rendering the markdown tables, and appending them to the results CSV.

Everything here groups by *configuration* -- the tuple of every swept axis except
the budget ratio, which is the x axis within a configuration (CONFIG_AXES). Rows
are read with `.get` and defaults throughout, so a CSV or a result list written
by an older version of this code -- which had fewer axes -- still groups
correctly, landing on the default that version implied.

The tables are built to be read in one order: per-configuration curves first,
then the marginal effect of each axis, then the Pareto front, which is the actual
shortlist -- everything off it is strictly dominated by something else in the
same sweep.
"""

import csv
import datetime
import os
import re

from mikv_config import (
    DEFAULT_BALANCE_SCHEME,
    HW_AGE_LUT_ENTRIES,
    DEFAULT_BUDGET_MODE,
    DEFAULT_HIGH_BITS,
    DEFAULT_LOW_BITS,
    DEFAULT_SCORE_BITS,
    DEFAULT_SCORE_DECAY_SCHEME,
    DEFAULT_SCORE_SCHEME,
    DEFAULT_WINDOW_RATIO,
    HW_SCORE_FRAC_BITS,
    HW_SCORE_LENGTH_BITS,
    HW_SCORE_SIGNED,
    BALANCE_SCHEME_LABELS,
    BUDGET_MODE_LABELS,
    format_score_tag,
    high_precision_of,
)

# ---- reading a sweep back out of its result rows ----
#
# Everything below groups, labels and plots results by *configuration*: the tuple
# of every swept axis except the budget ratio, which is the x axis within a
# configuration. Rows are read with `.get` and defaults throughout, so a CSV or a
# result list written by an older version of this script -- which had fewer axes
# -- still groups correctly, landing on the default that version implied.
#
# The score axes are folded into one string (`row_score_tag`) rather than kept
# as six separate components: they are one design decision (what the scoreboard
# register file looks like) and `format_score_tag` already renders them compactly.
CONFIG_AXES = (
    "budget_mode",
    "scheme",
    "window",
    "high_tier",
    "low_bits",
    "score_tag",
)


def row_score_tag(r: dict) -> str:
    """The scoreboard-precision tag of one result row, tolerating rows written
    before `score_scheme` existed (they were all native fp16)."""
    scheme = r.get("score_scheme") or DEFAULT_SCORE_SCHEME
    bits = r.get("score_bits") or DEFAULT_SCORE_BITS
    decay = r.get("score_decay_scheme") or DEFAULT_SCORE_DECAY_SCHEME
    # Rows written before this column existed were all compounding, so they must
    # NOT inherit today's default -- that would silently relabel them.
    site = r.get("score_decay_application") or "compounding"
    length = r.get("score_length_bits") or HW_SCORE_LENGTH_BITS
    frac = r.get("score_frac_bits") or HW_SCORE_FRAC_BITS
    signed = r.get("score_signed")
    signed = HW_SCORE_SIGNED if signed in (None, "") else str(signed).lower() in ("true", "1")
    lut = r.get("hw_age_lut_entries") or HW_AGE_LUT_ENTRIES
    return format_score_tag(
        scheme, int(bits), decay, site, int(length), int(frac), signed, int(lut)
    )


def row_high_tier(r: dict) -> str:
    """The `high_tiers` axis value of a row, from the two columns that encode it.
    A row with no `high_precision_native` column predates the flag, when the
    important bucket was always left native."""
    native = r.get("high_precision_native")
    native = True if native in (None, "") else (
        native if isinstance(native, bool) else str(native).lower() in ("true", "1")
    )
    return high_precision_of(int(r.get("high_bits") or DEFAULT_HIGH_BITS), native)


def row_window(r: dict) -> str:
    """A row's recency window, as one label covering both ways of setting it: an
    absolute token count (`window_tokens`) or a fraction of k (`window_ratio`).
    They are alternatives, never both, so one config axis holds either."""
    tokens = r.get("window_tokens")
    if tokens not in (None, ""):
        return f"w={int(tokens)}"
    ratio = r.get("window_ratio")
    ratio = DEFAULT_WINDOW_RATIO if ratio in (None, "") else float(ratio)
    return f"w={ratio:g}k"


def row_config(r: dict) -> tuple:
    """One row's configuration, as a tuple aligned with CONFIG_AXES."""
    return (
        r.get("budget_mode") or DEFAULT_BUDGET_MODE,
        r.get("scheme") or DEFAULT_BALANCE_SCHEME,
        row_window(r),
        row_high_tier(r),
        int(r.get("low_bits") or DEFAULT_LOW_BITS),
        row_score_tag(r),
    )


def config_get(config: tuple, axis: str):
    return config[CONFIG_AXES.index(axis)]


def configs_in(results: list[dict]) -> list[tuple]:
    """Distinct configurations present in `results`, in first-seen order."""
    seen = []
    for r in results:
        config = row_config(r)
        if config not in seen:
            seen.append(config)
    return seen


def varying_config_axes(results: list[dict]) -> tuple[str, ...]:
    """The CONFIG_AXES that take more than one value across `results`. Labels,
    titles and filenames name exactly these: naming an axis that never moved is
    noise, and omitting one that did makes two different runs look like one."""
    configs = configs_in(results)
    return tuple(
        axis for axis in CONFIG_AXES if len({config_get(c, axis) for c in configs}) > 1
    )


def rows_for(results: list[dict], config: tuple) -> list[dict]:
    return [r for r in results if row_config(r) == config]


def axis_value_label(axis: str, value) -> str:
    """One axis value, rendered for a human. Each axis needs its own form -- a
    bare `0.5` is meaningless without knowing whether it is w/k or r."""
    if axis == "budget_mode":
        return BUDGET_MODE_LABELS.get(value, value)
    if axis == "scheme":
        return BALANCE_SCHEME_LABELS.get(value, value)
    if axis == "window":
        return str(value)
    if axis == "high_tier":
        return f"HIGH {value}"
    if axis == "low_bits":
        return f"LOW int{value}"
    return str(value)


def axis_value_slug(axis: str, value) -> str:
    """Same, but filename-safe and terse."""
    if axis == "window":
        return str(value).replace("=", "")
    if axis == "high_tier":
        return f"hi{value}"
    if axis == "low_bits":
        return f"lo{value}"
    # Underscores are kept, not swapped for hyphens: the budget modes are
    # spelled "fixed_length"/"fixed_ratio", and rewriting them would rename every
    # figure this script has ever written.
    return re.sub(r"[^0-9A-Za-z._+-]+", "-", str(value))


def config_label(config: tuple, axes: tuple[str, ...] | None = None) -> str:
    """Human-readable name for one configuration, naming only `axes` (the axes
    that vary in this result set). With nothing varying -- a single-configuration
    sweep -- fall back to naming the budget mode and balancer, which is what the
    titles said before there were other axes."""
    axes = axes if axes else ("budget_mode", "scheme")
    return " / ".join(axis_value_label(a, config_get(config, a)) for a in axes)


def config_slug(config: tuple, axes: tuple[str, ...] | None = None) -> str:
    """Filename fragment for one configuration. The budget mode and balancer are
    always included and always first, so a default sweep keeps writing the same
    `..._<mode>_<scheme>.png` paths it always has; any other axis appears only
    when it varies."""
    axes = ("budget_mode", "scheme") + tuple(
        a for a in (axes or ()) if a not in ("budget_mode", "scheme")
    )
    return "_".join(axis_value_slug(a, config_get(config, a)) for a in axes)


def pareto_front(results: list[dict]) -> list[dict]:
    """
    The rows no other row beats on both axes at once -- i.e. for which nothing
    else is simultaneously smaller (`compression_pct`) and at least as accurate.

    This is the answer to "which configuration should the hardware use": every
    row off the front is strictly wasteful, because some other configuration in
    the same sweep is both smaller and no less accurate. Ties are kept (a row is
    dropped only when another is strictly better on one axis and no worse on the
    other), so two configurations that land on the same point both survive and
    the choice between them falls to what the sweep does not measure -- area,
    critical path, verification cost.
    """
    front = []
    for r in results:
        dominated = any(
            other is not r
            and other["compression_pct"] <= r["compression_pct"]
            and other["accuracy"] >= r["accuracy"]
            and (
                other["compression_pct"] < r["compression_pct"]
                or other["accuracy"] > r["accuracy"]
            )
            for other in results
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["compression_pct"])


def axis_effect_summary(results: list[dict]) -> dict[str, list[dict]]:
    """
    Per varying axis, one entry per value it takes: how many runs used it, and
    the mean / best accuracy and mean footprint over them.

    Read this as a *marginal* effect, and only that. Under sweep_mode="ofat" the
    other axes are held at the baseline, so the comparison is clean but local to
    that baseline. Under "grid" the mean is taken over whatever else the grid
    happened to cross, so it is an average over the other axes rather than a
    controlled comparison -- fine for ranking which knobs matter, wrong for
    reading an exact cost off. The compression column is what says which is
    which: an axis whose values sit at the same mean footprint (the balancer,
    the scoreboard) is being compared at equal cost; one whose values sit at
    different footprints (r, w, the bit widths) is not, and its accuracy
    difference is partly just a difference in how much cache it kept.
    """
    summary = {}
    for axis in varying_config_axes(results):
        buckets: dict[object, list[dict]] = {}
        for r in results:
            buckets.setdefault(config_get(row_config(r), axis), []).append(r)
        entries = []
        for value, rows in buckets.items():
            entries.append(
                dict(
                    value=value,
                    label=axis_value_label(axis, value),
                    runs=len(rows),
                    mean_accuracy=sum(x["accuracy"] for x in rows) / len(rows),
                    best_accuracy=max(x["accuracy"] for x in rows),
                    mean_compression=sum(x["compression_pct"] for x in rows) / len(rows),
                )
            )
        summary[axis] = sorted(entries, key=lambda e: -e["mean_accuracy"])
    return summary


def format_results_table(results: list[dict]) -> str:
    """
    Render the sweep as markdown: one section per configuration, then three
    cross-cutting summaries -- the per-axis marginal effect, the accuracy/
    footprint Pareto front, and a ranked list of the best configurations.
    Returned as a string so the caller can both print it (into the tee'd log)
    and paste it into the docs.

    Only the axes that actually varied are named anywhere, so a
    single-configuration sweep renders as compactly as it always did and a
    twelve-axis one stays readable.
    """
    if not results:
        return "(no results)"
    varying = varying_config_axes(results)
    configs = configs_in(results)
    lines = []

    baseline = results[0]["baseline_accuracy"] * 100
    seq_len = results[0]["seq_len"]
    kv_before_mb = results[0]["kv_size_before"] / 1e6
    lines.append(
        f"Sizing basis: fixed context seq_len={seq_len} tokens, "
        f"uncompressed KV = {kv_before_mb:.1f} MB. "
        f"Uncompressed baseline accuracy = {baseline:.1f}%."
    )
    lines.append(
        f"Configurations: {len(configs)} over {len(results)} runs. "
        + (f"Varying axes: {', '.join(varying)}." if varying else "A single configuration.")
    )
    lines.append("")

    for config in configs:
        rows = rows_for(results, config)
        lines.append(f"### {config_label(config, varying)}")
        lines.append("")
        lines.append("| ratio | k | KV after (MB) | KV size (% of uncompressed) | accuracy |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| - (uncompressed) | - | {kv_before_mb:.1f} | 100.0% | {baseline:.1f}% |")
        for r in sorted(rows, key=lambda x: x["ratio"]):
            lines.append(
                f"| {r['ratio']} | {r['k']} | {r['kv_size_after'] / 1e6:.1f} | "
                f"{r['compression_pct']:.1f}% | {r['accuracy'] * 100:.1f}% |"
            )
        lines.append("")

    # --- per-axis marginal effect ---
    if varying:
        lines.append("### Axis effects (marginal, at the footprint each value implies)")
        lines.append("")
        lines.append("| axis | value | runs | mean accuracy | best accuracy | mean KV size |")
        lines.append("|---|---|---|---|---|---|")
        for axis, entries in axis_effect_summary(results).items():
            for e in entries:
                lines.append(
                    f"| {axis} | {e['label']} | {e['runs']} | "
                    f"{e['mean_accuracy'] * 100:.1f}% | {e['best_accuracy'] * 100:.1f}% | "
                    f"{e['mean_compression']:.1f}% |"
                )
        lines.append("")
        lines.append(
            "Axes whose values sit at the *same* mean KV size (balancer, scoreboard, "
            "decay) are compared at equal cost -- an accuracy difference there is a "
            "real difference in selection quality, and a cheaper value that holds "
            "accuracy is free in area terms. Axes whose values sit at different KV "
            "sizes (r, w, the bit widths) are not: part of their accuracy "
            "difference is just keeping more cache."
        )
        lines.append("")

    # --- head-to-head across the balancer, at otherwise-identical settings ---
    schemes = sorted({config_get(c, "scheme") for c in configs})
    if len(schemes) > 1:
        lines.append("### Balancer head-to-head (accuracy at identical everything else)")
        lines.append("")
        other_axes = tuple(a for a in varying if a != "scheme")
        key_header = " | ".join(other_axes) if other_axes else "configuration"
        lines.append(f"| {key_header} | ratio | k | KV size (%) | " + " | ".join(schemes) + " |")
        lines.append("|---" * (len(other_axes or ("x",)) + 3) + "|---" * len(schemes) + "|")
        # Keyed on every varying axis *except* the balancer, plus the ratio: only
        # rows agreeing on all of those sit at the same footprint under the same
        # policy, so only those are a fair balancer-vs-balancer comparison.
        by_point: dict[tuple, dict[str, dict]] = {}
        for r in results:
            config = row_config(r)
            key = tuple(config_get(config, a) for a in other_axes) + (r["ratio"],)
            by_point.setdefault(key, {})[config_get(config, "scheme")] = r
        for key in sorted(by_point, key=lambda k: tuple(str(v) for v in k)):
            per_scheme = by_point[key]
            any_row = next(iter(per_scheme.values()))
            cells = [
                f"{per_scheme[s]['accuracy'] * 100:.1f}%" if s in per_scheme else "-" for s in schemes
            ]
            key_cells = [
                axis_value_label(a, v) for a, v in zip(other_axes, key[:-1])
            ] or ["(single)"]
            lines.append(
                "| " + " | ".join(key_cells) + f" | {key[-1]} | {any_row['k']} | "
                f"{any_row['compression_pct']:.1f}% | " + " | ".join(cells) + " |"
            )
        lines.append("")

    # --- what to actually build ---
    front = pareto_front(results)
    lines.append("### Pareto front (nothing else is both smaller and at least as accurate)")
    lines.append("")
    lines.append("| KV size (%) | accuracy | ratio | k | configuration |")
    lines.append("|---|---|---|---|---|")
    for r in front:
        lines.append(
            f"| {r['compression_pct']:.1f}% | {r['accuracy'] * 100:.1f}% | {r['ratio']} | "
            f"{r['k']} | {config_label(row_config(r), varying)} |"
        )
    lines.append("")

    lines.append("### Top configurations by accuracy")
    lines.append("")
    lines.append("| accuracy | KV size (%) | ratio | k | configuration |")
    lines.append("|---|---|---|---|---|")
    ranked = sorted(results, key=lambda r: (-r["accuracy"], r["compression_pct"]))[:15]
    for r in ranked:
        lines.append(
            f"| {r['accuracy'] * 100:.1f}% | {r['compression_pct']:.1f}% | {r['ratio']} | "
            f"{r['k']} | {config_label(row_config(r), varying)} |"
        )
    lines.append("")

    return "\n".join(lines)


# Column order for the results CSV. Fixed (rather than taken from the first
# row's keys) because the file is *appended* to across runs: every run has to
# write the same columns in the same order under one header, and a row missing
# a key -- an older result dict, say -- must land as an empty cell rather than
# silently shifting every subsequent column. Leading config columns identify
# what produced each row (budget mode, balancer, bit widths, ...), so rows from
# different runs stay distinguishable once they're interleaved in one file.
RESULTS_CSV_FIELDS = (
    "timestamp",
    "run_id",
    "model_name",
    "budget_mode",
    "scheme",
    "score_scheme",
    "score_bits",
    "score_decay_scheme",
    "score_decay_application",
    "score_length_bits",
    "score_frac_bits",
    "score_signed",
    "hw_delta_bits",
    "hw_age_lut_entries",
    "ratio",
    "window_tokens",
    "window_ratio",
    "high_bits",
    "low_bits",
    "high_precision_native",
    "sweep_mode",
    "num_samples",
    "num_records",
    "max_tokens",
    "seed",
    "t_p",
    "k",
    "seq_len",
    "accuracy",
    "baseline_accuracy",
    "kv_size_before",
    "kv_size_after",
    "compression_pct",
    "avg_seq_len",
    "kv_bytes_occupied",
)


def write_results_csv(
    results: list[dict],
    save_path: str = "docs/kv_compression_sweep.csv",
    run_id: str | None = None,
) -> str:
    """
    Append the sweep rows to `save_path` (one row per (budget mode, scheme,
    ratio)), creating the file with a header if it doesn't exist yet and
    reusing the existing header otherwise. Appending, not overwriting, so
    results accumulate across invocations -- each row carries the full
    configuration that produced it (see RESULTS_CSV_FIELDS) plus a
    `timestamp` and a `run_id` shared by every row of one sweep, so a
    single run can be pulled back out of the accumulated file.

    Rows are restricted to RESULTS_CSV_FIELDS; any extra key on a result
    dict is dropped rather than corrupting the column alignment of a file
    whose header was written by an earlier version.
    """
    if not results:
        return save_path
    stamp = datetime.datetime.now()
    run_id = run_id or f"{stamp:%Y%m%d_%H%M%S}"
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_header = not os.path.exists(save_path) or os.path.getsize(save_path) == 0
    if not write_header:
        # An existing file was written under some header; appending rows ordered by
        # RESULTS_CSV_FIELDS under a *different* one would put every value in the
        # wrong column, silently. If the headers don't match (a file from an older
        # version of this script, say), divert to a fresh timestamped sibling rather
        # than either corrupting it or raising -- a sweep is hours of GPU time and
        # must not lose its results to a bookkeeping mismatch.
        with open(save_path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != list(RESULTS_CSV_FIELDS):
            stem, ext = os.path.splitext(save_path)
            diverted = f"{stem}_{run_id}{ext}"
            print(
                f"[csv] WARNING: {save_path} has a different header "
                f"({len(existing)} columns vs. {len(RESULTS_CSV_FIELDS)} expected) -- "
                f"writing to {diverted} instead of appending",
                flush=True,
            )
            save_path, write_header = diverted, True
    with open(save_path, "a", newline="") as f:
        # extrasaction="ignore": keys outside RESULTS_CSV_FIELDS are skipped
        # instead of raising, so a caller-added diagnostic can't break the append.
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in RESULTS_CSV_FIELDS}
            row["timestamp"] = f"{stamp:%Y-%m-%d %H:%M:%S}"
            row["run_id"] = run_id
            writer.writerow(row)
    print(
        f"[csv] appended {len(results)} rows (run_id={run_id}) to {save_path}"
        f"{' (new file)' if write_header else ''}",
        flush=True,
    )
    return save_path
