"""
The GSM8K CSVs: the per-problem checkpoint rows every job writes, and the
per-run summary rows `mikv_gsm8k.py merge` builds from them. The GSM8K
counterpart of mikv_report, whose grouping (`row_config`) and markdown tables it
reuses unchanged.

--- the two files ---

gsm8k_problems.csv  one row per (run, problem) -- the source of truth. Written
                    one row at a time by each array task, flushed and fsync'd,
                    so a job killed by the walltime limit loses at most the
                    problem it was on, and the next submission resumes after the
                    last written row. Every statistic below is recomputed from
                    these rows at merge time, so sharding and resuming cannot
                    change a result.
gsm8k_summary.csv   one row per (sweep, run). Leading columns are Line
                    Retrieval's, spelled identically (budget_mode, scheme, ...,
                    ratio, k, accuracy, baseline_accuracy, compression_pct), so a
                    GSM8K row and a Line Retrieval row of the same configuration
                    carry the same `config_id` and can be joined on
                    (config_id, ratio); the GSM8K-specific statistics follow.

--- crash safety ---

Each part file is written by exactly one task, so there are no concurrent
writers. The completion is JSON-encoded so a record never spans lines, and every
record ends in `row_ok=1`: a line cut off mid-write is missing that sentinel (or
has the wrong field count) and is dropped on read, and the writer starts a fresh
line before appending, so a torn record never corrupts the next one.
"""

import csv
import glob
import math
import os
import statistics

from mikv_report import row_config

# Per-problem rows. Config columns match mikv_sweep.config_row_fields.
PROBLEM_CSV_FIELDS = (
    "timestamp",
    "run_key",
    "method",
    "job",
    "gpu",
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
    "n_shot",
    "max_new_tokens",
    "max_tokens",
    "prompt_version",
    "problem_idx",
    "gold",
    "pred_strict",
    "pred_flexible",
    "correct_strict",
    "correct_flexible",
    "t_p",
    "k",
    "w_eff",
    "k_h",
    "gen_tokens",
    "stop_reason",
    "seq_len",
    "avg_seq_len",
    "kv_size_before",
    "kv_size_after",
    "compression_pct",
    "compression_pct_occupied",
    "seconds",
    "completion",
    "row_ok",
)

SUMMARY_CSV_FIELDS = (
    # identity
    "sweep",
    "method",
    "run_key",
    "config_id",
    # configuration -- Line Retrieval's columns, same names, same encoding
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
    # progress
    "n_problems",
    "n_done",
    "complete",
    # accuracy (fractions, like Line Retrieval's `accuracy`)
    "accuracy",
    "correct",
    "accuracy_ci95_lo",
    "accuracy_ci95_hi",
    "baseline_accuracy",
    "n_paired",
    "delta_accuracy",
    "n_lost",
    "n_gained",
    "mcnemar_p",
    "accuracy_flexible",
    "baseline_accuracy_flexible",
    "no_answer_pct",
    "hit_max_new_tokens_pct",
    # budget and footprint
    "t_p",
    "k",
    "w_eff",
    "pure_recency_pct",
    "gen_tokens",
    "seq_len",
    "avg_seq_len",
    "kv_size_before",
    "kv_size_after",
    "compression_pct",
    "compression_pct_occupied",
    # cost
    "sec_per_problem",
    # the matching Line Retrieval rows (merge --line-retrieval-csv)
    "lr_accuracy",
    "lr_baseline_accuracy",
    "lr_matches",
    "lr_sources",
    # run settings
    "num_problems",
    "problem_seed",
    "n_shot",
    "max_new_tokens",
    "max_tokens",
    "prompt_version",
    "model_name",
)

# Columns that describe the configuration, copied verbatim from a run's rows.
CONFIG_COLUMNS = (
    "budget_mode", "scheme", "score_scheme", "score_bits", "score_decay_scheme",
    "score_decay_application", "score_length_bits", "score_frac_bits", "score_signed",
    "hw_delta_bits", "hw_age_lut_entries", "ratio", "window_tokens", "window_ratio",
    "high_bits", "low_bits", "high_precision_native",
)


# ---- the per-problem part files ----


class PartWriter:
    """Appends per-problem rows to one part file, durably, one row at a time."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        if not new:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
        self._file = open(path, "a", newline="")
        if not new and torn:
            self._file.write("\n")  # isolate a record cut off by a killed job
        self._writer = csv.DictWriter(self._file, fieldnames=PROBLEM_CSV_FIELDS,
                                      extrasaction="ignore", lineterminator="\n")
        if new:
            self._writer.writeheader()
            self._flush()

    def write(self, row: dict) -> None:
        self._writer.writerow({**{k: row.get(k, "") for k in PROBLEM_CSV_FIELDS}, "row_ok": 1})
        self._flush()

    def _flush(self) -> None:
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def _read_part(path: str) -> list[dict]:
    """The valid rows of one part file; torn or foreign-header lines are skipped."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows
        if tuple(header) != PROBLEM_CSV_FIELDS:
            print(f"[gsm8k] WARNING: {path} has an unexpected header; skipped", flush=True)
            return rows
        for record in reader:
            if len(record) != len(PROBLEM_CSV_FIELDS) or record[-1] != "1":
                continue
            row = dict(zip(PROBLEM_CSV_FIELDS, record))
            try:
                row["problem_idx"] = int(row["problem_idx"])
            except ValueError:
                continue
            rows.append(row)
    return rows


def part_files(parts_dir: str, run_key: str | None = None) -> list[str]:
    pattern = f"{run_key}__*.csv" if run_key else "*.csv"
    return sorted(glob.glob(os.path.join(parts_dir, pattern)))


def done_problems(parts_dir: str, run_key: str) -> set[int]:
    """Problems of `run_key` already written by any task -- including tasks of an
    earlier or concurrent submission, and of other sweeps sharing the run."""
    done = set()
    for path in part_files(parts_dir, run_key):
        done.update(r["problem_idx"] for r in _read_part(path))
    return done


def load_all_rows(parts_dir: str) -> tuple[dict[str, dict[int, dict]], int]:
    """
    Every valid per-problem row, as {run_key: {problem_idx: row}}, and the number
    of duplicates whose answer disagreed with the first copy.

    Duplicates arise legitimately -- two submissions racing on a shared run -- and
    greedy decoding should reproduce them exactly; a disagreement means GPU
    nondeterminism flipped an argmax somewhere, which is worth knowing about
    rather than silently averaging over.
    """
    runs: dict[str, dict[int, dict]] = {}
    conflicts = 0
    for path in part_files(parts_dir):
        for row in _read_part(path):
            per_run = runs.setdefault(row["run_key"], {})
            first = per_run.get(row["problem_idx"])
            if first is None:
                per_run[row["problem_idx"]] = row
            elif first["pred_strict"] != row["pred_strict"]:
                conflicts += 1
    return runs, conflicts


def measured_seconds(parts_dir: str) -> dict[str, float]:
    """
    Seconds per problem, per method, measured from every row written so far --
    what the plan sizes --time from once the probe has run.

    The slowest run's mean, not the overall mean: configurations differ in cost
    (the fixed-point score path runs in float64), and one --time covers every
    task in an array, so it has to fit the slowest of them.
    """
    per_run: dict[tuple[str, str], list[float]] = {}
    for path in part_files(parts_dir):
        for row in _read_part(path):
            try:
                per_run.setdefault((row["method"], row["run_key"]), []).append(float(row["seconds"]))
            except ValueError:
                pass
    slowest: dict[str, float] = {}
    for (method, _), values in per_run.items():
        slowest[method] = max(slowest.get(method, 0.0), statistics.fmean(values))
    return slowest


# ---- statistics ----


def wilson_interval(correct: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion -- well-behaved at the
    small n and low accuracies a GSM8K screen runs at, unlike the normal one."""
    if n == 0:
        return float("nan"), float("nan")
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact_p(lost: int, gained: int) -> float:
    """
    Two-sided exact McNemar test on the discordant pairs: `lost` problems the
    baseline got right and this run got wrong, `gained` the reverse. Under "the
    cache changes nothing" each discordant problem is a fair coin, so the p-value
    is a binomial tail. Concordant problems carry no information and do not enter
    -- which is why a paired test on 250 problems resolves differences far below
    the +/- 5 point CI of either accuracy alone.
    """
    n = lost + gained
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _mean(values) -> float | str:
    values = [float(v) for v in values if v not in ("", None)]
    return statistics.fmean(values) if values else ""


def config_id(row: dict) -> str:
    """The configuration as one string, from `mikv_report.row_config` -- identical
    for a GSM8K summary row and a Line Retrieval CSV row of the same config."""
    if row.get("method") == "baseline":
        return "uncompressed"
    return " / ".join(str(v) for v in row_config(row))


def summarize_run(
    sweep: str,
    run_key: str,
    rows_by_problem: dict[int, dict],
    problems: list[int],
    baseline_by_problem: dict[int, dict],
    num_problems: int,
    problem_seed: int,
) -> dict | None:
    """One summary row: `run_key` restricted to the sweep's `problems`, paired
    against the baseline on the problems both have. None if nothing is done yet."""
    rows = [rows_by_problem[i] for i in problems if i in rows_by_problem]
    if not rows:
        return None
    first = rows[0]
    n = len(rows)
    correct = sum(int(r["correct_strict"]) for r in rows)
    lo, hi = wilson_interval(correct, n)

    summary = {c: first.get(c, "") for c in CONFIG_COLUMNS}
    summary.update(
        sweep=sweep,
        method=first["method"],
        run_key=run_key,
        n_problems=len(problems),
        n_done=n,
        complete=n == len(problems),
        accuracy=correct / n,
        correct=correct,
        accuracy_ci95_lo=lo,
        accuracy_ci95_hi=hi,
        accuracy_flexible=sum(int(r["correct_flexible"]) for r in rows) / n,
        no_answer_pct=100 * sum(r["pred_strict"] == "" for r in rows) / n,
        hit_max_new_tokens_pct=100 * sum(r["stop_reason"] == "max_new_tokens" for r in rows) / n,
        t_p=_mean(r["t_p"] for r in rows),
        k=_mean(r["k"] for r in rows),
        w_eff=_mean(r["w_eff"] for r in rows),
        pure_recency_pct=(
            100 * sum(r["k_h"] == "0" for r in rows) / n if first["method"] != "baseline" else ""
        ),
        gen_tokens=_mean(r["gen_tokens"] for r in rows),
        seq_len=first["seq_len"],
        avg_seq_len=_mean(r["avg_seq_len"] for r in rows),
        kv_size_before=_mean(r["kv_size_before"] for r in rows),
        kv_size_after=_mean(r["kv_size_after"] for r in rows),
        compression_pct=_mean(r["compression_pct"] for r in rows),
        compression_pct_occupied=_mean(r["compression_pct_occupied"] for r in rows),
        sec_per_problem=_mean(r["seconds"] for r in rows),
        num_problems=num_problems,
        problem_seed=problem_seed,
        n_shot=first["n_shot"],
        max_new_tokens=first["max_new_tokens"],
        max_tokens=first["max_tokens"],
        prompt_version=first["prompt_version"],
        model_name=first["model_name"],
    )
    summary["config_id"] = config_id(summary)

    paired = [(r, baseline_by_problem[r["problem_idx"]]) for r in rows
              if r["problem_idx"] in baseline_by_problem]
    if paired:
        mine = [int(r["correct_strict"]) for r, _ in paired]
        base = [int(b["correct_strict"]) for _, b in paired]
        lost = sum(b and not m for m, b in zip(mine, base))
        gained = sum(m and not b for m, b in zip(mine, base))
        summary.update(
            baseline_accuracy=sum(base) / len(paired),
            baseline_accuracy_flexible=(
                sum(int(b["correct_flexible"]) for _, b in paired) / len(paired)
            ),
            n_paired=len(paired),
            delta_accuracy=(sum(mine) - sum(base)) / len(paired),
            n_lost=lost,
            n_gained=gained,
            mcnemar_p=mcnemar_exact_p(lost, gained),
        )
    return summary


# ---- joining Line Retrieval ----


def load_line_retrieval(patterns: list[str], num_records: int | None) -> dict[tuple, list[dict]]:
    """Line Retrieval CSV rows keyed by (row_config, ratio), from every file the
    glob `patterns` match. `num_records` keeps only rows at that record count, so
    S6's longer-context runs (70 records) do not average into the 40-record ones."""
    index: dict[tuple, list[dict]] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            # results/followup/<sweep>/legacy/sweep.csv -> "<sweep>"
            parts = os.path.normpath(path).split(os.sep)
            source = parts[-3] if len(parts) >= 3 else path
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        if num_records is not None and int(row.get("num_records") or 0) != num_records:
                            continue
                        key = (row_config(row), float(row["ratio"]))
                        row["_source"] = source
                        index.setdefault(key, []).append(row)
                    except (KeyError, ValueError):
                        continue
    return index


def attach_line_retrieval(summary: dict, index: dict[tuple, list[dict]]) -> None:
    if summary.get("method") == "baseline" or not index:
        return
    matches = index.get((row_config(summary), float(summary["ratio"])), [])
    if not matches:
        summary["lr_matches"] = 0
        return
    summary.update(
        lr_accuracy=statistics.fmean(float(r["accuracy"]) for r in matches),
        lr_baseline_accuracy=statistics.fmean(float(r["baseline_accuracy"]) for r in matches),
        lr_matches=len(matches),
        lr_sources=";".join(sorted({r["_source"] for r in matches})),
    )


# ---- writing ----


def _cell(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def write_csv_atomic(path: str, fields: tuple[str, ...], rows: list[dict]) -> None:
    """Write the whole file to a temporary sibling and rename it into place, so a
    reader never sees a half-written merge."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k, "")) for k in fields})
    os.replace(tmp, path)
