"""
HumanEval sweeps of the MiKV policy, run as SLURM job arrays: the HumanEval
counterpart of mikv_gsm8k.py, with the same plan / run / merge stages, the same
run identity (`mikv_gsm8k.run_key`, with HumanEval's generation settings) and the
same resume-from-parts behaviour.

    plan   expand a group (mikv_humaneval_sweeps) into units; no model, no GPU.
    run    one unit inside an array task: generate, extract, EXECUTE the tests,
           append one JSON line per problem to a part file.
    merge  rebuild the CSVs, the official-format sample JSONLs + meta sidecars,
           the per-problem pass matrix and (--plots) the Figure-6-style plot.
           Never executes model code, so it is safe on the login node.

Scoring happens inside `run`, per problem, through human_eval.execution's
check_correctness -- the exact function evaluate_functional_correctness calls,
with its default 3 s timeout -- so model-written code only ever executes on a
compute node. `humaneval_sweeps.sh rescore <run_key>` re-scores a merged sample
file with the official CLI (also as a SLURM job) as a cross-check.

Parts are JSONL, parts/<run_key>__<units file>_u<NNNN>.jsonl; every record ends
with "row_ok": 1 and a torn line fails to parse and is dropped.

Usage (see humaneval_sweeps.sh):
    python scripts/mikv_humaneval.py plan  --group tier1 [--write-units U.json --submit-info U.env]
    python scripts/mikv_humaneval.py run   --units-file U.json --unit "$SLURM_ARRAY_TASK_ID"
    python scripts/mikv_humaneval.py merge [--plots]
"""

import argparse
import csv
import datetime
import glob
import json
import os
import statistics
import sys
from dataclasses import asdict

from mikv_config import DEVICE, MODEL_NAME
from mikv_gsm8k import (
    TIME_CAP_MIN,
    TIME_MARGIN,
    _describe_run,
    _hms,
    _split,
    run_key,
)
from mikv_gsm8k_report import mcnemar_exact_p, wilson_interval, write_csv_atomic
from mikv_humaneval_bench import (
    MAX_NEW_TOKENS,
    STOP_STRINGS,
    budget_k_preview,
    default_data_path,
    evaluate_problem,
    file_sha256,
    generation_settings,
    load_problems,
    prompt_lengths,
    release_cuda_cache,
    select_problems,
)
from mikv_humaneval_sweeps import HUMANEVAL_SWEEPS, expand_group
from mikv_sweep import SweepPoint, config_row_fields, enumerate_sweep_points, format_sweep_plan

DEFAULT_RESULTS_ROOT = "results/humaneval"
# Until the probe measures them. GSM8K measured ~0.14 s/token for MiKV and ~0.07
# for the baseline; HumanEval completions run ~100-200 tokens, up to 512 when the
# model never emits a stop marker, plus up to 3 s of test execution.
DEFAULT_SEC_PER_PROBLEM = {"mikv": 35.0, "baseline": 18.0}
# Wider than GSM8K's 20 / 45 min: a task may first stage 13 GB onto node-local
# disk, and probe 27665 lost its whole 45 min to a slow NFS model load.
TIME_OVERHEAD_MIN = 30
TIME_FLOOR_MIN = 60
SEED = 0
CONFIG_FIELDS = tuple(config_row_fields(SweepPoint()).keys())


# ---- runs ----


def sweep_runs(spec: dict) -> tuple[list[tuple], list[SweepPoint], list]:
    runs, points, skipped = [], [], []
    if spec.get("baseline"):
        runs.append(("baseline", None, None))
    axes_list = spec.get("axes") or ()
    if isinstance(axes_list, dict):
        axes_list = (axes_list,)
    for axes in axes_list:
        got, dropped = enumerate_sweep_points(axes, sweep_mode="grid")
        points += [p for p in got if p not in points]
        skipped += dropped
    runs += [("mikv", p, r) for p in points for r in spec.get("budget_ratios", ())]
    return runs, points, skipped


# ---- part files ----


class PartWriter:
    """One JSON line per problem, flushed and fsync'd."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torn = False
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
        self._file = open(path, "a")
        if torn:
            self._file.write("\n")

    def write(self, row: dict) -> None:
        self._file.write(json.dumps({**row, "row_ok": 1}, ensure_ascii=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def part_files(parts_dir: str, key: str | None = None) -> list[str]:
    return sorted(glob.glob(os.path.join(parts_dir, f"{key}__*.jsonl" if key else "*.jsonl")))


def read_part(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("row_ok") == 1:
                rows.append(row)
    return rows


def done_problems(parts_dir: str, key: str) -> set[int]:
    return {r["problem_idx"] for p in part_files(parts_dir, key) for r in read_part(p)}


def load_all_rows(parts_dir: str) -> tuple[dict[str, dict[int, dict]], int]:
    """{run_key: {problem_idx: row}} and the number of duplicate rows whose
    completion differs from the first copy (GPU nondeterminism)."""
    runs: dict[str, dict[int, dict]] = {}
    conflicts = 0
    for path in part_files(parts_dir):
        for row in read_part(path):
            per_run = runs.setdefault(row["run_key"], {})
            first = per_run.get(row["problem_idx"])
            if first is None:
                per_run[row["problem_idx"]] = row
            elif first["completion"] != row["completion"]:
                conflicts += 1
    return runs, conflicts


def measured_seconds(parts_dir: str) -> dict[str, float]:
    """Slowest run's mean s/problem per method (generation + execution)."""
    per_run: dict[tuple[str, str], list[float]] = {}
    for path in part_files(parts_dir):
        for row in read_part(path):
            total = float(row["seconds"]) + float(row.get("exec_seconds", 0.0))
            per_run.setdefault((row["method"], row["run_key"]), []).append(total)
    slowest: dict[str, float] = {}
    for (method, _), values in per_run.items():
        slowest[method] = max(slowest.get(method, 0.0), statistics.fmean(values))
    return slowest


# ---- plan ----


def build_plan(group: str, data_path: str, parts_dir: str, problems_per_task: int,
               model_name: str) -> dict:
    sweeps = expand_group(group)
    problems = load_problems(data_path)
    settings = generation_settings()
    runs: dict[str, dict] = {}
    per_sweep: dict[str, dict] = {}
    for name in sweeps:
        spec = HUMANEVAL_SWEEPS[name]
        chosen = select_problems(len(problems), spec.get("problem_limit"))
        run_list, points, skipped = sweep_runs(spec)
        keys = []
        for method, point, ratio in run_list:
            key = run_key(method, point, ratio, settings, model_name)
            keys.append(key)
            entry = runs.setdefault(key, dict(run_key=key, method=method, point=point,
                                              ratio=ratio, problems=set(), sweeps=[]))
            entry["problems"].update(chosen)
            entry["sweeps"].append(name)
        per_sweep[name] = dict(spec=spec, problems=chosen, points=points, skipped=skipped, keys=keys)

    units = []
    for entry in runs.values():
        done = done_problems(parts_dir, entry["run_key"])
        entry["done"] = done & entry["problems"]
        chunks = _split(sorted(entry["problems"] - done), problems_per_task)
        for shard, chunk in enumerate(chunks):
            units.append(dict(
                run_key=entry["run_key"], method=entry["method"],
                point=asdict(entry["point"]) if entry["point"] is not None else None,
                ratio=entry["ratio"], problems=chunk, shard=shard, shards=len(chunks),
                sweeps=entry["sweeps"],
            ))
    return dict(
        group=group, sweeps=list(sweeps),
        created=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        data=dict(path=data_path, sha256=file_sha256(data_path), n_total=len(problems)),
        settings=settings, model_name=model_name, problems_per_task=problems_per_task,
        units=units, _runs=runs, _per_sweep=per_sweep, _problems=problems,
    )


def _k_preview(plan: dict) -> list[str]:
    """t_p over the problems and, per (sweep, point window, r), how many problems
    run as pure recency (k_H = 0). Tokenizer only; skipped if not loadable."""
    try:
        from mikv_runtime import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan["model_name"])
    except Exception as exc:  # noqa: BLE001 -- a preview must never fail a plan
        return [f"[k] preview skipped (tokenizer not loadable here: {type(exc).__name__}: {exc})"]
    all_idx = list(range(plan["data"]["n_total"]))
    t_all = prompt_lengths(tokenizer, plan["_problems"], all_idx)
    lines = [f"[k] t_p mean {statistics.fmean(t_all):.0f} (min {min(t_all)}, median "
             f"{statistics.median(t_all):.0f}, max {max(t_all)}) over {len(t_all)} problems"]
    for name, info in plan["_per_sweep"].items():
        t_ps = [t_all[i] for i in info["problems"]]
        seen = set()
        for point in info["points"]:
            window = f"w={point.window_tokens}" if point.window_tokens is not None else f"w={point.window_ratio}k"
            if window in seen:
                continue
            seen.add(window)
            for ratio in info["spec"].get("budget_ratios", ()):
                lo, hi, recency = budget_k_preview(t_ps, ratio, point)
                lines.append(f"[k] {name} {window} r={ratio}: k {lo}-{hi}; "
                             f"{recency}/{len(t_ps)} problems pure recency (k_H = 0)")
    return lines


def _seconds_per_problem(parts_dir: str, overrides: dict) -> dict[str, tuple[float, str]]:
    measured = measured_seconds(parts_dir)
    out = {}
    for method, default in DEFAULT_SEC_PER_PROBLEM.items():
        if overrides.get(method) is not None:
            out[method] = (overrides[method], "given")
        elif method in measured:
            out[method] = (measured[method], "measured, slowest run so far")
        else:
            out[method] = (default, "default guess -- submit `probe` to measure")
    return out


def cmd_plan(args) -> None:
    parts_dir = os.path.join(args.results_root, "parts")
    plan = build_plan(args.group, args.data, parts_dir, args.problems_per_task, args.model)
    runs, per_sweep = plan["_runs"], plan["_per_sweep"]

    print(f"=== HumanEval plan: {args.group} -> {', '.join(plan['sweeps'])} ===")
    print(f"data     : {args.data} ({plan['data']['n_total']} problems, sha256 {plan['data']['sha256'][:16]}...)")
    print(f"decoding : greedy, verbatim prompt, <= {MAX_NEW_TOKENS} new tokens, stop at {list(STOP_STRINGS)}")
    print(f"model    : {plan['model_name']}")
    for name, info in per_sweep.items():
        n = len(info["problems"])
        done_runs = sum(len(runs[k]["done"] & set(info["problems"])) == n for k in info["keys"])
        print(f"--- {name}: {n} problems; {len(info['keys'])} runs, {done_runs} complete")
        if info["spec"].get("baseline"):
            print("[plan]   + uncompressed baseline")
        if info["points"] or info["skipped"]:
            print(format_sweep_plan(info["points"], tuple(info["spec"].get("budget_ratios", ())),
                                    info["skipped"], num_samples=n))
    if not args.no_k_preview:
        print("\n".join(_k_preview(plan)))

    units = plan["units"]
    print(f"=== {len(runs)} distinct runs; {sum(not (e['problems'] - e['done']) for e in runs.values())} "
          f"complete; {len(units)} units pending ({sum(len(u['problems']) for u in units)} problem-generations) ===")
    for i, u in enumerate(units):
        entry = runs[u["run_key"]]
        print(f"  u{i:03d}  {u['run_key']}  {len(u['problems']):>4} problems "
              f"({len(entry['done'])}/{len(entry['problems'])} done)  [{','.join(u['sweeps'])}]  {_describe_run(u)}")

    sec = _seconds_per_problem(parts_dir, {"mikv": args.sec_per_problem,
                                           "baseline": args.sec_per_problem_baseline})
    for method, (s, source) in sec.items():
        print(f"estimate : {s:.1f} s/problem for {method} ({source})")
    unit_minutes = [len(u["problems"]) * sec[u["method"]][0] / 60 for u in units]
    gpu_hours = sum(unit_minutes) / 60
    longest = max(unit_minutes, default=0.0)
    wanted = longest * TIME_MARGIN + TIME_OVERHEAD_MIN
    time_limit = _hms(min(max(wanted, TIME_FLOOR_MIN), TIME_CAP_MIN))
    print(f"estimate : ~{gpu_hours:.1f} GPU-hours pending; longest unit ~{longest / 60:.1f} h "
          f"-> --time {time_limit} per array task")

    if args.write_units:
        os.makedirs(os.path.dirname(args.write_units) or ".", exist_ok=True)
        with open(args.write_units, "w") as f:
            json.dump({k: v for k, v in plan.items() if not k.startswith("_")}, f, indent=1)
        print(f"units    : {len(units)} written to {args.write_units}")
    if args.submit_info:
        with open(args.submit_info, "w") as f:
            f.write(f"NUNITS={len(units)}\nTIME={time_limit}\nGPU_HOURS={gpu_hours:.1f}\n")


# ---- run ----


def cmd_run(args) -> None:
    with open(args.units_file) as f:
        doc = json.load(f)
    if not 0 <= args.unit < len(doc["units"]):
        raise SystemExit(f"unit {args.unit} out of range: {args.units_file} has {len(doc['units'])}")
    unit = doc["units"][args.unit]
    if doc["settings"] != generation_settings():
        raise SystemExit(f"generation settings changed since this units file was planned "
                         f"({doc['settings']} -> {generation_settings()}); re-plan and resubmit")
    parts_dir = os.path.join(args.results_root, "parts")
    stem = os.path.splitext(os.path.basename(args.units_file))[0]
    part_path = os.path.join(parts_dir, f"{unit['run_key']}__{stem}_u{args.unit:04d}.jsonl")

    print(f"=== HumanEval unit {args.unit}/{len(doc['units']) - 1} of {args.units_file} ===", flush=True)
    print(f"run      : {unit['run_key']}  {_describe_run(unit)}", flush=True)
    print(f"sweeps   : {', '.join(unit['sweeps'])}", flush=True)
    done = done_problems(parts_dir, unit["run_key"])
    todo = [i for i in unit["problems"] if i not in done]
    print(f"problems : {len(unit['problems'])} in unit, {len(unit['problems']) - len(todo)} already done, "
          f"{len(todo)} to run -> {part_path}", flush=True)
    if not todo:
        print("=== nothing to do: unit already complete ===", flush=True)
        return
    if DEVICE != "cuda" and not args.allow_cpu:
        raise SystemExit(f"CUDA is not available on {os.uname().nodename}; refusing to run a 7B model on "
                         f"the CPU. Resubmit with EXCLUDE=<node>. Nothing was written for this unit.")

    data_path = args.data or doc["data"]["path"]
    if file_sha256(data_path) != doc["data"]["sha256"]:
        raise SystemExit(f"{data_path} is not the file this plan was built from (sha256 differs)")
    problems = load_problems(data_path)

    import transformers
    from mikv_policy import load_model
    from mikv_runtime import torch

    print(f"[humaneval] loading model at {datetime.datetime.now():%H:%M:%S}", flush=True)
    model, tokenizer = load_model(doc["model_name"])
    print(f"[humaneval] model on {DEVICE} at {datetime.datetime.now():%H:%M:%S}", flush=True)

    # Determinism: fixed seed, batch 1, and deterministic kernels where torch has
    # them (warn_only: ops without a deterministic CUDA kernel warn rather than
    # abort; CUBLAS_WORKSPACE_CONFIG is set by jobs_humaneval.sh before CUDA init).
    # Enabled only after loading: deterministic mode also fills every fresh
    # allocation, which is pointless work over 13 GB of weights about to be
    # overwritten.
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    point = SweepPoint(**unit["point"]) if unit["point"] is not None else None
    fixed = dict(
        run_key=unit["run_key"],
        job=(f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}"
             if "SLURM_ARRAY_JOB_ID" in os.environ else "interactive"),
        gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        model_name=doc["model_name"],
        commit=os.environ.get("MIKV_COMMIT", ""),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        seed=SEED,
        max_new_tokens=MAX_NEW_TOKENS,
        prompt_version=doc["settings"]["prompt_version"],
        **(config_row_fields(point) if point is not None else {}),
    )

    writer = PartWriter(part_path)
    passed = elapsed = 0.0
    try:
        for n, idx in enumerate(todo, 1):
            started = datetime.datetime.now()
            row = evaluate_problem(model, tokenizer, problems[idx], unit["method"], point, unit["ratio"])
            total = (datetime.datetime.now() - started).total_seconds()
            row.update(fixed, exec_seconds=round(total - row["seconds"], 2),
                       timestamp=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
            writer.write(row)
            release_cuda_cache()
            passed += row["passed"]
            elapsed += total
            print(f"[humaneval u{args.unit} {n}/{len(todo)}] {row['task_id']} "
                  f"{'PASS' if row['passed'] else 'FAIL'} ({row['result'][:40]}) t_p={row['t_p']} "
                  f"k={row['k']} k_h={row['k_h']} gen={row['gen_tokens']} ({row['stop_reason']}) "
                  f"{total:.1f}s | pass@1 {100 * passed / n:.1f}% avg {elapsed / n:.1f}s/problem", flush=True)
            if n <= 3:
                print(f"    completion[:200] = {row['completion'][:200]!r}", flush=True)
    finally:
        writer.close()
    print(f"=== unit done: {int(passed)}/{len(todo)} passed this task, {elapsed / len(todo):.1f} s/problem ===",
          flush=True)


# ---- merge ----

SUMMARY_FIELDS = (
    "sweep", "method", "run_key", "describe", *CONFIG_FIELDS, "ratio",
    "n_problems", "n_done", "complete",
    "pass_at_1", "n_passed", "pass_ci95_lo", "pass_ci95_hi",
    "baseline_pass_at_1", "n_paired", "delta_pass_at_1", "n_lost", "n_gained", "mcnemar_p",
    "t_p", "k", "w_eff", "k_h", "pure_recency_pct", "gen_tokens", "hit_max_new_tokens_pct",
    "compression_pct", "compression_pct_occupied", "avg_kv_bits_occupied", "sec_per_problem",
    "model_name", "commit", "torch_version", "transformers_version",
)
PROBLEM_FIELDS = (
    "run_key", "method", *CONFIG_FIELDS, "ratio", "task_id", "problem_idx", "passed", "result",
    "t_p", "k", "w_eff", "k_h", "gen_tokens", "stop_reason", "avg_seq_len",
    "compression_pct", "compression_pct_occupied", "avg_kv_bits_occupied",
    "seconds", "exec_seconds", "job", "gpu", "model_name", "commit", "timestamp", "completion",
)


def _mean(rows: list[dict], field: str):
    values = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
    return round(statistics.fmean(values), 4) if values else ""


def summarize_run(sweep: str, key: str, rows: dict[int, dict], chosen: list[int],
                  baseline: dict[int, dict]) -> dict | None:
    mine = [rows[i] for i in chosen if i in rows]
    if not mine:
        return None
    first = mine[0]
    n_passed = sum(int(r["passed"]) for r in mine)
    lo, hi = wilson_interval(n_passed, len(mine))
    paired = [i for i in chosen if i in rows and i in baseline]
    lost = sum(int(baseline[i]["passed"]) and not int(rows[i]["passed"]) for i in paired)
    gained = sum(int(rows[i]["passed"]) and not int(baseline[i]["passed"]) for i in paired)
    base_pass = sum(int(baseline[i]["passed"]) for i in paired)
    run_pass = sum(int(rows[i]["passed"]) for i in paired)
    return dict(
        sweep=sweep, method=first["method"], run_key=key,
        describe=_describe_row(first),
        **{f: first.get(f, "") for f in CONFIG_FIELDS}, ratio=first.get("ratio", ""),
        n_problems=len(chosen), n_done=len(mine), complete=len(mine) == len(chosen),
        pass_at_1=round(n_passed / len(mine), 4), n_passed=n_passed,
        pass_ci95_lo=round(lo, 4), pass_ci95_hi=round(hi, 4),
        baseline_pass_at_1=round(base_pass / len(paired), 4) if paired else "",
        n_paired=len(paired),
        delta_pass_at_1=round((run_pass - base_pass) / len(paired), 4) if paired else "",
        n_lost=lost, n_gained=gained,
        mcnemar_p=round(mcnemar_exact_p(lost, gained), 4) if paired else "",
        t_p=_mean(mine, "t_p"), k=_mean(mine, "k"), w_eff=_mean(mine, "w_eff"), k_h=_mean(mine, "k_h"),
        pure_recency_pct=(round(100 * sum(r["k_h"] == 0 for r in mine) / len(mine), 1)
                          if first["method"] == "mikv" else ""),
        gen_tokens=_mean(mine, "gen_tokens"),
        hit_max_new_tokens_pct=round(100 * sum(r["stop_reason"] == "max_new_tokens" for r in mine) / len(mine), 1),
        compression_pct=_mean(mine, "compression_pct"),
        compression_pct_occupied=_mean(mine, "compression_pct_occupied"),
        avg_kv_bits_occupied=_mean(mine, "avg_kv_bits_occupied"),
        sec_per_problem=round(statistics.fmean(float(r["seconds"]) + float(r.get("exec_seconds", 0))
                                               for r in mine), 2),
        model_name=first.get("model_name", ""), commit=first.get("commit", ""),
        torch_version=first.get("torch_version", ""), transformers_version=first.get("transformers_version", ""),
    )


def _describe_row(row: dict) -> str:
    if row["method"] == "baseline":
        return "uncompressed baseline"
    high = "fp16" if row["high_precision_native"] else f"int{row['high_bits']}"
    window = f"w={row['window_tokens']}" if row["window_tokens"] != "" else f"w={row['window_ratio']}k"
    score = (f"Q{row['score_length_bits']}.{row['score_frac_bits']}" if row["score_scheme"] == "fixed"
             else row["score_scheme"])
    return (f"{row['scheme']} {window} HIGH {high} LOW int{row['low_bits']} "
            f"{score}/{row['score_decay_scheme']}")


def _plot(summaries: list[dict], baseline_pass: float | None, path: str) -> None:
    import mikv_runtime  # noqa: F401 -- torch/transformers before matplotlib (see mikv_runtime)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[float, float]]] = {}
    for s in summaries:
        if s["method"] != "mikv" or not s["complete"] or s["compression_pct_occupied"] == "":
            continue
        series.setdefault(s["describe"], []).append((float(s["compression_pct_occupied"]),
                                                     100 * float(s["pass_at_1"])))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, pts in sorted(series.items()):
        pts = sorted(set(pts))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
    if baseline_pass is not None:
        ax.axhline(100 * baseline_pass, linestyle="--", color="black", label="Full cache")
    ax.set_xlabel("KV cache size (% of fp16, occupied length)")
    ax.set_ylabel("HumanEval pass@1 (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_merge(args) -> None:
    root = args.results_root
    parts_dir = os.path.join(root, "parts")
    runs, conflicts = load_all_rows(parts_dir)
    if not runs:
        raise SystemExit(f"no per-problem rows under {parts_dir} yet")
    problems = load_problems(args.data)
    n_total = len(problems)
    settings = generation_settings()
    baseline_key = run_key("baseline", None, None, settings, args.model)
    baseline = runs.get(baseline_key, {})

    submitted = set()
    for path in glob.glob(os.path.join(root, "plans", "*.json")):
        with open(path) as f:
            submitted.update(json.load(f).get("sweeps", []))

    summaries, covered = [], set()
    for name, spec in HUMANEVAL_SWEEPS.items():
        if submitted and name not in submitted:
            continue
        chosen = select_problems(n_total, spec.get("problem_limit"))
        for method, point, ratio in sweep_runs(spec)[0]:
            key = run_key(method, point, ratio, settings, args.model)
            covered.add(key)
            s = summarize_run(name, key, runs.get(key, {}), chosen, baseline)
            if s is not None:
                summaries.append(s)
    for key, rows in runs.items():
        if key not in covered:
            summaries.append(summarize_run("-", key, rows, sorted(rows), baseline))

    order = sorted(runs, key=lambda k: (k != baseline_key, k))
    problem_rows = [dict(r, completion=json.dumps(r["completion"], ensure_ascii=False))
                    for key in order for _, r in sorted(runs[key].items())]
    write_csv_atomic(os.path.join(root, "humaneval_problems.csv"), PROBLEM_FIELDS, problem_rows)
    write_csv_atomic(os.path.join(root, "humaneval_summary.csv"), SUMMARY_FIELDS, summaries)

    # Wide pass matrix: one row per task, one column per run -- which problems flip.
    matrix_path = os.path.join(root, "humaneval_passmatrix.csv")
    with open(matrix_path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", *order])
        w.writerow(["# describe", *[_describe_row(next(iter(runs[k].values()))) +
                                    (f" r={next(iter(runs[k].values()))['ratio']}"
                                     if k != baseline_key else "") for k in order]])
        for idx in range(n_total):
            w.writerow([f"HumanEval/{idx}", *[runs[k][idx]["passed"] if idx in runs[k] else "" for k in order]])
    os.replace(matrix_path + ".tmp", matrix_path)

    # Official-format samples + meta sidecars, for evaluate_functional_correctness.
    samples_dir = os.path.join(root, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    by_key = {s["run_key"]: s for s in summaries}
    for key, rows in runs.items():
        with open(os.path.join(samples_dir, f"{key}.jsonl"), "w") as f:
            for _, r in sorted(rows.items()):
                f.write(json.dumps({"task_id": r["task_id"], "completion": r["completion"]}) + "\n")
        s = by_key.get(key, {})
        first = next(iter(rows.values()))
        meta = dict(
            run_key=key, describe=_describe_row(first), ratio=first.get("ratio", ""),
            model=first.get("model_name"), method=first["method"],
            config={f: first.get(f, "") for f in CONFIG_FIELDS},
            settings=settings, seed=first.get("seed"), commit=first.get("commit"),
            torch_version=first.get("torch_version"), transformers_version=first.get("transformers_version"),
            n_done=len(rows), pass_at_1=s.get("pass_at_1"),
            wall_clock_s=round(sum(float(r["seconds"]) + float(r.get("exec_seconds", 0)) for r in rows.values()), 1),
            avg_kv_bits_occupied=s.get("avg_kv_bits_occupied"),
            compression_pct_occupied=s.get("compression_pct_occupied"),
        )
        with open(os.path.join(samples_dir, f"{key}.meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    print(f"[merge] {len(problem_rows)} problem rows over {len(runs)} runs -> {root}/humaneval_problems.csv")
    print(f"[merge] {len(summaries)} summary rows -> {root}/humaneval_summary.csv; "
          f"pass matrix -> {matrix_path}; samples -> {samples_dir}/")
    print(f"{'sweep':<9} {'run_key':<12} {'r':>4} {'pass@1':>7} {'lost':>4} {'gain':>4} {'p':>6} "
          f"{'recency%':>8} {'kv%occ':>6}  config")
    for s in summaries:
        progress = "" if s["complete"] else f"  ({s['n_done']}/{s['n_problems']})"
        print(f"{s['sweep']:<9} {s['run_key']:<12} {str(s['ratio']):>4} "
              f"{100 * s['pass_at_1']:>6.1f}% {s['n_lost']:>4} {s['n_gained']:>4} {str(s['mcnemar_p']):>6} "
              f"{str(s['pure_recency_pct']):>8} {str(s['compression_pct_occupied'])[:6]:>6}  "
              f"{s['describe']}{progress}")
    if not baseline:
        print("[merge] WARNING: no baseline rows yet -- submit `baseline`")
    if conflicts:
        print(f"[merge] WARNING: {conflicts} duplicated problems produced different completions "
              f"(nondeterminism); the first copy was kept")
    if args.plots:
        base = (sum(int(r["passed"]) for r in baseline.values()) / len(baseline)) if baseline else None
        _plot(summaries, base, os.path.join(root, "humaneval_pass_vs_kv.png"))
        print(f"[merge] plot -> {root}/humaneval_pass_vs_kv.png")


# ---- CLI ----


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mikv_humaneval.py",
                                     description="HumanEval sweeps of the MiKV KV-cache policy, as SLURM job arrays.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    plan.add_argument("--group", required=True, help="tier1, all, a sweep (baseline, probe, sanity, rtn, H1..H4) or a mix")
    plan.add_argument("--data", default=default_data_path())
    plan.add_argument("--model", default=os.environ.get("HUMANEVAL_MODEL", MODEL_NAME),
                      help="must already be in hf_cache (jobs run offline)")
    plan.add_argument("--problems-per-task", type=int, default=164)
    plan.add_argument("--sec-per-problem", type=float, default=None)
    plan.add_argument("--sec-per-problem-baseline", type=float, default=None)
    plan.add_argument("--write-units", default=None)
    plan.add_argument("--submit-info", default=None)
    plan.add_argument("--no-k-preview", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--units-file", required=True)
    run.add_argument("--unit", type=int, required=True)
    run.add_argument("--data", default=None)
    run.add_argument("--allow-cpu", action="store_true")

    merge = sub.add_parser("merge", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    merge.add_argument("--data", default=default_data_path())
    merge.add_argument("--model", default=os.environ.get("HUMANEVAL_MODEL", MODEL_NAME))
    merge.add_argument("--plots", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    {"plan": cmd_plan, "run": cmd_run, "merge": cmd_merge}[args.command](args)
    sys.stdout.flush()
