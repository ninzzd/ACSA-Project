"""
GSM8K sweeps of the MiKV policy, run as SLURM job arrays: the GSM8K counterpart
of mikv.py. Three subcommands, one per stage:

    plan   Expand a group of sweeps (mikv_gsm8k_sweeps) into *units* of work and
           print the plan -- no model, no GPU; runs on a login node. With
           --write-units it also freezes the pending units into a JSON file that
           one job array then executes, one unit per array task.
    run    Execute one unit (inside an array task): load the model, generate
           and score each of the unit's problems, append each row to a part CSV.
    merge  Rebuild every summary from the part CSVs -- see mikv_gsm8k_report.

--- units ---

A *run* is one (method, configuration, budget ratio): a MiKV point, or the
uncompressed baseline. Its identity is a hash of the configuration and the
generation settings (`run_key`) -- NOT of the sweep that asked for it -- so
sweeps sharing a configuration share its results, and a run already partly
measured by any earlier job resumes instead of restarting.

A *unit* is a run's pending problems, or a slice of them: runs are split so no
unit exceeds --problems-per-task, which keeps every array task a few hours long.
Short tasks schedule sooner (backfill), spread across every free GPU on the
partition, and lose little to a failure -- and resuming makes a timed-out task
cost nothing but a resubmit.

Only *pending* problems become units, so resubmitting a group after a partial
run submits exactly what is left. Plans are frozen at submission (the units
file), so editing mikv_gsm8k_sweeps while an array is queued cannot change what
its tasks run.

--- resuming ---

Each task writes its own part file, parts/<run_key>__<units file>_u<NNNN>.csv,
flushing after every problem. Before loading the model it reads every part file
of its run_key and skips problems already present, from any task of any
submission. A task killed by the walltime limit therefore loses at most one
problem, and `gsm8k_sweeps.sh submit <group>` again picks up the remainder.

Usage (from the repo root; see gsm8k_sweeps.sh for the job-level wrapper):
    python scripts/mikv_gsm8k.py plan  --group tier1
    python scripts/mikv_gsm8k.py plan  --group tier1 --write-units U.json --submit-info U.env
    python scripts/mikv_gsm8k.py run   --units-file U.json --unit "$SLURM_ARRAY_TASK_ID"
    python scripts/mikv_gsm8k.py merge --line-retrieval-csv 'results/followup/*/legacy/sweep.csv'
"""

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict

from mikv_config import DEVICE, MODEL_NAME
from mikv_gsm8k_bench import (
    DEFAULT_DATA_PATH,
    MAX_NEW_TOKENS,
    MAX_TOKENS,
    N_SHOT,
    PROMPT_VERSION,
    STOP_STRINGS,
    evaluate_problem,
    file_sha256,
    generation_settings,
    load_problems,
    prompt_lengths,
    release_cuda_cache,
    select_problems,
)
from mikv_gsm8k_report import (
    PROBLEM_CSV_FIELDS,
    SUMMARY_CSV_FIELDS,
    PartWriter,
    attach_line_retrieval,
    done_problems,
    load_all_rows,
    load_line_retrieval,
    measured_seconds,
    summarize_run,
    write_csv_atomic,
)
from mikv_gsm8k_sweeps import GSM8K_SWEEPS, expand_group
from mikv_report import format_results_table
from mikv_sweep import SweepPoint, config_row_fields, enumerate_sweep_points, format_sweep_plan

DEFAULT_RESULTS_ROOT = "results/gsm8k"
# Until the probe has measured them. Rough: the MiKV path decodes ~100-250 tokens
# per problem with the policy's per-step bookkeeping on top of eager attention.
DEFAULT_SEC_PER_PROBLEM = {"mikv": 30.0, "baseline": 15.0}
# --time = the largest unit's estimate x margin + load overhead, capped under 24 h.
TIME_MARGIN = 1.5
TIME_OVERHEAD_MIN = 20
TIME_FLOOR_MIN = 45
TIME_CAP_MIN = 23 * 60 + 55


# ---- runs ----


def run_key(method: str, point: SweepPoint | None, ratio: float | None,
            settings: dict, model_name: str) -> str:
    """A run's identity: configuration + generation settings + model, hashed.
    Deliberately excludes the sweep and the problem set -- see the module
    docstring -- so two sweeps asking for the same run share one result."""
    ident = dict(
        method=method,
        point=asdict(point) if point is not None else None,
        ratio=None if ratio is None else float(ratio),
        settings=settings,
        model=model_name,
    )
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def sweep_runs(spec: dict) -> tuple[list[tuple], list[SweepPoint], list]:
    """(runs, points, skipped) for one sweep spec. A run is (method, point, ratio);
    the points are canonicalized and deduped by the shared enumerator, exactly as
    a Line Retrieval grid sweep does."""
    runs, points, skipped = [], [], []
    if spec.get("baseline"):
        runs.append(("baseline", None, None))
    if spec.get("axes"):
        points, skipped = enumerate_sweep_points(spec["axes"], sweep_mode="grid")
        runs += [("mikv", p, r) for p in points for r in spec["budget_ratios"]]
    return runs, points, skipped


def sweep_problems(spec: dict, n_total: int) -> list[int]:
    return select_problems(n_total, spec["num_problems"], spec["problem_seed"],
                           spec.get("problem_limit"))


def _split(items: list[int], max_per_chunk: int) -> list[list[int]]:
    """Balanced contiguous chunks of at most `max_per_chunk` (sizes differ by <= 1)."""
    if not items:
        return []
    n = math.ceil(len(items) / max_per_chunk)
    base, extra = divmod(len(items), n)
    chunks, start = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


# ---- plan ----


def build_plan(group: str, data_path: str, parts_dir: str, problems_per_task: int) -> dict:
    sweeps = expand_group(group)
    problems = load_problems(data_path)
    n_total = len(problems)
    settings = generation_settings()

    runs: dict[str, dict] = {}
    per_sweep: dict[str, dict] = {}
    for name in sweeps:
        spec = GSM8K_SWEEPS[name]
        chosen = sweep_problems(spec, n_total)
        run_list, points, skipped = sweep_runs(spec)
        keys = []
        for method, point, ratio in run_list:
            key = run_key(method, point, ratio, settings, MODEL_NAME)
            keys.append(key)
            entry = runs.setdefault(key, dict(run_key=key, method=method, point=point,
                                              ratio=ratio, problems=set(), sweeps=[]))
            entry["problems"].update(chosen)
            entry["sweeps"].append(name)
        per_sweep[name] = dict(spec=spec, problems=chosen, points=points,
                               skipped=skipped, keys=keys)

    units = []
    for entry in runs.values():
        done = done_problems(parts_dir, entry["run_key"])
        entry["done"] = done & entry["problems"]
        pending = sorted(entry["problems"] - done)
        chunks = _split(pending, problems_per_task)
        for shard, chunk in enumerate(chunks):
            units.append(dict(
                run_key=entry["run_key"],
                method=entry["method"],
                point=asdict(entry["point"]) if entry["point"] is not None else None,
                ratio=entry["ratio"],
                problems=chunk,
                shard=shard,
                shards=len(chunks),
                sweeps=entry["sweeps"],
            ))
    return dict(
        group=group,
        sweeps=list(sweeps),
        created=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        data=dict(path=data_path, sha256=file_sha256(data_path), n_total=n_total),
        settings=settings,
        model_name=MODEL_NAME,
        problems_per_task=problems_per_task,
        units=units,
        _runs=runs,
        _per_sweep=per_sweep,
        _problems=problems,
    )


def _describe_run(entry_or_unit: dict) -> str:
    """One compact line per run: the fields that differ between GSM8K sweeps."""
    if entry_or_unit["method"] == "baseline":
        return "uncompressed baseline"
    point = entry_or_unit["point"]
    point = SweepPoint(**point) if isinstance(point, dict) else point
    window = f"w={point.window_tokens}" if point.window_tokens is not None else f"w={point.window_ratio}k"
    return (f"{point.budget_mode} {point.balance_scheme} {window} HIGH {point.high_tier} "
            f"LOW int{point.low_bits} {point.score_tag} r={entry_or_unit['ratio']}")


def _k_preview(plan: dict) -> list[str]:
    """Per sweep: t_p over its problems and, per (ratio, window), the k it implies
    and how many problems degenerate to pure recency (k <= w). Needs only the
    tokenizer, so it runs on a login node; skipped if the tokenizer is not in the
    local HF cache."""
    try:
        from mikv_runtime import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan["model_name"])
    except Exception as exc:  # noqa: BLE001 -- a preview must never fail a plan
        return [f"[k] preview skipped (tokenizer not loadable here: {type(exc).__name__}: {exc})"]
    lines = []
    cache: dict[int, int] = {}
    for name, info in plan["_per_sweep"].items():
        todo = [i for i in info["problems"] if i not in cache]
        for i, n in zip(todo, prompt_lengths(tokenizer, plan["_problems"], todo)):
            cache[i] = n
        t_ps = [cache[i] for i in info["problems"]]
        lines.append(
            f"[k] {name}: t_p mean {sum(t_ps) / len(t_ps):.0f} (min {min(t_ps)}, max {max(t_ps)}) "
            f"over {len(t_ps)} problems"
        )
        if not info["points"]:
            continue
        windows = sorted({p.window_tokens for p in info["points"]}, key=lambda w: (w is None, w))
        for ratio in info["spec"]["budget_ratios"]:
            ks = [max(1, math.floor(ratio * t)) for t in t_ps]
            parts = []
            for w in windows:
                if w is None:
                    continue
                recency = sum(k <= w for k in ks)
                parts.append(f"w={w}: " + (
                    "pure recency on every problem" if recency == len(ks)
                    else f"{recency} pure-recency problems" if recency else "k_H > 0 everywhere"
                ))
            lines.append(
                f"[k]   r={ratio}: k {min(ks)}-{max(ks)} (mean {sum(ks) / len(ks):.0f})"
                + (f"; {'; '.join(parts)}" if parts else "")
            )
    return lines


def _seconds_per_problem(parts_dir: str, overrides: dict[str, float | None]) -> dict[str, tuple[float, str]]:
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


def _hms(minutes: float) -> str:
    minutes = int(math.ceil(minutes / 5) * 5)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def cmd_plan(args) -> None:
    parts_dir = os.path.join(args.results_root, "parts")
    plan = build_plan(args.group, args.data, parts_dir, args.problems_per_task)
    runs, per_sweep = plan["_runs"], plan["_per_sweep"]

    print(f"=== GSM8K plan: {args.group} -> {', '.join(plan['sweeps'])} ===")
    print(f"data     : {args.data} ({plan['data']['n_total']} problems, "
          f"sha256 {plan['data']['sha256'][:16]}...)")
    print(f"prompt   : {N_SHOT}-shot CoT ({PROMPT_VERSION}), <= {MAX_NEW_TOKENS} new tokens, "
          f"stop at {list(STOP_STRINGS)} or EOS; footprint sized at {MAX_TOKENS} tokens")
    print(f"model    : {plan['model_name']}")
    for name, info in per_sweep.items():
        spec = info["spec"]
        n = len(info["problems"])
        which = ("full test set" if not spec["num_problems"]
                 else f"{spec['num_problems']} problems, seed {spec['problem_seed']}"
                 + (f", first {spec['problem_limit']}" if spec.get("problem_limit") else ""))
        done_runs = sum(len(runs[k]["done"] & set(info["problems"])) == n for k in info["keys"])
        print(f"--- {name}: {n} problems ({which}); {len(info['keys'])} runs, {done_runs} complete")
        if spec.get("baseline"):
            print("[plan]   + uncompressed baseline")
        if info["points"] or info["skipped"]:
            print(format_sweep_plan(info["points"], tuple(spec["budget_ratios"]),
                                    info["skipped"], num_samples=n))
    if not args.no_k_preview:
        print("\n".join(_k_preview(plan)))

    saved = sum(len(e["sweeps"]) - 1 for e in runs.values())
    units = plan["units"]
    pending_problems = sum(len(u["problems"]) for u in units)
    print(f"=== {len(runs)} distinct runs ({saved} more requested by overlapping sweeps, run once); "
          f"{sum(not (e['problems'] - e['done']) for e in runs.values())} complete; "
          f"{len(units)} units pending ({pending_problems} problem-generations) ===")
    for i, u in enumerate(units):
        entry = runs[u["run_key"]]
        print(f"  u{i:03d}  {u['run_key']}  {len(u['problems']):>4} problems "
              f"(shard {u['shard'] + 1}/{u['shards']}, {len(entry['done'])}/{len(entry['problems'])} done)  "
              f"[{','.join(u['sweeps'])}]  {_describe_run(u)}")

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
    if wanted > TIME_CAP_MIN:
        print(f"WARNING  : the longest unit needs ~{wanted / 60:.1f} h with margin, over the 24 h "
              f"limit; lower --problems-per-task (it resumes either way, but will need resubmits)")

    if args.write_units:
        os.makedirs(os.path.dirname(args.write_units) or ".", exist_ok=True)
        doc = {k: v for k, v in plan.items() if not k.startswith("_")}
        with open(args.write_units, "w") as f:
            json.dump(doc, f, indent=1)
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
        raise SystemExit(
            "generation settings in mikv_gsm8k_bench changed since this units file was "
            f"planned ({doc['settings']} -> {generation_settings()}); re-plan and resubmit"
        )
    parts_dir = os.path.join(args.results_root, "parts")
    stem = os.path.splitext(os.path.basename(args.units_file))[0]
    part_path = os.path.join(parts_dir, f"{unit['run_key']}__{stem}_u{args.unit:04d}.csv")

    print(f"=== GSM8K unit {args.unit}/{len(doc['units']) - 1} of {args.units_file} ===", flush=True)
    print(f"run      : {unit['run_key']}  {_describe_run(unit)}", flush=True)
    print(f"sweeps   : {', '.join(unit['sweeps'])}", flush=True)
    # Checked before the model loads: a unit finished by an earlier or concurrent
    # task exits in seconds instead of spending a minute loading 13 GB for nothing.
    done = done_problems(parts_dir, unit["run_key"])
    todo = [i for i in unit["problems"] if i not in done]
    print(f"problems : {len(unit['problems'])} in unit, {len(unit['problems']) - len(todo)} "
          f"already done, {len(todo)} to run -> {part_path}", flush=True)
    if not todo:
        print("=== nothing to do: unit already complete ===", flush=True)
        return

    # mikv_config falls back to the CPU when CUDA does not initialize (a node with
    # a broken driver, say). For a 7B model that is not a slow run but a useless
    # one -- fp32 weights, minutes per token -- so refuse instead of burning the
    # task's walltime on it.
    if DEVICE != "cuda" and not args.allow_cpu:
        raise SystemExit(
            f"CUDA is not available on {os.uname().nodename}; refusing to run the model on "
            f"the CPU. Check the node (./gsm8k_sweeps.sh nodecheck) and resubmit with "
            f"EXCLUDE=<node>. Nothing was written for this unit."
        )

    data_path = args.data or doc["data"]["path"]
    if file_sha256(data_path) != doc["data"]["sha256"]:
        raise SystemExit(f"{data_path} is not the file this plan was built from (sha256 differs)")
    problems = load_problems(data_path)

    # Imported here so `plan` and `merge` never touch the model path.
    from mikv_policy import load_model
    from mikv_runtime import torch

    model, tokenizer = load_model(doc["model_name"])
    point = SweepPoint(**unit["point"]) if unit["point"] is not None else None
    fixed = dict(
        run_key=unit["run_key"],
        method=unit["method"],
        job=(f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}"
             if "SLURM_ARRAY_JOB_ID" in os.environ else "interactive"),
        gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        model_name=doc["model_name"],
        n_shot=N_SHOT,
        max_new_tokens=MAX_NEW_TOKENS,
        max_tokens=MAX_TOKENS,
        prompt_version=PROMPT_VERSION,
        **(config_row_fields(point) if point is not None else {}),
    )

    writer = PartWriter(part_path)
    correct = elapsed = 0.0
    try:
        for n, idx in enumerate(todo, 1):
            row = evaluate_problem(model, tokenizer, problems[idx], unit["method"],
                                   point, unit["ratio"])
            row.update(fixed, timestamp=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
            writer.write(row)
            release_cuda_cache()
            correct += row["correct_strict"]
            elapsed += row["seconds"]
            print(
                f"[gsm8k u{args.unit} {n}/{len(todo)}] idx={idx} gold={row['gold']} "
                f"strict={row['pred_strict'] or '-'} flex={row['pred_flexible'] or '-'} "
                f"{'OK' if row['correct_strict'] else 'WRONG'} t_p={row['t_p']} "
                f"gen={row['gen_tokens']} ({row['stop_reason']}) {row['seconds']:.1f}s | "
                f"acc {100 * correct / n:.1f}% avg {elapsed / n:.1f}s/problem",
                flush=True,
            )
    finally:
        writer.close()
    print(f"=== unit done: {int(correct)}/{len(todo)} correct (strict) this task, "
          f"{elapsed / len(todo):.1f} s/problem ===", flush=True)


# ---- merge ----


def _table_rows(summaries: list[dict]) -> list[dict]:
    """Summary rows shaped for mikv_report.format_results_table: numeric fields it
    formats, and NaN where the baseline has not run yet."""
    rows = []
    for s in summaries:
        if s["method"] != "mikv":
            continue
        base = s.get("baseline_accuracy")
        rows.append(dict(
            s,
            ratio=float(s["ratio"]),
            k=round(float(s["k"]), 1),
            baseline_accuracy=float(base) if base not in ("", None) else float("nan"),
        ))
    return rows


def cmd_merge(args) -> None:
    parts_dir = os.path.join(args.results_root, "parts")
    runs, conflicts = load_all_rows(parts_dir)
    if not runs:
        raise SystemExit(f"no per-problem rows under {parts_dir} yet")
    n_total = len(load_problems(args.data))
    settings = generation_settings()

    baseline_key = run_key("baseline", None, None, settings, MODEL_NAME)
    baseline = runs.get(baseline_key, {})
    lr_index = load_line_retrieval(args.line_retrieval_csv, args.lr_num_records) \
        if args.line_retrieval_csv else {}

    # Only sweeps some submission actually planned: the _full definitions share
    # runs with the lean ones, and summarizing a sweep nobody submitted would list
    # it as partly done under its own name.
    submitted = set()
    for path in glob.glob(os.path.join(args.results_root, "plans", "*.json")):
        with open(path) as f:
            submitted.update(json.load(f).get("sweeps", []))

    summaries: list[dict] = []
    covered: set[str] = set()
    for name, spec in GSM8K_SWEEPS.items():
        if submitted and name not in submitted:
            continue
        chosen = sweep_problems(spec, n_total)
        for method, point, ratio in sweep_runs(spec)[0]:
            key = run_key(method, point, ratio, settings, MODEL_NAME)
            covered.add(key)
            summary = summarize_run(name, key, runs.get(key, {}), chosen, baseline,
                                    spec["num_problems"], spec["problem_seed"])
            if summary is not None:
                attach_line_retrieval(summary, lr_index)
                summaries.append(summary)
    # Runs no current sweep asks for (e.g. S7 finalists since edited away):
    # summarized over whatever problems they have, so nothing measured is lost.
    for key, rows in runs.items():
        if key in covered:
            continue
        summary = summarize_run("-", key, rows, sorted(rows), baseline, 0, 0)
        attach_line_retrieval(summary, lr_index)
        summaries.append(summary)

    # Baseline first, then run by run, each in problem order -- so the file reads
    # as blocks a spreadsheet filter on run_key splits cleanly.
    order = sorted(runs, key=lambda k: (k != baseline_key, k))
    problem_rows = [row for key in order for _, row in sorted(runs[key].items())]
    write_csv_atomic(os.path.join(args.results_root, "gsm8k_problems.csv"),
                     PROBLEM_CSV_FIELDS, problem_rows)
    write_csv_atomic(os.path.join(args.results_root, "gsm8k_summary.csv"),
                     SUMMARY_CSV_FIELDS, summaries)
    print(f"[merge] {len(problem_rows)} problem rows over {len(runs)} runs -> "
          f"{args.results_root}/gsm8k_problems.csv")
    print(f"[merge] {len(summaries)} summary rows -> {args.results_root}/gsm8k_summary.csv")

    for name in dict.fromkeys(s["sweep"] for s in summaries):
        mine = [s for s in summaries if s["sweep"] == name]
        label = "unassigned" if name == "-" else name
        out_dir = os.path.join(args.results_root, label)
        write_csv_atomic(os.path.join(out_dir, "summary.csv"), SUMMARY_CSV_FIELDS, mine)
        incomplete = sum(not s["complete"] for s in mine)
        table = _table_rows(mine)
        report = format_results_table(table) if table else "(baseline only)"
        with open(os.path.join(out_dir, "report.md"), "w") as f:
            f.write(f"# GSM8K {label}\n\n{report}\n")
        print(f"[merge] {label}: {len(mine)} rows ({incomplete} incomplete) -> {out_dir}/summary.csv, report.md")

    if not baseline:
        print("[merge] WARNING: no baseline rows yet -- the paired columns are empty until "
              "`gsm8k_sweeps.sh submit baseline` has run")
    if conflicts:
        print(f"[merge] WARNING: {conflicts} duplicated problems disagreed between copies "
              f"(GPU nondeterminism); the first copy was kept")
    for method, s in measured_seconds(parts_dir).items():
        print(f"[merge] slowest {method} run so far: {s:.1f} s/problem")


# ---- CLI ----


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mikv_gsm8k.py",
        description="GSM8K sweeps of the MiKV KV-cache policy, as SLURM job arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print (and optionally freeze) the pending units of a group",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    plan.add_argument("--group", required=True,
                      help="a group (tier1, all), a sweep (baseline, probe, S1..S7), "
                           "or a comma-separated mix")
    plan.add_argument("--data", default=DEFAULT_DATA_PATH)
    plan.add_argument("--problems-per-task", type=int, default=330,
                      help="largest unit; a 1319-problem run splits into 4 tasks, a 250-problem run into 1")
    plan.add_argument("--sec-per-problem", type=float, default=None,
                      help="MiKV s/problem for sizing --time (default: measured, else a guess)")
    plan.add_argument("--sec-per-problem-baseline", type=float, default=None)
    plan.add_argument("--write-units", default=None, help="freeze the pending units to this JSON file")
    plan.add_argument("--submit-info", default=None,
                      help="write NUNITS/TIME/GPU_HOURS as shell assignments to this file")
    plan.add_argument("--no-k-preview", action="store_true",
                      help="skip tokenizing the prompts to preview t_p and k")

    run = sub.add_parser("run", help="execute one unit of a frozen units file")
    run.add_argument("--units-file", required=True)
    run.add_argument("--unit", type=int, required=True)
    run.add_argument("--data", default=None, help="override the data path recorded in the plan")
    run.add_argument("--allow-cpu", action="store_true",
                     help="run even without CUDA (debugging only; hopeless for a 7B model)")

    merge = sub.add_parser("merge", help="rebuild the summary CSVs from the part files",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    merge.add_argument("--data", default=DEFAULT_DATA_PATH)
    merge.add_argument("--line-retrieval-csv", action="append", default=[],
                       help="glob of Line Retrieval sweep CSVs to join on (config, ratio); repeatable")
    merge.add_argument("--lr-num-records", type=int, default=40,
                       help="only join Line Retrieval rows at this record count (0 = any)")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if getattr(args, "lr_num_records", None) == 0:
        args.lr_num_records = None
    {"plan": cmd_plan, "run": cmd_run, "merge": cmd_merge}[args.command](args)
    sys.stdout.flush()
