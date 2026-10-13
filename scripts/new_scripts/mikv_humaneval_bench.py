"""
What a configuration is scored on for HumanEval: the verbatim prompt, the stop
markers, completion extraction, execution against the hidden tests, and one
problem's evaluation under the MiKV policy (or the uncompressed baseline) plus
the KV footprint that run implies. The HumanEval counterpart of mikv_gsm8k_bench.

Why HumanEval after GSM8K: Line Retrieval is a lookup and barely sees the
*precision* of what the cache kept. HumanEval decodes 50-500 tokens of code where
one flipped token is a syntax or logic error the unit tests catch, so it is
sensitive to the LOW width, the balancer and the importance policy.

Protocol (MiKV, Yang et al. 2024, App. D.2): all 164 problems, greedy, one sample
each, pass@1 = fraction whose tests pass.

--- the prompt ---

`problem["prompt"]` verbatim: imports, signature, docstring. No chat template, no
instruction -- raw completion, like the GSM8K plain-text prompt.

--- extracting the completion ---

Two traps, both of which silently score ~0%:

  1. Leading indentation. The body starts with "    ". SentencePiece drops the
     leading space of the *first* token when a continuation is decoded on its
     own, so decoding only the new ids can yield a 3-space body and an
     IndentationError. The completion is therefore taken as
     decode(prompt + new)[len(decode(prompt)):], never left-stripped.
  2. The prompt. evaluate_functional_correctness concatenates prompt + completion
     itself, so the completion must not contain it.

--- short prompts and the budget ---

t_p is ~40-400 tokens (mean ~130), so k = floor(r * t_p) at r <= 0.5 is under 64
on nearly every problem. With the RTL's w = 64 the policy is then pure recency
(k_H = 0) -- the plan's k preview counts it and every row records k_h. The
HumanEval sweeps therefore use w = k/2 as their reference and keep w = 64 as its
own control sweep.
"""

import gzip
import hashlib
import json
import time

from mikv_runtime import torch
from mikv_config import DEVICE, high_precision_knobs
from mikv_bench import kv_cache_size_bytes, resolve_budget_k
from mikv_gsm8k_bench import effective_window
from mikv_policy import _generate_tokens_baseline, _generate_tokens_with_mikv
from mikv_sweep import SweepPoint

HUMANEVAL_SIZE = 164

# Generation settings -- all folded into a run's key, as for GSM8K.
MAX_NEW_TOKENS = 512
# A stop marker is a top-level construct after the function: the body is done.
# Body lines are indented, so "\n#" / "\ndef " never cut a nested comment/helper.
STOP_STRINGS = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```")
MAX_TOKENS = 4096
EXEC_TIMEOUT_S = 3.0  # evaluate_functional_correctness's default
PROMPT_VERSION = "humaneval-verbatim-v1"


def generation_settings() -> dict:
    return dict(
        benchmark="humaneval",
        max_new_tokens=MAX_NEW_TOKENS,
        max_tokens=MAX_TOKENS,
        stop_strings=list(STOP_STRINGS),
        exec_timeout_s=EXEC_TIMEOUT_S,
        prompt_version=PROMPT_VERSION,
    )


# ---- the dataset ----


def default_data_path() -> str:
    """The HumanEval.jsonl.gz shipped inside the installed human-eval package.
    `human_eval.data` only reads files; nothing that executes code is imported."""
    from human_eval.data import HUMAN_EVAL
    return HUMAN_EVAL


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_problems(path: str) -> list[dict]:
    """The 164 problems ordered by task number, so problems[idx] is HumanEval/idx."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: int(r["task_id"].split("/")[-1]))
    for idx, r in enumerate(records):
        if r["task_id"] != f"HumanEval/{idx}":
            raise ValueError(f"{path}: expected HumanEval/{idx}, found {r['task_id']}")
        r["idx"] = idx
    return records


def select_problems(n_total: int, limit: int | None = None) -> list[int]:
    """Always the whole set, in order; `limit` takes a prefix (the probe), so a
    probe's rows are reused by the full run."""
    indices = list(range(n_total))
    return indices[:limit] if limit else indices


# ---- completion ----


def truncate_at_stop(text: str, stop_strings=STOP_STRINGS) -> tuple[str, bool]:
    """`text` cut at the earliest stop marker. The decode loop checks after each
    token, so the marker (and possibly a token past it) is in the raw text."""
    cut = min((i for i in (text.find(s) for s in stop_strings) if i >= 0), default=-1)
    return (text[:cut], True) if cut >= 0 else (text, False)


def decode_continuation(tokenizer, ids: list[int], t_p: int) -> str:
    """The new text with its leading whitespace intact -- see the module docstring."""
    prompt_text = tokenizer.decode(ids[:t_p], skip_special_tokens=True)
    full_text = tokenizer.decode(ids, skip_special_tokens=True)
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text):]
    # A byte-fallback merge across the boundary; fall back to the new ids alone.
    return tokenizer.decode(ids[t_p:], skip_special_tokens=True)


def check_completion(problem: dict, completion: str) -> tuple[bool, str]:
    """Run prompt + completion + tests in the harness's sandboxed subprocess.
    Executes model-written code: only ever call this inside a SLURM task."""
    from human_eval.execution import check_correctness
    out = check_correctness(problem, completion, EXEC_TIMEOUT_S)
    return bool(out["passed"]), str(out["result"])


# ---- one problem ----


def _run_generation(model, tokenizer, prompt: str, method: str, point: SweepPoint | None,
                    ratio: float | None) -> tuple[list[int], int]:
    """(all token ids, prompt length)."""
    if method == "baseline":
        generated, t_p = _generate_tokens_baseline(
            model, tokenizer, prompt, max_tokens=MAX_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS, stop_strings=STOP_STRINGS,
        )
    else:
        generated, t_p = _generate_tokens_with_mikv(
            model, tokenizer, prompt, max_tokens=MAX_TOKENS, budget_ratio=ratio,
            max_new_tokens=MAX_NEW_TOKENS, stop_strings=STOP_STRINGS,
            **point.policy_kwargs(),
        )
    return generated[0].tolist(), t_p


def evaluate_problem(model, tokenizer, problem: dict, method: str,
                     point: SweepPoint | None = None, ratio: float | None = None) -> dict:
    """Generate, extract, execute; return the per-problem row."""
    started = time.time()
    ids, t_p = _run_generation(model, tokenizer, problem["prompt"], method, point, ratio)
    gen_seconds = time.time() - started

    new_ids = ids[t_p:]
    completion, hit_stop = truncate_at_stop(decode_continuation(tokenizer, ids, t_p))
    gen_tokens = len(new_ids)
    if hit_stop:
        stop_reason = "stop_string"
    elif new_ids and new_ids[-1] == tokenizer.eos_token_id:
        stop_reason = "eos"
    elif gen_tokens >= MAX_NEW_TOKENS:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "max_tokens"

    passed, result = check_completion(problem, completion)
    occupied = t_p + gen_tokens
    row = dict(
        method=method,
        task_id=problem["task_id"],
        problem_idx=problem["idx"],
        passed=int(passed),
        result=result,
        t_p=t_p,
        gen_tokens=gen_tokens,
        stop_reason=stop_reason,
        seq_len=MAX_TOKENS,
        avg_seq_len=occupied,
        seconds=round(gen_seconds, 2),
        completion=completion,
    )
    if method == "baseline":
        return dict(row, ratio="", k="", w_eff="", k_h="", compression_pct=100.0,
                    compression_pct_occupied=100.0, avg_kv_bits_occupied=16.0)

    high_bits, native = high_precision_knobs(point.high_tier)
    widths = dict(high_bits=high_bits, low_bits=point.low_bits, high_precision_native=native)
    k = resolve_budget_k(point.budget_mode, ratio, t_p, MAX_TOKENS)
    prefill_k = resolve_budget_k(point.budget_mode, ratio, t_p, t_p)
    w = effective_window(point, prefill_k)
    before = kv_cache_size_bytes(model, MAX_TOKENS, k=None)
    after = kv_cache_size_bytes(model, MAX_TOKENS, k=k, **widths)
    k_occupied = resolve_budget_k(point.budget_mode, ratio, t_p, occupied)
    before_occ = kv_cache_size_bytes(model, occupied, k=None)
    after_occ = kv_cache_size_bytes(model, occupied, k=k_occupied, **widths)
    return dict(
        row,
        ratio=ratio,
        k=k,
        w_eff=w,
        k_h=prefill_k - w,
        kv_size_before=before,
        kv_size_after=after,
        compression_pct=100 * after / before,
        # What the cache really held -- the x-axis of MiKV's Figure 6.
        compression_pct_occupied=100 * after_occ / before_occ,
        avg_kv_bits_occupied=16 * after_occ / before_occ,
    )


def prompt_lengths(tokenizer, problems: list[dict], indices: list[int]) -> list[int]:
    return [len(tokenizer(problems[i]["prompt"]).input_ids) for i in indices]


def release_cuda_cache() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def budget_k_preview(t_ps: list[int], ratio: float, point: SweepPoint) -> tuple[int, int, int]:
    """(min k, max k, problems with k_H = 0) for one (ratio, point) over `t_ps`."""
    ks = [resolve_budget_k(point.budget_mode, ratio, t, t) for t in t_ps]
    recency = sum(effective_window(point, k) >= k for k in ks)
    return min(ks), max(ks), recency
