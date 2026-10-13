"""
What a configuration is scored on for GSM8K: the 8-shot chain-of-thought prompt,
answer extraction, which problems a sweep runs, and one problem's evaluation --
generation under the MiKV policy (or the uncompressed baseline) plus the KV
footprint that run implies. The GSM8K counterpart of mikv_bench.

Why GSM8K after Line Retrieval: retrieval answers are a handful of tokens, so
almost every eviction decision is taken once, at prefill. A GSM8K answer is a
100-250 token chain of thought the model has to keep attending over, so the decode
path -- per-step score accumulation, the age decay, sticky demotion -- is
exercised on every token, and an error in any step propagates to the final number.

--- the prompt ---

The standard 8-shot chain-of-thought exemplars of Wei et al. (2022), as used by
lm-evaluation-harness's `gsm8k_cot` task, delivered as plain text ("Q: ...\nA: ...")
rather than through the chat template. The chat checkpoint follows this format
without any instruction, and plain text keeps every prompt token the same across
problems except the question itself, so t_p varies only with question length.

--- scoring ---

Two extractions, both reported:
  strict   -- the number after the first "The answer is", the format every
              exemplar ends in. The primary metric: a compressed cache that
              breaks the model's ability to follow the format is a real loss.
  flexible -- the last number anywhere in the completion. Lenient on format, so
              the gap between the two separates "reasoned wrongly" from "stopped
              following the format".
Numbers are compared after stripping "$" and thousands separators, as decimals
(so "18.00" == "18").
"""

import hashlib
import json
import math
import os
import random
import re
import time
from decimal import Decimal, InvalidOperation

from mikv_runtime import torch
from mikv_config import DEVICE, high_precision_knobs
from mikv_bench import kv_cache_size_bytes, resolve_budget_k
from mikv_policy import _generate_tokens_baseline, _generate_tokens_with_mikv
from mikv_sweep import SweepPoint

GSM8K_TEST_SIZE = 1319
DEFAULT_DATA_PATH = "data/gsm8k/test.jsonl"

# Generation settings. Every one of these is folded into a run's key (see
# mikv_gsm8k.run_key), so changing one starts fresh results instead of silently
# resuming into a file produced under the old setting.
N_SHOT = 8
# lm-eval's gsm8k_cot default. Llama-2-7b-chat's chains run ~100-200 tokens, so
# hitting this cap is itself a symptom -- `hit_max_new_tokens_pct` reports it.
MAX_NEW_TOKENS = 256
# lm-eval's `until` for gsm8k_cot: after its answer the model starts writing the
# next "Q:" exemplar. EOS is handled separately, by token id.
STOP_STRINGS = ("Q:",)
# The context the footprint is sized at, identical to Line Retrieval's
# `--max-tokens` default so compression_pct is on the same basis in both.
MAX_TOKENS = 4096
PROMPT_VERSION = "wei2022-cot-8shot-plain-v1"

# Wei et al. (2022), Table 20 -- the exemplar set lm-eval's gsm8k_cot uses.
COT_EXEMPLARS = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


def generation_settings() -> dict:
    """Everything about how a problem is prompted and decoded, as one dict -- the
    part of a run's identity that is not the cache configuration."""
    return dict(
        n_shot=N_SHOT,
        max_new_tokens=MAX_NEW_TOKENS,
        max_tokens=MAX_TOKENS,
        stop_strings=list(STOP_STRINGS),
        prompt_version=PROMPT_VERSION,
    )


def build_prompt(question: str, n_shot: int = N_SHOT) -> str:
    """lm-eval's gsm8k_cot layout: exemplars as "Q: ...\\nA: ..." joined by a blank
    line, then the test question with an open "A:" for the model to continue."""
    shots = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in COT_EXEMPLARS[:n_shot])
    return f"{shots}Q: {question.strip()}\nA:"


# ---- the dataset ----


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_problems(path: str = DEFAULT_DATA_PATH) -> list[dict]:
    """The GSM8K test split as a list indexed by problem number (the line number in
    the official test.jsonl -- stable, and what every result row is keyed on)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- download it once on the login node with "
            f"`./gsm8k_sweeps.sh prep`"
        )
    problems = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            gold = normalize_number(record["answer"].split("####")[-1])
            if gold is None:
                raise ValueError(f"{path}:{lineno}: no parsable '#### <answer>' in the reference")
            # idx is the position in the list, so problems[idx] always holds.
            problems.append(dict(idx=len(problems), question=record["question"], gold=gold))
    return problems


def select_problems(
    n_total: int, num_problems: int, seed: int, limit: int | None = None
) -> list[int]:
    """
    The problem indices a sweep runs, deterministically from (num_problems, seed).

    num_problems = 0 (or >= the test size) is the whole test set. Otherwise a
    seeded random subset, *sorted*, so a run's progress walks the file in order.
    Every sweep that asks for the same (num_problems, seed) gets the same problems,
    which is what makes accuracy differences between configurations paired
    comparisons on identical inputs rather than comparisons across two samples.
    `limit` truncates that list (the probe runs its first 20), so a probe's
    results are a prefix of -- and are reused by -- the full screen.
    """
    if not num_problems or num_problems >= n_total:
        indices = list(range(n_total))
    else:
        indices = sorted(random.Random(seed).sample(range(n_total), num_problems))
    return indices[:limit] if limit else indices


# ---- answer extraction ----

_NUMBER = r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?"
_STRICT = re.compile(r"The answer is\s*(" + _NUMBER + ")")
_ANY_NUMBER = re.compile(_NUMBER)


def normalize_number(text: str | None) -> str | None:
    """A number as a canonical string ("$1,000.50" -> "1000.5", "18.00" -> "18"),
    or None when there is none to parse."""
    if text is None:
        return None
    cleaned = text.replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def truncate_at_stop(text: str, stop_strings=STOP_STRINGS) -> tuple[str, bool]:
    """`text` cut at the earliest stop string, and whether one was found. The
    decode loop stops *after* producing a stop string, so it is always in the
    raw continuation and must come off before scoring."""
    cut = min((i for i in (text.find(s) for s in stop_strings) if i >= 0), default=-1)
    return (text[:cut], True) if cut >= 0 else (text, False)


def extract_strict(completion: str) -> str | None:
    m = _STRICT.search(completion)
    return normalize_number(m.group(1)) if m else None


def extract_flexible(completion: str) -> str | None:
    numbers = _ANY_NUMBER.findall(completion)
    return normalize_number(numbers[-1]) if numbers else None


# ---- one problem ----


def effective_window(point: SweepPoint, k: int) -> int:
    """w as `MiKVPolicy.budget_for` resolves it: clamped into [0, k]. k_H = k - w
    is then 0 whenever the window reaches the budget, i.e. the policy degenerates
    to pure recency for that problem -- worth flagging per row, because on a short
    prompt it happens at small r (S1's r = 0.05) and wide w (S4's w = 256)."""
    if point.window_tokens is not None:
        return max(0, min(k, int(point.window_tokens)))
    return max(0, min(k, math.floor(point.window_ratio * k)))


def _run_generation(model, tokenizer, prompt: str, method: str, point: SweepPoint | None,
                    ratio: float | None) -> tuple[list[int], int]:
    """(new token ids, prompt length). The only place a generation loop is called,
    so a test can replace it without a model."""
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
    return generated[0, t_p:].tolist(), t_p


def evaluate_problem(
    model, tokenizer, problem: dict, method: str,
    point: SweepPoint | None = None, ratio: float | None = None,
) -> dict:
    """
    Generate one answer under `method` ("mikv" at `point`/`ratio`, or "baseline")
    and return its per-problem result row: the prediction and its correctness,
    the budget the policy actually ran with for this prompt, and the footprint.

    The footprint is reported on two bases, like Line Retrieval's CSV: sized at the
    provisioned MAX_TOKENS context (`compression_pct`, the column comparable across
    benchmarks), and at the length this generation actually occupied
    (`compression_pct_occupied`, what the cache really held).
    """
    started = time.time()
    new_ids, t_p = _run_generation(model, tokenizer, build_prompt(problem["question"]),
                                   method, point, ratio)
    seconds = time.time() - started

    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    completion, hit_stop = truncate_at_stop(raw)
    gen_tokens = len(new_ids)
    if hit_stop:
        stop_reason = "stop_string"
    elif new_ids and new_ids[-1] == tokenizer.eos_token_id:
        stop_reason = "eos"
    elif gen_tokens >= MAX_NEW_TOKENS:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "max_tokens"

    pred_strict = extract_strict(completion)
    pred_flexible = extract_flexible(completion)
    occupied = t_p + gen_tokens
    row = dict(
        method=method,
        problem_idx=problem["idx"],
        gold=problem["gold"],
        pred_strict="" if pred_strict is None else pred_strict,
        pred_flexible="" if pred_flexible is None else pred_flexible,
        correct_strict=int(pred_strict == problem["gold"]),
        correct_flexible=int(pred_flexible == problem["gold"]),
        t_p=t_p,
        gen_tokens=gen_tokens,
        stop_reason=stop_reason,
        seq_len=MAX_TOKENS,
        avg_seq_len=occupied,
        seconds=round(seconds, 2),
        # JSON-encoded so the row stays on one line -- see mikv_gsm8k_report.
        completion=json.dumps(completion, ensure_ascii=False),
    )
    if method == "baseline":
        return dict(row, ratio="", k="", w_eff="", k_h="", compression_pct=100.0,
                    compression_pct_occupied=100.0)

    high_bits, native = high_precision_knobs(point.high_tier)
    widths = dict(high_bits=high_bits, low_bits=point.low_bits, high_precision_native=native)
    # k at the sizing context, as Line Retrieval's `k` column is; under
    # "fixed_length" it equals the prefill budget the policy actually froze.
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
        compression_pct_occupied=100 * after_occ / before_occ,
    )


def prompt_lengths(tokenizer, problems: list[dict], indices: list[int]) -> list[int]:
    """t_p for each of `indices` -- used by the plan to preview k before any GPU
    time is spent, since k = floor(r * t_p) and t_p varies with the question."""
    return [len(tokenizer(build_prompt(problems[i]["question"])).input_ids) for i in indices]


def release_cuda_cache() -> None:
    """Between problems: each one grows a DynamicCache to a different length, and
    releasing the allocator's pool keeps it from fragmenting across hundreds of
    differently-sized generations (see the matching note in mikv_bench)."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
