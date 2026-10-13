"""
What a configuration is *scored on*: the Line Retrieval benchmark, the no-quant
baseline it is measured against, and the KV footprint arithmetic that turns a
budget into bytes.

Line Retrieval (Li et al., 2023a; MiKV Figure 3 / Appendix D.3) plants random
"line <name>: REGISTER_CONTENT is <value>" facts in the prompt and asks the model
to retrieve one by name. We reproduce the paper's system instruction and user
message, but deliver them through the tokenizer's own `apply_chat_template`
rather than hand-writing Llama-2-chat's [INST]/<<SYS>> syntax: that syntax is
specific to Llama-2-chat's training format, and on a model not trained on it
(verified against a base model: it treats "[/INST]" as arbitrary text and emits
EOS immediately) it elicits no real response at all. The chat template gets the
same semantic structure in whatever format the target model understands, so this
path works unchanged across chat-tuned models.

The footprint side is an *estimate*, deliberately: `quantize_kv` is a fake
quantizer that dequantizes back to the model's dtype, so it never shrinks the
real DynamicCache. `kv_cache_size_bytes` instead computes what the footprint
would be if the two tiers were genuinely stored at their widths, which is what
lets accuracy be traded against real compression.
"""

import math
import random
import re
from dataclasses import dataclass

from mikv_runtime import torch
from mikv_config import (
    BUDGET_MODES,
    HW_AGE_LUT_ENTRIES,
    DEFAULT_BALANCE_SCHEME,
    DEFAULT_BUDGET_MODE,
    DEFAULT_BUDGET_RATIO,
    DEFAULT_HIGH_BITS,
    DEFAULT_HIGH_PRECISION_NATIVE,
    DEFAULT_LOW_BITS,
    DEFAULT_SCORE_BITS,
    DEFAULT_SCORE_DECAY_APPLICATION,
    DEFAULT_SCORE_DECAY_SCHEME,
    DEFAULT_SCORE_SCHEME,
    DEFAULT_WINDOW_RATIO,
    DEFAULT_WINDOW_TOKENS,
    DEVICE,
    HW_SCORE_FRAC_BITS,
    HW_SCORE_LENGTH_BITS,
    HW_SCORE_SIGNED,
    format_score_tag,
)
from mikv_policy import _generate_tokens_with_mikv

# ---- Line Retrieval benchmark -- the prompt corpus ----
#
# The adjective/noun pools below generate the random line names the paper
# plants (their Figure 15). See the module docstring for why the prompt is
# delivered through `apply_chat_template` rather than literal [INST]/<<SYS>>.

_LINE_ADJECTIVES = [
    "billowy", "psychotic", "daffy", "exclusive", "enthusiastic", "handsome",
    "enchanting", "sour", "faithful", "picayune", "wee", "forgetful", "cagey",
    "childlike", "inconclusive", "delightful", "courageous", "scandalous",
    "mere", "annoyed", "brave", "curious", "distant", "eager", "fancy",
    "gentle", "hollow", "icy", "jolly", "keen",
]
_LINE_NOUNS = [
    "schizophrenic", "cement", "pancake", "bough", "navigation", "variability",
    "thrust", "hippopotamus", "tabernacle", "cookie", "basics", "struggle",
    "cargo", "polyp", "flesh", "location", "viability", "laboratory", "affect",
    "armrest", "canyon", "domino", "engine", "feather", "garden", "harbor",
    "island", "jacket", "kettle", "ladder",
]


@dataclass
class LineRetrievalSample:
    prompt: str
    target_line: str
    expected_value: int


def make_line_retrieval_sample(
    tokenizer, num_records: int = 20, value_digits: int = 5, rng: random.Random | None = None
) -> LineRetrievalSample:
    """Build one synthetic Line Retrieval prompt: `num_records` random facts, then a retrieval question."""
    rng = rng or random
    names = set()
    while len(names) < num_records:
        names.add(f"{rng.choice(_LINE_ADJECTIVES)}-{rng.choice(_LINE_NOUNS)}")
    names = list(names)
    rng.shuffle(names)

    low, high = 10 ** (value_digits - 1), 10**value_digits - 1
    values = {name: rng.randint(low, high) for name in names}
    target_line = rng.choice(names)

    lines = "\n".join(f"line {name}: REGISTER_CONTENT is <{values[name]}>" for name in names)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a record processing computer. Given a list of records, and a "
                "target <line index>, you retrieve the '<REGISTER_CONTENT>' number."
            ),
        },
        {
            "role": "user",
            "content": (
                "Below is a record of lines I want you to remember. Each line begins with "
                "'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the "
                "line as a numerical value. For each line index, memorize its corresponding "
                "<REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve "
                "the corresponding <REGISTER_CONTENT> of a certain line index. Now the "
                f"record start:\n\n{lines}\n\n"
                "Now the record is over. Tell me what is the <REGISTER_CONTENT> in line "
                f"{target_line}? I need the number."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return LineRetrievalSample(prompt=prompt, target_line=target_line, expected_value=values[target_line])


def _extract_number(text: str) -> int | None:
    m = re.search(r"<(-?\d+)>", text)          # prefer the paper's <...> format
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d{4,6}", text)      # else: first plausible register value
    return int(nums[0]) if nums else None


def run_line_retrieval_benchmark(
    model,
    tokenizer,
    num_samples: int = 20,
    num_records: int = 20,
    budget_ratio: float = DEFAULT_BUDGET_RATIO,
    window_ratio: float = DEFAULT_WINDOW_RATIO,
    window_tokens: int | None = DEFAULT_WINDOW_TOKENS,
    high_bits: int = DEFAULT_HIGH_BITS,
    low_bits: int = DEFAULT_LOW_BITS,
    high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
    balance_scheme: str = DEFAULT_BALANCE_SCHEME,
    budget_mode: str = DEFAULT_BUDGET_MODE,
    score_scheme: str = DEFAULT_SCORE_SCHEME,
    score_bits: int = DEFAULT_SCORE_BITS,
    score_decay_scheme: str = DEFAULT_SCORE_DECAY_SCHEME,
    score_decay_application: str = DEFAULT_SCORE_DECAY_APPLICATION,
    score_length_bits: int = HW_SCORE_LENGTH_BITS,
    score_frac_bits: int = HW_SCORE_FRAC_BITS,
    score_signed: bool = HW_SCORE_SIGNED,
    age_lut_entries: int = HW_AGE_LUT_ENTRIES,
    max_tokens: int = 4096,
    seed: int = 0,
) -> float:
    """
    Line Retrieval accuracy under the MiKV cache policy (paper Figure 3b /
    Table 1): generate `num_samples` synthetic line-retrieval prompts,
    greedily decode each under the mixed-precision policy, and score
    whether the retrieved value matches the planted one.

    `max_tokens` is the total context budget (prompt + generation), not
    just the new-token count -- e.g. 4096 to simulate a 4K-context run,
    however long the planted-record prompt happens to be.

    `balance_scheme` selects the channel balancer -- see BALANCE_SCHEMES.
    `budget_mode` selects how k is resolved from `budget_ratio` -- see
    BUDGET_MODES. `score_scheme`/`score_bits` select the precision the
    importance scoreboard accumulates at -- see SCORE_SCHEMES -- and
    `score_decay_scheme` the precision of the age-decay factor applied to it
    -- see SCORE_DECAY_SCHEMES.
    """
    print(
        f"[mikv] starting {num_samples} samples: budget_mode={budget_mode} "
        f"balance_scheme={balance_scheme} score_scheme={score_scheme} "
        f"score_bits={score_bits if score_scheme == 'quant' else '-'} "
        f"score_decay={score_decay_scheme}/{score_decay_application} "
        f"age_lut={age_lut_entries if score_decay_scheme == 'lut' else '-'} "
        f"fixed_point=({int(score_signed)},{score_length_bits},{score_frac_bits}) "
        f"budget_ratio={budget_ratio} "
        f"window={'w=' + str(window_tokens) if window_tokens is not None else 'ratio ' + str(window_ratio)} "
        f"high_bits={high_bits} low_bits={low_bits} "
        f"high_precision_native={high_precision_native}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed)
    correct = 0
    for i in range(num_samples):
        sample = make_line_retrieval_sample(tokenizer, num_records=num_records, rng=rng)
        prompt_len = tokenizer(sample.prompt, return_tensors="pt").input_ids.shape[-1]
        generated, _ = _generate_tokens_with_mikv(
            model,
            tokenizer,
            sample.prompt,
            max_tokens=max_tokens,
            budget_ratio=budget_ratio,
            window_ratio=window_ratio,
            window_tokens=window_tokens,
            high_bits=high_bits,
            low_bits=low_bits,
            high_precision_native=high_precision_native,
            balance_scheme=balance_scheme,
            budget_mode=budget_mode,
            score_scheme=score_scheme,
            score_bits=score_bits,
            score_decay_scheme=score_decay_scheme,
            score_decay_application=score_decay_application,
            score_length_bits=score_length_bits,
            score_frac_bits=score_frac_bits,
            score_signed=score_signed,
            age_lut_entries=age_lut_entries,
        )
        continuation = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        predicted = _extract_number(continuation)
        is_correct = predicted == sample.expected_value
        correct += int(is_correct)
        print(
            f"[mikv {i + 1}/{num_samples}] target={sample.target_line} expected={sample.expected_value} "
            f"predicted={predicted} {'OK' if is_correct else 'WRONG'}",
            flush=True,
        )
        # Each sample re-installs a fresh MiKVPolicy and grows a new DynamicCache to a
        # slightly different length (num_records draws a random count of unique names),
        # so the CUDA caching allocator accumulates many differently-sized cached blocks
        # over hundreds of samples. On a ~3.3GB-free GPU that's enough fragmentation to
        # trigger (recoverable) OOM retries -- releasing the cache each sample keeps the
        # allocator's pool from fragmenting across runs of varying shape.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / num_samples
    print(
        f"[mikv] Line Retrieval accuracy "
        f"({budget_mode}/{balance_scheme}/"
        f"{format_score_tag(score_scheme, score_bits, score_decay_scheme, score_decay_application, score_length_bits, score_frac_bits, score_signed, age_lut_entries)}): "
        f"{accuracy * 100:.1f}% "
        f"({correct}/{num_samples}) {_cuda_mem_str()}",
        flush=True,
    )
    return accuracy


# ---- KV cache size estimation & compression sweep ----
#
# The fake-quantizer in `quantize_kv` round-trips values through an N-bit
# affine quantizer but dequantizes back to the model's native dtype, so it
# never actually shrinks the DynamicCache's real memory footprint -- it's
# built to measure quantization's effect on accuracy, not to save memory.
# The functions below instead *compute* what the footprint would be if the
# high/low precision buckets were actually stored at their respective bit
# widths, so accuracy can be traded off against real compression.


def _kv_cache_shape(model) -> tuple[int, int, int]:
    """(num_layers, num_kv_heads, head_dim) sizing one cached token's K/V."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return num_layers, num_kv_heads, head_dim


def _cuda_mem_str() -> str:
    """
    Real (not estimated) CUDA memory in MB, for diagnosing GPU pressure:
    `allocated` is memory backing tensors actually live right now; `reserved`
    is the caching allocator's total pool held from the driver (allocated +
    cached-but-freed blocks kept around for reuse -- this is what nvidia-smi
    /nvtop report as this process's VRAM usage, so it reads higher than
    `allocated`); `peak_allocated` is the high-water mark for `allocated`
    since the last `torch.cuda.reset_peak_memory_stats()` call.
    """
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return f"cuda_mem: allocated={allocated:.0f}MB reserved={reserved:.0f}MB peak_allocated={peak:.0f}MB"


def kv_cache_size_bytes(
    model,
    seq_len: int,
    k: int | None = None,
    high_bits: int = DEFAULT_HIGH_BITS,
    low_bits: int = DEFAULT_LOW_BITS,
    high_precision_native: bool = DEFAULT_HIGH_PRECISION_NATIVE,
) -> float:
    """
    Estimate the KV cache footprint (bytes, K+V across all layers/heads)
    for a sequence of length `seq_len`.

    k=None: uncompressed baseline -- every position at the model's native
    16-bit dtype.
    k=<int>: MiKV steady state -- per `MiKVPolicy._importance_set`, the
    importance set holds exactly k positions once the cache is longer than
    k, so exactly min(seq_len, k) positions stay HIGH precision (native 16-bit if
    `high_precision_native`, else quantized to `high_bits` -- must match
    whatever the accuracy run actually used) and the remaining
    max(0, seq_len - k) are compressed to `low_bits` (N).

    k is the budget the run *ends up at* by `seq_len`, which differs by
    budget mode -- frozen at r*t_p under "fixed_length", grown to r*seq_len
    under "fixed_ratio". Use `resolve_budget_k` to get it rather than
    computing r*t_p by hand.
    """
    num_layers, num_kv_heads, head_dim = _kv_cache_shape(model)
    elements_per_token = num_layers * num_kv_heads * head_dim * 2  # *2 for K and V

    if k is None:
        return elements_per_token * seq_len * 16 / 8

    high_bit_width = 16 if high_precision_native else high_bits
    num_high = min(seq_len, k)
    num_low = max(0, seq_len - k)
    return elements_per_token * (num_high * high_bit_width / 8 + num_low * low_bits / 8)


def resolve_budget_k(budget_mode: str, budget_ratio: float, t_p: int, seq_len: int) -> int:
    """
    The steady-state budget k a run under `budget_mode` ends up holding once
    the cache has grown to `seq_len` tokens -- the number `kv_cache_size_bytes`
    must be sized against.

    "fixed_length": k was frozen at prefill from the prompt length, so it is
    floor(r * t_p) no matter how far decoding ran. "fixed_ratio": k tracks the
    cache, so at `seq_len` tokens it is floor(r * seq_len). Mirrors
    `MiKVPolicy.budget_for` (same floor, same max(1, ...) guard) -- the two must
    agree or the reported compression won't describe the run that produced the
    accuracy next to it.
    """
    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode must be one of {BUDGET_MODES}, got {budget_mode!r}")
    t = t_p if budget_mode == "fixed_length" else seq_len
    return max(1, math.floor(budget_ratio * t))


def run_line_retrieval_no_quant(
    model,
    tokenizer,
    num_samples: int = 20,
    num_records: int = 20,
    max_tokens: int = 4096,
    seed: int = 0,
) -> tuple[float, int, float, float]:
    """
    Baseline Line Retrieval pass with the stock HF generation loop -- no
    MiKV balancing, scoring, or quantization, so the KV cache stays at the
    model's native 16-bit dtype throughout. Establishes (a) t_p, the
    prompt length at the first prefill stage (constant across samples
    since every prompt plants the same `num_records` facts), and (b) the
    "before compression" KV cache size at the end of generation -- both
    needed to evaluate the MiKV sweep in `sweep_kv_compression`.

    Must run before any MiKV call (`generate_with_mikv`,
    `run_line_retrieval_benchmark`, ...): those monkey-patch each layer's
    attention forward in place and never restore the original, so once
    installed there is no unpatched model left to baseline against.

    `max_tokens` is the total context budget (prompt + generation), passed
    straight to `model.generate(..., max_length=max_tokens)` so it means
    the same thing here as it does in `run_line_retrieval_benchmark`.

    Returns (accuracy, t_p, avg_kv_cache_bytes_occupied, avg_seq_len).
    The last two describe what generation actually *occupied* (decoding
    stops at EOS, typically well short of `max_tokens`) and are reported
    as diagnostics only -- `sweep_kv_compression` sizes its compression
    ratio at the fixed provisioned context instead, since mixing an
    occupancy figure with a provisioned one overstates compression.
    """
    print(f"[no-quant] starting {num_samples} baseline samples (num_records={num_records})...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    correct = 0
    t_p = None
    total_bytes = 0.0
    total_seq_len = 0
    for i in range(num_samples):
        sample = make_line_retrieval_sample(tokenizer, num_records=num_records, rng=rng)
        input_ids = tokenizer(sample.prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_len = input_ids.shape[-1]
        if t_p is None:
            t_p = prompt_len
            print(f"[no-quant] t_p (prompt length) = {t_p} tokens", flush=True)

        output = model.generate(input_ids, max_length=max_tokens, do_sample=False, pad_token_id=pad_token_id)
        seq_len = output.shape[-1]
        total_seq_len += seq_len
        total_bytes += kv_cache_size_bytes(model, seq_len, k=None)

        continuation = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
        predicted = _extract_number(continuation)
        is_correct = predicted == sample.expected_value
        correct += int(is_correct)
        print(
            f"[no-quant {i + 1}/{num_samples}] target={sample.target_line} expected={sample.expected_value} "
            f"predicted={predicted} {'OK' if is_correct else 'WRONG'}",
            flush=True,
        )
        if torch.cuda.is_available():  # see the matching note in run_line_retrieval_benchmark
            torch.cuda.empty_cache()

    accuracy = correct / num_samples
    avg_bytes = total_bytes / num_samples
    print(
        f"[no-quant] baseline done: accuracy={accuracy * 100:.1f}% ({correct}/{num_samples}) "
        f"t_p={t_p} avg KV cache={avg_bytes / 1e6:.2f} MB {_cuda_mem_str()}",
        flush=True,
    )

    avg_seq_len = total_seq_len / num_samples
    return accuracy, t_p, avg_bytes, avg_seq_len
