"""
Boilerplate for autoregressive inference on Qwen2.5-0.5B implementing the
MiKV decode-time policy: at every decode step, use the freshly computed
attention row to score every cached KV pair (per layer, per head) and
fake-quantize whichever single pair currently scores lowest.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # eager attention is required to get real softmaxed attention weights
    # back out of the forward pass (output_attentions=True is not
    # supported by the sdpa/flash-attention backends).
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, attn_implementation="eager"
    )
    model.to(DEVICE)
    model.eval()  # disables dropout etc.; inference-only forward passes
    return model, tokenizer


def quantize_kv(tensor: torch.Tensor, bits: int = 2) -> torch.Tensor:  # does not include channel biasing factor b
    """
    Fake-quantize K/V vectors: round-trip them through a `bits`-wide affine
    quantizer (per vector, i.e. per last dim) and dequantize back to floats.

    tensor: (..., head_dim)
    """
    qmax = 2**bits - 1
    t_min = tensor.amin(dim=-1, keepdim=True)
    t_max = tensor.amax(dim=-1, keepdim=True)
    scale = (t_max - t_min).clamp(min=1e-8) / qmax
    quantized = torch.round((tensor - t_min) / scale)
    dequantized = quantized * scale + t_min
    return dequantized.to(tensor.dtype)


def apply_kv_quantization(
    cache: DynamicCache,
    attentions: tuple[torch.Tensor, ...],
    num_query_heads: int,
    num_kv_heads: int,
    bits: int = 2,
) -> None:
    """
    Mutate a DynamicCache in place per the MiKV decode-step policy.

    For every layer and every KV head, use this decode step's attention
    row (the query is the single just-decoded token, so there is exactly
    one row) as an importance scoreboard over all cached token positions
    0..n. The single lowest-scoring position i<=n has its K/V vector for
    that head fake-quantized to `bits`; every other position/head is left
    untouched.

    attentions: one (batch, num_query_heads, q_len=1, kv_len) tensor per
    layer, as returned by the model with output_attentions=True during a
    single-token decode step.
    """
    groups = num_query_heads // num_kv_heads

    for layer_idx, layer in enumerate(cache.layers):
        if layer.keys is None or layer.values is None:
            continue

        attn = attentions[layer_idx]  # (batch, num_query_heads, 1, kv_len)
        batch, kv_len = attn.shape[0], attn.shape[-1]

        # score of every cached position, per kv-head: query heads sharing
        # a kv-head (GQA) are averaged together into one scoreboard.

        # this is a dummy representation of the 'scores' tensor, it must be a globally stored tensor
        # must change scores computation, assuming other mechanisms are fine
        scores = attn[:, :, -1, :].reshape(batch, num_kv_heads, groups, kv_len)
        scores = scores.mean(dim=2)  # (batch, num_kv_heads, kv_len)

        least_important = scores.argmin(dim=-1)  # (batch, num_kv_heads)

        batch_idx = torch.arange(batch, device=attn.device).view(-1, 1).expand(-1, num_kv_heads)
        head_idx = torch.arange(num_kv_heads, device=attn.device).view(1, -1).expand(batch, -1)

        selected_keys = layer.keys[batch_idx, head_idx, least_important, :]
        selected_values = layer.values[batch_idx, head_idx, least_important, :]

        layer.keys[batch_idx, head_idx, least_important, :] = quantize_kv(selected_keys, bits=bits)
        layer.values[batch_idx, head_idx, least_important, :] = quantize_kv(selected_values, bits=bits)


@torch.no_grad()  # disables autograd tracking; no backward pass needed for inference
def generate_with_mikv(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 4096,
    bits: int = 2,
) -> str:
    """
    Manual autoregressive generation loop applying the MiKV KV compression
    policy at every decode step (never during prefill, since the policy is
    defined in terms of a single freshly-decoded query row).
    """
    num_query_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_query_heads)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = input_ids
    cache = DynamicCache()

    # prefill: build the initial cache, no quantization (no decode row yet)
    outputs = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    cache = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=-1)
    next_input_ids = next_token

    for _ in range(max_tokens - input_ids.shape[-1] - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break

        outputs = model(
            input_ids=next_input_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
        cache = outputs.past_key_values

        apply_kv_quantization(
            cache, outputs.attentions, num_query_heads, num_kv_heads, bits=bits
        )

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        next_input_ids = next_token

    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()
    prompt = "The quick brown fox"
    output = generate_with_mikv(model, tokenizer, prompt, max_tokens=32, bits=2)
    print(output)
