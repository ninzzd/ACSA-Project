"""
Boilerplate for autoregressive inference on Qwen2.5-0.5B with a hook point
to mutate the KV cache between forward passes (e.g. to simulate KV
quantization for MiKV). Replace `quantize_kv` with the actual scheme
being evaluated.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def quantize_kv(tensor: torch.Tensor, bits: int = 2) -> torch.Tensor:
    """
    Fake-quantize a single K or V tensor: round-trip it through a
    `bits`-wide affine quantizer (per key/value vector, i.e. per last dim)
    and dequantize back to floats. This is the injection point for the
    actual MiKV quantization scheme.

    tensor: (batch, num_kv_heads, seq_len, head_dim)
    """
    qmax = 2**bits - 1
    t_min = tensor.amin(dim=-1, keepdim=True)
    t_max = tensor.amax(dim=-1, keepdim=True)
    scale = (t_max - t_min).clamp(min=1e-8) / qmax
    quantized = torch.round((tensor - t_min) / scale)
    dequantized = quantized * scale + t_min
    return dequantized.to(tensor.dtype)


def apply_kv_quantization(
    cache: DynamicCache, bits: int = 8, layers: set[int] | None = None
) -> None:
    """
    Mutate a DynamicCache in place, quantizing every layer's key/value
    tensors, or only `layers` (a set of layer indices) if given.
    """
    for layer_idx, layer in enumerate(cache.layers):
        if layers is not None and layer_idx not in layers:
            continue
        if layer.keys is not None:
            layer.keys = quantize_kv(layer.keys, bits=bits)
        if layer.values is not None:
            layer.values = quantize_kv(layer.values, bits=bits)


@torch.no_grad()
def generate_with_kv_quantization(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    bits: int = 8,
    quantize_every_step: bool = True,
) -> str:
    """
    Manual autoregressive generation loop. `past_key_values` is fetched
    back after every forward pass, giving a chance to mutate it (e.g.
    quantize it) before it's fed into the next pass.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = input_ids
    next_input_ids = input_ids
    cache = DynamicCache()

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=next_input_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cache = outputs.past_key_values

        # --- KV manipulation hook: mutate `cache` here every pass ---
        if quantize_every_step:
            apply_kv_quantization(cache, bits=bits)
        # --------------------------------------------------------------

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        next_input_ids = next_token

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()
    prompt = "The quick brown fox"
    output = generate_with_kv_quantization(
        model, tokenizer, prompt, max_new_tokens=32, bits=8
    )
    print(output)
