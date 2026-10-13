"""
Import-order bootstrap. Two orderings in this project are load-bearing rather
than stylistic -- both produce a bare segfault with no Python traceback -- and
this module is where they are enforced once instead of in every file:

1. `HF_HUB_DISABLE_XET=1` must be set before `transformers`/`huggingface_hub`
   import, since it is read once at import time. hf-xet (huggingface_hub's
   accelerated download backend) has been observed to segfault partway through a
   first-time weight download on this machine; the plain HTTP downloader is
   slower but does not crash. `setdefault`, so a shell-level override still wins.
2. `torch`/`transformers` must import before `matplotlib.pyplot`. Importing
   matplotlib first and transformers second segfaults here -- a native-library
   symbol conflict, order-dependent, whichever loads first wins.

Consequence for the rest of the package: any module that touches matplotlib must
import THIS module first (see mikv_plots), and nothing may import matplotlib at
a lower level than that.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

__all__ = [
    "torch",
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "DynamicCache",
    "ALL_ATTENTION_FUNCTIONS",
    "apply_rotary_pos_emb",
    "eager_attention_forward",
]
