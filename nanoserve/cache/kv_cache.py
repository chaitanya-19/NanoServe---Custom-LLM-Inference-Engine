"""Physical KV cache tensors.

We allocate one (num_blocks, block_size, num_kv_heads, head_dim) tensor for K
and one for V, per layer.  Stored in a Python list so each layer can pass its
own slice to the attention module without indexing into a giant 5-D tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from ..config import CacheConfig, ModelConfig


@dataclass
class PagedKVCache:
    k_caches: List[torch.Tensor]   # one per layer
    v_caches: List[torch.Tensor]
    num_blocks: int
    block_size: int

    @property
    def num_layers(self) -> int:
        return len(self.k_caches)


def allocate_kv_cache(
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> PagedKVCache:
    shape = (cache_cfg.num_blocks, cache_cfg.block_size, model_cfg.num_key_value_heads, model_cfg.head_dim)
    k_caches: List[torch.Tensor] = []
    v_caches: List[torch.Tensor] = []
    for _ in range(model_cfg.num_hidden_layers):
        k_caches.append(torch.zeros(shape, dtype=dtype, device=device))
        v_caches.append(torch.zeros(shape, dtype=dtype, device=device))
    return PagedKVCache(k_caches, v_caches, cache_cfg.num_blocks, cache_cfg.block_size)


def kv_cache_memory_bytes(
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    dtype: torch.dtype,
) -> int:
    elem = torch.tensor([], dtype=dtype).element_size()
    per_token_per_layer = 2 * model_cfg.num_key_value_heads * model_cfg.head_dim * elem
    return (
        model_cfg.num_hidden_layers
        * cache_cfg.num_blocks
        * cache_cfg.block_size
        * per_token_per_layer
    ) // 2  # we factored in the 2 (K+V) into per_token_per_layer
