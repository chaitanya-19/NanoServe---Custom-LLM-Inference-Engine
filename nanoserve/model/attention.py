"""Attention layer for the inference engine.

The attention module:
  1. Projects hidden states to Q, K, V.
  2. Applies RoPE to Q and K.
  3. Writes the new K, V into the paged KV cache at the slots given by
     ``slot_mapping``.
  4. Calls the paged-attention backend (pure-PyTorch or Triton).
  5. Projects the attention output back to hidden size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..kernels.paged_attention_torch import paged_attention_torch
from ..kernels.paged_attention_triton import paged_attention_triton
from .rope import apply_rope


@dataclass
class AttentionMetadata:
    """Metadata describing the current iteration's batched attention work.

    All tensors live on the GPU; integer tensors are int32.

    Attributes:
        slot_mapping: flat slot index (block_idx * block_size + offset) for each
            query token, telling the kv-cache writer where to store its K/V.
            Shape (total_q_tokens,).
        block_tables: per-sequence block table padded to ``max_blocks_per_seq``.
            Shape (num_seqs, max_blocks_per_seq).
        seq_lens: total KV length of each sequence after this iteration.
            Shape (num_seqs,).
        query_lens: number of Q tokens contributed by each sequence in this
            iteration. Shape (num_seqs,).
        query_start_loc: cumulative sum of ``query_lens`` (prefix sum starting
            at 0). Shape (num_seqs + 1,).
        use_triton: whether to dispatch to the Triton kernel.
    """

    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    query_start_loc: torch.Tensor
    block_size: int
    use_triton: bool = False


def write_kv_to_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter K/V into the paged cache.

    key, value: (total_q_tokens, num_kv_heads, head_dim)
    k_cache, v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    slot_mapping: (total_q_tokens,) flat slots into (num_blocks * block_size)
    """
    num_blocks, block_size = k_cache.shape[0], k_cache.shape[1]
    block_idx = slot_mapping // block_size
    block_off = slot_mapping % block_size
    k_cache[block_idx, block_off] = key
    v_cache[block_idx, block_off] = value


class Attention(nn.Module):
    """Grouped-query attention with paged KV cache."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection
        q_size = config.q_size
        kv_size = config.kv_size
        self.qkv_proj = nn.Linear(
            config.hidden_size, q_size + 2 * kv_size, bias=False
        )
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.q_size = q_size
        self.kv_size = kv_size

    def forward(
        self,
        hidden_states: torch.Tensor,         # (total_tokens, hidden_size)
        cos: torch.Tensor,                   # (total_tokens, head_dim/2)
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_meta: AttentionMetadata,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # RoPE on Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Write new K, V to cache before reading (so query tokens see themselves)
        write_kv_to_cache(k, v, k_cache, v_cache, attn_meta.slot_mapping)

        if attn_meta.use_triton:
            attn_out = paged_attention_triton(
                q, k_cache, v_cache,
                attn_meta.block_tables, attn_meta.seq_lens,
                attn_meta.query_start_loc, attn_meta.query_lens,
                self.scale, attn_meta.block_size,
            )
        else:
            attn_out = paged_attention_torch(
                q, k_cache, v_cache,
                attn_meta.block_tables, attn_meta.seq_lens,
                attn_meta.query_start_loc, attn_meta.query_lens,
                self.scale, attn_meta.block_size,
            )
        # attn_out: (total_q_tokens, num_heads, head_dim)
        out = attn_out.reshape(-1, self.num_heads * self.head_dim)
        return self.o_proj(out)
