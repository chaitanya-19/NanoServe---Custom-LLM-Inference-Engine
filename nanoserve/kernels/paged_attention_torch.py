"""Pure-PyTorch paged attention.

This is the "naive but correct" backend.  For each sequence in the batch we
gather its KV from the block table into a contiguous tensor, expand the KV
heads to match the Q heads (GQA), and call SDPA.  It is O(num_seqs) Python
overhead per layer per step, which on an L4 with concurrency <= 64 is small
enough to be useful for development and as a baseline for the Triton path.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gather_kv_for_seq(
    k_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,
    block_table_row: torch.Tensor,  # (max_blocks_per_seq,) int32
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    blocks = block_table_row[:num_blocks_needed]
    k = k_cache.index_select(0, blocks).reshape(-1, k_cache.shape[2], k_cache.shape[3])
    v = v_cache.index_select(0, blocks).reshape(-1, v_cache.shape[2], v_cache.shape[3])
    return k[:seq_len], v[:seq_len]


def paged_attention_torch(
    query: torch.Tensor,            # (total_q_tokens, num_heads, head_dim)
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,     # (num_seqs, max_blocks_per_seq) int32
    seq_lens: torch.Tensor,         # (num_seqs,)
    query_start_loc: torch.Tensor,  # (num_seqs + 1,)
    query_lens: torch.Tensor,       # (num_seqs,)
    scale: float,
    block_size: int,
) -> torch.Tensor:
    num_q_heads = query.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = query.shape[2]
    num_kv_groups = num_q_heads // num_kv_heads

    out = torch.empty_like(query)
    num_seqs = block_tables.shape[0]

    # Move host-side scalars in one shot to avoid repeated GPU->CPU syncs
    seq_lens_cpu = seq_lens.tolist()
    query_lens_cpu = query_lens.tolist()
    query_start_loc_cpu = query_start_loc.tolist()

    for i in range(num_seqs):
        q_start = query_start_loc_cpu[i]
        q_end = query_start_loc_cpu[i + 1]
        q_len = query_lens_cpu[i]
        kv_len = seq_lens_cpu[i]
        if q_len == 0:
            continue

        q_i = query[q_start:q_end]  # (q_len, num_q_heads, head_dim)
        k, v = _gather_kv_for_seq(
            k_cache, v_cache, block_tables[i], kv_len, block_size
        )  # (kv_len, num_kv_heads, head_dim)

        # GQA: repeat KV heads to match query heads
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)

        # SDPA expects (batch, heads, seq, head_dim)
        q_t = q_i.transpose(0, 1).unsqueeze(0)
        k_t = k.transpose(0, 1).unsqueeze(0)
        v_t = v.transpose(0, 1).unsqueeze(0)

        if q_len == kv_len:
            attn = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=scale
            )
        elif q_len == 1:
            # Decode: every KV is visible (causal trivially satisfied)
            attn = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        else:
            # Chunked prefill: queries are the *last* q_len positions of the
            # kv_len-token sequence; each query can attend to all preceding
            # KV plus itself.
            offset = kv_len - q_len
            kv_idx = torch.arange(kv_len, device=q_i.device)
            q_idx = torch.arange(q_len, device=q_i.device) + offset
            mask = kv_idx.unsqueeze(0) <= q_idx.unsqueeze(1)  # (q_len, kv_len)
            attn = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=scale
            )

        attn = attn.squeeze(0).transpose(0, 1)  # (q_len, num_q_heads, head_dim)
        out[q_start:q_end] = attn

    return out
