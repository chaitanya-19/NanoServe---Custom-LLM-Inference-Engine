"""Triton paged-attention kernel.

The kernel parallelizes over (q_token, head) so every query head of every
query token gets its own program.  Each program walks the KV blocks of its
sequence using online softmax (FlashAttention-style) so we never materialize
the full attention matrix.

We support GQA via NUM_KV_GROUPS and causal masking per-query (each query has
its own kv-position cutoff, which makes chunked-prefill and decode share one
code path).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_kernel(
    Q_ptr,                 # (total_q, num_heads_q, head_dim)
    K_cache_ptr,           # (num_blocks, block_size, num_kv_heads, head_dim)
    V_cache_ptr,
    Out_ptr,               # (total_q, num_heads_q, head_dim)
    block_tables_ptr,      # (num_seqs, max_blocks_per_seq) int32
    seq_lens_ptr,          # (num_seqs,) int32
    seq_for_q_ptr,         # (total_q,) int32
    kv_pos_for_q_ptr,      # (total_q,) int32  absolute kv pos of each q
    scale,
    stride_q_token, stride_q_head, stride_q_dim,
    stride_k_block, stride_k_token, stride_k_head, stride_k_dim,
    stride_v_block, stride_v_token, stride_v_head, stride_v_dim,
    stride_o_token, stride_o_head, stride_o_dim,
    stride_bt_seq, stride_bt_block,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
):
    q_idx = tl.program_id(0)
    h = tl.program_id(1)
    kv_h = h // NUM_KV_GROUPS

    seq_idx = tl.load(seq_for_q_ptr + q_idx)
    kv_pos = tl.load(kv_pos_for_q_ptr + q_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_bs = tl.arange(0, BLOCK_SIZE)

    # Load the query once, promote to fp32 for math
    q = tl.load(
        Q_ptr + q_idx * stride_q_token + h * stride_q_head + offs_d * stride_q_dim
    ).to(tl.float32)

    m_i = tl.full([], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for block_iter in range(0, num_blocks):
        phys_block = tl.load(
            block_tables_ptr + seq_idx * stride_bt_seq + block_iter * stride_bt_block
        )
        block_kv_start = block_iter * BLOCK_SIZE
        within_seq = (block_kv_start + offs_bs) < seq_len
        causal = (block_kv_start + offs_bs) <= kv_pos
        valid = within_seq & causal

        # K: (BLOCK_SIZE, HEAD_DIM)
        k_ptrs = (
            K_cache_ptr
            + phys_block * stride_k_block
            + offs_bs[:, None] * stride_k_token
            + kv_h * stride_k_head
            + offs_d[None, :] * stride_k_dim
        )
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

        # scores: (BLOCK_SIZE,) = sum_d q[d] * k[bs, d] * scale
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_new = alpha * l_i + tl.sum(p, axis=0)

        v_ptrs = (
            V_cache_ptr
            + phys_block * stride_v_block
            + offs_bs[:, None] * stride_v_token
            + kv_h * stride_v_head
            + offs_d[None, :] * stride_v_dim
        )
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new

    # If l_i is 0 (no valid keys, shouldn't happen with a well-formed batch),
    # guard against division by zero.
    out = acc / tl.where(l_i > 0, l_i, 1.0)
    o_ptr = Out_ptr + q_idx * stride_o_token + h * stride_o_head + offs_d * stride_o_dim
    tl.store(o_ptr, out.to(Out_ptr.dtype.element_ty))


def _build_seq_for_q(query_lens: torch.Tensor) -> torch.Tensor:
    # Expand [3, 1, 2] -> [0, 0, 0, 1, 2, 2]
    return torch.repeat_interleave(
        torch.arange(query_lens.numel(), device=query_lens.device, dtype=torch.int32),
        query_lens,
    )


def _build_kv_pos_for_q(
    seq_lens: torch.Tensor, query_lens: torch.Tensor
) -> torch.Tensor:
    # For seq with q_len=L_q, kv_len=L_kv, the positions are
    # [L_kv - L_q, L_kv - L_q + 1, ..., L_kv - 1]
    parts = []
    seq_lens_cpu = seq_lens.tolist()
    query_lens_cpu = query_lens.tolist()
    for kv_len, q_len in zip(seq_lens_cpu, query_lens_cpu):
        if q_len == 0:
            continue
        parts.append(
            torch.arange(kv_len - q_len, kv_len, device=seq_lens.device, dtype=torch.int32)
        )
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int32, device=seq_lens.device)


def paged_attention_triton(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_lens: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    total_q, num_heads_q, head_dim = query.shape
    _, _, num_kv_heads, _ = k_cache.shape
    num_kv_groups = num_heads_q // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]

    seq_for_q = _build_seq_for_q(query_lens)
    kv_pos_for_q = _build_kv_pos_for_q(seq_lens, query_lens)

    out = torch.empty_like(query)
    grid = (total_q, num_heads_q)
    _paged_attn_kernel[grid](
        query, k_cache, v_cache, out,
        block_tables, seq_lens, seq_for_q, kv_pos_for_q,
        scale,
        query.stride(0), query.stride(1), query.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        NUM_KV_GROUPS=num_kv_groups,
        MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
        num_warps=4, num_stages=2,
    )
    return out
