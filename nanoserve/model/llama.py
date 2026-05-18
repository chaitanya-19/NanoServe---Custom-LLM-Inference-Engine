"""Custom Llama implementation for paged-cache inference.

We implement only the inference forward pass; training-only paths
(dropout, gradient checkpointing, etc.) are deliberately omitted.

The model accepts flat token batches: a single 1-D ``input_ids`` of length
``total_tokens`` plus an ``AttentionMetadata`` that describes how those tokens
are grouped into sequences.  This matches the continuous-batching execution
model and avoids the overhead of padded 2-D tensors.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from ..config import ModelConfig
from .attention import Attention, AttentionMetadata
from .rope import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS in fp32 for numerical stability, like reference impls.
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32.to(in_dtype) * self.weight)


class LlamaMLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Fuse gate and up into a single GEMM
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_meta: AttentionMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, cos, sin, k_cache, v_cache, attn_meta)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None  # tied; we'll use embed_tokens.weight

        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
            low_freq_factor=config.rope_low_freq_factor,
            high_freq_factor=config.rope_high_freq_factor,
            original_max_pos=config.rope_original_max_pos,
            device=device,
            dtype=dtype,
        )
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        input_ids: torch.Tensor,           # (total_tokens,)
        positions: torch.Tensor,           # (total_tokens,)
        k_caches: List[torch.Tensor],      # one per layer
        v_caches: List[torch.Tensor],
        attn_meta: AttentionMetadata,
    ) -> torch.Tensor:
        with nvtx.range("embed"):
            h = self.embed_tokens(input_ids)
        cos, sin = self.rope.get(positions)

        for i, layer in enumerate(self.layers):
            with nvtx.range(f"layer_{i}"):
                h = layer(h, cos, sin, k_caches[i], v_caches[i], attn_meta)

        with nvtx.range("final_norm"):
            h = self.norm(h)
        return h  # logit computation deferred to sample-only positions

    def compute_logits(
        self,
        hidden_states: torch.Tensor,            # (num_sample_positions, hidden_size)
    ) -> torch.Tensor:
        with nvtx.range("lm_head"):
            if self.lm_head is None:
                return F.linear(hidden_states, self.embed_tokens.weight)
            return self.lm_head(hidden_states)
