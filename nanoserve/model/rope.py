"""Rotary positional embeddings with Llama 3 frequency scaling.

The Llama 3 scaling formula extends the base RoPE so that the model trained
on 8K context generalizes to 128K. Low frequencies (long-wavelength dims) are
divided by `factor`, high frequencies are left untouched, and there is a
smooth interpolation band in between.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def compute_llama3_inv_freq(
    head_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_pos: int,
    device: torch.device,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    low_freq_wavelen = original_max_pos / low_freq_factor
    high_freq_wavelen = original_max_pos / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # Scale low-frequency (long-wavelength) components
    inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    # Smooth interpolation in the medium-frequency band
    smooth = (original_max_pos / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * (inv_freq / factor) + smooth * inv_freq
    is_medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_scaled)


class RotaryEmbedding:
    """Precomputed cos/sin caches keyed by token position."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_pos: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.head_dim = head_dim
        inv_freq = compute_llama3_inv_freq(
            head_dim,
            base,
            scaling_factor,
            low_freq_factor,
            high_freq_factor,
            original_max_pos,
            device,
        )
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.cos = freqs.cos().to(dtype)  # (max_pos, head_dim/2)
        self.sin = freqs.sin().to(dtype)

    def get(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[positions], self.sin[positions]


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to (num_tokens, num_heads, head_dim).

    cos/sin are (num_tokens, head_dim/2).  We use the standard
    "interleaved-half" convention used by HuggingFace Llama.
    """
    # Split into halves: x = [x1 | x2], rotate as (x1*cos - x2*sin, x2*cos + x1*sin)
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)  # (num_tokens, 1, head_dim/2)
    sin = sin.unsqueeze(1)
    rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rot
