"""Batched sampler.

Each request can have different sampling params, so the sampler accepts
parallel arrays of (temperature, top_p, top_k, ...) and applies them in a
vectorized way.  For repetition penalty we need the per-request history,
which the engine passes as a packed list of CPU ints.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..config import SamplingParams


class Sampler:
    def __init__(self, device: torch.device):
        self.device = device

    def sample(
        self,
        logits: torch.Tensor,                # (num_seqs, vocab_size)
        sampling_params_list: List[SamplingParams],
        history_token_ids: List[List[int]],  # for repetition penalty
        generators: Optional[List[Optional[torch.Generator]]] = None,
    ) -> torch.Tensor:
        num_seqs, vocab = logits.shape
        device = logits.device

        # Repetition penalty (HF formulation)
        for i, p in enumerate(sampling_params_list):
            if p.repetition_penalty != 1.0 and history_token_ids[i]:
                ids = torch.tensor(history_token_ids[i], device=device, dtype=torch.long)
                score = logits[i].index_select(0, ids)
                # If score < 0 we multiply by penalty; else divide
                score = torch.where(score < 0, score * p.repetition_penalty, score / p.repetition_penalty)
                logits[i].index_copy_(0, ids, score)

        # Temperature & greedy
        temps = torch.tensor(
            [max(p.temperature, 1e-6) if p.temperature > 0 else 1.0
             for p in sampling_params_list],
            device=device,
            dtype=logits.dtype,
        )
        greedy = torch.tensor(
            [p.temperature == 0.0 for p in sampling_params_list],
            device=device,
        )
        logits = logits / temps.unsqueeze(1)

        # Apply top-k
        for i, p in enumerate(sampling_params_list):
            if p.top_k > 0 and p.top_k < vocab:
                topk_vals, _ = torch.topk(logits[i], p.top_k)
                threshold = topk_vals[-1]
                logits[i] = torch.where(logits[i] < threshold, torch.full_like(logits[i], -float("inf")), logits[i])

        # Apply top-p (nucleus): mask everything outside the smallest
        # prefix whose cumulative probability exceeds p
        for i, p in enumerate(sampling_params_list):
            if p.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits[i], descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                # tokens to remove: cum_probs > top_p, but always keep the top-1
                remove_sorted = cum_probs > p.top_p
                remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                remove_sorted[..., 0] = False
                remove_orig = torch.zeros_like(remove_sorted)
                remove_orig.scatter_(0, sorted_idx, remove_sorted)
                logits[i] = logits[i].masked_fill(remove_orig, -float("inf"))

        # Sample
        probs = torch.softmax(logits, dim=-1)
        # Replace NaNs (can occur if a whole row was -inf) with uniform
        nan_rows = torch.isnan(probs).any(dim=-1)
        if nan_rows.any():
            probs[nan_rows] = 1.0 / vocab

        # Greedy rows: take argmax
        argmax = torch.argmax(logits, dim=-1)
        # Multinomial sample for non-greedy
        # multinomial needs a per-row generator support, which torch lacks; we
        # do one multinomial call and override greedy rows
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled = torch.where(greedy, argmax, sampled)
        return sampled
