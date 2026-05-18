"""Iteration-level continuous-batching scheduler.

Each call to ``step()`` decides which sequences run this iteration and how
many tokens each one contributes.  The scheduler supports:

  * Mid-decode admission: new requests join the in-flight batch as long as
    there is sequence-, token-, and block-budget.
  * Chunked prefill: a long prompt is fed in pieces of at most
    ``chunked_prefill_size`` tokens so we never block decodes behind a
    multi-thousand-token prefill.
  * Mixed batches: prefilling and decoding sequences share one forward pass.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch

from ..cache import BlockAllocator
from ..config import SchedulerConfig
from .request import FinishReason, Request, RequestStatus


@dataclass
class SchedulerOutput:
    """Everything the engine needs to run one forward pass."""

    # Tensors on device
    input_ids: torch.Tensor          # (total_q,)
    positions: torch.Tensor          # (total_q,)
    slot_mapping: torch.Tensor       # (total_q,)
    block_tables: torch.Tensor       # (num_seqs, max_blocks_per_seq)
    seq_lens: torch.Tensor           # (num_seqs,)
    query_lens: torch.Tensor         # (num_seqs,)
    query_start_loc: torch.Tensor    # (num_seqs + 1,)
    sample_indices: torch.Tensor     # indices in [0, total_q) of tokens to sample from

    # Bookkeeping for the engine
    requests: List[Request]                    # one per seq, in batch order
    sampling_request_indices: List[int]        # indices in `requests` that sample this iter


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        block_size: int,
        allocator: BlockAllocator,
        device: torch.device,
        eos_token_ids: List[int],
    ):
        self.config = config
        self.block_size = block_size
        self.allocator = allocator
        self.device = device
        self.eos_token_ids = set(eos_token_ids)

        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        # Requests that finished this step; engine picks them up
        self.finished_this_step: List[Request] = []

    # ------------------------------------------------------------------ public
    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def abort_request(self, request_id: str) -> None:
        for r in list(self.waiting):
            if r.request_id == request_id:
                self.waiting.remove(r)
                r.status = RequestStatus.FINISHED
                r.finish_reason = FinishReason.ABORT
                self.finished_this_step.append(r)
                return
        for r in self.running:
            if r.request_id == request_id:
                r.status = RequestStatus.FINISHED
                r.finish_reason = FinishReason.ABORT
                return

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def num_waiting(self) -> int:
        return len(self.waiting)

    def num_running(self) -> int:
        return len(self.running)

    # ---------------------------------------------------------------- step()
    def step(self) -> Optional[SchedulerOutput]:
        self.finished_this_step = []

        # ---- 1) Plan token contributions ---------------------------------
        # decoding sequences contribute exactly 1 token each
        decoding_running = [r for r in self.running if r.is_prompt_done]
        prefilling_running = [r for r in self.running if not r.is_prompt_done]

        token_budget = self.config.max_num_batched_tokens
        token_budget -= len(decoding_running)
        if token_budget < 0:
            # Too many decoders for budget; trim to fit by dropping nothing
            # (decoders always run; the budget is a soft target). Reset to 0.
            token_budget = 0

        per_req_chunk: Dict[str, int] = {}
        for r in prefilling_running:
            remaining = r.num_prompt_tokens_remaining
            chunk = min(remaining, self.config.chunked_prefill_size, token_budget)
            if chunk <= 0:
                continue
            per_req_chunk[r.request_id] = chunk
            token_budget -= chunk

        # ---- 2) Try to admit waiting requests ----------------------------
        while (
            self.waiting
            and len(self.running) < self.config.max_num_seqs
            and token_budget > 0
        ):
            req = self.waiting[0]
            if req.num_prompt_tokens > self.config.max_model_len:
                self.waiting.popleft()
                req.status = RequestStatus.FINISHED
                req.finish_reason = FinishReason.LENGTH
                self.finished_this_step.append(req)
                continue
            prompt_blocks = max(
                1,
                (req.num_prompt_tokens + self.block_size - 1) // self.block_size,
            )
            if not self.allocator.can_allocate(prompt_blocks):
                break
            chunk = min(
                req.num_prompt_tokens,
                self.config.chunked_prefill_size,
                token_budget,
            )
            if chunk <= 0:
                break
            self.waiting.popleft()
            req.block_table = self.allocator.allocate(prompt_blocks)
            req.status = RequestStatus.PREFILLING
            self.running.append(req)
            per_req_chunk[req.request_id] = chunk
            token_budget -= chunk

        if not self.running:
            return None

        # ---- 3) Determine q_len and tokens for each running request ------
        batch_entries: List[dict] = []
        sampling_req_indices: List[int] = []
        for r in self.running:
            if r.status == RequestStatus.FINISHED:
                continue

            if r.is_prompt_done:
                # Decode: feed last sampled token
                token_id = r.output_token_ids[-1]
                q_tokens = [token_id]
                q_positions = [r.num_computed_tokens]
                q_len = 1

                # Ensure we have a block for this new token
                blocks_needed = (r.num_computed_tokens + 1 + self.block_size - 1) // self.block_size
                while blocks_needed > len(r.block_table):
                    if not self.allocator.can_allocate(1):
                        r.status = RequestStatus.FINISHED
                        r.finish_reason = FinishReason.LENGTH
                        self.finished_this_step.append(r)
                        break
                    r.block_table.extend(self.allocator.allocate(1))
                if r.status == RequestStatus.FINISHED:
                    continue
                samples_this_iter = True
            else:
                q_len = per_req_chunk.get(r.request_id, 0)
                if q_len <= 0:
                    continue
                q_tokens = r.prompt_token_ids[
                    r.num_computed_tokens : r.num_computed_tokens + q_len
                ]
                q_positions = list(
                    range(r.num_computed_tokens, r.num_computed_tokens + q_len)
                )
                # Sample only if this chunk completes the prompt
                samples_this_iter = (r.num_computed_tokens + q_len) >= r.num_prompt_tokens

            batch_entries.append(
                dict(
                    request=r,
                    tokens=q_tokens,
                    positions=q_positions,
                    q_len=q_len,
                    samples=samples_this_iter,
                )
            )

        # Drop finished
        self.running = [r for r in self.running if r.status != RequestStatus.FINISHED]
        if not batch_entries:
            return None

        # ---- 4) Build flat tensors ---------------------------------------
        all_tokens: List[int] = []
        all_positions: List[int] = []
        all_slots: List[int] = []
        query_lens: List[int] = []
        seq_lens: List[int] = []
        request_list: List[Request] = []

        max_blocks_per_seq = 0
        for e in batch_entries:
            r: Request = e["request"]
            all_tokens.extend(e["tokens"])
            all_positions.extend(e["positions"])
            query_lens.append(e["q_len"])
            seq_lens.append(r.num_computed_tokens + e["q_len"])
            request_list.append(r)
            for pos in e["positions"]:
                blk_idx = pos // self.block_size
                blk_off = pos % self.block_size
                phys = r.block_table[blk_idx]
                all_slots.append(phys * self.block_size + blk_off)
            max_blocks_per_seq = max(max_blocks_per_seq, len(r.block_table))

        # Pad block tables
        block_tables = [
            r.block_table + [self.allocator.padding_block] * (max_blocks_per_seq - len(r.block_table))
            for r in request_list
        ]

        query_start_loc = [0]
        for q in query_lens:
            query_start_loc.append(query_start_loc[-1] + q)

        # sample_indices: indices in flat q tensor of tokens we want logits from
        sample_indices: List[int] = []
        for idx, e in enumerate(batch_entries):
            if e["samples"]:
                # The position of this seq's last q token in the flat tensor
                sample_indices.append(query_start_loc[idx] + e["q_len"] - 1)
                sampling_req_indices.append(idx)

        device = self.device

        return SchedulerOutput(
            input_ids=torch.tensor(all_tokens, dtype=torch.long, device=device),
            positions=torch.tensor(all_positions, dtype=torch.long, device=device),
            slot_mapping=torch.tensor(all_slots, dtype=torch.long, device=device),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device=device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            query_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
            sample_indices=torch.tensor(sample_indices, dtype=torch.long, device=device),
            requests=request_list,
            sampling_request_indices=sampling_req_indices,
        )

    # ----------------------------------------------------- post-iter update
    def update_with_sampled(
        self,
        output: SchedulerOutput,
        sampled_token_ids: List[int],
        now: float,
    ) -> List[Request]:
        """Append sampled tokens, check stop conditions, free finished blocks.

        Returns the list of requests that just *finished* this step.
        """
        # First: advance num_computed for every batched request by its q_len
        query_lens = output.query_lens.tolist()
        for i, r in enumerate(output.requests):
            r.num_computed_tokens += query_lens[i]

        # Append sampled tokens to the sampling requests
        finished: List[Request] = []
        for samp_idx, req_idx in enumerate(output.sampling_request_indices):
            r = output.requests[req_idx]
            tok = sampled_token_ids[samp_idx]
            r.output_token_ids.append(tok)
            r.last_token_time = now
            if r.first_token_time is None:
                r.first_token_time = now
            else:
                # ITLs are deltas between consecutive output tokens
                if len(r.inter_token_latencies) >= 0 and len(r.output_token_ids) >= 2:
                    # the previous last_token_time was the "true" previous time
                    pass  # handled by engine if needed via last_token_time

            # Stop conditions
            if not r.sampling_params.ignore_eos and tok in self.eos_token_ids:
                r.status = RequestStatus.FINISHED
                r.finish_reason = FinishReason.EOS
            elif tok in r.sampling_params.stop_token_ids:
                r.status = RequestStatus.FINISHED
                r.finish_reason = FinishReason.STOP_TOKEN
            elif r.num_output_tokens >= r.sampling_params.max_tokens:
                r.status = RequestStatus.FINISHED
                r.finish_reason = FinishReason.LENGTH
            elif r.total_tokens >= self.config.max_model_len:
                r.status = RequestStatus.FINISHED
                r.finish_reason = FinishReason.LENGTH
            else:
                if r.status == RequestStatus.PREFILLING and r.is_prompt_done:
                    r.status = RequestStatus.DECODING

            if r.status == RequestStatus.FINISHED:
                finished.append(r)

        # Free blocks for finished requests
        for r in finished:
            self.allocator.free(r.block_table)
            r.block_table = []

        # Combine with finished_this_step from step() (e.g. OOM, aborts, length>max)
        all_finished = finished + self.finished_this_step
        # Remove finished from running
        finished_ids = {r.request_id for r in all_finished}
        self.running = [r for r in self.running if r.request_id not in finished_ids]
        self.finished_this_step = []
        return all_finished
