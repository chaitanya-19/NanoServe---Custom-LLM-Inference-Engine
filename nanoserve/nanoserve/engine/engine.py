"""High-level inference engine.

The engine owns the model, tokenizer, KV cache, scheduler, and sampler.  It
runs a synchronous main loop in a background thread.  External callers add
requests via ``add_request`` and consume token deltas from per-request
``asyncio.Queue`` instances; the engine bridges thread -> asyncio with
``loop.call_soon_threadsafe``.
"""
from __future__ import annotations

import asyncio
import logging
import queue as stdqueue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.cuda.nvtx as nvtx

from ..cache import BlockAllocator, allocate_kv_cache
from ..config import EngineConfig, SamplingParams
from ..model import AttentionMetadata, load_model
from ..scheduler import Request, RequestStatus, Scheduler, Sampler


log = logging.getLogger("nanoserve.engine")


@dataclass
class StreamingOutput:
    request_id: str
    delta_text: str
    token_id: int
    is_finished: bool
    finish_reason: Optional[str]
    num_prompt_tokens: int
    num_generated_tokens: int


class InferenceEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

        self.model, self.tokenizer, self.model_cfg = load_model(config)
        self.dtype = next(self.model.parameters()).dtype

        # Allocate KV cache
        self.kv_cache = allocate_kv_cache(
            self.model_cfg, config.cache, self.device, self.dtype
        )
        log.info(
            f"KV cache allocated: {config.cache.num_blocks} blocks x "
            f"{config.cache.block_size} tokens = "
            f"{config.cache.num_blocks * config.cache.block_size} token slots"
        )

        # Allocator (blocks 1..N-1 are usable; 0 is padding)
        self.allocator = BlockAllocator(config.cache.num_blocks)

        # Sampler
        self.sampler = Sampler(self.device)

        # EOS tokens (Llama 3 uses <|end_of_text|> and <|eot_id|>)
        eos_ids = []
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        for tok in ("<|eot_id|>", "<|end_of_text|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                eos_ids.append(tid)
        self.eos_token_ids = list(set(eos_ids))
        log.info(f"EOS token ids: {self.eos_token_ids}")

        # Scheduler
        self.scheduler = Scheduler(
            config.scheduler,
            block_size=config.cache.block_size,
            allocator=self.allocator,
            device=self.device,
            eos_token_ids=self.eos_token_ids,
        )

        # Threading
        self._add_q: "stdqueue.Queue[Request]" = stdqueue.Queue()
        self._abort_q: "stdqueue.Queue[str]" = stdqueue.Queue()
        self._output_qs: Dict[str, asyncio.Queue] = {}
        self._loops: Dict[str, asyncio.AbstractEventLoop] = {}
        # Per-request rolling text used for incremental detokenization
        self._reported_text_len: Dict[str, int] = {}
        self._reported_token_count: Dict[str, int] = {}

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._engine_loop, daemon=True)
            self._thread.start()
            log.info("engine started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            log.info("engine stopped")

    # ------------------------------------------------------------- requests
    def add_request(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        loop: asyncio.AbstractEventLoop,
        request_id: Optional[str] = None,
    ) -> tuple[str, asyncio.Queue]:
        request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        out_q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._output_qs[request_id] = out_q
            self._loops[request_id] = loop
            self._reported_text_len[request_id] = 0
            self._reported_token_count[request_id] = 0
        req = Request(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
        )
        self._add_q.put(req)
        return request_id, out_q

    def abort_request(self, request_id: str) -> None:
        self._abort_q.put(request_id)

    # ------------------------------------------------------------- internals
    def _push(self, request_id: str, item: StreamingOutput) -> None:
        with self._lock:
            loop = self._loops.get(request_id)
            q = self._output_qs.get(request_id)
        if loop is None or q is None:
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError:
            # Loop closed; nothing to do
            pass

    def _cleanup_request(self, request_id: str) -> None:
        with self._lock:
            self._output_qs.pop(request_id, None)
            self._loops.pop(request_id, None)
            self._reported_text_len.pop(request_id, None)
            self._reported_token_count.pop(request_id, None)

    def _decode_incremental(self, r: Request) -> str:
        """Decode all output tokens, return the delta vs last call.

        We decode the FULL output sequence every time, then take the suffix.
        This is O(num_output) per token, but correct around multi-byte chars.
        """
        # Build the slice to decode: a small lookback gives the tokenizer the
        # context it needs (Llama BPE merges may have multi-token effects).
        text = self.tokenizer.decode(
            r.output_token_ids, skip_special_tokens=True
        )
        last_len = self._reported_text_len.get(r.request_id, 0)
        if last_len > len(text):
            last_len = len(text)
        delta = text[last_len:]
        self._reported_text_len[r.request_id] = len(text)
        return delta

    @torch.inference_mode()
    def _engine_loop(self) -> None:
        log.info("engine loop running")
        idle_counter = 0
        while not self._stop.is_set():
            # Drain incoming
            while True:
                try:
                    self.scheduler.add_request(self._add_q.get_nowait())
                except stdqueue.Empty:
                    break
            while True:
                try:
                    self.scheduler.abort_request(self._abort_q.get_nowait())
                except stdqueue.Empty:
                    break

            if not self.scheduler.has_unfinished_requests():
                idle_counter += 1
                # Backoff when idle to avoid pegging a CPU
                time.sleep(min(0.01, 0.0001 * idle_counter))
                continue
            idle_counter = 0

            nvtx.range_push("scheduler.step")
            out = self.scheduler.step()
            nvtx.range_pop()
            if out is None:
                time.sleep(0.0001)
                continue

            attn_meta = AttentionMetadata(
                slot_mapping=out.slot_mapping,
                block_tables=out.block_tables,
                seq_lens=out.seq_lens,
                query_lens=out.query_lens,
                query_start_loc=out.query_start_loc,
                block_size=self.config.cache.block_size,
                use_triton=self.config.use_triton_kernel,
            )

            nvtx.range_push("model.forward")
            hidden = self.model(
                out.input_ids,
                out.positions,
                self.kv_cache.k_caches,
                self.kv_cache.v_caches,
                attn_meta,
            )
            nvtx.range_pop()

            # Compute logits only for sample positions
            sampled_ids: List[int] = []
            now = time.monotonic()
            if out.sample_indices.numel() > 0:
                nvtx.range_push("logits+sample")
                sample_hidden = hidden.index_select(0, out.sample_indices)
                logits = self.model.compute_logits(sample_hidden).float()

                sampling_params_list = [
                    out.requests[i].sampling_params
                    for i in out.sampling_request_indices
                ]
                history = [
                    out.requests[i].prompt_token_ids
                    + out.requests[i].output_token_ids
                    for i in out.sampling_request_indices
                ]
                sampled = self.sampler.sample(
                    logits, sampling_params_list, history
                )
                sampled_ids = sampled.tolist()
                nvtx.range_pop()

            nvtx.range_push("post_step")
            finished = self.scheduler.update_with_sampled(out, sampled_ids, now)

            # Stream deltas to subscribers
            for samp_idx, req_idx in enumerate(out.sampling_request_indices):
                r = out.requests[req_idx]
                tok = sampled_ids[samp_idx]
                delta_text = self._decode_incremental(r)
                self._push(
                    r.request_id,
                    StreamingOutput(
                        request_id=r.request_id,
                        delta_text=delta_text,
                        token_id=tok,
                        is_finished=(r.status == RequestStatus.FINISHED),
                        finish_reason=(
                            r.finish_reason.value if r.finish_reason else None
                        ),
                        num_prompt_tokens=r.num_prompt_tokens,
                        num_generated_tokens=r.num_output_tokens,
                    ),
                )

            # For requests finished without sampling this step (e.g. aborted),
            # also emit a final marker.
            for r in finished:
                # If we already sent a final-token marker above (because the
                # request finished due to EOS during sampling), skip.
                if r.request_id not in self._output_qs:
                    continue
                if r.finish_reason and r.finish_reason.value in ("abort",) and r.num_output_tokens == 0:
                    self._push(
                        r.request_id,
                        StreamingOutput(
                            request_id=r.request_id,
                            delta_text="",
                            token_id=-1,
                            is_finished=True,
                            finish_reason=r.finish_reason.value,
                            num_prompt_tokens=r.num_prompt_tokens,
                            num_generated_tokens=r.num_output_tokens,
                        ),
                    )
                # Defer cleanup until consumer drains the queue: we add a
                # sentinel None and the consumer cleans up on receipt.
                self._push_sentinel(r.request_id)
            nvtx.range_pop()

        log.info("engine loop exiting")

    def _push_sentinel(self, request_id: str) -> None:
        with self._lock:
            loop = self._loops.get(request_id)
            q = self._output_qs.get(request_id)
        if loop is None or q is None:
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, None)  # None = stream end
        except RuntimeError:
            pass
        # The consumer holds its own reference to ``q``; dropping ours now
        # avoids a per-request leak in long-running servers.
        self._cleanup_request(request_id)
