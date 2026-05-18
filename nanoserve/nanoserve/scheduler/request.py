"""Per-request state."""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import SamplingParams


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"  # has been admitted, still has prompt tokens to consume
    DECODING = "decoding"      # all prompt consumed, generating output
    FINISHED = "finished"


class FinishReason(enum.Enum):
    EOS = "stop"
    LENGTH = "length"
    STOP_TOKEN = "stop"
    STOP_STRING = "stop"
    ABORT = "abort"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)

    # Runtime state (set by scheduler/engine)
    block_table: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0   # prompt+output tokens already passed through the model
    output_token_ids: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: Optional[FinishReason] = None

    # Streaming book-keeping
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    inter_token_latencies: List[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """How many tokens this request *has* committed (prompt + emitted)."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_prompt_tokens_remaining(self) -> int:
        return max(0, self.num_prompt_tokens - self.num_computed_tokens)

    @property
    def is_prompt_done(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def blocks_needed(self, block_size: int) -> int:
        """Total *logical* blocks to hold all committed tokens so far."""
        return max(1, (self.total_tokens + block_size - 1) // block_size)

    def get_token_id_at(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        return self.output_token_ids[idx - self.num_prompt_tokens]
