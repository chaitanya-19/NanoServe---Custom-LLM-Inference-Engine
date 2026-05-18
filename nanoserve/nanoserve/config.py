"""Configuration objects for model, cache, scheduler, and engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Llama 3.2 3B Instruct architecture defaults."""
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 32.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_pos: int = 8192
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass
class CacheConfig:
    """Paged KV cache configuration."""
    num_blocks: int = 4096
    block_size: int = 16


@dataclass
class SchedulerConfig:
    """Continuous-batching scheduler limits."""
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    max_model_len: int = 8192
    chunked_prefill_size: int = 512


@dataclass
class EngineConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dtype: str = "bfloat16"
    device: str = "cuda"
    use_triton_kernel: bool = False
    seed: int = 0


@dataclass
class SamplingParams:
    """Per-request sampling configuration."""
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop_token_ids: list = field(default_factory=list)
    stop: list = field(default_factory=list)
    ignore_eos: bool = False
    seed: Optional[int] = None
