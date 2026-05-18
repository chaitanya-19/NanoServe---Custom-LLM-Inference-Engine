"""Load HuggingFace Llama checkpoints into the custom model.

We download the model with huggingface_hub, then map HF parameter names to
ours.  We fuse Q/K/V into one GEMM and gate/up into another, which gives a
modest speedup for free and lets us call only two matmuls per attention or
MLP block.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from ..config import EngineConfig, ModelConfig
from .llama import LlamaForCausalLM


HF_TO_NANO_LAYER = {
    # per-layer mapping; keys are HF, values are nano-side
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
}


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def _load_safetensors(model_path: Path) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    # Single- or sharded-safetensors layouts
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_map = json.load(f)["weight_map"]
        shards = sorted(set(shard_map.values()))
        for shard in shards:
            with safe_open(model_path / shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
    else:
        single = model_path / "model.safetensors"
        with safe_open(single, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    return tensors


def _hf_config_to_model_config(hf_cfg: dict) -> ModelConfig:
    rope_scaling = hf_cfg.get("rope_scaling") or {}
    return ModelConfig(
        hidden_size=hf_cfg["hidden_size"],
        intermediate_size=hf_cfg["intermediate_size"],
        num_hidden_layers=hf_cfg["num_hidden_layers"],
        num_attention_heads=hf_cfg["num_attention_heads"],
        num_key_value_heads=hf_cfg.get("num_key_value_heads", hf_cfg["num_attention_heads"]),
        head_dim=hf_cfg.get(
            "head_dim", hf_cfg["hidden_size"] // hf_cfg["num_attention_heads"]
        ),
        vocab_size=hf_cfg["vocab_size"],
        max_position_embeddings=hf_cfg.get("max_position_embeddings", 8192),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        rope_scaling_factor=float(rope_scaling.get("factor", 1.0)),
        rope_low_freq_factor=float(rope_scaling.get("low_freq_factor", 1.0)),
        rope_high_freq_factor=float(rope_scaling.get("high_freq_factor", 4.0)),
        rope_original_max_pos=int(
            rope_scaling.get("original_max_position_embeddings", 8192)
        ),
        rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=hf_cfg.get("tie_word_embeddings", False),
    )


def load_model(engine_cfg: EngineConfig):
    """Download, instantiate, and weight-load the model.  Returns (model, tokenizer, model_config)."""
    device = torch.device(engine_cfg.device)
    dtype = _resolve_dtype(engine_cfg.dtype)

    print(f"[loader] downloading {engine_cfg.model_name} ...", flush=True)
    model_path = Path(
        snapshot_download(
            repo_id=engine_cfg.model_name,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer.model",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
    )
    with open(model_path / "config.json") as f:
        hf_cfg = json.load(f)
    model_cfg = _hf_config_to_model_config(hf_cfg)
    engine_cfg.model = model_cfg

    print("[loader] instantiating model ...", flush=True)
    model = LlamaForCausalLM(model_cfg, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype)
    model.eval()

    print("[loader] reading safetensors ...", flush=True)
    hf_state = _load_safetensors(model_path)

    print("[loader] copying weights ...", flush=True)
    own_state = {}

    # embed
    own_state["embed_tokens.weight"] = hf_state["model.embed_tokens.weight"]
    # final norm
    own_state["norm.weight"] = hf_state["model.norm.weight"]

    # optional lm_head
    if not model_cfg.tie_word_embeddings and "lm_head.weight" in hf_state:
        own_state["lm_head.weight"] = hf_state["lm_head.weight"]

    # per-layer
    for i in range(model_cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        # norms
        own_state[f"layers.{i}.input_layernorm.weight"] = hf_state[
            f"{p}.input_layernorm.weight"
        ]
        own_state[f"layers.{i}.post_attention_layernorm.weight"] = hf_state[
            f"{p}.post_attention_layernorm.weight"
        ]
        # fused QKV
        q = hf_state[f"{p}.self_attn.q_proj.weight"]
        k = hf_state[f"{p}.self_attn.k_proj.weight"]
        v = hf_state[f"{p}.self_attn.v_proj.weight"]
        own_state[f"layers.{i}.self_attn.qkv_proj.weight"] = torch.cat([q, k, v], dim=0)
        # output proj
        own_state[f"layers.{i}.self_attn.o_proj.weight"] = hf_state[
            f"{p}.self_attn.o_proj.weight"
        ]
        # fused gate/up
        gate = hf_state[f"{p}.mlp.gate_proj.weight"]
        up = hf_state[f"{p}.mlp.up_proj.weight"]
        own_state[f"layers.{i}.mlp.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)
        own_state[f"layers.{i}.mlp.down_proj.weight"] = hf_state[
            f"{p}.mlp.down_proj.weight"
        ]

    # Cast & move all tensors before load_state_dict
    own_state = {
        k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        for k, v in own_state.items()
    }
    missing, unexpected = model.load_state_dict(own_state, strict=False)
    if unexpected:
        print(f"[loader] WARNING: unexpected keys: {unexpected[:5]} ...", flush=True)
    # `rope.cos`/`rope.sin` are not registered as parameters; expect them in missing.
    real_missing = [m for m in missing if not m.startswith("rope.")]
    if real_missing:
        raise RuntimeError(f"missing keys at load: {real_missing[:5]}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print(f"[loader] done. dtype={dtype} device={device}", flush=True)
    return model, tokenizer, model_cfg
