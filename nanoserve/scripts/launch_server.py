"""Launch the NanoServe HTTP server.

Usage:
    python scripts/launch_server.py \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --port 8000 \\
        --num-blocks 4096 \\
        --block-size 16 \\
        --max-num-seqs 64 \\
        --max-num-batched-tokens 2048 \\
        --chunked-prefill-size 512
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from nanoserve.api.server import create_app
from nanoserve.config import CacheConfig, EngineConfig, SchedulerConfig
from nanoserve.engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("nanoserve")
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-blocks", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--chunked-prefill-size", type=int, default=512)
    p.add_argument("--use-triton", action="store_true",
                   help="dispatch attention to the custom Triton kernel "
                        "(default: pure PyTorch backend)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="info")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        stream=sys.stderr,
    )

    config = EngineConfig(
        model_name=args.model,
        cache=CacheConfig(num_blocks=args.num_blocks, block_size=args.block_size),
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            chunked_prefill_size=args.chunked_prefill_size,
        ),
        dtype=args.dtype,
        device=args.device,
        use_triton_kernel=args.use_triton,
        seed=args.seed,
    )

    engine = InferenceEngine(config)
    engine.start()

    app = create_app(engine, config)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
