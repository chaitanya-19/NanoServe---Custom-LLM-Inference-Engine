"""Async load generator for any OpenAI-compatible chat-completions endpoint.

Measures, per request:
  - TTFT (time to first content token in the SSE stream)
  - ITL (inter-token latencies)
  - end-to-end latency
  - output token count

Aggregates across requests: throughput (tokens/sec), requests/sec, P50/P95/P99.

Usage:
    python -m benchmark.client \\
        --url http://localhost:8000/v1/chat/completions \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --prompts benchmark/prompts.json \\
        --concurrency 1 4 16 32 \\
        --max-tokens 128 \\
        --output results_nanoserve.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


@dataclass
class RequestResult:
    prompt_idx: int
    success: bool
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    output_tokens: int = 0
    itls_s: List[float] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ConcurrencyResult:
    concurrency: int
    num_requests: int
    num_success: int
    duration_s: float
    total_output_tokens: int

    # latency percentiles
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    e2e_p50: float
    e2e_p95: float
    e2e_p99: float
    itl_p50: float
    itl_p95: float
    itl_p99: float

    requests_per_sec: float
    output_tokens_per_sec: float
    raw_requests: List[Dict[str, Any]]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * p
    f = int(k)
    c = min(f + 1, len(sv) - 1)
    if f == c:
        return sv[f]
    return sv[f] + (sv[c] - sv[f]) * (k - f)


async def _run_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    prompt_idx: int,
) -> RequestResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    t0 = time.monotonic()
    ttft: Optional[float] = None
    itls: List[float] = []
    prev_t: Optional[float] = None
    output_tokens = 0
    try:
        async with session.post(
            url, json=body, headers={"Accept": "text/event-stream"}
        ) as resp:
            if resp.status != 200:
                return RequestResult(
                    prompt_idx=prompt_idx,
                    success=False,
                    error=f"HTTP {resp.status}: {await resp.text()}",
                )
            async for line in resp.content:
                if not line:
                    continue
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta is None or delta == "":
                    # Role-only opener or empty heartbeat; don't count as a token
                    if choices[0].get("finish_reason"):
                        break
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                    prev_t = now
                else:
                    itls.append(now - prev_t)
                    prev_t = now
                output_tokens += 1
                if choices[0].get("finish_reason"):
                    break
        e2e = time.monotonic() - t0
        return RequestResult(
            prompt_idx=prompt_idx,
            success=True,
            ttft_s=ttft if ttft is not None else 0.0,
            e2e_s=e2e,
            output_tokens=output_tokens,
            itls_s=itls,
        )
    except Exception as e:
        return RequestResult(
            prompt_idx=prompt_idx, success=False, error=repr(e)
        )


async def _run_concurrency(
    url: str,
    model: str,
    prompts: List[str],
    concurrency: int,
    requests_per_level: int,
    max_tokens: int,
    temperature: float,
) -> ConcurrencyResult:
    sem = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []

    async def _wrapped(i: int):
        async with sem:
            prompt = prompts[i % len(prompts)]
            return await _run_one(
                session, url, model, prompt, max_tokens, temperature, i
            )

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 32))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        t0 = time.monotonic()
        tasks = [asyncio.create_task(_wrapped(i)) for i in range(requests_per_level)]
        for t in asyncio.as_completed(tasks):
            r = await t
            results.append(r)
        duration = time.monotonic() - t0

    ok = [r for r in results if r.success]
    ttfts = [r.ttft_s for r in ok if r.ttft_s > 0]
    e2es = [r.e2e_s for r in ok]
    all_itls = [v for r in ok for v in r.itls_s]
    total_out = sum(r.output_tokens for r in ok)

    return ConcurrencyResult(
        concurrency=concurrency,
        num_requests=len(results),
        num_success=len(ok),
        duration_s=duration,
        total_output_tokens=total_out,
        ttft_p50=_percentile(ttfts, 0.50),
        ttft_p95=_percentile(ttfts, 0.95),
        ttft_p99=_percentile(ttfts, 0.99),
        e2e_p50=_percentile(e2es, 0.50),
        e2e_p95=_percentile(e2es, 0.95),
        e2e_p99=_percentile(e2es, 0.99),
        itl_p50=_percentile(all_itls, 0.50),
        itl_p95=_percentile(all_itls, 0.95),
        itl_p99=_percentile(all_itls, 0.99),
        requests_per_sec=len(ok) / duration if duration > 0 else 0.0,
        output_tokens_per_sec=total_out / duration if duration > 0 else 0.0,
        raw_requests=[asdict(r) for r in results],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("nanoserve-bench")
    p.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", default="benchmark/prompts.json")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--requests-per-level", type=int, default=64,
                   help="number of requests to send at each concurrency level")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy, makes the benchmark deterministic")
    p.add_argument("--warmup-requests", type=int, default=4)
    p.add_argument("--output", default="results.json")
    p.add_argument("--engine-name", default="nanoserve",
                   help="label written to the output JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.prompts) as f:
        prompts = json.load(f)
    print(f"loaded {len(prompts)} prompts")

    async def _main():
        # Warmup at concurrency 1
        if args.warmup_requests > 0:
            print(f"warmup: {args.warmup_requests} requests ...")
            await _run_concurrency(
                args.url, args.model, prompts,
                concurrency=1,
                requests_per_level=args.warmup_requests,
                max_tokens=min(32, args.max_tokens),
                temperature=args.temperature,
            )

        all_results = []
        for c in args.concurrency:
            print(f"\n=== concurrency = {c} ===")
            r = await _run_concurrency(
                args.url, args.model, prompts,
                concurrency=c,
                requests_per_level=args.requests_per_level,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(
                f"  ok={r.num_success}/{r.num_requests}  "
                f"duration={r.duration_s:.2f}s  "
                f"req/s={r.requests_per_sec:.2f}  "
                f"out_tok/s={r.output_tokens_per_sec:.1f}\n"
                f"  TTFT  P50={r.ttft_p50*1000:.0f}ms "
                f"P95={r.ttft_p95*1000:.0f}ms "
                f"P99={r.ttft_p99*1000:.0f}ms\n"
                f"  ITL   P50={r.itl_p50*1000:.1f}ms "
                f"P95={r.itl_p95*1000:.1f}ms "
                f"P99={r.itl_p99*1000:.1f}ms\n"
                f"  E2E   P50={r.e2e_p50:.2f}s "
                f"P95={r.e2e_p95:.2f}s "
                f"P99={r.e2e_p99:.2f}s"
            )
            all_results.append(asdict(r))

        Path(args.output).write_text(
            json.dumps(
                {"engine_name": args.engine_name, "model": args.model, "results": all_results},
                indent=2,
            )
        )
        print(f"\nwrote {args.output}")

    asyncio.run(_main())


if __name__ == "__main__":
    main()
