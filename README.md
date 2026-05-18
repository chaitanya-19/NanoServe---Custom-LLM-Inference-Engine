# NanoServe

A minimal-but-real LLM inference engine in PyTorch + Triton. Paged KV cache,
continuous batching with chunked prefill, OpenAI-compatible streaming API,
benchmarked against vLLM on a single L4. Built to understand and explain
the moving parts of a modern serving stack, not to replace one.

> Target model: **Llama 3.2 3B Instruct** (GQA 24/8, head_dim 128, 28 layers,
> tied embeddings, Llama 3 RoPE scaling). Other Llama-family architectures
> work without code changes if the HF config and weights are available.

---

## Why it exists

Production inference servers (vLLM, TGI, TensorRT-LLM) are excellent black
boxes. This repo unpacks the box: every primitive you would expect — block
allocator, paged attention, scheduler, sampler, streaming API — is here in
~2.5k lines of Python and Triton, with NVTX hooks and a one-command profile
flow. The goal is for someone reading the code to come away knowing exactly
where TTFT, ITL, and throughput come from in the call graph.

What's intentionally **out** of scope: tensor parallelism, quantization
(weights or KV), prefix caching, speculative decoding, LoRA adapters,
preemption/swap-out, CUDA graphs. Each of these is a known extension point
and is called out in the code where it would slot in.

## Architecture at a glance

```
┌─────────────────┐    POST /v1/chat/completions   ┌──────────────────┐
│  HTTP client    │ ─────────────────────────────► │  FastAPI server  │
│  (OpenAI SDK,   │                                │  (api/server.py) │
│   bench client) │ ◄─── SSE stream of chunks ──── │                  │
└─────────────────┘                                └────────┬─────────┘
                                                            │
                                                  add_request(prompt,
                                                              params)
                                                            │
                                                            ▼
                                              ┌──────────────────────────┐
                                              │   InferenceEngine        │
                                              │   (engine/engine.py)     │
                                              │   - bg thread main loop  │
                                              │   - call_soon_threadsafe │
                                              │     to async queues      │
                                              └──┬───────────────────────┘
                                                 │ each iteration:
                                                 ▼
                              ┌────────────────────────────────────────┐
                              │  Scheduler.step()                      │
                              │  - admit waiting if budget allows      │
                              │  - chunked prefill + decode in 1 batch │
                              │  - build flat tensors (input_ids,      │
                              │    positions, slot_mapping,            │
                              │    block_tables, seq_lens, ...)        │
                              └──┬─────────────────────────────────────┘
                                 ▼
                              ┌─────────────────────────────────────┐
                              │  LlamaForCausalLM.forward()         │
                              │  per layer:                         │
                              │    QKV proj → RoPE → write_kv →     │
                              │    paged_attention(torch|triton) →  │
                              │    o_proj → MLP                     │
                              └──┬──────────────────────────────────┘
                                 ▼
                              ┌─────────────────────────────────────┐
                              │  Sampler.sample()                   │
                              │  (only on sample_indices)           │
                              └──┬──────────────────────────────────┘
                                 ▼
                              ┌─────────────────────────────────────┐
                              │  Scheduler.update_with_sampled()    │
                              │  - append tokens, check EOS/length  │
                              │  - free blocks of finished requests │
                              └──┬──────────────────────────────────┘
                                 ▼
                              push StreamingOutput to per-request queue
```

## Repo layout

```
nanoserve/
├── nanoserve/
│   ├── config.py                       # ModelConfig, CacheConfig, ...
│   ├── model/
│   │   ├── rope.py                     # Llama 3 RoPE with freq scaling
│   │   ├── attention.py                # QKV, RoPE, KV write, kernel dispatch
│   │   ├── llama.py                    # RMSNorm + SwiGLU MLP + decoder + LM
│   │   └── loader.py                   # HF → custom weight mapping
│   ├── cache/
│   │   ├── block_allocator.py          # free-list allocator, block 0 = sentinel
│   │   └── kv_cache.py                 # per-layer K/V tensors
│   ├── kernels/
│   │   ├── paged_attention_torch.py    # per-seq SDPA, GQA via repeat_interleave
│   │   └── paged_attention_triton.py   # FlashAttention-style online softmax
│   ├── scheduler/
│   │   ├── request.py                  # Request state machine
│   │   ├── sampler.py                  # temp / top-k / top-p / rep-penalty
│   │   └── scheduler.py                # continuous batching + chunked prefill
│   ├── engine/engine.py                # main loop, thread↔asyncio bridge
│   └── api/{protocol.py,server.py}     # OpenAI schemas + FastAPI
├── scripts/launch_server.py            # entry point
├── benchmark/
│   ├── client.py                       # async load gen w/ TTFT/ITL/throughput
│   ├── compare.py                      # markdown + matplotlib plots
│   ├── prompts.json                    # 25 ShareGPT-style prompts
│   └── run_{nanoserve,vllm}.sh
├── profile/
│   ├── nsys_capture.sh                 # end-to-end NVTX trace
│   └── ncu_capture.sh                  # per-kernel counters for paged attn
├── Dockerfile
├── setup.sh
└── requirements.txt
```

## RunPod L4 quickstart

1. **Pod**: NVIDIA L4 (24 GB), template `pytorch:2.4.0-py3.10-cuda12.1.0-devel`,
   80 GB container disk, port 8000 exposed.

2. **Clone + setup** (5 min, mostly weight download):
   ```bash
   git clone <this-repo> nanoserve && cd nanoserve
   export HF_TOKEN=hf_xxx        # Llama 3.2 is gated; accept the license first
   bash setup.sh
   source venv/bin/activate
   ```

3. **Serve**:
   ```bash
   python scripts/launch_server.py \
       --model meta-llama/Llama-3.2-3B-Instruct \
       --port 8000 \
       --num-blocks 4096 \
       --block-size 16 \
       --max-num-seqs 64 \
       --max-num-batched-tokens 2048 \
       --chunked-prefill-size 512
   ```
   Add `--use-triton` to dispatch attention to the custom Triton kernel.

4. **Smoke test**:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{"model":"meta-llama/Llama-3.2-3B-Instruct",
            "messages":[{"role":"user","content":"Hello!"}],
            "max_tokens":64, "stream":true}'
   ```

5. **Benchmark vs vLLM** (vLLM in a sibling venv to avoid torch/triton
   pinning fights):
   ```bash
   # nanoserve on port 8000, vllm on port 8001
   bash benchmark/run_nanoserve.sh
   bash benchmark/run_vllm.sh
   python -m benchmark.compare \
       --inputs results_nanoserve_triton.json results_vllm.json \
       --output-md comparison.md --output-dir plots/
   ```

## Design notes

### Paged KV cache

K and V tensors per layer have shape
`(num_blocks, block_size, num_kv_heads, head_dim)` and live for the whole
process. A `BlockAllocator` hands out integer block IDs from a free list;
block 0 is reserved as a padding sentinel so we can build a uniform 2-D
`block_tables` tensor without checking sentinel reads at kernel time (the
kernel masks on `seq_len` anyway). On Llama 3.2 3B in bf16 the cache cost
per token across all 28 layers is `28 × 2 × 8 × 128 × 2 = 112 KB`; the
default 4096-block × 16-token configuration gives 64K token slots in ~7 GB,
which sits comfortably on a 24 GB L4 alongside the 6.4 GB of weights.

### Continuous batching with chunked prefill

The scheduler is iteration-level. Every step it:

1. Reserves token budget for in-flight decodes (1 token each).
2. Assigns chunked-prefill budget to in-flight prefills (up to
   `chunked_prefill_size` per request).
3. Admits waiting requests until any of `max_num_seqs`,
   `max_num_batched_tokens`, or block budget runs out.
4. Builds a single flat batch where each token carries its own
   `(seq_idx, position, slot)`. Prefill and decode share one forward pass.

This is the design that gives modern servers their throughput: a long
prefill never blocks a queue of short decodes. The "did this iter consume
the last prompt token?" check decides whether a request produces a sampled
output token at the end of its prefill chunk.

### Attention: two backends, one interface

Both backends expose:
```
paged_attention(query, k_cache, v_cache, block_tables, seq_lens,
                query_start_loc, query_lens, scale, block_size) -> attn_out
```

`paged_attention_torch` is a per-sequence SDPA loop. For each sequence we
gather its KV via `index_select`, `repeat_interleave` the KV heads to match
Q heads (GQA), and call `F.scaled_dot_product_attention` with one of three
masks: `is_causal=True` for fresh prefill, no mask for decode (q_len=1),
custom mask for mid-chunk prefill (queries are the last `q_len` positions
of a longer sequence). It is correct and small but pays O(num_seqs) Python
overhead per layer per step.

`paged_attention_triton` parallelizes over `(q_token, head)`. Each program
walks its sequence's blocks once, using online softmax to never materialize
the full attention matrix. Causal masking is per-query (each query carries
its own kv-position cutoff via a precomputed `kv_pos_for_q` tensor), so
prefill and decode share the same kernel. GQA is a constexpr divisor —
each query head reads from `head // NUM_KV_GROUPS`.

### Sampling

`Sampler.sample` is batched across requests with heterogeneous parameters.
Temperature, top-k, top-p, and repetition penalty are applied as a vector
of per-row transforms, with greedy rows (temperature == 0) overlaid on top
of the multinomial sample at the end. Rep penalty uses the
HuggingFace formulation (divide-if-positive, multiply-if-negative).

### Streaming API

The engine's main loop runs in a daemon thread; the FastAPI handler is
async. They communicate through per-request `asyncio.Queue` instances.
The engine schedules pushes onto each queue from the worker thread via
`loop.call_soon_threadsafe(queue.put_nowait, item)`. A `None` sentinel on
the queue marks end-of-stream so the SSE handler can flush `data: [DONE]`
and close cleanly. Disconnects are detected with `Request.is_disconnected`
and trigger `engine.abort_request`, which frees the request's blocks
through the scheduler.

## Profiling

NVTX ranges are emitted at three nested levels:

- engine loop: `scheduler.step`, `model.forward`, `logits+sample`, `post_step`
- model: `embed`, `layer_0`...`layer_N`, `final_norm`, `lm_head`
- (implicit) each PyTorch kernel inside the layers

Capture an end-to-end timeline:
```bash
bash profile/nsys_capture.sh           # produces nanoserve.nsys-rep
nsys-ui nanoserve.nsys-rep
```

Capture per-launch counters for the Triton paged-attention kernel:
```bash
bash profile/ncu_capture.sh            # produces nanoserve_attn.ncu-rep
ncu-ui nanoserve_attn.ncu-rep
```

Look for: (1) the relative cost of prefill vs decode iterations,
(2) the gap between `scheduler.step` and `model.forward` (host overhead),
(3) `_paged_attn_kernel`'s achieved memory throughput vs L4 peak
(~300 GB/s), and (4) whether the bf16→fp32 promotion in the kernel is
register-bound.

## What this is not

This is a learning artifact for an L4-class single-GPU workload. It will
underperform vLLM at high concurrency because vLLM has years of work in
CUDA-graphed decoding, custom CUDA paged-attention kernels (with shared-K
optimizations specific to recent GPU SMs), prefix caching, and a much more
sophisticated scheduler with preemption. Where this repo gets within ~2x
on TTFT and ITL at moderate concurrency, that's a teaching success, not
a production claim.

## Acknowledgements

- The paged-attention idea is from the vLLM paper (Kwon et al., 2023).
- The chunked-prefill scheduler design follows Sarathi-Serve (Agrawal et al., 2024).
- The Llama 3 RoPE scaling formula is from the
  [Llama 3 reference implementation](https://github.com/meta-llama/llama3).

## License

MIT.
