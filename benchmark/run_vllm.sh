#!/usr/bin/env bash
# Launch vLLM with matched config, run the same load generator,
# and write results to results_vllm.json.
#
# Assumes vllm is installed in the active environment.
# We run vllm in a sibling venv to avoid torch/triton conflicts; see README.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOK="${MAX_NUM_BATCHED_TOK:-2048}"
CONCURRENCY="${CONCURRENCY:-1 4 16 32}"
REQUESTS="${REQUESTS:-64}"
MAX_TOKENS="${MAX_TOKENS:-128}"
OUTPUT="${OUTPUT:-results_vllm.json}"

echo "[run_vllm] launching vLLM server on port $PORT"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOK" \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --disable-log-requests &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "[run_vllm] waiting for /health ..."
for i in $(seq 1 300); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then
        echo "[run_vllm] server up after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -sf "http://localhost:$PORT/health" >/dev/null; then
    echo "[run_vllm] server never came up; aborting" >&2
    exit 1
fi

echo "[run_vllm] running benchmark client ..."
python -m benchmark.client \
    --url "http://localhost:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --prompts benchmark/prompts.json \
    --concurrency $CONCURRENCY \
    --requests-per-level "$REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --engine-name "vllm" \
    --output "$OUTPUT"

echo "[run_vllm] done. results -> $OUTPUT"
