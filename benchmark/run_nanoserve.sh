#!/usr/bin/env bash
# Launch nanoserve, wait for it to warm up, run the load generator,
# and write results to results_nanoserve.json.
#
# Assumes you are in the repo root and have activated the venv.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${PORT:-8000}"
BACKEND="${BACKEND:-torch}"   # torch | triton
NUM_BLOCKS="${NUM_BLOCKS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_BATCHED_TOK="${MAX_BATCHED_TOK:-2048}"
CHUNK="${CHUNK:-512}"
CONCURRENCY="${CONCURRENCY:-1 4 16 32}"
REQUESTS="${REQUESTS:-64}"
MAX_TOKENS="${MAX_TOKENS:-128}"
OUTPUT="${OUTPUT:-results_nanoserve_${BACKEND}.json}"

EXTRA_ARGS=""
if [[ "$BACKEND" == "triton" ]]; then
    EXTRA_ARGS="--use-triton"
fi

echo "[run_nanoserve] launching server on port $PORT (backend=$BACKEND)"
python scripts/launch_server.py \
    --model "$MODEL" \
    --port "$PORT" \
    --num-blocks "$NUM_BLOCKS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOK" \
    --chunked-prefill-size "$CHUNK" \
    $EXTRA_ARGS &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "[run_nanoserve] waiting for /health ..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then
        echo "[run_nanoserve] server up after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -sf "http://localhost:$PORT/health" >/dev/null; then
    echo "[run_nanoserve] server never came up; aborting" >&2
    exit 1
fi

echo "[run_nanoserve] running benchmark client ..."
python -m benchmark.client \
    --url "http://localhost:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --prompts benchmark/prompts.json \
    --concurrency $CONCURRENCY \
    --requests-per-level "$REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --engine-name "nanoserve-${BACKEND}" \
    --output "$OUTPUT"

echo "[run_nanoserve] done. results -> $OUTPUT"
