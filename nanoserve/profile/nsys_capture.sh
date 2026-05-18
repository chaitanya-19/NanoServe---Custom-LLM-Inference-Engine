#!/usr/bin/env bash
# Capture an end-to-end Nsight Systems trace of nanoserve serving a small
# benchmark.  Use `nsys-ui nanoserve.nsys-rep` to open it.
#
# The model forward and scheduler step are annotated with NVTX ranges
# (see nanoserve/engine/engine.py and nanoserve/model/llama.py).

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${PORT:-8000}"
OUTPUT="${OUTPUT:-nanoserve}"
BACKEND="${BACKEND:-triton}"

EXTRA_ARGS=""
if [[ "$BACKEND" == "triton" ]]; then
    EXTRA_ARGS="--use-triton"
fi

echo "[nsys_capture] starting nanoserve under nsys ..."
nsys profile \
    --trace=cuda,nvtx,osrt,cublas \
    --cuda-memory-usage=true \
    --output="${OUTPUT}" \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python scripts/launch_server.py \
        --model "$MODEL" \
        --port "$PORT" \
        --max-num-seqs 32 \
        --max-num-batched-tokens 1024 \
        $EXTRA_ARGS &
SERVER_PID=$!
trap 'kill -SIGINT $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "[nsys_capture] waiting for /health ..."
for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then
        break
    fi
    sleep 1
done

echo "[nsys_capture] running a short load to fill the trace ..."
python -m benchmark.client \
    --url "http://localhost:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --prompts benchmark/prompts.json \
    --concurrency 8 \
    --requests-per-level 16 \
    --max-tokens 64 \
    --warmup-requests 2 \
    --engine-name "nanoserve-trace" \
    --output /tmp/trace_run.json

echo "[nsys_capture] stopping server (will flush trace) ..."
kill -SIGINT $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo "[nsys_capture] trace written to ${OUTPUT}.nsys-rep"
echo "                Open with:  nsys-ui ${OUTPUT}.nsys-rep"
