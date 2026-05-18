#!/usr/bin/env bash
# Capture per-launch metrics for the Triton paged-attention kernel with
# Nsight Compute.  ncu replays each kernel launch many times to gather
# detailed counters, so we keep the workload small (a handful of decode
# iterations is enough).
#
# Output:  nanoserve_attn.ncu-rep    (open with `ncu-ui`)

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${PORT:-8000}"
OUTPUT="${OUTPUT:-nanoserve_attn}"

echo "[ncu_capture] launching server (Triton backend) ..."
python scripts/launch_server.py \
    --model "$MODEL" \
    --port "$PORT" \
    --use-triton \
    --max-num-seqs 16 \
    --max-num-batched-tokens 512 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then break; fi
    sleep 1
done

echo "[ncu_capture] sending a few requests to warm up Triton autotuner ..."
python -m benchmark.client \
    --url "http://localhost:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --prompts benchmark/prompts.json \
    --concurrency 1 \
    --requests-per-level 2 \
    --max-tokens 16 \
    --warmup-requests 0 \
    --engine-name warmup \
    --output /tmp/ncu_warmup.json

# Now attach ncu to the server process.  We filter to the paged-attention
# kernel only to keep capture time reasonable.
echo "[ncu_capture] attaching ncu to PID $SERVER_PID ..."
ncu --target-processes all \
    --set full \
    --kernel-name regex:_paged_attn_kernel \
    --launch-skip 0 \
    --launch-count 8 \
    --export "$OUTPUT" \
    --force-overwrite \
    --replay-mode kernel \
    --attach "$SERVER_PID" &
NCU_PID=$!

# Fire load while ncu is attached.
python -m benchmark.client \
    --url "http://localhost:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --prompts benchmark/prompts.json \
    --concurrency 4 \
    --requests-per-level 8 \
    --max-tokens 32 \
    --warmup-requests 0 \
    --engine-name ncu_run \
    --output /tmp/ncu_run.json

wait $NCU_PID 2>/dev/null || true

echo "[ncu_capture] report written to ${OUTPUT}.ncu-rep"
echo "                Open with:  ncu-ui ${OUTPUT}.ncu-rep"
