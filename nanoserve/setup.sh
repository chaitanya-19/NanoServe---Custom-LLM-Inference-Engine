#!/usr/bin/env bash
# One-shot setup for a RunPod L4 (or any CUDA 12.1+ host).
#
# What it does:
#   1. apt-installs build deps
#   2. creates a Python 3.10 venv at ./venv
#   3. pip-installs requirements
#   4. exports HF_TOKEN if you've set it (Llama 3.2 needs a license-accepted token)
#   5. pre-downloads the model so the first launch is fast
#
# After this script finishes:
#     source venv/bin/activate
#     python scripts/launch_server.py --model meta-llama/Llama-3.2-3B-Instruct

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
SKIP_DL="${SKIP_DL:-0}"

if ! command -v python3 >/dev/null; then
    echo "[setup] installing python ..."
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-venv python3-pip
fi

if [[ ! -d venv ]]; then
    echo "[setup] creating venv ..."
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo "[setup] upgrading pip ..."
pip install --upgrade pip wheel

echo "[setup] installing requirements ..."
pip install -r requirements.txt
pip install hf_transfer

echo "[setup] installing nanoserve (editable) ..."
pip install -e .

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[setup] WARNING: HF_TOKEN is not set. Llama 3.2 is gated; export it before pre-downloading:"
    echo "          export HF_TOKEN=hf_..."
fi

if [[ "$SKIP_DL" == "0" && -n "${HF_TOKEN:-}" ]]; then
    echo "[setup] pre-downloading model: $MODEL"
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    python - <<EOF
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "$MODEL",
    token=os.environ.get("HF_TOKEN"),
    allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
)
print("[setup] model cached.")
EOF
fi

echo
echo "[setup] done. To start the server:"
echo "  source venv/bin/activate"
echo "  python scripts/launch_server.py --model $MODEL"
