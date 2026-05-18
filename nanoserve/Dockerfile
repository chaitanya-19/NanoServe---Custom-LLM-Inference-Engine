# NanoServe container.
#
# Targets a CUDA 12.1 host (matches torch 2.4 + triton 3.0 wheels).
# On RunPod L4 instances the recommended base image is the official
# pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime; we use the devel variant so
# that triton can JIT-compile.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/nanoserve
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install hf_transfer

COPY nanoserve ./nanoserve
COPY scripts ./scripts
COPY benchmark ./benchmark
COPY profile ./profile
COPY README.md ./README.md

RUN pip install -e .

EXPOSE 8000
CMD ["python", "scripts/launch_server.py", \
     "--model", "meta-llama/Llama-3.2-3B-Instruct", \
     "--port", "8000"]
