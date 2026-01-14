#!/usr/bin/env bash
set -euo pipefail

WITH_FEATURES="${1:-cuda,nccl,graph,python,flash-attn,flash-context}"
CHINA_MIRROR="${2:-0}"          # 0=off, 1=on
IMAGE_TAG="${3:-vllm-rs:latest}"

docker build --network=host -t "${IMAGE_TAG}" \
  --build-arg CUDA_VERSION=12.9.0 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg WITH_FEATURES="${WITH_FEATURES}" \
  --build-arg CUDA_COMPUTE_CAP=80 \
  --build-arg CHINA_MIRROR="${CHINA_MIRROR}" \
  .

cat <<EOF

============================================================
Build finished: ${IMAGE_TAG}

WITH_FEATURES: ${WITH_FEATURES}

China mirror mode: ${CHINA_MIRROR}
  - 0 = disabled
  - 1 = enabled (Rustup/Cargo mirrors)

Two commands are available in the image:

1) vLLM.rs CLI:
   docker run --rm --gpus all ${IMAGE_TAG} vllm-rs --help
   # Expose host file system and running ports
   docker run --rm --gpus all -v "$HOME":/workspace -p 8000:8000 -p 8001:8001 ${IMAGE_TAG} vllm-rs --m Qwen/Qwen3-0.6B --ui-server --port 8000

2) Server shortcut (equivalent to: python3 -m vllm_rs.server):
   docker run --rm --gpus all -p 80:80 ${IMAGE_TAG} vllm-rs-server --help

3) Run interactively:
   docker run --rm -it --gpus all ${IMAGE_TAG} bash
============================================================

EOF
