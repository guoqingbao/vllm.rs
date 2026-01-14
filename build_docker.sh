#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-vllm-rs:latest}"
CHINA_MIRROR="${2:-0}"  # 0=off, 1=on

docker build -t "${IMAGE_TAG}" \
  --build-arg CUDA_VERSION=12.9.0 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg WITH_FEATURES=cuda,nccl,graph,python,flash-attn,flash-context \
  --build-arg CUDA_COMPUTE_CAP=80 \
  --build-arg CHINA_MIRROR="${CHINA_MIRROR}" \
  .

cat <<EOF

============================================================
Build finished: ${IMAGE_TAG}

China mirror mode: ${CHINA_MIRROR}
  - 0 = disabled
  - 1 = enabled (Rustup/Cargo mirrors)
  
Two commands are available in the image:

1) vLLM.rs CLI:
   docker run --rm --gpus all ${IMAGE_TAG} vllm-rs --help

2) Server shortcut (equivalent to: python3 -m vllm_rs.server):
   docker run --rm --gpus all -p 80:80 ${IMAGE_TAG} vllm-rs-server --help
============================================================

EOF
