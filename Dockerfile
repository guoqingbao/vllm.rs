# syntax=docker/dockerfile:1

FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        libnccl-dev=2.29.2-1+cuda12.9 \
        libnccl2=2.29.2-1+cuda12.9 \
        curl \
        libssl-dev \
        pkg-config \
        clang \
        libclang-dev \
        python3-pip && \

    rm -rf /var/lib/apt/lists/*
HEREDOC

RUN pip3 install maturin patchelf cffi

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup update nightly
RUN rustup default nightly

WORKDIR /vllm.rs

COPY . .

# Rayon threads are limited to minimize memory requirements in CI, avoiding OOM
# Rust threads are increased with a nightly feature for faster compilation (single-threaded by default)
ARG CUDA_COMPUTE_CAP=120
ARG RAYON_NUM_THREADS=64
ARG RUST_NUM_THREADS=64
ARG RUSTFLAGS="-Z threads=${RUST_NUM_THREADS}"
ARG WITH_FEATURES="cuda,nccl,python,flash-attn"
# Build both output types - server and CLI bins, avoid shell games: echo and sed
RUN ./build.sh --release --features "${WITH_FEATURES}" && cargo build --release --features $(echo $WITH_FEATURES|sed 's|,python||')

FROM docker.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04 AS base
ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80

ARG DEBIAN_FRONTEND=noninteractive

RUN <<HEREDOC
    apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        libnccl2=2.29.2-1+cuda12.9 \
        libomp-dev \
        ca-certificates \
        libssl-dev \
        curl \
        pkg-config \
        python3-pip && \

    rm -rf /var/lib/apt/lists/*
HEREDOC

RUN pip3 install fastapi uvicorn cffi

FROM base

# Server components
COPY --from=builder /vllm.rs/target/release/libvllm_rs.so /usr/lib64/libvllm_rs.so
COPY --from=builder /vllm.rs/target/release/runner /usr/local/bin/runner
RUN chmod +x /usr/local/bin/runner

COPY --from=builder /vllm.rs/target/wheels wheels
RUN pip3 install wheels/* && rm -rf wheels
RUN echo -e '#!/bin/bash\npython3 -m vllm_rs.server  "$@"' > /usr/local/bin/vllm-rs-server && chmod +x /usr/local/bin/vllm-rs-server

# CLI component
COPY --from=builder /vllm.rs/target/release/vllm-rs /usr/local/bin/vllm-rs
RUN chmod +x /usr/local/bin/vllm-rs

# Only the `devel` builder image provides symlinks, restore the `libnccl.so` symlink:
RUN ln -s libnccl.so.2 /usr/lib/$(uname -m)-linux-gnu/libnccl.so

