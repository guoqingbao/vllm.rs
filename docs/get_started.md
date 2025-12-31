# Get Started

This guide walks through building and running vLLM.rs across CUDA/Metal, different model formats, multi-rank, PD Disaggregation, and OpenAI-compatible APIs. Commands assume repo root and `./run.sh` (wrapper around `cargo build/run`).

## Build & features
- **Backends**: `--features cuda[,nccl,graph,flash-attn,flash-context]` or `--features metal`. CPU-only is supported but slow.
- **Quant/accel toggles**: `--fp8-kvcache` (KV in FP8, CUDA), `--flash-context` (Ampere+, long compile), `--prefix-cache` (prefix KV reuse).
- **Python bindings**: add feature `python` when building wheels (`./build.sh --features python`).

### Build (CUDA)
```shell
# Remove `nccl` for single-gpu usage
# Remove `flash-attn,flash-context` for V100 or older hardware
./build.sh --release --features cuda,nccl,graph,flash-attn,flash-context
```

### Build (Metal)
```shell
cargo build --release --features metal
```

## Model formats
- **Safetensors (HF layout)**: `--m <hf_id>` for cached download, or `--w <local_dir>` for offline weights + configs.
- **Safetensors (HF layout) + ISQ**: in-situ quantize into GGUF with args `--isq <q4k|q2k|q6k|...>`.
- **GGUF**: `--f <gguf_file>`; no configs needed.
- **Vision-Language** (Qwen3-VL, Gemma3, Mistral3-VL): require image tokens; use `--ui-server` for uploads or send image_url/base64 in the request.


## 3) Run patterns (single host)
- **CUDA text model (chat/server)**  
  ```bash
  target/release/vllm-rs --m Qwen/Qwen2.5-7B-Instruct --max-model-len 131072 \
    --kv-fraction 0.6 --prefix-cache --ui-server
  ```
- **Metal (Mac) text model**  
  ```bash
  target/release/vllm-rs --m meta-llama/Llama-3-8b --max-model-len 32768 --ui-server
  ```
- **GGUF quantized**  
  ```bash
  target/release/vllm-rs --f /path/model-Q4_K_M.gguf --max-model-len 65536 --prefix-cache
  ```
- **Embeddings** (same server; OpenAI `/v1/embeddings`)  
  ```bash
  target/release/vllm-rs --m Qwen/Qwen2.5-7B-Instruct --prefix-cache
  # curl -d '{"input":"hello","embedding_type":"mean"}' http://localhost:8000/v1/embeddings
  ```
- **Multimodal**  
  ```bash
  # Update image in the Chat UI
  target/release/vllm-rs --m Qwen/Qwen3-VL-8B-Instruct --ui-server --prefix-cache
  ```

Common runtime knobs: `--max-model-len`, `--max-num-seqs`, `--kv-fraction` (CUDA KV share), `--cpu-mem-fold` (CPU swap ratio), `--port`, `--fp8-kvcache`, `--prefix-cache`, `--prefix-cache-max-tokens`, `--ui-server`, `--batch` (perf test).

## 4) Multi-rank (single node)
- **NCCL multi-GPU**  
  ```bash
  target/release/vllm-rs --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --max-num-seqs 2 --kv-fraction 0.5
  ```
- **Graph capture (Ampere+)**: add `--features graph,flash-context` for fastest long-context prefill/decoding (compilation time increases).

## 5) PD Disaggregation (prefill/decoding split)
- **PD server (prefill host, usually memory-rich)**  
  ```bash
  target/release/vllm-rs --pd-server --port 8000 \
    --m Qwen/Qwen3-30B-A3B-Instruct-2507 --prefix-cache
  ```
- **PD client (decode host)**  
  ```bash
  target/release/vllm-rs --server --pd-client --pd-url 0.0.0.0:8000 \
    --m Qwen/Qwen3-30B-A3B-Instruct-2507 --prefix-cache
  ```
- Same weights/config on both ends; Local IPC used automatically on same node CUDA, TCP when `--pd-url` is set. Monitor logs for transfer and swap events.

## Prefix cache
- Enable with `--prefix-cache` (CUDA/Metal). Prefix reuse is automatic; no `session_id` required.
- Use `--prefix-cache-max-tokens` to cap the cache size (rounded down to block size).
- Tune `--max-model-len`, `--kv-fraction`, `--cpu-mem-fold`; avoid overcommitting KV or cache will swap/evict.

## APIs (OpenAI-style)
- Chat: `POST /v1/chat/completions` (supports `stream=true`, images for VL models).
- Embeddings: `POST /v1/embeddings` (`embedding_type=mean|last`, `encoding_format=float|base64`).
- Models: `GET /v1/models`; Usage: `GET /v1/usage?session_id=...`.
- UI: add `--ui-server` to expose the built-in web UI on port 8001.

## Troubleshooting & tuning
- Use `--log` to view loading/progress; watch for “swap” messages (KV pressure).
- If OOM on Metal, lower `--max-model-len` and batch; on CUDA, reduce `--kv-fraction` or `--max-num-seqs`.
- For GGUF/ISQ, keep `--max-num-seqs` moderate to avoid bandwidth bottlenecks; consider `--fp8-kvcache` only on Ampere+.
