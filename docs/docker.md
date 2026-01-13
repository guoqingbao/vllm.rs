# vLLM-rs Docker Image

This repository provides a Docker image for **vLLM-rs**, a high-performance inference engine for large language models (LLMs), built using Rust and optimized for NVIDIA GPUs.

The image includes:
- A **command-line interface (CLI)** tool (`vllm-rs`)
- An **OpenAI-compatible REST API server** (`vllm-rs-server`), a shell wrapper around the Python service that fronts the Rust runtime

---

## Build Options

The image is built with the following features enabled by default:

```text
cuda,nccl,graph,python,flash-attn,flash-context
```

Graph capture is enabled by default; remove `graph` if you don't want it (or if your GPU does not support it).
`flash-context` increases build time but improves long-context prefill/decoding performance.
For V100, remove `flash-attn` and `flash-context`.
For single-GPU machines, remove `nccl`.

Default build args:
- `WITH_FEATURES=cuda,nccl,graph,python,flash-attn,flash-context` (or `BUILD_FEATURES=...`)
- `CUDA_COMPUTE_CAP=80`

---

## Example Usage (Docker Run)

## Build From Dockerfile

To build this Docker image locally, choose the feature list and compute capability:

```bash
docker build -t vllm-rs:latest \
  --build-arg WITH_FEATURES=cuda,nccl,graph,python,flash-attn,flash-context \
  --build-arg CUDA_COMPUTE_CAP=89 \
  .
```

### Run with a Hugging Face Model

```bash
docker run --gpus all -p 80:80 \
  vllm-rs:latest \
  vllm-rs-server --m meta-llama/Llama-3.2-1B --host 0.0.0.0 --port 80
```

### Run CLI Inference

```bash
docker run --gpus all vllm-rs:latest \
  vllm-rs --m meta-llama/Llama-3.2-1B --max-tokens 100
```

---

## Available Tools

### 1. `vllm-rs` – Rust CLI/REST API Inference Tool

```bash
vllm-rs [OPTIONS] [HF_TOKEN] [HF_TOKEN_PATH]
```

**Usage Example:**

```bash
vllm-rs --m meta-llama/Llama-3.2-1B --max-tokens 512 --temperature 0.7 --server --port 8000
```

**Common Options:**
| Option | Description |
|--------|-------------|
| `--m <MODEL_ID>` | Hugging Face model ID |
| `--w <WEIGHT_PATH>` | Path to safetensor weights |
| `--f <WEIGHT_FILE>` | GGUF file path or name |
| `--max-tokens <MAX_TOKENS>` | Max tokens per request (default: 4096) |
| `--temperature <TEMP>` | Sampling temperature |
| `--top-p <TOP_P>` | Top-p sampling |
| `--cpu` | Run on CPU instead of GPU |
| `--d <DEVICE_IDS>` | GPU device IDs to use |
| `--context-cache` | Enable context caching for better performance |
| `--server` | Enable serving native Rust chat API instead of interactive mode |
| `--port` | Port on which to bind 0.0.0.0 serving the HTTP API |

See `vllm-rs --help` for full list of options.

---

### 2. `vllm-rs-server` – Python REST API Server

```bash
vllm-rs-server --host 0.0.0.0 --port 80 --m meta-llama/Llama-3.2-1B
```

**Endpoint:**  
`POST /v1/completions`

**Example Request:**
```bash
curl http://localhost:80/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, how are you?", "max_tokens": 100}'
```

**Common Options:**
| Option | Description |
|--------|-------------|
| `--host <HOST>` | Host to bind server to |
| `--port <PORT>` | Port to listen on |
| `--m <MODEL_ID>` | Hugging Face model ID |
| `--w <WEIGHT_PATH>` | Path to safetensor weights |
| `--f <WEIGHT_FILE>` | GGUF file path or name |
| `--dtype <DTYPE>` | Data type (f16, bf16, f32) |
| `--max-num-seqs <MAX_SEQS>` | Max concurrent sequences |
| `--context-cache` | Enable context caching |
| `--temperature <TEMP>` | Sampling temperature |

See `vllm-rs-server --help` for full list of options.