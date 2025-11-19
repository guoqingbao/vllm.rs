# 🚀 **vLLM.rs** – A Minimalist vLLM in Rust

A blazing-fast ⚡, lightweight **Rust** 🦀 implementation of vLLM.

---

<p align="center">
  <a href="./ReadMe.md">English</a> |
  <a href="./ReadMe-CN.md">简体中文</a> |
</p>

## ✨ Key Features

* 🔧 **Pure Rust Backend** – Absolutely **no** PyTorch required
* 🚀 **High Performance** (with **Context-cache** and **PD Disaggregation**) – Superior than Python counterparts
* 🧠 **Minimalist Core** – Core logic written in **<3000 lines** of clean Rust
* 💻 **Cross-Platform** – Supports **CUDA** (Linux/Windows) and **Metal** (macOS)
* 🤖 **Built-in Chatbot/API Server** – Native Rust server for both CUDA and Metal
* 🐍 **Lightweight Python Interface** – PyO3-powered bindings for chat completion
* 🤝 **Open for Contributions** – PRs, issues, and stars are welcome!

---
### 💬 Chat Performace

> **A100** (Single Card, 40G)

| Model | Format | Size| Decoding Speed |
|------------------|---------------|----------|------------------------|
| Llama-3.1-8B | ISQ (BF16->Q4K) | 8B | **90.19** tokens/s |
| DeepSeek-R1-Distill-Llama-8B | Q2_K | 8B | **94.47** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | **95** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | **70.38** tokens/s |
| QwQ-32B | Q4_K_M | 32B | **35.69** tokens/s |
| **Qwen3-30B-A3B** | Q4_K_M | **30B (MoE)**| **75.91** tokens/s  |

> **Metal (Apple Silicon, M4)**
  </details>

  <details>
    <summary>More details</summary>

> Models: Qwen3-0.6B (BF16), Qwen3-4B (Q4_K_M), Qwen3-8B (Q2_K)；
> Concurrent Requests: 1 - 128；
> Max Model Length: 512 - 2048；
> Max Output Tokens / Request: 512 - 2048；

| Model | Batch Size | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|--------|--------|---------|-------------|
| Qwen3-0.6B (BF16) |  128  | 63488       | 83.13s    | 763.73     |
| Qwen3-0.6B (BF16) |  32      | 15872       | 23.53s    | 674.43    |
| Qwen3-0.6B (BF16) | 1       | 456       | 9.23s    | 49.42       |
| Qwen3-4B (Q4_K_M)  | 1       | 1683       | 52.62s    | 31.98     |
| Qwen3-8B (Q2_K)  | 1       | 1300       | 80.88s    | 16.07     |

</details>

### Performance Comparison

> Model: Qwen3-0.6B (BF16); 
> Concurrent Requests: 256; 
> Max Model Length: 1024; 
> Max Output Tokens / Request: 1024

| Inference Engine | Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------|----------|------------------------|
| vLLM (RTX 4070) (Reference)          | 133,966       | 98.37    | 1361.84                |
| Nano-vLLM (RTX 4070) (Reference)      | 133,966       | 93.41    | 1434.13                |
| **vLLM.rs** (**A100**)        | 262,144       | 23.88s    | **10977.55** (**40%+ speedup**)               |
| Nano-vLLM (A100)       | 262,144       | 34.22s    |   7660.26      | 

<a href="python/ReadMe.md">Reproducible steps</a>



## 🧠 Supported Architectures

* ✅ LLaMa (LLaMa2, LLaMa3)
* ✅ Qwen (Qwen2, Qwen3)
* ✅ Qwen2 Moe
* ✅ Qwen3 Moe
* ✅ Mistral
* ✅ GLM4 (0414, **Not ChatGLM**)

Supports both **Safetensor** (including GPTQ and AWQ formats) and **GGUF** formats.


## 📘 Usage in Python

### 📦 Install with pip
   💡 1. Manual build required for CUDA compute capability < 8.0 (e.g., V100, no flash-attn support)

   💡 2. Prebuilt package has native support of `context cache` using flash attention (built with `flash-context` feature), `fp8-kvcache` feature is not compatible with `flash-context`, manual build required to use FP8 KvCache.
```shell
# You might install NCCL library for multi-gpu inference
python3 -m pip install vllm_rs fastapi uvicorn
```

### 🌐✨ API Server

💡You can use **any client compatible with the OpenAI API** to interact.

🤖 <a href="python/ReadMe.md">Here are notes on using Context-cache with clients</a>

  <details open>
    <summary>Single GPU + GGUF model + FP8 KvCache</summary>

```bash
# Each request has a default maximum output tokens (`--max-tokens`)

# Default client configuration (if the client and API Server are on the same system):
# openai.base_url = "[http://localhost:8000/v1/](http://localhost:8000/v1/)"
# openai.api_key = "EMPTY"
# `--m`: model_id, `--f`: GGUF file name
# To enable FP8 KV Cache (`--fp8-kvcache`), python package need to be built without `flash-context` feature
python -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --host 0.0.0.0 --port 8000 --max-tokens 32768 --max-model-len 128000
```

  </details>

  <details open>
    <summary>Multi-GPU + Local GGUF model</summary>

```bash
python -m vllm_rs.server --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --d 0,1 --host 0.0.0.0 --port 8000 --max-model-len 64000
```

  </details>

  <details open>
    <summary>Non-quantized to quantized models (ISQ)</summary>

```bash
# Safetensors model multi-GPU inference (also quantizes weights to Q4K format, enabling maximum context length):
python -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --d 0,1 --host 0.0.0.0 --port 8000 --max-model-len 262144 --max-num-seqs 1
```

  </details>

  <details>
    <summary>GPTQ/AWQ Marlin-compatible model</summary>

```bash
python -m vllm_rs.server --w /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin --host 0.0.0.0 --port 8000
```

  </details>

  <details>
    <summary>Multi-GPU + GGUF model + Context-Cache</summary>

When context cache is enabled, pass `session_id` in the `extra_body` field when sending requests through the OpenAI API.
`session_id` stays unchanged during one conversation; new conversations need a new `session_id`. No other settings need to be changed.

```bash
python -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --d 0,1 --host 0.0.0.0 --port 8000 --max-model-len 64000 --max-num-seqs 8 --context-cache
```

  </details>

### 🤖✨ Interactive Chat and Batch Processing

  <details open>
    <summary>Load with Huggingface model_id</summary>

```bash
python -m vllm_rs.chat --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

  </details>

  <details open>
    <summary>Run GGUF models with Huggingface Model ID</summary>

```bash
# Context-cache will be automatically enabled under chat mode
python -m vllm_rs.chat --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf

```

  </details>

  <details open>
    <summary>Unquantized to GGUF model (ISQ)</summary>

```bash
# Enable maximum context (262144 tokens), two ranks (`--d 0,1`) (`--f` for local gguf file)
python -m vllm_rs.chat --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 262144
```

  </details>

  <details>
    <summary>Batch Completion</summary>

```bash
python -m vllm_rs.completion --f /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```

```bash
python -m vllm_rs.completion --w /home/GLM-4-9B-0414 --d 0,1 --batch 8 --max-model-len 1024 --max-tokens 1024
```

  </details>

#### 🐍 Python API

  <details>
    <summary>Details</summary>

```python
from vllm_rs import Engine, EngineConfig, SamplingParams, Message
cfg = EngineConfig(weight_path="/path/Qwen3-8B-Q2_K.gguf", max_model_len=4096)
engine = Engine(cfg, "bf16")
params = SamplingParams(temperature=0.6, max_tokens=256)
prompt = engine.apply_chat_template([Message("user", "How are you?")], True)

# Synchronous batch generation
outputs = engine.generate_sync([params,params], [prompt, prompt])
print(outputs)

params.session_id = xxx  # Pass session_id to enable context cache

# Single-request streaming generation
(seq_id, prompt_length, stream) = engine.generate_stream(params, prompt)
for item in stream:
   # item.datatype == "TOKEN"
   print(item.data)
```

  </details>

## 📘 Usage (Rust)

Use `--i` to enable interactive mode 🤖, `--server` to enable service mode 🌐, `--m` to specify a Huggingface model, or `--w` for a local Safetensors model path, or `--f` for a GGUF model file:

> Chat mode

  <details open>
    <summary>Single GPU inference + built-in Context Cache</summary>

  ```bash
  cargo run --release --features cuda,nccl -- --i --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --max-model-len 262144 --context-cache
  ```

  </details>

  <details open>
    <summary>Multi-GPU inference + Flash attention</summary>

  ```bash
  # Requires using run.sh to generate a separate runner
  ./run.sh --release --features cuda,nccl,graph,flash-attn -- --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --max-model-len 100000 --server --port 8000 --context-cache
  ```

  </details>

> Multi-GPU inference server (requires run.sh to generate independent runner)

  <details open>
    <summary>Run non-quantized Qwen3-30B-A3B model with CUDA Graph (4 GPUs)</summary>

  ```bash
  ./run.sh --release --features cuda,nccl,graph,flash-attn -- --d 0,1,2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --max-model-len 100000 --max-num-seqs 4 --server --port 8000
  ```

  </details>

  <details open>
    <summary>Run quantized Qwen3-30B-A3B on multiple GPUs</summary>

  ```bash
  ./run.sh --release --features cuda,nccl,graph,flash-attn -- --server --d 0,1 --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --max-model-len 262144 --context-cache
  ```

  </details>

  <details>
    <summary>Run non-quantized Qwen3-30B-A3B as Q4K quantized model with FP8 KVCache</summary>

  ```bash
  ./run.sh --release --features cuda,nccl,flash-attn -- --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 100000 --max-num-seqs 4 --server --port 8000 --fp8-kvcache
  ```

  </details>

  <details>
    <summary>Further enable Context-Cache functionality</summary>

  Using built-in context cache, without Flash Attention, supports V100 and Metal platforms:

  ```bash
  ./run.sh --release --features cuda,nccl -- --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 100000 --max-num-seqs 4 --server --port 8000 --context-cache
  ```

  Using Flash Attention for both context-cache and decoding (requires Ampere+ hardware; long compilation time; best performance for long-text prefill):

  ```bash
  ./run.sh --release --features cuda,nccl,flash-attn,flash-context -- --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 100000 --max-num-seqs 4 --server --port 8000 --context-cache
  ```

  </details>

> MacOS/Metal platform

  <details open>
    <summary>Run Q2K quantized model</summary>

  ```bash
  cargo run --release --features metal -- --server --f /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf
  ```

  </details>

  <details>
    <summary>Run non-quantized model as Q6K quantized model with context-cache</summary>

  ```bash
  cargo run --release --features metal -- --server --w /path/Qwen3-0.6B --isq q6k --context-cache
  ```

  </details>

## 🔀 Prefill-Decode Separation (PD Separation)

  <details>
    <summary>Start PD server</summary>

  No need to specify `port`, since the server does not directly handle user requests.

  ```bash
  # Build with `flash-context` for maximum speed in long-context prefill
  ./run.sh --release --features cuda,nccl,flash-context -- --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 200000 --max-num-seqs 2 --server --pd-server
  ```

  PD server can also be started with Python (dependency: pip install vllm_rs fastapi uvicorn)

  ```bash
  python3 -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 200000 --max-num-seqs 2 --d 0,1 --pd-server
  ```

  </details>

  <details>
    <summary>Start PD client</summary>

  ```bash
  ./run.sh --release --features cuda,nccl,flash-attn -- --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 200000 --max-num-seqs 2 --server --port 8000 --pd-client
  ```

  PD client can also be started with Python:

  ```bash
  python3 -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 200000 --max-num-seqs 2 --d 2,3 --port 8000 --pd-client
  ```

  </details>

  <details>
    <summary>Multi-container / Multi-machine setup</summary>

  The PD server and client must use the same model and rank count (GPU count). They may use different *formats* of the same model (e.g., server uses unquantized Safetensor, client uses GGUF).
  If `--pd-url` is specified (e.g., server: 0.0.0.0:8100, client: server_ip:8100), the PD server/client will bind or connect to that address.
  The client will attempt to connect to the server using the given URL (Metal platform does not support LocalIPC, so pd-url is required).
  In this case, the server and client may run on different machines.
  For single machine multi-GPU, when PD server and client run in different Docker containers, Docker must be started with `--ipc=host`.

  </details>

---



## 📽️ Demo Video

Watch it in action 🎉 

<video src="https://github.com/user-attachments/assets/7fc6aa0b-78ac-4323-923f-d761dd12857f" width="1000px"></video>


## 🔨 Build Python Package from source (Optional)

> ⚠️ The first build may take time if `Flash Attention` is enabled.

> ⚠️ When enabling context caching or multi-GPU inference, you also need to compile `Runner` (using `build.sh` or `run.sh`).


### 🛠️ Prerequisites

* Install the [Rust toolchain](https://www.rust-lang.org/tools/install)
* On **macOS**, install [Xcode command line tools](https://mac.install.guide/commandlinetools/)
* For Python bindings, install [Maturin](https://github.com/PyO3/maturin)

### Building steps
1. **Install Maturin**

```bash
# install build dependencies (Linux)
sudo apt install libssl-dev pkg-config -y
pip install maturin
pip install maturin[patchelf]  # For Linux/Windows
```

2. **Build the Python package**

```bash
# Naive CUDA (single GPU only) 
maturin build --release --features cuda,python

# Naive CUDA (+CUDA Graph, experimental)
./build.sh --release --features cuda,graph,python

# CUDA (with context-cache and FP8 KV Cache, no Flash Attention) 
./build.sh --release --features cuda,nccl,python

# CUDA (+Flash Attention, only used in prefill stage) 
./build.sh --release --features cuda,nccl,flash-attn,python

# CUDA (+Flash Attention, used in both prefill and decode stage, long time to build) 
./build.sh --release --features cuda,nccl,flash-context,python

# macOS (Metal, single GPU only, with Context-cache and FP8 kvcache)
maturin build --release --features metal,python
```

3. **Install packages**

```bash
# the package you built
pip install target/wheels/vllm_rs-*-cp38-abi3-*.whl --force-reinstall
pip install fastapi uvicorn
```


## ⚙️ Command Line Arguments

| Flag        | Description                                                      |    |
| ----------- | ---------------------------------------------------------------- | -- |
| `--m`       | Hugginface Model ID                 |    |
| `--w`       | Path to Safetensors model                 |    |
| `--f`       | GGUF filename when model_id given or GGUF file path                 |    |
| `--d`       | Device ID (e.g. `--d 0`)                                         |    |
| `--max-num-seqs`   | Maximum number of concurrent requests (default: `32`, `8` on macOS)                            |    |
| `--max-tokens`     | Max tokens per response (default: `4096`, up to `max_model_len`) |    |
| `--batch`     | Only used for benchmark (this will replace `max-num-seqs` and ignore `prompts`) |    |
| `--prompts` | Prompts separated by \| |
| `--dtype`   | KV cache dtype: `bf16` (default), `f16`, or `f32`                |    |
| `--isq`   | Load unquantized model as GGUF quantized format such as `q2k`, `q4k`, etc.   |       |
| `--temperature`   | Controls randomness: lower (0.) → deterministic, higher (1.0) → creative/random.  |       |
| `--top-k`   | Limits choices to the top k highest-probability tokens. smaller k → more stable；larger k → more random   |       |
| `--top-p`   | Dynamically chooses the smallest set of tokens whose cumulative probability ≥ p. Range: 0.8 ~ 0.95   |       |
| `--presence-penalty` | Presence penalty, controls whether the model avoids reusing `tokens that have already appeared`. <br> Range [-2, 2]. Higher positive values → more likely to introduce new tokens; negative values → more likely to repeat previously used tokens | |
| `--frequency-penalty` | Frequency penalty, controls whether the model reduces the probability of `tokens that appear too often`. <br> Range [-2, 2]. Higher positive values → stronger penalty for frequently repeated tokens; negative values → encourages more repetition | |
| `--server`       | server mode used in Rust CLI, while Python use `python -m vllm.server`        |       |
| `--fp8-kvcache`       | Use FP8 KV Cache (when flash-context not enabled)                 |    |
| `--cpu-mem-fold`       | The percentage of CPU KVCache memory size compare to GPU (default 1.0, range from 0.1 to 10.0)              |    |
| `--pd-server`       | When using PD Disaggregation, specify the current instance as the PD server (this server is only used for Prefill) |    |
| `--pd-client`       | When using PD Disaggregation, specify the current instance as the PD client (this client sends long-context Prefill requests to the PD server for processing) |    |
| `--pd-url`          | When using PD Disaggregation, if specified `pd-url`, communication will occur via TCP/IP (used when the PD server and client are on different machines) |    |

## 📌 Project Status

> 🚧 **Under active development – breaking changes may occur!**


## 🛠️ Roadmap

* [x] Batched inference (Metal)
* [x] GGUF format support
* [x] FlashAttention (CUDA)
* [x] CUDA Graph
* [x] OpenAI-compatible API (streaming support)
* [x] Continuous batching
* [x] Multi-gpu inference (Safetensors, GPTQ, AWQ, GGUF)
* [x] Speedup prompt processing on Metal/macOS
* [x] Chunked Prefill
* [x] Session-based context cache (available on `CUDA` when `context-cache` enabled)
* [x] Model loading from hugginface hub
* [ ] Model loading from ModelScope (China)
* [x] Context cache for Metal/macOS
* [x] FP8 KV Cache (CUDA)
* [x] FP8 KV Cache (Metal)
* [ ] FP8 KV Cache (with Flash-Attn)
* [ ] Additional model support (GLM 4.6, Kimi K2 Thinking, etc.)
* [x] CPU KV Cache Offloading
* [x] Prefill-decode Disaggregation (CUDA)
* [x] Prefill-decode Disaggregation (Metal)
---

## 📚 References

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm

---

💡 **Like this project? Give it a ⭐ and contribute!**
