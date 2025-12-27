# 🚀 **vLLM.rs** – A Minimalist vLLM in Rust

A blazing-fast ⚡, lightweight **Rust** 🦀 implementation of vLLM.

---

<p align="center">
  <a href="./ReadMe.md">English</a> |
  <a href="./ReadMe-CN.md">简体中文</a>
</p>

## ✨ Key Features

* 🔧 **Pure Rust Backend** – Absolutely **no** PyTorch required
* 🚀 **High Performance** (with **Context-cache** and **PD Disaggregation**)
* 🧠 **Minimalist Core** – Core logic written in **<3000 lines** of clean Rust
* 💻 **Cross-Platform** – Supports **CUDA** (Linux/Windows) and **Metal** (macOS)
* 🤖 **Built-in API Server and ChatGPT-like Web UI** – Native Rust server for both CUDA and Metal
* 🔌 **MCP Integration** – Model Context Protocol for tool calling support
* 📊 **Embedding & Tokenizer APIs** – Full text processing support
* 🐍 **Lightweight Python Interface** – PyO3-powered bindings for chat completion

---

## 📈 Performance

### 💬 Chat Performance

> **A100** (Single Card, 40G)

| Model | Format | Size| Decoding Speed |
|------------------|---------------|----------|------------------------|
| Ministral-3-3B (Multimodal) | BF16 | 3B | **118.49** tokens/s |
| Ministral-3-3B (Multimodal) | ISQ (BF16->Q4K) | 3B | **171.92** tokens/s |
| Qwen3-VL-8B-Instruct (**Multimodal**) | Q8_0 | 8B | **105.31** tokens/s |
| Llama-3.1-8B | ISQ (BF16->Q4K) | 8B | **120.74** tokens/s |
| DeepSeek-R1-Distill-Llama-8B | Q2_K | 8B | **126.89** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | **124.87** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | **70.38** tokens/s |
| QwQ-32B | Q4_K_M | 32B | **41.36** tokens/s |
| **Qwen3-30B-A3B** | Q4_K_M | **30B (MoE)**| **97.16** tokens/s  |

> **Metal (Apple Silicon, M4)**
  <details>

| Model | Batch Size | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|--------|--------|---------|-------------|
| Qwen3-0.6B (BF16) |  128  | 63488       | 83.13s    | 763.73     |
| Qwen3-0.6B (BF16) |  32      | 15872       | 23.53s    | 674.43    |
| Qwen3-0.6B (BF16) | 1       | 456       | 9.23s    | 49.42       |
| Qwen3-4B (Q4_K_M)  | 1       | 1683       | 52.62s    | 31.98     |
| Qwen3-8B (Q2_K)  | 1       | 1300       | 80.88s    | 16.07     |

  </details>

See [**Full Performance Benchmarks →**](docs/performance.md)


## 🧠 Supported Architectures

* ✅ LLaMa (LLaMa2, LLaMa3)
* ✅ Qwen (Qwen2, Qwen3)
* ✅ Qwen2 Moe
* ✅ Qwen3 Moe
* ✅ Mistral v1, v2
* ✅ Mistral-3-VL Reasoning (3B, 8B, 14B, Multimodal model)
* ✅ GLM4 (0414, **Not ChatGLM**)
* ✅ Phi3 / Phi4 (Phi-3, Phi-4, Phi-4-mini, etc.)
* ✅ Gemma3 (Multimodal model, No flash-attn support)
* ✅ Qwen3-VL (Dense, Multimodal model)

Supports both **Safetensor** (including GPTQ and AWQ formats) and **GGUF** formats.

---
## 📚 Guides
- [Get Started](docs/get_started.md)
- [MCP Integration and Tool Calling](docs/mcp_tool_calling.md)
- [Embedding](docs/embeddings.md)
- [Multimodal (Qwen3-VL, Gemma3, Mistral3-VL)](docs/multimodal.md)
- [Context cache](docs/context-cache.md)
- [Rust crate](docs/rust_crate.md)
- [Tokenize/Detokenize](docs/tokenize.md)
- [Performance Benchmarks](docs/performance.md)


## 📘 Usage in Python

### 📦 Install with pip
   💡 1. Manual build required for CUDA compute capability < 8.0 (e.g., V100, no flash-attn support) (or use Rust mode)

   💡 2. Prebuilt package built with `flash-context` feature, manual build required to use FP8 KvCache (remove `flash-context` build flag).

```shell
# NCCL library is required for CUDA. (For single GPU usage, you may choose other prebuilt packages or Rust mode.)
python3 -m pip install vllm_rs
```

### 🌐✨ API Server + Built-in ChatGPT-like Web Server

💡Start with `--ui-server` will also start ChatGPT-like web server, no external chat client required in that case.

💡Use the Rust PD Server (see **PD Disaggregation**) if decoding stalls during prefilling of long-context requests.

💡When `context-cache` enabled, fingerprint-based session detection is enabled by default if `session_id` not provided in the client side.

 ⚠️Low quantization may cause **"Thinking Process Truncated"** for reasoning models, use BF16, Q6K/Q8_0, GPTQ/AWQ can mitigate the problem, or disable `context-cache` or disable reasoning through `thinking=False` / `enable_thinking=False` might also help.

  <details open>
    <summary>Single GPU + GGUF model</summary>

```bash
# CUDA
python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --kv-fraction 0.6 --ui-server --context-cache
# Metal/MacOS (response can be seriously degradated on MacOS pre-Tahoe, use a smaller `--max-model-len` or `--kv-fraction` parameter)
python3 -m vllm_rs.server --m unsloth/Qwen3-4B-GGUF --f Qwen3-4B-Q4_K_M.gguf --ui-server --max-model-len 32768 --context-cache
```

  </details>

  <details open>
    <summary>Multi-GPU + Safetensors model</summary>

```bash
python3 -m vllm_rs.server --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --ui-server --context-cache
```

  </details>

  <details open>
    <summary>Unquantized load as GGUF model (ISQ)</summary>

```bash
# Load as Q4K format, enable maximum context length:
python3 -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --d 0,1 --max-model-len 262144 --max-num-seqs 1 --ui-server --context-cache
```

  </details>


  <details open>
    <summary>Multimodal model (Qwen3 VL, with images)</summary>

```bash
# Use the built-in ChatUI to upload images or refer image url (ended with '.bmp', '.gif', '.jpeg', '.png', '.tiff', or '.webp')
python3 -m vllm_rs.server --m Qwen/Qwen3-VL-8B-Instruct --ui-server --context-cache
```

  </details>

  <details open>
    <summary>GPTQ/AWQ Marlin-compatible model</summary>

```bash
python3 -m vllm_rs.server --w /home/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-Marlin
```
  </details>

See [**More Python Examples →**](python/ReadMe.md)

## 📘 Usage (Rust)

Use `--i` to enable interactive mode 🤖, `--ui-server` or `--server` to enable service mode 🌐, `--m` to specify a Huggingface model, or `--w` for a local Safetensors model path, or `--f` for a GGUF model file:

### Build (CUDA 11+, 12+)
```shell
# Remove `nccl` for single-gpu usage
# Remove `flash-attn,flash-context` for V100 or older hardware
./build.sh --release --features cuda,nccl,graph,flash-attn,flash-context
```

### Build (Metal)
```shell
cargo build --release --features metal
```

> API server + Web UI

  <details open>
    <summary>Single GPU</summary>

  ```bash
  # CUDA
  target/release/vllm-rs --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --context-cache
  # Metal/MacOS
  target/release/vllm-rs --m Qwen/Qwen3-4B-GGUF --f Qwen3-4B-Q4_K_M.gguf --ui-server --context-cache
  ```
  
  <details open>
    <summary>Multi-GPU + Unquantized Model</summary>

  ```bash
  # Replace "--ui-server" with "--server" will only start API server
  target/release/vllm-rs --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --ui-server --max-model-len 128000 --max-num-seqs 2 --port 8000 --context-cache
  ```

  </details>

  <details open>
    <summary>Multi-GPU + GGUF Model</summary>

  ```bash
  target/release/vllm-rs --d 0,1 --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --max-model-len 262144 --context-cache
  ```

  </details>

  <details open>
    <summary>ISQ model + FP8 KvCache</summary>

  ```bash
  # CUDA: Disabled flash-context feature to use fp8-kvcache
  ./run.sh --release --features cuda,nccl,flash-attn --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 100000 --max-num-seqs 4 --ui-server --port 8000 --fp8-kvcache
  # MacOS/Metal
  cargo run --release --features metal -- --ui-server --w /path/Qwen3-4B --isq q6k
  ```

  </details>

---

## 🔌 MCP Integration (Tool Calling)

Enable LLMs to call external tools via Model Context Protocol.

```bash
# Start with multiple mcp servers
python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --context-cache --mcp-config ./mcp.json
```

See [**MCP Documentation →**](docs/mcp_tool_calling.md)

---

## 🔀 Prefill-Decode Separation (PD Disaggregation)

  <details>
    <summary>Start PD server</summary>
  Metal/MacOS Platform or PD Server/PD Client not within same OS, `--pd-url` is required for both server and client（e.g., 0.0.0.0:8100）

  No need to specify `port`, since the server does not directly handle user requests.
  The size of KvCache is controlled by `--max-model-len` and `--max-num-seqs`。

  ```bash
  # Build with `flash-context` for maximum speed in long-context prefill
  # Use unquantized model to obtain maximum prefill speed (~3000 tokens/s)
  target/release/vllm-rs --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --pd-server
  ```

  Or, use prebuilt Python package as PD server:
  ```bash
  python3 -m vllm_rs.server --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --pd-server
  ```
  </details>

  <details>
    <summary>Start PD client</summary>

  ```bash
  # Client can use different format of the same model
  # Use Q4K to obtain higher decoding speed for small batches
  target/release/vllm-rs --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --ui-server --port 8000 --pd-client
  ```

  Or, start with prebuild Python package:
  ```bash
  python3 -m vllm_rs.server --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --ui-server --port 8000 --pd-client
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

# CUDA (with context-cache and FP8 KV Cache, no Flash Attention, compatible with V100) 
./build.sh --release --features cuda,nccl,python

# CUDA (+Flash Attention, only used in prefill stage) 
./build.sh --release --features cuda,nccl,flash-attn,python

# CUDA (+Flash Attention for decoding, +high prefill throughput, long time to build) 
./build.sh --release --features cuda,nccl,flash-context,python

# macOS (Metal, single GPU only, with Context-cache and FP8 kvcache)
maturin build --release --features metal,python
```

3. **Install packages**

```bash
# the package you built
pip install target/wheels/vllm_rs-*-cp38-abi3-*.whl --force-reinstall
```


## ⚙️ Command Line Arguments

| Flag        | Description                                                      |
| ----------- | ---------------------------------------------------------------- |
| `--m`       | Hugginface Model ID                 |
| `--w`       | Path to Safetensors model                 |
| `--f`       | GGUF filename when model_id given or GGUF file path                 |
| `--d`       | Device ID (e.g. `--d 0`)                                         |
| `--max-num-seqs`   | Maximum number of concurrent requests (default: `32`, `8` on macOS)                            |
| `--max-tokens`     | Max tokens per response (default: `4096`, up to `max_model_len`) |
| `--batch`     | Only used for benchmark (this will replace `max-num-seqs` and ignore `prompts`) |
| `--prompts` | Prompts separated by \| |
| `--dtype`   | KV cache dtype: `bf16` (default), `f16`, or `f32`                |
| `--isq`   | Load unquantized model as GGUF quantized format such as `q2k`, `q4k`, etc.   |
| `--temperature`   | Controls randomness: lower (0.) → deterministic, higher (1.0) → creative/random.  |
| `--top-k`   | Limits choices to the top k highest-probability tokens. smaller k → more stable；larger k → more random   |
| `--top-p`   | Dynamically chooses the smallest set of tokens whose cumulative probability ≥ p. Range: 0.8 ~ 0.95   |
| `--presence-penalty` | Presence penalty, controls whether the model avoids reusing `tokens that have already appeared`. <br> Range [-2, 2]. Higher positive values → more likely to introduce new tokens; negative values → more likely to repeat previously used tokens |
| `--frequency-penalty` | Frequency penalty, controls whether the model reduces the probability of `tokens that appear too often`. <br> Range [-2, 2]. Higher positive values → stronger penalty for frequently repeated tokens; negative values → encourages more repetition |
| `--server`       | server mode used in Rust CLI, while Python use `python -m vllm.server`        |
| `--fp8-kvcache`       | Use FP8 KV Cache (when flash-context not enabled)                 |
| `--cpu-mem-fold`       | The percentage of CPU KVCache memory size compare to GPU (default 0.5, range from 0.1 to 10.0)              |
| `--pd-server`       | When using PD Disaggregation, specify the current instance as the PD server (this server is only used for Prefill) |
| `--pd-client`       | When using PD Disaggregation, specify the current instance as the PD client (this client sends long-context Prefill requests to the PD server for processing) |
| `--pd-url`          | When using PD Disaggregation, if specified `pd-url`, communication will occur via TCP/IP (used when the PD server and client are on different machines) |
| `--ui-server`       |  server mode: start the API server and also start the ChatGPT-like web server |
| `--kv-fraction`       |  control kvcache usage (percentage of remaining gpu memory after model loading) |
| `--context-cache`   | Enable context caching for multi-turn conversations |

### MCP Configuration

| Flag | Description |
|------|-------------|
| `--mcp-command` | Path to single MCP server executable |
| `--mcp-args` | Comma-separated arguments for MCP server |
| `--mcp-config` | Path to JSON config file for multiple MCP servers |

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
* [x] Built-in ChatGPT-like Web Server
* [x] **Embedding API**
* [x] **Tokenize/Detokenize API**
* [x] **MCP Integration & Tool Calling**
---

## 📚 References

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm

---

💡 **Like this project? Give it a ⭐ and contribute!**
