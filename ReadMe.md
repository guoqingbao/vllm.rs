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

* ✅ LLaMa (LLaMa2, LLaMa3, IQuest-Coder)
* ✅ Qwen (Qwen2, Qwen3) (+Hardware FP8，SM90+)
* ✅ Qwen2/Qwen3 Moe (+Hardware FP8, SM90+)
* ✅ Qwen3 Next (+Hardware FP8，SM90+)
* ✅ Qwen3.5 Dense/Moe (Test TBD)
* ✅ Mistral v1, v2
* ✅ Mistral-3-VL Reasoning (3B, 8B, 14B, Multimodal model)
* ✅ GLM4 (0414, **Not ChatGLM**)
* ✅ GLM4 MoE (4.6/4.7)
* ✅ Phi3 / Phi4 (Phi-3, Phi-4, Phi-4-mini, etc.)
* ✅ Gemma3 (Multimodal model, No flash-attn support)
* ✅ Qwen3-VL (Dense, Multimodal model)
* ✅ MiroThinker-v1.5 (30B, 235B)

Supports both **Safetensor** (including GPTQ and AWQ formats) and **GGUF** formats.

All models support hardware FP8 KV-cache acceleration (requires SM90+ and disable `flash-context`).

---
## 📚 Guides
- [Get Started](docs/get_started.md)
- [Docker Build](docs/docker.md)
- [Tool Parsing](docs/tool_parsing.md)
- [MCP Integration and Tool Calling](docs/mcp_tool_calling.md)
- [Work with Claude Code](docs/claude_code.md)
- [Work with OpenCode](docs/open_code.md)
- [Work with Goose AI Agent](docs/goose.md)
- [Embedding](docs/embeddings.md)
- [Multimodal (Qwen3-VL, Gemma3, Mistral3-VL)](docs/multimodal.md)
- [Prefix cache](docs/prefix-cache.md)
- [Rust crate](docs/rust_crate.md)
- [Tokenize/Detokenize](docs/tokenize.md)
- [Performance Benchmarks](docs/performance.md)


## 📘 Usage in Python

### 📦 Install with pip
- 💡 **CUDA compute capability < 8.0** (e.g., V100) requires a **manual build**  
  (no `flash-attn` support; alternatively use **Rust mode**).
- 💡 The **prebuilt wheel** is built with the `flash-context` feature enabled.  
  To use **FP8 KV Cache**, you must **build manually** (remove the `flash-context` build flag).


> 🍎 Metal (macOS)
```shell
python3 -m pip install vllm_rs
````

> 🟩 CUDA (Linux)

#### Ampere / Ada (SM80+)
```shell
#(Optional) Install NCCL
apt-get install -y libnccl2 libnccl-dev
python3 -m pip install vllm_rs
```

#### Hopper (SM90+) / Blackwell (SM120+)

Download the wheel from the [Release Assets](https://github.com/guoqingbao/vllm.rs/releases/tag/v0.8.7), unzip it, then install the `.whl`

### 🌐✨ API Server + Built-in ChatGPT-like Web Server

💡Start with `--ui-server` will also start ChatGPT-like web server, no external chat client required in that case.

💡Use the Rust PD Server (see **PD Disaggregation**) if decoding stalls during prefilling of long-context requests.

💡Prefix cache is automatic and does not require `session_id`.

  <details open>
    <summary>Single GPU + GGUF model</summary>

```bash
# CUDA
python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --kv-fraction 0.6 --ui-server --prefix-cache
# Metal/MacOS (response can be seriously degradated on MacOS pre-Tahoe, use a smaller `--max-model-len` or `--kv-fraction` parameter)
python3 -m vllm_rs.server --m unsloth/Qwen3-4B-GGUF --f Qwen3-4B-Q4_K_M.gguf --ui-server --max-model-len 32768 --prefix-cache
```

  </details>

  <details open>
    <summary>Multi-GPU + Safetensors model</summary>

```bash
python3 -m vllm_rs.server --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --ui-server --prefix-cache
```

  </details>

  <details open>
    <summary>Unquantized load as GGUF model (ISQ)</summary>

```bash
# Load as Q4K format, other options (q2k, q3k, q5k, q6k, q8_0):
python3 -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --d 0,1 --ui-server --prefix-cache
```

  </details>

  <details open>
    <summary>FP8 Model</summary>

```bash
python3 -m vllm_rs.server --m Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --ui-server --prefix-cache
```

  </details>

  <details open>
    <summary>Multimodal model (Qwen3 VL, with images)</summary>

```bash
# Use the built-in ChatUI to upload images or refer image url (ended with '.bmp', '.gif', '.jpeg', '.png', '.tiff', or '.webp')
python3 -m vllm_rs.server --m Qwen/Qwen3-VL-8B-Instruct --ui-server --prefix-cache
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

### Install on CUDA (CUDA 11+, 12+, 13.0)

> **Option 1:** Install into Docker
   <details>

```bash
cd vllm.rs
# Use one of the following build methods

# change `sm_80` to your hardware spec, e.g., sm_75 (V100), sm_80 (A100), sm_90 (Hopper), sm_100/sm_120 (Blackwell)
./build_docker.sh "cuda,nccl,graph,flash-attn,flash-context,python" sm_80

# +cutlass feature for optimized fp8 models (Qwen3 series, sm90+) with CUDA 13
./build_docker.sh "cuda,nccl,graph,flash-attn,flash-context,cutlass,python" sm_90 13.0.0

# Pass 1 to enable rust crate mirror (Chinese Mainland)
./build_docker.sh "cuda,nccl,graph,flash-attn,flash-context,python" sm_80 12.9.0 1

# Pass `--prod` to build the production image (uses `Dockerfile.prod`)
./build_docker.sh --prod "cuda,nccl,graph,flash-attn,flash-context,cutlass,python" sm_90 13.0.0

# New（Use FlashInfer backend）
./build_docker.sh "cuda,nccl,flashinfer,graph,cutlass,python" sm_90 13.0.0
```
  </details>

See [**Run vLLM.rs in docker →**](docs/docker.md)

> **Option 2:** Manual Installation

   <details open>

Install the Rust toolchain
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install build dependencies

```sh
sudo apt-get update
sudo apt-get install -y git build-essential libssl-dev pkg-config
```

Install CUDA toolkit (optional)

```sh
# CUDA 12.9
sudo apt-get install -y \
  cuda-nvcc-12-9 \
  cuda-nvrtc-dev-12-9 \
  libcublas-dev-12-9 \
  libcurand-dev-12-9

# NCCL
sudo apt-get install -y libnccl2 libnccl-dev
```

Install vLLM.rs
```shell
# Remove `nccl` for single-gpu usage
# Remove `flash-attn,flash-context` for V100 or older hardware
# Add `cutlass` for sm90+ (fp8 models)
# Use `--dst` to change installation folder
sudo ./build.sh --install --features cuda,nccl,graph,flash-attn,flash-context

# New（Use FlashInfer backend）
./build.sh --install --features cuda,nccl,flashinfer,graph,cutlass
```
  </details>

### Install on MacOS/Metal

Install [Xcode command line tools](https://mac.install.guide/commandlinetools/)

Install with `metal` feature
```shell
cargo install --features metal
```

### Running
Use `--i` to enable interactive mode 🤖, `--ui-server` or `--server` to enable service mode 🌐, `--m` to specify a Huggingface model, or `--w` for a local Safetensors model path, or `--f` for a GGUF model file:


> API server + Web UI

  <details open>
    <summary>Single GPU</summary>

  ```bash
  # CUDA
  vllm-rs --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --prefix-cache
  # Metal/MacOS
  vllm-rs --m Qwen/Qwen3-4B-GGUF --f Qwen3-4B-Q4_K_M.gguf --ui-server --prefix-cache
  ```
  
  <details open>
    <summary>Multi-GPU + Unquantized Model</summary>

  ```bash
  # Replace "--ui-server" with "--server" will only start API server
  vllm-rs --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --ui-server --prefix-cache
  ```

  </details>

  <details open>
    <summary>Multi-GPU + GGUF Model</summary>

  ```bash
  vllm-rs --d 0,1 --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --prefix-cache
  ```

  </details>

  <details open>
    <summary>FP8 Model</summary>

```bash
# CUDA (MoE, Dense), be sure to enable `cutlass` feature on sm90+
vllm-rs --m Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --ui-server --prefix-cache
# Or Qwen3-Next 80B
./run.sh --features cuda,nccl,graph,flashinfer,cutlass,graph --release --m Qwen/Qwen3-Coder-Next-FP8 --ui-server --d 0,1 --prefix-cache
# MacOS/Metal (Dense)
vllm-rs --m Qwen/Qwen3-4B-Instruct-2507-FP8 --ui-server --prefix-cache
```

  </details>

  <details open>
    <summary>ISQ model + FP8 KvCache</summary>

  ```bash
  # CUDA: Disabled flash-context feature to use fp8-kvcache
  ./run.sh --release --features cuda,nccl,flash-attn --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --isq q4k --fp8-kvcache
  # MacOS/Metal
  vllm-rs --ui-server --w /path/Qwen3-4B --isq q6k
  ```

  </details>

---

## 🔌 MCP Integration (Tool Calling)

Enable LLMs to call external tools via Model Context Protocol.

```bash
# Start with multiple mcp servers
python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --prefix-cache --mcp-config ./mcp.json
```

See [**MCP Documentation →**](docs/mcp_tool_calling.md)

---

## 🔀 Prefill-Decode Separation (PD Disaggregation)

PD Disaggregation separates prefill (prompt processing) and decode (token generation) into separate instances. This helps avoid decoding stalls during long-context prefilling.

### Connection Modes

| Mode | URL Format | Use Case |
|------|------------|----------|
| LocalIPC (default) | No `--pd-url` | Same machine, CUDA only |
| File-based IPC | `file:///path/to/sock` | Containers with shared volume |
| Remote TCP | `tcp://host:port` or `http://host:port` | Different machines |

  <details>
    <summary>Start PD server</summary>

  No need to specify `port`, since the server does not directly handle user requests.
  The size of KvCache is controlled by `--max-model-len` and `--max-num-seqs`.

  ```bash
  # Build with `flash-context` for maximum speed in long-context prefill
  # Use unquantized model to obtain maximum prefill speed (~3000 tokens/s)
  vllm-rs --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --pd-server
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
  vllm-rs --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --ui-server --port 8000 --pd-client
  ```

  Or, start with prebuild Python package:
  ```bash
  python3 -m vllm_rs.server --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --ui-server --port 8000 --pd-client
  ```

  </details>

  <details>
    <summary>Multi-container setup with shared filesystem (file:// mode)</summary>

  When running PD server and client in different Docker containers on the same machine, use a shared volume for socket communication:

  ```bash
  # Create shared directory
  mkdir -p /tmp/pd-sockets

  # Start PD server container with shared volume
  docker run --gpus '"device=0,1"' -v /tmp/pd-sockets:/sockets ...
  target/release/vllm-rs --d 0,1 --m Qwen/... --pd-server --pd-url file:///sockets

  # Start PD client container with same shared volume
  docker run --gpus '"device=2,3"' -v /tmp/pd-sockets:/sockets ...
  target/release/vllm-rs --d 0,1 --w /path/... --pd-client --pd-url file:///sockets --ui-server --port 8000
  ```

  </details>

  <details>
    <summary>Multi-machine setup (tcp:// or http:// mode)</summary>

  The PD server and client must use the same model and rank count (GPU count). They may use different *formats* of the same model (e.g., server uses unquantized Safetensor, client uses GGUF).

  ```bash
  # On server machine (e.g., 192.168.1.100)
  target/release/vllm-rs --d 0,1 --m Qwen/... --pd-server --pd-url tcp://0.0.0.0:8100

  # On client machine
  target/release/vllm-rs --d 0,1 --w /path/... --pd-client --pd-url tcp://192.168.1.100:8100 --ui-server --port 8000
  ```

  > **Note**: Metal/macOS does not support LocalIPC, so `--pd-url` is required for PD disaggregation on macOS.

  </details>

---



## 📽️ Demo Video

Watch it in action 🎉 

<video src="https://github.com/user-attachments/assets/7fc6aa0b-78ac-4323-923f-d761dd12857f" width="1000px"></video>


## 🔨 Build Python Package from source (Optional)

> ⚠️ The first build may take time if `Flash Attention` is enabled.

> ⚠️ When enabling context caching or multi-GPU inference, you also need to compile `Runner` (using `build.sh` or `run.sh`).


### 🛠️ Prerequisites
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

# CUDA (with prefix-cache and FP8 KV Cache, no Flash Attention, compatible with V100) 
./build.sh --release --features cuda,nccl,python

# CUDA (+Flash Attention, only used in prefill stage) 
./build.sh --release --features cuda,nccl,flash-attn,python

# CUDA (+cutlass (sm90+), +Flash Attention for decoding, +high prefill throughput, long time to build) 
./build.sh --release --features cuda,nccl,flash-attn,flash-context,cutlass,python

# macOS (Metal, single GPU only, with prefix-cache and FP8 kvcache)
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
| `--cpu-mem-fold`       | The percentage of CPU KVCache memory size compare to GPU (default 0.2, range from 0.1 to 10.0)              |
| `--pd-server`       | When using PD Disaggregation, specify the current instance as the PD server (this server is only used for Prefill) |
| `--pd-client`       | When using PD Disaggregation, specify the current instance as the PD client (this client sends long-context Prefill requests to the PD server for processing) |
| `--pd-url`          | PD communication URL: `tcp://host:port` or `http://host:port` for remote TCP, `file:///path` for filesystem socket (containers), or omit for local IPC |
| `--ui-server`       |  server mode: start the API server and also start the ChatGPT-like web server |
| `--kv-fraction`       |  control kvcache usage (percentage of remaining gpu memory after model loading) |
| `--prefix-cache`   | Enable prefix caching for multi-turn conversations |
| `--prefix-cache-max-tokens`   | Cap prefix cache size in tokens (rounded down to block size) |

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
* [x] Prefix cache (available on `CUDA` when `prefix-cache` enabled)
* [x] Model loading from hugginface hub
* [ ] Model loading from ModelScope (China)
* [x] Prefix cache for Metal/macOS
* [x] FP8 KV Cache (CUDA)
* [x] FP8 KV Cache (Metal)
* [ ] FP8 KV Cache (with Flash-Attn / Flashinfer)
* [x] FP8 Models (CUDA: MoE, Dense; Metal: Dense)
* [ ] Additional model support (LLaMa 4, Kimi K2 Thinking, etc.)
* [x] CPU KV Cache Offloading
* [x] Prefill-decode Disaggregation (CUDA)
* [x] Prefill-decode Disaggregation (Metal)
* [x] Built-in ChatGPT-like Web Server
* [x] **Embedding API**
* [x] **Tokenize/Detokenize API**
* [x] **MCP Integration & Tool Calling**
* [x] **Prefix Caching**
* [x] **Claude/Anthropic-compatible API Server**
* [x] **Support CUDA 13**
* [x] **Support FlashInfer backend**
* [ ] TentorRT-LLM 
---

## 📚 References

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm

---

💡 **Like this project? Give it a ⭐ and contribute!**
