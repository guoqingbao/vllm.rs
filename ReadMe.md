# 🚀 **vLLM.rs** – A Minimalist vLLM in Rust

A blazing-fast ⚡, lightweight **Rust** 🦀 implementation of vLLM.

---

<p align="center">
  <a href="./ReadMe.md">English</a> |
  <a href="./ReadMe-CN.md">简体中文</a> |
</p>

## ✨ Key Features

* 🔧 **Pure Rust Backend** – Absolutely **no** PyTorch required
* 🚀 **High Performance** (with **session-based context cache**) – Superior than Python counterparts
* 🧠 **Minimalist Core** – Core logic written in **< 2000 lines** of clean Rust
* 💻 **Cross-Platform** – Supports **CUDA** (Linux/Windows) and **Metal** (macOS)
* 🤖 **Built-in Chatbot/API Server** – Native Rust server for both CUDA and Metal
* 🐍 **Lightweight Python Interface** – PyO3-powered bindings for chat completion
* 🤝 **Open for Contributions** – PRs, issues, and stars are welcome!

---
### Chat Performace

> **A100** (Single Card, 40G)

| Model | Format | Size| Decoding Speed |
|------------------|---------------|----------|------------------------|
| Llama-3.1-8B | ISQ (BF16->Q4K) | 8B | **90.19** tokens/s |
| DeepSeek-R1-Distill-Llama-8B | Q2_K | 8B | **94.47** tokens/s |
| DeepSeek-R1-0528-Qwen3-8B | Q4_K_M | 8B | **95** tokens/s |
| GLM-4-9B-0414 | Q4_K_M | 9B | **70.38** tokens/s |
| QwQ-32B | Q4_K_M | 32B | **35.69** tokens/s |
| **Qwen3-30B-A3B** | Q4_K_M | **30B (MoE)**| **75.91** tokens/s  |

#### Performance of vLLM.rs on **Metal (Apple Silicon, M4)**
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


## 📽️ Demo Video

Watch it in action 🎉 <video src="https://github.com/user-attachments/assets/7fc6aa0b-78ac-4323-923f-d761dd12857f" width="1000px"></video>


## 📦 Install with pip
   💡 1. Manual build required for CUDA compute capability < 8.0 (e.g., V100)

   💡 2. Prebuilt package has native `context cache` feature without relying on flash attention, manual build required to use `flash-context` feature.
```shell
python3 -m pip install vllm_rs
```

## 📘 Usage in Python

### 🌐✨ API Server Mode
   💡 You can use any client compatible with the OpenAI API.

   🤖 <a href="python/ReadMe.md">Here is the client usage of context cache</a>

```bash
# install server dependency
pip install fastapi uvicorn
# Start OpenAI API Server (default http://0.0.0.0:8000）
# openai.base_url = "http://localhost:8000/v1/"
# openai.api_key = "EMPTY"

# Local gguf file (`--f`), max output tokens for each request (`--max-tokens`), FP8 KV Cache (`--fp8-kvcache`, slight accuracy degradation)
python -m vllm_rs.server --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --host 0.0.0.0 --port 8000 --max-tokens 32768 --max-model-len 128000 --fp8-kvcache

# Use model weights from huggingface (`--m`: model_id, `--f`: gguf file)
python -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --host 0.0.0.0 --port 8000

# Multi-GPU (`--d`)
python -m vllm_rs.server --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --d 0,1 --host 0.0.0.0 --port 8000 --max-model-len 64000

# Multi-GPU for safetensors model: local safetensors model (`--w`) with in-situ quant to Q4K during model loading (enable maximum context length)
python -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --host 0.0.0.0 --port 8000 --isq q4k --max-model-len 262144 --max-num-seqs 1

# multi-GPU inference + context caching for GGUF model (to cache context, you need to include a `session_id` in the `extra_body` field when making a request through the OpenAI API. The session_id should remain the same throughout a conversation, and a new `session_id` should be used for a new conversation, unsed session cache will be cleared. No need to change other settings of the API).
python -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --d 0,1 --host 0.0.0.0 --port 8000 --max-model-len 64000 --max-num-seqs 8 --context-cache
```


### Interactive Chat and completion

```bash
# Interactive chat
# Load with model id
python -m vllm_rs.chat --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --fp8-kvcache

# local gguf file on second device (device order 1，`--d 1`)
python -m vllm_rs.chat --d 1 --f /path/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf

# Load unquantized safetensors model as GGUF quantized (e.g., q4k), with maximum model context length
python -m vllm_rs.chat --d 0 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 262144 --max-num-seqs 1 --max-tokens 16384

# Enable context cache for fast response (CUDA)
python -m vllm_rs.chat --d 0,1 --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --max-model-len 262144 --max-num-seqs 1 --context-cache

# ISQ q4k (macOS/Metal recommended, optional `--context-cache`)
python -m vllm_rs.chat --w /path/Qwen3-0.6B --isq q4k

# Chat completion
python -m vllm_rs.completion --f /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"

# Chat completion (Multi-GPU, CUDA)
python -m vllm_rs.completion --w /home/GLM-4-9B-0414 --d 0,1 --batch 8 --max-model-len 1024 --max-tokens 1024
```

### 🐍 Python API

```python
from vllm_rs import Engine, EngineConfig, SamplingParams, Message
cfg = EngineConfig(weight_path="/path/Qwen3-8B-Q2_K.gguf", max_model_len=4096)
engine = Engine(cfg, "bf16")
params = SamplingParams(temperature=0.6, max_tokens=256)
prompt = engine.apply_chat_template([Message("user", "How are you?")], True)

# Synchronous generation for batched input
outputs = engine.generate_sync([params,params], [prompt, prompt])
print(outputs)

params.session_id = xxx # pass session to use context cache
# Streaming generation for single request
(seq_id, prompt_length, stream) = engine.generate_stream(params, prompt)
for item in stream:
    # item.datatype == "TOKEN"
    print(item.data)
```

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

## 📘 Usage in Rust
### 🤖✨ Rust CLI Mode

Run with `--i` for interactive chat and `--w` to specify safetensors model path, or `--f` load local gguf file:

```bash
# Naive CUDA (single card only, optional `--fp8-kvcache`)
cargo run --release --features cuda,nccl -- --i --d 0 --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --max-model-len 262144 --context-cache

# Multi-GPU CUDA (+Flash Attention, this scirpt help build the runner)
./run.sh --release --features cuda,nccl,flash-attn -- --i --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 262144 --context-cache

# Multi-GPU CUDA (unquantized models)
./run.sh --release --features cuda,nccl,flash-attn -- --d 0,1,2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --max-model-len 100000 --max-num-seqs 4 --server --port 8000

# Multi-GPU server mode (with `--fp8-kvcache` or `--context-cache`)
./run.sh --release --features cuda,nccl,flash-attn -- --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 100000 --max-num-seqs 4 --server --port 8000 --fp8-kvcache

# Multi-GPU server mode (with `--context-cache`, Flash Attention used in both prefill/decode, long time to build)
./run.sh --release --features cuda,nccl,flash-context -- --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 100000 --max-num-seqs 4 --server --port 8000 --context-cache

# Naive CUDA (+CUDA Graph, experimental)
cargo run --release --features cuda,graph -- --i --f /path/qwq-32b-q4_k_m.gguf --presence-penalty 1.2 --frequency-penalty 1.2

# macOS (Metal)
cargo run --release --features metal -- --i --f /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf

#macOS (Metal, ISQ)
cargo run --release --features metal -- --i --w /path/Qwen3-0.6B --isq q4k --context-cache
```


Safetensor Models (Unquantized)

```bash
# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --prompts "How are you today?"

# Metal
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "How are you today?"

# Multi-GPUs (interactive mode)
./run.sh --release --features cuda,nccl -- --w /home/GLM-4-9B-0414 --d 0,1 --i --max-tokens 1024 --max-model-len 1024

# Multi-GPUs (server mode)
./run.sh --release --features cuda,nccl -- --w /home/GLM-4-9B-0414 --d 0,1 --max-tokens 1024 --max-model-len 1024 --server

# Multi-GPUs with Context Cache (interactive mode)
./run.sh --release --features cuda,nccl,flash-attn -- --w /home/GLM-4-9B-0414 --d 0,1 --i --max-tokens 1024 --max-model-len 1024 --context-cache
```

## Prefill-decode Disaggregation
```shell
# Start the PD server (not `port` required since it does not directly respond request(s))
# Rust
./run.sh --release --features cuda,nccl,flash-attn -- --d 0,1 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 200000 --max-num-seqs 2 --server --pd-server

# Python
python3 -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 200000 --max-num-seqs 2 --d 0,1 --pd-server


# Start the corresponding PD client (with exactly same args except device ids and pd mode)
# Rust
./run.sh --release --features cuda,nccl,flash-attn -- --d 2,3 --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 200000 --max-num-seqs 2 --server --port 8000 --pd-client

# Python
python3 -m vllm_rs.server --w /path/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 200000 --max-num-seqs 2 --d 2,3 --port 8000 --pd-client

# If `--pd-url` (e.g., 192.168.0.10:8888) is provided, the PD server will try to bind to the given address,
# and the client will attempt to connect to the server using the specified URL.
# In this scenario, the server and client can be located on different machines.
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

## 🗜️ In-Situ Quantization (GGUF Conversion during loading)

   💡 Run any unquantized models as GGUF quantized format, but it may takes few minutes for `--isq` other than q4k and q8_0.



```bash
# macOS
cargo run --release --features metal -- --w /path/Qwen3-0.6B/ --isq q4k --prompts "How are you today?"

# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --isq q4k --prompts "How are you today?"
```


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
* [ ] Additional model support
* [x] CPU KV Cache Offloading
* [x] Prefill-decode Disaggregation
---

## 📚 References

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm

---

💡 **Like this project? Give it a ⭐ and contribute!**
