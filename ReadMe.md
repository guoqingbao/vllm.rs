# 🚀 **vLLM.rs** – A Minimalist vLLM in Rust

A blazing-fast ⚡, lightweight **Rust** 🦀 implementation of vLLM.

---

<p align="center">
  <a href="./ReadMe.md">English</a> |
  <a href="./ReadMe-CN.md">简体中文</a> |
</p>

## ✨ Key Features

* 🔧 **Pure Rust Backend** – Absolutely **no** PyTorch required
* 🚀 **High Performance** – Superior than vLLM and Nano-vLLM
* 🧠 **Minimalist Core** – Core logic written in **< 1000 lines** of clean Rust
* 💻 **Cross-Platform** – Supports **CUDA** (Linux/Windows) and **Metal** (macOS)
* 🤖 **Built-in Chatbot/API Server** – Native Rust server for both CUDA and Metal
* 🐍 **Lightweight Python Interface** – PyO3-powered bindings for chat completion
* 🤝 **Open for Contributions** – PRs, issues, and stars are welcome!

---

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

#### How to reproduce?
**vLLM.rs**
```shell
pip install vllm-rs
python example/completion.py --w /home/Qwen3-0.6B/ --batch 256 --max-tokens 1024 --max-model-len 1024

# Log
Allocating 8192 KV blocks (28672 MB) for [256 seqs x 1024 tokens]
Maximum batched tokens 262144 (8192 blocks x Block_Size 32 for KV cache).
Start inference with 256 prompts
--- Performance Metrics ---
⏱️ Prompt tokens: 4096 in 0.28s (14894.55 tokens/s)
⏱️ Decoded tokens: 258048 in 23.60s (10944.62 tokens/s)
```


**Nano-vLLM** 

   💡 To ensure a fair comparison, revise each request to have a maximum of 1024 output tokens, instead of a random number between 100 and 1024.
```shell
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
# with cuda graph, flash attention and model warmup
python3 bench.py
# log
Generating: 100%|██████████████████| 1/1 [00:02<00:00,  2.65s/it, Prefill=1tok/s, Decode=369tok/s]
Total: 262144tok, Time: 34.22s, Throughput: 7660.26tok/s
```

### Performance of vLLM.rs on **Metal (Apple Silicon, M4)**
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


## 📦 Install with pip

```shell
# flash-attn built-in for prefilling
pip install vllm-rs
```


## 🔨 Build from source

> ⚠️ The first build may take time if `Flash Attention` is enabled.

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
# CUDA (normal context)
maturin build --release --features cuda,python

# CUDA (with CUDA Graph)
maturin build --release --features cuda,graph,python

# CUDA with Flash Attention
maturin build --release --features cuda,flash-attn,graph,python

# macOS (Metal)
maturin build --release --features metal,python
```

3. **Install packages**

```bash
# the package you built
pip install target/wheels/vllm_rs-*-cp38-abi3-*.whl --force-reinstall
pip install fastapi uvicorn
```

## 📘 Usage

### 🐍 Quick Python Example

```python
from vllm_rs import Engine, EngineConfig, SamplingParams, Message
cfg = EngineConfig(model_path="/path/Qwen3-8B-Q2_K.gguf", max_model_len=4096)
engine = Engine(cfg, "bf16")
params = SamplingParams(temperature=0.6, max_tokens=256)
prompt = engine.apply_chat_template([Message("user", "How are you?")], True)

# Synchronous generation for batched input
outputs = engine.generate_sync([params,params], [prompt, prompt])
print(outputs)

# Streaming generation for single request
stream = engine.generate_stream(params, prompt)
for token in stream:
    print(token)
```


### 🌐✨ API Server Mode (Python Interface)
   💡 You can use any client compatible with the OpenAI API.

```bash
# Start OpenAI API Server (default http://0.0.0.0:8000）
# openai.base_url = "http://localhost:8000/v1/"
# openai.api_key = "EMPTY"
python example/server.py --w /path/qwq-32b-q4_k_m.gguf --host 0.0.0.0 --port 8000
```

### Interactive Chat and completion (Python)

```bash
# Interactive chat
python3 example/chat.py --i --w /path/qwq-32b-q4_k_m.gguf

# Use the second device (device order 1，`--d 1`)
python3 example/chat.py --i --d 1 --w /path/GLM-4-9B-0414-Q4_K_M.gguf

# Chat completion
python3 example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```


### 🤖✨ Rust CLI Mode

Run with `--i` for interactive chat and `--w` to specify model path:

```bash
# CUDA (normal context)
cargo run --release --features cuda -- --i --w /path/qwq-32b-q4_k_m.gguf

# Use the third device (device order 2，`--d 2`)
cargo run --release --features cuda -- --i --d 2 --w /path/GLM-4-9B-0414-Q4_K_M.gguf

# CUDA (with CUDA Graph)
cargo run --release --features cuda,graph -- --i --w /path/qwq-32b-q4_k_m.gguf

# CUDA with Flash Attention (extra-long context, e.g., 32k tokens)
cargo run --release --features cuda,flash-attn,graph -- --i --w /path/qwq-32b-q4_k_m.gguf

# macOS (Metal)
cargo run --release --features metal -- --i --w /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf
```


Safetensor Models (Unquantized)

```bash
# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --prompts "How are you today?"

# Metal
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "How are you today?"
```

## ⚙️ CLI Flags

| Flag        | Description                                                      |    |
| ----------- | ---------------------------------------------------------------- | -- |
| `--w`       | Path to model folder (Safetensor) or file (GGUF)                 |    |
| `--d`       | Device ID (e.g. `--d 0`)                                         |    |
| `--max-num-seqs`   | Maximum number of concurrent requests (default: `32`, `8` on macOS)                            |    |
| `--max-tokens`     | Max tokens per response (default: `4096`, up to `max_model_len`) |    |
| `--batch`     | Only used for benchmark (this will replace `max-num-seqs` and ignore `prompts`) |    |
| `--prompts` | Prompts separated by \| |
| `--dtype`   | KV cache dtype: `bf16` (default), `f16`, or `f32`                |    |


## 📽️ Demo Video

Watch it in action 🎉 <video src="https://github.com/user-attachments/assets/0751471b-a0c4-45d7-acc6-99a3e91e4c91" width="70%"></video>


## 🗜️ In-Situ Quantization (GGUF Conversion)

This may take several minutes:

```bash
# macOS
cargo run --release --features metal -- --w /path/Qwen3-0.6B/ --quant q4k --prompts "How are you today?"

# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --quant q4k --prompts "How are you today?"
```


## 🧠 Supported Architectures

* ✅ LLaMa (LLaMa2, LLaMa3)
* ✅ Qwen (Qwen2, Qwen3)
* ✅ Mistral
* ✅ GLM4 (0414, **Not ChatGLM**)

Supports both **Safetensor** and **GGUF** formats.


## 📌 Project Status

> 🚧 **Under active development – breaking changes may occur!**


## 🛠️ Roadmap

* [x] Batched inference (Metal)
* [x] GGUF format support
* [x] FlashAttention (CUDA)
* [x] CUDA Graph
* [x] OpenAI-compatible API (streaming support)
* [x] Continuous batching
* [ ] Multi-rank inference
* [ ] Additional model support
* [ ] Speedup prompt processing on Metal/macOS
---

## 📚 References

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm

---

💡 **Like this project? Give it a ⭐ and contribute!**
