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

### Performance

> Model: Qwen3-0.6B (BF16); 
> Concurrent Requests: 256; 
> Max Model Length: 1024; 
> Max Tokens / Request: 1024

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------|----------|------------------------|
| vLLM (RTX 4070) (Reference)          | 133,966       | 98.37    | 1361.84                |
| Nano-vLLM (RTX 4070) (Reference)      | 133,966       | 93.41    | 1434.13                |
| **vLLM.rs** (**A100**)        | 257,792 (prompts not counted)      | **25.21s**    | **10216.44**  (**30%+**)             |
| Nano-vLLM (A100)       | 262,144       | 34.22s    |   7660.26      | 

#### How to reproduce?
**vLLM.rs**
```shell
# w/o cuda graph, no flash attention and model warmup (final report)
cargo run --release --features cuda -- --w /home/Qwen3-0.6B --batch 256 --max-tokens 1024 --max-model-len 1024
# report
2025-07-16T10:32:32.632729Z  INFO vllm_rs: --- Performance Metrics ---
2025-07-16T10:32:32.632764Z  INFO vllm_rs: ⏱️ Prompt tokens: 4096 in 12.56s (326.17 tokens/s)
2025-07-16T10:32:32.632781Z  INFO vllm_rs: ⏱️ Decoded tokens: 257792 in 25.21s (10216.44 tokens/s)

# enable cuda graph for higher performance
cargo run --release --features cuda,graph -- --w /home/Qwen3-0.6B --batch 256 --max-tokens 1024 --max-model-len 1024
# enable cuda graph and flash attention for even higher performance (take some times to build flash-attn kernels)
cargo run --release --features cuda,flash-attn,graph -- --w /home/Qwen3-0.6B --batch 256 --max-tokens 1024 --max-model-len 1024 --flash
```


**Nano-vLLM** 

   💡 (to make a fair comparison, revise each request to maximum of 1024 output tokens instead of random 100-1024 tokens)
```shell
# with cuda graph, flash attention and model warmup
python3 bench.py
# report
Generating: 100%|██████████████████| 1/1 [00:02<00:00,  2.65s/it, Prefill=1tok/s, Decode=369tok/s]
Total: 262144tok, Time: 34.22s, Throughput: 7660.26tok/s
```

## 📦 Installation & Usage

> ⚠️ The first build may take time if Flash Attention is enabled.

### 🛠️ Prerequisites

* Install the [Rust toolchain](https://www.rust-lang.org/tools/install)
* Install **Linux** build dependencies: `sudo apt install libssl-dev pkg-config -y`
* On **macOS**, install [Xcode command line tools](https://mac.install.guide/commandlinetools/)
* For Python bindings, install [Maturin](https://github.com/PyO3/maturin)

---

## 🐍 Quick Python Example
   💡 To compile vllm.rs python whl, please refer to `API Server Mode (Python Interface)`

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

---

## 🤖✨ Interactive Mode (Rust CLI)

Run with `--i` for interactive chat and `--w` to specify model path:

```bash
# CUDA (normal context)
cargo run --release --features cuda -- --i --w /path/qwq-32b-q4_k_m.gguf

# CUDA (with CUDA Graph)
cargo run --release --features cuda,graph -- --i --w /path/qwq-32b-q4_k_m.gguf

# CUDA with Flash Attention (extra-long context, e.g., 32k tokens)
cargo run --release --features cuda,flash-attn,graph -- --i --w /path/qwq-32b-q4_k_m.gguf

# macOS (Metal)
cargo run --release --features metal -- --i --w /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf
```

---

## 🌐✨ API Server Mode (Python Interface)

1. **Install Maturin**

```bash
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

3. **Install and Setup Chat Server**

```bash
pip install target/wheels/vllm_rs-0.1.0-cp38-abi3-*.whl
pip install fastapi uvicorn
```

4. **Start OpenAI API Server**
   💡 You can use any client compatible with the OpenAI API.
```bash
# Start OpenAI API Server (default http://0.0.0.0:8000）
# openai.base_url = "http://localhost:2000/v1/"
# openai.api_key = "EMPTY"
# add `--flash` to enalbe flash attention decoding (`flash-attn` feature required for maturin build）
python example/server.py --w /path/qwq-32b-q4_k_m.gguf --host 0.0.0.0 --port 8000
```

### Other Examples:

```bash
# Interactive chat
python3 example/chat.py --i --w /path/qwq-32b-q4_k_m.gguf

# Chat completion
python3 example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```

---

### 📽️ Demo Video

Watch it in action 🎉 <video src="https://github.com/user-attachments/assets/0751471b-a0c4-45d7-acc6-99a3e91e4c91" width="70%"></video>

---

## 🧾 Completion Mode (Rust CLI)

### GGUF Models

```bash
# CUDA
cargo run --release --features cuda,graph -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"

# CUDA + Flash Attention
cargo run --release --features cuda,flash-attn,graph -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"

# Metal (macOS)
cargo run --release --features metal -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"
```

### With Python:

```bash
python example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```

### Safetensor Models (Unquantized)

```bash
# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --prompts "How are you today?"

# Metal
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "How are you today?"
```

---

## 📚 Batched Requests

Use `|` to separate prompts:

```bash
# GGUF (Rust)
cargo run --release --features cuda,graph,flash-attn -- --w /path/qwq-32b-q4_k_m.gguf --prompts "Talk about China. | Talk about America." --max-model-len 1024

# Safetensor (Rust)
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "Talk about China. | Talk about America."

# GGUF (Python)
python3 example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?" --max-model-len 1024
```

---

## 🗜️ In-Situ Quantization (GGUF Conversion)

This may take several minutes:

```bash
# macOS
cargo run --release --features metal -- --w /path/Qwen3-0.6B/ --quant q4k --prompts "How are you today?"

# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --quant q4k --prompts "How are you today?"
```

---

## 📄 Sample Output

**Single request** (Qwen3-0.6B, BF16, macOS Metal):

```bash
cargo run --features metal -- --w /path/Qwen3-0.6B/ --prompts "How are you today?"
```

```
<think>
Okay, the user asked, "How are you today?"...
</think>

Hi there! How are you today? I'm here to help you with anything! 😊 Let me know if there's anything you need!
```

---

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
| `--flash`   | Enable flash attention **decoding** (default `False`, use Paged Attention decoding), build flag `flash-attn` required   |    |
---

## 🧠 Supported Architectures

* ✅ LLaMa (LLaMa2, LLaMa3)
* ✅ Qwen (Qwen2, Qwen3)
* ✅ Mistral

Supports both **Safetensor** and **GGUF** formats.

---

## 📌 Project Status

> 🚧 **Under active development – breaking changes may occur!**

---

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

Inspired by:

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm

---

💡 **Like this project? Give it a ⭐ and contribute!**
