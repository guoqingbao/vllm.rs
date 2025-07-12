# 🚀 **vLLM.rs** – A Minimalist vLLM in Rust

A blazing-fast ⚡, lightweight **Rust** 🦀 implementation of vLLM.

---

<p align="center">
  <a href="./ReadMe.md">English</a> |
  <a href="./ReadMe-CN.md">简体中文</a> |
</p>

## ✨ Key Features

* 🔧 **Pure Rust Backend** – Absolutely **no** PyTorch required
* 🚀 **High Performance with CUDA graph** – Comparable to original vLLM (PyTorch + ATen)
* 🧠 **Minimalist Core** – Core logic written in **< 1000 lines** of clean Rust
* 💻 **Cross-Platform** – Supports **CUDA** (Linux/Windows) and **Metal** (macOS)
* 🤖 **Built-in Chatbot/API Server** – Native Rust server for both CUDA and Metal
* 🐍 **Lightweight Python Interface** – PyO3-powered bindings for chat completion
* 🤝 **Open for Contributions** – PRs, issues, and stars are welcome!

---

### Performance

Model: Qwen3-0.6B (BF16)
Concurrent Requests: 256

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------|----------|------------------------|
| vLLM (RTX 4070)            | 133,966       | 98.37    | 1361.84                |
| Nano-vLLM (RTX 4070)       | 133,966       | 93.41    | 1434.13                |
| **vLLM.rs** (**A100**)        | 25,600       | 5.23s    | **5092.50**                |
| vLLM (A100)            | -       | -    | TODO                |
| Nano-vLLM (A100)       | -       | -    | TODO                |

## 📦 Installation & Usage

> ⚠️ The first build may take time if Flash Attention is enabled.

### 🛠️ Prerequisites

* Install the [Rust toolchain](https://www.rust-lang.org/tools/install)
* On macOS, install [Xcode command line tools](https://mac.install.guide/commandlinetools/)
* For Python bindings, install [Maturin](https://github.com/PyO3/maturin)

---

## 🐍 Quick Python Example

```python
cfg = EngineConfig(model_path="/path/Qwen3-8B-Q2_K.gguf", ...)
engine = Engine(cfg, "bf16")
params = SamplingParams(temperature=0.6, max_tokens=256)
prompt = engine.apply_chat_template([Message("user", "How are you?")], True)

# Synchronous generation for batched input
outputs = engine.generate_sync(params, [prompt, prompt])
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

   💡 Specify Python version with `-i`, e.g., `-i python3.9`

```bash
# CUDA (normal context)
maturin build --release --features cuda,python

# CUDA (with CUDA Graph)
maturin build --release --features cuda,graph,python

# CUDA with Flash Attention
maturin build --release --features cuda,flash-attn,graph,python -i 3.9

# macOS (Metal)
maturin build --release --features metal,python
```

3. **Install and Setup Chat Server**

```bash
pip install target/wheels/vllm_rs-0.1.0*.whl
pip install fastapi uvicorn
```

4. **Start OpenAI API Server**

```bash
python example/server.py --w /path/qwq-32b-q4_k_m.gguf --host 0.0.0.0 --port 8000
```
💡 You can use any client compatible with the OpenAI API.

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
cargo run --release --features cuda,graph,flash-attn -- --w /path/qwq-32b-q4_k_m.gguf --prompts "Talk about China. | Talk about America."

# Safetensor (Rust)
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "Talk about China. | Talk about America."

# GGUF (Python)
python3 example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
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

## 📊 Batched Output Examples

**LLaMa3.1-8B (BF16, A100, 16 requests)**

```
8450 tokens generated in 14.28s (591.82 tokens/s)
```

**QwQ-32B GGUF Q4K (A100, 4 requests)**

```
4000 tokens in 48.23s (82.93 tokens/s)
```

---

## ⚙️ CLI Flags

| Flag        | Description                                                      |    |
| ----------- | ---------------------------------------------------------------- | -- |
| `--w`       | Path to model folder (Safetensor) or file (GGUF)                 |    |
| `--d`       | Device ID (e.g. `--d 0`)                                         |    |
| `--kvmem`   | KV cache size in MB (default: `4096`)                            |    |
| `--max`     | Max tokens per response (default: `4096`, up to `max_model_len`) |    |
| `--prompts` | Prompts, separated by \`                                         | \` |
| `--dtype`   | KV cache dtype: `bf16` (default), `f16`, or `f32`                |    |

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

---

## 📚 References

Inspired by:

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm

---

💡 **Like this project? Give it a ⭐ and contribute!**
