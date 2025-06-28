# 🚀 **vLLM.rs** – A Minimalist vLLM in Rust

A blazing-fast ⚡, lightweight **Rust** 🦀 implementation of vLLM.

---

## ✨ Key Features

* 🔧 **Pure Rust** – Absolutely **no** PyTorch or Python required
* 🚀 **High Performance** – On par with the original vLLM (PyTorch + ATen)
* 🧠 **Minimalist Core** – Core logic in **< 1000 lines** of clean Rust code
* 💻 **Cross-Platform** – Works on both **CUDA** (Linux/Windows) and **Metal** (macOS)
* 🤖 **Built-in Chatbot** – Built-in Rust Chatbot work with **CUDA** and **Metal**
* 🤝 **Open for Contributions** – PRs, issues, and stars are welcome!

---

## 📦 Usage

Make sure you have the [Rust toolchain](https://www.rust-lang.org/tools/install) installed.

Mac OS Platform (Metal) requires installation of [XCode command line tools](https://mac.install.guide/commandlinetools/).

---

### 🔥 CUDA (Linux/Windows) and 🍎 Metal (macOS)

⚠️ First run may take a while on CUDA (if flash attention enabled).

---

### 🤖✨ Interactive Mode

Simply run the program with `--i` parameter:

```bash
# 🔥 CUDA
cargo run --release --features cuda -- --i --w /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf

# 🔥 CUDA with 🚀 Flash Attention (build takes longer time)
cargo run --release --features cuda,flash-attn -- --i --w /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf

# 🍎 Metal (macOS)
cargo run --release --features metal -- --i --w /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf

```


### 📽️ Demo Video

Watch a quick demo of how it works! 🎉

<video src="https://github.com/user-attachments/assets/0751471b-a0c4-45d7-acc6-99a3e91e4c91" width="70%"></video>


### 🧾✨ Completion Mode

#### GGUF model:

```bash
# 🔥 CUDA
cargo run --release --features cuda -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"

# 🔥 CUDA with 🚀 Flash Attention (build takes longer time)
cargo run --release --features cuda,flash-attn -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"

# 🍎 Metal (macOS)
cargo run --release --features cuda -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"
```

#### Safetensor model:

```bash

# 🔥 CUDA
cargo run --release --features cuda -- --w /path/Qwen3-8B/ --prompts "How are you today?"

# 🍎 Metal (macOS)
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "How are you today?"

```

---

### 📚 Batched Requests

Prompts are separated by `|`

```bash
# GGUF model
cargo run --release --features cuda -- --w /path/qwq-32b-q4_k_m.gguf --prompts "Please talk about China. | Please talk about America."

# Safetensor model
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "Please talk about China. | Please talk about America."
```

---

### 🗜️ In-situ Quantization (GGUF format conversion)

Takes a few minutes for quantization.

```bash
# macOS
cargo run --release --features metal -- --w /path/Qwen3-0.6B/ --quant q4k --prompts "How are you today?"

# CUDA
cargo run --release --features cuda -- --w /path/Qwen3-8B/ --quant q4k --prompts "How are you today?"
```

---

## 📄 Sample Output

**Single request** with Qwen3-0.6B (BF16) on macOS/Metal:

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

### 📊 Batched Results (Examples)

**LLaMa3.1-8B** BF16 (16 requests on A100):

```bash
8450 tokens generated in 14.28 s (decoding throughput: 591.82 tokens/s)
```

**QwQ-32B** GGUF Q4K (4 requests on A100):

```
4000 tokens in 48.23s (avg throughput: 82.93 tokens/s)
```

---

## ⚙️ Command-Line Parameters

| Flag        | Description                                       |    |
| ----------- | ------------------------------------------------- | -- |
| `--w`       | Path to model folder (Safetensor) or file (GGUF)  |    |
| `--d`       | Device ID (e.g. `--d "0"`)                        |    |
| `--kvmem`   | KV cache size in MB (default: `4096`)               |    |
| `--max`   | Maximum number of tokens in each chat response (default: `4096`, up to `max_model_len`) |    |
| `--prompts` | Input prompts separated by "\|" |
| `--dtype`   | KV cache dtype: `bf16` (default), `f16`, or `f32` |    |

---

## 🧠 Supported Architectures

* ✅ LLaMa (LLaMa2, LLaMa3)
* ✅ Qwen (Qwen2, Qwen3)
* ✅ Mistral

Supports both **Safetensor** and **GGUF** formats.

---

## 📌 Status

> **Project is under active development. Expect changes.**

---

## 🛠️ TODO

* [x] 🔧 Fix batched inference on `Metal`
* [ ] 🛰️ Multi-rank inference
* [ ] 🧠 More model support
* [x] 🧾 GGUF support
* [ ] 🌐 OpenAI-compatible API server (w/ streaming)
* [x] ⚡ FlashAttention (CUDA)
* [ ] ♻️ Continuous batching

---

## 📚 Reference

Core ideas inspired by:

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm

---

💡 **Like the project? Star it ⭐ and contribute!**

---
