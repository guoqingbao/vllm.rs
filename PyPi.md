# 🚀 **vLLM.rs** – A Minimalist vLLM in Rust

A blazing-fast ⚡, lightweight **Rust** 🦀 implementation of vLLM.

## ✨ Key Features

* 🔧 **Pure Rust Backend** – Absolutely **no** PyTorch required
* 🚀 **High Performance** – Superior than vLLM and Nano-vLLM
* 🧠 **Minimalist Core** – Core logic written in **< 1000 lines** of clean Rust
* 💻 **Cross-Platform** – Supports **CUDA** (Linux/Windows) and **Metal** (macOS)
* 🤖 **Built-in Chatbot/API Server** – Native Rust server for both CUDA and Metal
* 🐍 **Lightweight Python Interface** – PyO3-powered bindings for chat completion
* 🤝 **Open for Contributions** – PRs, issues, and stars are welcome!


## 📘 Usage

### 📦 Install with pip

```shell
# flash-attn built-in for prefilling (on CUDA)
pip install vllm-rs
```

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

# Chat completion
python3 example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```

## More Examples

[vLLM.rs](https://github.com/guoqingbao/vllm.rs/tree/main/example)

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



## 🧠 Supported Architectures

* ✅ LLaMa (LLaMa2, LLaMa3)
* ✅ Qwen (Qwen2, Qwen3)
* ✅ Mistral

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
