# 🚀 **vLLM.rs** – 用 Rust 实现的极简 vLLM

一个极速 ⚡、轻量的 🦀**Rust 实现版 vLLM**。

---

<p align="center">
  <a href="./ReadMe.md">English</a> |
  <a href="./ReadMe-CN.md">简体中文</a> |
</p>

## ✨ 主要特性

* 🔧 **纯 Rust 后端** – 完全**不依赖 PyTorch**
* 🚀 **高性能** – 性能优于 vLLM 和 Nano-vLLM
* 🧠 **极简核心** – 核心逻辑仅 **< 1000 行** Rust 代码
* 💻 **跨平台支持** – 支持 **CUDA**（Linux/Windows）与 **Metal**（macOS）
* 🤖 **内置聊天/API 服务** – Rust 原生实现的聊天与 API 服务
* 🐍 **轻量 Python 接口** – 使用 PyO3 构建的 Python 聊天接口
* 🤝 **欢迎贡献** – 欢迎提交 PR、问题或给项目点亮 ⭐！

---

### 性能对比

> 模型: Qwen3-0.6B (BF16)；
> 并发请求数: 256；
> Max Model Length: 1024；
> 每个请求最大输出: 1024

| 推理引擎 | 输出Tokens | 耗时 (s) | 吞吐率 (tokens/s) |
|------------------|---------------|----------|------------------------|
| vLLM (RTX 4070) (Reference)          | 133,966       | 98.37    | 1361.84                |
| Nano-vLLM (RTX 4070) (Reference)      | 133,966       | 93.41    | 1434.13                |
| **vLLM.rs** (**A100**)        | 257,792       | 25.21s    | **10216.44** (**提升30%+**)               |
| Nano-vLLM (A100)       | 262144       | 34.22s    |   7660.26      | 

#### 复现步骤

**vLLM.rs**
```shell
# 未启用Cuda Graph，未启用Flash Attention，无模型预热 (最终报告)
cargo run --release --features cuda -- --w /home/Qwen3-0.6B --batch 256 --max-tokens 1024 --max-model-len 1024
# 日志
2025-07-16T10:32:32.632729Z  INFO vllm_rs: --- Performance Metrics ---
2025-07-16T10:32:32.632764Z  INFO vllm_rs: ⏱️ Prompt tokens: 4096 in 12.56s (326.17 tokens/s)
2025-07-16T10:32:32.632781Z  INFO vllm_rs: ⏱️ Decoded tokens: 257792 in 25.21s (10216.44 tokens/s)

# 启用cuda graph 获得更高性能
cargo run --release --features cuda,graph -- --w /home/Qwen3-0.6B --batch 256 --max-tokens 1024 --max-model-len 1024
# 启用cuda graph和flash attention获得最高性能 (编译flash attention需要较长时间)
cargo run --release --features cuda,flash-attn,graph -- --w /home/Qwen3-0.6B --batch 256 --max-tokens 1024 --max-model-len 1024 --flash
```

***Nano-vLLM** 

   💡 (为公平比较，请修改所有请求最长输出为固定值（如1024），而非随机值（100-1024)）
```shell
# 默认启用cuda graph，启用flash attention 与模型预热
python3 bench.py
# 日志
Generating: 100%|██████████████████| 1/1 [00:02<00:00,  2.65s/it, Prefill=1tok/s, Decode=369tok/s]
Total: 262144tok, Time: 34.22s, Throughput: 7660.26tok/s
```

## 📦 安装与使用

> ⚠️ 启用 Flash Attention（CUDA）时，首次编译可能需要较长时间。

### 🛠️ 环境要求

* 安装 [Rust 工具链](https://www.rust-lang.org/tools/install)
* macOS 平台需安装 [Xcode 命令行工具](https://mac.install.guide/commandlinetools/)
* 构建 Python 接口需安装 [Maturin](https://github.com/PyO3/maturin)

---

## 🐍 快速 Python 示例
   💡 编译vllm.rs Python包可参见`API 服务模式（Python 接口）`
```python
from vllm_rs import Engine, EngineConfig, SamplingParams, Message
cfg = EngineConfig(model_path="/path/Qwen3-8B-Q2_K.gguf", max_model_len=4096)
engine = Engine(cfg, "bf16")
params = SamplingParams(temperature=0.6, max_tokens=256)
prompt = engine.apply_chat_template([Message("user", "How are you?")], True)

# 同步批量生成
outputs = engine.generate_sync([params,params], [prompt, prompt])
print(outputs)

# 单请求流式生成
stream = engine.generate_stream(params, prompt)
for token in stream:
    print(token)
```

---

## 🤖✨ 交互模式（纯 Rust CLI）

使用 `--i` 启用交互模式，`--w` 指定模型路径：

```bash
# CUDA（短上下文）
cargo run --release --features cuda -- --i --w /path/qwq-32b-q4_k_m.gguf

# CUDA + Flash Attention（超长上下文，如 32k tokens）
cargo run --release --features cuda,flash-attn -- --i --w /path/qwq-32b-q4_k_m.gguf

# macOS（Metal）
cargo run --release --features metal -- --i --w /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf
```

---

## 🌐✨ API 服务模式（Python 接口）

1. **安装 Maturin**

```bash
pip install maturin
pip install maturin[patchelf]  # Linux/Windows 平台
```

2. **构建 Python 包**

```bash
# CUDA（较短上下文）
maturin build --release --features cuda,python

# CUDA + Flash Attention (超长上下文 (>32k时) 推荐启用）
maturin build --release --features cuda,flash-attn,python

# macOS（Metal）
maturin build --release --features metal,python
```

3. **安装构建好的包与依赖**

```bash
pip install target/wheels/vllm_rs-0.1.0-cp38-abi3-*.whl
pip install fastapi uvicorn
```

4. **启动 OpenAI API 服务**
   💡你可以使用**任何兼容 OpenAI API 的客户端**进行交互。
```bash
# 启动 OpenAI 兼容的 API 服务（监听 http://0.0.0.0:8000）
# openai.base_url = "http://localhost:2000/v1/"
# openai.api_key = "EMPTY"
# 添加`--flash`选项以启用Flash attention （maturin生成whl包时需添加`flash-attn`）
python example/server.py --w /path/qwq-32b-q4_k_m.gguf --host 0.0.0.0 --port 8000
```


---

### 其他 Python 示例

```bash
# 交互式聊天
python3 example/chat.py --i --w /path/qwq-32b-q4_k_m.gguf

# 批量同步示例
python3 example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```

---

### 📽️ 演示视频

🎉 观看项目运行演示：

<video src="https://github.com/user-attachments/assets/0751471b-a0c4-45d7-acc6-99a3e91e4c91" width="70%"></video>

---

## 🧾 补全模式（Rust CLI）

### GGUF 模型

```bash
# CUDA
cargo run --release --features cuda -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"

# CUDA + Flash Attention
cargo run --release --features cuda,flash-attn -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"

# Metal（macOS）
cargo run --release --features metal -- --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you today?"
```

### Python 调用：

```bash
python example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```

### Safetensor 模型（未量化）

```bash
# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --prompts "How are you today?"

# Metal（macOS）
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "How are you today?"
```

---

## 📚 批量请求支持

多个 prompt 使用 `|` 分隔：

```bash
# GGUF 模型（Rust）
cargo run --release --features cuda,flash-attn -- --w /path/qwq-32b-q4_k_m.gguf --prompts "Talk about China. | Talk about America." --max-model-len 1024

# Safetensor 模型（Rust）
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "Talk about China. | Talk about America."

# GGUF 模型（Python）
python3 example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?" --max-model-len 1024
```

---

## 🗜️ 实时量化（GGUF 格式转换）

量化过程可能需要几分钟时间：

```bash
# macOS
cargo run --release --features metal -- --w /path/Qwen3-0.6B/ --quant q4k --prompts "How are you today?"

# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --quant q4k --prompts "How are you today?"
```

---

## 📄 示例输出

**单条请求**（Qwen3-0.6B，BF16，macOS Metal）：

```bash
cargo run --features metal -- --w /path/Qwen3-0.6B/ --prompts "How are you today?"
```

```
<think>
用户提问："How are you today?"...
</think>

你好呀！今天感觉怎么样？我在这里可以帮你解答任何问题！😊 有需要尽管告诉我！
```

---


## ⚙️ 命令行参数说明

| 参数          | 描述                                     |       |
| ----------- | -------------------------------------- | ----- |
| `--w`       | 模型路径（Safetensor 目录或 GGUF 文件）           |       |
| `--d`       | 设备 ID，例如 `--d 0`                       |       |
| `--max_num_seqs`   | 同时处理的最大请求数（默认 `32`）               |       |
| `--max_tokens`     | 单次最大输出 token 数（默认 `4096`，上限为模型支持的最大长度） |       |
| `--batch`     | 仅用于性能 (启用后会忽略 `max-num-seqs` 与 `prompts`) |    |
| `--prompts` | 输入的 prompt，多个使用 \`                     | \` 分隔 |
| `--dtype`   | KV 缓存数据类型：`bf16`（默认）、`f16` 或 `f32`     |       |

---

## 🧠 支持的模型架构

* ✅ LLaMa 系列（LLaMa2、LLaMa3）
* ✅ Qwen 系列（Qwen2、Qwen3）
* ✅ Mistral

支持 **Safetensor** 和 **GGUF** 格式。

---

## 📌 项目状态

> 🚧 **项目仍在积极开发中，接口与功能可能发生变更。**

---

## 🛠️ 开发计划（TODO）

* [x] Metal 平台支持批量推理
* [x] 支持 GGUF 格式
* [x] CUDA 平台 Flash Attention 支持
* [x] OpenAI API 兼容服务器（支持流式输出）
* [x] 持续批处理
* [ ] 多卡并行推理
* [ ] 支持更多模型类型

---

## 📚 参考项目

核心思路参考：

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm 项目

---

💡 **喜欢这个项目？欢迎 ⭐ 收藏和参与贡献！**
