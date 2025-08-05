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

| 推理引擎 | Tokens | 耗时 (s) | 吞吐率 (tokens/s) |
|------------------|---------------|----------|------------------------|
| vLLM (RTX 4070) (Reference)          | 133,966       | 98.37    | 1361.84                |
| Nano-vLLM (RTX 4070) (Reference)      | 133,966       | 93.41    | 1434.13                |
| **vLLM.rs** (**A100**)        | 262,144       | 23.88s    | **10977.55** (**提升40%+**)               |
| Nano-vLLM (A100)       | 262,144       | 34.22s    |   7660.26      | 

#### 复现步骤

**vLLM.rs**
```shell
pip install vllm-rs
python example/completion.py --w /home/Qwen3-0.6B/ --batch 256 --max-tokens 1024 --max-model-len 1024

# 日志输出
Allocating 8192 KV blocks (28672 MB) for [256 seqs x 1024 tokens]
Maximum batched tokens 262144 (8192 blocks x Block_Size 32 for KV cache).
Start inference with 256 prompts
--- Performance Metrics ---
⏱️ Prompt tokens: 4096 in 0.28s (14894.55 tokens/s)
⏱️ Decoded tokens: 258048 in 23.60s (10944.62 tokens/s)
```

**Nano-vLLM** 

   💡 为公平比较，请修改所有请求最长输出为固定值（如1024），而非随机值（100-1024)
```shell
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
python3 bench.py
# 日志输出
Generating: 100%|██████████████████| 1/1 [00:02<00:00,  2.65s/it, Prefill=1tok/s, Decode=369tok/s]
Total: 262144tok, Time: 34.22s, Throughput: 7660.26tok/s
```

### vLLM.rs 在 **Metal (Apple Silicon, M4)** 上的性能
> 模型: Qwen3-0.6B (BF16), Qwen3-4B (Q4_K_M), Qwen3-8B (Q2_K)；
> 并发请求数: 1 - 128；
> Max Model Length: 512 - 2048；
> 每个请求最大输出: 512 - 2048；

| 模型 | 并发数 | 输出Tokens | 耗时 (s) | 吞吐量 (tokens/s) |
|------------------|--------|--------|---------|-------------|
| Qwen3-0.6B (BF16) |  128  | 63488       | 83.13s    | 763.73     |
| Qwen3-0.6B (BF16) |  32      | 15872       | 23.53s    | 674.43    |
| Qwen3-0.6B (BF16) | 1       | 456       | 9.23s    | 49.42       |
| Qwen3-4B (Q4_K_M)  | 1       | 1683       | 52.62s    | 31.98     |
| Qwen3-8B (Q2_K)  | 1       | 1300       | 80.88s    | 16.07     |

## 🧠 支持的模型架构

* ✅ LLaMa 系列（LLaMa2、LLaMa3）
* ✅ Qwen 系列（Qwen2、Qwen3）
* ✅ Qwen3 MoE 系列
* ✅ Mistral
* ✅ GLM4 (0414版本, **非ChatGLM**)

支持 **Safetensor** 和 **GGUF** 格式。

## 📦 从pip安装

```shell
# 默认支持flash-attn prefilling
python3 -m pip install vllm-rs
```


## 📘 使用方法（Python）

### 🐍 快速 Python 示例
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

### 🌐✨ API Server
   💡你可以使用**任何兼容 OpenAI API 的客户端**进行交互。

```bash
# 启动 OpenAI 兼容的 API 服务（监听 http://0.0.0.0:8000）
# openai.base_url = "http://localhost:8000/v1/"
# openai.api_key = "EMPTY"
python3 example/server.py --w /path/qwq-32b-q4_k_m.gguf --host 0.0.0.0 --port 8000
# 或，多GPU推理服务：
python3 example/server.py --w /path/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --host 0.0.0.0 --port 8000
```

### 🤖✨ 交互式聊天与批处理

```bash
# 交互式聊天
python3 example/chat.py --i --w /path/qwq-32b-q4_k_m.gguf

# 指定设备2 (设备序号为1，`--d 1`)
python3 example/chat.py --i --d 1 --w /path/GLM-4-9B-0414-Q4_K_M.gguf

# 批量同步示例
python3 example/completion.py --w /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"

# 批量同步示例 (多GPU)
python3 example/completion.py --w /home/GLM-4-9B-0414 --d 0,1 --batch 8 --max-model-len 1024 --max-tokens 1024
```

## 🔨 从源代码编译安装（可选）

> ⚠️ 启用 Flash Attention（CUDA）时，首次编译可能需要较长时间。

### 🛠️ 环境要求

* 安装 [Rust 工具链](https://www.rust-lang.org/tools/install)
* **macOS** 平台需安装 [Xcode 命令行工具](https://mac.install.guide/commandlinetools/)
* 构建 Python 接口需安装 [Maturin](https://github.com/PyO3/maturin)

### 编译步骤
1. **安装 Maturin**

```bash
sudo apt install libssl-dev pkg-config -y # 编译依赖 (Linux)
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

# 多GPU推理 (CUDA, 生成独立的runner，运行于不同进程)
./build.sh --release --features cuda,nccl,flash-attn,python
```

3. **安装构建好的包与依赖**

```bash
pip install target/wheels/vllm_rs-*-cp38-abi3-*.whl --force-reinstall
pip install fastapi uvicorn
```

## 📘 使用方法（Rust）
### 🤖✨ Rust CLI 模式

使用 `--i` 启用交互模式，`--w` 指定模型路径：

```bash
# CUDA（短上下文）
cargo run --release --features cuda -- --i --w /path/qwq-32b-q4_k_m.gguf

# 使用第三个设备 (设备序号2，`--d 2`)
cargo run --release --features cuda -- --i --d 2 --w /path/GLM-4-9B-0414-Q4_K_M.gguf

# CUDA + Flash Attention（超长上下文，如 32k tokens）
cargo run --release --features cuda,flash-attn -- --i --w /path/qwq-32b-q4_k_m.gguf

# macOS（Metal）
cargo run --release --features metal -- --i --w /path/DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf
```

Safetensor 模型（未量化）

```bash
# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --prompts "How are you today?"

# Metal（macOS）, 多个 prompt 使用 `|` 分隔
cargo run --release --features metal -- --w /path/Qwen3-8B/ --prompts "Talk about China. | Talk about America."

# 多GPU推理（交互模式）
./run.sh run --release --features cuda,flash-attn,nccl -- --w /home/GLM-4-9B-0414 --d 0,1 --i --max-tokens 1024 --max-model-len 1024
```

### ⚙️ 命令行参数说明

| 参数          | 描述                                     |       |
| ----------- | -------------------------------------- | ----- |
| `--w`       | 模型路径（Safetensor 目录或 GGUF 文件）           |       |
| `--d`       | 设备 ID，例如 `--d 0`                       |       |
| `--max-num-seqs`   | 同时处理的最大请求数（默认 `32`, macOS平台为`8`）   |       |
| `--max-tokens`     | 单次最大输出 token 数（默认 `4096`，上限为模型支持的最大长度） |       |
| `--batch`     | 仅用于性能 (启用后会忽略 `max-num-seqs` 与 `prompts`) |    |
| `--prompts` | 输入的 prompt，多个使用 \| 分隔 |
| `--dtype`   | KV 缓存数据类型：`bf16`（默认）、`f16` 或 `f32`     |       |

## 📽️ 演示视频

🎉 观看项目运行演示：

<video src="https://github.com/user-attachments/assets/0751471b-a0c4-45d7-acc6-99a3e91e4c91" width="70%"></video>


## 🗜️ 实时量化（GGUF 格式转换）

量化过程可能需要几分钟时间：

```bash
# macOS
cargo run --release --features metal -- --w /path/Qwen3-0.6B/ --quant q4k --prompts "How are you today?"

# CUDA
cargo run --release --features cuda,flash-attn -- --w /path/Qwen3-8B/ --quant q4k --prompts "How are you today?"
```


## 📌 项目状态

> 🚧 **项目仍在积极开发中，接口与功能可能发生变更。**

## 🛠️ 开发计划（TODO）

* [x] Metal 平台支持批量推理
* [x] 支持 GGUF 格式
* [x] CUDA 平台 Flash Attention 支持
* [x] OpenAI API 兼容服务器（支持流式输出）
* [x] 持续批处理
* [x] 多卡并行推理 （目前支持Safetensors非量化格式模型，GGUF多卡推理待支持）
* [x] Metal/macOS平台Prompt处理加速
* [ ] 支持更多模型类型


## 📚 参考项目

参考：

* [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm)
* Python nano-vllm 项目

---

💡 **喜欢这个项目？欢迎 ⭐ 收藏和参与贡献！**
