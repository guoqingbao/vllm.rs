#### How to reproduce?
**vLLM.rs**
```shell
pip install vllm_rs
python -m vllm_rs.completion --w /home/Qwen3-0.6B/ --batch 256 --max-tokens 1024 --max-model-len 1024

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

#### 复现步骤

**vLLM.rs**
```shell
pip install vllm_rs
python -m vllm_rs.completion --w /home/Qwen3-0.6B/ --batch 256 --max-tokens 1024 --max-model-len 1024

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

---

### 🤖 Client Usage of Context Cache

**Key changes for the client:**

```python
import uuid
import openai
use_context_cache = True #flag to use context_cache
# create session_id for each new chat session and use it throughout that session (session cache will be cleared if the client aborted the connection)
session_id = str(uuid.uuid4())
extra_body = {"session_id": session_id if use_context_cache else None }

# vllm.rs service url
openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

response = openai.chat.completions.create(
   model="",
   messages=messages + [user_msg],
   stream=True,
   max_tokens = max_tokens,
   temperature = temperature,
   top_p = top_p,
   extra_body = extra_body, #pass session_id through extra_body
)

```

### 🤖 客户端使用上下文缓存特性

**主要修改点**

```python
import uuid
import openai
use_context_cache = True #是否启用上下文缓存特性
# 为每一个新对话创建一个session_id，并在此对话中一直使用（当客户端主动中断对话时，此对话缓存会被立即清理）
session_id = str(uuid.uuid4())
extra_body = {"session_id": session_id if use_context_cache else None }

# vllm.rs服务地址
openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

response = openai.chat.completions.create(
   model="",
   messages=messages + [user_msg],
   stream=True,
   max_tokens = max_tokens,
   temperature = temperature,
   top_p = top_p,
   extra_body = extra_body, #将session_id通过extra_body传入
)

```
---