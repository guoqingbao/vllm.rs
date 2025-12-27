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
---

### 🐍 Python API

```python
from vllm_rs import Engine, EngineConfig, SamplingParams, Message
cfg = EngineConfig(weight_path="/path/Qwen3-8B-Q2_K.gguf", max_model_len=4096)
engine = Engine(cfg, "bf16")
params = SamplingParams(temperature=0.6, max_tokens=256)
message = Message("user", "How are you?")]

# Synchronous batch generation
outputs = engine.generate_sync([params, params], [[message], [message]])
print(outputs)

params.session_id = xxx  # Pass session_id to enable context cache

# Single-request streaming generation
(seq_id, prompt_length, stream) = engine.generate_stream(params, [message])
for item in stream:
   # item.datatype == "TOKEN"
   print(item.data)
```

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
openai.base_url = "http://localhost:8000/v1"

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

### 🤖✨ Interactive Chat and Batch Processing

> Interactive Chat
  <details open>
    <summary>Chat with Qwen3-32B-A3B model</summary>

```bash
# Context-cache automatically enabled under chat mode
python3 -m vllm_rs.chat --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

  </details>

  <details open>
    <summary>Chat with local unquantized model</summary>

```bash
python3 -m vllm_rs.chat --w /path/Qwen3-30B-A3B-Instruct-2507 --d 0,1
```

  </details>

  <details open>
    <summary>Chat with model quantized instantly (ISQ)</summary>

```bash
# Enable maximum context (262144 tokens), two ranks (`--d 0,1`)
python3 -m vllm_rs.chat --d 0,1 --m Qwen/Qwen3-30B-A3B-Instruct-2507 --isq q4k --max-model-len 262144
```

  </details>

> Batch Processing
  <details>
    <summary>Batch Completion</summary>

```bash
python3 -m vllm_rs.completion --f /path/qwq-32b-q4_k_m.gguf --prompts "How are you? | How to make money?"
```

```bash
python3 -m vllm_rs.completion --w /home/GLM-4-9B-0414 --d 0,1 --batch 8 --max-model-len 1024 --max-tokens 1024
```

  </details>

### 🧰 MCP Multi-Server Demo (Python Client)

Start vLLM.rs with an MCP config file:

```shell
target/release/vllm-rs --m <model_id> --server --mcp-config ./mcp.json
```

Then call a prefixed MCP tool from Python:

```python
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1"

response = openai.chat.completions.create(
   model="",
   messages=[{"role": "user", "content": "Use filesystem_read_file to read README.md"}],
)
print(response.choices[0].message)
```
