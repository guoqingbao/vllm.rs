# OpenCode + vLLM.rs (OpenAI-compatible endpoint)

This guide connects OpenCode directly to vLLM.rs using the built-in
OpenAI-compatible `/v1/chat_completions` API. No proxy required.

```
OpenCode -> vLLM.rs (OpenAI-compatible)
```

## 1) Start vLLM.rs on port 8000

```bash
# Rust
./run.sh --features cuda,nccl,graph,flash-attn,flash-context,cutlass,graph --release --m Qwen/Qwen3-Coder-Next-FP8 --server --d 0,1 --prefix-cache

# or
./run.sh --features cuda,nccl,graph,flashinfer,cutlass,graph --release --m Qwen/Qwen3-Coder-Next-FP8 --server --d 0,1 --prefix-cache

# Different model
./run.sh --features cuda,nccl,graph,flash-attn,flash-context --release --m miromind-ai/MiroThinker-v1.5-30B --d 0,1 --server --prefix-cache

# Python
python3 -m vllm_rs.server --m Qwen/Qwen3-Coder-Next-FP8 --d 0,1 --prefix-cache
```

## 2) Configure OpenCode

Install opencode

```shell
curl -fsSL https://opencode.ai/install | bash
```

Or install with npm

```shell
npm i -g opencode-ai
```

Export config into `~/.config/opencode/config.json`


```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-vllm-rs": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM.rs Local",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "qwen3-coder": {
          "name": "Qwen3 Coder 80B"
        }
      }
    }
  },
  "model": "local-vllm-rs/qwen3-coder"
}
```

## 3) Run OpenCode

run opencode

```shell
opencode
```

### Trouble shooting

1. Use the chat logger to monitor detailed interactions between OpenCode and vLLM.rs.

```shell
# Log into files (in folder ./log)
export VLLM_RS_CHAT_LOGGER=1
```