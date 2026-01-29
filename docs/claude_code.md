# Claude Code + vLLM.rs (Anthropic-compatible endpoint)

This guide connects Claude Code directly to vLLM.rs using the built-in
Anthropic-compatible `/v1/messages` API. No proxy required.

```
Claude Code -> vLLM.rs (Anthropic-compatible)
```

## 1) Start vLLM.rs on port 8000

```bash
# Rust
./run.sh --features cuda,nccl,graph,flash-attn,flash-context --release --m miromind-ai/MiroThinker-v1.5-30B --d 0,1 --ui-server --prefix-cache
# Python
python3 -m vllm_rs.server --m miromind-ai/MiroThinker-v1.5-30B --d 0,1 --server --prefix-cache
```

## 2) Configure Claude Code

Install claude code

```shell
npm install -g @anthropic-ai/claude-code
```

Export config

```shell
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="sk-dummy"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
```

Or make it permanent

Set `~/.claude/settings.json` (or copy from `example/claude/settings.json`):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_MODEL": "default",
    "ANTHROPIC_SMALL_FAST_MODEL": "default",
    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

## 3) Run Claude Code

run claude code

```shell
claude
```

or verify with a direct request (optional)

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "Hello from Claude Code"}
    ]
  }'
```

## Notes

- Streaming uses server-sent events (SSE) on `/v1/messages` with `stream: true`.
- Token counting is available at `POST /v1/messages/count_tokens`.
- Embeddings are not part of the Anthropic API and are not exposed here.

### Trouble shooting

1. Use the chat logger to monitor detailed interactions between Claude Code and vLLM.rs.

```shell
# Log into files (in folder ./log)
export VLLM_RS_CHAT_LOGGER=1
```

2. Use custom tool prompt (for example, to guide tool use)

```shell
./run.sh --features cuda,nccl,graph,flash-attn,flash-context --release --m Qwen/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --ui-server --prefix-cache --tool-prompt ./example/tool_prompt.json
```