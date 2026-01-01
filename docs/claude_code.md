# Claude Code + vLLM.rs (via LiteLLM)

This guide wires **Claude Code** to your local vLLM.rs server using a LiteLLM proxy. The flow looks like this:

```
Claude Code 🤖 -> LiteLLM proxy (Anthropic-compatible) -> vLLM.rs (OpenAI-compatible)
```

## 1) Install Claude Code + set `settings.json`

Install Claude Code, then set `~/.claude/settings.json` to:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8010",
    "ANTHROPIC_MODEL": "default",
    "ANTHROPIC_SMALL_FAST_MODEL": "default",
    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

## 2) Install LiteLLM + create `config.yaml`

Install LiteLLM and create a `config.yaml` like this:

```bash
pip install litellm
```

```yaml
model_list:
  - model_name: default
    litellm_params:
      model: openai/default
      api_base: http://127.0.0.1:8000/v1
      api_key: empty
      stream: true
      max_tokens: 32768
      drop_params: True

  - model_name: anthropic/*
    litellm_params:
      model: openai/default
      api_base: http://127.0.0.1:8000/v1
      api_key: empty
      stream: true
      max_tokens: 32768
      drop_params: True
```

Start the proxy on port 8010 (matches `ANTHROPIC_BASE_URL`):

```bash
litellm --config /path/to/config.yaml --port 8010
```

## 3) Start vLLM.rs on port 8000 + run Claude Code

Run vLLM.rs with an OpenAI-compatible server on port 8000:

```bash
# Rust
./target/release/vllm-rs --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --server --prefix-cache
# Python
python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --server --prefix-cache
```

Now open Claude Code and send requests. ✅

## How the pieces fit together

- Claude Code sends Anthropic-style requests to `http://127.0.0.1:8010`.
- LiteLLM translates those requests to OpenAI-compatible calls.
- vLLM.rs serves the actual inference at `http://127.0.0.1:8000/v1`.

That's it - all tool calls and chat completions go through your local vLLM.rs server. 🚀
