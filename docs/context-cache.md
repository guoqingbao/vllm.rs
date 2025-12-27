# Context Cache Guide

Context-cache lets the server reuse KV cache across conversation turns when `--context-cache` is enabled (CUDA/Metal). This reduces prefill latency for multi-turn conversations.

## Session Detection: Automatic vs Explicit

When context-cache is enabled, the server uses **two detection modes**:

| Scenario | Behavior |
|----------|----------|
| Client provides `session_id` | Uses that session ID for cache lookup |
| No `session_id` provided | **Automatic fingerprint detection** matches conversations |

---

## Enabling Context Cache

```bash
# CUDA
target/release/vllm-rs --server --m Qwen/Qwen3-30B-A3B-Instruct-2507 --context-cache

# Metal (macOS)
target/release/vllm-rs --server --m Qwen/Qwen3-4B-GGUF --f Qwen3-4B-Q4_K_M.gguf --context-cache --max-model-len 32768
```

---

## Using Explicit `session_id` (Recommended for Multi-User)

Client provides the same `session_id` across conversation turns.

```json
// Turn 1: Creates cache
{"model":"default","messages":[{"role":"user","content":"Explain KV cache"}],"session_id":"chat-123"}

// Turn 2: Reuses cache (only new tokens prefilled)
{"model":"default","messages":[
  {"role":"user","content":"Explain KV cache"},
  {"role":"assistant","content":"KV cache is..."},
  {"role":"user","content":"Tell me more"}
],"session_id":"chat-123"}
```

**Pros:** Full control, works with multiple concurrent users
**Cons:** Requires client to track session ID

---

## Automatic Fingerprint Detection (Zero Client Changes)

When no `session_id` is provided, the server automatically detects multi-turn conversations by fingerprinting message content.

```json
// Turn 1: Server creates auto_session_0
{"model":"default","messages":[{"role":"user","content":"Hello"}]}

// Turn 2: Server automatically matches to auto_session_0
{"model":"default","messages":[
  {"role":"user","content":"Hello"},
  {"role":"assistant","content":"Hi there!"},
  {"role":"user","content":"How are you?"}
]}
```

### How It Works
1. Server computes a fingerprint from message roles and content
2. For continuation requests, fingerprint of previous messages is matched
3. If matched → reuse cached KV; if not → create new session

### Fingerprint Matching Rules
- **User/System messages**: Match on `role + text_length + first_32_chars + last_32_chars`
- **Assistant messages**: Match on `role + last_32_chars` only (handles reasoning/thinking content)

**Pros:** Zero client changes, automatic cache reuse
**Cons:** May not work if client modifies message content

---

## When Fingerprint Detection Fails

Automatic detection **will NOT work** if:
- Client adds dynamic content (e.g., timestamps in `<info-msg>` tags)
- Client truncates or summarizes previous messages
- Multiple different conversations run in parallel

In these cases, use explicit `session_id` instead.

---

## Python Server Usage

```python
from vllm_rs import Engine, EngineConfig

# Enable context-cache (fingerprint detection is automatic)
cfg = EngineConfig(model_id="Qwen/Qwen3-4B", flash_context=True)

engine = Engine(cfg, "bf16")
engine.start_server(8000, False)
```

---

## Configuration Tips

- Set `--max-model-len` and `--kv-fraction` to balance cache size vs decode headroom
- On Metal, prefer smaller `--max-model-len` values
- `--fp8-kvcache` is an optional CUDA optimization (Ampere+)

## Logs

When fingerprint detection is active:
```
FingerprintManager: registered new session auto_session_0
FingerprintManager: matched existing session auto_session_0
FingerprintManager: extended session auto_session_0 fingerprint with assistant response
```
