# Context Cache Guide

Context-cache lets the server reuse KV cache across conversation turns when `--context-cache` is enabled (CUDA/Metal). This reduces prefill latency for multi-turn conversations.

## Session Detection Strategies

There are **two ways** to enable context cache reuse:

| Strategy | Flag | How it works | Best for |
|----------|------|--------------|----------|
| Explicit `session_id` | `--context-cache` | Client provides `session_id` in each request | Full control, multi-client scenarios |
| Automatic fingerprint | `--context-cache --force-cache` | Server detects sessions via message fingerprinting | Simple clients, single-user scenarios |

---

## Strategy 1: Explicit `session_id` (Recommended for Production)

Client provides the same `session_id` across conversation turns.

### Enable
```bash
target/release/vllm-rs --server --m Qwen/Qwen3-30B-A3B-Instruct-2507 --context-cache
```

### Usage
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

### Pros & Cons
- ✅ Full control over session lifecycle
- ✅ Works with multiple concurrent users/clients
- ✅ Client can manage cache explicitly
- ❌ Requires client to track and send `session_id`

---

## Strategy 2: Automatic Fingerprint Detection (Zero Client Changes)

Server automatically detects multi-turn conversations by fingerprinting message content. **No `session_id` needed!**

### Enable
```bash
target/release/vllm-rs --server --m Qwen/Qwen3-30B-A3B-Instruct-2507 --context-cache --force-cache
```

### Usage
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
2. For continuation requests, fingerprint of previous messages (excluding last user message) is matched
3. If matched → reuse cached KV; if not → create new session

### Fingerprint Matching Logic
- **User/System messages**: Match on `role + text_length + first_32_chars + last_32_chars`
- **Assistant messages**: Match on `role + last_32_chars` only (handles thinking/reasoning content)

### Pros & Cons
- ✅ Zero client changes needed
- ✅ Automatic cache reuse for stateless APIs
- ✅ Works with reasoning models (thinking content stripped)
- ❌ May not match if client modifies message content (e.g., adds timestamps)
- ❌ Not suitable for multi-user scenarios (concurrent requests may conflict)

### When Fingerprint Matching Fails
Fingerprint detection **will NOT work** if:
- Client adds dynamic content to messages (e.g., `<info-msg>timestamp</info-msg>`)
- Client truncates or summarizes previous messages
- Multiple different conversations run in parallel with similar prefixes

In these cases, use explicit `session_id` instead.

---

## Python Server Usage

```python
from vllm_rs import Engine, EngineConfig

# Strategy 1: Explicit session_id only
cfg = EngineConfig(model_id="Qwen/Qwen3-4B", flash_context=True)

# Strategy 2: Automatic fingerprint detection
cfg = EngineConfig(model_id="Qwen/Qwen3-4B", flash_context=True, force_cache=True)

engine = Engine(cfg, "bf16")
engine.start_server(8000, False)
```

---

## Configuration Tips

- Set `--max-model-len` and `--kv-fraction` to balance cache size vs decode headroom
- On Metal, prefer smaller `--max-model-len` values
- `--fp8-kvcache` is an optional CUDA optimization (Ampere+)
- Avoid heavy streaming loads with context-cache if cache swaps hurt throughput

## Logs

When fingerprint detection is active, you'll see logs like:
```
FingerprintManager: registered new session auto_session_0
FingerprintManager: matched existing session auto_session_0
FingerprintManager: extended session auto_session_0 fingerprint with assistant response
```
