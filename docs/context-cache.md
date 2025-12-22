# Context Cache Guide

Context-cache lets the server reuse KV cache across turns via `session_id` when `--context-cache` is enabled (CUDA/Metal). This reduces prefill latency for long conversations.

## Enabling
- Start server with cache on:  
  ```bash
  target/release/vllm-rs --server --m Qwen/Qwen3-30B-A3B-Instruct-2507 --context-cache
  ```
- Metal example:  
  ```bash
  target/release/vllm-rs --server --m Qwen/Qwen3-4B-GGUF --f Qwen3-4B-Q4_K_M.gguf --context-cache --max-model-len 32768
  ```

## Using `session_id`
- First turn (creates cache):  
  ```json
  {"model":"default","messages":[{"role":"user","content":"Explain KV cache"}],"session_id":"chat-123"}
  ```
- Follow-up reuses cache; only the new message is sent:  
  ```json
  {"model":"default","messages":[{"role":"user","content":"continue"}],"session_id":"chat-123"}
  ```
- Cache limits follow `max_model_len` and block allocation; server logs warn when swapping or evicting.

## Notes
- Set `--max-model-len` and `--kv-fraction` to balance cache size vs decode headroom; on Metal prefer smaller lengths.
- `--fp8-kvcache` and `--flash-context` are optional CUDA optimizations (Ampere+); enable when building with the relevant features.
- Avoid mixing context-cache with streaming `session_id` on heavy loads unless you need persistence; throughput may drop if cache swaps.
