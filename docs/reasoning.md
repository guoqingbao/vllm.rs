# Reasoning content

Reasoning-capable models (Qwen3, DeepSeek-R1, …) emit a "thinking"
pass wrapped in `<think>…</think>` (or `<|think|>…<|/think|>`,
`[THINK]…[/THINK]`) before the final answer. vllm.rs splits the two
phases into separate response fields, matching the convention used by
vLLM, SGLang, DeepSeek, and LiteLLM.

## Non-streaming

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "7 times 8 equals 56.",
      "reasoning_content": "Okay, so I need to figure out what 7 times 8 is..."
    }
  }]
}
```

## Streaming

```
data: {"choices":[{"delta":{"reasoning_content":"Okay, so I need..."}}]}
data: {"choices":[{"delta":{"content":"7 times 8 equals 56."}}]}
```

The reasoning markers themselves are stripped before they reach the
client.

## Opting out

```bash
export VLLM_RS_STREAM_AS_REASONING_CONTENT=false
```

Restores the previous shape: markers stay inside `content`,
`reasoning_content` stays empty.

## Non-reasoning models

`reasoning_content` is omitted entirely (non-streaming) and never
appears in `delta` (streaming). The JSON shape is unchanged.

## Token counts

When `usage.completion_tokens_details.reasoning_tokens` is present, it
reports the count of tokens inside the reasoning block.
