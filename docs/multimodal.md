# Multimodal Model Usage

This project supports vision-language models (Qwen3-VL dense/MoE, Gemma3, Mistral3-VL). The server exposes `/v1/chat/completions` with mixed text+image content and optional web UI.

## Starting servers
- Qwen3-VL (CUDA):  
  ```bash
  ./run.sh --release --features cuda -- --server \
    --m Qwen/Qwen3-VL-8B-Instruct --ui-server --context-cache
  ```
- Qwen3-VL (Metal/Mac):  
  ```bash
  ./run.sh --release --features metal -- --server \
    --m Qwen/Qwen3-VL-8B-Instruct --max-model-len 32768 --ui-server
  ```
- Gemma3 (vision):  
  ```bash
  ./run.sh --release --features cuda -- --server \
    --m google/gemma-3-4b-it --ui-server --context-cache
  ```
- Mistral3-VL (vision):  
  ```bash
  ./run.sh --release --features cuda -- --server \
    --m mistralai/Ministral-3-8B-Reasoning --ui-server --context-cache
  ```

## Request payloads (OpenAI-compatible)
- Text + image URL:  
  ```json
  {
    "model": "default",
    "messages": [
      {"role":"user","content":[
        {"type":"text","text":"Describe this image"},
        {"type":"image_url","image_url":"https://example.com/cat.png"}
      ]}
    ]
  }
  ```
- Text + base64 image: `{"type":"image_base64","image_base64":"data:image/png;base64,..."}`

## Tips
- Use smaller `--max-model-len` on Metal if VRAM is tight; consider `--kv-fraction` on CUDA to reserve cache.
- For batch image inputs, keep concurrent images modest; too many images will increase prefill time.
- `--ui-server` opens the built-in chat UI for uploads; without it, send HTTP requests directly.***
