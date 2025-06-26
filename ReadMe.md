## vLLM.rs (Minimalist vLLM for Rust)

## Usage
Install Rust compiler and execute the following demo

**CUDA (it takes some time for the first run, for building CUDA kernels):**
```
#GGUF model
cargo run --release --features cuda -- --weight-path /Users/bob/Downloads/qwq-32b-q4_k_m.gguf --prompts "How are you today?"

#Safetensor model
cargo run --release --features cuda -- --weight-path /Users/bob/Downloads/Qwen3-8B/ --prompts "How are you today?"
```

**Metal (Mac)**
```
#GGUF model
cargo run --release --features metal -- --weight-path /Users/bob/Downloads/qwq-32b-q4_k_m.gguf --prompts "How are you today?"

#Safetensor model
cargo run --release --features metal -- --weight-path /Users/bob/Downloads/Qwen3-0.6B/ --prompts "How are you today?"
```

**Batched request (CUDA only, prompts separated by "|")**
```
#GGUF model
cargo run --release --features metal -- --weight-path /Users/bob/Downloads/qwq-32b-q4_k_m.gguf --prompts "Please talk about China. | Please talk about America."

#Safetensor model
cargo run --release --features cuda -- --weight-path /Users/bob/Downloads/Qwen3-8B/ --prompts "Please talk about China. | Please talk about America."
```

**In-situ quantization (format conversion takes few minutes)**
```
#Mac (load model into quantized gguf (q4k) format)
cargo run --release --features metal -- --weight-path /Users/bob/Downloads/Qwen3-0.6B/ --quant q4k --prompts "How are you today?"

#CUDA
cargo run --release --features cuda -- --weight-path /Users/bob/Downloads/Qwen3-8B/ --quant q4k --prompts "How are you today?"
```


Sample result (Qwen3-0.6B BF16, 1 requests)

```
bob@Mac4 vllm.rs % cargo run --features metal -- --weight-path /Users/bob/Downloads/Qwen3-0.6B/ --prompts "How are you today?"
   Compiling vllm-rs v0.1.0 (/Users/bob/vllm.rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.21s
     Running `target/debug/vllm-rs --weight-path /Users/bob/Downloads/Qwen3-0.6B/ --prompts 'How are you today?'`
<think>
Okay, the user asked, "How are you today?" I need to respond in a friendly and helpful way. Let me start by acknowledging their question. I should say something like, "Hi there! How are you today?" to keep it open-ended.

Next, I should check if they need help. Maybe offer assistance. For example, "I'm here to help you with anything!" That shows I'm available. I should also mention that I'm here to support them, so they feel comfortable asking questions.

I should keep the tone positive and approachable. Avoid any negative language. Make sure the response is concise but covers the necessary points. Let me put that all together in a natural way.
</think>

Hi there! How are you today? I'm here to help you with anything! �� Let me know if there's anything you need!%                                                                                
```

Sample batched result (**LLaMa3.1-8B** BF16, **16** requests, on A100)
```shell
8450 tokens generated in 14.28 s (decoding throughput 591.82 tokens/s)
```

Sample batched result (**QwQ-32B** GGUF Q4K, **4** requests, on A100)
```shell
2025-06-26T07:24:15.509470Z  INFO vllm_rs::core::engine: [4 requests] 400 tokens generated in 5.55 s (avg decoding throughput 72.03 tokens/s)
2025-06-26T07:24:21.179117Z  INFO vllm_rs::core::engine: [4 requests] 800 tokens generated in 11.22 s (avg decoding throughput 71.28 tokens/s)
2025-06-26T07:24:26.892348Z  INFO vllm_rs::core::engine: [4 requests] 1200 tokens generated in 16.94 s (avg decoding throughput 70.85 tokens/s)
2025-06-26T07:24:32.444857Z  INFO vllm_rs::core::engine: [4 requests] 1600 tokens generated in 22.49 s (avg decoding throughput 71.15 tokens/s)
2025-06-26T07:24:37.281107Z  INFO vllm_rs::core::engine: [4 requests] 2000 tokens generated in 27.33 s (avg decoding throughput 73.19 tokens/s)
2025-06-26T07:24:42.136550Z  INFO vllm_rs::core::engine: [4 requests] 2400 tokens generated in 32.18 s (avg decoding throughput 74.58 tokens/s)
2025-06-26T07:24:46.998532Z  INFO vllm_rs::core::engine: [4 requests] 2800 tokens generated in 37.04 s (avg decoding throughput 75.59 tokens/s)
2025-06-26T07:24:51.906541Z  INFO vllm_rs::core::engine: [4 requests] 3200 tokens generated in 41.95 s (avg decoding throughput 76.28 tokens/s)
2025-06-26T07:24:55.110979Z  INFO vllm_rs::core::engine: [4 requests] 3600 tokens generated in 45.15 s (avg decoding throughput 79.73 tokens/s)
2025-06-26T07:24:58.190300Z  INFO vllm_rs::core::engine: [4 requests] 4000 tokens generated in 48.23 s (avg decoding throughput 82.93 tokens/s)
```

## Supported Model Arch (Safetensor, GGUF)
1) LLaMa (LLaMa2/LLaMa3)
2) Qwen (Qwen2/Qwen3)

## Note
**This project is under the initial stage.**

## TODO

1. Fix bugs for batched inference on `Metal`
2. Support Multi-rank inference
3. Add more model support
4. _Add GGUF support (finished)_
5. Add an OpenAI API server (support streaming)
6. _Flash attention integration (finished for CUDA)_
7. Continuous batching

## Reference

Core concepts were coming from our another project [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm) and Python nano-vllm.