## vLLM.rs (Minimalist vLLM for Rust)

## Usage
Install Rust compiler and execute the following demo

**CUDA (it takes some time for the first run, for building CUDA kernels):**
```
cargo run --release --features cuda -- --weight-path /Users/bob/Downloads/Qwen3-8B/ --prompts "How are you today?"
```

**Metal (Mac)**
```
cargo run --release --features metal -- --weight-path /Users/bob/Downloads/Qwen3-0.6B/ --prompts "How are you today?"
```

**Batched request (CUDA only, prompts separated by "|")**
```
cargo run --release --features cuda -- --weight-path /Users/bob/Downloads/Qwen3-8B/ --prompts "Please talk about China. | Please talk about America."
```

**In-situ quantization**
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
8450 tokens generated in 14.28 s (decoding thourghput 591.82 tokens/s)
```

## Supported Model Arch
1) LLaMa (LLaMa2/LLaMa3)
2) Qwen (Qwen2/Qwen3)

## Note
**This project is under the initial stage.**

## TODO

1. Fix bugs for batched inference on `Metal`
2. Support Multi-rank inference
3. Add more model support
4. Add GGUF support
5. Add an OpenAI API server (support streaming)
6. _Flash attention integration (finished for CUDA)_
7. Continuous batching

## Reference

Core concepts were coming from our another project [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm) and Python nano-vllm.