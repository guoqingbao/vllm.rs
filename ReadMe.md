## vLLM.rs (Minimalist vLLM for Rust)

## Usage
Install Rust compiler and execute demo

```
cargo run
```

Change model_path in `EngineConfig` (qwen3 models)

Sample result (Qwen3-8B BF16)

```
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.14s
     Running `target/debug/vllm-rs`
seq 0 finished
Prompt 1: What's the capital of France?
Response: <think>
Okay, the user is asking for the capital of France. Let me think. I know that France is a country in Europe, and its capital is a well-known city. The most common answer is Paris. But wait, I should make sure there's no confusion with other cities. For example, some people might think it's Lyon or Marseille, but those are major cities, not the capital. Let me confirm. Yes, Paris has been the capital of France since the 3rd
```

## Note
**This project is under the initial stage.**

## TODO

1. Fix bugs for batched inference
2. Add more model support
3. Add GGUF support
4. Add an OpenAI API server (support streaming)
5. Support Metal
6. Support AMD GPU
7. Flash attention integration
8. Continuous batching

## Reference

Core concepts were coming from our another project [Candle-vLLM](https://github.com/EricLBuehler/candle-vllm) and Python nano-vllm.