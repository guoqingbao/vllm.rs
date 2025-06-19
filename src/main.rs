use candle_core::{DType, Result};
use vllm_rs::core::engine::LLMEngine;
use vllm_rs::utils::config::{EngineConfig, SamplingParams};

fn main() -> Result<()> {
    let econfig = EngineConfig {
        model_path: "/home/data/Qwen3-8B".to_string(),
        tokenizer: None,
        tokenizer_config: None,
        num_blocks: 128,
        block_size: 32,
        max_num_seqs: 512,
        max_num_batched_tokens: 32768,
        temperature: 0.7,
        max_model_len: 32768,
        quant: None,
        kvcache_mem_gpu: Some(4096),
        num_shards: Some(1),
        device_id: None,
    };

    let mut engine = LLMEngine::new(&econfig, DType::BF16)?;
    let prompts = &[
        "What's the capital of France?",
        "Explain quantum computing in simple terms",
    ];

    let params = SamplingParams {
        temperature: 0.6,
        max_tokens: 100,
        ignore_eos: false,
        top_k: None,
        top_p: None,
    };

    let outputs = engine.generate(prompts, &params)?;
    for (i, output) in outputs.iter().enumerate() {
        println!("Prompt {}: {}", i + 1, prompts[i]);
        println!("Response: {}\n", output);
    }

    Ok(())
}
