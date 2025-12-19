use vllm_rs::{ChatMessage, Engine, EngineConfig, SamplingParams, get_dtype};

fn main() -> candle_core::Result<()> {
    let econfig = EngineConfig::new(
        Some("Qwen/Qwen2.5-7B-Instruct".to_string()),
        None,
        None,
        None,
        None,
        Some(4),
        None,
        Some(8192),
        Some(256),
        None,
        None,
        Some(vec![0]),
        None,
        None,
        Some(false),
        None,
        Some(true),
        None,
        None,
        None,
    );

    let engine = Engine::new(econfig, get_dtype(Some("bf16".to_string())))?;
    let params = SamplingParams::new_with_max_tokens(128);
    let messages = vec![ChatMessage::text("user", "Explain vLLM.rs in one sentence.")];
    let output = engine.generate(params, messages)?;
    println!("{}", output.decode_output);
    Ok(())
}
