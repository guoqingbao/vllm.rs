use vllm_rs::api::Engine;
use vllm_rs::server::{ChatMessage, MessageContent, MessageContentType};
use vllm_rs::utils::config::{EngineConfig, SamplingParams};

fn main() -> candle_core::Result<()> {
    let mut config = EngineConfig::for_model("Qwen/Qwen3-0.6B");
    config.max_model_len = Some(4096);
    let mut engine = Engine::new(config, Some("bf16".to_string()))?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: MessageContentType::Multi(vec![MessageContent::Text {
            text: "Say hello from the Rust API.".to_string(),
        }]),
    }];

    let params = SamplingParams::default();
    let output = engine.generate_chat(params, messages)?;
    println!("{}", output.decode_output);
    Ok(())
}
