use vllm_rs::api::{EngineBuilder, ModelRepo};
use vllm_rs::server::{ChatMessage, MessageContent, MessageContentType};
use vllm_rs::utils::config::SamplingParams;

fn main() -> candle_core::Result<()> {
    let mut engine =
        EngineBuilder::new(ModelRepo::ModelID(("Qwen/Qwen3-0.6B".to_string(), None))).build()?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: MessageContentType::Multi(vec![MessageContent::Text {
            text: "Say hello from the Rust API.".to_string(),
        }]),
    }];

    let params = SamplingParams::default();
    let output = engine.generate(params, messages)?;
    println!("{}", output.decode_output);
    Ok(())
}
