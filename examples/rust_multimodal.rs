use vllm_rs::{ChatMessage, Engine, EngineConfig, MessageContent, SamplingParams, get_dtype};

fn main() -> candle_core::Result<()> {
    let econfig = EngineConfig::new(
        Some("Qwen/Qwen3-VL-8B-Instruct".to_string()),
        None,
        None,
        None,
        None,
        Some(2),
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
    let messages = vec![ChatMessage::multimodal(
        "user",
        vec![
            MessageContent::Text {
                text: "Describe the image in one sentence.".to_string(),
            },
            MessageContent::ImageUrl {
                image_url: "https://example.com/demo.png".to_string(),
            },
        ],
    )];

    let output = engine.generate(params, messages)?;
    println!("{}", output.decode_output);
    Ok(())
}
