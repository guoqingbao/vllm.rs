# Rust crate usage

This crate exposes a Rust-facing API for loading models, running generation, and optionally running
an OpenAI-compatible service without changing the existing project structure.

## Add dependency

```toml
[dependencies]
vllm-rs = { path = "/path/to/vllm.rs" }
```

Use the same Cargo features you would use for the CLI (`cuda`, `metal`, `nccl`, etc.).

## Direct generation (text)

```rust
use vllm_rs::api::Engine;
use vllm_rs::utils::config::{EngineConfig, SamplingParams};

fn main() -> candle_core::Result<()> {
    let mut config = EngineConfig::for_model("Qwen/Qwen3-0.6B");
    config.max_model_len = Some(4096);
    let mut engine = Engine::new(config, Some("bf16".to_string()))?;

    let mut params = SamplingParams::default();
    params.temperature = Some(0.7);

    let output = engine.generate_prompt(params, "Hello from Rust!")?;
    println!("{}", output.decode_output);

    Ok(())
}
```

## Multimodal request (URL or base64)

```rust
use vllm_rs::api::Engine;
use vllm_rs::server::{ChatMessage, MessageContent, MessageContentType};
use vllm_rs::utils::config::{EngineConfig, SamplingParams};

fn main() -> candle_core::Result<()> {
    let config = EngineConfig::for_model("Qwen/Qwen3-VL-8B-Instruct");
    let mut engine = Engine::new(config, Some("bf16".to_string()))?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: MessageContentType::Multi(vec![
            MessageContent::Text {
                text: "Describe this image:".to_string(),
            },
            MessageContent::ImageUrl {
                image_url: "https://example.com/cat.png".to_string(),
            },
        ]),
    }];

    let params = SamplingParams::default();
    let output = engine.generate_chat(params, messages)?;
    println!("{}", output.decode_output);

    Ok(())
}
```

## Serve API

```rust
use vllm_rs::api::Engine;
use vllm_rs::utils::config::EngineConfig;

fn main() -> candle_core::Result<()> {
    let config = EngineConfig::for_model("Qwen/Qwen3-0.6B");
    let engine = Engine::new(config, Some("bf16".to_string()))?;

    engine.start_server(8000, true)?;
    Ok(())
}
```

## Multi-rank / multi-GPU

Provide `device_ids` in `EngineConfig` (e.g. `Some(vec![0, 1])`) along with the same CUDA/NCCL
features you use for the CLI. The Rust API reuses the same engine and scheduler path.
