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
use vllm_rs::api::{EngineBuilder, ModelRepo};
use vllm_rs::server::{ChatMessage, MessageContent, MessageContentType};
use vllm_rs::utils::config::SamplingParams;

fn main() -> candle_core::Result<()> {
    let mut engine = EngineBuilder::new(ModelRepo::ModelID((
        "Qwen/Qwen3-0.6B".to_string(),
        None,
    )))
    .build()?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: MessageContentType::Multi(vec![MessageContent::Text {
            text: "Hello from Rust!".to_string(),
        }]),
    }];

    let params = SamplingParams::default();
    let output = engine.generate(params, messages)?;
    println!("{}", output.decode_output);

    Ok(())
}
```

## Multimodal request (URL or base64)

```rust
use vllm_rs::api::{EngineBuilder, ModelRepo};
use vllm_rs::server::{ChatMessage, MessageContent, MessageContentType};
use vllm_rs::utils::config::SamplingParams;

fn main() -> candle_core::Result<()> {
    let mut engine = EngineBuilder::new(ModelRepo::ModelID((
        "Qwen/Qwen3-VL-8B-Instruct".to_string(),
        None,
    )))
    .build()?;

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
    let output = engine.generate(params, messages)?;
    println!("{}", output.decode_output);

    Ok(())
}
```

## Serve API

```rust
use vllm_rs::api::{EngineBuilder, ModelRepo};

fn main() -> candle_core::Result<()> {
    let mut engine = EngineBuilder::new(ModelRepo::ModelID((
        "Qwen/Qwen3-0.6B".to_string(),
        None,
    )))
    .build()?;

    engine.start_server(8000, true, false)?;
    Ok(())
}
```

## Multi-rank / multi-GPU

Provide `device_ids` with `with_multirank` (e.g. `"0,1"`) along with the same CUDA/NCCL features
you use for the CLI. The Rust API reuses the same engine and scheduler path.

```rust
use vllm_rs::api::{EngineBuilder, ModelRepo};

fn main() -> candle_core::Result<()> {
    let mut engine = EngineBuilder::new(ModelRepo::ModelFile(vec![
        "/path/Qwen3-VL-8B-Instruct-GGUF-Q4_KM.gguf".to_string(),
    ]))
    .with_multirank("0,1")?
    .build()?;

    engine.start_server(8000, true, true)?;
    Ok(())
}
```
