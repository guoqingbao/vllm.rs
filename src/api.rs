use crate::core::engine::{LLMEngine, StreamItem, GLOBAL_RT};
use crate::core::GenerationOutput;
use crate::server::convert_chat_message;
pub use crate::server::{ChatMessage, MessageContent, MessageContentType};
use crate::server::{EmbeddingStrategy, ServerData};
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, ModelType, SamplingParams};
use crate::utils::image::{get_tensor_raw_data, ImageData, ImageProcessTrait, ImageProcessor};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use candle_core::{DType, Result, Tensor};
use parking_lot::RwLock;
use rustchatui::start_ui_server;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc;
use tower_http::cors::{Any, CorsLayer};

/// Rust-friendly engine wrapper for direct inference or API serving.
pub struct Engine {
    engine: Arc<RwLock<LLMEngine>>,
    econfig: EngineConfig,
}

/// Stream response wrapper for direct generation.
pub struct EngineStream {
    pub seq_id: usize,
    pub prompt_length: usize,
    pub receiver: mpsc::Receiver<StreamItem>,
}

impl ChatMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: MessageContentType::PureText(content.into()),
        }
    }

    pub fn multimodal(role: impl Into<String>, content: Vec<MessageContent>) -> Self {
        ChatMessage {
            role: role.into(),
            content: MessageContentType::Multi(content),
        }
    }
}

impl Engine {
    pub fn new(econfig: EngineConfig, dtype: DType) -> Result<Self> {
        let engine = LLMEngine::new(&econfig, dtype)?;
        Ok(Self { engine, econfig })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.econfig
    }

    pub fn engine(&self) -> Arc<RwLock<LLMEngine>> {
        self.engine.clone()
    }

    pub fn get_model_info(&self) -> (bool, String) {
        let engine = self.engine.read();
        engine.get_model_info()
    }

    pub fn generate(
        &self,
        params: SamplingParams,
        messages: Vec<ChatMessage>,
    ) -> Result<GenerationOutput> {
        let (messages, images) = self.prepare_messages(&messages)?;

        let receivers = {
            let mut engine = self.engine.write();
            engine.generate_sync(&vec![params], &vec![messages], images)?
        };

        let tokenizer = {
            let engine = self.engine.read();
            Arc::new(engine.tokenizer.clone())
        };

        let results = tokio::task::block_in_place(|| {
            GLOBAL_RT.block_on(async { LLMEngine::collect_sync_results(receivers, tokenizer).await })
        })?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| candle_core::Error::Msg("No generation output produced".into()))
    }

    pub fn generate_batch(
        &self,
        params: Vec<SamplingParams>,
        messages: Vec<Vec<ChatMessage>>,
    ) -> Result<Vec<GenerationOutput>> {
        let message_list = messages
            .iter()
            .map(|batch| self.prepare_text_only(batch))
            .collect::<Result<Vec<_>>>()?;

        let receivers = {
            let mut engine = self.engine.write();
            engine.generate_sync(&params, &message_list, None)?
        };

        let tokenizer = {
            let engine = self.engine.read();
            Arc::new(engine.tokenizer.clone())
        };

        tokio::task::block_in_place(|| {
            GLOBAL_RT.block_on(async { LLMEngine::collect_sync_results(receivers, tokenizer).await })
        })
    }

    pub fn generate_stream(
        &self,
        params: SamplingParams,
        messages: Vec<ChatMessage>,
    ) -> Result<EngineStream> {
        let (messages, images) = self.prepare_messages(&messages)?;
        let (seq_id, prompt_length, receiver) = {
            let mut engine = self.engine.write();
            engine.generate_stream(&params, &messages, images)?
        };
        Ok(EngineStream {
            seq_id,
            prompt_length,
            receiver,
        })
    }

    pub fn embed(
        &self,
        inputs: &[String],
        strategy: EmbeddingStrategy,
    ) -> Result<(Vec<Vec<f32>>, usize)> {
        let mut engine = self.engine.write();
        engine.embed(inputs, strategy)
    }

    pub fn start_server(&self, port: usize, with_ui_server: bool) -> Result<()> {
        let is_pd_server = if let Some(cfg) = &self.econfig.pd_config {
            matches!(cfg.role, crate::transfer::PdRole::Server)
        } else {
            false
        };

        let server_data = ServerData {
            engine: self.engine.clone(),
            econfig: self.econfig.clone(),
        };

        let (has_vision, model_name) = {
            let e = self.engine.read();
            e.get_model_info()
        };
        let has_vision = Arc::new(has_vision);

        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        let app = Router::new()
            .route(
                "/v1/models",
                get(|| async move {
                    let m = if *has_vision {
                        vec!["text", "image"]
                    } else {
                        vec!["text"]
                    };
                    Json(json!({
                        "object": "list",
                        "data": [
                            {
                                "id": model_name,
                                "object": "model",
                                "created": std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis() as i64,
                                "owned_by": "vllm.rs",
                                "permission": [],
                                "modalities": m,
                            }
                        ]
                    }))
                }),
            )
            .route("/v1/chat/completions", post(crate::server::server::chat_completion))
            .route("/v1/usage", get(crate::server::server::get_usage))
            .layer(cors)
            .with_state(Arc::new(server_data));

        let addr = if is_pd_server {
            crate::log_warn!("🚀 PD server started, waiting for prefill request(s)...",);
            format!("0.0.0.0:{}", 0)
        } else {
            crate::log_warn!("🚀 Chat server listening on http://0.0.0.0:{}/v1/", port);
            format!("0.0.0.0:{}", port)
        };

        GLOBAL_RT.block_on(async move {
            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

            let mut tasks = Vec::new();
            tasks.push(tokio::spawn(async move {
                if let Err(e) = axum::serve(listener, app).await {
                    eprintln!("Chat API server error: {e:?}");
                }
            }));

            if with_ui_server {
                tasks.push(tokio::spawn(async move {
                    start_ui_server((port + 1) as u16, Some(port as u16), None, None)
                        .await
                        .unwrap();
                }));
            }

            tokio::select! {
                _ = futures::future::try_join_all(tasks) => {},
                _ = tokio::signal::ctrl_c() => {
                    println!("Received CTRL+C, shutting down server...");
                },
            }
        });

        Ok(())
    }

    fn prepare_messages(&self, messages: &[ChatMessage]) -> Result<(Vec<Message>, Option<ImageData>)> {
        use crate::models::qwen3_vl::input::Qwen3VLImageProcessor;

        let img_cfg = {
            let engine = self.engine.read();
            engine.img_cfg.clone()
        };

        if img_cfg.is_none() && Self::messages_have_images(messages) {
            candle_core::bail!("Model does not support multimodal inputs.");
        }

        let mut processor: Option<Box<dyn ImageProcessTrait + Send>> = if let Some(cfg) = &img_cfg {
            if matches!(cfg.model_type, ModelType::Qwen3VL) {
                Some(Box::new(Qwen3VLImageProcessor::default(cfg)))
            } else {
                Some(Box::new(ImageProcessor::new(cfg)))
            }
        } else {
            None
        };

        let mut images: Vec<(Tensor, Vec<(usize, usize)>)> = vec![];
        let messages: Vec<Message> = messages
            .iter()
            .map(|m| convert_chat_message(m, &mut processor, &mut images))
            .collect::<Result<Vec<_>>>()?;

        let image_data = if !images.is_empty() && img_cfg.is_some() {
            let mut image_sizes = Vec::new();
            let mut image_tensors = Vec::new();
            for (t, s) in &images {
                image_tensors.push(t);
                image_sizes.extend(s);
            }
            let images_tensor = Tensor::cat(&image_tensors, 0)?;
            let (images_raw, images_shape) = get_tensor_raw_data(&images_tensor)?;
            crate::log_info!(
                "{} images detected in the chat message, combined image shape {:?}",
                images_shape[0],
                images_shape
            );
            Some(ImageData {
                raw: images_raw,
                shape: images_shape,
                patches: image_sizes,
                image_idx: 0,
            })
        } else {
            None
        };

        Ok((messages, image_data))
    }

    fn prepare_text_only(&self, messages: &[ChatMessage]) -> Result<Vec<Message>> {
        let mut output = Vec::with_capacity(messages.len());
        for msg in messages {
            let prompt = match &msg.content {
                MessageContentType::PureText(text) => text.clone(),
                MessageContentType::Multi(items) => {
                    let mut combined = String::new();
                    for item in items {
                        match item {
                            MessageContent::Text { text } => {
                                combined.push_str(text);
                                combined.push(' ');
                            }
                            MessageContent::ImageUrl { .. }
                            | MessageContent::ImageBase64 { .. } => {
                                candle_core::bail!(
                                    "Batch generation only supports text; use generate/generate_stream for multimodal."
                                );
                            }
                        }
                    }
                    combined.trim().to_string()
                }
            };
            output.push(Message {
                role: msg.role.clone(),
                content: prompt,
                num_images: 0,
            });
        }
        Ok(output)
    }

    fn messages_have_images(messages: &[ChatMessage]) -> bool {
        messages.iter().any(|msg| match &msg.content {
            MessageContentType::PureText(_) => false,
            MessageContentType::Multi(items) => items.iter().any(|item| {
                matches!(
                    item,
                    MessageContent::ImageUrl { .. } | MessageContent::ImageBase64 { .. }
                )
            }),
        })
    }
}
