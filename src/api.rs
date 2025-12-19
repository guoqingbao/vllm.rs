use crate::core::engine::{LLMEngine, StreamItem, GLOBAL_RT};
use crate::core::GenerationOutput;
use crate::server::{build_messages_and_images, ChatMessage, ServerData};
use crate::transfer::PdRole;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, SamplingParams};
use crate::utils::get_dtype;
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use candle_core::Result;
use parking_lot::RwLock;
use rustchatui::start_ui_server;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc;
use tower_http::cors::{Any, CorsLayer};

pub struct Engine {
    engine: Arc<RwLock<LLMEngine>>,
    econfig: EngineConfig,
}

impl Engine {
    pub fn new(econfig: EngineConfig, dtype: Option<String>) -> Result<Self> {
        let dtype = get_dtype(dtype);
        let engine = LLMEngine::new(&econfig, dtype)?;
        Ok(Self { engine, econfig })
    }

    pub fn start_server(&self, port: usize, with_ui_server: bool) -> Result<()> {
        let is_pd_server = if let Some(cfg) = &self.econfig.pd_config {
            matches!(cfg.role, PdRole::Server)
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
            .route(
                "/v1/chat/completions",
                post(crate::server::server::chat_completion),
            )
            .route("/v1/usage", get(crate::server::server::get_usage))
            .layer(cors)
            .with_state(Arc::new(server_data));

        let addr = if is_pd_server {
            crate::log_warn!("🚀 PD server started, waiting for prefill request(s)...");
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

    pub fn generate_prompt(
        &mut self,
        params: SamplingParams,
        prompt: impl Into<String>,
    ) -> Result<GenerationOutput> {
        let messages = vec![Message::new("user".to_string(), prompt.into(), 0)];
        self.generate_messages(params, messages, None)
    }

    pub fn generate_messages(
        &mut self,
        params: SamplingParams,
        messages: Vec<Message>,
        images: Option<crate::utils::image::ImageData>,
    ) -> Result<GenerationOutput> {
        let (receivers, tokenizer) = {
            let mut engine = self.engine.write();
            (
                engine.generate_sync(&vec![params], &vec![messages], images)?,
                Arc::new(engine.tokenizer.clone()),
            )
        };

        let mut results = GLOBAL_RT
            .block_on(async { LLMEngine::collect_sync_results(receivers, tokenizer).await })?;

        results
            .pop()
            .ok_or_else(|| candle_core::Error::msg("No generation output returned"))
    }

    pub fn generate_chat(
        &mut self,
        params: SamplingParams,
        messages: Vec<ChatMessage>,
    ) -> Result<GenerationOutput> {
        let img_cfg = { self.engine.read().img_cfg.clone() };
        let (messages, image_data) = build_messages_and_images(&messages, img_cfg.as_ref())?;
        self.generate_messages(params, messages, image_data)
    }

    pub fn generate_chat_stream(
        &mut self,
        params: SamplingParams,
        messages: Vec<ChatMessage>,
    ) -> Result<EngineStream> {
        let img_cfg = { self.engine.read().img_cfg.clone() };
        let (messages, image_data) = build_messages_and_images(&messages, img_cfg.as_ref())?;

        let (seq_id, prompt_length, stream) = {
            let mut engine = self.engine.write();
            engine.generate_stream(&params, &messages, image_data)?
        };

        Ok(EngineStream {
            engine: self.engine.clone(),
            rx: stream,
            finished: false,
            seq_id,
            prompt_length,
            cancelled: false,
        })
    }

    pub fn get_num_cached_tokens(&self) -> usize {
        let engine = self.engine.read();
        engine.get_num_cached_tokens()
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let engine = self.engine.read();
        engine.get_available_kv_tokens()
    }
}

pub struct EngineStream {
    engine: Arc<RwLock<LLMEngine>>,
    rx: mpsc::Receiver<StreamItem>,
    finished: bool,
    pub seq_id: usize,
    pub prompt_length: usize,
    cancelled: bool,
}

impl EngineStream {
    pub fn cancel(&mut self) {
        self.cancelled = true;
        let mut engine_guard = self.engine.write();
        engine_guard.cancel(self.seq_id);
    }

    pub async fn recv(&mut self) -> Option<StreamItem> {
        if self.finished {
            return None;
        }
        let item = self.rx.recv().await;
        if matches!(item, Some(StreamItem::Done(_) | StreamItem::Error(_))) {
            self.finished = true;
        }
        item
    }

    pub fn recv_blocking(&mut self) -> Option<StreamItem> {
        if self.finished {
            return None;
        }
        let item = GLOBAL_RT.block_on(self.rx.recv());
        if matches!(item, Some(StreamItem::Done(_) | StreamItem::Error(_))) {
            self.finished = true;
        }
        item
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}
