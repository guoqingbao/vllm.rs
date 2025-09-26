use clap::Parser;
use serde::{Deserialize, Serialize};
#[cfg(not(feature = "python"))]
pub mod server;
pub mod streaming;
use crate::core::engine::LLMEngine;
use crate::server::streaming::Streamer;
use crate::utils::config::EngineConfig;
use axum::extract::Json;
use axum::http::{self, StatusCode};
use axum::response::{IntoResponse, Sse};
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_k: Option<isize>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stream: Option<bool>,
    pub session_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceChunk>,
    pub usage: Option<Usage>,
}

#[derive(Serialize)]
pub struct ChatChoiceChunk {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Delta {
    pub content: Option<String>,
}

pub struct ServerData {
    pub engine: Arc<RwLock<LLMEngine>>,
    pub econfig: EngineConfig,
}

trait ErrorToResponse: Serialize {
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

#[derive(Serialize)]
struct JsonError {
    message: String,
}

impl JsonError {
    fn new(message: String) -> Self {
        Self { message }
    }
}
impl ErrorToResponse for JsonError {}

pub enum ChatResponder {
    Streamer(Sse<Streamer>),
    Completion(ChatCompletionResponse),
    ModelError(String),
    InternalError(String),
    ValidationError(String),
}

impl IntoResponse for ChatResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatResponder::Streamer(s) => s.into_response(),
            ChatResponder::Completion(s) => Json(s).into_response(),
            ChatResponder::InternalError(e) => {
                JsonError::new(e).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ChatResponder::ValidationError(e) => {
                JsonError::new(e).to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            ChatResponder::ModelError(msg) => {
                JsonError::new(msg).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Maximum number of concurrent sequences to allow, default 1 for interactive chat
    #[arg(long, default_value_t = 1)]
    pub max_num_seqs: usize,

    /// Size of a block
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// if weight_path is passed, it will ignore the model_id
    #[arg(long = "m")]
    pub model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long = "w")]
    pub weight_path: Option<String>,

    /// gguf file path or gguf file name when model_id is given
    #[arg(long = "f")]
    pub weight_file: Option<String>,

    pub hf_token: Option<String>,

    pub hf_token_path: Option<String>,

    #[arg(long)]
    pub dtype: Option<String>,

    #[arg(long, default_value_t = false)]
    pub cpu: bool,

    #[arg(long = "d", value_delimiter = ',')]
    pub device_ids: Option<Vec<usize>>,

    #[arg(long, default_value_t = false)]
    pub log: bool,

    #[arg(long, value_delimiter = '|')]
    pub prompts: Option<Vec<String>>,

    // in-site quantization, e.g. q4_k, q2_k, q8_0, etc.
    // if not provided, it will not perform in-situ quantization for the original model
    // do not use this option if you are using a gguf file
    #[arg(long, default_value = None)]
    pub isq: Option<String>,

    #[arg(long = "i", default_value_t = false)]
    pub interactive: bool,

    /// max tokens for each request
    #[arg(long, default_value_t = 16384)]
    pub max_tokens: usize,

    /// for batch performance tetst
    #[arg(long, default_value = None)]
    pub batch: Option<usize>,

    #[arg(long, default_value = None)]
    pub temperature: Option<f32>,

    #[arg(long, default_value = None)]
    pub top_k: Option<isize>,

    #[arg(long, default_value = None)]
    pub top_p: Option<f32>,

    #[arg(long, default_value = None)]
    pub frequency_penalty: Option<f32>,

    #[arg(long, default_value = None)]
    pub presence_penalty: Option<f32>,

    #[arg(long, default_value = None)]
    pub seed: Option<u64>, //seed for reproduce the results

    #[arg(long, default_value_t = false)]
    pub context_cache: bool,

    #[arg(long, default_value_t = false)]
    pub server: bool, //server mode

    #[arg(long, default_value_t = 8000)]
    pub port: usize,
}
