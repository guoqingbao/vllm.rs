use clap::Parser;
use serde::{Deserialize, Serialize};
pub mod server;
pub mod streaming;
use crate::core::engine::LLMEngine;
use crate::server::streaming::Streamer;
use crate::utils::chat_template::Message;
use crate::utils::config::EngineConfig;
use crate::utils::image::{
    load_image_from_base64, load_image_from_url, ImageProcessTrait, IMAGE_PLACEHOLDER,
};
use axum::extract::Json;
use axum::http::{self, StatusCode};
use axum::response::{IntoResponse, Sse};
use candle_core::{Result, Tensor};
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

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

impl Default for EncodingFormat {
    fn default() -> Self {
        Self::Float
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingStrategy {
    Mean,
    Last,
}

impl Default for EmbeddingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum MessageContent {
    // pure text (classic chat format)
    #[serde(alias = "input_text")]
    Text { text: String },

    // URL image: "image_url": "https://..."
    #[serde(alias = "image_url")]
    ImageUrl { image_url: String },

    // Base64 format: "data:image/jpeg;base64,xxxxx"
    #[serde(alias = "image_base64")]
    ImageBase64 { image_base64: String },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum MessageContentType {
    PureText(String),
    Multi(Vec<MessageContent>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContentType,
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
pub struct ErrorMsg {
    pub message: Option<String>,
}

#[derive(Serialize)]
pub struct ChatChoiceChunk {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    pub error: Option<Vec<ErrorMsg>>,
}

#[derive(Serialize)]
pub struct Delta {
    pub content: Option<String>,
}

#[derive(Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: EmbeddingOutput,
    pub index: usize,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum EmbeddingOutput {
    Vector(Vec<f32>),
    Base64(String),
}

#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

#[derive(Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    #[serde(default)]
    pub embedding_type: EmbeddingStrategy,
}

#[derive(Deserialize)]
pub struct UsageQuery {
    pub session_id: Option<String>,
    // pub user_id: Option<String>,
    // pub detail: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct UsageResponse {
    pub token_used: usize,
    pub max_model_len: usize,
    pub used_kvcache_tokens: usize,
    pub total_kv_cache_tokens: usize,
    pub swap_used: f32,
    pub total_swap_memory: f32,
    pub session_status: String,
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
    Usage(UsageResponse),
    Embedding(EmbeddingResponse),
    ModelError(String),
    InternalError(String),
    ValidationError(String),
}

impl IntoResponse for ChatResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatResponder::Streamer(s) => s.into_response(),
            ChatResponder::Completion(s) => Json(s).into_response(),
            ChatResponder::Usage(s) => Json(s).into_response(),
            ChatResponder::Embedding(s) => Json(s).into_response(),
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

    /// for batch performance test
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

    #[arg(long, default_value_t = false)]
    pub fp8_kvcache: bool, //kv cache and attention with quantization

    // After model loading, the percentage of the remaining gpu memory for kvcache
    #[arg(long, default_value = None)]
    pub kv_fraction: Option<f32>,

    #[arg(long, default_value = None)]
    pub cpu_mem_fold: Option<f32>, //the percentage of cpu vs. gpu kvcache size

    #[arg(long, default_value_t = false)]
    pub pd_server: bool, //PD server mode

    #[arg(long, default_value_t = false)]
    pub pd_client: bool, //PD client mode

    #[arg(long)]
    pub pd_url: Option<String>, //Url for PD server mode (server in remote)

    #[arg(long, default_value_t = false)]
    pub ui_server: bool, //Start the web chat
}

pub fn convert_chat_message(
    msg: &ChatMessage,
    processor: &mut Option<Box<dyn ImageProcessTrait + Send>>,
    images_tensors: &mut Vec<(Tensor, Vec<(usize, usize)>)>,
) -> Result<Message> {
    let role = msg.role.clone();
    let mut prompt = String::new();
    let mut images = Vec::new();

    match &msg.content {
        MessageContentType::PureText(text) => {
            prompt.push_str(text);
        }
        MessageContentType::Multi(items) => {
            for item in items {
                match item {
                    MessageContent::Text { text } => {
                        prompt.push_str(text);
                    }
                    MessageContent::ImageUrl { image_url } => {
                        let img = load_image_from_url(image_url)?;
                        crate::log_info!(
                            "Chat image downloaded: {} x {}",
                            img.width(),
                            img.height()
                        );
                        prompt.push_str(&IMAGE_PLACEHOLDER);
                        images.push(img);
                    }
                    MessageContent::ImageBase64 { image_base64 } => {
                        let img = load_image_from_base64(image_base64)?;
                        crate::log_info!("Chat image decoded: {} x {}", img.width(), img.height());
                        prompt.push_str(&IMAGE_PLACEHOLDER);
                        images.push(img);
                    }
                }
                prompt.push(' '); // keep spacing readable
            }
        }
    }

    if !images.is_empty() && processor.is_some() {
        if let Some(processor) = processor.as_mut() {
            let (images_tensor, image_sizes) = processor.process_inputs(&mut prompt, &images)?;
            images_tensors.push((images_tensor, image_sizes));
        }
    }

    Ok(Message::new(role, prompt.trim().to_owned(), images.len()))
}
