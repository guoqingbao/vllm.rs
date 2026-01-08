use clap::Parser;
use serde::{Deserialize, Serialize};
pub mod claude_server;
pub mod logger;
pub mod parser;
pub mod server;
pub mod streaming;
use crate::core::engine::LLMEngine;
use crate::server::streaming::Streamer;
use crate::transfer::PdRole;
use crate::utils::chat_template::Message;
use crate::utils::config::EngineConfig;
use crate::utils::image::{
    compute_tokens_per_image, get_tensor_raw_data, load_image_from_base64, load_image_from_url,
    ImageData, ImageProcessConfig, ImageProcessTrait, IMAGE_PLACEHOLDER,
};
use axum::http::{self, StatusCode};
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use candle_core::{Result, Tensor};
use colored::*;
use local_ip_address::local_ip;
use parking_lot::RwLock;
use rustchatui::start_ui_server;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_k: Option<isize>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    #[serde(alias = "enable_thinking")]
    pub thinking: Option<bool>,
    pub stream: Option<bool>,
    pub session_id: Option<String>,
    /// Tools available for the model to call
    #[serde(default)]
    pub tools: Option<Vec<crate::tools::Tool>>,
    /// How the model should choose which tool to call
    #[serde(default)]
    pub tool_choice: Option<crate::tools::ToolChoice>,
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
#[serde(untagged)]
pub enum ImageUrlContent {
    Url(String),
    Object {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

impl ImageUrlContent {
    pub fn url(&self) -> &str {
        match self {
            Self::Url(url) => url,
            Self::Object { url, .. } => url,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum MessageContent {
    // pure text (classic chat format)
    #[serde(alias = "input_text", alias = "text")]
    Text { text: String },

    // URL image: "image_url": "https://..."
    #[serde(alias = "image_url")]
    ImageUrl { image_url: ImageUrlContent },

    // Base64 format: "data:image/jpeg;base64,xxxxx"
    #[serde(alias = "image_base64")]
    ImageBase64 { image_base64: String },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum MessageContentType {
    PureText(String),
    Single(MessageContent),
    Multi(Vec<MessageContent>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContentType>,
    /// Tool calls made by the assistant
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tools::ToolCall>>,
    /// Tool call ID when role is "tool"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create a simple text message
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(MessageContentType::PureText(content.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message with tool calls
    pub fn with_tool_calls(tool_calls: Vec<crate::tools::ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Create a tool result message
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(MessageContentType::PureText(content.into())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
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
    pub message: ChatResponseMessage,
    pub finish_reason: Option<String>,
}

/// Message in the response (may contain tool calls)
#[derive(Serialize)]
pub struct ChatResponseMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tools::ToolCall>>,
}

#[derive(Serialize, Debug)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceChunk>,
    pub usage: Option<Usage>,
}

#[derive(Serialize, Debug)]
pub struct ErrorMsg {
    pub message: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct ChatChoiceChunk {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    pub error: Option<Vec<ErrorMsg>>,
}

#[derive(Serialize, Debug)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tools::ToolCall>>,
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

// === Tokenize API ===

/// Input for tokenize request - either plain text or chat messages
#[derive(Deserialize)]
#[serde(untagged)]
pub enum TokenizeInput {
    /// Chat messages input (will apply chat template)
    Messages { messages: Vec<ChatMessage> },
    /// Plain text input
    Text { prompt: String },
}

/// Request body for /tokenize endpoint
#[derive(Deserialize)]
pub struct TokenizeRequest {
    pub model: Option<String>,
    #[serde(flatten)]
    pub input: TokenizeInput,
    /// Whether to add special tokens (default: true)
    #[serde(default)]
    pub add_special_tokens: Option<bool>,
}

/// Response from /tokenize endpoint
#[derive(Serialize)]
pub struct TokenizeResponse {
    /// List of token IDs
    pub tokens: Vec<u32>,
    /// Number of tokens
    pub count: usize,
    /// Maximum model context length (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
}

// === Detokenize API ===

/// Request body for /detokenize endpoint
#[derive(Deserialize)]
pub struct DetokenizeRequest {
    pub model: Option<String>,
    /// Token IDs to decode
    pub tokens: Vec<u32>,
    /// Whether to skip special tokens in output (default: true)
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
}

/// Response from /detokenize endpoint
#[derive(Serialize)]
pub struct DetokenizeResponse {
    /// Decoded text
    pub prompt: String,
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
    pub mcp_manager: Option<Arc<crate::mcp::McpClientManager>>,
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
    Tokenize(TokenizeResponse),
    Detokenize(DetokenizeResponse),
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
            ChatResponder::Tokenize(s) => Json(s).into_response(),
            ChatResponder::Detokenize(s) => Json(s).into_response(),
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

    #[arg(long, default_value_t = false)]
    pub no_flash_attn: bool,

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

    #[arg(long)]
    pub tool_prompt: Option<String>,

    #[arg(long, default_value_t = false)]
    pub prefix_cache: bool,

    /// Max cached prefix size in tokens (rounded down to block size).
    #[arg(long, default_value = None)]
    pub prefix_cache_max_tokens: Option<usize>,

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

    /// MCP server command to spawn for tool discovery and calls
    #[arg(long, default_value = None)]
    pub mcp_command: Option<String>,

    /// MCP config file path for multi-server setups
    #[arg(long, default_value = None)]
    pub mcp_config: Option<String>,

    /// MCP server arguments (comma-separated)
    #[arg(long, value_delimiter = ',', default_value = None)]
    pub mcp_args: Option<Vec<String>>,
}

/// Result of executing tool calls via MCP
#[allow(dead_code)]
pub struct ToolExecutionResult {
    /// Messages to add for follow-up (assistant tool_calls + tool results)
    followup_messages: Vec<ChatMessage>,
    /// The tool calls that were executed
    tool_calls: Vec<crate::tools::ToolCall>,
}

/// Default timeout for individual tool calls (60 seconds)
const TOOL_CALL_TIMEOUT_SECS: u64 = 60;

/// Execute tool calls via MCP manager and return messages for follow-up generation
/// Each tool call has a timeout of TOOL_CALL_TIMEOUT_SECS seconds
pub async fn execute_mcp_tool_calls_async(
    tool_calls: Vec<crate::tools::ToolCall>,
    mcp_manager: std::sync::Arc<crate::mcp::McpClientManager>,
    base_messages: Vec<ChatMessage>,
) -> ToolExecutionResult {
    let mut followup_messages = base_messages.clone();
    followup_messages.push(ChatMessage::with_tool_calls(tool_calls.clone()));

    for call in &tool_calls {
        let args_value: serde_json::Value = serde_json::from_str(&call.function.arguments)
            .unwrap_or_else(|_| serde_json::json!({"raw": call.function.arguments}));
        let args_map = args_value
            .as_object()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<HashMap<String, serde_json::Value>>();

        let call_name = call.function.name.clone();
        let call_id = call.id.clone();
        let mcp_manager_clone = mcp_manager.clone();
        crate::log_info!(
            "Executing tool call: {} with args {:?}",
            call_name,
            args_map
        );

        let start = std::time::Instant::now();

        // Execute tool call with timeout using spawn_blocking
        let timeout_duration = std::time::Duration::from_secs(TOOL_CALL_TIMEOUT_SECS);
        let tool_result = match tokio::time::timeout(
            timeout_duration,
            tokio::task::spawn_blocking(move || mcp_manager_clone.call_tool(&call_name, args_map)),
        )
        .await
        {
            Ok(Ok(Ok(result))) => {
                // Success: spawn_blocking succeeded, call_tool succeeded
                let elapsed = start.elapsed();
                crate::log_info!(
                    "Tool '{}' completed in {:.2}s",
                    call.function.name,
                    elapsed.as_secs_f32()
                );
                let content = result
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        crate::mcp::ToolContent::Text { text } => Some(text.clone()),
                        crate::mcp::ToolContent::Resource { text, .. } => text.clone(),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                ChatMessage::tool_result(call_id, content)
            }
            Ok(Ok(Err(err))) => {
                // Tool execution failed
                let elapsed = start.elapsed();
                crate::log_error!(
                    "Tool '{}' failed after {:.2}s: {:?}",
                    call.function.name,
                    elapsed.as_secs_f32(),
                    err
                );
                ChatMessage::tool_result(call_id, format!("Tool execution failed: {err}"))
            }
            Ok(Err(join_err)) => {
                // spawn_blocking panicked
                crate::log_error!(
                    "Tool '{}' task panicked: {:?}",
                    call.function.name,
                    join_err
                );
                ChatMessage::tool_result(call_id, format!("Tool execution panicked: {join_err}"))
            }
            Err(_timeout_err) => {
                // Timeout occurred
                crate::log_error!(
                    "Tool '{}' timed out after {}s",
                    call.function.name,
                    TOOL_CALL_TIMEOUT_SECS
                );
                ChatMessage::tool_result(
                    call_id,
                    format!("Tool execution timed out after {}s", TOOL_CALL_TIMEOUT_SECS),
                )
            }
        };
        followup_messages.push(tool_result);
    }

    ToolExecutionResult {
        followup_messages,
        tool_calls,
    }
}

pub fn convert_chat_message(
    msg: &ChatMessage,
    processor: &mut Option<Box<dyn ImageProcessTrait + Send>>,
    images_tensors: &mut Vec<(Tensor, Vec<(usize, usize)>)>,
) -> Result<Message> {
    let role = msg.role.clone();
    let mut prompt = String::new();
    let mut images = Vec::new();

    // Handle tool call messages specially
    if role == "tool" {
        if let Some(tool_call_id) = &msg.tool_call_id {
            if let Some(content) = &msg.content {
                let mut tool_text = String::new();
                match content {
                    MessageContentType::PureText(text) => {
                        tool_text.push_str(text);
                    }
                    MessageContentType::Single(item) => {
                        if let MessageContent::Text { text } = item {
                            tool_text.push_str(text);
                        }
                    }
                    MessageContentType::Multi(items) => {
                        for item in items {
                            if let MessageContent::Text { text } = item {
                                tool_text.push_str(text);
                                tool_text.push(' ');
                            }
                        }
                    }
                }
                let tool_text_trimmed = tool_text.trim();
                if !tool_text_trimmed.is_empty() {
                    prompt = format!("[Tool Result for {}]: {}", tool_call_id, tool_text_trimmed);
                }
            }
        }
        return Ok(Message::new(role, prompt.trim().to_owned(), 0));
    }

    // // Handle assistant messages with tool calls
    // if msg.tool_calls.is_some() {
    //     if let Some(tool_calls) = &msg.tool_calls {
    //         for tc in tool_calls {
    //             prompt.push_str(&format!(
    //                 "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>\n",
    //                 tc.function.name, tc.function.arguments
    //             ));
    //         }
    //     }
    //     return Ok(Message::new(role, prompt.trim().to_owned(), 0));
    // }

    // Normal message handling
    if let Some(content) = &msg.content {
        match content {
            MessageContentType::PureText(text) => {
                prompt.push_str(text);
            }
            MessageContentType::Single(item) => {
                append_message_item(item, &mut prompt, &mut images)?;
                prompt.push(' '); // keep spacing readable
            }
            MessageContentType::Multi(items) => {
                for item in items {
                    append_message_item(item, &mut prompt, &mut images)?;
                    prompt.push(' '); // keep spacing readable
                }
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

fn append_message_item(
    item: &MessageContent,
    prompt: &mut String,
    images: &mut Vec<image::DynamicImage>,
) -> Result<()> {
    match item {
        MessageContent::Text { text } => {
            prompt.push_str(text);
        }
        MessageContent::ImageUrl { image_url } => {
            let url = image_url.url();
            let img = if url.starts_with("data:") {
                let img = load_image_from_base64(url)?;
                crate::log_info!("Chat image decoded: {} x {}", img.width(), img.height());
                img
            } else {
                let img = load_image_from_url(url)?;
                crate::log_info!("Chat image downloaded: {} x {}", img.width(), img.height());
                img
            };
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
    Ok(())
}

pub fn build_messages_and_images(
    messages: &[ChatMessage],
    img_cfg: Option<&ImageProcessConfig>,
) -> Result<(Vec<Message>, Option<ImageData>)> {
    use crate::models::qwen3_vl::input::Qwen3VLImageProcessor;
    use crate::utils::config::ModelType;
    use crate::utils::image::ImageProcessor;

    let mut processor: Option<Box<dyn ImageProcessTrait + Send>> = if let Some(cfg) = img_cfg {
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
        let cfg = img_cfg.unwrap();
        let tokens_per_image = compute_tokens_per_image(cfg, &image_sizes);
        Some(ImageData {
            raw: images_raw,
            shape: images_shape,
            patches: image_sizes,
            image_idx: 0,
            image_token_offset: 0,
            tokens_per_image,
            image_token_id: cfg.image_token_id,
        })
    } else {
        None
    };

    Ok((messages, image_data))
}

pub async fn run_server(
    engine: Arc<RwLock<LLMEngine>>,
    econfig: EngineConfig,
    port: usize,
    with_ui_server: bool,
) -> Result<()> {
    use axum::extract::DefaultBodyLimit;
    let (has_vision, model_name) = {
        let e = engine.read();
        e.get_model_info()
    };
    let has_vision = Arc::new(has_vision);

    let is_pd_server = if let Some(cfg) = &econfig.pd_config {
        matches!(cfg.role, PdRole::Server)
    } else {
        false
    };

    let mcp_manager_config = if let Some(path) = &econfig.mcp_config {
        match crate::mcp::manager::McpManagerConfig::from_file(path) {
            Ok(cfg) => Some(cfg),
            Err(err) => {
                crate::log_error!("Failed to load MCP config file: {:?}", err);
                None
            }
        }
    } else if let Some(command) = econfig.mcp_command.clone() {
        Some(crate::mcp::manager::McpManagerConfig::from_single(
            crate::mcp::manager::McpToolConfig::new(
                command,
                econfig.mcp_args.clone().unwrap_or_default(),
            ),
        ))
    } else {
        None
    };

    let mcp_manager = if let Some(cfg) = mcp_manager_config {
        match crate::mcp::McpClientManager::new(cfg) {
            Ok(manager) => Some(Arc::new(manager)),
            Err(err) => {
                crate::log_error!("Failed to start MCP client manager: {:?}", err);
                None
            }
        }
    } else {
        None
    };

    let server_data = ServerData {
        engine,
        econfig,
        mcp_manager,
    };

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
                    vec!["text", "embedding"]
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
        .route("/v1/chat/completions", post(server::chat_completion))
        .route("/v1/messages", post(claude_server::messages))
        .route(
            "/v1/messages/count_tokens",
            post(claude_server::count_tokens),
        )
        .route("/v1/embeddings", post(server::create_embeddings))
        .route("/v1/usage", get(server::get_usage))
        .route("/tokenize", post(server::tokenize))
        .route("/detokenize", post(server::detokenize))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // 100MB body size limit
        .layer(cors)
        .with_state(Arc::new(server_data));

    let addr = if is_pd_server {
        crate::log_warn!("🚀 PD server started, waiting for prefill request(s)...",);
        format!("0.0.0.0:{}", 0)
    } else {
        let ip = local_ip().unwrap_or("127.0.0.1".parse().unwrap());
        let local_url = format!("http://localhost:{port}/v1/");
        let lan_url = format!("http://{ip}:{port}/v1/");

        let api_server_url = format!(
            "🧠 API server running at:\n   -  {} (Local Access) \n   -  {} (Remote Access)",
            local_url, lan_url
        );
        println!("{}", api_server_url.cyan());

        println!(
            "{}",
            format!("📡 Supported endpoints (OpenAI/Claude):").yellow()
        );
        println!("{}", format!("   - POST /v1/chat/completions").yellow());
        println!("{}", format!("   - POST /v1/messages").yellow());
        println!(
            "{}",
            format!("   - POST /v1/messages/count_tokens").yellow()
        );
        println!("{}", format!("   - POST /v1/embeddings").yellow());
        println!("{}", format!("   - GET  /v1/models").yellow());
        println!("{}", format!("   - GET  /v1/usage").yellow());
        println!("{}", format!("   - POST /tokenize").yellow());
        println!("{}", format!("   - POST /detokenize").yellow());
        println!("");
        format!("0.0.0.0:{}", port)
    };

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let mut tasks = Vec::new();
    tasks.push(tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("API server error: {e:?}");
        }
    }));

    if with_ui_server {
        tasks.push(tokio::spawn(async move {
            start_ui_server((port + 1) as u16, Some(port as u16), None, None)
                .await
                .unwrap();
        }));
    }

    futures::future::try_join_all(tasks)
        .await
        .map_err(candle_core::Error::wrap)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_without_images() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContentType::PureText("hello world".to_string())),
            tool_calls: None,
            tool_call_id: None,
        }];

        let (converted, images) = build_messages_and_images(&messages, None).unwrap();

        assert!(images.is_none());
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[0].content, "hello world");
        assert_eq!(converted[0].num_images, 0);
    }

    #[test]
    fn test_chat_message_helpers() {
        let text_msg = ChatMessage::text("user", "Hello!");
        assert_eq!(text_msg.role, "user");
        assert!(text_msg.content.is_some());

        let tool_result = ChatMessage::tool_result("call_123", r#"{"result": 42}"#);
        assert_eq!(tool_result.role, "tool");
        assert_eq!(tool_result.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_tokenize_request_text_parsing() {
        let json = r#"{"prompt": "Hello, world!"}"#;
        let request: TokenizeRequest = serde_json::from_str(json).unwrap();
        match request.input {
            TokenizeInput::Text { prompt } => assert_eq!(prompt, "Hello, world!"),
            _ => panic!("Expected TokenizeInput::Text"),
        }
    }

    #[test]
    fn test_tokenize_request_messages_parsing() {
        let json = r#"{"messages": [{"role": "user", "content": "Hello"}]}"#;
        let request: TokenizeRequest = serde_json::from_str(json).unwrap();
        match request.input {
            TokenizeInput::Messages { messages } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].role, "user");
            }
            _ => panic!("Expected TokenizeInput::Messages"),
        }
    }

    #[test]
    fn test_tokenize_request_with_options() {
        let json = r#"{"prompt": "test", "add_special_tokens": false}"#;
        let request: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.add_special_tokens, Some(false));
    }

    #[test]
    fn test_detokenize_request_parsing() {
        let json = r#"{"tokens": [1, 2, 3, 4]}"#;
        let request: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_detokenize_request_with_options() {
        let json = r#"{"tokens": [1, 2], "skip_special_tokens": false}"#;
        let request: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.skip_special_tokens, Some(false));
    }

    #[test]
    fn test_tokenize_response_serialization() {
        let response = TokenizeResponse {
            tokens: vec![1, 2, 3],
            count: 3,
            max_model_len: Some(4096),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"count\":3"));
        assert!(json.contains("\"tokens\":[1,2,3]"));
    }

    #[test]
    fn test_detokenize_response_serialization() {
        let response = DetokenizeResponse {
            prompt: "Hello, world!".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"prompt\":\"Hello, world!\""));
    }
}
