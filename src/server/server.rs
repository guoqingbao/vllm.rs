// src/server/server.rs
use super::logger::ChatCompletionLogger;
use super::{
    build_messages_and_images,
    streaming::{ChatResponse, Streamer, StreamingStatus},
    ChatResponder, DetokenizeRequest, DetokenizeResponse, EmbeddingRequest, EmbeddingResponse,
    EncodingFormat, TokenizeInput, TokenizeRequest, TokenizeResponse,
};
use super::{
    ChatChoice, ChatChoiceChunk, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChatResponseMessage, Delta, EmbeddingData,
    EmbeddingOutput, EmbeddingUsage, ErrorMsg, ServerData, Usage, UsageQuery, UsageResponse,
};
use crate::core::engine::{LLMEngine, StreamItem};
use crate::server::parser::{ParserState, StreamResult, StreamToolParser};
use crate::tools::helpers::{
    build_tool_schema_map, filter_tool_calls, log_tool_calls, resolve_tools,
};
use crate::tools::parser::ToolParser;
use crate::tools::{ToolChoice, ToolFormat};
use crate::utils::config::SamplingParams;
use axum::{
    extract::{Json, Query, State},
    response::{sse::KeepAlive, Sse},
};
use base64::Engine;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use tokio::task;
use uuid::Uuid;

/// Helper struct to manage streaming response chunks
/// Provides clean API for sending tokens, errors, and status notifications
struct StreamingContext {
    seq_id: usize,
    model_id: String,
    created: u64,
    response_tx: flume::Sender<ChatResponse>,
}

impl StreamingContext {
    fn new(
        seq_id: usize,
        model_id: String,
        created: u64,
        response_tx: flume::Sender<ChatResponse>,
    ) -> Self {
        Self {
            seq_id,
            model_id,
            created,
            response_tx,
        }
    }

    /// Send a content token chunk. Returns false if client disconnected.
    fn send_token(&self, token: &str) -> bool {
        let chunk = ChatCompletionChunk {
            id: format!("seq-{}", self.seq_id),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model_id.clone(),
            choices: vec![ChatChoiceChunk {
                index: 0,
                delta: Delta {
                    content: Some(token.to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
                error: None,
            }],
            usage: None,
        };
        self.response_tx
            .try_send(ChatResponse::Chunk(chunk))
            .is_ok()
    }
}

#[utoipa::path(
    post,
    tag = "vllm-rs",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((status = 200, description = "Chat completions"))
)]
pub async fn chat_completion(
    State(data): State<Arc<ServerData>>,
    request: Json<ChatCompletionRequest>,
) -> ChatResponder {
    // Create logger for this request (None if VLLM_RS_CHAT_LOGGER not set to true)
    let logger = ChatCompletionLogger::new();
    if let Some(ref l) = logger {
        l.log_request(&request);
    }

    let use_stream = request.stream.unwrap_or(false);

    let model_id = request.model.clone().unwrap_or("default".to_string());
    let max_tokens = request
        .max_tokens
        .unwrap_or(data.econfig.max_tokens.unwrap_or(16384));

    let mut params = SamplingParams::new_with_max_tokens(max_tokens);
    params.temperature = request.temperature;
    params.top_k = request.top_k;
    params.top_p = request.top_p;
    params.frequency_penalty = request.frequency_penalty;
    params.presence_penalty = request.presence_penalty;
    params.session_id = request.session_id.clone();
    params.thinking = request.thinking.clone();
    let (img_cfg, model_type, tool_config) = {
        let e = data.engine.read();
        (
            e.img_cfg.clone(),
            e.model_type.clone(),
            e.tool_config.clone(),
        )
    };

    let mcp_tools = data
        .mcp_manager
        .as_ref()
        .map(|manager| manager.cached_tools())
        .unwrap_or_default();
    let mut resolved_tools = resolve_tools(request.tools.as_deref(), &mcp_tools);
    let mut forced_tool_name: Option<String> = None;
    let mut tool_choice_instruction: Option<String> = None;
    let mut tool_choice_required = false;

    // Set tool mode for streaming tool call handling:
    // - None: No tools, ignore </tool_call> detection
    // - Some(true): Tools enabled, finish stream at </tool_call> for external handling
    match request.tool_choice.as_ref() {
        Some(ToolChoice::None(_)) => {
            resolved_tools.clear();
        }
        Some(ToolChoice::Function { function, .. }) => {
            tool_choice_required = true;
            forced_tool_name = Some(function.name.clone());
        }
        Some(ToolChoice::Auto(_)) | None => {}
    }

    if let Some(name) = forced_tool_name.clone() {
        let selected = resolved_tools
            .iter()
            .find(|tool| tool.function.name == name)
            .cloned();
        match selected {
            Some(tool) => {
                resolved_tools = vec![tool];
                tool_choice_instruction = Some(format!(
                    "Tool choice enforced: you MUST call the `{}` tool. Do not answer with plain text. Return only a tool call.",
                    name
                ));
            }
            None => {
                return ChatResponder::ValidationError(format!(
                    "tool_choice requires tool '{}' but it was not provided",
                    name
                ));
            }
        }
    }

    let tool_schemas = Arc::new(build_tool_schema_map(&resolved_tools));
    let has_tools = !resolved_tools.is_empty();
    params.mcp_mode = if has_tools { Some(true) } else { None };

    if params.mcp_mode.is_some() {
        crate::log_warn!("Tools enabled for request");
    }

    let mut chat_messages = request.messages.clone();
    if has_tools {
        let tool_prompt_template = data.engine.read().econfig.tool_prompt_template.clone();
        let mut tool_prompt = if let Some(template) = tool_prompt_template {
            template
        } else {
            ToolFormat::get_tool_prompt(&model_type)
        };
        if let Some(instruction) = tool_choice_instruction.as_ref() {
            tool_prompt = format!("{tool_prompt}\n\n{instruction}");
        }

        // Merge with existing system prompt if present, otherwise insert new one
        if !chat_messages.is_empty() && chat_messages[0].role == "system" {
            // Merge: tool prompt + newline + existing system content
            if let Some(ref content) = chat_messages[0].content {
                let existing_content = match content {
                    super::MessageContentType::PureText(text) => text.clone(),
                    super::MessageContentType::Single(item) => match item {
                        super::MessageContent::Text { text } => text.clone(),
                        _ => String::new(),
                    },
                    super::MessageContentType::Multi(items) => items
                        .iter()
                        .filter_map(|item| match item {
                            super::MessageContent::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                };
                let merged = format!("{}\n\n{}", existing_content, tool_prompt);
                chat_messages[0] = ChatMessage::text("system", merged);
            } else {
                // System message exists but has no content, just use tool prompt
                chat_messages[0] = ChatMessage::text("system", tool_prompt);
            }
        } else {
            // No existing system prompt, insert tool prompt as first message
            chat_messages.insert(0, ChatMessage::text("system", tool_prompt));
        }
    }

    let (messages, image_data) = match build_messages_and_images(&chat_messages, img_cfg.as_ref()) {
        Ok(output) => output,
        Err(e) => {
            crate::log_error!("Image processing failed: {:?}", e);
            return ChatResponder::InternalError(format!("Internal server error {:?}", e));
        }
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    if use_stream {
        let session_id = params.session_id.clone();
        if let Some(sid) = session_id {
            crate::log_warn!("Stream request has session_id {sid}");
        }
        let (seq_id, prompt_length, stream) = {
            let mut e = data.engine.write();
            match e.generate_stream(&params, &messages, image_data, &resolved_tools, &logger) {
                Ok((seq_id, prompt_length, stream)) => (seq_id, prompt_length, stream),
                Err(e) => {
                    crate::log_error!("Stream generation failed: {:?}", e);
                    return ChatResponder::ValidationError(format!(
                        "Stream generation failed: {:?}",
                        e
                    ));
                }
            }
        };

        let stream = stream;
        let (response_tx, client_rx) = flume::unbounded();

        // Clone data needed for the async task
        let engine_clone = data.engine.clone();
        let params_clone = params.clone();
        let _img_cfg_clone = img_cfg.clone();

        let tool_config = tool_config.clone();
        let tool_parser =
            StreamToolParser::new_with_config(&model_type, model_id.to_string(), tool_config);
        let forced_tool_name = forced_tool_name.clone();
        let stream_tool_schemas = tool_schemas.clone();
        if let Some(ref l) = logger {
            l.log_start_response();
        }
        let stream_logger = logger.clone();

        task::spawn(async move {
            #[allow(unused_assignments)]
            let mut decode_start_time = 0u64;
            let mut total_decoded_tokens = 0usize;
            let mut pending_tool_calls: Vec<crate::tools::ToolCall> = Vec::new();

            // Create streaming context for clean helper methods
            let stream_ctx =
                StreamingContext::new(seq_id, model_id.to_string(), created, response_tx.clone());

            // Initialize the stream tool parser (handles all tool call detection internally)
            let mut tool_parser = tool_parser;
            let should_parse_tools = params_clone.mcp_mode.is_some();

            let mut current_stream = stream;
            let current_seq_id = seq_id;

            while let Some(item) = current_stream.recv().await {
                match item {
                    StreamItem::Token(token, token_id) => {
                        if decode_start_time == 0 {
                            decode_start_time = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;
                        }

                        // Use StreamToolParser for all tool call detection and buffering
                        if should_parse_tools {
                            match tool_parser.process_token(token_id, &token) {
                                StreamResult::Content(text) => {
                                    if text.is_empty() {
                                        continue;
                                    }
                                    // Send content to client
                                    if let Some(ref l) = stream_logger {
                                        l.log_stream_token(&text);
                                    }
                                    if !stream_ctx.send_token(&text) {
                                        crate::log_error!(
                                            "[Seq {}] Stream send error (disconnected)",
                                            current_seq_id
                                        );
                                        let mut e = engine_clone.write();
                                        e.cancel(current_seq_id);
                                        break;
                                    }
                                }
                                StreamResult::Buffering => {
                                    // Parser is buffering, don't send anything
                                }
                                StreamResult::FlushBuffer(text) => {
                                    if text.is_empty() {
                                        continue;
                                    }
                                    // False positive - flush buffered content as text
                                    crate::log_info!(
                                        "[Seq {}] Flushing {} chars (false positive)",
                                        current_seq_id,
                                        text.len()
                                    );
                                    if let Some(ref l) = stream_logger {
                                        l.log_stream_token(&text);
                                    }
                                    if !stream_ctx.send_token(&text) {
                                        let mut e = engine_clone.write();
                                        e.cancel(current_seq_id);
                                        break;
                                    }
                                }
                                StreamResult::ToolCalls(tools) => {
                                    pending_tool_calls.extend(tools);
                                }
                            }
                        } else {
                            // No tool parsing - stream directly
                            if token.is_empty() {
                                continue;
                            }
                            if let Some(ref l) = stream_logger {
                                l.log_stream_token(&token);
                            }
                            if !stream_ctx.send_token(&token) {
                                crate::log_error!(
                                    "[Seq {}] Stream send error (disconnected)",
                                    current_seq_id
                                );
                                let mut e = engine_clone.write();
                                e.cancel(current_seq_id);
                                break;
                            }
                        }
                    }
                    StreamItem::Done((
                        prompt_start_time,
                        decode_start_time_done,
                        decode_finish_time,
                        final_decoded_length,
                    )) => {
                        total_decoded_tokens += final_decoded_length;

                        // Finalize tool parsing and collect any remaining tool calls
                        if should_parse_tools {
                            match tool_parser.state() {
                                ParserState::Buffering => {
                                    // Finalize any buffered content
                                    if let Some(mut parsed) = tool_parser.finalize() {
                                        if !parsed.is_empty() {
                                            crate::log_info!(
                                                "[Seq {}] Parsed {} tool call(s)",
                                                current_seq_id,
                                                parsed.len()
                                            );
                                        }
                                        pending_tool_calls.append(&mut parsed);
                                    } else {
                                        // Parse failed - flush any remaining buffer as text
                                        let buffer = tool_parser.take_buffer();
                                        if !buffer.is_empty() {
                                            crate::log_warn!(
                                                "[Seq {}] Tool parse failed, flushing {} chars",
                                                current_seq_id,
                                                buffer.len()
                                            );
                                            stream_ctx.send_token(&buffer);
                                        }
                                    }
                                }
                                ParserState::MaybeStart => {
                                    let buffer = tool_parser.take_buffer();
                                    if !buffer.is_empty() {
                                        crate::log_warn!(
                                            "[Seq {}] Tool parse partial, flushing {} chars",
                                            current_seq_id,
                                            buffer.len()
                                        );
                                        stream_ctx.send_token(&buffer);
                                    }
                                }
                                ParserState::Normal => {}
                            }
                        }

                        if let Some(ref forced_name) = forced_tool_name {
                            let before = pending_tool_calls.len();
                            pending_tool_calls.retain(|call| call.function.name == *forced_name);
                            let dropped = before - pending_tool_calls.len();
                            if dropped > 0 {
                                crate::log_warn!(
                                    "[Seq {}] Dropped {} tool call(s) that did not match tool_choice '{}'",
                                    current_seq_id,
                                    dropped,
                                    forced_name
                                );
                            }
                        }

                        let (valid_calls, invalid_calls) =
                            filter_tool_calls(&pending_tool_calls, stream_tool_schemas.as_ref());
                        if !invalid_calls.is_empty() {
                            crate::log_warn!(
                                "[Seq {}] Dropped {} invalid tool call(s)",
                                current_seq_id,
                                invalid_calls.len()
                            );
                            log_tool_calls("Invalid", &invalid_calls);
                            if let Some(ref l) = logger {
                                l.log_tool_calls("Invalid", &invalid_calls);
                            }
                        }

                        let tool_calls = if valid_calls.is_empty() {
                            None
                        } else {
                            log_tool_calls("Valid", &valid_calls);
                            Some(valid_calls)
                        };
                        let has_any_tool_calls = tool_calls.is_some();
                        if tool_choice_required && !has_any_tool_calls {
                            crate::log_warn!(
                                "[Seq {}] Tool choice required but no tool calls were produced",
                                current_seq_id
                            );
                        }
                        // Send final chunk
                        let final_chunk = ChatCompletionChunk {
                            id: format!("seq-{}", current_seq_id),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id.to_string(),
                            choices: vec![ChatChoiceChunk {
                                index: 0,
                                delta: Delta {
                                    content: None,
                                    tool_calls,
                                },
                                finish_reason: if has_any_tool_calls {
                                    Some("tool_calls".to_string())
                                } else if total_decoded_tokens >= max_tokens {
                                    Some("length".to_string())
                                } else {
                                    Some("stop".to_string())
                                },
                                error: None,
                            }],
                            usage: Some(Usage {
                                prompt_tokens: prompt_length,
                                completion_tokens: total_decoded_tokens,
                                total_tokens: prompt_length + total_decoded_tokens,
                            }),
                        };

                        if has_any_tool_calls {
                            crate::log_info!("Final chunk with tool calls: {:?}", final_chunk);
                        }
                        if let Some(ref l) = stream_logger {
                            l.log_stream_end(&final_chunk);
                        }
                        let _ = response_tx.try_send(ChatResponse::Chunk(final_chunk));

                        // Performance metrics
                        let prompt_time_taken = if decode_start_time_done > prompt_start_time {
                            (decode_start_time_done - prompt_start_time) as f32 / 1000.0
                        } else {
                            0.0
                        };
                        let decode_time_taken = if decode_finish_time > decode_start_time_done {
                            (decode_finish_time - decode_start_time_done) as f32 / 1000.0
                        } else {
                            0.0
                        };

                        crate::log_warn!("--- Performance Metrics ---");
                        if prompt_time_taken > 0.0 {
                            crate::log_info!(
                                "[Seq {}] ⏱️ Prompt: {} tokens in {:.2}s ({:.2} t/s)",
                                current_seq_id,
                                prompt_length,
                                prompt_time_taken,
                                prompt_length as f32 / prompt_time_taken.max(0.001)
                            );
                        } else {
                            crate::log_info!(
                                "[Seq {}] ⏱️ Prompt: {} tokens (cached)",
                                current_seq_id,
                                prompt_length
                            );
                        }
                        crate::log_info!(
                            "[Seq {}] ⏱️ Decoded: {} tokens in {:.2}s ({:.2} t/s)",
                            current_seq_id,
                            total_decoded_tokens,
                            decode_time_taken,
                            total_decoded_tokens as f32 / decode_time_taken.max(0.001)
                        );

                        break;
                    }
                    StreamItem::Error(e) => {
                        crate::log_error!("[Seq {}] Stream error: {}", current_seq_id, e);
                        let error_chunk = ChatCompletionChunk {
                            id: format!("seq-{}", current_seq_id),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id.to_string(),
                            choices: vec![ChatChoiceChunk {
                                index: 0,
                                delta: Delta {
                                    content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                                error: Some(vec![ErrorMsg { message: Some(e) }]),
                            }],
                            usage: None,
                        };

                        let _ = response_tx.try_send(ChatResponse::Chunk(error_chunk));
                        break;
                    }
                    _ => {}
                }
            }

            let _ = response_tx.try_send(ChatResponse::Done);
        });

        ChatResponder::Streamer(
            Sse::new(Streamer {
                rx: client_rx,
                status: StreamingStatus::Uninitialized,
            })
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(100))
                            .unwrap_or(100),
                    ))
                    .text("keep-alive-text"),
            ),
        )
    } else {
        // Non-streaming
        let current_params = params.clone();
        let mut total_prompt_tokens = 0;
        let mut total_decoded_tokens = 0;
        let mut total_prompt_time_taken = 0f32;
        let mut total_decoded_time_taken = 0f32;
        let mut choices = Vec::new();
        let tokenizer = {
            let e = data.engine.read();
            Arc::new(e.tokenizer.clone())
        };

        crate::log_info!(
            "Received completion request with {} messages",
            messages.len()
        );
        let receivers = {
            let mut e = data.engine.write();
            match e.generate_sync(
                &vec![current_params.clone()],
                &vec![messages],
                image_data,
                &resolved_tools,
                &logger,
            ) {
                Ok(receivers) => receivers,
                Err(e) => {
                    crate::log_error!("Completion generation failed: {:?}", e);
                    return ChatResponder::InternalError(format!("Internal server error {:?}", e));
                }
            }
        };
        if let Some(ref l) = logger {
            l.log_start_response();
        }
        let results =
            match LLMEngine::collect_sync_results(receivers, tokenizer.clone(), logger.clone())
                .await
            {
                Ok(results) => results,
                Err(e) => {
                    crate::log_error!("Failed to collect completion results: {:?}", e);
                    return ChatResponder::InternalError(format!("Internal server error {:?}", e));
                }
            };

        for output in results {
            total_prompt_tokens += output.prompt_length;
            total_decoded_tokens += output.decoded_length;
            let prompt_time_taken =
                (output.decode_start_time - output.prompt_start_time) as f32 / 1000.0;
            let decode_time_taken =
                (output.decode_finish_time - output.decode_start_time) as f32 / 1000.0;
            total_prompt_time_taken += prompt_time_taken;
            total_decoded_time_taken += decode_time_taken;

            // Parse tool calls from the model output if tools were provided
            let tool_parser = ToolParser::new();

            let (content, tool_calls) = if has_tools {
                let mut parsed_calls = tool_parser.parse(&output.decode_output);
                if let Some(ref forced_name) = forced_tool_name {
                    let before = parsed_calls.len();
                    parsed_calls.retain(|call| call.function.name == *forced_name);
                    let dropped = before - parsed_calls.len();
                    if dropped > 0 {
                        crate::log_warn!(
                            "Dropped {} tool call(s) that did not match tool_choice '{}'",
                            dropped,
                            forced_name
                        );
                    }
                }
                let (valid_calls, invalid_calls) =
                    filter_tool_calls(&parsed_calls, tool_schemas.as_ref());
                if !invalid_calls.is_empty() {
                    crate::log_warn!("Dropped {} invalid tool call(s)", invalid_calls.len());
                    log_tool_calls("Invalid", &invalid_calls);
                }
                if valid_calls.is_empty() {
                    if tool_choice_required {
                        crate::log_warn!("Tool choice required but no tool calls were produced");
                    }
                    (Some(output.decode_output), None)
                } else {
                    log_tool_calls("Valid", &valid_calls);
                    (None, Some(valid_calls))
                }
            } else {
                (Some(output.decode_output), None)
            };

            // For external tool calls (not MCP), return to client
            let has_tool_calls = tool_calls.is_some();
            choices.push(ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls,
                },
                finish_reason: if has_tool_calls {
                    Some("tool_calls".to_string())
                } else {
                    Some("stop".to_string())
                },
            });
        }

        crate::log_warn!("--- Performance Metrics ---");
        crate::log_info!(
            "[{} requests] ⏱️ Prompt tokens: {} in {:.2}s ({:.2} t/s)",
            choices.len(),
            total_prompt_tokens,
            total_prompt_time_taken,
            total_prompt_tokens as f32 / total_prompt_time_taken.max(0.001)
        );
        crate::log_info!(
            "[{} requests] ⏱️ Decoded tokens: {} in {:.2}s ({:.2} t/s)",
            choices.len(),
            total_decoded_tokens,
            total_decoded_time_taken,
            total_decoded_tokens as f32 / total_decoded_time_taken.max(0.001)
        );

        let response = ChatCompletionResponse {
            id: "cmpl-".to_string() + &Uuid::new_v4().to_string()[..8],
            object: "chat.completion",
            created,
            model: model_id.to_string(),
            choices,
            usage: Usage {
                prompt_tokens: total_prompt_tokens,
                completion_tokens: total_decoded_tokens,
                total_tokens: total_prompt_tokens + total_decoded_tokens,
            },
        };

        if let Some(ref l) = logger {
            l.log_response(&response);
        }
        ChatResponder::Completion(response)
    }
}

#[utoipa::path(
    post,
    tag = "vllm-rs",
    path = "/v1/embeddings",
    request_body = EmbeddingRequest,
    responses((status = 200, description = "Embeddings"))
)]
pub async fn create_embeddings(
    State(data): State<Arc<ServerData>>,
    request: Json<EmbeddingRequest>,
) -> ChatResponder {
    let EmbeddingRequest {
        model,
        input,
        encoding_format,
        embedding_type,
    } = request.0;
    let inputs = input.into_vec();
    if inputs.is_empty() {
        return ChatResponder::ValidationError("input cannot be empty".to_string());
    }

    let model_name = model.unwrap_or_else(|| "default".to_string());

    let mut engine = data.engine.write();
    let (vectors, prompt_tokens) = match engine.embed(&inputs, embedding_type.clone()) {
        Ok(res) => res,
        Err(e) => return ChatResponder::ModelError(format!("Embedding generation failed: {e:?}")),
    };

    crate::log_warn!(
        "Finished with {} embedding vectors and {} prompt tokens",
        vectors.len(),
        prompt_tokens
    );
    let data: Vec<EmbeddingData> = vectors
        .into_iter()
        .enumerate()
        .map(|(idx, vec)| {
            let embedding = match encoding_format {
                EncodingFormat::Float => EmbeddingOutput::Vector(vec),
                EncodingFormat::Base64 => {
                    let bytes = bytemuck::cast_slice::<f32, u8>(&vec);
                    EmbeddingOutput::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
                }
            };
            EmbeddingData {
                object: "embedding",
                embedding,
                index: idx,
            }
        })
        .collect();

    ChatResponder::Embedding(EmbeddingResponse {
        object: "list",
        data,
        model: model_name,
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
}

#[utoipa::path(
    get,
    tag = "vllm-rs",
    path = "/v1/usage",
    request_body = UsageQuery,
    responses((status = 200, description = "Token Usage Request"))
)]
pub async fn get_usage(
    State(state): State<Arc<ServerData>>,
    Query(request): Query<UsageQuery>,
) -> ChatResponder {
    let engine = state.engine.read();
    let stats = match engine.get_usage_stats(request.session_id.clone()) {
        Ok(s) => s,
        Err(e) => {
            return ChatResponder::InternalError(format!("Failed to obtain usage status {:?}", e));
        }
    };

    ChatResponder::Usage(UsageResponse {
        token_used: stats.token_used,
        max_model_len: stats.max_model_len,
        used_kvcache_tokens: stats.used_kvcache_tokens,
        total_kv_cache_tokens: stats.total_kv_cache_tokens,
        swap_used: stats.swap_used,
        total_swap_memory: stats.total_swap_memory,
        session_status: stats.session_status,
    })
}

#[utoipa::path(
    post,
    tag = "vllm-rs",
    path = "/tokenize",
    request_body = TokenizeRequest,
    responses((status = 200, description = "Tokenize text or messages"))
)]
pub async fn tokenize(
    State(data): State<Arc<ServerData>>,
    request: Json<TokenizeRequest>,
) -> ChatResponder {
    let add_special_tokens = request.add_special_tokens.unwrap_or(true);

    // Get text to tokenize based on input type
    let (text, input_type) = match &request.0.input {
        TokenizeInput::Text { prompt } => (prompt.clone(), "text"),
        TokenizeInput::Messages { messages } => {
            // For messages, we need to apply chat template
            // First convert to internal Message format
            let img_cfg = {
                let e = data.engine.read();
                e.img_cfg.clone()
            };
            let (converted_messages, _) =
                match build_messages_and_images(messages, img_cfg.as_ref()) {
                    Ok(result) => result,
                    Err(e) => {
                        return ChatResponder::ValidationError(format!(
                            "Message processing failed: {:?}",
                            e
                        ));
                    }
                };

            // Apply chat template using engine's template
            let engine = data.engine.read();
            let mut template = engine.get_chat_template();
            template.set_messages(&converted_messages);
            let prompt = match template.apply_chat_template(&Vec::new(), false) {
                Ok(prompt) => prompt,
                Err(e) => {
                    return ChatResponder::InternalError(format!(
                        "Failed to apply chat template: {:?}",
                        e
                    ));
                }
            };
            (prompt, "messages")
        }
    };

    let input_chars = text.len();

    // Get tokenizer and tokenize
    let tokenizer = {
        let e = data.engine.read();
        e.tokenizer.clone()
    };

    let encoding = match tokenizer.encode(text.as_str(), add_special_tokens) {
        Ok(enc) => enc,
        Err(e) => {
            return ChatResponder::InternalError(format!("Tokenization failed: {:?}", e));
        }
    };

    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    let count = tokens.len();

    crate::log_info!(
        "[Tokenize] input_type={}, input_chars={}, output_tokens={}",
        input_type,
        input_chars,
        count
    );

    ChatResponder::Tokenize(TokenizeResponse {
        tokens,
        count,
        max_model_len: data.econfig.max_model_len,
    })
}

#[utoipa::path(
    post,
    tag = "vllm-rs",
    path = "/detokenize",
    request_body = DetokenizeRequest,
    responses((status = 200, description = "Detokenize tokens to text"))
)]
pub async fn detokenize(
    State(data): State<Arc<ServerData>>,
    request: Json<DetokenizeRequest>,
) -> ChatResponder {
    let skip_special_tokens = request.skip_special_tokens.unwrap_or(true);

    let tokenizer = {
        let e = data.engine.read();
        e.tokenizer.clone()
    };

    let input_tokens = request.tokens.len();

    let prompt = match tokenizer.decode(&request.tokens, skip_special_tokens) {
        Ok(text) => text,
        Err(e) => {
            return ChatResponder::InternalError(format!("Detokenization failed: {:?}", e));
        }
    };

    crate::log_info!(
        "[Detokenize] input_tokens={}, output_chars={}",
        input_tokens,
        prompt.len()
    );

    ChatResponder::Detokenize(DetokenizeResponse { prompt })
}
