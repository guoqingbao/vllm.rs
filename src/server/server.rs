// src/server/server.rs
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

use crate::tools::parser::ToolParser;
use crate::tools::{Tool, ToolFormat};
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

fn resolve_tools(request_tools: Option<&[Tool]>, mcp_tools: &[Tool]) -> Vec<Tool> {
    if let Some(tools) = request_tools {
        if !tools.is_empty() {
            return tools.to_vec();
        }
    }
    mcp_tools.to_vec()
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
    let img_cfg = {
        let e = data.engine.read();
        e.img_cfg.clone()
    };

    let requested_tools = request.tools.as_deref().unwrap_or_default();
    let has_request_tools = !requested_tools.is_empty();
    let mcp_tools = data
        .mcp_manager
        .as_ref()
        .map(|manager| manager.cached_tools())
        .unwrap_or_default();
    let resolved_tools = resolve_tools(request.tools.as_deref(), &mcp_tools);
    let mcp_injected_tools = !has_request_tools && !mcp_tools.is_empty();

    // Set tool mode for streaming tool call handling:
    // - None: No tools, ignore </tool_call> detection
    // - Some(true): Tools enabled, finish stream at </tool_call> for external handling
    params.mcp_mode = if mcp_injected_tools || has_request_tools {
        Some(true) // Tools enabled
    } else {
        None // No tools at all
    };

    if params.mcp_mode.is_some() {
        crate::log_warn!("Tools enabled for request");
    }

    let has_tools = !resolved_tools.is_empty();
    let mut chat_messages = request.messages.clone();
    if has_tools {
        let tool_prompt = ToolFormat::format_tools(&resolved_tools);

        // Merge with existing system prompt if present, otherwise insert new one
        if !chat_messages.is_empty() && chat_messages[0].role == "system" {
            // Merge: tool prompt + newline + existing system content
            if let Some(ref content) = chat_messages[0].content {
                let existing_content = match content {
                    super::MessageContentType::PureText(text) => text.clone(),
                    super::MessageContentType::Multi(items) => items
                        .iter()
                        .filter_map(|item| match item {
                            super::MessageContent::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                };
                let merged = format!("{}\n\n{}", tool_prompt, existing_content);
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
            match e.generate_stream(&params, &messages, image_data) {
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
        let chat_messages_clone = chat_messages.clone();
        let params_clone = params.clone();
        let _img_cfg_clone = img_cfg.clone();

        task::spawn(async move {
            #[allow(unused_assignments)]
            let mut decode_start_time = 0u64;
            #[allow(unused_assignments, unused_variables)]
            let mut decoded_length = 0usize;
            let mut accumulated_output = String::new();
            let mut total_decoded_tokens = 0usize;

            // Create streaming context for clean helper methods
            let stream_ctx =
                StreamingContext::new(seq_id, model_id.to_string(), created, response_tx.clone());

            // State machine for tool call detection
            // Normal: streaming normally
            // MaybeToolCall: detected potential start (partial tag), buffering for confirmation
            // InToolCall: confirmed tool call, fully buffering until end
            #[derive(Debug, Clone, PartialEq)]
            enum ToolCallState {
                Normal,
                MaybeToolCall,
                InToolCall,
            }
            let mut tool_call_state = ToolCallState::Normal;
            let mut tool_call_buffer = String::new(); // Buffer for potential/confirmed tool call content

            // Reasoning marker pairs: (start, end) - only matched end closes the block
            const REASONING_MARKERS: &[(&str, &str)] = &[
                ("<think>", "</think>"),
                ("<|think|>", "<|/think|>"),
                ("[THINK]", "[/THINK]"),
                ("<thought>", "</thought>"),
            ];
            // Track which reasoning marker pair is active (None = not in reasoning)
            let mut active_reasoning_end: Option<&'static str> = None;
            // Track if we're inside a code block (```) - don't detect tool calls inside code blocks
            let mut in_code_block = false;
            // Check if we should buffer tool calls (when tools are enabled)
            let should_buffer_tool_calls = params_clone.mcp_mode.is_some();

            // Helper function to check for partial tool call tag
            fn could_be_partial_tag(text: &str) -> bool {
                const TAG: &str = "<tool_call>";
                for i in 1..TAG.len() {
                    if text.ends_with(&TAG[..i]) {
                        return true;
                    }
                }
                false
            }

            // Context that accumulates across tool call cycles
            let _current_messages = chat_messages_clone.clone();
            let mut current_stream = stream;
            let current_seq_id = seq_id;

            while let Some(item) = current_stream.recv().await {
                match item {
                    StreamItem::Token(token) => {
                        if decode_start_time == 0 {
                            decode_start_time = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;
                        }
                        decoded_length += 1;

                        // Always accumulate for tool call parsing
                        accumulated_output.push_str(&token);

                        // Track reasoning block state using paired markers
                        if active_reasoning_end.is_none() {
                            // Check for any reasoning start marker
                            for &(start, end) in REASONING_MARKERS {
                                if token.contains(start) || accumulated_output.ends_with(start) {
                                    active_reasoning_end = Some(end);
                                    break;
                                }
                            }
                        } else if let Some(end_marker) = active_reasoning_end {
                            // Check only for the paired end marker
                            if token.contains(end_marker)
                                || accumulated_output.ends_with(end_marker)
                            {
                                active_reasoning_end = None;
                            }
                        }

                        // Track code block state (```) - toggle on each occurrence
                        // Don't detect tool calls inside code blocks (they're examples/documentation)
                        if token.contains("```") || accumulated_output.ends_with("```") {
                            in_code_block = !in_code_block;
                        }

                        // When tools are available, use state machine for tool call detection
                        // But ONLY if we're not inside a reasoning block OR a code block
                        if should_buffer_tool_calls
                            && active_reasoning_end.is_none()
                            && !in_code_block
                        {
                            match tool_call_state.clone() {
                                ToolCallState::Normal => {
                                    // First check: does the current token contain the FULL tag?
                                    // This handles cases where <tool_call> arrives in one token
                                    if let Some(pos) = token.find("<tool_call>") {
                                        crate::log_info!(
                                                "[Seq {}] Detected <tool_call> in token, buffering started",
                                                current_seq_id
                                            );
                                        tool_call_state = ToolCallState::InToolCall;
                                        tool_call_buffer.clear();
                                        // Capture any content that came AFTER the tag in this token
                                        let after_tag = &token[pos + 11..]; // len("<tool_call>") = 11
                                        if !after_tag.is_empty() {
                                            tool_call_buffer.push_str(after_tag);
                                        }
                                        continue; // Don't send this token
                                    }
                                    // Second check: did the tag just complete across tokens?
                                    // This handles cases where tag spans multiple tokens
                                    if accumulated_output.ends_with("<tool_call>") {
                                        crate::log_info!(
                                                "[Seq {}] Detected <tool_call> at end of accumulated output, buffering started",
                                                current_seq_id
                                            );
                                        tool_call_state = ToolCallState::InToolCall;
                                        tool_call_buffer.clear();
                                        continue; // Don't send this token
                                    }
                                    // Third check: partial tag match (could be starting a tool call)
                                    if could_be_partial_tag(&accumulated_output) {
                                        tool_call_state = ToolCallState::MaybeToolCall;
                                        tool_call_buffer.push_str(&token);
                                        continue; // Hold this token
                                    }
                                    // Normal streaming - send token
                                }

                                ToolCallState::MaybeToolCall => {
                                    tool_call_buffer.push_str(&token);
                                    // Check if we now have the full tag anywhere in the buffer
                                    // This handles cases where the tag completes in the middle of a token
                                    if let Some(tag_pos) = tool_call_buffer.find("<tool_call>") {
                                        crate::log_info!(
                                                "[Seq {}] Confirmed <tool_call> in buffer after partial match",
                                                current_seq_id
                                            );
                                        tool_call_state = ToolCallState::InToolCall;
                                        // Only keep content after the tag
                                        let after_tag_start = tag_pos + 11;
                                        let after_tag =
                                            tool_call_buffer[after_tag_start..].to_string();
                                        tool_call_buffer.clear();
                                        if !after_tag.is_empty() {
                                            tool_call_buffer.push_str(&after_tag);
                                        }
                                        continue;
                                    }

                                    // Check if it's still a potential partial match
                                    if could_be_partial_tag(&accumulated_output) {
                                        continue; // Keep waiting
                                    }
                                    // False alarm - not a tool call tag
                                    // Flush the buffered content as normal text

                                    crate::log_info!(
                                            "[Seq {}] False positive partial tag detected, flushing {} chars",
                                            current_seq_id,
                                            tool_call_buffer.len()
                                        );
                                    if !stream_ctx.send_token(&tool_call_buffer) {
                                        crate::log_error!(
                                            "[Seq {}] Stream send to client error (disconnected)",
                                            current_seq_id
                                        );
                                        let mut e = engine_clone.write();

                                        e.cancel(current_seq_id);
                                        break;
                                    }
                                    tool_call_buffer.clear();
                                    tool_call_state = ToolCallState::Normal;
                                    continue; // Token already in buffer, was sent
                                }
                                ToolCallState::InToolCall => {
                                    // Keep buffering - scheduler will detect </tool_call>
                                    tool_call_buffer.push_str(&token);
                                    continue;
                                }
                            }
                        }

                        // Send token to client using helper (only if not buffering tool call)
                        if !stream_ctx.send_token(&token) {
                            crate::log_error!(
                                "[Seq {}] Stream send to client error (disconnected)",
                                current_seq_id
                            );
                            let mut e = engine_clone.write();
                            e.cancel(current_seq_id);
                            break;
                        }
                    }
                    StreamItem::Done((
                        prompt_start_time,
                        decode_start_time_done,
                        decode_finish_time,
                        final_decoded_length,
                    )) => {
                        total_decoded_tokens += final_decoded_length;

                        // Check if we need to parse tool calls (tools enabled)
                        // Only parse if we actually detected a tool call during streaming
                        // (i.e., tool_call_state is InToolCall, not Normal or MaybeToolCall)
                        // This prevents parsing tool calls that are inside code blocks
                        // (examples/documentation) which were never buffered
                        let should_parse_tool_calls = params_clone.mcp_mode.is_some()
                            && !accumulated_output.is_empty()
                            && tool_call_state == ToolCallState::InToolCall;

                        let (tool_calls, has_tool_calls) = if should_parse_tool_calls {
                            let tool_parser = ToolParser::new();
                            let parsed = tool_parser.parse(&accumulated_output);
                            let has_calls = !parsed.is_empty();
                            crate::log_warn!(
                                "Parse tool call content: {:?}, result: {}",
                                accumulated_output,
                                if has_calls { "success" } else { "failed" }
                            );

                            // If parsing failed but we have buffered content, flush it as text
                            // This handles incomplete/truncated tool calls gracefully
                            if !has_calls
                                && tool_call_state != ToolCallState::Normal
                                && !tool_call_buffer.is_empty()
                            {
                                crate::log_warn!(
                                    "[Seq {}] Tool call parsing failed, flushing {} chars as text",
                                    current_seq_id,
                                    tool_call_buffer.len()
                                );
                                stream_ctx.send_token(&tool_call_buffer);
                            }

                            (if has_calls { Some(parsed) } else { None }, has_calls)
                        } else {
                            (None, false)
                        };

                        // Send final chunk with tool_calls if applicable
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
                                finish_reason: if has_tool_calls {
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

                        if has_tool_calls {
                            crate::log_info!("Dump final chunk for tool call: {:?}", final_chunk);
                        }
                        let _ = response_tx.try_send(ChatResponse::Chunk(final_chunk));

                        // Performance metrics
                        // Note: For resumed generation with cached context, timing may be unusual
                        // Use saturating_sub to prevent underflow
                        let prompt_time_taken = if decode_start_time_done > prompt_start_time {
                            (decode_start_time_done - prompt_start_time) as f32 / 1000.0
                        } else {
                            0.0 // Cached context, no real prompt time
                        };
                        let decode_time_taken = if decode_finish_time > decode_start_time_done {
                            (decode_finish_time - decode_start_time_done) as f32 / 1000.0
                        } else {
                            0.0
                        };

                        crate::log_warn!("--- Performance Metrics ---");
                        if prompt_time_taken > 0.0 {
                            crate::log_info!(
                                "[Seq {}] ⏱️ Prompt tokens: {} in {:.2}s ({:.2} t/s)",
                                current_seq_id,
                                prompt_length,
                                prompt_time_taken,
                                prompt_length as f32 / prompt_time_taken.max(0.001)
                            );
                        } else {
                            crate::log_info!(
                                "[Seq {}] ⏱️ Prompt tokens: {} (cached context)",
                                current_seq_id,
                                prompt_length
                            );
                        }
                        crate::log_info!(
                            "[Seq {}] ⏱️ Decoded tokens: {} in {:.2}s ({:.2} t/s)",
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
            // Stream ended without Done signal

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
        // Non-streaming with loop-based MCP tool calling (like stream path)
        let current_messages = chat_messages.clone();
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

        // MCP tool calling loop - continues until no more tool calls
        let (input_messages, input_images) =
            match build_messages_and_images(&current_messages, img_cfg.as_ref()) {
                Ok(output) => output,
                Err(e) => {
                    crate::log_error!("Message processing failed: {:?}", e);
                    return ChatResponder::InternalError(format!("Internal server error {:?}", e));
                }
            };

        crate::log_info!(
            "Received completion request with {} messages",
            input_messages.len()
        );
        let receivers = {
            let mut e = data.engine.write();
            match e.generate_sync(
                &vec![current_params.clone()],
                &vec![input_messages],
                input_images,
            ) {
                Ok(receivers) => receivers,
                Err(e) => {
                    crate::log_error!("Completion generation failed: {:?}", e);
                    return ChatResponder::InternalError(format!("Internal server error {:?}", e));
                }
            }
        };

        let results = match LLMEngine::collect_sync_results(receivers, tokenizer.clone()).await {
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
                let parsed_calls = tool_parser.parse(&output.decode_output);
                if parsed_calls.is_empty() {
                    (Some(output.decode_output), None)
                } else {
                    (None, Some(parsed_calls))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Tool;

    #[test]
    fn resolve_tools_prefers_request_tools() {
        let request_tools = vec![Tool::function("local", "Local tool").build()];
        let mcp_tools = vec![Tool::function("mcp", "MCP tool").build()];

        let resolved = resolve_tools(Some(&request_tools), &mcp_tools);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].function.name, "local");
    }

    #[test]
    fn resolve_tools_falls_back_to_mcp() {
        let request_tools: Vec<Tool> = vec![];
        let mcp_tools = vec![Tool::function("mcp", "MCP tool").build()];

        let resolved = resolve_tools(Some(&request_tools), &mcp_tools);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].function.name, "mcp");
    }

    #[test]
    fn tool_choice_schema_for_function() {
        let tool_choice = ToolChoice::function("get_weather");
        let schema = tool_choice_schema(&Some(tool_choice)).unwrap();
        assert_eq!(schema["properties"]["name"]["const"], "get_weather");
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&"name".into()));
    }
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
            let prompt = match template.apply_chat_template(false) {
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
