// src/server/server.rs
use super::{
    build_messages_and_images,
    streaming::{ChatResponse, Streamer, StreamingStatus},
    ChatResponder, DetokenizeRequest, DetokenizeResponse, EmbeddingRequest, EmbeddingResponse,
    EncodingFormat, TokenizeInput, TokenizeRequest, TokenizeResponse,
};
use super::{
    execute_mcp_tool_calls_async, ChatChoice, ChatChoiceChunk, ChatCompletionChunk,
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatResponseMessage, Delta,
    EmbeddingData, EmbeddingOutput, EmbeddingUsage, ErrorMsg, ServerData, Usage, UsageQuery,
    UsageResponse,
};
use crate::core::engine::{LLMEngine, StreamItem};
use crate::tools::parser::ToolParser;
use crate::tools::{Tool, ToolChoice, ToolFormat};
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

fn tool_format_for_model(model_id: &str) -> ToolFormat {
    let model_id = model_id.to_lowercase();
    if model_id.contains("qwen") {
        ToolFormat::Qwen
    } else if model_id.contains("llama") || model_id.contains("mistral") {
        ToolFormat::Llama
    } else {
        ToolFormat::Generic
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

fn tool_choice_schema(tool_choice: &Option<ToolChoice>) -> Option<serde_json::Value> {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => Some(serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "const": function.name },
                "arguments": { "type": "object" }
            },
            "required": ["name", "arguments"],
            "additionalProperties": false
        })),
        _ => None,
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
    params.guided_json_schema = request
        .guided_json_schema
        .clone()
        .or_else(|| tool_choice_schema(&request.tool_choice));
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
    // - Some(false): External tools (user-provided), finish stream at </tool_call>
    // - Some(true): MCP internal tools, pause stream, execute, resume
    params.mcp_mode = if mcp_injected_tools {
        Some(true) // MCP internal execution
    } else if has_request_tools {
        Some(false) // External tool handling
    } else {
        None // No tools at all
    };

    let has_tools = !resolved_tools.is_empty();
    let mut chat_messages = request.messages.clone();
    if has_tools {
        let model_hint = request.model.clone().unwrap_or_else(|| {
            let e = data.engine.read();
            e.get_model_info().1
        });
        let tool_prompt = tool_format_for_model(&model_hint).format_tools(&resolved_tools);
        chat_messages.insert(0, ChatMessage::text("system", tool_prompt));
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
        let img_cfg_clone = img_cfg.clone();

        task::spawn(async move {
            #[allow(unused_assignments)]
            let mut decode_start_time = 0u64;
            #[allow(unused_assignments, unused_variables)]
            let mut decoded_length = 0usize;
            let mut accumulated_output = String::new();
            let mut total_decoded_tokens = 0usize;

            // Track if we're inside a tool call (for MCP mode token buffering)
            let mut in_tool_call = false;
            // Check if MCP mode is enabled (internal tool execution)
            let is_mcp_mode = params_clone.mcp_mode == Some(true);

            // Context that accumulates across tool call cycles
            let mut current_messages = chat_messages_clone.clone();
            let mut current_stream = stream;
            let mut current_seq_id = seq_id;

            'outer: loop {
                while let Some(item) = current_stream.recv().await {
                    match item {
                        StreamItem::Token(token) => {
                            if decode_start_time == 0 {
                                decode_start_time = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis()
                                    as u64;
                            }
                            decoded_length += 1;

                            // Always accumulate for tool call parsing
                            accumulated_output.push_str(&token);

                            // In MCP mode, detect tool call start and buffer tokens
                            if is_mcp_mode {
                                // Check if we're entering a tool call
                                if !in_tool_call && accumulated_output.contains("<tool_call>") {
                                    in_tool_call = true;
                                    // Don't send tool call content to client
                                    continue;
                                }

                                // If we're inside a tool call, don't send to client
                                if in_tool_call {
                                    continue;
                                }
                            }

                            // Send token to client (only if not buffering tool call)
                            let chunk = ChatCompletionChunk {
                                id: format!("seq-{}", current_seq_id),
                                object: "chat.completion.chunk",
                                created,
                                model: model_id.to_string(),
                                choices: vec![ChatChoiceChunk {
                                    index: 0,
                                    delta: Delta {
                                        content: Some(token.clone()),
                                    },
                                    finish_reason: None,
                                    error: None,
                                }],
                                usage: None,
                            };

                            if let Err(e) = response_tx.try_send(ChatResponse::Chunk(chunk)) {
                                crate::log_error!(
                                    "[Seq {}] Stream send to client error: {:?}",
                                    current_seq_id,
                                    e
                                );
                                let mut e = engine_clone.write();
                                e.cancel(current_seq_id);
                                break 'outer;
                            }
                        }
                        StreamItem::ToolCallPause {
                            session_id,
                            prompt_start_time: _,
                            decode_start_time: _,
                            decoded_length: pause_decoded_length,
                        } => {
                            // Scheduler detected tool call end, execute MCP and resume
                            crate::log_info!(
                                "[Seq {}] Received ToolCallPause, session_id: {}",
                                current_seq_id,
                                session_id
                            );

                            total_decoded_tokens += pause_decoded_length;

                            // Reset in_tool_call state
                            in_tool_call = false;

                            // Parse tool calls from accumulated output
                            let tool_parser = ToolParser::new();
                            let parsed_calls = tool_parser.parse(&accumulated_output);

                            if !parsed_calls.is_empty() {
                                if let Some(mcp_manager) = data.mcp_manager.as_ref() {
                                    // Check if client is still connected before executing tools
                                    if response_tx.is_disconnected() {
                                        crate::log_warn!(
                                            "[Seq {}] Client disconnected, aborting tool execution",
                                            current_seq_id
                                        );
                                        let mut e = engine_clone.write();
                                        e.cancel(current_seq_id);
                                        break 'outer;
                                    }

                                    crate::log_info!(
                                        "[Seq {}] Executing {} tool call(s) via MCP (with 60s timeout)",
                                        current_seq_id,
                                        parsed_calls.len()
                                    );

                                    // Execute tool calls with timeout using async version
                                    let exec_result = execute_mcp_tool_calls_async(
                                        parsed_calls.clone(),
                                        mcp_manager.clone(),
                                        current_messages.clone(),
                                    )
                                    .await;

                                    // Update context with tool calls and results
                                    current_messages = exec_result.followup_messages.clone();

                                    // Build follow-up prompt
                                    let (followup_inputs, followup_images) =
                                        match build_messages_and_images(
                                            &exec_result.followup_messages,
                                            img_cfg_clone.as_ref(),
                                        ) {
                                            Ok(output) => output,
                                            Err(e) => {
                                                crate::log_error!(
                                                    "Tool follow-up processing failed: {:?}",
                                                    e
                                                );
                                                break 'outer;
                                            }
                                        };

                                    // Resume generation with session_id to use cached context
                                    // Note: mcp_mode is preserved to support multi-tool scenarios
                                    // The scheduler clears tool_call_session_id on cache resume
                                    let mut resume_params = params_clone.clone();
                                    resume_params.session_id = Some(session_id.clone());

                                    let new_stream = {
                                        let mut e = engine_clone.write();
                                        match e.generate_stream(
                                            &resume_params,
                                            &followup_inputs,
                                            followup_images,
                                        ) {
                                            Ok((new_seq_id, _, stream)) => {
                                                crate::log_info!(
                                                    "[Seq {}] Resumed generation with session_id: {} (new seq: {})",
                                                    current_seq_id,
                                                    session_id,
                                                    new_seq_id
                                                );
                                                current_seq_id = new_seq_id;
                                                stream
                                            }
                                            Err(e) => {
                                                crate::log_error!(
                                                    "Tool follow-up generation failed: {:?}",
                                                    e
                                                );
                                                break 'outer;
                                            }
                                        }
                                    };

                                    // Reset state for new generation
                                    decoded_length = 0;
                                    decode_start_time = 0;
                                    accumulated_output.clear();
                                    current_stream = new_stream;

                                    // Continue outer loop to process new stream
                                    continue 'outer;
                                }
                            }

                            // If no tool calls parsed or no MCP manager, treat as normal completion
                            crate::log_warn!(
                                "[Seq {}] ToolCallPause received but no tool calls to execute",
                                current_seq_id
                            );
                        }
                        StreamItem::Done((
                            prompt_start_time,
                            decode_start_time_done,
                            decode_finish_time,
                            final_decoded_length,
                        )) => {
                            total_decoded_tokens += final_decoded_length;

                            // Send final chunk
                            let final_chunk = ChatCompletionChunk {
                                id: format!("seq-{}", current_seq_id),
                                object: "chat.completion.chunk",
                                created,
                                model: model_id.to_string(),
                                choices: vec![ChatChoiceChunk {
                                    index: 0,
                                    delta: Delta { content: None },
                                    finish_reason: if total_decoded_tokens >= max_tokens {
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

                            break 'outer;
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
                                    delta: Delta { content: None },
                                    finish_reason: None,
                                    error: Some(vec![ErrorMsg { message: Some(e) }]),
                                }],
                                usage: None,
                            };

                            let _ = response_tx.try_send(ChatResponse::Chunk(error_chunk));
                            break 'outer;
                        }
                        _ => {}
                    }
                }
                // Stream ended without Done signal
                break 'outer;
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
        let (receivers, tokenizer) = {
            let mut e = data.engine.write();
            (
                match e.generate_sync(&vec![params.clone()], &vec![messages], image_data) {
                    Ok(receivers) => receivers,
                    Err(e) => {
                        crate::log_error!("Completion generation failed: {:?}", e);
                        return ChatResponder::InternalError(format!(
                            "Internal server error {:?}",
                            e
                        ));
                    }
                },
                Arc::new(e.tokenizer.clone()),
            )
        };

        let outputs = match LLMEngine::collect_sync_results(receivers, tokenizer).await {
            Ok(outputs) => outputs,
            Err(e) => {
                crate::log_error!("Failed to collect completion results: {:?}", e);
                return ChatResponder::InternalError(format!("Internal server error {:?}", e));
            }
        };

        let mut total_prompt_tokens = 0;
        let mut total_decoded_tokens = 0;
        let mut total_prompt_time_taken = 0.;
        let mut total_decoded_time_taken = 0.;

        let mut choices = Vec::new();
        for output in outputs {
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

            if let (Some(tool_calls), Some(mcp_manager)) = (
                &tool_calls,
                data.mcp_manager.as_ref().filter(|_| mcp_injected_tools),
            ) {
                crate::log_info!(
                    "Detected {} tool call(s) in completion, executing via MCP (with 60s timeout)",
                    tool_calls.len()
                );

                // Use async helper for tool execution with timeout
                let exec_result = execute_mcp_tool_calls_async(
                    tool_calls.clone(),
                    mcp_manager.clone(),
                    chat_messages.clone(),
                )
                .await;

                let (followup_inputs, followup_images) = match build_messages_and_images(
                    &exec_result.followup_messages,
                    img_cfg.as_ref(),
                ) {
                    Ok(output) => output,
                    Err(e) => {
                        crate::log_error!("Tool follow-up processing failed: {:?}", e);
                        return ChatResponder::InternalError(format!(
                            "Internal server error {:?}",
                            e
                        ));
                    }
                };

                let (receivers, tokenizer) = {
                    let mut e = data.engine.write();
                    match e.generate_sync(
                        &vec![params.clone()],
                        &vec![followup_inputs],
                        followup_images,
                    ) {
                        Ok(receivers) => (receivers, Arc::new(e.tokenizer.clone())),
                        Err(e) => {
                            crate::log_error!("Tool follow-up generation failed: {:?}", e);
                            return ChatResponder::InternalError(format!(
                                "Internal server error {:?}",
                                e
                            ));
                        }
                    }
                };

                let outputs = match LLMEngine::collect_sync_results(receivers, tokenizer).await {
                    Ok(outputs) => outputs,
                    Err(e) => {
                        crate::log_error!("Failed to collect tool follow-up results: {:?}", e);
                        return ChatResponder::InternalError(format!(
                            "Internal server error {:?}",
                            e
                        ));
                    }
                };

                let followup_output = outputs
                    .first()
                    .map(|o| o.decode_output.clone())
                    .unwrap_or_default();
                if let Some(followup_metrics) = outputs.first() {
                    total_prompt_tokens += followup_metrics.prompt_length;
                    total_decoded_tokens += followup_metrics.decoded_length;
                    let prompt_time_taken = (followup_metrics.decode_start_time
                        - followup_metrics.prompt_start_time)
                        as f32
                        / 1000.0;
                    let decode_time_taken = (followup_metrics.decode_finish_time
                        - followup_metrics.decode_start_time)
                        as f32
                        / 1000.0;
                    total_prompt_time_taken += prompt_time_taken;
                    total_decoded_time_taken += decode_time_taken;
                }

                choices.push(ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant".to_string(),
                        content: Some(followup_output),
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_string()),
                });
            } else {
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
