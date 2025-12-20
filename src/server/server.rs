// src/server/server.rs
use super::{
    build_messages_and_images,
    streaming::{ChatResponse, Streamer, StreamingStatus},
    ChatResponder, EmbeddingRequest, EmbeddingResponse, EncodingFormat,
};
use super::{
    ChatChoice, ChatChoiceChunk, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChatResponseMessage, Delta, EmbeddingData,
    EmbeddingOutput, EmbeddingUsage, ErrorMsg, ServerData, Usage, UsageQuery, UsageResponse,
};
use crate::core::engine::{LLMEngine, StreamItem};
use crate::tools::parser::ToolParser;
use crate::tools::ToolFormat;
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
    let img_cfg = {
        let e = data.engine.read();
        e.img_cfg.clone()
    };

    let mut chat_messages = request.messages.clone();
    if let Some(tools) = request.tools.as_ref().filter(|tools| !tools.is_empty()) {
        let model_hint = request.model.clone().unwrap_or_else(|| {
            let e = data.engine.read();
            e.get_model_info().1
        });
        let tool_prompt = tool_format_for_model(&model_hint).format_tools(tools);
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

        let mut stream = stream;
        let (response_tx, client_rx) = flume::unbounded();
        task::spawn(async move {
            let mut decode_start_time = 0;
            let mut decoded_length = 0;
            let engine_clone = data.engine.clone();
            while let Some(item) = stream.recv().await {
                match item {
                    StreamItem::Token(token) => {
                        if decode_start_time == 0 {
                            decode_start_time = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;
                        }
                        decoded_length += 1;
                        let chunk = ChatCompletionChunk {
                            id: format!("seq-{}", seq_id),
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

                        let result = response_tx.try_send(ChatResponse::Chunk(chunk));
                        if let Err(e) = result {
                            crate::log_error!(
                                "[seq_id {}] Stream send to client error: {:?}",
                                seq_id,
                                e
                            );

                            if decoded_length > 0 {
                                // Performance metrics
                                let prompt_time_taken =
                                    (decode_start_time - created) as f32 / 1000.0;
                                let decode_time_taken = (std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis()
                                    as u64
                                    - decode_start_time)
                                    as f32
                                    / 1000.0;
                                crate::log_warn!("--- Performance Metrics ---");
                                crate::log_info!(
                                    "[Unfinished seq_id {}] ⏱️ Prompt tokens: {} in {:.2}s ({:.2} t/s)",
                                    seq_id,
                                    prompt_length,
                                    prompt_time_taken,
                                    prompt_length as f32 / prompt_time_taken.max(0.001)
                                );
                                crate::log_info!(
                                    "[Unfinished seq_id {}] ⏱️ Decoded tokens: {} in {:.2}s ({:.2} t/s)",
                                    seq_id,
                                    decoded_length,
                                    decode_time_taken,
                                    decoded_length as f32 / decode_time_taken.max(0.001)
                                );
                            }

                            let mut e = engine_clone.write();
                            e.cancel(seq_id);
                            break;
                        }
                    }
                    StreamItem::Done((
                        prompt_start_time,
                        decode_start_time,
                        decode_finish_time,
                        decoded_length,
                    )) => {
                        let final_chunk = ChatCompletionChunk {
                            id: format!("seq{}", seq_id),
                            object: "chat.completion.chunk",
                            created,
                            model: model_id.to_string(),
                            choices: vec![ChatChoiceChunk {
                                index: 0,
                                delta: Delta { content: None },
                                finish_reason: if decoded_length >= max_tokens {
                                    Some("length".to_string())
                                } else {
                                    Some("stop".to_string())
                                },
                                error: None,
                            }],
                            usage: Some(Usage {
                                prompt_tokens: prompt_length,
                                completion_tokens: decoded_length,
                                total_tokens: prompt_length + decoded_length,
                            }),
                        };

                        let _ = response_tx.try_send(ChatResponse::Chunk(final_chunk));

                        // Performance metrics
                        let prompt_time_taken =
                            (decode_start_time - prompt_start_time) as f32 / 1000.0;
                        let decode_time_taken =
                            (decode_finish_time - decode_start_time) as f32 / 1000.0;

                        crate::log_warn!("--- Performance Metrics ---");
                        crate::log_info!(
                            "[seq_id {}] ⏱️ Prompt tokens: {} in {:.2}s ({:.2} t/s)",
                            seq_id,
                            prompt_length,
                            prompt_time_taken,
                            prompt_length as f32 / prompt_time_taken.max(0.001)
                        );
                        crate::log_info!(
                            "[seq_id {}] ⏱️ Decoded tokens: {} in {:.2}s ({:.2} t/s)",
                            seq_id,
                            decoded_length,
                            decode_time_taken,
                            decoded_length as f32 / decode_time_taken.max(0.001)
                        );

                        break;
                    }
                    StreamItem::Error(e) => {
                        crate::log_error!("[seq_id {}] Stream error: {}", seq_id, e);
                        let error_chunk = ChatCompletionChunk {
                            id: format!("seq{}", seq_id),
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
            let has_tools = request.tools.is_some() && !request.tools.as_ref().unwrap().is_empty();
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
