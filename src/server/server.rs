// src/server/server.rs
use super::{
    streaming::{ChatResponse, Streamer, StreamingStatus},
    ChatResponder,
};
use super::{
    ChatChoice, ChatChoiceChunk, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, Delta, ServerData, Usage,
};
use crate::core::engine::{LLMEngine, StreamItem};
use crate::utils::chat_template::Message;
use crate::utils::config::SamplingParams;
use axum::{
    extract::{Json, State},
    response::{sse::KeepAlive, Sse},
};
use std::env;
use std::sync::Arc;
use std::time::Duration;
use tokio::task;
use uuid::Uuid;

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

    let messages: Vec<Message> = request
        .messages
        .iter()
        .map(|m| Message::new(m.role.clone(), m.content.clone()))
        .collect();

    let prompt = {
        let engine = data.engine.read();
        engine.apply_chat_template(&params, &messages, false)
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
            match e.generate_stream(&params, prompt) {
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
                                crate::log_info!("--- Performance Metrics ---");
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

                        crate::log_info!("--- Performance Metrics ---");
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
                match e.generate_sync(&vec![params.clone()], vec![prompt.clone()]) {
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

            choices.push(ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: output.decode_output,
                },
                finish_reason: Some("stop".to_string()),
            });
        }

        crate::log_info!("--- Performance Metrics ---");
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
