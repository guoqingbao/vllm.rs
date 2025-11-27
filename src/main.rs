use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use candle_core::Result;
use clap::Parser;
use colored::Colorize;
use reedline::{DefaultPrompt, DefaultPromptSegment, Reedline, Signal};
use rustchatui::start_ui_server;
use serde_json::json;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;
use vllm_rs::core::engine::StreamItem;
use vllm_rs::core::engine::GLOBAL_RT;
use vllm_rs::core::{engine::LLMEngine, GenerationOutput};
use vllm_rs::log_error;
use vllm_rs::server::Args;
use vllm_rs::server::{
    server::{chat_completion, get_usage},
    ServerData,
};
use vllm_rs::transfer::{PdConfig, PdMethod, PdRole};
use vllm_rs::utils::chat_template::Message;
use vllm_rs::utils::config::GenerationConfig;
use vllm_rs::utils::config::{EngineConfig, SamplingParams};
use vllm_rs::utils::get_dtype;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();

    if args.model_id.is_none() && args.weight_path.is_none() && args.weight_file.is_none() {
        candle_core::bail!("Must provide model_id or weight_path or weight_file!");
    }

    let dtype = get_dtype(args.dtype);

    let (max_num_seqs, interactive) = if args.batch.is_some() {
        tracing::warn!("max_num_seqs is ignored in batch performance test.");
        if args.interactive {
            tracing::warn!("interactive mode is ignored in batch performance test.");
        }
        (args.batch.unwrap(), false)
    } else {
        (args.max_num_seqs, args.interactive)
    };

    assert!(
        !(interactive && args.server),
        "You selected both interactive and server mode, which is not valid!"
    );

    // Align with Python interface
    let default_max_model_len = if cfg!(target_os = "macos") {
        32768
    } else {
        32768 * 2
    };

    let max_model_len = if args.max_model_len.is_none() && interactive {
        let max_model_len = if args.interactive {
            default_max_model_len
        } else {
            default_max_model_len / max_num_seqs
        };
        Some(max_model_len)
    } else {
        // if not set under server mode, the backend will auto decide
        assert!(
            args.max_model_len == None || args.kv_fraction == None,
            "You provided both max_model_len and kv_fraction!"
        );
        args.max_model_len
    };

    let prompts = match (&args.prompts, interactive) {
        (Some(prompts), false) => prompts.clone(),
        (None, false) => {
            if args.server {
                tracing::warn!("Enter server mode.");
                vec![]
            } else {
                vec!["Please talk about China in more details.".to_string()]
            }
        }
        (Some(_), true) => {
            tracing::warn!("Interactive mode does not support predefined prompts, these prompts will be ignored.");
            vec![]
        }
        (None, true) => {
            tracing::warn!("Enter interactive mode.");
            vec![]
        }
    };

    let prompts = if args.batch.is_some() {
        let prompts = if prompts.len() > 0 {
            vec![prompts.clone()[0].clone()]
        } else {
            vec!["Please talk about China in more details.".to_string()]
        };
        let repeated: Vec<String> = (0..args.batch.unwrap())
            .flat_map(|_| prompts.iter().cloned())
            .collect();
        repeated
    } else {
        prompts
    };

    let generation_cfg = if (args.temperature.is_some()
        && (args.top_k.is_some() && args.top_p.is_some()))
        || args.frequency_penalty.is_some()
        || args.presence_penalty.is_some()
    {
        Some(GenerationConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            frequency_penalty: args.frequency_penalty,
            presence_penalty: args.presence_penalty,
        })
    } else {
        None
    };

    if args.pd_server && args.pd_client {
        candle_core::bail!("This program can only be served as PD server or PD client, not both!");
    }

    #[cfg(not(feature = "cuda"))]
    if (args.pd_server || args.pd_client) && args.pd_url.is_none() {
        candle_core::bail!("Non-CUDA platform does not support LocalIPC, please provide pd-url (e.g., 0.0.0.0:8100)!");
    }

    let pd_config = if args.pd_server || args.pd_client {
        let pd_role = if args.pd_server {
            PdRole::Server
        } else {
            PdRole::Client
        };
        let pd_method = if args.pd_url.is_some() {
            PdMethod::RemoteTcp
        } else {
            PdMethod::LocalIpc
        };
        Some(PdConfig {
            role: pd_role,
            method: pd_method,
            url: args.pd_url,
        })
    } else {
        None
    };

    let mut context_cache = args.context_cache;

    if interactive && !args.server && !args.pd_server {
        // force to use context_cache in chat mode
        context_cache = true;
    }
    let econfig = EngineConfig::new(
        args.model_id,
        args.weight_path,
        args.weight_file,
        args.hf_token,
        args.hf_token_path,
        Some(std::cmp::max(max_num_seqs, prompts.len())),
        None,
        max_model_len,
        Some(args.max_tokens),
        args.isq.clone(),
        Some(1),
        args.device_ids.clone(),
        generation_cfg,
        args.seed,
        Some(context_cache),
        Some(args.fp8_kvcache),
        Some(args.server || args.ui_server || !interactive),
        args.cpu_mem_fold,
        args.kv_fraction,
        pd_config,
    );

    let engine = LLMEngine::new(&econfig, dtype)?;

    if args.server || args.ui_server || args.pd_server {
        let server_data = ServerData {
            engine: engine.clone(),
            econfig: econfig.clone(),
        };
        // CORS config
        let cors = CorsLayer::new()
            .allow_origin(Any) // same as "*"
            .allow_methods(Any)
            .allow_headers(Any);
        // Build axum app
        let app = Router::new()
            .route(
                "/v1/models",
                get(|| async {
                    Json(json!({
                        "object": "list",
                        "data": [
                            {
                                "id": "default",
                                "object": "model",
                                "created": std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis() as i64,
                                "owned_by": "vllm.rs",
                                "permission": []
                            }
                        ]
                    }))
                }),
            )
            .route("/v1/chat/completions", post(chat_completion))
            .route("/v1/usage", get(get_usage))
            .layer(cors)
            .with_state(Arc::new(server_data));

        let addr = if args.pd_server {
            // Start PD server
            vllm_rs::log_warn!("🚀 PD server started, waiting for prefill request(s)...",);
            format!("0.0.0.0:{}", 0)
        } else {
            // Start server
            vllm_rs::log_warn!(
                "🚀 Chat server listening on http://0.0.0.0:{}/v1/",
                args.port
            );
            format!("0.0.0.0:{}", args.port)
        };
        let listener = tokio::net::TcpListener::bind(addr).await?;

        let mut tasks = Vec::new();
        tasks.push(tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("Chat API server error: {e:?}");
            }
        }));

        if args.ui_server {
            tasks.push(tokio::spawn(async move {
                start_ui_server((args.port + 1) as u16, Some(args.port as u16), None, None)
                    .await
                    .unwrap();
            }));
        }

        futures::future::try_join_all(tasks)
            .await
            .map_err(candle_core::Error::wrap)?;
        return Ok(());
    }

    if !interactive && args.prompts.is_none() {
        eprintln!(
            "{}",
            String::from("⛔️ No prompts provided for completion, using default prompt! Interactive (`--i`) and server mode (`--server`) are no specified!").red()
        );
    }

    let mut params = Vec::new();

    let mut prompt_processed = Vec::new();
    // let mut rng = rand::rng();
    if !interactive && prompts.len() > 0 {
        if prompts.len() > 1 {
            tracing::warn!("Live output muted for more than one prompt!\n");
        }
        for prompt in prompts.iter() {
            let msg = Message::new("user".to_string(), prompt.clone());
            let param = SamplingParams::new_with_max_tokens(args.max_tokens);
            let e = engine.read();
            let prompt = e.apply_chat_template(&param, &vec![msg], !args.batch.is_some());
            prompt_processed.push(prompt);
            params.push(param);
        }
        if let Some(max_model_len) = args.max_model_len {
            if args.max_tokens > max_model_len {
                log_error!(
                    "Requested max_tokens {} larger than max_model_len {}",
                    args.max_tokens,
                    max_model_len
                );
            }
        }
    } else {
        params.push(SamplingParams::new_with_max_tokens(args.max_tokens));
    }

    let total_available_tokens: i64 = max_num_seqs as i64 * max_model_len.unwrap_or(4096) as i64;
    let mut chat_context_left = total_available_tokens;
    let mut line_editor = Reedline::create();
    let mut prompt = DefaultPrompt {
        left_prompt: DefaultPromptSegment::WorkingDirectory,
        right_prompt: DefaultPromptSegment::Basic(format!(
            "Tokens left: {} (full)",
            chat_context_left
        )),
    };

    let mut request_params = params[0].clone();
    request_params.session_id = if context_cache {
        Some(format!("{}", Uuid::new_v4()))
    } else {
        None
    };

    let mut chat_history = Vec::<Message>::new();
    loop {
        if interactive {
            if chat_history.is_empty() {
                print!("🤖✨ Enter a new prompt (Press Ctrl+C to exit):");
            } else {
                print!("🤖✨ Enter another prompt to continue current chat (Press Ctrl+C to start a new chat):\n");
            }
            let sig = line_editor.read_line(&prompt);
            if chat_context_left < 0 {
                tracing::error!("No tokens left, press Ctrl+C to start a new chat session!");
                chat_context_left = 0;
                continue;
            }
            match sig {
                Ok(Signal::Success(buffer)) => {
                    let trimmed = buffer.trim();
                    if !trimmed.is_empty() {
                        let msg = Message::new("user".to_string(), trimmed.to_string());
                        chat_history.push(msg.clone());
                        prompt_processed.clear();
                        let e = engine.read();
                        prompt_processed.push(e.apply_chat_template(
                            &request_params,
                            &chat_history,
                            false,
                        ));
                    } else {
                        print!("\n No prompt was given.");
                        continue;
                    }
                }
                Ok(Signal::CtrlD) | Ok(Signal::CtrlC) => {
                    if chat_history.is_empty() {
                        print!("\n👋 Exiting.");
                        std::process::exit(0); // Ctrl+C to exit
                    } else {
                        print!("\n🌀 Chat history cleared. Start a new conversation.\n");
                        chat_history.clear(); //start a new chat
                        if context_cache {
                            let e = engine.read();
                            chat_context_left =
                                total_available_tokens - e.get_num_cached_tokens() as i64;
                        } else {
                            chat_context_left = total_available_tokens;
                        }
                        prompt.right_prompt = DefaultPromptSegment::Basic(format!(
                            "Tokens left: {} (full)",
                            chat_context_left
                        ));
                        request_params.session_id = if context_cache {
                            Some(format!("{}", Uuid::new_v4()))
                        } else {
                            None
                        };
                        continue;
                    }
                }
                _ => {}
            }
        }

        let mut outputs = {
            if interactive {
                let (seq_id, prompt_length, stream) = {
                    let mut e = engine.write();
                    match e.generate_stream(&request_params, prompt_processed[0].clone()) {
                        Ok((seq_id, prompt_length, stream)) => (seq_id, prompt_length, stream),
                        Err(e) => {
                            tracing::error!("Session unexpectedly ended because: {:?}", e);
                            print!("\n🌀 Chat history cleared. Start a new conversation.\n");
                            chat_history.clear(); //start a new chat
                            prompt.right_prompt = DefaultPromptSegment::Basic(format!(
                                "Tokens left: {} (full)",
                                total_available_tokens
                            ));
                            continue;
                        }
                    }
                };
                let handle: tokio::task::JoinHandle<(usize, usize, usize, usize, String)> =
                    GLOBAL_RT.spawn(async move {
                        let mut tickets: (usize, usize, usize, usize) = (0, 0, 0, 0);
                        let mut decode_output = "".to_string();
                        let mut rx = stream;
                        while let Some(item) = rx.recv().await {
                            match item {
                                StreamItem::Token(t) => {
                                    decode_output += &t.to_string();
                                    print!("{}", t);
                                    use std::io::Write;
                                    let _ = std::io::stdout().flush();
                                }
                                StreamItem::TokenID(_) | StreamItem::Completion(_) => {
                                    break;
                                }
                                StreamItem::Done((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    length,
                                )) => {
                                    tickets = (
                                        prompt_start_time,
                                        decode_start_time,
                                        decode_finish_time,
                                        length,
                                    );
                                    eprintln!(
                                        "{}",
                                        String::from("\r\nGeneration completed!").yellow()
                                    );
                                    break;
                                }
                                StreamItem::Error(e) => eprintln!("Error: {}", e),
                            }
                        }
                        (tickets.0, tickets.1, tickets.2, tickets.3, decode_output)
                    });
                let (
                    prompt_start_time,
                    decode_start_time,
                    decode_finish_time,
                    decoded_length,
                    decode_output,
                ) = handle.await.map_err(candle_core::Error::wrap)?;
                if context_cache {
                    let e = engine.read();
                    chat_context_left = total_available_tokens - e.get_num_cached_tokens() as i64;
                } else {
                    chat_context_left =
                        total_available_tokens - prompt_length as i64 - decoded_length as i64;
                }
                prompt.right_prompt =
                    DefaultPromptSegment::Basic(format!("Tokens left: {}", chat_context_left));
                vec![GenerationOutput {
                    seq_id,
                    prompt_length,
                    prompt_start_time,
                    decode_start_time,
                    decode_finish_time,
                    decoded_length,
                    decode_output,
                }]
            } else {
                vllm_rs::log_warn!("Starting the inference...");

                let (receivers, tokenizer) = {
                    let mut e = engine.write();
                    (
                        e.generate_sync(&params, prompt_processed.clone())?,
                        Arc::new(e.tokenizer.clone()),
                    )
                };
                LLMEngine::collect_sync_results(receivers, tokenizer).await?
            }
        };

        outputs.sort_by_key(|o| o.seq_id);

        let mut all_decode_time_taken = 0f32;
        let mut prompt_time_taken = 0f32;
        let mut total_decoded_tokens = 0;
        let mut total_prompt_tokens = 0;

        for (
            i,
            GenerationOutput {
                seq_id,
                prompt_length,
                prompt_start_time,
                decode_start_time,
                decode_finish_time,
                decoded_length,
                decode_output,
            },
        ) in outputs.iter().enumerate()
        {
            if !interactive && args.batch.is_none() {
                tracing::info!("[seq_id {}] 📚✨ Prompt {}: {}", seq_id, i + 1, prompts[i]);
                tracing::info!("[seq_id {}] 📄✨ Response: {}\n", seq_id, decode_output);
            }
            total_prompt_tokens += prompt_length;
            total_decoded_tokens += decoded_length;
            let duration_prompt = (decode_start_time - prompt_start_time) as f32 / 1000.0; //maximum time costs for prompt
            if duration_prompt > prompt_time_taken {
                prompt_time_taken = duration_prompt;
            }

            let duration = (decode_finish_time - decode_start_time) as f32 / 1000.0; //maximum time costs for decoding
            all_decode_time_taken += duration;

            if interactive {
                let msg = Message::new("assistant".to_string(), decode_output.to_string());
                chat_history.push(msg.clone());
            }
        }

        let decode_time_taken = all_decode_time_taken / outputs.len() as f32;
        vllm_rs::log_info!("--- Performance Metrics ---");

        tracing::info!(
            "⏱️ Prompt tokens: {} in {:.2}s ({:.2} tokens/s)",
            total_prompt_tokens,
            prompt_time_taken,
            total_prompt_tokens as f32 / prompt_time_taken,
        );
        tracing::info!(
            "⏱️ Decoded tokens: {} in {:.2}s ({:.2} tokens/s)",
            total_decoded_tokens,
            decode_time_taken,
            total_decoded_tokens as f32 / decode_time_taken,
        );

        if !interactive {
            break;
        }
    }

    Ok(())
}
