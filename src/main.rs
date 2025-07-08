use candle_core::{DType, Result};
use clap::Parser;
use reedline::{DefaultPrompt, Reedline, Signal};
use std::time::{SystemTime, UNIX_EPOCH};
use vllm_rs::core::engine::StreamItem;
use vllm_rs::core::engine::GLOBAL_RT;
use vllm_rs::core::{engine::LLMEngine, GenerationOutput};
use vllm_rs::utils::chat_template::Message;
use vllm_rs::utils::config::{EngineConfig, SamplingParams};
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Maximum number of concurrent sequences to allow
    #[arg(long, default_value_t = 64)]
    max_num_seqs: usize,

    /// Size of a block
    #[arg(long, default_value_t = 32)]
    block_size: usize,

    /// if weight_path is passed, it will ignore the model_id
    #[arg(long)]
    model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    /// or the path to a gguf file
    #[arg(long = "w")]
    weight_path: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Available GPU memory for kvcache (MB)
    #[arg(long = "kvmem", default_value_t = 4096)]
    kvcache_mem_gpu: usize,

    #[arg(long = "d", value_delimiter = ',')]
    device_ids: Option<Vec<usize>>,

    //Whether the program running in multiprocess or multithread model for parallel inference
    #[arg(long, default_value_t = false)]
    multi_process: bool,

    #[arg(long, default_value_t = false)]
    log: bool,

    #[arg(long, value_delimiter = '|')]
    prompts: Option<Vec<String>>,

    // in-site quantization, e.g. q4_k, q2_k, q8_0, etc.
    // if not provided, it will not perform in-situ quantization for the original model
    // do not use this option if you are using a gguf file
    #[arg(long, default_value = None)]
    quant: Option<String>,

    #[arg(long = "i", default_value_t = false)]
    interactive: bool,

    #[arg(long = "max", default_value_t = 4096)]
    max_tokens: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    if args.weight_path.is_none() {
        candle_core::bail!(
            "Must provide weight-path (folder of safetensors or path of gguf file)!"
        );
    }
    if args.device_ids.is_some() && args.device_ids.as_ref().unwrap().len() > 1 {
        candle_core::bail!("Multi-rank inference is under development!");
    }

    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => panic!("Unsupported dtype {dtype}"),
        None => DType::BF16,
    };

    let econfig = EngineConfig::new(
        args.weight_path.unwrap(),
        Some(args.block_size),
        Some(args.max_num_seqs),
        args.quant.clone(),
        Some(1),
        Some(args.kvcache_mem_gpu),
        args.device_ids.clone(),
    );

    let engine = LLMEngine::new(&econfig, dtype)?;
    let prompts = match (args.prompts, args.interactive) {
        (Some(prompts), false) => prompts.clone(),
        (None, false) => {
            println!("⛔️ No prompts provided, using default prompt.");
            vec!["How are you today?".to_string()]
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

    let params = SamplingParams::default();

    tracing::info!("{:?}\n", params);

    let mut prompt_processed = Vec::new();

    if !args.interactive && prompts.len() > 0 {
        if prompts.len() > 1 {
            tracing::warn!("Live output muted for more than one prompt!\n");
        }
        for prompt in prompts.iter() {
            let msg = Message::new("user".to_string(), prompt.clone());
            let e = engine.read();
            let prompt = e.apply_chat_template(&vec![msg], true);
            prompt_processed.push(prompt);
        }
    }

    let mut line_editor = Reedline::create();
    let prompt = DefaultPrompt::default();

    let mut chat_history = Vec::<Message>::new();
    loop {
        if args.interactive {
            if chat_history.is_empty() {
                print!("🤖✨ Enter a new prompt (Press Ctrl+C to exit):");
            } else {
                print!("🤖✨ Enter another prompt to continue current chat (Press Ctrl+C to start a new chat):\n");
            }

            let sig = line_editor.read_line(&prompt);
            match sig {
                Ok(Signal::Success(buffer)) => {
                    let trimmed = buffer.trim();
                    if !trimmed.is_empty() {
                        let msg = Message::new("user".to_string(), trimmed.to_string());
                        chat_history.push(msg.clone());
                        prompt_processed.clear();
                        let e = engine.read();
                        prompt_processed.push(e.apply_chat_template(&chat_history, false));
                    }
                }
                Ok(Signal::CtrlD) | Ok(Signal::CtrlC) => {
                    if chat_history.is_empty() {
                        print!("\n👋 Exiting.");
                        std::process::exit(0); // Ctrl+C to exit
                    } else {
                        print!("\n🌀 Chat history cleared. Start a new conversation.\n");
                        chat_history.clear(); //start a new chat
                        continue;
                    }
                }
                _ => {}
            }
        }

        let prompt_start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        let mut outputs = {
            if args.interactive {
                let (seq_id, prompt_length, stream) = {
                    let mut e = engine.write();
                    e.generate_stream(&params, prompt_processed[0].clone())?
                };
                let handle: tokio::task::JoinHandle<(usize, usize, String)> =
                    GLOBAL_RT.spawn(async move {
                        let mut length = 0;
                        let mut decode_start_time = 0;
                        let mut decode_output = "".to_string();
                        let mut rx = stream;
                        while let Some(item) = rx.recv().await {
                            match item {
                                StreamItem::Token(t) => {
                                    if decode_start_time == 0 {
                                        decode_start_time = SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .expect("Time went backwards")
                                            .as_millis()
                                            as usize;
                                    }
                                    length += 1;
                                    decode_output += &t.to_string();
                                    print!("{}", t);
                                    use std::io::Write;
                                    let _ = std::io::stdout().flush();
                                }
                                StreamItem::Completion(_) => {}
                                StreamItem::Done(_) => tracing::info!("Generation completed!"),
                                StreamItem::Error(e) => eprintln!("Error: {}", e),
                            }
                        }
                        (decode_start_time, length, decode_output)
                    });
                let (decode_start_time, decoded_length, decode_output) =
                    handle.await.map_err(candle_core::Error::wrap)?;
                vec![GenerationOutput {
                    seq_id,
                    prompt_length,
                    decode_start_time,
                    decoded_length,
                    decode_output,
                }]
            } else {
                let receivers = {
                    let mut e = engine.write();
                    e.generate_sync(&params, prompt_processed.clone())?
                };
                LLMEngine::collect_sync_results(receivers).await?
            }
        };

        outputs.sort_by_key(|o| o.seq_id);

        let mut decode_time_taken = 0f32;
        let mut prompt_time_taken = 0f32;
        let mut total_decoded_tokens = 0;
        let mut total_prompt_tokens = 0;

        for (
            i,
            GenerationOutput {
                seq_id,
                prompt_length,
                decode_start_time,
                decoded_length,
                decode_output,
            },
        ) in outputs.iter().enumerate()
        {
            if !args.interactive {
                tracing::info!("[seq_id {}] 📚✨ Prompt {}: {}", seq_id, i + 1, prompts[i]);
                tracing::info!("[seq_id {}] 📄✨ Response: {}\n", seq_id, decode_output);
            }
            total_prompt_tokens += prompt_length;
            total_decoded_tokens += decoded_length;
            let duration_prompt = (decode_start_time - prompt_start_time) as f32 / 1000.0; //maximum time costs for prompt
            if duration_prompt > prompt_time_taken {
                prompt_time_taken = duration_prompt;
            }

            let duration = (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as usize
                - decode_start_time) as f32
                / 1000.0; //maximum time costs for decoding
            if duration > decode_time_taken {
                decode_time_taken = duration;
            }

            if args.interactive {
                let msg = Message::new("assistant".to_string(), decode_output.to_string());
                chat_history.push(msg.clone());
            }
        }

        tracing::info!("--- Performance Metrics ---");

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

        if !args.interactive {
            break;
        }
    }

    Ok(())
}
