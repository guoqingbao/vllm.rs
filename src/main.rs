use candle_core::{DType, Result};
use clap::Parser;
use rustyline::{error::ReadlineError, DefaultEditor};
use std::time::{SystemTime, UNIX_EPOCH};
use vllm_rs::core::engine::LLMEngine;
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
}

fn main() -> Result<()> {
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
        args.block_size,
        args.max_num_seqs,
        args.quant.clone(),
        Some(1),
        Some(args.kvcache_mem_gpu),
        args.device_ids.clone(),
    );

    let mut engine = LLMEngine::new(&econfig, dtype)?;
    let prompts = match (args.prompts, args.interactive) {
        (Some(prompts), false) => prompts.clone(),
        (None, false) => {
            println!("No prompts provided, using default prompt.");
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

    let params = SamplingParams {
        temperature: 0.6,
        max_tokens: 2048,
        ignore_eos: false,
        top_k: None,
        top_p: None,
    };

    tracing::info!("{:?}\n", params);

    let mut prompt_processed = Vec::new();

    if !args.interactive && prompts.len() > 0 {
        if prompts.len() > 1 {
            tracing::info!("Live output muted for more than one prompt!\n");
        }
        for prompt in prompts.iter() {
            let msg = Message {
                role: "user".to_string(),
                content: prompt.clone(),
            };
            let prompt = engine.apply_chat_template(&vec![msg], true);
            prompt_processed.push(prompt);
        }
    }

    let mut chat_history = Vec::<Message>::new();
    let mut editor = DefaultEditor::new().expect("Failed to open input");
    loop {
        if args.interactive {
            if chat_history.is_empty() {
                print!("🤖✨ Enter a new prompt (Press Ctrl+C to exit):\n");
            } else {
                print!("🤖✨ Enter another prompt to continue current chat (Press Ctrl+C to start a new chat):\n");
            }

            let r = editor.readline("> ");
            match r {
                Err(ReadlineError::Interrupted) => {
                    if chat_history.is_empty() {
                        std::process::exit(0); // Ctrl+C to exit
                    } else {
                        chat_history.clear(); //start a new chat
                        continue;
                    }
                }
                Err(ReadlineError::Eof) => {
                    std::process::exit(0); // CTRL-D to force exist
                }
                Err(e) => {
                    tracing::error!("Error reading input: {e:?}");
                    std::process::exit(1);
                }
                Ok(prompt) => {
                    let msg = Message {
                        role: "user".to_string(),
                        content: prompt,
                    };
                    chat_history.push(msg.clone());
                    prompt_processed.clear();
                    prompt_processed.push(engine.apply_chat_template(&chat_history, false));
                }
            }
        }

        let outputs = engine.generate(&prompt_processed, &params)?;
        let mut decode_time_taken = 0f32;
        let mut total_decoded_tokens = 0;
        for (i, (seq_id, decode_starting_time, length, output)) in outputs.iter().enumerate() {
            if !args.interactive && prompts.len() > 1 {
                tracing::info!("[seq_id {}] Prompt {}: {}", seq_id, i + 1, prompts[i]);
                tracing::info!("[seq_id {}] Response: {}\n", seq_id, output);
            }
            total_decoded_tokens += length;
            let duration = (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as usize
                - decode_starting_time) as f32
                / 1000.0; //maximum time costs for decoding
            if duration > decode_time_taken {
                decode_time_taken = duration;
            }

            if args.interactive {
                let msg = Message {
                    role: "assistant".to_string(),
                    content: output.to_string(),
                };
                chat_history.push(msg.clone());
            }
        }

        println!("");

        if !args.interactive {
            tracing::info!("Generation completed!");
        }

        tracing::warn!(
            "{} tokens generated in {:.2} s (decoding throughput {:.2} tokens/s)",
            total_decoded_tokens,
            decode_time_taken,
            total_decoded_tokens as f32 / decode_time_taken
        );

        if !args.interactive {
            break;
        }
    }

    Ok(())
}
