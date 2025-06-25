use candle_core::{DType, Result};
use clap::Parser;
use std::time::{SystemTime, UNIX_EPOCH};
use vllm_rs::core::engine::LLMEngine;
use vllm_rs::utils::config::{EngineConfig, SamplingParams};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Maximum number of sequences to allow
    #[arg(long, default_value_t = 256)]
    max_num_seqs: usize,

    /// Size of a block
    #[arg(long, default_value_t = 32)]
    block_size: usize,

    /// if weight_path is passed, it will ignore the model_id
    #[arg(long)]
    model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online), path must include last "/"
    #[arg(long)]
    weight_path: Option<String>,

    /// The quantized weight file name (for gguf/ggml file)
    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Available GPU memory for kvcache (MB)
    #[arg(long, default_value_t = 4096)]
    kvcache_mem_gpu: usize,

    #[arg(long, value_delimiter = ',')]
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
    #[arg(long, default_value = None)]
    quant: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.weight_path.is_none() {
        candle_core::bail!("Must provide weight-path (folder of qwen3 safetensors)!");
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
        32768,
        0.7,
        args.quant.clone(),
        Some(1),
        Some(args.kvcache_mem_gpu),
        args.device_ids.clone(),
    );

    let mut engine = LLMEngine::new(&econfig, dtype)?;
    let prompts = match args.prompts {
        Some(prompts) => prompts.clone(),
        _ => vec!["How are you today?".to_string()],
    };

    let params = SamplingParams {
        temperature: 0.6,
        max_tokens: 2048,
        ignore_eos: false,
        top_k: None,
        top_p: None,
    };

    println!("{:?}\n", params);

    if prompts.len() > 1 {
        println!("Live output muted for more than one prompt!\n");
    }

    let outputs = engine.generate(&prompts, &params)?;

    let mut decode_time_taken = 0f32;
    let mut total_decoded_tokens = 0;
    for (i, (seq_id, decode_starting_time, length, output)) in outputs.iter().enumerate() {
        if prompts.len() > 1 {
            println!("[{}] Prompt {}: {}", seq_id, i + 1, prompts[i]);
            println!("[{}] Response: {}\n", seq_id, output);
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
    }

    println!(
        "\n\n{} tokens generated in {:.2} s (decoding thourghput {:.2} tokens/s)",
        total_decoded_tokens,
        decode_time_taken,
        total_decoded_tokens as f32 / decode_time_taken
    );

    Ok(())
}
