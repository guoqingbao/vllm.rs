//src/core/engine.rs
use super::runner::ModelRunner;
use super::scheduler::Scheduler;
use super::sequence::Sequence;
use crate::utils::config::{Config, EngineConfig, ModelType, SamplingParams, TokenizerConfig};
use crate::utils::{
    chat_template::ChatTemplate, get_kvcache_blocks, hub_load_local_safetensors, new_device,
};
use candle_core::{DType, Result};
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

pub struct LLMEngine {
    model_runner: ModelRunner,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    econfig: EngineConfig,
    config_tokenizer: TokenizerConfig,
}

impl LLMEngine {
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Self> {
        let device = new_device(econfig.device_id.unwrap_or(0))?;
        println!("Loading model...");
        let weight_files = if Path::new(&econfig.model_path)
            .join("model.safetensors.index.json")
            .exists()
        {
            hub_load_local_safetensors(&econfig.model_path, "model.safetensors.index.json")?
        } else if Path::new(&econfig.model_path)
            .join("model.safetensors")
            .exists()
        {
            vec![Path::new(&econfig.model_path).join("model.safetensors")]
        } else {
            candle_core::bail!("Safetensors files not found in path {}", econfig.model_path);
        };

        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(&weight_files, dtype, &device)
                .unwrap()
        };
        let config_path = format!("{}/config.json", econfig.model_path);
        let mut config: Config =
            serde_json::from_slice(&std::fs::read(config_path).map_err(candle_core::Error::wrap)?)
                .map_err(candle_core::Error::wrap)?;
        config.quant = econfig.quant.clone();

        let config_path = format!("{}/tokenizer_config.json", econfig.model_path);
        let config_tokenizer: TokenizerConfig =
            serde_json::from_slice(&std::fs::read(config_path).map_err(candle_core::Error::wrap)?)
                .map_err(candle_core::Error::wrap)?;

        println!("{:?}\n", config_tokenizer);

        println!("Model loaded.\n");

        let mut econfig = econfig.clone();
        econfig.max_model_len = std::cmp::min(
            econfig.max_model_len,
            config.max_model_len.unwrap_or(
                config_tokenizer
                    .model_max_length
                    .unwrap_or(config.max_position_embeddings as f64) as usize,
            ),
        );

        let tokenizer_file = if econfig.tokenizer.is_some() {
            econfig.tokenizer.clone().unwrap()
        } else {
            econfig.model_path.clone() + "/tokenizer.json"
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(candle_core::Error::wrap)?;

        assert!(
            config.architectures.len() == 1,
            "Only one architecture is supported at the moment!"
        );

        let model_type = match config.architectures[0].as_str() {
            "Qwen2ForCausalLM" | "Qwen2ForConditionalGeneration" => ModelType::Qwen3,
            "Qwen3ForCausalLM" | "Qwen3ForConditionalGeneration" => ModelType::Qwen3,
            "LlamaForCausalLM" | "LlamaForConditionalGeneration" => ModelType::LLaMa,
            _ => candle_core::bail!("Unsupported architecture: {}", config.architectures[0]),
        };

        let num_blocks = get_kvcache_blocks(
            econfig.kvcache_mem_gpu.unwrap_or(4096),
            econfig.block_size,
            &config,
            econfig.num_shards.unwrap_or(1),
            dtype,
        );

        econfig.num_blocks = num_blocks;
        econfig.max_num_batched_tokens = num_blocks * econfig.block_size;
        println!(
            "Maximum batched tokens {} ({} blocks x Block_Size {} for KV cache).",
            econfig.max_num_batched_tokens, num_blocks, econfig.block_size
        );

        let model_runner =
            ModelRunner::new(model_type, vb, &econfig, &config, dtype, device.clone())?;
        let scheduler = Scheduler::new(&econfig, &config);

        Ok(Self {
            model_runner,
            scheduler,
            tokenizer,
            econfig,
            config_tokenizer,
        })
    }

    pub fn add_request(&mut self, prompt: &str, params: SamplingParams) -> Result<()> {
        let tokens = self.tokenizer.encode(prompt, true).expect("encode failed!");
        let token_ids: Vec<u32> = tokens.get_ids().iter().map(|&x| x as u32).collect();
        let seq = Sequence::new(token_ids, self.econfig.block_size, params);
        self.scheduler.add(seq);
        Ok(())
    }

    pub fn step(&mut self, log: bool) -> Result<Vec<(usize, Vec<u32>)>> {
        // Get scheduled sequence indexes and prefill flag
        let (scheduled_ids, is_prefill) = self.scheduler.schedule();
        if scheduled_ids.is_empty() {
            return Ok(vec![]);
        }

        // Get immutable references to scheduled sequences for model_runner
        let seqs = self.scheduler.get_sequences(&scheduled_ids);

        // Run model on the scheduled sequences
        let output_ids = self.model_runner.run(&seqs, is_prefill)?;

        // Postprocess sequences by modifying them inside the scheduler
        self.scheduler.postprocess(&scheduled_ids, &output_ids);

        // Collect outputs of finished sequences
        let outputs: Vec<_> = scheduled_ids
            .iter()
            .filter_map(|&idx| match self.scheduler.get_running(idx) {
                Some(s) => {
                    if s.is_finished() {
                        Some((s.id, s.output_ids.clone()))
                    } else {
                        if log {
                            let output = self
                                .tokenizer
                                .decode(&[s.last_token], true)
                                .expect("unable to decode!");
                            print!("{}", output);
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                        }
                        None
                    }
                }
                _ => None,
            })
            .collect();
        self.scheduler.clear_finished();
        Ok(outputs)
    }

    pub fn generate(
        &mut self,
        prompts: &Vec<String>,
        params: &SamplingParams,
    ) -> Result<Vec<(usize, usize, usize, String)>> {
        let mut params = params.clone();
        if prompts.len() * params.max_tokens > self.econfig.max_num_batched_tokens {
            params.max_tokens = self.econfig.max_num_batched_tokens / prompts.len();
            println!("Adjusted max_tokens to {}", params.max_tokens);
        }
        for prompt in prompts {
            let prompt = ChatTemplate::new(
                None,
                self.config_tokenizer.chat_template.clone(),
                self.config_tokenizer.bos_token.clone(),
                self.config_tokenizer.eos_token.clone(),
                Some(prompt.clone()),
                false,
                false,
            );
            let prompt = prompt
                .apply_chat_template()
                .map_err(candle_core::Error::wrap)?;
            println!("Prompt after applying Chat Template: {}", prompt);
            self.add_request(&prompt.as_str(), params.clone())?;
        }

        let mut decode_start_time = 0;
        let mut outputs = HashMap::new();
        while !self.scheduler.is_finished() {
            let step_output = self.step(prompts.len() == 1)?;
            if decode_start_time == 0 {
                decode_start_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize;
            }
            for (seq_id, token_ids) in step_output {
                outputs.insert(seq_id, token_ids);
            }
        }

        let mut sorted_outputs: Vec<_> = outputs.into_iter().collect();
        sorted_outputs.sort_by_key(|(seq_id, _)| *seq_id);

        let mut results = vec![];
        for (seq_id, token_ids) in sorted_outputs {
            let output = self
                .tokenizer
                .decode(&token_ids, true)
                .expect("unable to decode!");
            results.push((seq_id, decode_start_time, token_ids.len(), output));
        }

        Ok(results)
    }
}
