//src/core/engine.rs
use super::runner::ModelRunner;
use super::scheduler::Scheduler;
use super::sequence::Sequence;
use crate::models::layers::VarBuilderX;
use crate::utils::config::{EngineConfig, ModelType, SamplingParams, TokenizerConfig};
use crate::utils::init_config_tokenizer;
use crate::utils::{chat_template::ChatTemplate, get_kvcache_blocks, new_device};
use candle_core::{DType, Result};
use either::Either;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
pub struct LLMEngine {
    model_runner: ModelRunner,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    econfig: EngineConfig,
    config_tokenizer: TokenizerConfig,
    default_chat_template: String,
}

impl LLMEngine {
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Self> {
        let device = new_device(econfig.device_id.unwrap_or(0))?;
        tracing::info!("Loading model...");
        let (config, config_tokenizer, tokenizer) = init_config_tokenizer(econfig)?;
        let vb = VarBuilderX::new(&econfig.model_path, dtype, &device)?;
        tracing::info!("{:?}\n", config_tokenizer);

        let mut econfig = econfig.clone();
        econfig.max_model_len = std::cmp::min(
            econfig.max_model_len,
            config.max_model_len.unwrap_or(
                config_tokenizer
                    .model_max_length
                    .unwrap_or(config.max_position_embeddings as f64) as usize,
            ),
        );

        assert!(
            config.architectures.len() == 1,
            "Only one architecture is supported at the moment!"
        );

        let (model_type, default_chat_template, is_rope_i) = match config.architectures[0].as_str()
        {
            "Qwen2ForCausalLM"
            | "Qwen2ForConditionalGeneration"
            | "Qwen3ForCausalLM"
            | "Qwen3ForConditionalGeneration"
            | "qwen2"
            | "qwen3" => (
                ModelType::Qwen3,
                "<|im_start|>user\n {} <|im_end|>".to_string(),
                false,
            ),
            "LlamaForCausalLM"
            | "LlamaForConditionalGeneration"
            | "llama"
            | "llama2"
            | "llama3" => {
                if let Some(_) = tokenizer
                    .get_vocab(true)
                    .get("<|start_header_id|>")
                    .copied()
                {
                    //llama3
                    (
                        ModelType::LLaMa,
                        "<|start_header_id|>user<|end_header_id|>\n\n {} <|eot_id|>".to_string(),
                        false,
                    )
                } else {
                    //llama2
                    (ModelType::LLaMa, "[INST] {} [/INST]".to_string(), true)
                }
            }
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
        tracing::info!("{:?}", econfig);

        tracing::info!(
            "Maximum batched tokens {} ({} blocks x Block_Size {} for KV cache).",
            econfig.max_num_batched_tokens,
            num_blocks,
            econfig.block_size
        );

        let model_runner = ModelRunner::new(
            model_type,
            &vb,
            &econfig,
            &config,
            dtype,
            is_rope_i,
            device.clone(),
        )?;
        let scheduler = Scheduler::new(&econfig, &config);
        tracing::info!("Model loaded.\n");

        Ok(Self {
            model_runner,
            scheduler,
            tokenizer,
            econfig,
            config_tokenizer,
            default_chat_template,
        })
    }

    pub fn add_request(&mut self, prompt: &str, params: SamplingParams) -> Result<()> {
        let tokens = self.tokenizer.encode(prompt, true).expect("encode failed!");
        let token_ids: Vec<u32> = tokens.get_ids().iter().map(|&x| x).collect();
        let seq = Sequence::new(token_ids, self.econfig.block_size, params);
        self.scheduler.add(seq);
        Ok(())
    }

    pub fn step(&mut self) -> Result<Vec<(usize, Either<u32, Vec<u32>>)>> {
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
                        Some((s.id, Either::Right(s.output_ids.clone())))
                    } else {
                        Some((s.id, Either::Left(s.last_token)))
                    }
                }
                _ => None,
            })
            .collect();
        self.scheduler.clear_finished();
        Ok(outputs)
    }

    fn apply_chat_template(&self, prompt: &String) -> String {
        let prompt_template = ChatTemplate::new(
            None,
            self.config_tokenizer.chat_template.clone(),
            self.config_tokenizer.bos_token.clone(),
            self.config_tokenizer.eos_token.clone(),
            Some(prompt.clone()),
            false,
            false,
        );
        let prompt_processed = prompt_template
            .apply_chat_template()
            .map_err(candle_core::Error::wrap);
        let prompt = if prompt_processed.is_ok() {
            prompt_processed.unwrap()
        } else {
            tracing::error!(
                "Applying Chat Template failed: {:?}, use default template!",
                prompt_processed
            );
            self.default_chat_template.replace("{}", &prompt)
        };

        tracing::info!("Prompt after applying Chat Template: {}", prompt);
        prompt
    }

    pub fn generate(
        &mut self,
        prompts: &Vec<String>,
        params: &SamplingParams,
    ) -> Result<Vec<(usize, usize, usize, String)>> {
        let mut params = params.clone();
        if prompts.len() * params.max_tokens > self.econfig.max_num_batched_tokens {
            params.max_tokens = self.econfig.max_num_batched_tokens / prompts.len();
            tracing::info!("Adjusted max_tokens to {}", params.max_tokens);
        }
        for prompt in prompts {
            self.add_request(self.apply_chat_template(prompt).as_str(), params.clone())?;
        }

        let mut decode_start_time = 0;
        let mut outputs = HashMap::new();
        let mut total_decoded_tokens = 0;
        let mut decode_time_taken = 0f32;

        let tokenizer = self.tokenizer.clone();
        let mut stream_decoder = tokenizer.decode_stream(true);
        while !self.scheduler.is_finished() {
            let step_output = self.step()?;
            if decode_start_time == 0 {
                decode_start_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize;
            }
            for (seq_id, token_ids) in step_output {
                match token_ids {
                    Either::Left(token_id) => {
                        if prompts.len() == 1 {
                            if let Ok(Some(output)) = stream_decoder.step(token_id) {
                                print!("{}", output);
                                use std::io::Write;
                                let _ = std::io::stdout().flush();
                            }
                        }
                    }
                    Either::Right(ids) => {
                        outputs.insert(seq_id, ids);
                    }
                }
            }

            total_decoded_tokens += prompts.len();
            if prompts.len() > 1 && (total_decoded_tokens / prompts.len()) % 100 == 0 {
                let duration = (SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize
                    - decode_start_time) as f32
                    / 1000.0; //maximum time costs for decoding
                if duration > decode_time_taken {
                    decode_time_taken = duration;
                }

                tracing::info!(
                    "[{} requests] {} tokens generated in {:.2} s (avg decoding throughput {:.2} tokens/s)",
                    prompts.len(),
                    total_decoded_tokens,
                    decode_time_taken,
                    total_decoded_tokens as f32 / decode_time_taken
                );
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
