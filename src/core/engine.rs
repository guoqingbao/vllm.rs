//src/core/engine.rs
use super::runner::ModelRunner;
use super::scheduler::Scheduler;
use super::sequence::Sequence;
use super::GenerationOutput;
use crate::log_info;
use crate::models::layers::VarBuilderX;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, ModelType, SamplingParams};
use crate::utils::init_config_tokenizer;
use crate::utils::{chat_template::ChatTemplate, get_kvcache_blocks, new_device};
use candle_core::{DType, Result};
use either::Either;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

pub struct LLMEngine {
    model_runner: ModelRunner,
    pub scheduler: Scheduler,
    pub tokenizer: Tokenizer,
    econfig: EngineConfig,
    default_chat_template: String,
    template: ChatTemplate,
}

impl LLMEngine {
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Self> {
        let device = new_device(econfig.device_id.unwrap_or(0))?;
        log_info!("Loading model...");
        let (config, config_tokenizer, tokenizer) = init_config_tokenizer(econfig)?;
        let vb = VarBuilderX::new(&econfig.model_path, dtype, &device)?;
        log_info!("{:?}\n", config);

        log_info!("{:?}\n", config_tokenizer);

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

        let rope_key_map: HashMap<&str, bool> = [
            ("Qwen2ForCausalLM", false),
            ("Qwen3ForCausalLM", false),
            ("MistralForCausalLM", false),
            ("LlamaForCausalLM", false),
            ("qwen2", false),
            ("qwen3", false),
            ("llama", true),
            ("mistral", true),
        ]
        .iter()
        .cloned()
        .collect();

        let arch = config.architectures[0].as_str();
        let (model_type, default_chat_template) = match arch {
            "Qwen2ForCausalLM"
            | "Qwen2ForConditionalGeneration"
            | "Qwen3ForCausalLM"
            | "Qwen3ForConditionalGeneration"
            | "qwen2"
            | "qwen3" => (
                ModelType::Qwen3,
                "<|im_start|>user\n {} <|im_end|>".to_string(),
            ),
            "LlamaForCausalLM"
            | "MistralForCausalLM"
            | "LlamaForConditionalGeneration"
            | "llama"
            | "mistral"
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
                    )
                } else {
                    //llama2
                    (ModelType::LLaMa, "[INST] {} [/INST]".to_string())
                }
            }
            _ => candle_core::bail!("Unsupported architecture: {}", config.architectures[0]),
        };

        let is_rope_i = if rope_key_map.contains_key(arch) {
            rope_key_map[arch]
        } else {
            false
        };

        log_info!("Use ROPE interleaved {is_rope_i}");

        let num_blocks = get_kvcache_blocks(
            econfig.kvcache_mem_gpu.unwrap_or(4096),
            econfig.block_size,
            &config,
            econfig.num_shards.unwrap_or(1),
            dtype,
        );

        econfig.num_blocks = num_blocks;
        econfig.max_num_batched_tokens = num_blocks * econfig.block_size;
        log_info!("{:?}", econfig);

        log_info!(
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
        log_info!("Model loaded.\n");

        let template = ChatTemplate::new(
            None,
            config_tokenizer.chat_template.clone(),
            config_tokenizer.bos_token.clone(),
            config_tokenizer.eos_token.clone(),
            None,
            true,
            true,
        );

        Ok(Self {
            model_runner,
            scheduler,
            tokenizer,
            econfig,
            default_chat_template,
            template,
        })
    }

    pub fn add_request(&mut self, params: SamplingParams, prompt: &str) -> Result<(usize, usize)> {
        let tokens = self.tokenizer.encode(prompt, true).expect("encode failed!");
        let token_ids: Vec<u32> = tokens.get_ids().iter().map(|&x| x).collect();
        let length = token_ids.len();
        let seq = Sequence::new(token_ids, self.econfig.block_size, params);
        let id = self.scheduler.add(seq);
        Ok((id, length))
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

    pub fn cancel(&mut self, seq_id: usize) {
        self.scheduler.cancel(seq_id);
        self.scheduler.clear_finished();
    }

    pub fn apply_chat_template(&self, messages: &Vec<Message>, log: bool) -> String {
        let mut prompt_template = self.template.clone();
        prompt_template.set_messages(messages);
        let prompt_processed = prompt_template
            .apply_chat_template(log)
            .map_err(candle_core::Error::wrap);
        let prompt = if prompt_processed.is_ok() {
            prompt_processed.unwrap()
        } else {
            if log {
                tracing::error!(
                    "Applying Chat Template failed: {:?}, use default template!",
                    prompt_processed
                );
            }

            let mut prompt = "".to_string();
            for message in messages {
                if message.role == "user" {
                    prompt += &self.default_chat_template.replace("{}", &message.content);
                    prompt += "\n";
                }
            }
            prompt
        };

        if log {
            log_info!(
                "Prompt after applying Chat Template: {}",
                prompt.replace("\n", "")
            );
        }
        prompt
    }

    pub fn generate(
        &mut self,
        params: &SamplingParams,
        prompts: &Vec<String>,
    ) -> Result<Vec<GenerationOutput>> {
        let mut params = params.clone();
        if prompts.len() * params.max_tokens > self.econfig.max_num_batched_tokens {
            params.max_tokens = self.econfig.max_num_batched_tokens / prompts.len();
            log_info!("Adjusted max_tokens to {}", params.max_tokens);
        }
        let mut map_prompt_length = HashMap::<usize, usize>::new();
        for prompt in prompts {
            let (seq_id, length) = self.add_request(params.clone(), prompt)?;
            map_prompt_length.insert(seq_id, length);
        }

        let mut decode_start_time = 0;
        let mut outputs = HashMap::new();
        let mut total_decoded_tokens = 0;
        let mut decode_time_taken = 0f32;

        let tokenizer = self.tokenizer.clone();
        let mut stream_decoder = tokenizer.decode_stream(false);
        while !self.scheduler.is_finished() {
            let step_output = self.step()?;
            if decode_start_time == 0 {
                decode_start_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize;
            }
            let mut vec_ids = Vec::new();
            for (seq_id, token_ids) in step_output {
                match token_ids {
                    Either::Left(token_id) => {
                        total_decoded_tokens += 1;
                        if prompts.len() == 1 {
                            if let Ok(Some(output)) = stream_decoder.step(token_id) {
                                print!("{}", output);
                                use std::io::Write;
                                let _ = std::io::stdout().flush();
                            }
                        }
                    }
                    Either::Right(ids) => {
                        if prompts.len() > 1 {
                            log_info!(
                                "[seq_id {}] finished ({} tokens generated)",
                                seq_id,
                                ids.len()
                            );
                        }
                        outputs.insert(seq_id, ids);
                    }
                }
                vec_ids.push(seq_id);
            }

            if prompts.len() > 1 && total_decoded_tokens % (prompts.len() * 50) == 0 {
                let duration = (SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize
                    - decode_start_time) as f32
                    / 1000.0; //maximum time costs for decoding
                if duration > decode_time_taken {
                    decode_time_taken = duration;
                }

                log_info!(
                    "[{} request(s)] {} tokens generated in {:.2} s (avg decoding throughput {:.2} tokens/s)",
                    vec_ids.len(),
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
            let decode_output = self
                .tokenizer
                .decode(&token_ids, true)
                .expect("unable to decode!");
            let output = GenerationOutput {
                seq_id,
                prompt_length: map_prompt_length[&seq_id],
                decode_start_time,
                decoded_length: token_ids.len(),
                decode_output,
            };
            results.push(output);
        }

        Ok(results)
    }
}
