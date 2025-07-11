//src/core/engine.rs
use super::runner::ModelRunner;
use super::scheduler::Scheduler;
use super::sequence::Sequence;
use crate::core::GenerationOutput;
use crate::log_info;
use crate::models::layers::VarBuilderX;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, ModelType, SamplingParams};
use crate::utils::init_config_tokenizer;
use crate::utils::{chat_template::ChatTemplate, get_kvcache_blocks, new_device};
use candle_core::{DType, Result};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{channel, Receiver, Sender};

pub static GLOBAL_RT: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build global Tokio runtime")
});

#[derive(Debug)]
pub enum StreamItem {
    Token(String),
    Completion((usize, usize, String)),
    Done(String),
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestType {
    Stream,
    Completion,
}

pub struct LLMEngine {
    model_runner: ModelRunner,
    pub scheduler: Scheduler,
    pub tokenizer: Tokenizer,
    econfig: EngineConfig,
    default_chat_template: String,
    template: ChatTemplate,
    stream_decoders: HashMap<usize, super::DecodeStreamType>,
    stream_senders: HashMap<usize, Sender<StreamItem>>,
    request_types: HashMap<usize, RequestType>,
    decode_start_times: HashMap<usize, usize>,
    active_requests: HashSet<usize>,
}

impl LLMEngine {
    #[allow(unused_mut)]
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Arc<RwLock<Self>>> {
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

        let mut model_runner = ModelRunner::new(
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

        #[cfg(all(feature = "cuda", feature = "graph"))]
        match model_runner.warmup_capture() {
            Ok(_) => {
                log_info!("Cuda graph captured for performance enhancement!")
            }
            Err(e) => crate::log_error!("Unable to capture cuda graph {:?}!", e),
        }

        let engine = Arc::new(RwLock::new(Self {
            model_runner,
            scheduler,
            tokenizer,
            econfig,
            default_chat_template,
            template,
            stream_decoders: HashMap::new(),
            stream_senders: HashMap::new(),
            request_types: HashMap::new(),
            decode_start_times: HashMap::new(),
            active_requests: HashSet::new(),
        }));
        Self::start_engine(engine.clone());
        Ok(engine)
    }

    fn add_request_(&mut self, params: &SamplingParams, prompt: &str) -> Result<(usize, usize)> {
        let tokens = self.tokenizer.encode(prompt, true).expect("encode failed!");
        let token_ids: Vec<u32> = tokens.get_ids().iter().map(|&x| x).collect();
        let length = token_ids.len();
        let seq = Sequence::new(token_ids, self.econfig.block_size, params.clone());
        let seq_id = self.scheduler.add(seq);

        let tokenizer = self.tokenizer.clone();
        let leaked: &'static _ = Box::leak(Box::new(tokenizer));
        let decoder = leaked.decode_stream(false);
        let wrapped = super::StreamWithTokenizer {
            _tokenizer: unsafe { Box::from_raw(leaked as *const _ as *mut _) },
            stream: decoder,
        };
        let boxed_decoder: Box<dyn super::DecodeStreamTrait + Send + Sync> = Box::new(wrapped);

        self.stream_decoders.insert(seq_id, boxed_decoder);
        self.active_requests.insert(seq_id);
        Ok((seq_id, length))
    }

    pub fn add_request(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        request_type: RequestType,
    ) -> Result<(usize, usize, Receiver<StreamItem>)> {
        let (seq_id, prompt_length) = self.add_request_(params, prompt)?;
        let (tx, rx) = channel(16);
        self.stream_senders.insert(seq_id, tx);
        self.request_types.insert(seq_id, request_type.clone());
        if request_type != RequestType::Completion {
            log_info!(
                "[{:?}] A new request [Seq_id {}] with prompt length {} added for inference!",
                request_type,
                seq_id,
                prompt_length
            );
        }
        Ok((seq_id, prompt_length, rx))
    }

    pub fn step(&mut self) -> Result<()> {
        // Get scheduled sequence indexes and prefill flag
        let (scheduled_ids, is_prefill) = self.scheduler.schedule();
        if scheduled_ids.is_empty() {
            return Ok(());
        }

        // Get immutable references to scheduled sequences for model_runner
        let seqs = self.scheduler.get_sequences(&scheduled_ids);

        // Run model on the scheduled sequences
        let output_ids = self.model_runner.run(&seqs, is_prefill)?;

        // Postprocess sequences by modifying them inside the scheduler
        self.scheduler.postprocess(&scheduled_ids, &output_ids);

        for &idx in &scheduled_ids {
            if let Some(s) = self.scheduler.get_running(idx) {
                let seq_id = s.id;
                if s.is_finished() {
                    if let Some(_) = self.stream_decoders.get_mut(&seq_id) {
                        if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                            if let Some(request_type) = self.request_types.get(&seq_id) {
                                if *request_type == RequestType::Stream {
                                    let _ = sender
                                        .try_send(StreamItem::Done("data: [DONE]".to_string()));
                                } else {
                                    let decode_output = self
                                        .tokenizer
                                        .decode(&s.output_ids, true)
                                        .expect("unable to decode!");
                                    let decode_start_time = if let Some(decode_start_time) =
                                        self.decode_start_times.get(&seq_id)
                                    {
                                        *decode_start_time
                                    } else {
                                        0usize
                                    };
                                    let _ = sender.try_send(StreamItem::Completion((
                                        decode_start_time,
                                        s.output_ids.len(),
                                        decode_output,
                                    )));
                                }
                            }
                            self.stream_senders.remove(&seq_id);
                        }
                        self.stream_decoders.remove(&seq_id);
                    }

                    if let Some(_) = self.active_requests.get(&seq_id) {
                        self.active_requests.remove(&seq_id);
                    }
                } else {
                    if !self.decode_start_times.contains_key(&seq_id) {
                        let decode_start_time = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis() as usize;
                        self.decode_start_times.insert(seq_id, decode_start_time);
                    }

                    let token_id = s.last_token;
                    if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                        if let Some(tok) = decoder.step(token_id) {
                            if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                                if let Some(request_type) = self.request_types.get(&seq_id) {
                                    let result = sender.try_send(StreamItem::Token(tok.clone()));
                                    if *request_type == RequestType::Stream {
                                        if result.is_err() {
                                            log_info!(
                                                "[seq_id {}] Error when sending token to client",
                                                seq_id
                                            );
                                            self.scheduler.cancel(seq_id);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.scheduler.clear_finished();
        Ok(())
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

    pub fn is_idle(&self) -> bool {
        self.active_requests.is_empty()
    }

    pub fn generate_sync(
        &mut self,
        params: &SamplingParams,
        prompts: Vec<String>,
    ) -> Result<Vec<(usize, usize, mpsc::Receiver<StreamItem>)>> {
        let mut receivers = Vec::new();
        for prompt in &prompts {
            if let Ok((seq_id, prompt_length, rx)) =
                self.add_request(params, prompt, RequestType::Completion)
            {
                receivers.push((seq_id, prompt_length, rx));
            }
        }

        Ok(receivers)
    }

    pub async fn collect_sync_results(
        mut receivers: Vec<(usize, usize, mpsc::Receiver<StreamItem>)>,
    ) -> Result<Vec<GenerationOutput>> {
        let mut results = Vec::new();
        let batch_size = receivers.len();
        let mut decode_start_time = 0;
        let mut decoded_tokens = 0;
        let mut map_finished = HashMap::<usize, bool>::new();

        loop {
            if results.len() >= batch_size || map_finished.len() >= batch_size {
                println!(
                    "[{} request(s)] finished with {} tokens!",
                    batch_size, decoded_tokens
                );
                break;
            }

            for (seq_id, prompt_length, rx) in &mut receivers {
                if map_finished.contains_key(seq_id) {
                    continue;
                }
                let num_active_requests = batch_size - map_finished.keys().len();
                match rx.recv().await {
                    Some(StreamItem::Completion((
                        decode_start_time,
                        decoded_length,
                        decode_output,
                    ))) => {
                        // println!(
                        //     "Sequence [seq_id {}] finished with {} tokens!",
                        //     *seq_id, decoded_length
                        // );
                        let decode_finish_time = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis() as usize;
                        results.push(GenerationOutput {
                            seq_id: *seq_id,
                            prompt_length: *prompt_length,
                            decode_start_time,
                            decode_finish_time,
                            decoded_length,
                            decode_output,
                        });
                        map_finished.insert(*seq_id, true);
                    }
                    Some(StreamItem::Token(_)) => {
                        decoded_tokens += 1;
                        if decode_start_time == 0 {
                            decode_start_time = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time went backwards")
                                .as_millis()
                                as usize;
                        }

                        let decode_time_taken = (SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis() as usize
                            - decode_start_time)
                            as f32
                            / 1000.0;

                        if decoded_tokens % (50 * batch_size) == 0 {
                            println!(
                                "[{} active request(s)] {} tokens generated in {:.2} s (avg decoding throughput {:.2} tokens/s)",
                                num_active_requests,
                                decoded_tokens,
                                decode_time_taken,
                                decoded_tokens as f32 / decode_time_taken
                            );
                        }
                    }
                    Some(StreamItem::Done(_)) => {
                        println!("Sequence [seq_id {}] finished!", *seq_id);
                    }
                    Some(StreamItem::Error(e)) => {
                        eprintln!("Error: {}", e);
                        break;
                    }
                    _ => {
                        println!(
                            "Sequence [seq_id {}] error occurred while decoding!",
                            *seq_id
                        );
                        break;
                    }
                }
            }
        }
        Ok(results)
    }

    pub fn generate_stream(
        &mut self,
        params: &SamplingParams,
        prompt: String,
    ) -> Result<(usize, usize, mpsc::Receiver<StreamItem>)> {
        if let Ok((seq_id, prompt_length, rx)) =
            self.add_request(params, &prompt, RequestType::Stream)
        {
            Ok((seq_id, prompt_length, rx))
        } else {
            candle_core::bail!("Failed to create stream!")
        }
    }

    pub fn start_engine(engine: Arc<RwLock<Self>>) {
        GLOBAL_RT.spawn(async move {
            let engine = engine.clone();
            loop {
                let idle = {
                    let guard = engine.read();
                    guard.is_idle()
                };

                if idle {
                    tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
                    continue;
                }

                {
                    let mut guard = engine.write();
                    if let Err(e) = guard.step() {
                        panic!("[Engine Loop] Step error: {:?}", e);
                    }
                }

                tokio::task::yield_now().await;
            }
        });
    }
}
