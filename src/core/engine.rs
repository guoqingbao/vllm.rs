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
use either::Either;
use futures::future::join_all;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::{task, time::sleep};

pub static GLOBAL_RT: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build global Tokio runtime")
});

#[derive(Debug)]
pub enum StreamItem {
    Token(String),                               //streaming
    TokenID(u32),                                //completion
    Completion((usize, usize, usize, Vec<u32>)), //completion
    Done((usize, String)),
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
    prompt_start_times: HashMap<usize, usize>,
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
        let config_model_len = config.max_model_len.unwrap_or(
            config_tokenizer
                .model_max_length
                .unwrap_or(config.max_position_embeddings as f64) as usize,
        );

        econfig.max_model_len = Some(std::cmp::min(
            econfig.max_model_len.unwrap_or(4096),
            config_model_len,
        ));

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
            econfig.max_num_seqs,
            econfig.max_model_len.unwrap_or(4096),
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
            prompt_start_times: HashMap::new(),
            decode_start_times: HashMap::new(),
            active_requests: HashSet::new(),
        }));
        Self::start_engine(engine.clone());
        Ok(engine)
    }

    fn add_request_(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        request_type: &RequestType,
    ) -> Result<(usize, usize)> {
        let tokens = self
            .tokenizer
            .encode_fast(prompt, true)
            .expect("encode failed!");
        let token_ids: Vec<u32> = tokens.get_ids().iter().map(|&x| x).collect();
        let length = token_ids.len();
        let mut params = params.clone();
        let max_model_len = self.econfig.max_model_len.unwrap_or(params.max_tokens);
        //we also need to consider prompt length
        if length + params.max_tokens > max_model_len {
            params.max_tokens = max_model_len - length;
        }
        let seq = Sequence::new(token_ids, self.econfig.block_size, params);
        let seq_id = self.scheduler.add(seq);

        if *request_type == RequestType::Stream {
            let tokenizer = self.tokenizer.clone();
            let leaked: &'static _ = Box::leak(Box::new(tokenizer));
            let decoder = leaked.decode_stream(false);
            let wrapped = super::StreamWithTokenizer {
                _tokenizer: unsafe { Box::from_raw(leaked as *const _ as *mut _) },
                stream: decoder,
            };
            let boxed_decoder: Box<dyn super::DecodeStreamTrait + Send + Sync> = Box::new(wrapped);
            self.stream_decoders.insert(seq_id, boxed_decoder);
        }
        self.active_requests.insert(seq_id);
        Ok((seq_id, length))
    }

    pub fn add_request(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        request_type: RequestType,
    ) -> Result<(usize, usize, Receiver<StreamItem>)> {
        let (seq_id, prompt_length) = self.add_request_(params, prompt, &request_type)?;
        let (tx, rx) = channel(if request_type == RequestType::Stream {
            16
        } else {
            256
        });
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
        pub struct DecodedIds(Either<Vec<usize>, Vec<usize>>);

        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        // Get scheduled sequence indexes and prefill flag
        let (scheduled_ids, is_prefill) = self.scheduler.schedule();
        let decoded_ids = if !scheduled_ids.is_empty() {
            // Get immutable references to scheduled sequences for model_runner
            let seqs = self.scheduler.get_sequences(&scheduled_ids);

            // Run model on the scheduled sequences
            let output_ids = self.model_runner.run(&seqs, is_prefill)?;

            // Postprocess sequences by modifying them inside the scheduler
            self.scheduler.postprocess(&scheduled_ids, &output_ids);
            DecodedIds(Either::Left(scheduled_ids))
        } else {
            crate::log_info!("No more kv cache available, free all resources!");
            DecodedIds(Either::Right(self.scheduler.release_all_waitings()))
        };

        let (indices, is_running): (&Vec<usize>, bool) = match &decoded_ids.0 {
            Either::Left(indices) => (indices, true),
            Either::Right(indices) => (indices, false),
        };

        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        for &idx in indices {
            let sq = if is_running {
                self.scheduler.get_running(idx)
            } else {
                self.scheduler.get_waiting(idx)
            };
            if let Some(s) = sq {
                let seq_id = s.id;
                if s.is_finished() {
                    if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                        if let Some(request_type) = self.request_types.get(&seq_id) {
                            let prompt_start_time = if let Some(prompt_start_time) =
                                self.prompt_start_times.get(&seq_id)
                            {
                                *prompt_start_time
                            } else {
                                start_time
                            };

                            if *request_type == RequestType::Stream {
                                let _ = sender.try_send(StreamItem::Done((
                                    prompt_start_time,
                                    "data: [DONE]".to_string(),
                                )));
                            } else {
                                let decode_finish_time = cur_time;
                                let decode_start_time = if let Some(decode_start_time) =
                                    self.decode_start_times.get(&seq_id)
                                {
                                    *decode_start_time
                                } else {
                                    cur_time
                                };

                                let _ = sender.try_send(StreamItem::Completion((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    s.output_ids.clone(),
                                )));
                            }
                        }
                    }
                    self.stream_decoders.remove(&seq_id);
                    if let Some(_) = self.active_requests.get(&seq_id) {
                        self.active_requests.remove(&seq_id);
                    }
                } else {
                    if !self.decode_start_times.contains_key(&seq_id) {
                        self.decode_start_times.insert(seq_id, cur_time);
                    }
                    if is_prefill && !self.prompt_start_times.contains_key(&seq_id) {
                        self.prompt_start_times.insert(seq_id, start_time);
                    }
                    let token_id = s.last_token;
                    if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                        if let Some(request_type) = self.request_types.get(&seq_id) {
                            if *request_type == RequestType::Stream {
                                if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                                    if let Some(tok) = decoder.step(token_id) {
                                        let result =
                                            sender.try_send(StreamItem::Token(tok.clone()));
                                        if result.is_err() {
                                            log_info!(
                                                "[seq_id {}] Error when sending token to client",
                                                seq_id
                                            );
                                            self.scheduler.cancel(seq_id);
                                        }
                                    }
                                }
                            } else {
                                //completion request will be decoded at the final stage (at once)
                                let _ = sender.try_send(StreamItem::TokenID(token_id));
                            }
                        }
                    }
                }
            }
        }
        self.scheduler.clear_finished();

        if indices.is_empty() {
            self.active_requests.clear();
        }
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
                crate::log_error!(
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
        params: &Vec<SamplingParams>,
        prompts: Vec<String>,
    ) -> Result<Vec<(usize, usize, mpsc::Receiver<StreamItem>)>> {
        if params.len() != prompts.len() {
            candle_core::bail!("size of sampling parameters is not match with size of prompts!");
        }
        let mut receivers = Vec::new();
        for (param, prompt) in params.iter().zip(prompts.iter()) {
            if let Ok((seq_id, prompt_length, rx)) =
                self.add_request(param, prompt, RequestType::Completion)
            {
                receivers.push((seq_id, prompt_length, rx));
            }
        }

        Ok(receivers)
    }

    pub async fn collect_sync_results(
        receivers: Vec<(usize, usize, mpsc::Receiver<StreamItem>)>,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Vec<GenerationOutput>> {
        let decoded_tokens = Arc::new(AtomicUsize::new(0));
        let decode_start_time = Arc::new(AtomicUsize::new(0));
        let decode_start_time_clone = Arc::clone(&decode_start_time);

        // Spawn a background reporting task
        let decoded_tokens_clone = decoded_tokens.clone();
        let reporter = task::spawn(async move {
            let mut last_logged = 0;
            loop {
                sleep(Duration::from_secs(1)).await;
                let start_time = decode_start_time_clone.load(Ordering::SeqCst);
                if start_time > 0 {
                    let count = decoded_tokens_clone.load(Ordering::Relaxed);
                    if count == last_logged {
                        crate::log_info!("Finalizing...");
                        break;
                    }
                    last_logged = count;

                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as usize;

                    let elapsed = (now - start_time) as f32 / 1000.0;

                    println!(
                        "[Live Throughput] {} tokens in {:.2}s ({:.2} tokens/s)",
                        count,
                        elapsed,
                        count as f32 / elapsed
                    );
                }
            }
        });

        // Spawn tasks for each receiver
        let tasks = receivers
            .into_iter()
            .map(|(seq_id, prompt_length, mut rx)| {
                let decoded_tokens = decoded_tokens.clone();
                let decode_start_time_clone = decode_start_time.clone();
                let tokenizer = Arc::clone(&tokenizer);
                task::spawn(async move {
                    let mut output: Option<GenerationOutput> = None;

                    while let Some(msg) = rx.recv().await {
                        match msg {
                            StreamItem::Completion((
                                prompt_start,
                                decode_start,
                                decode_finish,
                                decoded_ids,
                            )) => {
                                let decoded_len = decoded_ids.len();
                                let decode_output = tokenizer
                                    .decode(&decoded_ids, true)
                                    .expect("unable to decode!");

                                output = Some(GenerationOutput {
                                    seq_id,
                                    prompt_length,
                                    prompt_start_time: prompt_start,
                                    decode_start_time: decode_start,
                                    decode_finish_time: decode_finish,
                                    decoded_length: decoded_len,
                                    decode_output,
                                });
                                break;
                            }
                            StreamItem::Token(_) | StreamItem::TokenID(_) => {
                                decoded_tokens.fetch_add(1, Ordering::Relaxed);

                                decode_start_time_clone
                                    .compare_exchange(
                                        0,
                                        SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis()
                                            as usize,
                                        Ordering::SeqCst,
                                        Ordering::SeqCst,
                                    )
                                    .ok(); // set it only if it was 0
                            }
                            StreamItem::Done(_) => {
                                println!("Sequence [seq_id {}] finished!", seq_id);
                                break;
                            }
                            StreamItem::Error(e) => {
                                eprintln!("Error in seq {}: {}", seq_id, e);
                                break;
                            }
                        }
                    }

                    output
                })
            });

        // Wait for all decoding tasks
        let outputs = join_all(tasks).await;

        // Wait for final reporter update (1s grace)
        reporter.await.unwrap();

        // Collect successful outputs
        let results: Vec<_> = outputs
            .into_iter()
            .filter_map(|r| r.ok().flatten())
            .collect();

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
