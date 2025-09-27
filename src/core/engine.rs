//src/core/engine.rs
use super::runner::{ModelRunner, Seqs};
use super::scheduler::Scheduler;
use super::sequence::Sequence;
use crate::core::sequence::DecodeSequence;
use crate::core::GenerationOutput;
use crate::models::layers::distributed::Comm;
#[cfg(feature = "nccl")]
use crate::models::layers::distributed::Id;
use crate::models::layers::VarBuilderX;
use crate::runner::{receive_local, send_local, MessageType, RunnerInitRequest};
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, SamplingParams};
use crate::utils::progress::{progress_worker, ProgressReporter};
use crate::utils::progress::{spawn_progress_thread, ProgressLike};
use crate::utils::{chat_template::ChatTemplate, get_kvcache_blocks};
use crate::utils::{get_runner_path, init_config_tokenizer, spawn_runner};
use crate::{log_info, log_warn};
use candle_core::{DType, Result};
use either::Either;
use futures::future::join_all;
use interprocess::local_socket::traits::Listener;
use interprocess::local_socket::{GenericNamespaced, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use interprocess::TryClone;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader};
use std::rc::Rc;
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

#[derive(Debug, Clone)]
pub enum StreamItem {
    Token(String),                               //streaming
    TokenID(u32),                                //completion
    Completion((usize, usize, usize, Vec<u32>)), //completion
    Done((usize, usize, usize, usize)),          //streaming end
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestType {
    Stream,
    Completion,
}

pub enum RunnerType {
    Thread(ModelRunner),
    Process(Vec<LocalStream>),
}

pub struct LLMEngine {
    pub runners: RunnerType,
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
    active_sessions: VecDeque<(usize, String)>,
    cancelled_sequences: Vec<usize>,
}

impl LLMEngine {
    #[allow(unused_mut)]
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Arc<RwLock<Self>>> {
        let (model_pathes, is_gguf, mut config, config_tokenizer, tokenizer, mut generation_cfg) =
            init_config_tokenizer(econfig)?;

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

        if econfig.max_model_len.unwrap() < config_model_len {
            log_warn!(
                "***This model has maximum context {} but only {} is set to use in Engine config!***",
                config_model_len,
                econfig.max_model_len.unwrap()
            );
        }

        assert!(
            config.architectures.len() == 1,
            "Only one architecture is supported at the moment!"
        );

        let (model_type, default_chat_template, is_rope_i) =
            crate::utils::get_arch_rope(&tokenizer, config.architectures[0].clone())?;
        log_info!("Use ROPE interleaved {is_rope_i}");

        match (&generation_cfg, &mut econfig.generation_cfg) {
            (Some(gen_cfg), None) => {
                econfig.generation_cfg = Some(gen_cfg.clone());
            }
            (Some(gen_cfg), Some(egen_cfg)) => {
                if egen_cfg.frequency_penalty.is_none() {
                    egen_cfg.frequency_penalty = gen_cfg.frequency_penalty;
                }
                if egen_cfg.presence_penalty.is_none() {
                    egen_cfg.presence_penalty = gen_cfg.presence_penalty;
                }
                if egen_cfg.temperature.is_none() {
                    egen_cfg.temperature = gen_cfg.temperature;
                    egen_cfg.top_p = gen_cfg.top_p;
                    egen_cfg.top_k = gen_cfg.top_k;
                }
            }
            _ => {
                crate::log_warn!("No generation config found for this model!");
            }
        }

        let mut device_ids = econfig.device_ids.clone().unwrap_or_default();
        if device_ids.is_empty() {
            device_ids.push(0);
        }

        let num_shards = device_ids.len();

        let num_blocks = get_kvcache_blocks(
            econfig.max_num_seqs,
            econfig.max_model_len.unwrap_or(4096),
            econfig.block_size,
            &config,
            num_shards,
            if econfig.fp8_kvcache.unwrap_or(false) {
                DType::U8
            } else {
                dtype
            },
        );

        econfig.num_blocks = num_blocks;
        econfig.max_num_batched_tokens = num_blocks * econfig.block_size;
        econfig.num_shards = Some(num_shards);
        config.fp8_kvcache = econfig.fp8_kvcache;
        log_info!("{:?}", econfig);

        log_info!("{:?}\n", config_tokenizer);

        log_info!("{:?}\n", config);

        log_info!(
            "Maximum batched tokens {} ({} blocks x Block_Size {} for KV cache).",
            econfig.max_num_batched_tokens,
            num_blocks,
            econfig.block_size
        );

        #[cfg(not(feature = "nccl"))]
        assert!(
            num_shards == 1,
            "Multi-rank inference is only available when `nccl` feature is enabled!"
        );

        #[cfg(feature = "nccl")]
        let use_runner = if num_shards > 1 {
            // if !econfig.flash_context.unwrap_or(false) {
            //     crate::log_warn!("Context cache is forced to be enabled under multi-rank inference if context-cache or flash-context feature built-in!");
            //     econfig.flash_context = Some(true);
            // }
            true
        } else {
            if cfg!(feature = "flash-attn") || cfg!(feature = "python") {
                econfig.flash_context.unwrap_or(false)
            } else {
                false
            }
        };

        #[cfg(not(feature = "nccl"))]
        assert!(
            num_shards == 1,
            "Multi-gpu inference is only available when `cuda` and `nccl` features enabled!"
        );
        #[cfg(not(feature = "nccl"))]
        let use_runner = num_shards > 1;

        log_info!("Check use_runner {:?}", use_runner);

        let runners = if !use_runner {
            let device = crate::utils::new_device(device_ids[0])?;
            log_info!("Loading model...");
            let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                Arc::new(RwLock::new(Box::new(ProgressReporter::new(0))));
            let handle = progress_worker(1, config.num_hidden_layers, &reporter);
            let vb = VarBuilderX::new(&model_pathes, is_gguf, dtype, &device)?;
            let mut model_runner = ModelRunner::new(
                model_type,
                &vb,
                #[cfg(not(feature = "nccl"))]
                Rc::new(Comm::default()),
                #[cfg(feature = "nccl")]
                Rc::new(
                    Comm::from_rank(
                        device.as_cuda_device().unwrap().cuda_device(),
                        0,
                        1,
                        Id::new().unwrap(),
                    )
                    .unwrap(),
                ),
                &econfig,
                &config,
                dtype,
                is_rope_i,
                device.clone(),
                reporter,
            )?;

            #[cfg(all(feature = "cuda", feature = "graph"))]
            match model_runner.warmup_capture() {
                Ok(_) => {
                    log_info!("Cuda graph captured for performance enhancement!")
                }
                Err(e) => crate::log_error!("Unable to capture cuda graph {:?}!", e),
            }

            let _ = handle.join();
            RunnerType::Thread(model_runner)
        } else {
            log_info!("Loading model with runner(s)...");

            #[cfg(feature = "nccl")]
            let nccl_id = Id::new().unwrap();

            let runner_path = get_runner_path()?;

            #[cfg(feature = "python")]
            pyo3::Python::with_gil(|py| {
                for (rank, _) in device_ids.iter().enumerate() {
                    let sock_name = format!("@vllm-rs-runner-{}", rank);
                    spawn_runner(py, &runner_path.display().to_string(), &sock_name)
                        .expect("Failed to spawn runner. \n\r*****Tips: runner is not built within this package, use 'build.sh' script to build package with runner!");
                }
            });

            #[cfg(not(feature = "python"))]
            for (rank, _) in device_ids.iter().enumerate() {
                let sock_name = format!("@vllm-rs-runner-{}", rank);
                spawn_runner(&runner_path.display().to_string(), &sock_name)
                    .expect("Failed to spawn runner. \n\r*****Tips: runner is not built, use 'run.sh' script instead of 'cargo run'!");
            }

            let progress_sock_name = "@vllm-rs-progress".to_string();
            let progress_handle =
                spawn_progress_thread(num_shards, config.num_hidden_layers, progress_sock_name);

            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::ParallelIterator;
            let runner_streams: Result<Vec<LocalStream>> = device_ids
                .par_iter()
                .enumerate()
                .map(|(rank, dev_id)| {
                    let model_type = model_type.clone();
                    let config = config.clone();
                    let econfig = econfig.clone();
                    let sock_name = format!("@vllm-rs-runner-{}", rank);
                    let listener = ListenerOptions::new()
                        .name(
                            sock_name
                                .clone()
                                .to_ns_name::<GenericNamespaced>()
                                .expect("Failed to to_ns_name"),
                        )
                        .create_sync()
                        .expect("Failed to create listener");

                    crate::log_info!("listener starting accepting runner {}", rank);

                    // Accept one connection
                    let mut stream = listener.accept()?;
                    crate::log_info!("Accepted runner {}", rank);

                    // Wait for "ready"
                    let mut reader = BufReader::new(&mut stream);
                    let mut message = String::new();
                    reader.read_line(&mut message)?;
                    if message.trim() != "ready" {
                        return Err(candle_core::Error::Msg(format!(
                            "Runner {} did not send ready",
                            rank
                        )));
                    }

                    // Build init message
                    let init_msg = MessageType::Init(RunnerInitRequest {
                        rank,
                        dev_id: *dev_id,
                        num_shards,
                        model_type,
                        config,
                        econfig,
                        model_pathes: model_pathes.clone(),
                        is_gguf,
                        dtype: dtype.into(),
                        is_rope_i,
                        #[cfg(feature = "nccl")]
                        nccl_id: crate::runner::NcclId(nccl_id.clone()),
                    });

                    send_local(&mut vec![stream.try_clone()?], &init_msg, true)?;

                    crate::log_info!("Waiting runner {} response...", rank);

                    if let MessageType::InitAck(ack) = receive_local(&mut stream, false)? {
                        if !ack {
                            candle_core::bail!("Runner {} failed to initialize", rank);
                        }
                    } else {
                        candle_core::bail!("Runner {} unable to initialize", rank);
                    }

                    crate::log_info!("Runner {} started!", rank);

                    Ok(stream)
                })
                .collect();

            if let Ok(Some(handle)) = progress_handle.join() {
                let _ = handle.join();
            }
            RunnerType::Process(runner_streams?)
        };

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

        let engine = Arc::new(RwLock::new(Self {
            runners,
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
            active_sessions: VecDeque::new(),
            cancelled_sequences: Vec::new(),
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
        if let Some(max_model_len) = self.econfig.max_model_len {
            if length > max_model_len - 1 {
                candle_core::bail!(
                    "Inputs token length {} exceed max_model_len {}",
                    length,
                    max_model_len
                );
            }
        }
        let mut params = params.clone();
        params.max_tokens = Some(
            params
                .max_tokens
                .unwrap_or(self.econfig.max_tokens.unwrap_or(16384)),
        );
        let max_tokens = params.max_tokens.unwrap();

        let max_model_len = self.econfig.max_model_len.unwrap_or(max_tokens);
        //we also need to consider prompt length
        if length + max_tokens > max_model_len {
            params.max_tokens = Some(max_model_len - length);
        }

        let remain_tokens = (self.econfig.max_num_seqs * max_model_len) as isize
            - self.get_num_cached_tokens() as isize;

        if remain_tokens < 1 || length as isize > remain_tokens {
            candle_core::bail!(
                "Remaining {} kvcache tokens, but your prompt length is {}, please request later!",
                remain_tokens,
                length
            );
        }

        if let Some(gen_cfg) = &self.econfig.generation_cfg {
            let temperature = params.temperature.or(gen_cfg.temperature);
            let top_k = params.top_k.or(gen_cfg.top_k);
            let top_p = params.top_p.or(gen_cfg.top_p);
            params.temperature = temperature;
            params.top_k = top_k;
            params.top_p = top_p;
        }
        let session_id = params.session_id.clone();

        let session_id = if session_id.is_some() && !self.econfig.flash_context.unwrap_or(false) {
            crate::log_error!("`session_id` detected but `context-cache` is not enabled!");
            None
        } else {
            session_id.clone()
        };

        let seq_id = if let Some(session_id) = session_id {
            crate::log_warn!(
                "Cached {} sessions: {:?} ({} tokens cached).",
                self.active_sessions.len(),
                self.active_sessions,
                self.get_num_cached_tokens(),
            );
            if self.scheduler.has_cache(&session_id) {
                self.scheduler.get_cache(&session_id, token_ids)?
            } else {
                let seq = Sequence::new(token_ids, self.econfig.block_size, params);
                let seq_id = self.scheduler.add(seq);
                self.active_sessions.push_back((seq_id, session_id.clone()));
                seq_id
            }
        } else {
            let seq = Sequence::new(token_ids, self.econfig.block_size, params);
            let seq_id = self.scheduler.add(seq);
            seq_id
        };

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
                "[{:?}] A new request [Seq_id {}] with prompt length {} added for inference! (session_id {:?})\n",
                request_type,
                seq_id,
                prompt_length,
                params.session_id,
            );
        }
        Ok((seq_id, prompt_length, rx))
    }

    pub fn get_num_cached_tokens(&self) -> usize {
        self.scheduler.get_num_cached_tokens()
    }

    pub fn notify_runner_finished(&mut self, id: usize) -> Result<()> {
        match &mut self.runners {
            RunnerType::Thread(model_runner) => Ok(model_runner.finished(id)),
            RunnerType::Process(ref mut runner_streams) => {
                for stream in runner_streams {
                    send_local(
                        &mut vec![stream.try_clone()?],
                        &MessageType::FinishDecode(id),
                        false,
                    )?;
                }
                Ok(())
            }
        }
    }

    pub fn step(&mut self) -> Result<()> {
        pub struct DecodedIds(Either<Vec<usize>, Vec<usize>>);

        // Get scheduled sequence indexes and prefill flag
        let (scheduled_ids, is_prefill) = match self.scheduler.schedule() {
            Ok((ids, prefill)) => (ids, prefill),
            Err(_) => (vec![], true),
        };
        let decoded_ids = if !scheduled_ids.is_empty() {
            // Get immutable references to scheduled sequences for model_runner
            let seqs = self.scheduler.get_sequences(&scheduled_ids);

            let output_ids = match &mut self.runners {
                RunnerType::Thread(model_runner) => {
                    // Run model on the scheduled sequences in the main thread
                    model_runner.run(Seqs::SeqRefs(&seqs), is_prefill)?
                }
                RunnerType::Process(ref mut runner_streams) => {
                    let request = if is_prefill {
                        let sequences = seqs.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
                        MessageType::RunPrefill((sequences, true))
                    } else {
                        let sequences = seqs
                            .iter()
                            .map(|s| DecodeSequence::new(s))
                            .collect::<Vec<_>>();
                        MessageType::RunDecode((sequences, false))
                    };

                    let cloned_streams: Vec<LocalStream> = runner_streams
                        .iter_mut()
                        .map(|s| s.try_clone().expect("clone failed"))
                        .collect();

                    use rayon::iter::IntoParallelIterator;
                    use rayon::iter::ParallelIterator;
                    let all_outputs: Result<Vec<Vec<u32>>> = cloned_streams
                        .into_par_iter()
                        .map(|mut stream| {
                            let msg = request.clone();
                            send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                            let response = receive_local(&mut stream, false)?;

                            match response {
                                MessageType::RunResponse(output_ids) => Ok(output_ids),
                                other => {
                                    candle_core::bail!("Unexpected response type: {:?}", other)
                                }
                            }
                        })
                        .collect();

                    let all_outputs = all_outputs.map_err(candle_core::Error::wrap)?;
                    // Only run postprocess once after all runners finish (use first result)
                    if let Some(output_ids) = all_outputs.first() {
                        output_ids.clone()
                        // self.scheduler.postprocess(&scheduled_ids, output_ids);
                    } else {
                        candle_core::bail!("No output ids received from model runners");
                    }
                }
            };
            // Postprocess sequences by modifying them inside the scheduler
            if is_prefill {
                let (indices, finished_indices) =
                    self.scheduler.filter_prefill_finished(&scheduled_ids);
                if indices.is_empty() {
                    //chunked prefill, no finished
                    return Ok(());
                } else {
                    let output_ids: Vec<u32> = indices.iter().map(|&i| output_ids[i]).collect();
                    self.scheduler.postprocess(
                        &finished_indices,
                        &output_ids,
                        &self.active_sessions,
                    );
                    DecodedIds(Either::Left(finished_indices))
                }
            } else {
                self.scheduler
                    .postprocess(&scheduled_ids, &output_ids, &self.active_sessions);
                DecodedIds(Either::Left(scheduled_ids))
            }
        } else {
            DecodedIds(Either::Right(vec![]))
        };

        let (indices, is_running): (&Vec<usize>, bool) = match &decoded_ids.0 {
            Either::Left(indices) => (indices, true),
            Either::Right(indices) => (indices, false),
        };

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
                            let prompt_start_time = s.created_time();

                            let decode_finish_time = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time went backwards")
                                .as_millis()
                                as usize;
                            let decode_start_time = if let Some(decode_start_time) =
                                self.decode_start_times.get(&seq_id)
                            {
                                *decode_start_time
                            } else {
                                decode_finish_time
                            };

                            if *request_type == RequestType::Stream {
                                let _ = sender.try_send(StreamItem::Done((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    s.output_ids.len(),
                                )));
                            } else {
                                let _ = sender.try_send(StreamItem::Completion((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    s.output_ids.clone(),
                                )));
                            }
                        }
                    }
                    self.stream_senders.remove(&seq_id);
                    self.stream_decoders.remove(&seq_id);
                    if let Some(_) = self.active_requests.get(&seq_id) {
                        self.active_requests.remove(&seq_id);
                    }
                    self.decode_start_times.remove(&seq_id);
                    let _ = self.notify_runner_finished(seq_id);
                } else {
                    if !self.decode_start_times.contains_key(&seq_id) {
                        self.decode_start_times.insert(
                            seq_id,
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time went backwards")
                                .as_millis() as usize,
                        );
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
                                            self.cancelled_sequences.push(seq_id);
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
        self.check_cache();
        self.check_canceled();
        Ok(())
    }

    pub fn check_cache(&mut self) {
        //kvcache approach 95%, we need release cached requests
        let has_tokens_left = self.econfig.max_num_seqs * self.econfig.max_model_len.unwrap()
            > (self.get_num_cached_tokens() as f32 * 1.05) as usize;
        if self.active_sessions.len() > 0
            && (self.active_sessions.len() > self.econfig.max_num_seqs && !has_tokens_left)
        {
            if let Some((seq_id, session_id)) = self.active_sessions.pop_front() {
                self.scheduler.release_cache(seq_id);
                crate::log_warn!(
                    "***Cache removed for Seq {} (session id {})!\n",
                    seq_id,
                    session_id
                );
            }
        }
    }

    pub fn check_canceled(&mut self) {
        if self.cancelled_sequences.is_empty() {
            return;
        }
        for i in 0..self.cancelled_sequences.len() {
            let seq_id = self.cancelled_sequences[i];
            self.scheduler.cancel(seq_id);
            if let Some(pos) = self
                .active_sessions
                .iter()
                .position(|(id, _)| *id == seq_id)
            {
                self.active_sessions.remove(pos);
            }
            if let Some(_) = self.active_requests.get(&seq_id) {
                self.active_requests.remove(&seq_id);
            }
            self.stream_decoders.remove(&seq_id);
            self.decode_start_times.remove(&seq_id);
            self.stream_senders.remove(&seq_id);
        }
        self.scheduler.clear_finished();
        self.cancelled_sequences.clear();
    }

    pub fn cancel(&mut self, seq_id: usize) {
        self.cancelled_sequences.push(seq_id);
    }

    pub fn apply_chat_template(
        &self,
        params: &SamplingParams,
        messages: &Vec<Message>,
        log: bool,
    ) -> String {
        let mut prompt_template = self.template.clone();
        if let Some(session_id) = &params.session_id {
            if self.scheduler.has_cache(&session_id) {
                //context cache, only retrieve the last message
                prompt_template.set_messages(&vec![messages[messages.len() - 1].clone()]);
            } else {
                prompt_template.set_messages(messages);
            }
        } else {
            prompt_template.set_messages(messages);
        }
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
                                eprintln!("Error in Seq {}: {}", seq_id, e);
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

    pub fn free_resources(&mut self) {
        crate::log_error!("***Release all resources for future usage!");
        self.scheduler.clear_finished();
        self.scheduler.release_waitings();
        self.active_sessions.clear();
    }

    pub fn generate_stream(
        &mut self,
        params: &SamplingParams,
        prompt: String,
    ) -> Result<(usize, usize, mpsc::Receiver<StreamItem>)> {
        match self.add_request(params, &prompt, RequestType::Stream) {
            Ok((seq_id, prompt_length, rx)) => Ok((seq_id, prompt_length, rx)),
            Err(e) => {
                self.free_resources();
                candle_core::bail!("{:?}", e)
            }
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
                        crate::log_error!("\n\n[Engine Loop] Step error: {:?}", e);
                        std::process::exit(1);
                    }
                }

                tokio::task::yield_now().await;
            }
        });
    }
}
