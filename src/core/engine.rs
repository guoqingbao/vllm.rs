//src/core/engine.rs
use super::fingerprint::FingerprintManager;
use super::runner::{ModelRunner, RunnerType, Seqs};
use super::scheduler::{Scheduler, KVCACHE_SWAP_THRESHOLD};
use super::sequence::Sequence;
use crate::core::scheduler::PD_PREFILL_STATUS_CHECK_COOLING_PERIOD;
use crate::core::sequence::{DecodeSequence, SequenceStatus};
use crate::core::{GenerationOutput, SyncCollectionResult};
use crate::models::layers::distributed::Comm;
#[cfg(feature = "nccl")]
use crate::models::layers::distributed::Id;
use crate::models::layers::VarBuilderX;
use crate::runner::{
    receive_local, send_and_expect_ack, send_local, MessageType, RunnerInitRequest,
};
use crate::server::{EmbeddingStrategy, UsageResponse};
use crate::transfer::PdRole;
use crate::transfer::Transfer;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, EosTokenId, SamplingParams};
use crate::utils::guidance::load_toktrie_from_path;
use crate::utils::heartbeat::heartbeat_worker;
use crate::utils::image::{get_image_config, ImageData, ImageProcessConfig};
use crate::utils::progress::{progress_worker, ProgressReporter};
use crate::utils::progress::{spawn_progress_thread, ProgressLike};
use crate::utils::{chat_template::ChatTemplate, prepare_engine_config};
use crate::utils::{get_runner_path, init_config_tokenizer, spawn_runner, update_kvcache_config};
use crate::{log_info, log_warn};
use candle_core::{DType, Result};
use colored::Colorize;
use either::Either;
use futures::future::join_all;
use interprocess::local_socket::traits::Listener;
use interprocess::local_socket::{GenericNamespaced, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use interprocess::TryClone;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
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
    /// Tool call pause - generation paused for MCP execution
    /// Contains: (session_id for resume, prompt_start_time, decode_start_time, decoded_length)
    ToolCallPause {
        session_id: String,
        prompt_start_time: usize,
        decode_start_time: usize,
        decoded_length: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestType {
    Stream,
    Completion,
}

#[allow(dead_code)]
pub struct LLMEngine {
    pub runners: Arc<RwLock<RunnerType>>,
    pub scheduler: Scheduler,
    pub tokenizer: Tokenizer,
    econfig: EngineConfig,
    default_chat_template: String,
    template: ChatTemplate,
    stream_decoders: HashMap<usize, super::DecodeStreamType>,
    stream_senders: HashMap<usize, Sender<StreamItem>>,
    request_types: HashMap<usize, RequestType>,
    decode_start_times: HashMap<usize, usize>,
    decode_length: HashMap<usize, usize>,
    last_check_throughput_time: usize,
    active_requests: HashSet<usize>,
    active_sessions: VecDeque<(usize, String)>,
    cancelled_sequences: Vec<usize>,
    stop_flag: Arc<AtomicBool>,
    has_vision: bool,
    model_name: String,
    pub img_cfg: Option<ImageProcessConfig>,
    /// Fingerprint manager for automatic session detection
    fingerprint_manager: FingerprintManager,
}

impl LLMEngine {
    #[allow(unused_mut)]
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Arc<RwLock<Self>>> {
        let (model_pathes, is_gguf, mut config, config_tokenizer, tokenizer, mut generation_cfg) =
            init_config_tokenizer(econfig)?;
        let toktrie = load_toktrie_from_path(&model_pathes.get_tokenizer_filename()).map(Arc::new);
        if toktrie.is_none() {
            crate::log_warn!("Guided decoding disabled: tokenizer trie unavailable.");
        }

        let stop_flag = Arc::new(AtomicBool::new(false));
        let model_loaded = Arc::new(AtomicBool::new(false));
        let (mut econfig, use_runner) = prepare_engine_config(
            econfig,
            &config,
            dtype,
            &config_tokenizer,
            &mut generation_cfg,
        );
        config.fp8_kvcache = econfig.fp8_kvcache;

        // In case config file missing bos and eos configuratioin
        config.apply_generation_cfg(generation_cfg.as_ref());
        if config.eos_token_id.is_none() {
            if let Some(eos) = &config_tokenizer.eos_token {
                if let Some(token) = tokenizer.get_vocab(true).get(eos).copied() {
                    config.eos_token_id = Some(EosTokenId::Single(token));
                };
            }
        }
        assert!(
            config.architectures.is_some() && config.architectures.as_ref().unwrap().len() == 1,
            "Only one architecture is supported at the moment!"
        );
        let arch = config.architectures.as_ref().unwrap()[0].clone();

        let (model_type, default_chat_template, is_rope_i) =
            crate::utils::get_arch_rope(&tokenizer, arch.clone())?;
        log_info!("Use ROPE interleaved {is_rope_i}");

        let is_pd_server = if let Some(p_cfg) = &econfig.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        };

        log_info!("{:?}", econfig);

        log_info!("{:?}\n", config_tokenizer);

        log_info!("{:?}\n", config);

        let device_ids = if let Some(ids) = &econfig.device_ids {
            ids.clone()
        } else {
            vec![0]
        };

        let is_gemma = arch == "Gemma3ForConditionalGeneration".to_string()
            || arch == "Gemma3ForCausalLM".to_string();
        // Gemma3 must use conventional attention
        if is_gemma {
            econfig.disable_flash_attn = Some(true);
        }
        let runners = if !use_runner {
            let device = crate::utils::new_device(device_ids[0])?;
            log_info!("Loading model...");
            let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                Arc::new(RwLock::new(Box::new(ProgressReporter::new(0))));
            let handle = progress_worker(1, config.num_hidden_layers, &reporter);
            let vb = VarBuilderX::new(&model_pathes, is_gguf, dtype, &device)?;
            let transfer = if let Some(p_cfg) = &econfig.pd_config {
                Some(Arc::new(Transfer::new(
                    p_cfg.clone(),
                    0,
                    model_loaded.clone(),
                    stop_flag.clone(),
                )?))
            } else {
                None
            };

            let mut model_runner = ModelRunner::new(
                model_type.clone(),
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
                &mut econfig,
                &config,
                dtype,
                is_rope_i,
                device.clone(),
                reporter,
                transfer,
                toktrie.clone(),
                None,
            )?;

            if !is_pd_server {
                //No graph capture for PD server
                #[cfg(all(feature = "cuda", feature = "graph"))]
                if crate::utils::is_no_cuda_graph_supprt(arch.clone()) {
                    log_info!("{arch} does not supprt CUDA graph");
                } else {
                    match model_runner.warmup_capture() {
                        Ok(_) => {
                            log_info!("Cuda graph captured for performance enhancement!")
                        }
                        Err(e) => crate::log_error!("Unable to capture cuda graph {:?}!", e),
                    }
                }
            }

            model_loaded.store(true, Ordering::SeqCst);
            let _ = handle.join();
            RunnerType::Thread(model_runner)
        } else {
            log_info!("Loading model with runner(s)...");

            #[cfg(feature = "nccl")]
            let nccl_id = Id::new().unwrap();

            let runner_path = get_runner_path()?;

            let uuid_str = uuid::Uuid::new_v4();
            let uuid_str = uuid_str.to_string();
            let unique_id = if let Some(l_uuid) = uuid_str.split('-').last() {
                l_uuid
            } else {
                ""
            };

            #[cfg(feature = "python")]
            pyo3::Python::with_gil(|py| {
                for (rank, _) in device_ids.iter().enumerate() {
                    let sock_name = format!("{}@vllm-rs-runner-{}", unique_id, rank);
                    spawn_runner(py, &runner_path.display().to_string(), &sock_name, unique_id)
                        .expect("Failed to spawn runner. \n\r*****Tips: runner is not built within this package, use 'build.sh' script to build package with runner!");
                }
            });

            #[cfg(not(feature = "python"))]
            for (rank, _) in device_ids.iter().enumerate() {
                let sock_name = format!("{}@vllm-rs-runner-{}", unique_id, rank);
                spawn_runner(&runner_path.display().to_string(), &sock_name, unique_id)
                    .expect("Failed to spawn runner. \n\r*****Tips: runner is not built, use 'run.sh' script instead of 'cargo run'!");
            }

            let progress_sock_name = format!("{}@vllm-rs-progress", unique_id);
            let progress_handle = spawn_progress_thread(
                econfig.num_shards.unwrap_or(1),
                config.num_hidden_layers,
                progress_sock_name,
            );
            heartbeat_worker(Some(device_ids.len()), false, stop_flag.clone(), unique_id);
            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::ParallelIterator;
            let engine_config = std::sync::OnceLock::<EngineConfig>::new();
            let runner_streams: Result<Vec<LocalStream>> = device_ids
                .par_iter()
                .enumerate()
                .map(|(rank, dev_id)| {
                    let model_type = model_type.clone();
                    let sock_name = format!("{}@vllm-rs-runner-{}", unique_id, rank);
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
                        num_shards: econfig.num_shards.unwrap_or(1),
                        model_type,
                        config: config.clone(),
                        econfig: econfig.clone(),
                        model_pathes: model_pathes.clone(),
                        is_gguf,
                        dtype: dtype.into(),
                        is_rope_i,
                        #[cfg(feature = "nccl")]
                        nccl_id: crate::runner::NcclId(nccl_id.clone()),
                    });

                    send_and_expect_ack(&mut stream, &init_msg, "initialize", rank)?;

                    if econfig.max_model_len.is_none() {
                        let ecfg = engine_config.get_or_init(|| {
                            let mut econfig = econfig.clone();
                            update_kvcache_config(&mut econfig, &config.clone(), dtype);
                            econfig
                        });

                        send_and_expect_ack(
                            &mut stream,
                            &MessageType::UsableMemoryLeft(ecfg.clone()),
                            "init kvcache",
                            rank,
                        )?;
                    }

                    crate::log_info!("Runner {} started!", rank);
                    Ok(stream)
                })
                .collect();

            if let Ok(Some(handle)) = progress_handle.join() {
                let _ = handle.join();
            }
            if let Some(cfg) = engine_config.get() {
                econfig = cfg.clone();
            } else if econfig.max_model_len.is_none() {
                candle_core::bail!("Unable to update EngineConfig!");
            }
            RunnerType::Process(runner_streams?)
        };

        if econfig.max_model_len.is_none() {
            println!(
                "\n{} is not given, default to {}, Max usable kvcache tokens {}.\n",
                format!("Warn: max_model_len").yellow().bold(),
                format!("{}", 32768).red().bold(),
                format!("{}", 32768 * econfig.max_num_seqs).red().bold(),
            );
            econfig.max_model_len = Some(32768);
        }
        let runners = Arc::new(RwLock::new(runners));
        let mut scheduler = Scheduler::new(runners.clone(), &econfig, &config);

        // Initialize tool call end tokens for detection
        // Tokenize "</tool_call>" to get the token IDs for tool call end detection
        if let Ok(tokens) = tokenizer.encode("</tool_call>", false) {
            let tool_call_end_ids: Vec<u32> = tokens.get_ids().to_vec();
            // We want to detect when the LAST token of "</tool_call>" is generated
            // So we just need the last token ID
            if let Some(&last_token) = tool_call_end_ids.last() {
                scheduler.set_tool_call_end_tokens(vec![last_token]);
                log_info!("Tool call end token ID set to: {}", last_token);
            }
        }

        // Set tokenizer for JSON tool call detection (for models like Qwen3 that output raw JSON)
        scheduler.set_tokenizer(Arc::new(tokenizer.clone()));

        log_warn!(
            "Maximum batched tokens {} ({} blocks x Block_Size {} for KV cache). Additional CPU KV Cache blocks {}.",
            econfig.max_num_batched_tokens,
            econfig.num_blocks,
            econfig.block_size,
            (econfig.num_blocks as f32 * econfig.cpu_mem_fold.unwrap_or(0.5f32)) as usize
        );
        log_warn!("Model loaded.\n");

        let template = ChatTemplate::new(
            None,
            config_tokenizer.chat_template.clone(),
            config_tokenizer.bos_token.clone(),
            config_tokenizer.eos_token.clone(),
            None,
            true,
            true,
        );

        let img_cfg = get_image_config(model_type, &config)?;

        let model_name = if let Some(archs) = &config.architectures {
            archs[0].to_string()
        } else {
            "default".to_string()
        };

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
            decode_length: HashMap::new(),
            last_check_throughput_time: 0,
            active_requests: HashSet::new(),
            active_sessions: VecDeque::new(),
            cancelled_sequences: Vec::new(),
            stop_flag: stop_flag.clone(),
            has_vision: config.is_multi_model.unwrap_or(false),
            img_cfg,
            model_name,
            fingerprint_manager: FingerprintManager::new(),
        }));
        Self::start_engine(engine.clone());
        Ok(engine)
    }

    fn add_request_(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        request_type: &RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
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

        let tokens_required = length.div_ceil(self.econfig.block_size) * self.econfig.block_size;
        let available_tokens = self.scheduler.get_available_kv_tokens();

        if tokens_required >= available_tokens {
            // Release cache for unactive sessions
            let _ = self.try_release_cache(tokens_required);
            if tokens_required > self.scheduler.get_available_kv_tokens() {
                candle_core::bail!(
                    "Remaining {} kvcache tokens, but your prompt length requires {}, please request later!",
                    self.scheduler.get_available_kv_tokens(),
                    tokens_required
                );
            }
        }

        if let Some(gen_cfg) = &self.econfig.generation_cfg {
            let temperature = params.temperature.or(gen_cfg.temperature);
            let top_k = params.top_k.or(gen_cfg.top_k);
            let top_p = params.top_p.or(gen_cfg.top_p);
            let frequency_penalty = params.frequency_penalty.or(gen_cfg.frequency_penalty);
            let presence_penalty = params.presence_penalty.or(gen_cfg.presence_penalty);
            params.temperature = temperature;
            params.top_k = top_k;
            params.top_p = top_p;
            params.frequency_penalty = frequency_penalty;
            params.presence_penalty = presence_penalty;
        }
        let session_id = params.session_id.clone();
        let session_id = if session_id.is_some() && !self.econfig.flash_context.unwrap_or(false) {
            None
        } else {
            session_id.clone()
        };

        let seq_id = if let Some(session_id) = session_id {
            if self.econfig.server_mode.unwrap_or(true) {
                crate::log_warn!(
                    "Cached {} sessions: {:?} ({} tokens cached).",
                    self.active_sessions.len(),
                    self.active_sessions,
                    self.get_num_cached_tokens(),
                );
            }

            // Try to reuse cached sequence if available
            let seq_id = if self.scheduler.has_cache(&session_id) {
                match self.scheduler.get_cache(
                    &session_id,
                    token_ids.clone(),
                    &mut self.active_sessions,
                    images,
                    image_idx,
                ) {
                    Ok(id) => Some(id),
                    Err(e) => {
                        crate::log_warn!("Get cache error: {:?}", e);
                        None
                    }
                }
            } else {
                None
            };

            let seq_id = seq_id.unwrap_or_else(|| {
                // Cache miss: create new sequence
                let seq = Sequence::new(
                    token_ids,
                    self.econfig.block_size,
                    params,
                    images,
                    image_idx,
                );
                let id = self.scheduler.add(seq);

                // Update active_sessions queue
                if let Some(pos) = self
                    .active_sessions
                    .iter()
                    .position(|(_, s)| s == &session_id)
                {
                    self.active_sessions.remove(pos);
                }
                self.active_sessions.push_back((id, session_id.clone()));
                id
            });

            seq_id
        } else {
            let seq = Sequence::new(
                token_ids,
                self.econfig.block_size,
                params,
                images,
                image_idx,
            );
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
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize, Receiver<StreamItem>)> {
        let (seq_id, prompt_length) =
            self.add_request_(params, prompt, &request_type, images, image_idx)?;
        let (tx, rx) = channel(4096);
        self.stream_senders.insert(seq_id, tx);
        self.request_types.insert(seq_id, request_type.clone());
        if self.econfig.server_mode.unwrap_or(true) && request_type != RequestType::Completion {
            log_warn!(
                "[{:?}] New request [Seq_id {}, {} tokens] received! (session_id: {:?})\n",
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

    pub fn get_available_kv_tokens(&self) -> usize {
        self.scheduler.get_available_kv_tokens()
    }

    pub fn notify_runner_finished(&mut self, id: usize) -> Result<()> {
        match &mut *self.runners.write() {
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

    pub fn step(&mut self) -> Result<usize> {
        pub struct DecodedIds(Either<Vec<usize>, Vec<usize>>);

        // Get scheduled sequence indexes and prefill flag
        let (scheduled_ids, is_prefill) = match self.scheduler.schedule(&mut self.active_sessions) {
            Ok((ids, prefill)) => (ids, prefill),
            Err(_) => (vec![], true),
        };
        let decoded_ids = if !scheduled_ids.is_empty() {
            // Get immutable references to scheduled sequences for model_runner
            let seqs = self.scheduler.get_sequences(&scheduled_ids);

            let output_ids = match &mut *self.runners.write() {
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

                    let all_outputs: Result<Vec<Vec<u32>>> = cloned_streams
                        .into_par_iter()
                        .map(|mut stream| {
                            let msg = request.clone();
                            send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                            let response = receive_local(&mut stream, false)?;

                            match response {
                                MessageType::RunResponse(output_ids) => {
                                    if output_ids.len() == 0 {
                                        candle_core::bail!("Runner step error, no response!")
                                    } else {
                                        Ok(output_ids)
                                    }
                                }
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
                    return Ok(0);
                } else {
                    let output_ids: Vec<u32> = indices.iter().map(|&i| output_ids[i]).collect();
                    self.scheduler.postprocess(
                        &finished_indices,
                        &output_ids,
                        &mut self.active_sessions,
                    );
                    DecodedIds(Either::Left(finished_indices))
                }
            } else {
                self.scheduler
                    .postprocess(&scheduled_ids, &output_ids, &mut self.active_sessions);
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
                    // Check if this is a tool call pause (not a real finish)
                    if let Some(session_id) = &s.tool_call_session_id {
                        if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                            if let Some(request_type) = self.request_types.get(&seq_id) {
                                let prompt_start_time = s.created_time();
                                let decode_start_time = self
                                    .decode_start_times
                                    .get(&seq_id)
                                    .copied()
                                    .unwrap_or(prompt_start_time);

                                crate::log_info!(
                                    "[Seq {}] Sending ToolCallPause with session_id: {} (request_type: {:?})",
                                    seq_id,
                                    session_id,
                                    request_type
                                );

                                // For stream requests, send the last token before pause
                                if *request_type == RequestType::Stream {
                                    if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                                        if let Some(tok) = decoder.step(s.last_token) {
                                            let _ = sender.try_send(StreamItem::Token(tok.clone()));
                                        }
                                    }
                                } else {
                                    let _ = sender.try_send(StreamItem::TokenID(s.last_token));
                                }

                                // Emit ToolCallPause for both stream and completion requests
                                let _ = sender.try_send(StreamItem::ToolCallPause {
                                    session_id: session_id.clone(),
                                    prompt_start_time,
                                    decode_start_time,
                                    decoded_length: s.output_ids.len(),
                                });
                            }
                        }
                        // Don't remove senders/decoders - they will be reused for follow-up
                        // The sequence will be resumed via get_cache with the session_id
                        if self.econfig.server_mode.unwrap_or(true) {
                            self.scheduler.print_free_blocks();
                        }
                        continue; // Skip normal finish handling
                    }

                    // Normal finish handling - tool call detection is now done in scheduler

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

                    // Extend fingerprint for auto-sessions when sequence completes
                    // This allows subsequent requests to match the conversation
                    if let Some((_, session_id)) =
                        self.active_sessions.iter().find(|(id, _)| *id == seq_id)
                    {
                        if session_id.starts_with("auto_session_") {
                            // Decode the output to get the assistant response text
                            if let Ok(decoded_text) = self.tokenizer.decode(&s.output_ids, true) {
                                self.fingerprint_manager
                                    .extend_session_fingerprint(&session_id.clone(), &decoded_text);
                            }
                        }
                    }

                    self.stream_senders.remove(&seq_id);
                    self.stream_decoders.remove(&seq_id);
                    if let Some(_) = self.active_requests.get(&seq_id) {
                        self.active_requests.remove(&seq_id);
                    }
                    self.decode_start_times.remove(&seq_id);
                    self.decode_length.remove(&seq_id);
                    let _ = self.notify_runner_finished(seq_id);
                    if self.econfig.server_mode.unwrap_or(true) {
                        self.scheduler.print_free_blocks();
                    }
                } else {
                    if !self.decode_start_times.contains_key(&seq_id) {
                        let cur_time = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis() as usize;
                        self.decode_start_times.insert(seq_id, cur_time);
                        self.decode_length.insert(seq_id, 1);

                        let time_costs = cur_time - s.created_time();
                        if time_costs / 100 > 0 && s.len() > 0 {
                            crate::log_info!(
                                "Prefilling [seq_id {}]: {} tokens in {:.2}s ({:.2} tokens/s{})",
                                seq_id,
                                s.len(),
                                time_costs as f32 / 1000f32,
                                s.len() as f32 / (time_costs as f32 * 1.0f32 / 1000f32),
                                if let Some(_) =
                                    self.active_sessions.iter().position(|(id, _)| *id == s.id)
                                {
                                    ", cache included"
                                } else {
                                    ""
                                },
                            )
                        }
                    }

                    if let Some(length) = self.decode_length.get_mut(&seq_id) {
                        *length = s.output_len();
                    }

                    let token_ids =
                        if self.is_pd_mode() && s.pd_first_token.is_some() && s.output_len() == 2 {
                            // Special case, the real first token is generated on PD server
                            vec![s.pd_first_token.unwrap_or(s.last_token), s.last_token]
                        } else {
                            vec![s.last_token]
                        };

                    if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                        if let Some(request_type) = self.request_types.get(&seq_id) {
                            if *request_type == RequestType::Stream {
                                if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                                    for token_id in token_ids {
                                        if let Some(tok) = decoder.step(token_id) {
                                            let result =
                                                sender.try_send(StreamItem::Token(tok.clone()));
                                            if result.is_err() {
                                                crate::log_error!(
                                                    "Error when sending token to client [seq_id {}]",
                                                    seq_id
                                                );
                                                self.cancelled_sequences.push(seq_id);
                                            }
                                        }
                                    }
                                }
                            } else {
                                //completion request will be decoded at the final stage (at once)
                                for token_id in token_ids {
                                    let _ = sender.try_send(StreamItem::TokenID(token_id));
                                }
                            }
                        }
                    }
                }
            }
        }
        self.scheduler.clear_finished();

        if indices.is_empty()
            && self.scheduler.kv_cache_usage_percent() > KVCACHE_SWAP_THRESHOLD + 0.01f32
        {
            if !self.try_release_cache(0) {
                if let Some(oldest_seq_id) = self.active_requests.clone().iter().min() {
                    crate::log_error!(
                        "Unable to schedule task(s), drop the oldest active request (seq_id: {:?})",
                        oldest_seq_id
                    );
                    self.cancelled_sequences.push(*oldest_seq_id);
                    self.check_canceled(Some(
                        "Unable to schedule task(s), this request has been dropped!".to_string(),
                    ));
                }
                if self.scheduler.kv_cache_usage_percent() > 0.99f32 {
                    self.free_resources();
                }
            }
        } else {
            self.check_canceled(None);
        }
        if self.econfig.server_mode.unwrap_or(true) && is_running {
            self.may_print_decoding_throughput(&indices);
        }
        Ok(indices.len())
    }

    pub fn try_release_cache(&mut self, tokens_required: usize) -> bool {
        let initial_available_tokens = self.get_available_kv_tokens();
        let mut active_session_reqeusts = VecDeque::new();
        while let Some((seq_id, session_id)) = self.active_sessions.pop_front() {
            if self.active_requests.contains(&seq_id) {
                active_session_reqeusts.push_back((seq_id, session_id.clone()));
                continue;
            };
            // Release GPU cached, not swapped out
            if self.scheduler.get_cached_status(&session_id) == SequenceStatus::Cached {
                if !self.scheduler.try_swap_out_by_id(seq_id, false) {
                    self.scheduler.release_cache(seq_id);
                    // Also remove fingerprint for auto-sessions
                    if session_id.starts_with("auto_session_") {
                        self.fingerprint_manager.remove_session(&session_id);
                    }
                    if self.econfig.server_mode.unwrap_or(true) {
                        crate::log_warn!(
                            "🗑️ Seq {} - cache removed (session id {})!\n",
                            seq_id,
                            session_id
                        );
                    }
                } else {
                    // Swapped out but still in cached sessions
                    crate::log_warn!(
                        "Swapped out cache for Seq {} (session_id {}).",
                        seq_id,
                        session_id,
                    );
                    active_session_reqeusts.push_back((seq_id, session_id.clone()));
                }
            }
            if self.scheduler.get_available_kv_tokens() > tokens_required {
                break;
            }
        }
        // Push back active session requests
        self.active_sessions.extend(active_session_reqeusts);

        initial_available_tokens - self.get_available_kv_tokens() > 0
    }

    pub fn check_canceled(&mut self, reason: Option<String>) {
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
            if let Some(r) = &reason {
                if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                    if let Some(request_type) = self.request_types.get(&seq_id) {
                        if *request_type == RequestType::Stream {
                            let _ = sender.try_send(StreamItem::Error(r.clone()));
                        }
                    }
                }
            }
            self.stream_senders.remove(&seq_id);
        }
        self.scheduler.clear_finished();
        self.cancelled_sequences.clear();
    }

    pub fn may_print_decoding_throughput(&mut self, active_indices: &Vec<usize>) {
        if active_indices.is_empty() || self.active_requests.is_empty() {
            return;
        }
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        if cur_time - self.last_check_throughput_time < 5000 {
            return;
        }
        self.last_check_throughput_time = cur_time;

        let mut total_decoded_length = 0;
        let mut total_decoded_time_costs = 0;

        for &idx in active_indices {
            if let Some(seq) = self.scheduler.get_running(idx) {
                if let Some(length) = self.decode_length.get(&seq.id) {
                    total_decoded_length += length;
                }
                if let Some(start_time) = self.decode_start_times.get(&seq.id) {
                    total_decoded_time_costs += cur_time - start_time;
                }
            }
        }

        if total_decoded_length > 0 && total_decoded_time_costs / 1000 > 0 {
            let avg_throughput = total_decoded_length / (total_decoded_time_costs / 1000);
            crate::log_info!(
                "Decoding: {} active request(s), avg. {} tokens/s per request (total: {} tokens/s)",
                active_indices.len(),
                avg_throughput,
                avg_throughput * active_indices.len()
            )
        }

        if total_decoded_length % 100 > 50 {
            self.scheduler.print_free_blocks();
        }
    }

    pub fn cancel(&mut self, seq_id: usize) {
        self.cancelled_sequences.push(seq_id);
    }

    pub fn cancel_all_with_reason(&mut self, reason: Option<String>) -> bool {
        for seq_id in &self.active_requests {
            self.cancelled_sequences.push(*seq_id);
        }
        let has_requests_to_cancel = self.cancelled_sequences.len() > 0;
        self.check_canceled(reason);
        has_requests_to_cancel
    }

    fn apply_chat_template(
        &mut self,
        params: &SamplingParams,
        messages: &Vec<Message>,
        log: bool,
    ) -> (String, i32) {
        // let mut collected_images = Vec::new();
        let mut prompt_template = self.template.clone();

        // Apply user's thinking preference - default to false if not specified
        let enable_thinking = params.thinking.unwrap_or(false);
        prompt_template.set_enable_thinking(enable_thinking);

        let mut context_cache_reqeust = false;

        let session_id =
            if params.session_id.is_some() && !self.econfig.flash_context.unwrap_or(false) {
                crate::log_warn!("`session_id` detected but `context-cache` is not enabled!");
                None
            } else {
                params.session_id.clone()
            };

        if let Some(session_id) = &session_id {
            if self.scheduler.has_cache(&session_id) {
                //context cache, only retrieve the last message
                context_cache_reqeust = true;
                prompt_template.set_messages(&vec![messages[messages.len() - 1].clone()]);
            }
        }

        let image_idx: i32 = if context_cache_reqeust {
            if messages[messages.len() - 1].num_images > 0 {
                // Find the image index of the current context request
                let mut idx = 0;
                for i in 0..messages.len() - 1 {
                    idx += messages[i].num_images;
                }
                idx as i32
            } else {
                -1 // No image for the continued request
            }
        } else {
            prompt_template.set_messages(messages);
            0 // Start from beginning (no context-cache)
        };
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
        (prompt, image_idx)
    }

    pub fn is_idle(&self) -> bool {
        self.active_requests.is_empty()
    }

    pub fn is_pd_mode(&self) -> bool {
        self.econfig.pd_config.is_some()
    }

    pub fn is_pd_server(&self) -> bool {
        if let Some(p_cfg) = &self.econfig.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        }
    }

    pub fn generate_sync(
        &mut self,
        params: &Vec<SamplingParams>,
        message_list: &Vec<Vec<Message>>,
        images: Option<ImageData>,
    ) -> Result<Vec<(usize, usize, mpsc::Receiver<StreamItem>)>> {
        if params.len() != message_list.len() {
            candle_core::bail!("size of sampling parameters is not match with size of prompts!");
        }
        let mut receivers = Vec::new();
        for (param, messages) in params.iter().zip(message_list.iter()) {
            // Fingerprint-based automatic session detection
            // When context-cache is enabled and no session_id provided, auto-detect via fingerprint
            let mut param = param.clone();
            let has_session = param.session_id.is_some();
            let flash_enabled = self.econfig.flash_context.unwrap_or(false);

            if !has_session && flash_enabled {
                // Build messages as (role, content) tuples for fingerprint computation
                let msg_tuples: Vec<(String, String)> = messages
                    .iter()
                    .map(|m| (m.role.clone(), m.content.clone()))
                    .collect();

                let fingerprint = FingerprintManager::compute_fingerprint(&msg_tuples);
                let session_id = self
                    .fingerprint_manager
                    .find_or_create_session(fingerprint, &msg_tuples);
                param.session_id = Some(session_id);
            }

            let (prompt, image_idx) = self.apply_chat_template(&param, messages, false);
            if let Ok((seq_id, prompt_length, rx)) =
                self.add_request(&param, &prompt, RequestType::Completion, &images, image_idx)
            {
                receivers.push((seq_id, prompt_length, rx));
            }
        }

        Ok(receivers)
    }

    pub async fn collect_sync_results(
        receivers: Vec<(usize, usize, mpsc::Receiver<StreamItem>)>,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Vec<SyncCollectionResult>> {
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

                    let s = format!(
                        "[Live Throughput] {} tokens in {:.2}s ({:.2} tokens/s)",
                        count,
                        elapsed,
                        count as f32 / elapsed
                    );
                    eprintln!("{}", String::from(s).yellow());
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
                    let mut output: Option<SyncCollectionResult> = None;
                    let mut collected_token_ids: Vec<u32> = Vec::new();

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

                                output = Some(SyncCollectionResult::Completed(GenerationOutput {
                                    seq_id,
                                    prompt_length,
                                    prompt_start_time: prompt_start,
                                    decode_start_time: decode_start,
                                    decode_finish_time: decode_finish,
                                    decoded_length: decoded_len,
                                    decode_output,
                                }));
                                break;
                            }
                            StreamItem::Token(_) => {
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
                                    .ok();
                            }
                            StreamItem::TokenID(id) => {
                                decoded_tokens.fetch_add(1, Ordering::Relaxed);
                                collected_token_ids.push(id);

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
                                    .ok();
                            }
                            StreamItem::Done(_) => {
                                crate::log_info!("Sequence [seq_id {}] finished!", seq_id);
                                break;
                            }
                            StreamItem::ToolCallPause {
                                session_id,
                                prompt_start_time,
                                decode_start_time,
                                decoded_length,
                            } => {
                                // Return tool call pause info for caller to handle
                                crate::log_info!(
                                    "Sequence [seq_id {}] tool call pause with session_id: {}",
                                    seq_id,
                                    session_id
                                );
                                output = Some(SyncCollectionResult::ToolCallPause {
                                    session_id,
                                    seq_id,
                                    prompt_length,
                                    output_ids: collected_token_ids.clone(),
                                    prompt_start_time,
                                    decode_start_time,
                                    decoded_length,
                                });
                                break;
                            }
                            StreamItem::Error(e) => {
                                crate::log_error!("Error in Seq {}: {}", seq_id, e);
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
        // Reset fingerprint manager when all resources are freed
        self.fingerprint_manager = FingerprintManager::new();
    }

    pub fn generate_stream(
        &mut self,
        params: &SamplingParams,
        messages: &Vec<Message>,
        images: Option<ImageData>, //collection of images of the full conversation
    ) -> Result<(usize, usize, mpsc::Receiver<StreamItem>)> {
        // Fingerprint-based automatic session detection
        // When context-cache is enabled and no session_id provided, auto-detect via fingerprint
        let mut params = params.clone();
        let has_session = params.session_id.is_some();
        let flash_enabled = self.econfig.flash_context.unwrap_or(false);

        if !has_session && flash_enabled {
            // Build messages as (role, content) tuples for fingerprint computation
            let msg_tuples: Vec<(String, String)> = messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();

            let fingerprint = FingerprintManager::compute_fingerprint(&msg_tuples);
            let session_id = self
                .fingerprint_manager
                .find_or_create_session(fingerprint, &msg_tuples);
            params.session_id = Some(session_id);
        }

        let (prompt, image_idx) = self.apply_chat_template(&params, messages, false);
        match self.add_request(&params, &prompt, RequestType::Stream, &images, image_idx) {
            Ok((seq_id, prompt_length, rx)) => Ok((seq_id, prompt_length, rx)),
            Err(e) => {
                candle_core::bail!("{:?}", e)
            }
        }
    }

    pub fn get_usage_stats(&self, session_id: Option<String>) -> Result<UsageResponse> {
        match session_id {
            Some(sid) => {
                let available_kvcache_tokens = self.scheduler.get_available_kv_tokens();
                let max_model_len = self
                    .econfig
                    .max_model_len
                    .ok_or_else(|| candle_core::Error::msg("max_model_len not set!"))?;
                let token_used = if let Some((seq_id, _)) =
                    self.active_sessions.iter().find(|(_, v)| *v == sid)
                {
                    self.scheduler.get_seq_token_usage(*seq_id)?
                } else {
                    candle_core::bail!("Seq with session_id {} not found", sid)
                };

                let total_kv_cache_tokens = self.scheduler.get_total_kv_tokens();
                let (swap_used, total_swap_memory) = self.scheduler.get_cpu_swap_usage();

                let mut session_status = "Waiting".to_string();
                if self.scheduler.has_cache(&sid) {
                    let status = self.scheduler.get_cached_status(&sid).to_string();
                    if status == "FinishSwapped" {
                        session_status = "Swapped".to_string();
                    } else {
                        session_status = status.clone();
                    }
                } else if let Some((seq_id, _)) =
                    self.active_sessions.iter().find(|(_, v)| v == &sid)
                {
                    if self.active_requests.contains(&seq_id) {
                        session_status = "Running".to_string();
                    }
                } else {
                    session_status = "Finished".to_string()
                }

                Ok(UsageResponse {
                    token_used,
                    max_model_len,
                    used_kvcache_tokens: total_kv_cache_tokens - available_kvcache_tokens,
                    total_kv_cache_tokens,
                    swap_used,
                    total_swap_memory,
                    session_status,
                })
            }
            _ => {
                candle_core::bail!("No session_id provided")
            }
        }
    }

    pub fn embed(
        &mut self,
        inputs: &[String],
        strategy: EmbeddingStrategy,
    ) -> Result<(Vec<Vec<f32>>, usize)> {
        let mut outputs = Vec::new();
        let mut prompt_tokens = 0;

        for input in inputs {
            let tokens = self
                .tokenizer
                .encode_fast(input.as_str(), true)
                .map_err(candle_core::Error::wrap)?;
            let token_ids: Vec<u32> = tokens.get_ids().iter().copied().collect();
            if token_ids.is_empty() {
                candle_core::bail!("Embedding input cannot be empty");
            }

            if let Some(max_model_len) = self.econfig.max_model_len {
                if token_ids.len() > max_model_len - 1 {
                    candle_core::bail!(
                        "Embedding input length {} exceeds max_model_len {}",
                        token_ids.len(),
                        max_model_len
                    );
                }
            }

            let tokens_required =
                token_ids.len().div_ceil(self.econfig.block_size) * self.econfig.block_size;
            if tokens_required > self.scheduler.get_available_kv_tokens() {
                candle_core::bail!(
                    "Remaining {} kvcache tokens, but embedding requires {}, please retry later",
                    self.scheduler.get_available_kv_tokens(),
                    tokens_required
                );
            }

            let mut seq = Sequence::new(
                token_ids.clone(),
                self.econfig.block_size,
                SamplingParams::default(),
                &None,
                -1,
            );
            self.scheduler.block_manager.allocate(&mut seq)?;

            let mut chunked_mean: Option<Vec<f32>> = None;
            let mut last_vec: Option<Vec<f32>> = None;
            let mut processed_tokens = 0usize;
            let chunk_size =
                if self.econfig.flash_context.unwrap_or(false) && cfg!(feature = "flash-context") {
                    4096
                } else {
                    8192
                };

            while seq.num_cached_tokens < seq.len() {
                let remaining = seq.len() - seq.num_cached_tokens;
                let chunk_tokens = std::cmp::min(chunk_size, remaining);
                let embedding_result = match &mut *self.runners.write() {
                    RunnerType::Thread(model_runner) => model_runner.embed(&[&seq], &strategy),
                    RunnerType::Process(ref mut runner_streams) => {
                        let request = MessageType::RunEmbed((vec![seq.clone()], strategy.clone()));
                        let cloned_streams: Vec<LocalStream> = runner_streams
                            .iter_mut()
                            .map(|s| s.try_clone().expect("clone failed"))
                            .collect();

                        let all_outputs: Result<Vec<Vec<Vec<f32>>>> = cloned_streams
                            .into_par_iter()
                            .map(|mut stream| {
                                let msg = request.clone();
                                send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                                let response = receive_local(&mut stream, false)?;

                                match response {
                                    MessageType::RunResponseEmbed(output_embed) => {
                                        if output_embed.len() == 0 {
                                            candle_core::bail!("Runner step error, no response!")
                                        } else {
                                            Ok(output_embed)
                                        }
                                    }
                                    other => {
                                        candle_core::bail!("Unexpected response type: {:?}", other)
                                    }
                                }
                            })
                            .collect();
                        let all_outputs = all_outputs.map_err(candle_core::Error::wrap)?;
                        if let Some(output_embed) = all_outputs.first() {
                            Ok(output_embed.clone())
                        } else {
                            candle_core::bail!(
                                "No output embedding states received from model runners"
                            );
                        }
                    }
                };
                let embedding_vecs = match embedding_result {
                    Ok(v) => v,
                    Err(e) => {
                        self.scheduler.block_manager.deallocate(&seq);
                        return Err(e);
                    }
                };
                if let Some(vec) = embedding_vecs.into_iter().next() {
                    match strategy {
                        EmbeddingStrategy::Mean => {
                            if chunked_mean.is_none() {
                                chunked_mean = Some(vec![0.0; vec.len()]);
                            }
                            if let Some(sum) = chunked_mean.as_mut() {
                                for (dst, val) in sum.iter_mut().zip(vec.iter()) {
                                    *dst += val * chunk_tokens as f32;
                                }
                            }
                        }
                        EmbeddingStrategy::Last => {
                            last_vec = Some(vec);
                        }
                    }
                }

                seq.num_cached_tokens += chunk_tokens;
                processed_tokens += chunk_tokens;
                if seq.len() > chunk_size {
                    if chunk_tokens < chunk_size {
                        crate::log_info!(
                            "Embedding chunk prefilled finished {}/{} (Seq {})",
                            seq.num_cached_tokens,
                            seq.len(),
                            seq.id
                        );
                    } else {
                        crate::log_info!(
                            "Embedding chunk prefilled {}/{} (Seq {})",
                            seq.num_cached_tokens,
                            seq.len(),
                            seq.id
                        );
                    }
                }
            }

            self.scheduler.block_manager.deallocate(&seq);
            prompt_tokens += seq.len();

            let final_vec = match strategy {
                EmbeddingStrategy::Mean => {
                    let mut sum = chunked_mean.unwrap_or_default();
                    if processed_tokens > 0 {
                        for v in sum.iter_mut() {
                            *v /= processed_tokens as f32;
                        }
                    }
                    sum
                }
                EmbeddingStrategy::Last => last_vec.unwrap_or_default(),
            };
            outputs.push(final_vec);
        }

        Ok((outputs, prompt_tokens))
    }

    pub fn start_engine(engine: Arc<RwLock<Self>>) {
        GLOBAL_RT.spawn(async move {
            let engine = engine.clone();
            let is_pd_server = {
                let guard = engine.read();
                guard.is_pd_mode() && guard.is_pd_server()
            };
            loop {
                let idle = {
                    let guard = engine.read();
                    //no add_request in PD server, marking it always active
                    guard.is_idle() && !is_pd_server
                };

                if idle {
                    tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
                    continue;
                }

                let mut task_processed = 0;
                {
                    let mut guard = engine.write();
                    match guard.step() {
                        Ok(n_tasks) => {
                            task_processed = n_tasks;
                        }
                        Err(e) => {
                            crate::log_error!("[Engine Loop] Step error: {:?}", e);
                            if !guard.cancel_all_with_reason(Some(e.to_string())) {
                                std::process::exit(1);
                            }
                        }
                    }
                }
                if task_processed == 0 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(if is_pd_server {
                        PD_PREFILL_STATUS_CHECK_COOLING_PERIOD as u64
                    } else {
                        1
                    }))
                    .await;
                }
                tokio::task::yield_now().await;
            }
        });
    }

    pub fn get_model_info(&self) -> (bool, String) {
        (self.has_vision, self.model_name.clone())
    }

    /// Get a clone of the chat template for external use (e.g., tokenization without generation)
    pub fn get_chat_template(&self) -> ChatTemplate {
        self.template.clone()
    }
}
