// src/core/runner.rs
use crate::models::layers::distributed::Comm;
use crate::models::layers::VarBuilderX;
use crate::transfer::Transfer;
use crate::utils::config::SamplingParams;
#[cfg(all(feature = "cuda", feature = "graph"))]
use crate::utils::graph::{CudaGraphFn, CudaGraphWrapper, GraphCapturer, ModelFn};
use crate::utils::logits_processor::LogitsProcessor;
use crate::utils::progress::ProgressLike;
use crate::{
    core::sequence::{DecodeSequence, Sequence, ToDecodeInput},
    models::glm4::GLM4ForCausalLM,
    models::llama::LLaMaForCausalLM,
    models::qwen3::Qwen3ForCausalLM,
    models::qwen3_moe::Qwen3MoEForCausalLM,
    utils::config::{Config, EngineConfig, ModelType},
};
use attention_rs::cache;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use interprocess::local_socket::Stream as LocalStream;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
const MAX_PARALLEL_SAMPLING: usize = 32;

pub enum Seqs<'a> {
    SeqRefs(&'a [&'a Sequence]),
    DecodeVec(&'a Vec<DecodeSequence>),
}

pub enum Model {
    Qwen3(Arc<Qwen3ForCausalLM>),
    Qwen3MoE(Arc<Qwen3MoEForCausalLM>),
    LLaMa(Arc<LLaMaForCausalLM>),
    GLM4(Arc<GLM4ForCausalLM>),
    // Gemma(GemmaForCausalLM),
    // Phi(PhiForCausalLM),
    // Mistral(MistralForCausalLM),
    // Yi(YiForCausalLM),
    // StableLM(StableLMForCausalLM),
    // DeepSeek(DeepSeekForCausalLM),
}

pub enum RunnerType {
    Thread(ModelRunner),
    Process(Vec<LocalStream>),
}

pub struct ModelRunner {
    model: Model,
    gpu_kv_cache: Arc<Mutex<Vec<(Tensor, Tensor)>>>,
    cpu_kv_cache: Arc<Mutex<Vec<(Tensor, Tensor)>>>,
    device: Device,
    config: EngineConfig,
    #[cfg(all(feature = "cuda", feature = "graph"))]
    pub capturer: GraphCapturer<CudaGraphWrapper<CudaGraphFn>>,
    logit_processor: LogitsProcessor,
    sampling_params: RwLock<SamplingParams>,
    seq_tokens: RwLock<HashMap<usize, Vec<u32>>>,
    transfer: Option<Arc<Transfer>>,
}

impl ModelRunner {
    pub fn new(
        model_type: ModelType,
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        econfig: &mut EngineConfig,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: Device,
        reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
        transfer: Option<Arc<Transfer>>,
        stream: Option<LocalStream>,
    ) -> Result<Self> {
        let model = match model_type {
            ModelType::Qwen3 => Model::Qwen3(Arc::new(Qwen3ForCausalLM::new(
                vb,
                comm.clone(),
                config,
                dtype,
                is_rope_i,
                &device,
                Arc::clone(&reporter),
            )?)),
            ModelType::Qwen3MoE => Model::Qwen3MoE(Arc::new(Qwen3MoEForCausalLM::new(
                vb,
                comm.clone(),
                config,
                dtype,
                is_rope_i,
                &device,
                Arc::clone(&reporter),
            )?)),
            ModelType::LLaMa => Model::LLaMa(Arc::new(LLaMaForCausalLM::new(
                vb,
                comm.clone(),
                config,
                dtype,
                is_rope_i,
                &device,
                Arc::clone(&reporter),
            )?)),
            ModelType::GLM4 => Model::GLM4(Arc::new(GLM4ForCausalLM::new(
                vb,
                comm.clone(),
                config,
                dtype,
                is_rope_i,
                &device,
                Arc::clone(&reporter),
            )?)),
            // ModelType::Gemma => GemmaForCausalLM::new(vb, config, dtype, &device)?,
            // ModelType::Phi => PhiForCausalLM::new(vb, config, dtype, &device)?,
            // ModelType::Mistral => MistralForCausalLM::new(vb, config, dtype, &device)?,
            // ModelType::Yi => YiForCausalLM::new(vb, config, dtype, &device)?,
            // ModelType::StableLM => StableLMForCausalLM::new(vb, config, dtype, &device)?,
            // ModelType::DeepSeek => DeepSeekForCausalLM::new(vb, config, dtype, &device)?,
            _ => {
                candle_core::bail!("Unsupported model type: {:?}", model_type);
            }
        };

        #[cfg(all(feature = "cuda", feature = "graph"))]
        let wrapper = match &model {
            Model::Qwen3(m) => {
                let model_arc = Arc::clone(m);
                let closure = move |input_ids: &Tensor,
                                    positions: &Tensor,
                                    kv_caches: Option<&Vec<(Tensor, Tensor)>>,
                                    input_metadata: &InputMetadata| {
                    model_arc.forward(input_ids, positions, kv_caches, input_metadata)
                };
                let boxed_closure: Box<ModelFn> = Box::new(closure);
                CudaGraphWrapper::new(boxed_closure, device.as_cuda_device()?.clone().into())
            }
            Model::Qwen3MoE(m) => {
                let model_arc = Arc::clone(m);
                let closure = move |input_ids: &Tensor,
                                    positions: &Tensor,
                                    kv_caches: Option<&Vec<(Tensor, Tensor)>>,
                                    input_metadata: &InputMetadata| {
                    model_arc.forward(input_ids, positions, kv_caches, input_metadata)
                };
                let boxed_closure: Box<ModelFn> = Box::new(closure);
                CudaGraphWrapper::new(boxed_closure, device.as_cuda_device()?.clone().into())
            }
            Model::LLaMa(m) => {
                let model_arc = Arc::clone(m);
                let closure = move |input_ids: &Tensor,
                                    positions: &Tensor,
                                    kv_caches: Option<&Vec<(Tensor, Tensor)>>,
                                    input_metadata: &InputMetadata| {
                    model_arc.forward(input_ids, positions, kv_caches, input_metadata)
                };
                let boxed_closure: Box<ModelFn> = Box::new(closure);
                CudaGraphWrapper::new(boxed_closure, device.as_cuda_device()?.clone().into())
            }
            Model::GLM4(m) => {
                let model_arc = Arc::clone(m);
                let closure = move |input_ids: &Tensor,
                                    positions: &Tensor,
                                    kv_caches: Option<&Vec<(Tensor, Tensor)>>,
                                    input_metadata: &InputMetadata| {
                    model_arc.forward(input_ids, positions, kv_caches, input_metadata)
                };
                let boxed_closure: Box<ModelFn> = Box::new(closure);
                CudaGraphWrapper::new(boxed_closure, device.as_cuda_device()?.clone().into())
            }
        };

        let (gpu_kv_cache, cpu_kv_cache) = if let Some(s) = stream {
            use crate::runner::{receive_local, send_local, MessageType};
            use interprocess::TryClone;
            send_local(
                &mut vec![s.try_clone()?],
                &MessageType::InitAck(true),
                false,
            )?;
            let msg = receive_local(&mut s.try_clone()?, true)?;
            if let MessageType::UsableMemoryLeft(ecfg) = msg {
                *econfig = ecfg.clone(); // Update Engine config
                let (gpu_kv_cache, cpu_kv_cache) =
                    Self::init_kv_cache(&econfig, config, dtype, &device)?;
                (gpu_kv_cache, cpu_kv_cache)
            } else {
                Self::init_kv_cache(&econfig, config, dtype, &device)?
            }
        } else {
            if econfig.max_model_len.is_none() {
                use crate::utils::update_kvcache_config;
                update_kvcache_config(econfig, &config.clone(), dtype);
            }
            Self::init_kv_cache(&econfig, config, dtype, &device)?
        };

        let (temperature, top_k, top_p) = if econfig.generation_cfg.is_some() {
            (
                econfig.generation_cfg.as_ref().unwrap().temperature.clone(),
                econfig.generation_cfg.as_ref().unwrap().top_k.clone(),
                econfig.generation_cfg.as_ref().unwrap().top_p.clone(),
            )
        } else {
            (None, None, None)
        };

        let seed = if econfig.seed.is_none() {
            rand::random::<u64>()
        } else {
            econfig.seed.unwrap()
        };
        Ok(Self {
            model,
            gpu_kv_cache: Arc::new(Mutex::new(gpu_kv_cache)),
            cpu_kv_cache: Arc::new(Mutex::new(cpu_kv_cache)),
            device,
            config: econfig.clone(),
            #[cfg(all(feature = "cuda", feature = "graph"))]
            capturer: GraphCapturer::new(
                wrapper,
                econfig.max_num_seqs,
                econfig.max_model_len.unwrap_or(4096),
                econfig.block_size,
                config.hidden_size,
            ),
            logit_processor: LogitsProcessor::new(seed, temperature, top_k, top_p),
            sampling_params: RwLock::new(SamplingParams::default()),
            seq_tokens: RwLock::new(HashMap::new()),
            transfer,
        })
    }

    //[num_blocks, block_size, num_kv_heads, head_size]
    fn calculate_flash_key_value_block_shape(
        cfg: &Config,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize) {
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        (block_size, cfg.num_key_value_heads / num_shards, head_dim)
    }

    fn calculate_key_block_shape(
        cfg: &Config,
        dtype: DType,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        let x = 16 / element_size;
        (
            cfg.num_key_value_heads / num_shards,
            head_dim / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        cfg: &Config,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize) {
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        (cfg.num_key_value_heads / num_shards, head_dim, block_size)
    }

    fn init_kv_cache(
        econfig: &EngineConfig,
        config: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
        let num_gpu_blocks = econfig.num_blocks;
        #[cfg(feature = "cuda")]
        let num_cpu_blocks =
            (econfig.num_blocks as f32 * econfig.cpu_mem_fold.unwrap_or(1.0f32)) as usize;
        #[cfg(not(feature = "cuda"))]
        let num_cpu_blocks = 1;

        #[cfg(not(feature = "cuda"))]
        let sync_alloc = true;

        #[allow(unused)]
        #[cfg(feature = "cuda")]
        let sync_alloc = if let Some(p_cfg) = &econfig.pd_config {
            matches!(p_cfg.role, crate::transfer::PdRole::Server)
        } else {
            false
        };

        if cfg!(feature = "flash-context") {
            assert!(
                !econfig.fp8_kvcache.unwrap_or(false),
                "fp8 kvcache is not compatible with flash-context feature!"
            );
            let kv_shape = Self::calculate_flash_key_value_block_shape(
                config,
                econfig.block_size,
                econfig.num_shards.unwrap_or(1),
            );

            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for _ in 0..config.num_hidden_layers {
                let key_blocks = Tensor::empty(
                    (num_gpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    device,
                    Some(sync_alloc),
                )?;
                let value_blocks = Tensor::empty(
                    (num_gpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    device,
                    Some(sync_alloc),
                )?;
                gpu_cache.push((key_blocks, value_blocks));
            }
            for _ in 0..config.num_hidden_layers {
                let key_blocks = Tensor::zeros(
                    (num_cpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    &Device::Cpu,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_cpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    &Device::Cpu,
                )?;
                cpu_cache.push((key_blocks, value_blocks));
            }
            Ok((gpu_cache, cpu_cache))
        } else {
            let fp8_kvcache = econfig.fp8_kvcache.unwrap_or(false);
            let cache_dtype = if fp8_kvcache { DType::U8 } else { dtype };
            crate::log_warn!(
                "Using FP8 KV Cache? {}, cache dtype {:?}",
                fp8_kvcache,
                cache_dtype
            );
            let kshape = Self::calculate_key_block_shape(
                config,
                cache_dtype,
                econfig.block_size,
                econfig.num_shards.unwrap_or(1),
            );
            let vshape = Self::calculate_value_block_shape(
                config,
                econfig.block_size,
                econfig.num_shards.unwrap_or(1),
            );
            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for _ in 0..config.num_hidden_layers {
                let key_blocks = Tensor::empty(
                    (num_gpu_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                let value_blocks = Tensor::empty(
                    (num_gpu_blocks, vshape.0, vshape.1, vshape.2),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                gpu_cache.push((key_blocks, value_blocks));
            }
            for _ in 0..config.num_hidden_layers {
                let key_blocks = Tensor::zeros(
                    (num_cpu_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_cpu_blocks, vshape.0, vshape.1, vshape.2),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                cpu_cache.push((key_blocks, value_blocks));
            }
            Ok((gpu_cache, cpu_cache))
        }
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<(Tensor, Tensor)>> {
        loop {
            if let Ok(v) = self.gpu_kv_cache.try_lock() {
                return v;
            }
        }
    }

    pub fn get_cpu_kv_cache(&self) -> MutexGuard<'_, Vec<(Tensor, Tensor)>> {
        loop {
            if let Ok(v) = self.cpu_kv_cache.try_lock() {
                return v;
            }
        }
    }

    pub fn run(&self, seqs: Seqs, is_prefill: bool) -> Result<Vec<u32>> {
        let (input_ids, positions, input_metadata) = if is_prefill {
            match seqs {
                Seqs::SeqRefs(seqs) => self.prepare_prefill(seqs)?,
                Seqs::DecodeVec(_) => {
                    candle_core::bail!(
                        "Decode sequences are not supported for prefill. Use SeqRefs instead."
                    );
                }
            }
        } else {
            match seqs {
                Seqs::SeqRefs(seqs) => self.prepare_decode(seqs)?,
                Seqs::DecodeVec(decode_seqs) => self.prepare_decode(decode_seqs.iter())?,
            }
        };

        #[cfg(all(feature = "cuda", feature = "graph"))]
        if !is_prefill && self.capturer.is_captured(input_ids.dim(0)?) {
            let logits = self
                .capturer
                .replay(&input_ids, &positions, &input_metadata)?;
            let output_ids = self.sample(&logits, seqs, is_prefill)?;
            return Ok(output_ids);
        }

        let logits = match &self.model {
            Model::Qwen3(model) => model.forward(
                &input_ids,
                &positions,
                Some(&self.get_kv_cache()),
                &input_metadata,
            )?,
            Model::Qwen3MoE(model) => model.forward(
                &input_ids,
                &positions,
                Some(&self.get_kv_cache()),
                &input_metadata,
            )?,
            Model::LLaMa(model) => model.forward(
                &input_ids,
                &positions,
                Some(&self.get_kv_cache()),
                &input_metadata,
            )?,
            Model::GLM4(model) => model.forward(
                &input_ids,
                &positions,
                Some(&self.get_kv_cache()),
                &input_metadata,
            )?,
            // _ => {
            //     candle_core::bail!("Unsupported model type for forward pass");
            // }
        };
        let output_ids = self.sample(&logits, seqs, is_prefill)?;
        Ok(output_ids)
    }

    fn prepare_block_tables<'a, I, S>(&self, seqs: I) -> Result<Tensor>
    where
        I: IntoIterator<Item = &'a S>,
        S: ToDecodeInput + 'a,
    {
        let seq_refs: Vec<&'a S> = seqs.into_iter().collect(); // only references, no clone
        let len = seq_refs.len();

        let max_len = seq_refs
            .iter()
            .map(|seq| seq.block_table().len())
            .max()
            .unwrap_or(0);

        let mut flat: Vec<u32> = Vec::with_capacity(len * max_len);
        for seq in &seq_refs {
            let bt = seq.block_table();
            flat.extend_from_slice(bt);
            flat.extend(std::iter::repeat(0).take(max_len - bt.len()));
        }

        Tensor::from_vec(flat, (len, max_len), &self.device)
    }

    #[allow(non_snake_case)]
    fn prepare_prefill(&self, seqs: &[&Sequence]) -> Result<(Tensor, Tensor, InputMetadata)> {
        let mut input_ids: Vec<u32> = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0];
        let mut cu_seqlens_k = vec![0];
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        let mut slot_mapping = Vec::new();
        let CHUNK_SIZE: usize = if self.config.flash_context.unwrap_or(false) {
            2048
        } else {
            8192
        };
        let mut max_context_len = 0;
        for seq in seqs {
            let seqlen = seq.len();
            let num_tokens = std::cmp::min(CHUNK_SIZE, seqlen - seq.num_cached_tokens);
            input_ids
                .extend(&seq.token_ids[seq.num_cached_tokens..seq.num_cached_tokens + num_tokens]);
            positions.extend(
                (seq.num_cached_tokens as i64..(seq.num_cached_tokens + num_tokens) as i64)
                    .collect::<Vec<_>>(),
            );
            if seqlen > max_context_len {
                max_context_len = seqlen;
            }
            let seqlen_q = num_tokens; //seqlen - seq.num_cached_tokens;
            let seqlen_k = if self.config.flash_context.unwrap_or(false)
                || (seq.num_cached_tokens > 0 && cfg!(feature = "flash-context"))
            {
                seq.num_cached_tokens + num_tokens
            } else {
                num_tokens
            };
            cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + seqlen_q as u32);
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + seqlen_k as u32);
            max_seqlen_q = std::cmp::max(max_seqlen_q, seqlen_q);
            max_seqlen_k = std::cmp::max(max_seqlen_k, seqlen_k);

            let mut slot_mapping_tokens: i64 = 0;
            for i in seq.num_cached_blocks()..seq.num_blocks() {
                let start = (seq.block_table[i] * self.config.block_size as u32) as i64;
                let start = if i == seq.num_cached_blocks() {
                    start + (seq.num_cached_tokens as i64 % self.config.block_size as i64)
                } else {
                    start
                };
                let end = start
                    + std::cmp::min(
                        num_tokens as i64 - slot_mapping_tokens,
                        self.config.block_size as i64,
                    );
                slot_mapping.extend((start..end).collect::<Vec<i64>>());
                slot_mapping_tokens += end - start;
                if slot_mapping_tokens >= num_tokens as i64 {
                    break;
                }
            }
        }

        assert!(
            input_ids.len() > 0 && positions.len() > 0 && slot_mapping.len() > 0,
            "Invalid inputs!"
        );
        // Validate lengths
        if input_ids.len() != slot_mapping.len() {
            candle_core::bail!(
                "input_ids and slot_mapping must have same length: {}, {}",
                input_ids.len(),
                slot_mapping.len()
            );
        }
        if input_ids.len() != *cu_seqlens_q.last().unwrap() as usize {
            candle_core::bail!("input_ids length must match last cu_seqlens_q",);
        }
        // crate::log_info!("input_ids {:?}, positions {:?}, slot_mapping {:?}", input_ids, positions, slot_mapping);

        // Create tensors
        let length = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), &self.device)?;
        let positions = Tensor::from_vec(positions, (length,), &self.device)?;
        let q_len = cu_seqlens_q.len();
        let k_len = cu_seqlens_k.len();
        let s_len = slot_mapping.len();

        let slot_mapping = Tensor::from_vec(slot_mapping, (s_len,), &self.device)?;

        // Handle context cache
        let (context_lens, block_tables) = if cu_seqlens_k.last() > cu_seqlens_q.last() {
            let context_lens: Vec<u32> = seqs.iter().map(|seq| seq.len() as u32).collect();
            let context_lens_t = Tensor::from_vec(context_lens, seqs.len(), &self.device)?;
            let block_tables_t = self.prepare_block_tables(seqs)?;
            (Some(context_lens_t), Some(block_tables_t))
        } else {
            (None, None)
        };
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (q_len,), &self.device)?;
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, (k_len,), &self.device)?;

        let input_metadata = InputMetadata {
            is_prefill: true,
            slot_mapping,
            block_tables,
            context_lens,
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            max_seqlen_q,
            max_seqlen_k,
            max_context_len,
        };

        Ok((input_ids, positions, input_metadata))
    }

    fn prepare_decode<'a, I, S>(&self, seqs: I) -> Result<(Tensor, Tensor, InputMetadata)>
    where
        I: IntoIterator<Item = &'a S>,
        S: ToDecodeInput + 'a,
    {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut context_lens = Vec::new();

        let seq_refs: Vec<&'a S> = seqs.into_iter().collect(); // only references, no clone

        for seq in &seq_refs {
            input_ids.push(seq.last_token());
            positions.push((seq.len() - 1) as i64);
            context_lens.push(seq.len() as u32);
            let slot = seq.block_table_last() * self.config.block_size as u32
                + seq.last_block_tokens() as u32
                - 1;
            slot_mapping.push(slot as i64);
        }

        // Create tensors
        let length = positions.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), &self.device)?;
        let positions = Tensor::from_vec(positions, (length,), &self.device)?;
        let s_len = slot_mapping.len();
        let c_len = context_lens.len();
        let max_context_len = context_lens.clone().into_iter().max().unwrap() as usize;

        let slot_mapping = Tensor::from_vec(slot_mapping, (s_len,), &self.device)?;
        let context_lens = Tensor::from_vec(context_lens, (c_len,), &self.device)?;
        let block_tables = self.prepare_block_tables(seq_refs)?;
        let input_metadata = InputMetadata {
            is_prefill: false,
            slot_mapping,
            block_tables: Some(block_tables),
            context_lens: Some(context_lens),
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            max_context_len,
        };

        Ok((input_ids, positions, input_metadata))
    }

    fn sample(&self, logits: &Tensor, seqs: Seqs, is_prefill: bool) -> Result<Vec<u32>> {
        let seq_ids: Vec<usize> = match seqs {
            Seqs::SeqRefs(seqs) => seqs.into_iter().map(|s| s.id()).collect(),
            Seqs::DecodeVec(v) => v.into_iter().map(|s| s.id()).collect(),
        };

        let logits = if let Some(cfg) = &self.config.generation_cfg {
            if cfg.frequency_penalty.is_some() || cfg.presence_penalty.is_some() {
                let frequency_penalty = cfg.frequency_penalty.unwrap_or(0.);
                let presence_penalty = cfg.presence_penalty.unwrap_or(0.);
                let seq_tokens = self.seq_tokens.write();
                let reference_tokens: Vec<Vec<u32>> = seq_ids
                    .iter()
                    .map(|id| {
                        if let Some(tokens) = seq_tokens.get(&id) {
                            if tokens.len() > 128 {
                                tokens[tokens.len().saturating_sub(128)..].to_vec()
                            } else {
                                vec![]
                            }
                        } else {
                            vec![]
                        }
                    })
                    .collect();

                self.logit_processor.apply_batch_repeat_penalty(
                    logits,
                    vec![frequency_penalty; reference_tokens.len()],
                    vec![presence_penalty; reference_tokens.len()],
                    reference_tokens,
                )?
            } else {
                logits.to_owned()
            }
        } else {
            logits.to_owned()
        };

        let tokens = match seqs {
            Seqs::SeqRefs(seqs) => {
                if is_prefill {
                    *self.sampling_params.write() = seqs[0].sampling_params.clone();
                }
                if seqs.len() <= std::cmp::min(MAX_PARALLEL_SAMPLING, self.config.max_num_seqs) {
                    self.logit_processor
                        .sample(&logits, &Some(seqs[0].sampling_params.clone()))?
                } else {
                    logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?
                }
            }
            Seqs::DecodeVec(v) => {
                if v.len() <= std::cmp::min(MAX_PARALLEL_SAMPLING, self.config.max_num_seqs) {
                    let sampling_params = self.sampling_params.read();
                    self.logit_processor
                        .sample(&logits, &Some(sampling_params.clone()))?
                } else {
                    logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?
                }
            }
        };

        if let Some(cfg) = &self.config.generation_cfg {
            if cfg.frequency_penalty.is_some() || cfg.presence_penalty.is_some() {
                let mut seq_tokens = self.seq_tokens.write();
                for i in 0..seq_ids.len() {
                    if seq_tokens.contains_key(&seq_ids[i]) {
                        seq_tokens
                            .get_mut(&seq_ids[i])
                            .expect("no entry")
                            .push(tokens[i]);
                    } else {
                        seq_tokens.insert(seq_ids[i], vec![tokens[i]].into());
                    }
                }
            }
        }
        Ok(tokens)
    }

    pub fn finished(&self, id: usize) {
        let mut seq_tokens = self.seq_tokens.write();
        let _ = seq_tokens.remove(&id);
    }

    pub fn get_model_vocab_size(&self) -> usize {
        match &self.model {
            Model::Qwen3(model) => model.get_vocab_size(),
            Model::Qwen3MoE(model) => model.get_vocab_size(),
            Model::LLaMa(model) => model.get_vocab_size(),
            Model::GLM4(model) => model.get_vocab_size(),
        }
    }

    #[cfg(all(feature = "cuda", feature = "graph"))]
    pub fn warmup_capture(&mut self) -> Result<()> {
        let kv_cache_lock = self.gpu_kv_cache.lock().unwrap(); // no custom method call on `self`
        self.capturer.capture(&self.device, Some(&kv_cache_lock))
    }

    pub fn swap_kvcache(&self, mappings: HashMap<usize, usize>, swap_in: bool) -> Result<bool> {
        fn cache_swap(
            gpu_cache: &Vec<(Tensor, Tensor)>,
            cpu_cache: &Vec<(Tensor, Tensor)>,
            mappings: &HashMap<usize, usize>,
            swap_in: bool,
        ) -> Result<bool> {
            assert!(
                gpu_cache.len() > 0 && cpu_cache.len() > 0,
                "Invalid kvcache tensors!"
            );
            let block_size_bytes = cpu_cache[0].0.elem_count() / cpu_cache[0].0.dim(0)?
                * cpu_cache[0].0.dtype().size_in_bytes();
            for i in 0..gpu_cache.len() {
                if swap_in {
                    cache::swap_blocks(&cpu_cache[i].0, &gpu_cache[i].0, mappings)?;
                    cache::swap_blocks(&cpu_cache[i].1, &gpu_cache[i].1, mappings)?;
                } else {
                    cache::swap_blocks(&gpu_cache[i].0, &cpu_cache[i].0, mappings)?;
                    cache::swap_blocks(&gpu_cache[i].1, &cpu_cache[i].1, mappings)?;
                }
            }
            let total_mb_bytes_swapped =
                (block_size_bytes * mappings.len() * gpu_cache.len() * 2) as f32 / 1024.0 / 1024.0;
            if swap_in {
                crate::log_info!(
                    "{:.2} MB CPU KV cached blocks swapped in GPU!",
                    total_mb_bytes_swapped
                );
            } else {
                crate::log_info!(
                    "{:.2} MB GPU KV cached blocks swapped out to CPU!",
                    total_mb_bytes_swapped
                );
            }
            Ok(true)
        }
        cache_swap(
            &*self.get_kv_cache(),
            &*self.get_cpu_kv_cache(),
            &mappings,
            swap_in,
        )
    }

    pub fn transfer_prefill(&self, seq: &Sequence) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_client() {
                candle_core::bail!(
                    "PD server does not support prefill transfer, call this in the client!"
                )
            }
            transfer.transfer_prefill(seq)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn try_receive_prefill(&self, available_tokens: usize) -> Result<(bool, Option<Sequence>)> {
        if let Some(transfer) = &self.transfer {
            if transfer.is_client() {
                candle_core::bail!("PD client does not support try_receive_prefill!");
            }
            transfer.try_receive_prefill_request(available_tokens)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!");
        }
    }

    pub fn check_prefill_status(&self, seq_id: usize) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_client() {
                candle_core::bail!("PD server does not support check prefill status!");
            }
            transfer.check_prefill_finished(seq_id)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!");
        }
    }

    pub fn send_kvcache(&self, seq: &Sequence, first_token: u32) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_server() {
                candle_core::bail!(
                    "PD client does not support send_kvcache, call this in the PD server!"
                )
            }
            transfer.transfer_kv_cache(seq, &*self.get_kv_cache(), first_token)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn receive_kvcache(&self, seq: &Sequence) -> Result<(bool, u32, usize)> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_client() {
                candle_core::bail!(
                    "PD server does not support receive_kvcache, call this in the PD client!"
                )
            }
            transfer.receive_kv_cache(seq, &*self.get_kv_cache())
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn release_remote_kvcache(&self, seq_id: usize) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if !transfer.is_client() {
                candle_core::bail!("release_remote_kvcache should be called from PD client!")
            }
            transfer.release_remote_kvcache(seq_id)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn check_kvcache_release(&self, seq_id: usize) -> Result<bool> {
        if let Some(transfer) = &self.transfer {
            if transfer.is_client() {
                candle_core::bail!("try_check_kvcache_release should be called from PD server!")
            }
            transfer.check_kvcache_release(seq_id)
        } else {
            candle_core::bail!("KV Cache transfer engine is not initialized!")
        }
    }

    pub fn clear_blocks(&self, block_ids: Vec<u32>) -> Result<bool> {
        fn cache_clear(gpu_cache: &Vec<(Tensor, Tensor)>, block_ids: &Vec<u32>) -> Result<bool> {
            if gpu_cache.is_empty() || block_ids.is_empty() {
                return Ok(true);
            }

            let block_size_bytes = gpu_cache[0].0.elem_count() / gpu_cache[0].0.dim(0)?
                * gpu_cache[0].0.dtype().size_in_bytes();

            for i in 0..gpu_cache.len() {
                cache::clear_blocks(&gpu_cache[i].0, block_ids)?;
                cache::clear_blocks(&gpu_cache[i].1, block_ids)?;
            }

            // Log the total memory cleared
            let total_mb_bytes_cleared =
                (block_size_bytes * block_ids.len() * gpu_cache.len() * 2) as f32 / 1024.0 / 1024.0;

            crate::log_info!(
                "ClearBlock: {:.2} MB KV cached blocks zeroed out on GPU!",
                total_mb_bytes_cleared
            );

            Ok(true)
        }

        cache_clear(&*self.get_kv_cache(), &block_ids)
    }
}
