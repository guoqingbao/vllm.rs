use crate::models::gemma3::Gemma3ForConditionalGeneration;
// src/core/runner.rs
use crate::models::layers::distributed::Comm;
use crate::models::layers::VarBuilderX;
use crate::server::EmbeddingStrategy;
use crate::transfer::Transfer;
#[cfg(all(feature = "cuda", feature = "graph"))]
use crate::utils::graph::{CudaGraphFn, CudaGraphWrapper, GraphCapturer, ModelFn};
use crate::utils::guidance::GuidanceState;
use crate::utils::image::compute_image_slice;
use crate::utils::logits_processor::{LogitsProcessor, Sampling};
use crate::utils::progress::ProgressLike;
use crate::{
    core::sequence::{DecodeSequence, Sequence, ToDecodeInput},
    models::glm4::GLM4ForCausalLM,
    models::glm4_moe::GLM4MoEForCausalLM,
    models::llama::LLaMaForCausalLM,
    models::mistral3_vl::Mistral3ForConditionalGeneration,
    models::phi4::Phi4ForCausalLM,
    models::qwen3::Qwen3ForCausalLM,
    models::qwen3_moe::Qwen3MoEForCausalLM,
    models::qwen3_vl::Qwen3VLForConditionalGeneration,
    utils::config::{Config, EngineConfig, ModelType},
    utils::kvcache_allocator::KVCacheAllocator,
};
use attention_rs::cache;
use attention_rs::FlashInferMetadata;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use interprocess::local_socket::Stream as LocalStream;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};
use toktrie::TokTrie;

/// Cached sampling parameters computed once during prefill, reused during decode
#[derive(Clone, Debug)]
pub struct CachedSamplingParams {
    pub sampling: Sampling,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

pub enum Seqs<'a> {
    SeqRefs(&'a [&'a Sequence]),
    DecodeVec(&'a Vec<DecodeSequence>),
}

pub enum Model {
    Qwen3(Arc<Qwen3ForCausalLM>),
    Qwen3MoE(Arc<Qwen3MoEForCausalLM>),
    LLaMa(Arc<LLaMaForCausalLM>),
    Phi4(Arc<Phi4ForCausalLM>),
    GLM4(Arc<GLM4ForCausalLM>),
    GLM4MoE(Arc<GLM4MoEForCausalLM>),
    Mistral3VL(Arc<Mistral3ForConditionalGeneration>),
    Gemma3(Arc<Gemma3ForConditionalGeneration>),
    Qwen3VL(Arc<Qwen3VLForConditionalGeneration>),
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
    /// Cached sampling strategy computed once during prefill, reused during decode
    cached_sampling: RwLock<Option<CachedSamplingParams>>,
    seq_tokens: RwLock<HashMap<usize, Vec<u32>>>,
    guidance_states: RwLock<HashMap<usize, GuidanceState>>,
    transfer: Option<Arc<Transfer>>,
    /// Whether this runner is on the first rank (for logging)
    is_first_rank: bool,
    model_type: ModelType,
}

impl ModelRunner {
    #[allow(unused)]
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
        toktrie: Option<Arc<TokTrie>>,
        stream: Option<LocalStream>,
    ) -> Result<Self> {
        let model = crate::build_model!(
            model_type,
            vb,
            comm,
            config,
            dtype,
            is_rope_i,
            &device,
            reporter,
            {
                Qwen3 => Qwen3ForCausalLM,
                Qwen3MoE => Qwen3MoEForCausalLM,
                LLaMa => LLaMaForCausalLM,
                Phi4 => Phi4ForCausalLM,
                GLM4 => GLM4ForCausalLM,
                GLM4MoE => GLM4MoEForCausalLM,
                Mistral3VL => Mistral3ForConditionalGeneration,
                Gemma3 => Gemma3ForConditionalGeneration,
                Qwen3VL => Qwen3VLForConditionalGeneration,
            }
        )?;

        #[cfg(all(feature = "cuda", feature = "graph"))]
        let wrapper = crate::graph_wrapper!(
            &model,
            device,
            {
                Qwen3 => EmbedInputs,
                Qwen3MoE => EmbedInputs,
                LLaMa => EmbedInputs,
                Phi4 => EmbedInputs,
                GLM4 => EmbedInputs,
                GLM4MoE => EmbedInputs,
                Mistral3VL => NoneArg,
                Gemma3 => NoneArg,
                Qwen3VL => NoneArg,
            }
        );

        let allocator = if let Some(s) = stream {
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
            }
            KVCacheAllocator::new(econfig, config, dtype)
        } else {
            let allocator = KVCacheAllocator::new(&econfig, &config, dtype);
            let device_ids = econfig.device_ids.clone().unwrap_or(vec![0]);
            match allocator.plan(&device_ids, econfig) {
                Ok(_) => {
                    crate::log_info!("KVCache allocation successfully planned!");
                }
                Err(e) => {
                    candle_core::bail!("KVCache allocation failed: {}", e);
                }
            }
            allocator
        };

        let allocation = crate::utils::kvcache_allocator::KVCacheAllocation {
            num_gpu_blocks: econfig.num_blocks,
            #[cfg(feature = "cuda")]
            num_cpu_blocks: (econfig.num_blocks as f32 * econfig.cpu_mem_fold.unwrap_or(0.2))
                as usize,
            #[cfg(not(feature = "cuda"))]
            num_cpu_blocks: 1, // dummy for non-CUDA platform
            max_num_seqs: econfig.max_num_seqs,
            max_model_len: econfig.max_model_len.unwrap_or(32768),
            kvcache_memory_bytes: econfig.kvcache_memory_bytes,
            max_num_batched_tokens: econfig.max_num_batched_tokens,
        };
        let (gpu_kv_cache, cpu_kv_cache) =
            allocator.init_kv_cache(&allocation, dtype, &device, econfig.pd_config.as_ref())?;

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
            cached_sampling: RwLock::new(None),
            seq_tokens: RwLock::new(HashMap::new()),
            guidance_states: RwLock::new(HashMap::new()),
            transfer,
            is_first_rank: comm.rank() == 0,
            model_type,
        })
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

        let images = if let Seqs::SeqRefs(s) = &seqs {
            // We do not batch multimodel prefill
            if let Some(images) = &s[0].images {
                if images.image_idx == -1 || !is_prefill {
                    None
                } else {
                    compute_image_slice(&s[0].token_ids, s[0].num_cached_tokens, images).map(
                        |(image_idx, token_offset)| {
                            let mut images = images.clone();
                            images.image_idx = image_idx;
                            images.image_token_offset = token_offset;
                            images
                        },
                    )
                }
            } else {
                None
            }
        } else {
            None
        };
        let images = images.as_ref();

        let logits = crate::model_call!(
            &self.model,
            forward,
            (&input_ids, &positions, Some(&self.get_kv_cache()), &input_metadata),
            {
                Qwen3 => false,
                Qwen3MoE => false,
                LLaMa => false,
                Phi4 => false,
                GLM4 => false,
                GLM4MoE => false,
                Mistral3VL => images,
                Gemma3 => images,
                Qwen3VL => images,
            }
        )?;
        let output_ids = self.sample(&logits, seqs, is_prefill)?;
        Ok(output_ids)
    }

    pub fn embed(&self, seqs: &[&Sequence], strategy: &EmbeddingStrategy) -> Result<Vec<Vec<f32>>> {
        let (input_ids, positions, input_metadata) = self.prepare_prefill(seqs)?;

        let hidden = crate::model_call!(
            &self.model,
            forward_embedding,
            (&input_ids, &positions, Some(&self.get_kv_cache()), &input_metadata),
            {
                Qwen3 => false,
                Qwen3MoE => false,
                LLaMa => false,
                Phi4 => false,
                GLM4 => false,
                Gemma3 => None,
            },
            candle_core::bail!("Embedding is not supported for this model type")
        )?;

        crate::log_info!(
            "Embedding forward finished with hidden shape {:?}",
            hidden.shape()
        );
        let hidden = hidden.to_dtype(DType::F32)?;
        let dims = hidden.dims();
        if dims.len() != 2 {
            candle_core::bail!("Unexpected embedding tensor dims {:?}", dims);
        }

        let mut start = 0;
        let mut outputs = Vec::new();
        for seq in seqs {
            let len = seq.len().saturating_sub(seq.num_cached_tokens);
            crate::log_info!(
                "Extracting embedding state for Seq {} (start {start}, len {len})",
                seq.id
            );
            let slice = hidden.narrow(0, start, len)?;
            let pooled = match strategy {
                EmbeddingStrategy::Mean => slice.mean(D::Minus2)?,
                EmbeddingStrategy::Last => slice.narrow(0, len.saturating_sub(1), 1)?.squeeze(0)?,
            };
            outputs.push(pooled.to_vec1::<f32>()?);
            start += len;
        }

        Ok(outputs)
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
        let mut batch_indices_vec: Vec<u32> = Vec::new();
        let mut positions_vec: Vec<u32> = Vec::new();
        let mut prefill_tokens: Vec<usize> = Vec::new();
        let mut cu_seqlens_q = vec![0];
        let mut cu_seqlens_k = vec![0];
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        let mut slot_mapping = Vec::new();
        let CHUNK_SIZE: usize = 8192;
        let mut max_context_len = 0;
        for (seq_idx, seq) in seqs.iter().enumerate() {
            let seqlen = seq.len();
            let num_tokens = std::cmp::min(CHUNK_SIZE, seqlen - seq.num_cached_tokens);
            input_ids
                .extend(&seq.token_ids[seq.num_cached_tokens..seq.num_cached_tokens + num_tokens]);
            positions.extend(
                (seq.num_cached_tokens as i64..(seq.num_cached_tokens + num_tokens) as i64)
                    .collect::<Vec<_>>(),
            );
            for pos in 0..num_tokens {
                batch_indices_vec.push(seq_idx as u32);
                positions_vec.push((seq.num_cached_tokens + pos) as u32);
            }
            prefill_tokens.push(num_tokens);
            if seqlen > max_context_len {
                max_context_len = seqlen;
            }
            let seqlen_q = num_tokens; //seqlen - seq.num_cached_tokens;
            let seqlen_k = if self.config.prefix_cache.unwrap_or(false)
                || self.config.prefix_cache.unwrap_or(false)
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

        // Handle cached prefix KV reuse
        let (context_lens, block_tables) = if cu_seqlens_k.last() > cu_seqlens_q.last() {
            let context_lens: Vec<u32> = seqs.iter().map(|seq| seq.len() as u32).collect();
            let context_lens_t = Tensor::from_vec(context_lens, seqs.len(), &self.device)?;
            let block_tables_t = self.prepare_block_tables(seqs)?;
            (Some(context_lens_t), Some(block_tables_t))
        } else {
            (None, None)
        };
        let cu_seqlens_q_vec = cu_seqlens_q.clone();
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (q_len,), &self.device)?;
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, (k_len,), &self.device)?;

        let disable_flash_attn = if matches!(self.model_type, ModelType::Gemma3) {
            Some(true)
        } else {
            None
        };

        let flashinfer_metadata = if cfg!(feature = "flashinfer") {
            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for (seq, &num_tokens) in seqs.iter().zip(prefill_tokens.iter()) {
                let effective_len = seq.num_cached_tokens + num_tokens;
                let max_blocks = seq.block_table.len();
                let num_blocks = if effective_len == 0 {
                    0
                } else {
                    (effective_len + self.config.block_size - 1) / self.config.block_size
                };
                let num_blocks = std::cmp::min(num_blocks, max_blocks);
                let bt = &seq.block_table[..num_blocks];
                indices.extend(bt.iter().map(|&x| x as u32));
                indptr.push(indices.len() as u32);
                let last = if effective_len == 0 {
                    0
                } else {
                    (effective_len - 1) % self.config.block_size + 1
                };
                last_len.push(last as u32);
            }

            let indptr_host = indptr.clone();
            let indptr_len = indptr.len();
            let indices_len = indices.len();
            let last_len_val = last_len.len();
            let batch_indices_len = batch_indices_vec.len();
            let positions_len = positions_vec.len();

            let indptr = Tensor::from_vec(indptr, (indptr_len,), &self.device)?;
            let indices = Tensor::from_vec(indices, (indices_len,), &self.device)?;
            let last_len = Tensor::from_vec(last_len, (last_len_val,), &self.device)?;
            let batch_indices =
                Tensor::from_vec(batch_indices_vec, (batch_indices_len,), &self.device)?;
            let positions = Tensor::from_vec(positions_vec, (positions_len,), &self.device)?;

            Some(FlashInferMetadata {
                indptr,
                indptr_host,
                indices,
                last_len,
                cu_seqlens_q_host: Some(cu_seqlens_q_vec.iter().map(|&x| x as u32).collect()),
                total_num_rows: Some(*cu_seqlens_q_vec.last().unwrap() as u32),
                batch_indices: Some(batch_indices),
                positions: Some(positions),
                use_cuda_graph: false,
            })
        } else {
            None
        };

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
            disable_flash_attn,
            seqlens: Some(cu_seqlens_q_vec[1..].to_vec()),
            flashinfer_metadata,
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
        let block_tables = self.prepare_block_tables(seq_refs.clone())?;

        let flashinfer_metadata = if cfg!(feature = "flashinfer") {
            #[cfg(all(feature = "cuda", feature = "graph"))]
            let use_cuda_graph = self.capturer.is_captured(seq_refs.len());
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            let use_cuda_graph = false;

            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for seq in &seq_refs {
                let bt = seq.block_table();
                indices.extend(bt.iter().map(|&x| x as u32));
                indptr.push(indices.len() as u32);
                let len = seq.len();
                let last = if len == 0 {
                    0
                } else {
                    (len - 1) % self.config.block_size + 1
                };
                last_len.push(last as u32);
            }
            let indptr_host = indptr.clone();
            let indptr_len = indptr.len();
            let indices_len = indices.len();
            let last_len_val = last_len.len();

            let indptr = Tensor::from_vec(indptr, (indptr_len,), &self.device)?;
            let indices = Tensor::from_vec(indices, (indices_len,), &self.device)?;
            let last_len = Tensor::from_vec(last_len, (last_len_val,), &self.device)?;

            Some(FlashInferMetadata {
                indptr,
                indptr_host,
                indices,
                last_len,
                cu_seqlens_q_host: None,
                total_num_rows: None,
                batch_indices: None,
                positions: None,
                use_cuda_graph,
            })
        } else {
            None
        };

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
            disable_flash_attn: None,
            seqlens: None,
            flashinfer_metadata,
        };

        Ok((input_ids, positions, input_metadata))
    }

    fn sample(&self, logits: &Tensor, seqs: Seqs, is_prefill: bool) -> Result<Vec<u32>> {
        let seq_ids: Vec<usize> = match &seqs {
            Seqs::SeqRefs(seqs) => seqs.iter().map(|s| s.id()).collect(),
            Seqs::DecodeVec(v) => v.iter().map(|s| s.id()).collect(),
        };

        // Get the batch size for deciding whether to use parallel sampling
        let batch_size = match seqs {
            Seqs::SeqRefs(seqs) => seqs.len(),
            Seqs::DecodeVec(v) => v.len(),
        };

        // Compute and cache sampling params (including penalties) during prefill, reuse during decode
        let cached_params = match (is_prefill, &seqs) {
            // Prefill: compute sampling strategy and penalties, cache for decode phase
            (true, Seqs::SeqRefs(seqs)) => {
                // Check if generation_cfg has valid sampling params (temperature AND top_k/top_p)
                let has_valid_sampling_cfg =
                    self.config.generation_cfg.as_ref().map_or(false, |cfg| {
                        cfg.temperature.is_some() && (cfg.top_k.is_some() || cfg.top_p.is_some())
                    });
                let user_params = &seqs[0].sampling_params;

                // Log thinking parameter only from first rank to avoid duplicate logs in multi-GPU
                if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                    crate::log_info!(
                        "User's thinking preference for reasoning models: {:?}",
                        user_params.thinking
                    );
                }

                // Determine frequency/presence penalties (user params > generation_cfg)
                let gen_cfg_freq = self
                    .config
                    .generation_cfg
                    .as_ref()
                    .and_then(|c| c.frequency_penalty);
                let gen_cfg_pres = self
                    .config
                    .generation_cfg
                    .as_ref()
                    .and_then(|c| c.presence_penalty);
                let frequency_penalty = user_params.frequency_penalty.or(gen_cfg_freq);
                let presence_penalty = user_params.presence_penalty.or(gen_cfg_pres);

                let sampling = if has_valid_sampling_cfg {
                    let cfg = self.config.generation_cfg.as_ref().unwrap();
                    if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                        crate::log_warn!(
                            "Using sampling from generation_config: temp={:?}, top_k={:?}, top_p={:?}, freq_penalty={:?}, pres_penalty={:?}",
                            cfg.temperature,
                            cfg.top_k,
                            cfg.top_p,
                            frequency_penalty,
                            presence_penalty
                        );
                    }
                    LogitsProcessor::get_strategy(cfg.temperature, cfg.top_k, cfg.top_p)
                } else {
                    let has_user_config = matches!(user_params.temperature, Some(t) if t != 0.0 && t != 1.0)
                        && (matches!(user_params.top_k, Some(k) if k > 0)
                            || matches!(user_params.top_p, Some(p) if p != 0.0 && p != 1.0));
                    if has_user_config {
                        if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                            crate::log_warn!(
                                "Using user's sampling params: temp={:?}, top_k={:?}, top_p={:?}, freq_penalty={:?}, pres_penalty={:?}",
                                user_params.temperature,
                                user_params.top_k,
                                user_params.top_p,
                                frequency_penalty,
                                presence_penalty
                            );
                        }
                        LogitsProcessor::get_strategy(
                            user_params.temperature,
                            user_params.top_k,
                            user_params.top_p,
                        )
                    } else {
                        if self.is_first_rank && seqs[0].num_cached_tokens == 0 {
                            crate::log_warn!(
                                "No generation_config, using default sampling (temperature=0.7, top_k=32, top_p=0.95)"
                            );
                        }
                        Sampling::TopKThenTopP {
                            k: 32,
                            p: 0.95,
                            temperature: 0.7,
                        }
                    }
                };

                let cached = CachedSamplingParams {
                    sampling,
                    frequency_penalty,
                    presence_penalty,
                };

                // Cache for decode phase
                *self.cached_sampling.write() = Some(cached.clone());
                cached
            }
            // Decode or non-SeqRefs: use cached parameters
            _ => self
                .cached_sampling
                .read()
                .clone()
                .unwrap_or(CachedSamplingParams {
                    sampling: Sampling::TopKThenTopP {
                        k: 32,
                        p: 0.95,
                        temperature: 0.7,
                    },
                    frequency_penalty: None,
                    presence_penalty: None,
                }),
        };

        // Apply penalties using cached values (same for all sequences in batch)
        let has_any_penalty =
            cached_params.frequency_penalty.is_some() || cached_params.presence_penalty.is_some();

        let logits = if !is_prefill && has_any_penalty {
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
                vec![cached_params.frequency_penalty.unwrap_or(0.0); batch_size],
                vec![cached_params.presence_penalty.unwrap_or(0.0); batch_size],
                reference_tokens,
            )?
        } else {
            logits.to_owned()
        };

        let tokens = self
            .logit_processor
            .sample_with_strategy(&logits, &cached_params.sampling)?;

        // Track tokens for sequences when penalties are enabled
        if has_any_penalty {
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
        Ok(tokens)
    }

    pub fn finished(&self, id: usize) {
        let mut seq_tokens = self.seq_tokens.write();
        let _ = seq_tokens.remove(&id);
        let mut guidance_states = self.guidance_states.write();
        let _ = guidance_states.remove(&id);
    }

    pub fn get_model_vocab_size(&self) -> usize {
        match &self.model {
            Model::Qwen3(model) => model.get_vocab_size(),
            Model::Qwen3MoE(model) => model.get_vocab_size(),
            Model::LLaMa(model) => model.get_vocab_size(),
            Model::Phi4(model) => model.get_vocab_size(),
            Model::GLM4(model) => model.get_vocab_size(),
            Model::GLM4MoE(model) => model.get_vocab_size(),
            Model::Mistral3VL(model) => model.get_vocab_size(),
            Model::Gemma3(model) => model.get_vocab_size(),
            Model::Qwen3VL(model) => model.get_vocab_size(),
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

    pub fn clear_blocks(&self, _block_ids: Vec<u32>) -> Result<bool> {
        Ok(true)
        // fn cache_clear(gpu_cache: &Vec<(Tensor, Tensor)>, block_ids: &Vec<u32>) -> Result<bool> {
        //     if gpu_cache.is_empty() || block_ids.is_empty() {
        //         return Ok(true);
        //     }

        //     for i in 0..gpu_cache.len() {
        //         cache::clear_blocks(&gpu_cache[i].0, block_ids)?;
        //         cache::clear_blocks(&gpu_cache[i].1, block_ids)?;
        //     }

        //     Ok(true)
        // }

        // cache_clear(&*self.get_kv_cache(), &block_ids)
    }
}
