// src/core/runner.rs
use crate::models::layers::VarBuilderX;
use crate::utils::progress::{progress_worker, ProgressReporter};
use crate::{
    core::sequence::Sequence,
    models::llama::LLaMaForCausalLM,
    models::qwen3::Qwen3ForCausalLM,
    utils::config::{Config, EngineConfig, ModelType},
};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use std::sync::RwLock;
use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(all(feature = "cuda", feature = "graph"))]
use crate::utils::graph::{CudaGraphFn, CudaGraphWrapper, GraphCapturer, ModelFn};

pub enum Model {
    Qwen3(Arc<Qwen3ForCausalLM>),
    LLaMa(Arc<LLaMaForCausalLM>),
    // Gemma(GemmaForCausalLM),
    // Phi(PhiForCausalLM),
    // Mistral(MistralForCausalLM),
    // GLM4(GLM4ForCausalLM),
    // Yi(YiForCausalLM),
    // StableLM(StableLMForCausalLM),
    // DeepSeek(DeepSeekForCausalLM),
}

pub struct ModelRunner {
    model: Model,
    kv_cache: Arc<Mutex<Vec<(Tensor, Tensor)>>>,
    device: Device,
    config: EngineConfig,
    pub use_flash_attn: Option<bool>,
    #[cfg(all(feature = "cuda", feature = "graph"))]
    pub capturer: GraphCapturer<CudaGraphWrapper<CudaGraphFn>>,
}

impl ModelRunner {
    pub fn new(
        model_type: ModelType,
        vb: &VarBuilderX,
        econfig: &EngineConfig,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: Device,
    ) -> Result<Self> {
        let reporter = Arc::new(RwLock::new(ProgressReporter::new(0)));
        progress_worker(Some(1), config.num_hidden_layers, Arc::clone(&reporter));
        let model = match model_type {
            ModelType::Qwen3 => Model::Qwen3(Arc::new(Qwen3ForCausalLM::new(
                vb,
                config,
                dtype,
                is_rope_i,
                &device,
                Arc::clone(&reporter),
            )?)),
            ModelType::LLaMa => Model::LLaMa(Arc::new(LLaMaForCausalLM::new(
                vb,
                config,
                dtype,
                is_rope_i,
                &device,
                Arc::clone(&reporter),
            )?)),
            // ModelType::Gemma => GemmaForCausalLM::new(vb, config, dtype, &device)?,
            // ModelType::Phi => PhiForCausalLM::new(vb, config, dtype, &device)?,
            // ModelType::Mistral => MistralForCausalLM::new(vb, config, dtype, &device)?,
            // ModelType::GLM4 => GLM4ForCausalLM::new(vb, config, dtype, &device)?,
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
        };

        let kv_cache =
            Self::init_kv_cache(econfig, config, dtype, &device, econfig.use_flash_attn)?;

        Ok(Self {
            model,
            kv_cache: Arc::new(Mutex::new(kv_cache)),
            device,
            config: econfig.clone(),
            use_flash_attn: econfig.use_flash_attn,
            #[cfg(all(feature = "cuda", feature = "graph"))]
            capturer: GraphCapturer::new(
                wrapper,
                econfig.max_num_seqs,
                econfig.max_model_len.unwrap_or(4096),
                econfig.block_size,
                config.hidden_size,
            ),
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
        flash_attn: Option<bool>,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let num_gpu_blocks = econfig.num_blocks;
        let flash_attn = flash_attn.unwrap_or(false);
        if flash_attn {
            let kv_shape = Self::calculate_flash_key_value_block_shape(
                config,
                econfig.block_size,
                econfig.num_shards.unwrap_or(1),
            );

            let mut gpu_cache = Vec::new();
            for _ in 0..config.num_hidden_layers {
                let key_blocks = Tensor::zeros(
                    (num_gpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    device,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_gpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    device,
                )?;
                gpu_cache.push((key_blocks, value_blocks));
            }
            Ok(gpu_cache)
        } else {
            let kshape = Self::calculate_key_block_shape(
                config,
                dtype,
                econfig.block_size,
                econfig.num_shards.unwrap_or(1),
            );
            let vshape = Self::calculate_value_block_shape(
                config,
                econfig.block_size,
                econfig.num_shards.unwrap_or(1),
            );
            let mut gpu_cache = Vec::new();
            for _ in 0..config.num_hidden_layers {
                let key_blocks = Tensor::zeros(
                    (num_gpu_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    dtype,
                    device,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_gpu_blocks, vshape.0, vshape.1, vshape.2),
                    dtype,
                    device,
                )?;
                gpu_cache.push((key_blocks, value_blocks));
            }
            Ok(gpu_cache)
        }
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<(Tensor, Tensor)>> {
        loop {
            if let Ok(v) = self.kv_cache.try_lock() {
                return v;
            }
        }
    }

    pub fn run(&mut self, seqs: &[&Sequence], is_prefill: bool) -> Result<Vec<u32>> {
        let (input_ids, positions, input_metadata) = if is_prefill {
            self.prepare_prefill(seqs)
        } else {
            self.prepare_decode(seqs)
        }?;

        #[cfg(all(feature = "cuda", feature = "graph"))]
        if !is_prefill && self.capturer.is_captured(input_ids.dim(0)?) {
            let logits = self
                .capturer
                .replay(&input_ids, &positions, &input_metadata)?;
            let output_ids = self.sample(&logits)?;
            return Ok(output_ids);
        }

        let logits = match &self.model {
            Model::Qwen3(model) => model.forward(
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
            // _ => {
            //     candle_core::bail!("Unsupported model type for forward pass");
            // }
        };
        let output_ids = self.sample(&logits)?;
        Ok(output_ids)
    }

    fn prepare_block_tables(&self, seqs: &[&Sequence]) -> Result<Tensor> {
        let max_len = seqs
            .iter()
            .map(|seq| seq.block_table.len())
            .max()
            .unwrap_or(0);

        let mut block_tables: Vec<u32> = Vec::with_capacity(seqs.len() * max_len);
        for seq in seqs {
            let row_len = seq.block_table.len();
            block_tables.extend(
                seq.block_table
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<u32>>(),
            );
            //i32?
            block_tables.extend(std::iter::repeat(0u32).take(max_len - row_len));
        }

        Tensor::from_vec(block_tables, (seqs.len(), max_len), &self.device)
    }

    fn prepare_prefill(&self, seqs: &[&Sequence]) -> Result<(Tensor, Tensor, InputMetadata)> {
        let mut input_ids: Vec<u32> = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0];
        let mut cu_seqlens_k = vec![0];
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        let mut slot_mapping = Vec::new();

        for seq in seqs {
            let seqlen = seq.len();
            input_ids.extend(&seq.token_ids[seq.num_cached_tokens..]);
            positions.extend((seq.num_cached_tokens as i64..seqlen as i64).collect::<Vec<_>>());

            let seqlen_q = seqlen - seq.num_cached_tokens;
            let seqlen_k = seqlen;
            cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + seqlen_q as u32);
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + seqlen_k as u32);
            max_seqlen_q = std::cmp::max(max_seqlen_q, seqlen_q);
            max_seqlen_k = std::cmp::max(max_seqlen_k, seqlen_k);

            for i in seq.num_cached_blocks()..seq.num_blocks() {
                let start = (seq.block_table[i] * self.config.block_size) as i64;
                let end = if i == seq.num_blocks() - 1 {
                    start + seq.last_block_num_tokens() as i64
                } else {
                    start + self.config.block_size as i64
                };
                slot_mapping.extend((start..end).collect::<Vec<i64>>());
            }
        }

        // Validate lengths
        if input_ids.len() != slot_mapping.len() {
            candle_core::bail!("input_ids and slot_mapping must have same length",);
        }
        if input_ids.len() != *cu_seqlens_q.last().unwrap() as usize {
            candle_core::bail!("input_ids length must match last cu_seqlens_q",);
        }

        // Create tensors
        let length = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), &self.device)?;
        let positions = Tensor::from_vec(positions, (length,), &self.device)?;
        let q_len = cu_seqlens_q.len();
        let k_len = cu_seqlens_k.len();
        let s_len = slot_mapping.len();

        let slot_mapping = Tensor::from_vec(slot_mapping, (s_len,), &self.device)?;

        // Handle prefix caching
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
            max_context_len: self.config.max_model_len.unwrap_or(4096),
            use_flash_attn: self.use_flash_attn,
        };

        Ok((input_ids, positions, input_metadata))
    }

    fn prepare_decode(&self, seqs: &[&Sequence]) -> Result<(Tensor, Tensor, InputMetadata)> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut context_lens = Vec::new();

        for seq in seqs {
            input_ids.push(seq.last_token);
            positions.push(seq.len() as i64);
            context_lens.push(seq.len() as u32);
            let slot = seq.block_table.last().unwrap() * self.config.block_size
                + seq.last_block_num_tokens()
                - 1;
            slot_mapping.push(slot as i64);
        }

        // Create tensors
        let length = positions.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), &self.device)?;
        let positions = Tensor::from_vec(positions, (length,), &self.device)?;
        let s_len = slot_mapping.len();
        let c_len = context_lens.len();

        let slot_mapping = Tensor::from_vec(slot_mapping, (s_len,), &self.device)?;
        let context_lens = Tensor::from_vec(context_lens, (c_len,), &self.device)?;
        let block_tables = self.prepare_block_tables(seqs)?;

        let input_metadata = InputMetadata {
            is_prefill: false,
            slot_mapping,
            block_tables: Some(block_tables),
            context_lens: Some(context_lens),
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            max_context_len: self.config.max_model_len.unwrap_or(4096),
            use_flash_attn: self.use_flash_attn,
        };

        Ok((input_ids, positions, input_metadata))
    }

    fn sample(&self, logits: &Tensor) -> Result<Vec<u32>> {
        logits.argmax(D::Minus1)?.to_vec1::<u32>()
    }

    pub fn get_model_vocab_size(&self) -> usize {
        match &self.model {
            Model::Qwen3(model) => model.get_vocab_size(),
            Model::LLaMa(model) => model.get_vocab_size(),
        }
    }

    #[cfg(all(feature = "cuda", feature = "graph"))]
    pub fn warmup_capture(&mut self) -> Result<()> {
        let kv_cache_lock = self.kv_cache.lock().unwrap(); // no custom method call on `self`
        self.capturer
            .capture(&self.device, Some(&kv_cache_lock), self.use_flash_attn)
    }
}
