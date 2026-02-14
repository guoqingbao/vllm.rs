// src/models/qwen3_5_moe.rs
// Qwen3.5 MoE variant with hybrid attention (full attention + GatedDeltaNet layers)
use crate::models::layers::deltanet::GatedDeltaNet;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::linear::LinearX as Linear;
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::{FusedMoe, FusedMoeFp8, FusedMoeGGUF, FusedMoeISQ};
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::models::qwen3_5_attention::Qwen3_5Attention;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use crate::utils::resolve_qwen3_hybrid_config;
use attention_rs::mamba_cache::MambaCache;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use either::Either;
use parking_lot::{RwLock, RwLockWriteGuard};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

// =============================================================================
// MoE or MLP dispatch (reused from qwen3_moe pattern)
// =============================================================================

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
    Mlp(MLP),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeGGUF(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeFp8(m) => m.forward(xs, is_prefill),
        }
    }
}

// =============================================================================
// Attention type: full attention or GatedDeltaNet
// =============================================================================

enum Qwen3_5MoEAttnType {
    FullAttention(Qwen3_5Attention),
    LinearAttention(GatedDeltaNet),
}

// =============================================================================
// Decoder layer: hybrid attention + MoE/MLP
// =============================================================================

pub struct Qwen3_5MoEDecoderLayer {
    attn: Qwen3_5MoEAttnType,
    mlp: MoeOrMlp,
    shared_gate: Option<Linear>,
    shared_expert: Option<MLP>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Option<Arc<ScalingRotaryEmbedding>>,
}

impl Qwen3_5MoEDecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        layer_type: &str,
        layer_idx: usize,
        gdn_layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        // Attention dispatch
        let attn = if layer_type == "full_attention" {
            Qwen3_5MoEAttnType::FullAttention(Qwen3_5Attention::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("self_attn").clone()
                },
                comm.clone(),
                config,
                None,
                config.sliding_window,
                dtype,
            )?)
        } else {
            Qwen3_5MoEAttnType::LinearAttention(GatedDeltaNet::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("linear_attn").clone()
                },
                comm.clone(),
                config,
                gdn_layer_idx,
                dtype,
            )?)
        };

        // MoE or MLP dispatch
        let moe_cfg = config
            .moe_cfg
            .as_ref()
            .expect("MoE config is not available!");
        let mlp = if !moe_cfg
            .mlp_only_layers
            .as_ref()
            .unwrap_or(&Vec::<usize>::new())
            .contains(&layer_idx)
            && (moe_cfg.num_experts.unwrap_or(0) > 0
                && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap_or(1) == 0)
        {
            if is_qvar_builder {
                MoeOrMlp::FusedMoeGGUF(FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?)
            } else if let Some(quant_config) = &config.quantization_config {
                if quant_config.quant_method == "fp8" {
                    MoeOrMlp::FusedMoeFp8(FusedMoeFp8::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                        quant_config,
                    )?)
                } else {
                    panic!("Unsupported quant method for MoE (use unquantized, gguf or fp8)!");
                }
            } else if config.quant.is_some() {
                MoeOrMlp::FusedMoeISQ(FusedMoeISQ::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            } else {
                MoeOrMlp::FusedMoe(FusedMoe::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            }
        } else {
            MoeOrMlp::Mlp(MLP::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("mlp").clone()
                },
                comm.clone(),
                config.hidden_size,
                config.intermediate_size,
                &config.hidden_act,
                &config.quantization_config,
                &config.quant,
                false,
                dtype,
                "",
            )?)
        };

        // Shared experts (Qwen2 MoE style)
        let (shared_gate, shared_expert) =
            if let Some(intermediate_size) = moe_cfg.shared_expert_intermediate_size {
                if intermediate_size > 0 {
                    let ws = match &vb.0 {
                        Either::Left(vb) => vb
                            .pp("mlp.shared_expert_gate")
                            .get((1, config.hidden_size), "weight")?,
                        Either::Right(vb) => {
                            let ws = vb
                                .pp("ffn_gate_inp_shexp")
                                .get((config.hidden_size,), "weight")?;
                            ws.dequantize(&vb.device())?
                                .reshape((1, config.hidden_size))?
                        }
                    }
                    .to_dtype(dtype)?;
                    let shared_gate = Linear::new(ws, None, &None)?;
                    let mlp = MLP::new(
                        if is_qvar_builder {
                            vb.clone()
                        } else {
                            vb.pp("mlp.shared_expert").clone()
                        },
                        comm.clone(),
                        config.hidden_size,
                        intermediate_size,
                        &config.hidden_act,
                        &config.quantization_config,
                        &config.quant,
                        false,
                        dtype,
                        if is_qvar_builder { "_shexp" } else { "" },
                    )?;
                    (Some(shared_gate), Some(mlp))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("attn_norm")
            } else {
                vb.pp("input_layernorm")
            },
            DType::F32,
            true,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("ffn_norm")
            } else {
                vb.pp("post_attention_layernorm")
            },
            DType::F32,
            true,
        )?;

        let rotary = if layer_type == "full_attention" {
            Some(rotary_emb)
        } else {
            None
        };

        Ok(Self {
            attn,
            mlp,
            shared_gate,
            shared_expert,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb: rotary,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        mamba_cache: &mut MambaCache,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        let attn_output = match &self.attn {
            Qwen3_5MoEAttnType::FullAttention(attn) => {
                let rope: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.as_ref().unwrap().clone();
                attn.forward(
                    &xs,
                    &Some(rope),
                    attention_mask,
                    positions,
                    cache,
                    input_metadata,
                )?
            }
            Qwen3_5MoEAttnType::LinearAttention(gdn) => {
                gdn.forward(&xs, mamba_cache, input_metadata, seq_slots)?
            }
        };

        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;

        // Shared experts
        let shared_output = match (&self.shared_gate, &self.shared_expert) {
            (Some(shared_gate), Some(shared_expert)) => {
                let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&xs)?)?;
                let shared_output = shared_expert.forward(&xs)?;
                Some(gate.broadcast_mul(&shared_output)?)
            }
            _ => None,
        };

        let mlp_output = self.mlp.forward(&xs, input_metadata.is_prefill)?;
        if let Some(shared_output) = shared_output {
            residual + (mlp_output + shared_output)?
        } else {
            residual + mlp_output
        }
    }

    pub fn is_full_attention(&self) -> bool {
        matches!(&self.attn, Qwen3_5MoEAttnType::FullAttention(_))
    }
}

// =============================================================================
// Qwen3.5 MoE Causal LM
// =============================================================================

pub struct Qwen3_5MoEForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3_5MoEDecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    mamba_cache: RwLock<MambaCache>,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
}

impl Qwen3_5MoEForCausalLM {
    fn resolve_seq_slots(input_metadata: &InputMetadata, token_count: usize) -> Result<Tensor> {
        if let Some(slot_mapping) = &input_metadata.mamba_slot_mapping {
            if slot_mapping.dtype() != DType::I64 {
                candle_core::bail!(
                    "Qwen3.5 MoE expects mamba_slot_mapping dtype I64, got {:?}",
                    slot_mapping.dtype()
                )
            }
            let slot_count = slot_mapping.dim(0)?;
            if slot_count == 0 {
                candle_core::bail!("Qwen3.5 MoE received empty mamba_slot_mapping")
            }
            if !input_metadata.is_prefill && slot_count != token_count {
                candle_core::bail!(
                    "Qwen3.5 MoE decode mamba_slot_mapping length mismatch: slots={} tokens={}",
                    slot_count,
                    token_count
                )
            }
            return Ok(slot_mapping.clone());
        }
        candle_core::bail!("Qwen3.5 MoE requires input_metadata.mamba_slot_mapping")
    }

    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        Self::new_with_prefix(
            vb,
            comm,
            config,
            dtype,
            is_rope_i,
            device,
            progress_reporter,
            None,
        )
    }

    pub fn new_with_prefix(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
        prefix: Option<String>,
    ) -> Result<Self> {
        let has_prefix = prefix.is_some();
        let mut prefix = prefix.unwrap_or("model.".to_string());
        let gguf_prefix = if has_prefix {
            prefix.clone()
        } else {
            "".to_string()
        };
        let key_map: HashMap<&str, &str> = [
            ("embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("norm", "output_norm"),
            ("layers", "blk"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();
        let reporter = progress_reporter.clone();
        let tie_word_embeddings = if !is_qvar_builder
            && vb.has_key("embed_tokens.weight")
            && !vb.has_key(&format!("{}embed_tokens.weight", prefix))
        {
            crate::log_error!("This model does not support decoding!");
            prefix.clear();
            Some(true)
        } else {
            config.tie_word_embeddings
        };

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
            } else {
                vb.pp(&format!("{}embed_tokens", prefix))
            },
            if is_qvar_builder || config.quant.is_some() {
                DType::F32
            } else {
                dtype
            },
        )?;

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder || config.quant.is_some() {
                DType::F32
            } else {
                dtype
            },
            config,
            &vb.device(),
            is_rope_i,
            config.rope_theta,
        )?);

        let hybrid = resolve_qwen3_hybrid_config(config);
        let layer_types = &hybrid.layer_types;

        let mut layers = Vec::new();
        let mut gdn_layer_idx = 0usize;

        for i in 0..config.num_hidden_layers {
            let layer_type = &layer_types[i];
            let current_gdn_idx = if layer_type == "linear_attention" {
                let idx = gdn_layer_idx;
                gdn_layer_idx += 1;
                idx
            } else {
                0
            };

            let layer = Qwen3_5MoEDecoderLayer::new(
                vb.pp(format!(
                    "{}.{}",
                    if is_qvar_builder {
                        format!("{}{}", gguf_prefix, key_map["layers"])
                    } else {
                        format!("{}layers", prefix)
                    },
                    i
                )
                .as_str()),
                comm.clone(),
                rotary_emb.clone(),
                config,
                layer_type,
                i,
                current_gdn_idx,
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let num_gdn_layers = gdn_layer_idx;

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["norm"]))
            } else {
                vb.pp(&format!("{}norm", prefix))
            },
            DType::F32,
            true,
        )?;

        let lm_head = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
                } else {
                    vb.pp(&format!("{}embed_tokens", prefix))
                }
            } else {
                if is_qvar_builder {
                    vb.pp(key_map["lm_head"])
                } else {
                    vb.pp("lm_head")
                }
            },
            &None,
            &None,
            dtype,
        )?;

        // Initialize MambaCache
        let world_size = comm.world_size();
        let num_v_heads = hybrid.num_v_heads;
        let num_k_heads = hybrid.num_k_heads;
        if num_v_heads % world_size != 0 || num_k_heads % world_size != 0 {
            candle_core::bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                num_v_heads,
                num_k_heads,
                world_size
            );
        }
        let num_v_heads = num_v_heads / world_size;
        let num_k_heads = num_k_heads / world_size;
        let key_head_dim = hybrid.key_head_dim;
        let value_head_dim = hybrid.value_head_dim;
        let conv_kernel_size = hybrid.conv_kernel_size;
        let d_conv = num_k_heads * key_head_dim * 2 + num_v_heads * value_head_dim;
        // Start small and let runner preallocate to the final engine capacity.
        let max_batch_size = 1;

        let mamba_cache = if num_gdn_layers > 0 {
            MambaCache::new(
                num_gdn_layers,
                max_batch_size,
                d_conv,
                conv_kernel_size,
                num_v_heads,
                key_head_dim,
                value_head_dim,
                dtype,
                DType::F32,
                device,
            )?
        } else {
            MambaCache::new(0, 1, 1, 2, 1, 1, 1, dtype, DType::F32, device)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mamba_cache: RwLock::new(mamba_cache),
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            is_qvar_builder,
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(xs)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        visual_pos_masks: &Option<Tensor>,
        deepstack_visual_embeds: &Option<Vec<Tensor>>,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );

        let mut xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_tokens.forward(input_ids)?
        };

        let mut kv_cache_idx = 0usize;
        let mut mamba_cache = self.mamba_cache.write();
        let seq_slots = Self::resolve_seq_slots(input_metadata, xs.dim(0)?)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = if layer.is_full_attention() {
                if let Some(kv_caches) = kv_caches {
                    let c = &kv_caches[kv_cache_idx];
                    kv_cache_idx += 1;
                    Some((&c.0, &c.1))
                } else {
                    None
                }
            } else {
                None
            };

            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                positions,
                cache,
                input_metadata,
                &mut mamba_cache,
                &seq_slots,
            )?;

            if let (Some(pos_mask), Some(deepstacks)) = (visual_pos_masks, deepstack_visual_embeds)
            {
                use crate::models::layers::deepstack::ApplyDeepStack;
                if i < deepstacks.len() {
                    xs = xs.apply_deep_stack(pos_mask, &deepstacks[i])?;
                }
            }
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        if return_hidden {
            xs.to_dtype(DType::F32)
        } else if self.is_qvar_builder {
            self.lm_head.forward(&xs)
        } else {
            self.lm_head
                .forward(&xs.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            true,
        )
    }

    pub fn forward_with_deepstack(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        visual_pos_masks: &Option<Tensor>,
        deepstack_visual_embeds: &Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            visual_pos_masks,
            deepstack_visual_embeds,
            false,
        )
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn release_sequence_state(&self, sequence_id: usize) {
        self.mamba_cache.write().free_slot(sequence_id);
    }

    pub fn ensure_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .ensure_slots_for_sequences(sequence_ids)
    }

    pub fn get_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .get_slots_for_sequences(sequence_ids)
    }

    pub fn lock_mamba_cache_for_graph(&self) -> RwLockWriteGuard<'_, MambaCache> {
        self.mamba_cache.write()
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        self.mamba_cache.write().reserve_capacity(max_num_seqs)
    }

    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        self.mamba_cache.write().set_prefix_cache_capacity(capacity);
    }

    pub fn capture_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        self.mamba_cache.write().capture_prefix_state(seq_id, hash)
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        self.mamba_cache.read().has_prefix_state(hash)
    }

    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        self.mamba_cache.write().restore_prefix_state(seq_id, hash)
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        self.mamba_cache.write().reset_all()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
