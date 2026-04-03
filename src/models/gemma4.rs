use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::{FusedMoe, FusedMoeGGUF, FusedMoeISQ, FusedMoeMxfp4};
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{
    ApplyRotaryEmbedding, RotaryEmbedding, ScalingRotaryEmbedding,
};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Module, Result, Tensor};
use either::Either;
use parking_lot::RwLock;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

enum Gemma4MoE {
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeMxfp4(FusedMoeMxfp4),
}

impl Gemma4MoE {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeGGUF(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeMxfp4(m) => m.forward(xs, is_prefill),
        }
    }
}

pub struct Gemma4DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    moe: Option<Gemma4MoE>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    pre_feedforward_layernorm: NormX,
    post_feedforward_layernorm: NormX,
    post_feedforward_layernorm_1: Option<NormX>,
    post_feedforward_layernorm_2: Option<NormX>,
    pre_feedforward_layernorm_2: Option<NormX>,
    layer_scalar: Tensor,
    is_sliding: bool,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
}

impl Gemma4DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        config: &Config,
        _layer_idx: usize,
        is_sliding: bool,
        enable_moe: bool,
        global_head_dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        let swa_head_dim = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("swa_head_dim")
                .or_else(|| v.get("text_config").and_then(|tc| tc.get("head_dim")))
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize
        } else {
            256
        };

        let head_dim = if is_sliding {
            swa_head_dim
        } else {
            global_head_dim
        };

        let mut layer_config = config.clone();
        layer_config.head_dim = Some(head_dim);

        if !is_sliding {
            if let Some(extra) = &config.extra_config_json {
                let v: serde_json::Value =
                    serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
                if let Some(gkv) = v.get("num_global_key_value_heads").and_then(|v| v.as_u64()) {
                    layer_config.num_key_value_heads = gkv as usize;
                }
            }
        }

        let self_attn = Attention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            &layer_config,
            Some(1.0),
            sliding_window,
            dtype,
        )?;

        let mlp = MLP::new(
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
        )?;

        let moe = if enable_moe && config.moe_cfg.is_some() {
            let m = if is_qvar_builder {
                Gemma4MoE::FusedMoeGGUF(FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?)
            } else if let Some(quant_config) = &config.quantization_config {
                if quant_config.quant_method == "mxfp4" {
                    Gemma4MoE::FusedMoeMxfp4(FusedMoeMxfp4::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                } else {
                    panic!(
                        "Unsupported quantization for Gemma4 MoE: {}",
                        quant_config.quant_method
                    );
                }
            } else if config.quant.is_some() {
                Gemma4MoE::FusedMoeISQ(FusedMoeISQ::new_with_gate(
                    config,
                    vb.pp("router").pp("proj").clone(),
                    vb.pp("experts").clone(),
                    comm.clone(),
                    dtype,
                )?)
            } else {
                Gemma4MoE::FusedMoe(FusedMoe::new_with_gate(
                    config,
                    vb.pp("router").pp("proj").clone(),
                    vb.pp("experts").clone(),
                    comm.clone(),
                    dtype,
                )?)
            };
            Some(m)
        } else {
            None
        };

        let norm_dtype = if is_qvar_builder || config.quant.is_some() {
            DType::F32
        } else {
            dtype
        };

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
            norm_dtype,
            true,
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            norm_dtype,
            true,
        )?;
        let pre_feedforward_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
            norm_dtype,
            true,
        )?;
        let post_feedforward_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
            norm_dtype,
            true,
        )?;

        let (
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
        ) = if enable_moe && config.moe_cfg.is_some() {
            (
                Some(rms_norm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_1"),
                    norm_dtype,
                    true,
                )?),
                Some(rms_norm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_2"),
                    norm_dtype,
                    true,
                )?),
                Some(rms_norm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    vb.pp("pre_feedforward_layernorm_2"),
                    norm_dtype,
                    true,
                )?),
            )
        } else {
            (None, None, None)
        };

        let layer_scalar = match &vb.0 {
            Either::Left(v) => v.get(1, "layer_scalar")?.to_dtype(dtype)?,
            Either::Right(v) => v
                .pp("layer_output_scale")
                .get((1,), "weight")?
                .dequantize(&v.device())?
                .to_dtype(dtype)?,
        };

        Ok(Self {
            self_attn,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
            layer_scalar,
            is_sliding,
            rotary_emb,
            rotary_emb_local,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        sliding_mask: Option<&Vec<Tensor>>,
        full_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        let mask = if self.is_sliding {
            sliding_mask
        } else {
            full_mask
        };

        let rope: Arc<dyn ApplyRotaryEmbedding> = if self.is_sliding {
            self.rotary_emb_local.clone()
        } else {
            self.rotary_emb.clone()
        };

        let attn_output =
            self.self_attn
                .forward(&xs, &Some(rope), mask, positions, cache, input_metadata)?;

        let mut xs = self.post_attention_layernorm.forward(&attn_output)?;
        xs = (xs + residual)?;

        let residual = &xs;

        let mlp_input = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&mlp_input)?;

        let combined = if let Some(moe) = &self.moe {
            let mlp_normed = self
                .post_feedforward_layernorm_1
                .as_ref()
                .unwrap()
                .forward(&mlp_output)?;

            let residual_flat = residual.flatten(0, residual.rank() - 2)?;
            let moe_input = self
                .pre_feedforward_layernorm_2
                .as_ref()
                .unwrap()
                .forward(&residual_flat)?;
            let moe_output = moe.forward(&moe_input, input_metadata.is_prefill)?;
            let moe_output = moe_output.reshape(residual.shape())?;

            let moe_normed = self
                .post_feedforward_layernorm_2
                .as_ref()
                .unwrap()
                .forward(&moe_output)?;

            (mlp_normed + moe_normed)?
        } else {
            mlp_output
        };

        let combined = self.post_feedforward_layernorm.forward(&combined)?;
        let xs = (residual + combined)?;

        xs.broadcast_mul(&self.layer_scalar)
    }
}

pub struct Gemma4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Gemma4DecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    embed_scale: f64,
    is_qvar_builder: bool,
    #[allow(dead_code)]
    layer_types: Vec<String>,
}

impl Gemma4ForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let reporter = progress_reporter.clone();
        let is_qvar_builder = vb.is_qvar_builder();

        let layer_types: Vec<String> = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("layer_types")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|v| v.as_str().unwrap_or("sliding_attention").to_string())
                        .collect()
                })
                .unwrap_or_else(|| {
                    (0..config.num_hidden_layers)
                        .map(|i| {
                            if (i + 1) % 6 == 0 {
                                "full_attention".to_string()
                            } else {
                                "sliding_attention".to_string()
                            }
                        })
                        .collect()
                })
        } else {
            (0..config.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % 6 == 0 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect()
        };

        let enable_moe = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("enable_moe_block")
                .and_then(|v| v.as_bool())
                .unwrap_or(config.moe_cfg.is_some())
        } else {
            config.moe_cfg.is_some()
        };

        let global_head_dim = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("global_head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(config.head_dim.unwrap_or(256) as u64) as usize
        } else {
            config.head_dim.unwrap_or(256)
        };

        let rope_local_base_freq = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("rope_local_base_freq")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0)
        } else {
            10000.0
        };

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp("model.embed_tokens")
            } else {
                vb.pp("language_model.model.embed_tokens")
            },
            dtype,
        )?;

        let embed_scale = (config.hidden_size as f64).sqrt();

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

        let swa_head_dim_for_rope = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("swa_head_dim")
                .or_else(|| v.get("text_config").and_then(|tc| tc.get("head_dim")))
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize
        } else {
            256
        };

        let mut local_config = config.clone();
        local_config.head_dim = Some(swa_head_dim_for_rope);
        local_config.partial_rotary_factor = None;

        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            if is_qvar_builder || config.quant.is_some() {
                DType::F32
            } else {
                dtype
            },
            &local_config,
            &vb.device(),
            is_rope_i,
            Some(rope_local_base_freq),
            None,
            None,
        )?);

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let is_sliding = layer_types
                .get(i)
                .map(|t| t == "sliding_attention")
                .unwrap_or(true);
            let layer_prefix = if is_qvar_builder {
                format!("model.layers.{}", i)
            } else {
                format!("language_model.model.layers.{}", i)
            };
            let layer = Gemma4DecoderLayer::new(
                vb.pp(&layer_prefix),
                comm.clone(),
                rotary_emb.clone(),
                rotary_emb_local.clone(),
                config,
                i,
                is_sliding,
                enable_moe,
                global_head_dim,
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("model.norm")
            } else {
                vb.pp("language_model.model.norm")
            },
            if is_qvar_builder || config.quant.is_some() {
                DType::F32
            } else {
                dtype
            },
            true,
        )?;

        let tie_word_embeddings = config.tie_word_embeddings.unwrap_or(true);
        let lm_head = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if tie_word_embeddings {
                if is_qvar_builder {
                    vb.pp("model.embed_tokens")
                } else {
                    vb.pp("language_model.model.embed_tokens")
                }
            } else if is_qvar_builder {
                vb.pp("model.output")
            } else {
                vb.pp("lm_head")
            },
            &None,
            &None,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            embed_scale,
            is_qvar_builder,
            layer_types,
        })
    }

    fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let xs =
            if (self.is_qvar_builder || self.config.quant.is_some()) && xs.dtype() != DType::F32 {
                xs.to_dtype(DType::F32)?
            } else {
                xs
            };
        xs * self.embed_scale
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let mut xs = self.embed_forward(input_ids)?;

        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();

        let full_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            None,
            input_metadata.is_prefill,
        );

        let sliding_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), (_i, layer)) in
                zip(kv_caches.iter(), self.layers.iter().enumerate())
            {
                xs = layer.forward(
                    &xs,
                    sliding_mask.as_ref(),
                    full_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
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
        } else {
            let logits = if self.is_qvar_builder {
                self.lm_head.forward(&xs)?
            } else {
                self.lm_head.forward(&xs.to_dtype(self.dtype)?)?
            };

            let final_logit_softcapping = if let Some(extra) = &self.config.extra_config_json {
                let v: serde_json::Value =
                    serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
                v.get("final_logit_softcapping").and_then(|v| v.as_f64())
            } else {
                self.config.final_logit_softcapping
            };

            if let Some(cap) = final_logit_softcapping {
                let scaled = (logits / cap)?;
                let tanh = scaled.tanh()?;
                tanh * cap
            } else {
                Ok(logits)
            }
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        _embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        _embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, positions, kv_caches, input_metadata, true)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}
