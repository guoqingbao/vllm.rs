// src/models/mtp.rs
// Common MTP utility module for Qwen3.5 and Qwen3.5-MoE models
// MTP (Multi-Token Prediction) is an optional speculative decoding path
// that only activates when CLI flag is passed

use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::{FusedMoe, FusedMoeFp8, FusedMoeGGUF, FusedMoeISQ};
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use attention_rs::InputMetadata;
use candle_core::{DType, Result, Tensor, D};
use candle_nn::Module;
use either::Either;
use std::rc::Rc;
use std::sync::Arc;

// ============================================================================
// MTP Decoder Layer (full_attention only)
// ============================================================================

/// MTP decoder layer for Qwen3.5 dense model
pub struct Qwen3_5MTPDecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
}

impl Qwen3_5MTPDecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            vb.pp("self_attn").clone(),
            comm.clone(),
            config,
            None,
            config.sliding_window,
            dtype,
        )?;

        let mlp = MLP::new(
            vb.pp("mlp").clone(),
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

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm").clone(),
            DType::F32,
            false,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm").clone(),
            DType::F32,
            false,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb,
        })
    }

    pub fn forward(
        &self,
        positions: &Tensor,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        input_metadata: &InputMetadata,
    ) -> Result<(Tensor, Tensor)> {
        let residual = residual.unwrap_or(hidden_states);
        let hidden_states = self.input_layernorm.forward(hidden_states)?;

        let rope: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.clone();
        let attn_output = self.self_attn.forward(
            &hidden_states,
            &Some(rope),
            None,
            positions,
            None,
            input_metadata,
        )?;

        let hidden_states = (attn_output + residual)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;

        Ok(((residual + mlp_output)?, hidden_states))
    }
}

// ============================================================================
// MTP Decoder Layer for MoE (full_attention only)
// ============================================================================

/// MoE or MLP dispatch for MTP decoder layer
enum MoeOrMlp {
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeGGUF(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeFp8(m) => m.forward(xs, is_prefill),
        }
    }
}

/// MTP decoder layer for Qwen3.5 MoE model
pub struct Qwen3_5MoEMTPDecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    shared_gate: Option<crate::models::layers::linear::LinearX>,
    shared_expert: Option<MLP>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
}

impl Qwen3_5MoEMTPDecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            vb.pp("self_attn").clone(),
            comm.clone(),
            config,
            None,
            config.sliding_window,
            dtype,
        )?;

        let moe_cfg = config
            .moe_cfg
            .as_ref()
            .expect("MoE config required for MTP MoE decoder");

        // Determine MoE variant based on builder type
        let mlp = if vb.is_qvar_builder() {
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
                panic!("Unsupported quant method for MoE MTP (use unquantized, gguf or fp8)!");
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
        };

        // Shared experts (Qwen2 MoE style)
        let (shared_gate, shared_expert) = if let Some(intermediate_size) =
            moe_cfg.shared_expert_intermediate_size
        {
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

                let shared_gate = crate::models::layers::linear::LinearX::new(ws, None, &None)?;
                let mlp = MLP::new(
                    vb.pp("mlp.shared_expert").clone(),
                    comm.clone(),
                    config.hidden_size,
                    intermediate_size,
                    &config.hidden_act,
                    &config.quantization_config,
                    &config.quant,
                    false,
                    dtype,
                    "",
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
            vb.pp("input_layernorm").clone(),
            DType::F32,
            false,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm").clone(),
            DType::F32,
            false,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            shared_gate,
            shared_expert,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb,
        })
    }

    pub fn forward(
        &self,
        positions: &Tensor,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        input_metadata: &InputMetadata,
    ) -> Result<(Tensor, Tensor)> {
        let residual = residual.unwrap_or(hidden_states);
        let hidden_states = self.input_layernorm.forward(hidden_states)?;

        let rope: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.clone();
        let attn_output = self.self_attn.forward(
            &hidden_states,
            &Some(rope),
            None,
            positions,
            None,
            input_metadata,
        )?;

        let hidden_states = (attn_output + residual)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        // Shared experts
        let shared_output = match (&self.shared_gate, &self.shared_expert) {
            (Some(shared_gate), Some(shared_expert)) => {
                let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&hidden_states)?)?;
                let shared_output = shared_expert.forward(&hidden_states)?;
                Some(gate.broadcast_mul(&shared_output)?)
            }
            _ => None,
        };

        let mlp_output = self.mlp.forward(&hidden_states, input_metadata.is_prefill)?;
        let hidden_states_val = match shared_output {
            Some(shared) => (mlp_output + shared)?,
            None => mlp_output,
        };

        Ok(((residual + hidden_states_val)?, hidden_states))
    }
}

// ============================================================================
// MTP Predictor Base Structure
// ============================================================================

/// Common MTP predictor for Qwen3.5 dense model
pub struct Qwen3_5MultiTokenPredictor {
    embed_tokens: candle_nn::Embedding,
    fc: ReplicatedLinear,
    layers: Vec<Qwen3_5MTPDecoderLayer>,
    norm: NormX,
    pre_fc_norm_hidden: NormX,
    pre_fc_norm_embedding: NormX,
    num_mtp_layers: usize,
}

impl Qwen3_5MultiTokenPredictor {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        // Qwen3Next uses num_nextn_predict_layers, Qwen3.5 uses mtp_num_hidden_layers
        let num_mtp_layers = config.num_nextn_predict_layers
            .or(config.mtp_num_hidden_layers)
            .unwrap_or(1);

        let (embed_tokens, _vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens").clone(),
            dtype,
        )?;

        let fc = ReplicatedLinear::load_no_bias(
            config.hidden_size * 2, // [embed; hidden] concat
            config.hidden_size,
            vb.pp("fc").clone(),
            &None,
            &None,
            dtype,
        )?;

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            dtype,
            config,
            &vb.device(),
            false,
            config.rope_theta,
        )?);

        let mut layers = Vec::new();
        for i in 0..num_mtp_layers {
            let layer_vb = vb.pp(format!("layers.{}", i).as_str());
            layers.push(Qwen3_5MTPDecoderLayer::new(
                layer_vb,
                comm.clone(),
                Arc::clone(&rotary_emb),
                config,
                dtype,
            )?);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("norm").clone(),
            DType::F32,
            false,
        )?;

        let pre_fc_norm_hidden = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("pre_fc_norm_hidden").clone(),
            DType::F32,
            false,
        )?;

        let pre_fc_norm_embedding = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("pre_fc_norm_embedding").clone(),
            DType::F32,
            false,
        )?;

        Ok(Self {
            embed_tokens,
            fc,
            layers,
            norm,
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            num_mtp_layers,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        hidden_states: &Tensor,
        input_metadata: &InputMetadata,
        spec_step_idx: usize,
    ) -> Result<Tensor> {
        // 1. Embed input_ids
        let embedded = self.embed_tokens.forward(input_ids)?;

        // 2. Normalize before FC
        let embedded = self.pre_fc_norm_embedding.forward(&embedded)?;
        let hidden_states = self.pre_fc_norm_hidden.forward(&hidden_states)?;

        // 3. Concatenate [embedded; hidden_states]
        let hidden_states = Tensor::cat(&[embedded, hidden_states], D::Minus(1))?;

        // 4. FC projection
        let hidden_states = self.fc.forward(&hidden_states)?;

        // 5. Cycle through MTP layers
        let layer_idx = spec_step_idx % self.num_mtp_layers;
        let (hidden_states, _residual) = self.layers[layer_idx].forward(
            positions,
            &hidden_states,
            None,
            input_metadata,
        )?;

        // 6. Final normalization - only hidden_states, residual is discarded
        let hidden_states = self.norm.forward(&hidden_states)?;

        Ok(hidden_states)
    }
}

/// Common MTP predictor for Qwen3.5 MoE model
pub struct Qwen3_5MoEMultiTokenPredictor {
    embed_tokens: candle_nn::Embedding,
    fc: ReplicatedLinear,
    layers: Vec<Qwen3_5MoEMTPDecoderLayer>,
    norm: NormX,
    pre_fc_norm_hidden: NormX,
    pre_fc_norm_embedding: NormX,
    num_mtp_layers: usize,
}

impl Qwen3_5MoEMultiTokenPredictor {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        // Qwen3Next uses num_nextn_predict_layers, Qwen3.5 uses mtp_num_hidden_layers
        let num_mtp_layers = config.num_nextn_predict_layers
            .or(config.mtp_num_hidden_layers)
            .unwrap_or(1);

        let (embed_tokens, _vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens").clone(),
            dtype,
        )?;

        let fc = ReplicatedLinear::load_no_bias(
            config.hidden_size * 2, // [embed; hidden] concat
            config.hidden_size,
            vb.pp("fc").clone(),
            &None,
            &None,
            dtype,
        )?;

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            dtype,
            config,
            &vb.device(),
            false,
            config.rope_theta,
        )?);

        let mut layers = Vec::new();
        for i in 0..num_mtp_layers {
            let layer_vb = vb.pp(format!("layers.{}", i).as_str());
            layers.push(Qwen3_5MoEMTPDecoderLayer::new(
                layer_vb,
                comm.clone(),
                Arc::clone(&rotary_emb),
                config,
                dtype,
            )?);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("norm").clone(),
            DType::F32,
            false,
        )?;

        let pre_fc_norm_hidden = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("pre_fc_norm_hidden").clone(),
            DType::F32,
            false,
        )?;

        let pre_fc_norm_embedding = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("pre_fc_norm_embedding").clone(),
            DType::F32,
            false,
        )?;

        Ok(Self {
            embed_tokens,
            fc,
            layers,
            norm,
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            num_mtp_layers,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        hidden_states: &Tensor,
        input_metadata: &InputMetadata,
        spec_step_idx: usize,
    ) -> Result<Tensor> {
        // 1. Embed input_ids
        let embedded = self.embed_tokens.forward(input_ids)?;

        // 2. Normalize before FC
        let embedded = self.pre_fc_norm_embedding.forward(&embedded)?;
        let hidden_states = self.pre_fc_norm_hidden.forward(&hidden_states)?;

        // 3. Concatenate [embedded; hidden_states]
        let hidden_states = Tensor::cat(&[embedded, hidden_states], D::Minus(1))?;

        // 4. FC projection
        let hidden_states = self.fc.forward(&hidden_states)?;

        // 5. Cycle through MTP layers
        let layer_idx = spec_step_idx % self.num_mtp_layers;
        let (hidden_states, _residual) = self.layers[layer_idx].forward(
            positions,
            &hidden_states,
            None,
            input_metadata,
        )?;

        // 6. Final normalization - only hidden_states, residual is discarded
        let hidden_states = self.norm.forward(&hidden_states)?;

        Ok(hidden_states)
    }
}