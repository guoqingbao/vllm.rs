// src/models/mtp.rs
// Multi-Token Prediction (MTP) module for Qwen3.5 and Qwen3.5-MoE models.
//
// MTP uses lightweight predictor heads (shared embedding + FC + one decoder layer)
// to draft speculative tokens during decode. The main model then verifies drafts
// in a single batched forward pass, accepting tokens greedily until the first
// mismatch. This amortises the per-token latency of autoregressive decoding.

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
// MTP Layer Discovery Utilities
// ============================================================================

pub fn discover_mtp_layers(tensor_keys: &[String], prefix: &str) -> Vec<usize> {
    let mut layers = std::collections::HashSet::new();
    let layer_prefix = format!("{}.layers.", prefix.trim_end_matches('.'));

    for key in tensor_keys {
        if key.starts_with(&layer_prefix) {
            let rest = &key[layer_prefix.len()..];
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(layer_idx) = rest[..dot_pos].parse::<usize>() {
                    layers.insert(layer_idx);
                }
            }
        }
    }

    let mut layers_vec: Vec<usize> = layers.into_iter().collect();
    layers_vec.sort();
    layers_vec
}

pub fn count_mtp_layers(tensor_keys: &[String], prefix: &str) -> usize {
    discover_mtp_layers(tensor_keys, prefix).len()
}

pub fn get_mtp_layer_info(tensor_keys: &[String], prefix: &str) -> Vec<usize> {
    let layers = discover_mtp_layers(tensor_keys, prefix);
    crate::log_info!("Discovered MTP layers from '{}': {:?}", prefix, layers);
    layers
}

// ============================================================================
// MTP Decoder Layer (dense, full_attention only — no KV cache)
// ============================================================================

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
// MTP Decoder Layer for MoE (full_attention only — no KV cache)
// ============================================================================

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

        let shared_output = match (&self.shared_gate, &self.shared_expert) {
            (Some(shared_gate), Some(shared_expert)) => {
                let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&hidden_states)?)?;
                let shared_output = shared_expert.forward(&hidden_states)?;
                Some(gate.broadcast_mul(&shared_output)?)
            }
            _ => None,
        };

        let mlp_output = self
            .mlp
            .forward(&hidden_states, input_metadata.is_prefill)?;
        let hidden_states_val = match shared_output {
            Some(shared) => (mlp_output + shared)?,
            None => mlp_output,
        };

        Ok(((residual + hidden_states_val)?, hidden_states))
    }
}

// ============================================================================
// MTP Predictor — Dense variant
// ============================================================================

pub struct Qwen3_5MultiTokenPredictor {
    embed_tokens: candle_nn::Embedding,
    fc: ReplicatedLinear,
    layers: Vec<Qwen3_5MTPDecoderLayer>,
    norm: NormX,
    pre_fc_norm_hidden: NormX,
    pre_fc_norm_embedding: NormX,
    pub num_mtp_layers: usize,
}

impl Qwen3_5MultiTokenPredictor {
    pub fn new(vb: VarBuilderX, comm: Rc<Comm>, config: &Config, dtype: DType) -> Result<Self> {
        let config_num_layers = config
            .num_nextn_predict_layers
            .or(config.mtp_num_hidden_layers);

        let tensor_keys = vb.get_all_tensor_keys();
        let prefix = vb.module_path();
        let mtp_prefix = if prefix.is_empty() { "mtp" } else { &prefix };

        let discovered_layers = get_mtp_layer_info(&tensor_keys, mtp_prefix);
        let dynamic_num_layers = discovered_layers.len().max(1);
        let num_mtp_layers = config_num_layers.unwrap_or(dynamic_num_layers);

        if let Some(config_val) = config_num_layers {
            if config_val != dynamic_num_layers && !tensor_keys.is_empty() {
                crate::log_warn!(
                    "MTP config specifies {} layers but {} were found in weight file",
                    config_val,
                    dynamic_num_layers
                );
            }
        }

        crate::log_info!(
            "Initializing MTP predictor with {} layers (prefix: {}, config: {:?})",
            num_mtp_layers,
            mtp_prefix,
            config_num_layers
        );

        let (embed_tokens, _vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens").clone(),
            dtype,
        )?;

        let fc = ReplicatedLinear::load_no_bias(
            config.hidden_size * 2,
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

    /// Run one MTP prediction step.
    /// `input_ids`: the token(s) whose embedding is combined with `hidden_states`.
    /// `hidden_states`: last hidden state from the main model (or previous MTP step).
    /// `spec_step_idx`: which MTP layer to use (cycled modulo `num_mtp_layers`).
    /// Returns the predicted hidden state (before lm_head projection).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        hidden_states: &Tensor,
        input_metadata: &InputMetadata,
        spec_step_idx: usize,
    ) -> Result<Tensor> {
        let embedded = self.embed_tokens.forward(input_ids)?;

        let embedded = self.pre_fc_norm_embedding.forward(&embedded)?;
        let hidden_states = self.pre_fc_norm_hidden.forward(hidden_states)?;

        let hidden_states = Tensor::cat(&[embedded, hidden_states], D::Minus1)?;
        let hidden_states = self.fc.forward(&hidden_states)?;

        let layer_idx = spec_step_idx % self.num_mtp_layers;
        let (hidden_states, _residual) =
            self.layers[layer_idx].forward(positions, &hidden_states, None, input_metadata)?;

        self.norm.forward(&hidden_states)
    }
}

// ============================================================================
// MTP Predictor — MoE variant
// ============================================================================

pub struct Qwen3_5MoEMultiTokenPredictor {
    embed_tokens: candle_nn::Embedding,
    fc: ReplicatedLinear,
    layers: Vec<Qwen3_5MoEMTPDecoderLayer>,
    norm: NormX,
    pre_fc_norm_hidden: NormX,
    pre_fc_norm_embedding: NormX,
    pub num_mtp_layers: usize,
}

impl Qwen3_5MoEMultiTokenPredictor {
    pub fn new(vb: VarBuilderX, comm: Rc<Comm>, config: &Config, dtype: DType) -> Result<Self> {
        let config_num_layers = config
            .num_nextn_predict_layers
            .or(config.mtp_num_hidden_layers);

        let tensor_keys = vb.get_all_tensor_keys();
        let prefix = vb.module_path();
        let mtp_prefix = if prefix.is_empty() { "mtp" } else { &prefix };

        let discovered_layers = get_mtp_layer_info(&tensor_keys, mtp_prefix);
        let dynamic_num_layers = discovered_layers.len().max(1);
        let num_mtp_layers = config_num_layers.unwrap_or(dynamic_num_layers);

        if let Some(config_val) = config_num_layers {
            if config_val != dynamic_num_layers && !tensor_keys.is_empty() {
                crate::log_warn!(
                    "MTP config specifies {} layers but {} were found in weight file",
                    config_val,
                    dynamic_num_layers
                );
            }
        }

        crate::log_info!(
            "Initializing MTP MoE predictor with {} layers (prefix: {}, config: {:?})",
            num_mtp_layers,
            mtp_prefix,
            config_num_layers
        );

        let (embed_tokens, _vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens").clone(),
            dtype,
        )?;

        let fc = ReplicatedLinear::load_no_bias(
            config.hidden_size * 2,
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
        let embedded = self.embed_tokens.forward(input_ids)?;

        let embedded = self.pre_fc_norm_embedding.forward(&embedded)?;
        let hidden_states = self.pre_fc_norm_hidden.forward(hidden_states)?;

        let hidden_states = Tensor::cat(&[embedded, hidden_states], D::Minus1)?;
        let hidden_states = self.fc.forward(&hidden_states)?;

        let layer_idx = spec_step_idx % self.num_mtp_layers;
        let (hidden_states, _residual) =
            self.layers[layer_idx].forward(positions, &hidden_states, None, input_metadata)?;

        self.norm.forward(&hidden_states)
    }
}

// ============================================================================
// MTP Speculative Decoding Orchestrator
// ============================================================================

/// Result of one MTP speculative decoding round for a single sequence.
#[derive(Debug, Clone)]
pub struct MtpAcceptResult {
    /// Accepted token IDs (may be empty if draft was rejected at step 0).
    pub accepted_tokens: Vec<u32>,
    /// The bonus token from the target model at the first rejection point.
    pub bonus_token: u32,
}

/// Greedy verification: compare draft tokens against target model logits.
/// Returns the number of accepted draft tokens and the corrected token at
/// the first mismatch (or the next-token from target if all drafts accepted).
pub fn verify_draft_tokens_greedy(
    draft_tokens: &[u32],
    target_logits: &Tensor,
) -> Result<MtpAcceptResult> {
    let target_logits_f32 = target_logits.to_dtype(DType::F32)?;
    let num_positions = target_logits_f32.dim(0)?;

    let mut accepted_tokens = Vec::new();

    for i in 0..draft_tokens.len().min(num_positions.saturating_sub(1)) {
        let row = target_logits_f32.get(i)?;
        let target_token = row.argmax(D::Minus1)?.to_scalar::<u32>()?;

        if target_token == draft_tokens[i] {
            accepted_tokens.push(target_token);
        } else {
            return Ok(MtpAcceptResult {
                accepted_tokens,
                bonus_token: target_token,
            });
        }
    }

    let last_idx = draft_tokens.len().min(num_positions.saturating_sub(1));
    let last_row = if last_idx < num_positions {
        target_logits_f32.get(last_idx)?
    } else {
        target_logits_f32.get(num_positions - 1)?
    };
    let bonus_token = last_row.argmax(D::Minus1)?.to_scalar::<u32>()?;

    Ok(MtpAcceptResult {
        accepted_tokens,
        bonus_token,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn make_logits(argmax_tokens: &[u32], vocab_size: usize) -> Tensor {
        let device = Device::Cpu;
        let num_positions = argmax_tokens.len();
        let mut data = vec![0.0f32; num_positions * vocab_size];
        for (i, &tok) in argmax_tokens.iter().enumerate() {
            data[i * vocab_size + tok as usize] = 10.0;
        }
        Tensor::from_vec(data, (num_positions, vocab_size), &device).unwrap()
    }

    #[test]
    fn test_verify_all_accepted() {
        let draft = vec![10, 20, 30];
        let logits = make_logits(&[10, 20, 30, 99], 100);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert_eq!(result.accepted_tokens, vec![10, 20, 30]);
        assert_eq!(result.bonus_token, 99);
    }

    #[test]
    fn test_verify_none_accepted() {
        let draft = vec![10, 20, 30];
        let logits = make_logits(&[99, 20, 30, 50], 100);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.bonus_token, 99);
    }

    #[test]
    fn test_verify_partial_accept() {
        let draft = vec![10, 20, 30];
        let logits = make_logits(&[10, 20, 77, 50], 100);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert_eq!(result.accepted_tokens, vec![10, 20]);
        assert_eq!(result.bonus_token, 77);
    }

    #[test]
    fn test_verify_single_draft() {
        let draft = vec![42];
        let logits = make_logits(&[42, 55], 100);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert_eq!(result.accepted_tokens, vec![42]);
        assert_eq!(result.bonus_token, 55);
    }

    #[test]
    fn test_verify_single_draft_rejected() {
        let draft = vec![42];
        let logits = make_logits(&[99, 55], 100);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.bonus_token, 99);
    }

    #[test]
    fn test_verify_empty_draft() {
        let draft: Vec<u32> = vec![];
        let logits = make_logits(&[42], 100);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.bonus_token, 42);
    }

    #[test]
    fn test_verify_more_drafts_than_logit_positions() {
        let draft = vec![10, 20, 30, 40, 50];
        let logits = make_logits(&[10, 20, 77], 100);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert_eq!(result.accepted_tokens, vec![10, 20]);
        assert_eq!(result.bonus_token, 77);
    }

    #[test]
    fn test_verify_large_vocab() {
        let draft = vec![50000, 60000, 70000];
        let logits = make_logits(&[50000, 60000, 70000, 80000], 100000);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert_eq!(result.accepted_tokens, vec![50000, 60000, 70000]);
        assert_eq!(result.bonus_token, 80000);
    }

    #[test]
    fn test_verify_reject_at_first_position() {
        let draft = vec![1, 2, 3, 4, 5];
        let logits = make_logits(&[0, 2, 3, 4, 5, 6], 10);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.bonus_token, 0);
    }

    #[test]
    fn test_verify_reject_at_last_position() {
        let draft = vec![1, 2, 3];
        let logits = make_logits(&[1, 2, 9, 7], 10);
        let result = verify_draft_tokens_greedy(&draft, &logits).unwrap();
        assert_eq!(result.accepted_tokens, vec![1, 2]);
        assert_eq!(result.bonus_token, 9);
    }

    #[test]
    fn test_discover_mtp_layers_basic() {
        let keys = vec![
            "mtp.layers.0.self_attn.q_proj.weight".to_string(),
            "mtp.layers.0.self_attn.k_proj.weight".to_string(),
            "mtp.layers.1.self_attn.q_proj.weight".to_string(),
            "mtp.norm.weight".to_string(),
        ];
        let layers = discover_mtp_layers(&keys, "mtp");
        assert_eq!(layers, vec![0, 1]);
    }

    #[test]
    fn test_discover_mtp_layers_empty() {
        let keys: Vec<String> = vec![];
        let layers = discover_mtp_layers(&keys, "mtp");
        assert!(layers.is_empty());
    }

    #[test]
    fn test_discover_mtp_layers_no_match() {
        let keys = vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.1.self_attn.q_proj.weight".to_string(),
        ];
        let layers = discover_mtp_layers(&keys, "mtp");
        assert!(layers.is_empty());
    }

    #[test]
    fn test_discover_mtp_layers_with_prefix() {
        let keys = vec![
            "model.mtp.layers.0.mlp.gate.weight".to_string(),
            "model.mtp.layers.0.mlp.experts.0.weight".to_string(),
            "model.mtp.layers.2.self_attn.q_proj.weight".to_string(),
        ];
        let layers = discover_mtp_layers(&keys, "model.mtp");
        assert_eq!(layers, vec![0, 2]);
    }

    #[test]
    fn test_count_mtp_layers() {
        let keys = vec![
            "mtp.layers.0.self_attn.q_proj.weight".to_string(),
            "mtp.layers.0.mlp.gate.weight".to_string(),
            "mtp.layers.1.self_attn.q_proj.weight".to_string(),
            "mtp.layers.2.self_attn.q_proj.weight".to_string(),
        ];
        assert_eq!(count_mtp_layers(&keys, "mtp"), 3);
    }
}
