#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::models::layers::distributed::{
    AllReduce, Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::others::{embedding, masked_fill, rms_norm, NormX};
use crate::models::layers::rotary_emb::YarnRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::{Config, EosTokenId, QuantConfig};
use crate::utils::progress::ProgressLike;
use attention_rs::ops::NonZeroOp;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Activation, Embedding, Module};
use parking_lot::RwLock;
use serde::Deserialize;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

type TokenID = usize;

fn routed_scaling_factor() -> f64 {
    1.0
}

fn topk_method() -> TopkMethod {
    TopkMethod::Greedy
}

fn moe_layer_freq() -> usize {
    1
}

fn first_k_dense_replace() -> usize {
    0
}

fn norm_topk_prob() -> bool {
    false
}

fn scoring_func() -> ScoringFunc {
    ScoringFunc::Softmax
}

fn hidden_act() -> Activation {
    Activation::Silu
}

fn tie_word_embeddings() -> bool {
    false
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
enum TopkMethod {
    Greedy,
    #[serde(rename = "noaux_tc")]
    NoAuxTc,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
enum ScoringFunc {
    Softmax,
    Sigmoid,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DeepSeekRopeScaling {
    Linear {
        factor: f32,
        #[serde(default)]
        scaling_type: Option<String>,
    },
    Dynamic {
        factor: f32,
        #[serde(default)]
        scaling_type: Option<String>,
    },
    Yarn {
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        factor: f32,
        mscale: f32,
        mscale_all_dim: f32,
        #[serde(default)]
        scaling_type: Option<String>,
    },
}

#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekConfig {
    pub architectures: Option<Vec<String>>,
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) n_shared_experts: Option<usize>,
    pub(crate) n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    pub(crate) routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    topk_method: TopkMethod,
    pub(crate) num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    pub(crate) moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    pub(crate) first_k_dense_replace: usize,
    #[serde(default = "norm_topk_prob")]
    pub(crate) norm_topk_prob: bool,
    #[serde(default = "scoring_func")]
    scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<DeepSeekRopeScaling>,
    pub(crate) q_lora_rank: Option<usize>,
    pub(crate) qk_rope_head_dim: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) v_head_dim: usize,
    pub(crate) qk_nope_head_dim: usize,
    pub(crate) n_group: usize,
    pub(crate) topk_group: usize,
    pub(crate) sliding_window: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
    pub bos_token_id: TokenID,
    pub eos_token_id: TokenID,
}

pub fn deepseek_config_to_common(cfg: &DeepSeekConfig) -> Config {
    let architectures = cfg
        .architectures
        .clone()
        .unwrap_or_else(|| vec!["DeepseekV3ForCausalLM".to_string()]);
    Config {
        architectures: Some(architectures),
        head_dim: Some(cfg.hidden_size / cfg.num_attention_heads),
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads),
        max_position_embeddings: cfg.max_position_embeddings,
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        max_model_len: None,
        intermediate_size: cfg.intermediate_size,
        rms_norm_eps: cfg.rms_norm_eps,
        vocab_size: Some(cfg.vocab_size),
        rope_theta: Some(cfg.rope_theta as f64),
        attention_bias: Some(false),
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        tie_word_embeddings: Some(cfg.tie_word_embeddings),
        bos_token_id: Some(cfg.bos_token_id),
        eos_token_id: Some(EosTokenId::Single(cfg.eos_token_id as u32)),
        use_sliding_window: None,
        sliding_window: cfg.sliding_window,
        max_window_layers: None,
        partial_rotary_factor: None,
        hidden_act: cfg.hidden_act.clone(),
        rope_scaling: None,
        quant: None,
        moe_cfg: None,
        fp8_kvcache: None,
        quantization_config: cfg.quantization_config.clone(),
        is_multi_model: None,
        extra_config_json: None,
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekV2RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

pub struct DeepSeekV2RopeConfig {
    pub rope_scaling: Option<DeepSeekRopeScaling>,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub qk_rope_head_dim: usize,
}

impl DeepSeekV2RotaryEmbedding {
    fn new_unscaled(cfg: &DeepSeekV2RopeConfig, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.qk_rope_head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), &Device::Cpu)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let sin = freqs.sin()?.to_device(dev)?;
        let cos = freqs.cos()?.to_device(dev)?;

        Ok(Self { cos, sin })
    }

    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1.)
    }

    pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        YarnRotaryEmbedding::yarn_get_mscale(scale, mscale)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        cfg: &DeepSeekV2RopeConfig,
        dev: &Device,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        factor: f32,
        mscale: f32,
        mscale_all_dim: f32,
    ) -> Result<Self> {
        let freq_extra: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, &Device::Cpu)?;
        let freq_inter: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), &Device::Cpu)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            original_max_position_embeddings,
        );
        let inv_freq_mask =
            (1. - Self::yarn_linear_ramp_mask(low, high, cfg.qk_rope_head_dim / 2, &Device::Cpu)?)?;
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale =
            Self::yarn_get_mscale(factor, mscale) / Self::yarn_get_mscale(factor, mscale_all_dim);
        let sin = (freqs.sin()? * mscale as f64)?.to_device(dev)?;
        let cos = (freqs.cos()? * mscale as f64)?.to_device(dev)?;

        Ok(Self { cos, sin })
    }

    pub fn new(cfg: &DeepSeekV2RopeConfig, dev: &Device) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(DeepSeekRopeScaling::Linear { .. } | DeepSeekRopeScaling::Dynamic { .. }) => {
                candle_core::bail!("linear and dynamic rope are not implemented yet!")
            }
            Some(DeepSeekRopeScaling::Yarn {
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                factor,
                mscale,
                mscale_all_dim,
                ..
            }) => Self::new_yarn(
                cfg,
                dev,
                *original_max_position_embeddings,
                *beta_fast,
                *beta_slow,
                *factor,
                *mscale,
                *mscale_all_dim,
            ),
            None => Self::new_unscaled(cfg, dev),
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(input_positions, 0)?;
        let sin = self.sin.index_select(input_positions, 0)?;
        let q_embed = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Clone, Debug)]
struct DeepSeekMoEConfig {
    num_experts_per_tok: Option<usize>,
    n_routed_experts: usize,
    moe_intermediate_size: usize,
    scoring_func: ScoringFunc,
    topk_method: TopkMethod,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    n_shared_experts: Option<usize>,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    first_k_dense_replace: usize,
    moe_layer_freq: usize,
    rope_scaling: Option<DeepSeekRopeScaling>,
    q_lora_rank: Option<usize>,
    n_group: usize,
    topk_group: usize,
}

impl DeepSeekMoEConfig {
    pub(crate) fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    fn softmax_scale(&self) -> f32 {
        let mut softmax_scale = 1.0 / (self.q_head_dim() as f32).sqrt();
        if let Some(DeepSeekRopeScaling::Yarn {
            mscale_all_dim,
            factor,
            ..
        }) = self.rope_scaling
        {
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, mscale_all_dim);
            softmax_scale = softmax_scale * mscale * mscale;
        }
        softmax_scale
    }
}

pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

trait TopKLastDimOp {
    fn topk(&self, topk: usize) -> Result<TopKOutput>;
    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        let sorted_indices = self.arg_sort_last_dim(false)?;
        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
        Ok(TopKOutput {
            values: self.gather(&topk_indices, D::Minus1)?,
            indices: topk_indices,
        })
    }

    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput> {
        let sorted_indices_all = self.arg_sort_last_dim(false)?;
        let topk_indices_sorted = sorted_indices_all
            .narrow(D::Minus1, 0, topk)?
            .contiguous()?;
        let topk_values_sorted = self.gather(&topk_indices_sorted, D::Minus1)?;

        let reorder_indices = topk_indices_sorted.arg_sort_last_dim(true)?;
        let topk_indices_unsorted = topk_indices_sorted.gather(&reorder_indices, D::Minus1)?;
        let topk_values_unsorted = topk_values_sorted.gather(&reorder_indices, D::Minus1)?;
        Ok(TopKOutput {
            values: topk_values_unsorted,
            indices: topk_indices_unsorted,
        })
    }
}

enum QProj {
    Plain(TensorParallelColumnLinear),
    Lora {
        a: ReplicatedLinear,
        norm: NormX,
        b: TensorParallelColumnLinear,
    },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(xs)?)?),
            Self::Plain(lin) => lin.forward(xs),
        }
    }
}

struct Attention {
    q: QProj,
    kv_a_proj_with_mqa: ReplicatedLinear,
    kv_a_layernorm: NormX,
    kv_b_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    q_head_dim: usize,
    num_attention_heads: usize,
    attn: PagedAttention,
    moe_cfg: DeepSeekMoEConfig,
}

impl Attention {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        moe_cfg: DeepSeekMoEConfig,
        dtype: DType,
    ) -> Result<Self> {
        let q_head_dim = moe_cfg.q_head_dim();
        let attention_bias = cfg.attention_bias.unwrap_or(false);
        let q = match moe_cfg.q_lora_rank {
            Some(lora_rank) => {
                let a = ReplicatedLinear::load_b(
                    cfg.hidden_size,
                    lora_rank,
                    attention_bias,
                    vb.pp("q_a_proj"),
                    &cfg.quantization_config,
                    &cfg.quant,
                    dtype,
                )?;
                let norm = rms_norm(
                    lora_rank,
                    cfg.rms_norm_eps,
                    vb.pp("q_a_layernorm"),
                    dtype,
                    false,
                )?;
                let b = TensorParallelColumnLinear::load_with_hints(
                    lora_rank,
                    cfg.num_attention_heads * q_head_dim,
                    false,
                    vb.pp("q_b_proj"),
                    comm.clone(),
                    &cfg.quantization_config,
                    &cfg.quant,
                    dtype,
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(TensorParallelColumnLinear::load_with_hints(
                cfg.hidden_size,
                cfg.num_attention_heads * q_head_dim,
                false,
                vb.pp("q_proj"),
                comm.clone(),
                &cfg.quantization_config,
                &cfg.quant,
                dtype,
            )?),
        };

        let kv_a_proj_with_mqa = ReplicatedLinear::load_b(
            cfg.hidden_size,
            moe_cfg.kv_lora_rank + moe_cfg.qk_rope_head_dim,
            attention_bias,
            vb.pp("kv_a_proj_with_mqa"),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;
        let kv_a_layernorm = rms_norm(
            moe_cfg.kv_lora_rank,
            cfg.rms_norm_eps,
            vb.pp("kv_a_layernorm"),
            dtype,
            false,
        )?;
        let kv_b_proj = TensorParallelColumnLinear::load_with_hints(
            moe_cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - moe_cfg.qk_rope_head_dim + moe_cfg.v_head_dim),
            false,
            vb.pp("kv_b_proj"),
            comm.clone(),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;

        let o_proj = TensorParallelRowLinear::load_with_hints(
            cfg.num_attention_heads * moe_cfg.v_head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
            comm.clone(),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;

        let num_attention_heads = cfg.num_attention_heads / comm.world_size();
        let num_kv_heads = cfg.num_key_value_heads / comm.world_size();

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            q_head_dim,
            num_attention_heads,
            attn: PagedAttention::new(
                num_attention_heads,
                moe_cfg.v_head_dim,
                moe_cfg.softmax_scale(),
                Some(num_kv_heads),
                cfg.sliding_window,
                vb.device().clone(),
                None,
                cfg.fp8_kvcache.unwrap_or(false),
            )?,
            moe_cfg,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;
        let q = self.q.forward(xs)?;
        let q = q.reshape((1, seq_len, self.num_attention_heads, self.q_head_dim))?;
        let q_nope = q
            .narrow(D::Minus1, 0, self.moe_cfg.qk_nope_head_dim)?
            .transpose(1, 2)?;
        let q_pe = q
            .narrow(
                D::Minus1,
                self.moe_cfg.qk_nope_head_dim,
                self.moe_cfg.qk_rope_head_dim,
            )?
            .transpose(1, 2)?
            .contiguous()?;

        let compressed_kv = self.kv_a_proj_with_mqa.forward(xs)?;
        let mut k_pe = compressed_kv
            .narrow(
                D::Minus1,
                self.moe_cfg.kv_lora_rank,
                self.moe_cfg.qk_rope_head_dim,
            )?
            .reshape((1, seq_len, 1, self.moe_cfg.qk_rope_head_dim))?
            .transpose(1, 2)?;
        let compressed_kv = compressed_kv
            .narrow(D::Minus1, 0, self.moe_cfg.kv_lora_rank)?
            .contiguous()?;

        let kv = self
            .kv_b_proj
            .forward(&self.kv_a_layernorm.forward(&compressed_kv)?)?;
        let kv = kv
            .reshape((
                1,
                seq_len,
                self.num_attention_heads,
                self.moe_cfg.qk_nope_head_dim + self.moe_cfg.v_head_dim,
            ))?
            .transpose(1, 2)?;

        let k_nope = kv
            .narrow(D::Minus1, 0, self.moe_cfg.qk_nope_head_dim)?
            .contiguous()?;
        let mut v = kv
            .narrow(
                D::Minus1,
                self.moe_cfg.qk_nope_head_dim,
                self.moe_cfg.v_head_dim,
            )?
            .contiguous()?;

        let (q_pe, k_pe_out) = self.rotary_emb.forward(
            &q_pe.to_dtype(DType::F32)?,
            &k_pe.to_dtype(DType::F32)?,
            input_positions,
        )?;
        k_pe = k_pe_out;
        let (q_pe, k_pe) = (q_pe.to_dtype(v.dtype())?, k_pe.to_dtype(v.dtype())?);

        let q = Tensor::cat(&[q_nope, q_pe], D::Minus1)?.contiguous()?;
        let k_pe = k_pe.repeat((1, q.dim(1)?, 1, 1))?;
        let k = Tensor::cat(&[k_nope, k_pe], D::Minus1)?.contiguous()?;

        if self.q_head_dim != self.moe_cfg.v_head_dim {
            v = v
                .pad_with_zeros(D::Minus1, 0, self.q_head_dim - self.moe_cfg.v_head_dim)?
                .contiguous()?;
        }

        let mut y = self.attn.forward(
            &q,
            &k,
            &v,
            attention_mask,
            cache.map(|(k_, _)| k_.clone()),
            cache.map(|(_, v_)| v_.clone()),
            input_metadata,
            None,
        )?;

        if self.q_head_dim != self.moe_cfg.v_head_dim {
            y = y.narrow(D::Minus1, 0, self.moe_cfg.v_head_dim)?;
        }

        let y = y.reshape((seq_len, ()))?;
        self.o_proj.forward(&y)
    }
}

struct Mlp {
    gate: ReplicatedLinear,
    up: ReplicatedLinear,
    down: ReplicatedLinear,
    act: Activation,
}

impl Mlp {
    fn new(
        cfg: &Config,
        vb: VarBuilderX,
        hidden_size: Option<usize>,
        intermediate_size: Option<usize>,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = hidden_size.unwrap_or(cfg.hidden_size);
        let intermediate_size = intermediate_size.unwrap_or(cfg.intermediate_size);

        let gate = ReplicatedLinear::load_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("gate_proj"),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;

        let up = ReplicatedLinear::load_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("up_proj"),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;

        let down = ReplicatedLinear::load_no_bias(
            intermediate_size,
            hidden_size,
            vb.pp("down_proj"),
            &cfg.quantization_config,
            &cfg.quant,
            dtype,
        )?;

        Ok(Self {
            gate,
            up,
            down,
            act: cfg.hidden_act.clone(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?.apply(&self.act)?;
        let rhs = self.up.forward(xs)?;
        self.down.forward(&(&lhs * &rhs)?)
    }
}

struct MoeGate {
    weight: Tensor,
    top_k: usize,
    n_routed_experts: usize,
    e_score_correction_bias: Option<Tensor>,
    moe_cfg: DeepSeekMoEConfig,
}

impl MoeGate {
    fn new(
        cfg: &Config,
        moe_cfg: DeepSeekMoEConfig,
        vb: VarBuilderX,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let weight = match &vb.0 {
            either::Either::Left(vb) => vb.get((n_routed_experts, cfg.hidden_size), "weight")?,
            either::Either::Right(vb) => vb
                .get((n_routed_experts, cfg.hidden_size), "weight")?
                .dequantize(vb.device())?,
        };
        let e_score_correction_bias = if matches!(moe_cfg.topk_method, TopkMethod::NoAuxTc) {
            let bias = match &vb.0 {
                either::Either::Left(vb) => vb.get(n_routed_experts, "e_score_correction_bias")?,
                either::Either::Right(vb) => vb
                    .get(n_routed_experts, "e_score_correction_bias")?
                    .dequantize(vb.device())?,
            };
            Some(bias.to_dtype(DType::F32)?)
        } else {
            None
        };
        Ok(Self {
            weight: weight.to_dtype(DType::F32)?,
            top_k: moe_cfg.num_experts_per_tok.unwrap_or(1),
            n_routed_experts,
            e_score_correction_bias,
            moe_cfg,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (seq_len, _) = xs.dims2()?;
        let moe_cfg = &self.moe_cfg;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?)?;
        let scores = match moe_cfg.scoring_func {
            ScoringFunc::Softmax => candle_nn::ops::softmax_last_dim(&logits)?,
            ScoringFunc::Sigmoid => candle_nn::ops::sigmoid(&logits)?,
        };

        let (mut topk_weight, topk_idx) = match moe_cfg.topk_method {
            TopkMethod::Greedy => {
                let TopKOutput { values, indices } = scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
            TopkMethod::NoAuxTc => {
                let Some(e_score_correction_bias) = &self.e_score_correction_bias else {
                    candle_core::bail!("Expected e_score_correction_bias")
                };
                let scores_for_choice = scores
                    .reshape((seq_len, ()))?
                    .broadcast_add(&e_score_correction_bias.unsqueeze(0)?)?;
                let group_scores = scores_for_choice
                    .reshape((seq_len, moe_cfg.n_group, ()))?
                    .topk(2)?
                    .values
                    .sum(D::Minus1)?;
                let group_idx = group_scores.topk(moe_cfg.topk_group)?.indices;
                let mut group_mask = group_scores.zeros_like()?;
                group_mask = group_mask.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
                    1,
                )?;
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        seq_len,
                        moe_cfg.n_group,
                        self.n_routed_experts / moe_cfg.n_group,
                    ))?
                    .reshape((seq_len, ()))?;
                let tmp_scores = scores_for_choice.broadcast_mul(&score_mask)?;
                let topk_idx = tmp_scores.topk(self.top_k)?.indices;
                (scores.gather(&topk_idx, 1)?, topk_idx)
            }
            TopkMethod::GroupLimitedGreedy => {
                let group_scores = scores
                    .reshape((seq_len, moe_cfg.n_group, ()))?
                    .max(D::Minus1)?;
                let group_idx = group_scores.topk_unsorted(moe_cfg.topk_group)?.indices;
                let mut group_mask = group_scores.zeros_like()?;
                group_mask = group_mask.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
                    1,
                )?;
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        seq_len,
                        moe_cfg.n_group,
                        self.n_routed_experts / moe_cfg.n_group,
                    ))?
                    .reshape((seq_len, ()))?;
                let tmp_scores = masked_fill(&score_mask, &(1. - &score_mask.ne(0.)?)?, 0f32)?;
                let TopKOutput { values, indices } = tmp_scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
        };

        if self.top_k > 1 && moe_cfg.norm_topk_prob {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denominator)?;
        }

        topk_weight = (topk_weight * moe_cfg.routed_scaling_factor)?;
        Ok((topk_idx, topk_weight))
    }
}

struct Moe {
    experts: Vec<Option<Mlp>>,
    shared_experts: Option<Mlp>,
    gate: MoeGate,
    all_reduce: AllReduce,
    experts_start_idx: usize,
    experts_end_idx: usize,
    world_size: usize,
}

enum MoeOrMlp {
    Moe(Moe),
    Mlp(Mlp),
}

struct DecoderLayer {
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
}

pub struct DeepSeekForCausalLM {
    lm_head: ReplicatedLinear,
    embed_tokens: Embedding,
    norm: NormX,
    layers: Vec<DecoderLayer>,
    dtype: DType,
    device: Device,
    config: Config,
    vocab_size: usize,
    is_qvar_builder: bool,
}

impl Moe {
    fn new(
        cfg: &Config,
        moe_cfg: DeepSeekMoEConfig,
        vb: VarBuilderX,
        n_shared_experts: Option<usize>,
        n_routed_experts: usize,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let mut experts = Vec::with_capacity(n_routed_experts);
        let n_local_experts = n_routed_experts / comm.world_size();
        let experts_start_idx = comm.rank() * n_local_experts;
        let experts_end_idx = experts_start_idx + n_local_experts;
        for i in 0..n_routed_experts {
            if i >= experts_start_idx && i < experts_end_idx {
                let vb_e = vb.pp(&format!("experts.{}", i));
                experts.push(Some(Mlp::new(
                    cfg,
                    vb_e,
                    None,
                    Some(moe_cfg.moe_intermediate_size),
                    dtype,
                )?));
            } else {
                experts.push(None);
            }
        }

        let shared_experts = if let Some(n_shared_experts) = n_shared_experts {
            let intermediate_size = moe_cfg.moe_intermediate_size * n_shared_experts;
            Some(Mlp::new(
                cfg,
                vb.pp("shared_experts"),
                None,
                Some(intermediate_size),
                dtype,
            )?)
        } else {
            None
        };
        let gate = MoeGate::new(cfg, moe_cfg, vb.pp("gate"), n_routed_experts)?;
        let word_size = comm.world_size();
        Ok(Self {
            experts,
            shared_experts,
            gate,
            all_reduce: AllReduce::new(comm),
            experts_end_idx,
            experts_start_idx,
            world_size: word_size,
        })
    }

    fn moe_infer(&self, xs: &Tensor, topk_ids: &Tensor, topk_weight: &Tensor) -> Result<Tensor> {
        let mut y = xs.zeros_like()?;
        let topk_weight = if topk_weight.dtype() != xs.dtype() {
            topk_weight.to_dtype(xs.dtype())?
        } else {
            topk_weight.to_owned()
        };
        let unique_ids: HashSet<u32> = topk_ids
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?
            .into_iter()
            .collect();
        let mut cur_used_experts = Vec::<u32>::new();
        for i in self.experts_start_idx..self.experts_end_idx {
            if unique_ids.contains(&(i as u32)) {
                cur_used_experts.push(i as u32);
            }
        }

        for i in &cur_used_experts {
            let idx_top = topk_ids.eq(*i as u32)?.nonzero()?.t()?.contiguous()?;
            let idx = &idx_top.i(0)?.contiguous()?;
            let top = &idx_top.i(1)?.contiguous()?;
            let expert = self.experts[*i as usize]
                .as_ref()
                .expect("Expert is not present for this rank.");

            y = y.index_add(
                idx,
                &expert.forward(&xs.index_select(idx, 0)?)?.broadcast_mul(
                    &topk_weight
                        .index_select(idx, 0)?
                        .gather(&top.unsqueeze(1)?, 1)?
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)?,
                )?,
                0,
            )?;
        }

        if self.world_size > 1 {
            y = self.all_reduce.apply(&y)?;
        }
        Ok(y)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;
        let mut y = self.moe_infer(&xs, &topk_idx, &topk_weight)?;
        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }
        Ok(y)
    }
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilderX,
        layer_idx: usize,
        comm: Rc<Comm>,
        moe_cfg: DeepSeekMoEConfig,
        dtype: DType,
    ) -> Result<Self> {
        let attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            moe_cfg.clone(),
            dtype,
        )?;
        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
            dtype,
            false,
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            dtype,
            false,
        )?;
        let moe_or_mlp = if moe_cfg.n_routed_experts > 0
            && layer_idx >= moe_cfg.first_k_dense_replace
            && layer_idx % moe_cfg.moe_layer_freq == 0
        {
            MoeOrMlp::Moe(Moe::new(
                cfg,
                moe_cfg.clone(),
                vb.pp("mlp"),
                moe_cfg.n_shared_experts,
                moe_cfg.n_routed_experts,
                comm.clone(),
                dtype,
            )?)
        } else {
            MoeOrMlp::Mlp(Mlp::new(cfg, vb.pp("mlp"), None, None, dtype)?)
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .attn
            .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

impl DeepSeekForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        cfg: &Config,
        dtype: DType,
        _is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let Some(extra_cfg) = &cfg.extra_config_json else {
            candle_core::bail!("DeepSeek requires extra_config_json for full config")
        };
        let deepseek_cfg: DeepSeekConfig =
            serde_json::from_str(extra_cfg).map_err(candle_core::Error::wrap)?;
        let moe_cfg = DeepSeekMoEConfig {
            num_experts_per_tok: deepseek_cfg.num_experts_per_tok,
            n_routed_experts: deepseek_cfg.n_routed_experts.unwrap_or(0),
            moe_intermediate_size: deepseek_cfg.moe_intermediate_size,
            scoring_func: deepseek_cfg.scoring_func.clone(),
            topk_method: deepseek_cfg.topk_method.clone(),
            norm_topk_prob: deepseek_cfg.norm_topk_prob,
            routed_scaling_factor: deepseek_cfg.routed_scaling_factor,
            n_shared_experts: deepseek_cfg.n_shared_experts,
            qk_nope_head_dim: deepseek_cfg.qk_nope_head_dim,
            qk_rope_head_dim: deepseek_cfg.qk_rope_head_dim,
            v_head_dim: deepseek_cfg.v_head_dim,
            kv_lora_rank: deepseek_cfg.kv_lora_rank,
            first_k_dense_replace: deepseek_cfg.first_k_dense_replace,
            moe_layer_freq: deepseek_cfg.moe_layer_freq,
            rope_scaling: deepseek_cfg.rope_scaling.clone(),
            q_lora_rank: deepseek_cfg.q_lora_rank,
            n_group: deepseek_cfg.n_group,
            topk_group: deepseek_cfg.topk_group,
        };

        let is_qvar_builder = vb.is_qvar_builder();
        let vb_m = vb.pp("model");
        let (embed_tokens, vocab_size) = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
            if is_qvar_builder || cfg.quant.is_some() || cfg.quantization_config.is_some() {
                DType::F32
            } else {
                dtype
            },
        )?;

        let norm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_m.pp("norm"),
            dtype,
            false,
        )?;

        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: moe_cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta.unwrap_or(10000.0) as f32,
            qk_rope_head_dim: moe_cfg.qk_rope_head_dim,
        };
        let rotary_emb = Arc::new(DeepSeekV2RotaryEmbedding::new(&rope_cfg, device)?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_m.pp(&format!("layers.{}", layer_idx)),
                layer_idx,
                comm.clone(),
                moe_cfg.clone(),
                dtype,
            )?;
            layers.push(layer);
            progress_reporter.write().set_progress(layer_idx + 1);
        }

        let lm_head = if !cfg.tie_word_embeddings.unwrap_or(false) {
            ReplicatedLinear::load_no_bias(
                cfg.hidden_size,
                vocab_size,
                vb.pp("lm_head"),
                &None,
                &None,
                dtype,
            )?
        } else {
            ReplicatedLinear::from_weight_bias(embed_tokens.embeddings().clone(), None)?
        };

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            dtype,
            device: device.clone(),
            config: cfg.clone(),
            vocab_size,
            is_qvar_builder,
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(xs)
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let seqlens = if input_metadata.cu_seqlens_q.is_some() {
            input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..]
                .into()
        } else {
            Vec::new()
        };
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );
        let mut xs = if embeded_inputs {
            x.to_owned()
        } else {
            self.embed_tokens.forward(x)?
        };

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), block) in zip(kv_caches.iter(), &self.layers) {
                xs = block.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        } else {
            for block in &self.layers {
                xs = block.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
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
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            x,
            input_positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        self.forward_inner(
            x,
            input_positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            true,
        )
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
