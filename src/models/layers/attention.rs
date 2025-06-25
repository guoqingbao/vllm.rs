use crate::models::layers::linear::{
    linear_b_x as linear_b, linear_no_bias_x as linear_no_bias, LinearX as Linear,
};
use crate::models::layers::others::rms_norm;
use crate::models::layers::rotary_emb::RotaryEmbedding;
use crate::utils::config::Config;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::var_builder::Shard;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::RmsNorm;
use std::sync::Arc;

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn: PagedAttention,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        rotary_emb: Arc<RotaryEmbedding>,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_heads);
        let sd = Shard::default();
        let attention_bias = config.attention_bias.unwrap_or(true);
        let q_proj = linear_b(
            hidden_size,
            num_heads * head_dim,
            attention_bias,
            vb.pp("q_proj"),
            sd,
            &config.quant,
            dtype,
        )?;
        let k_proj = linear_b(
            hidden_size,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("k_proj"),
            sd,
            &config.quant,
            dtype,
        )?;
        let v_proj = linear_b(
            hidden_size,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("v_proj"),
            sd,
            &config.quant,
            dtype,
        )?;
        let o_proj = linear_no_bias(
            num_heads * head_dim,
            hidden_size,
            vb.pp("o_proj"),
            sd,
            &config.quant,
            dtype,
        )?;

        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"));
        let q_norm = if q_norm.is_ok() {
            Some(q_norm.unwrap())
        } else {
            None
        };

        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"));
        let k_norm = if k_norm.is_ok() {
            Some(k_norm.unwrap())
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            attn: PagedAttention::new(
                num_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(num_kv_heads),
                config.sliding_window,
                vb.device().clone(),
                None,
            )?,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = if self.q_norm.is_some() && self.k_norm.is_some() {
            //Per‑head RMSNorm in qwen3
            let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> (BHL, D) after transpose later
            let k_flat = k.flatten(0, 2)?;
            let q_flat = self.q_norm.as_ref().unwrap().forward(&q_flat)?;
            let k_flat = self.k_norm.as_ref().unwrap().forward(&k_flat)?;
            let q = q_flat.reshape((1, self.num_heads, seq_len, self.head_dim))?;
            let k = k_flat.reshape((1, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        // Apply rotary embeddings
        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)?;

        let y = self
            .attn
            .forward(
                &q,
                &k,
                &v,
                attention_mask,
                cache.map(|(k_, _)| k_.clone()),
                cache.map(|(_, v_)| v_.clone()),
                input_metadata,
                None,
            )?
            .reshape((seq_len, ()))?;

        self.o_proj.forward(&y)
    }
}
