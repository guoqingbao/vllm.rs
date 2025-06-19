// src/models/qwen3.rs
use crate::models::layers::linear::{
    LinearX as Linear, linear_b_x as linear_b, linear_no_bias_x as linear_no_bias,
};
use crate::models::layers::mask::get_attention_casual_mask;
use crate::models::layers::others::{embedding, rms_norm};
use crate::models::layers::rotary_emb::RotaryEmbedding;
use crate::utils::config::Config;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::var_builder::Shard;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Module, RmsNorm};
use std::iter::zip;
use std::sync::Arc;

pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cfg: Config,
    attn: PagedAttention,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Qwen3Attention {
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
            cfg: config.clone(),
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

pub struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3MLP {
    pub fn new(vb: VarBuilder, config: &Config, dtype: DType) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let sd = Shard::default();
        let gate_proj = linear_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("gate_proj"),
            sd,
            &config.quant,
            dtype,
        )?;
        let up_proj = linear_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("up_proj"),
            sd,
            &config.quant,
            dtype,
        )?;
        let down_proj = linear_no_bias(
            intermediate_size,
            hidden_size,
            vb.pp("down_proj"),
            sd,
            &config.quant,
            dtype,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

pub struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3DecoderLayer {
    pub fn new(
        vb: VarBuilder,
        rotary_emb: Arc<RotaryEmbedding>,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        let self_attn = Qwen3Attention::new(vb.pp("self_attn"), rotary_emb, config, dtype)?;
        let mlp = Qwen3MLP::new(vb.pp("mlp"), config, dtype)?;

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
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
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let attn_output =
            self.self_attn
                .forward(&xs, attention_mask, positions, cache, input_metadata)?;
        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;

        residual + mlp_output
    }
}

pub struct Qwen3ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    config: Config,
    dtype: DType,
}

impl Qwen3ForCausalLM {
    pub fn new(vb: VarBuilder, config: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, config, &vb.device())?);

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = Qwen3DecoderLayer::new(
                vb.pp(&format!("model.layers.{}", i)),
                rotary_emb.clone(),
                config,
                dtype,
            )?;
            layers.push(layer);
        }

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;

        let lm_head = linear_no_bias(
            config.hidden_size,
            config.vocab_size,
            vb.pp("lm_head"),
            Shard::default(),
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
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let seq_len = input_ids.dims1()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = get_attention_casual_mask(
                &self.device,
                self.dtype,
                seq_len,
                positions,
                self.config.sliding_window,
            )?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(&input_ids)?;

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    &input_metadata,
                )?;
            }
        } else {
            for layer in self.layers.iter() {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    None,
                    &input_metadata,
                )?
            }
        }

        if input_metadata.cu_seqlens_q.is_some() {
            let indices = &input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..];
            let indices: Vec<_> = indices.iter().map(|x| x - 1).collect();
            let length = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (length,), &xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }
}
