// src/models/llama.rs
use crate::models::layers::attention::Attention;
use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::models::layers::mask::get_attention_casual_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{embedding, rms_norm};
use crate::models::layers::rotary_emb::RotaryEmbedding;
use crate::utils::config::Config;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::var_builder::Shard;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Module, RmsNorm};
use std::iter::zip;
use std::sync::Arc;

pub struct LLaMaDecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LLaMaDecoderLayer {
    pub fn new(
        vb: VarBuilder,
        rotary_emb: Arc<RotaryEmbedding>,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        let self_attn = Attention::new(vb.pp("self_attn"), rotary_emb, config, dtype)?;
        let mlp = MLP::new(vb.pp("mlp"), config, dtype)?;

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

pub struct LLaMaForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<LLaMaDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    config: Config,
    dtype: DType,
}

impl LLaMaForCausalLM {
    pub fn new(vb: VarBuilder, config: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, config, vb.device())?);

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = LLaMaDecoderLayer::new(
                vb.pp(format!("model.layers.{}", i)),
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
            if config.tie_word_embeddings.is_some_and(|x| x) {
                vb.pp("model.embed_tokens")
            } else {
                vb.pp("lm_head")
            },
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
        let mut xs = self.embed_tokens.forward(input_ids)?;

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        } else {
            for layer in self.layers.iter() {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    None,
                    input_metadata,
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
            xs = xs.index_select(&Tensor::from_vec(indices, (length,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }
}
