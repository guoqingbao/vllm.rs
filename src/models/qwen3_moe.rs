// src/models/qwen3_moe.rs
use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::mask::get_attention_casual_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::{FusedMoeGGUF, FusedMoeISQ, MoeNaive};
use crate::models::layers::others::{embedding, rms_norm};
use crate::models::layers::rotary_emb::ScalingRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, RmsNorm};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

enum MoeOrMlp {
    MoeNaive(MoeNaive),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    Mlp(MLP),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::MoeNaive(m) => m.forward(xs),
            Self::FusedMoeGGUF(m) => m.forward(xs),
            Self::FusedMoeISQ(m) => m.forward(xs),
        }
    }
}

pub struct Qwen3DecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        dtype: DType,
        layer_idx: usize,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let self_attn = Attention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            rotary_emb,
            config,
            dtype,
        )?;

        let moe_cfg = config
            .moe_cfg
            .as_ref()
            .expect("MoE config is not available!");

        let mlp = if !moe_cfg
            .mlp_only_layers
            .as_ref()
            .unwrap()
            .contains(&layer_idx)
            && (moe_cfg.num_experts.unwrap() > 0
                && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap() == 0)
        {
            if is_qvar_builder {
                //experts weights packed
                MoeOrMlp::FusedMoeGGUF(FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?)
            } else if config.quant.is_some() {
                MoeOrMlp::FusedMoeISQ(FusedMoeISQ::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            } else {
                MoeOrMlp::MoeNaive(MoeNaive::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            }
        } else {
            let mlp = MLP::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("mlp").clone()
                },
                comm.clone(),
                config,
                config.intermediate_size,
                false,
                dtype,
            )?;

            MoeOrMlp::Mlp(mlp)
        };

        let is_qvar_builder = vb.is_qvar_builder();

        let key_map: HashMap<&str, &str> = [
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "ffn_norm"),
        ]
        .iter()
        .cloned()
        .collect();

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["input_layernorm"]).clone()
            } else {
                vb.pp("input_layernorm").clone()
            },
            dtype,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["post_attention_layernorm"]).clone()
            } else {
                vb.pp("post_attention_layernorm").clone()
            },
            dtype,
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
        attention_mask: Option<&Vec<Tensor>>,
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

pub struct Qwen3MoEForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
}

impl Qwen3MoEForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let key_map: HashMap<&str, &str> = [
            ("model.embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("model.norm", "output_norm"),
            ("model.layers", "blk"),
        ]
        .iter()
        .cloned()
        .collect();
        let reporter = progress_reporter.clone();

        let is_qvar_builder = vb.is_qvar_builder();
        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["model.embed_tokens"])
            } else {
                vb.pp("model.embed_tokens")
            },
            dtype,
        )?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder { DType::F32} else { dtype },
            config,
            &vb.device(),
            is_rope_i,
        )?);

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = Qwen3DecoderLayer::new(
                vb.pp(format!(
                    "{}.{}",
                    if is_qvar_builder {
                        key_map["model.layers"]
                    } else {
                        "model.layers"
                    },
                    i
                )
                .as_str()),
                comm.clone(),
                rotary_emb.clone(),
                config,
                dtype,
                i,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["model.norm"])
            } else {
                vb.pp("model.norm")
            },
            dtype,
        )?;

        let lm_head = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if config.tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp(key_map["model.embed_tokens"])
                } else {
                    vb.pp("model.embed_tokens")
                }
            } else {
                if is_qvar_builder {
                    vb.pp(key_map["lm_head"])
                } else {
                    vb.pp("lm_head")
                }
            },
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
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
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

        let attention_mask = get_attention_casual_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );
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

        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}
