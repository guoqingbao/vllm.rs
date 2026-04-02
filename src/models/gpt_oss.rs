use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::moe::{FusedMoe, FusedMoeGGUF, FusedMoeISQ, FusedMoeMxfp4, MoeActFn};
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

enum GptOssMoE {
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeMxfp4(FusedMoeMxfp4),
}

impl GptOssMoE {
    fn set_act(&mut self, act: MoeActFn) {
        match self {
            Self::FusedMoe(m) => m.set_act(act),
            Self::FusedMoeGGUF(m) => m.set_act(act),
            Self::FusedMoeISQ(m) => m.set_act(act),
            Self::FusedMoeMxfp4(m) => m.set_act(act),
        }
    }

    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeGGUF(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeMxfp4(m) => m.forward(xs, is_prefill),
        }
    }
}

pub struct GptOssDecoderLayer {
    self_attn: Attention,
    mlp: GptOssMoE,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    #[allow(dead_code)]
    sinks: Option<Tensor>,
    is_sliding: bool,
}

impl GptOssDecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        dtype: DType,
        _layer_idx: usize,
        is_sliding: bool,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        let self_attn = Attention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            config,
            None,
            sliding_window,
            dtype,
        )?;

        let sinks = if is_qvar_builder {
            vb.pp("attn_sinks")
                .get((config.num_attention_heads,), "weight")
                .ok()
                .map(|t| t.to_dtype(dtype))
                .transpose()?
        } else {
            vb.pp("self_attn")
                .get((config.num_attention_heads,), "sinks")
                .ok()
                .map(|t| t.to_dtype(dtype))
                .transpose()?
        };

        let _moe_cfg = config
            .moe_cfg
            .as_ref()
            .expect("GPT-OSS requires MoE config");

        let (swiglu_alpha, swiglu_limit) = if let Some(extra) = &config.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            (
                v.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.702) as f32,
                v.get("swiglu_limit")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(7.0) as f32,
            )
        } else {
            (1.702f32, 7.0f32)
        };

        let mut mlp = if is_qvar_builder {
            GptOssMoE::FusedMoeGGUF(FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?)
        } else if let Some(quant_config) = &config.quantization_config {
            if quant_config.quant_method == "mxfp4" {
                GptOssMoE::FusedMoeMxfp4(FusedMoeMxfp4::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            } else {
                panic!(
                    "GPT-OSS: unsupported quant method '{}' (use unquantized, gguf, or mxfp4)",
                    quant_config.quant_method
                );
            }
        } else if config.quant.is_some() {
            GptOssMoE::FusedMoeISQ(FusedMoeISQ::new(
                config,
                vb.pp("mlp").clone(),
                comm.clone(),
                dtype,
            )?)
        } else {
            GptOssMoE::FusedMoe(FusedMoe::new(
                config,
                vb.pp("mlp").clone(),
                comm.clone(),
                dtype,
            )?)
        };

        mlp.set_act(MoeActFn::GptOssSwiglu {
            alpha: swiglu_alpha,
            limit: swiglu_limit,
        });

        let key_map: HashMap<&str, &str> = [
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
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
            false,
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
            false,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb,
            sinks,
            is_sliding,
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
        let rope: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.clone();
        let mask = if self.is_sliding {
            sliding_mask
        } else {
            full_mask
        };
        let attn_output =
            self.self_attn
                .forward(&xs, &Some(rope), mask, positions, cache, input_metadata)?;
        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs, input_metadata.is_prefill)?;
        residual + mlp_output
    }
}

pub struct GptOssForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<GptOssDecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
    #[allow(dead_code)]
    layer_types: Vec<bool>,
}

fn parse_layer_types(config: &Config) -> Vec<bool> {
    if let Some(extra) = &config.extra_config_json {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(extra) {
            if let Some(arr) = v.get("layer_types").and_then(|v| v.as_array()) {
                return arr
                    .iter()
                    .map(|v| v.as_str() == Some("sliding_attention"))
                    .collect();
            }
        }
    }
    if config.sliding_window.is_some() {
        (0..config.num_hidden_layers).map(|i| i % 2 == 0).collect()
    } else {
        vec![false; config.num_hidden_layers]
    }
}

impl GptOssForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let prefix = "model.".to_string();
        let gguf_prefix = "".to_string();

        let key_map: HashMap<&str, &str> = [
            ("embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("norm", "output_norm"),
            ("layers", "blk"),
        ]
        .iter()
        .cloned()
        .collect();

        let reporter = progress_reporter.clone();

        let tie_word_embeddings = config.tie_word_embeddings;

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["embed_tokens"]))
            } else {
                vb.pp(&format!("{}embed_tokens", prefix))
            },
            dtype,
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

        let layer_types = parse_layer_types(config);

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let is_sliding = layer_types.get(i).copied().unwrap_or(false);
            let layer = GptOssDecoderLayer::new(
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
                dtype,
                i,
                is_sliding,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(&format!("{}{}", gguf_prefix, key_map["norm"]))
            } else {
                vb.pp(&format!("{}norm", prefix))
            },
            dtype,
            false,
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

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            is_qvar_builder,
            layer_types,
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(xs)?;
        if (self.is_qvar_builder || self.config.quant.is_some()) && xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)
        } else {
            Ok(xs)
        }
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();

        let sliding_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );

        let full_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            None,
            input_metadata.is_prefill,
        );

        let mut xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };

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

        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        if self.is_qvar_builder {
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
        )
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
