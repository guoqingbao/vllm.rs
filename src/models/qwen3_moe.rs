// src/models/qwen3_moe.rs
use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::linear::LinearX as Linear;
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::{FusedMoe, FusedMoeGGUF, FusedMoeISQ};
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::ScalingRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use either::Either;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    FusedMoeGGUF(FusedMoeGGUF),
    FusedMoeISQ(FusedMoeISQ),
    Mlp(MLP),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeGGUF(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
        }
    }
}

pub struct Qwen3DecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    shared_gate: Option<Linear>,
    shared_expert: Option<MLP>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
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
            .unwrap_or(&Vec::<usize>::new())
            .contains(&layer_idx)
            && (moe_cfg.num_experts.unwrap_or(0) > 0
                && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap_or(1) == 0)
        {
            if is_qvar_builder {
                //experts weights packed
                MoeOrMlp::FusedMoeGGUF(FusedMoeGGUF::new(config, vb.clone(), comm.clone(), dtype)?)
            } else if config.quantization_config.is_some() {
                panic!("This feature is under developement (use unquantized, gguf or isq to gguf instead)!");
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
                "",
            )?;

            MoeOrMlp::Mlp(mlp)
        };

        //shared experts weights in Qwen2 MoE models
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
                            //weight must be 2d+
                            ws.dequantize(&vb.device())?
                                .reshape((1, config.hidden_size))?
                        }
                    }
                    .to_dtype(dtype)?;

                    let shared_gate = Linear::new(ws, None, &None);

                    let mlp = MLP::new(
                        if is_qvar_builder {
                            vb.clone()
                        } else {
                            vb.pp("mlp.shared_expert").clone()
                        },
                        comm.clone(),
                        config,
                        intermediate_size,
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
            shared_gate,
            shared_expert,
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

        //shared experts for Qwen2 MoE models
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
}

pub struct Qwen3MoEForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
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
        let vb = if config.is_multi_model.is_some_and(|x| x) && !is_qvar_builder {
            vb.pp("language_model")
        } else {
            vb.clone()
        };
        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["model.embed_tokens"])
            } else {
                vb.pp("model.embed_tokens")
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
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(xs)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
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

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}
