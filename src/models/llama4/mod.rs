pub mod config;
mod vision;

use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::moe::FusedMoeNvfp4;
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::{Config, MoEConfig};
use crate::utils::image::ImageData;
use crate::utils::progress::ProgressLike;
use attention_rs::ops::NonZeroOp;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Activation, Module};
use config::{Llama4Config, TextConfig};
use parking_lot::RwLock;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;
use vision::Llama4VisionModel;

// ---------------------------------------------------------------------------
// Llama4 MoE with shared expert and sigmoid routing
// ---------------------------------------------------------------------------

enum Llama4Experts {
    Dense {
        router: ReplicatedLinear,
        gate_up_w: Tensor,
        down_w: Tensor,
        w_size_n: usize,
        topk: usize,
    },
    Nvfp4(FusedMoeNvfp4),
}

struct Llama4TextMoe {
    experts: Llama4Experts,
    shared_expert: MLP,
}

impl Llama4TextMoe {
    fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        text_cfg: &TextConfig,
        dtype: DType,
    ) -> Result<Self> {
        let is_nvfp4 = config
            .quantization_config
            .as_ref()
            .map_or(false, |q| q.quant_method == "nvfp4");

        let experts = if is_nvfp4 {
            let moe_cfg = MoEConfig {
                moe_intermediate_size: text_cfg.intermediate_size,
                shared_expert_intermediate_size: None,
                num_experts: Some(text_cfg.num_local_experts),
                mlp_only_layers: None,
                decoder_sparse_step: None,
                norm_topk_prob: false,
                num_experts_per_tok: text_cfg.num_experts_per_tok,
                first_k_dense_replace: None,
                n_shared_experts: None,
                routed_scaling_factor: None,
                n_group: None,
                topk_group: None,
                scoring_func: None,
                topk_method: None,
            };
            let mut nvfp4_cfg = config.clone();
            nvfp4_cfg.moe_cfg = Some(moe_cfg);
            let mut fused = FusedMoeNvfp4::new_with_gate(
                &nvfp4_cfg,
                vb.pp("router"),
                vb.pp("experts"),
                comm.clone(),
                dtype,
            )?;
            fused.set_sigmoid_routing();
            Llama4Experts::Nvfp4(fused)
        } else {
            let router = ReplicatedLinear::load_no_bias(
                config.hidden_size,
                text_cfg.num_local_experts,
                vb.pp("router"),
                &None,
                &None,
                dtype,
            )?;
            let experts_vb = vb.pp("experts");
            let n = text_cfg.num_local_experts;
            let inter = text_cfg.intermediate_size;
            let hidden = config.hidden_size;
            let (gate_up_w, down_w) = match &experts_vb.0 {
                either::Either::Left(svb) => {
                    if svb.contains_tensor("gate_up_proj") {
                        // gate_up_proj [E, hidden, 2*inter] -> transpose -> [E, 2*inter, hidden]
                        let gate_up_w = svb
                            .get((n, hidden, inter * 2), "gate_up_proj")?
                            .t()?
                            .contiguous()?;
                        // down_proj [E, inter, hidden] -> transpose -> [E, hidden, inter]
                        let down_w = svb
                            .get((n, inter, hidden), "down_proj")?
                            .t()?
                            .contiguous()?;
                        (gate_up_w, down_w)
                    } else {
                        let mut gate_v = Vec::new();
                        let mut up_v = Vec::new();
                        let mut down_v = Vec::new();
                        for i in 0..n {
                            let ev = svb.pp(i.to_string());
                            gate_v.push(
                                ev.pp("gate_proj")
                                    .get((inter, hidden), "weight")?,
                            );
                            up_v.push(
                                ev.pp("up_proj")
                                    .get((inter, hidden), "weight")?,
                            );
                            down_v.push(
                                ev.pp("down_proj")
                                    .get((hidden, inter), "weight")?,
                            );
                        }
                        let gate = Tensor::stack(&gate_v, 0)?;
                        let up = Tensor::stack(&up_v, 0)?;
                        let gate_up_w = Tensor::cat(&[&gate, &up], 1)?;
                        let down_w = Tensor::stack(&down_v, 0)?;
                        (gate_up_w, down_w)
                    }
                }
                _ => candle_core::bail!("Llama4 Dense MoE requires safetensors VarBuilder"),
            };
            let w_size_n = gate_up_w.dim(1)? / 2;
            Llama4Experts::Dense {
                router,
                gate_up_w,
                down_w,
                w_size_n,
                topk: text_cfg.num_experts_per_tok,
            }
        };

        let shared_expert = MLP::new(
            vb.pp("shared_expert"),
            comm.clone(),
            config.hidden_size,
            text_cfg.intermediate_size,
            &config.hidden_act,
            &config.quantization_config,
            &config.quant,
            false,
            dtype,
            "",
        )?;

        Ok(Self {
            experts,
            shared_expert,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.shape().clone();
        let hidden_dim = *orig_shape.dims().last().unwrap();
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let routed_output = match &self.experts {
            Llama4Experts::Nvfp4(fused_moe) => fused_moe.forward(&xs_flat, false)?,
            Llama4Experts::Dense {
                router,
                gate_up_w,
                down_w,
                w_size_n,
                topk,
            } => {
                let router_logits = router.forward(&xs_flat)?.to_dtype(DType::F32)?;

                let topk_ids = router_logits.arg_sort_last_dim(false)?.narrow(
                    D::Minus1,
                    0,
                    *topk,
                )?;
                let sigmoid_weights =
                    (router_logits.neg()?.exp()? + 1.0)?.recip()?;
                let topk_weights = sigmoid_weights.gather(&topk_ids, D::Minus1)?;
                let topk_ids = topk_ids.to_dtype(DType::U32)?;

                let (expert_ids, sorted_token_ids) =
                    topk_ids.flatten_all()?.sort_last_dim(true)?;

                let gate_up = attention_rs::moe::moe_gemm(
                    &xs_flat,
                    gate_up_w,
                    &None,
                    &sorted_token_ids,
                    &expert_ids,
                    *topk,
                    false,
                )?;
                let gate = gate_up
                    .narrow(D::Minus1, 0, *w_size_n)?
                    .contiguous()?;
                let up = gate_up
                    .narrow(D::Minus1, *w_size_n, *w_size_n)?
                    .contiguous()?;
                let down_inputs =
                    (up * Activation::Silu.forward(&gate)?)?;

                let topk_weights = topk_weights.to_dtype(xs.dtype())?;
                attention_rs::moe::moe_gemm(
                    &down_inputs,
                    down_w,
                    &Some(topk_weights),
                    &sorted_token_ids,
                    &expert_ids,
                    *topk,
                    false,
                )?
                .reshape((xs_flat.dim(0)?, *topk, hidden_dim))?
                .sum(1)?
            }
        };

        let routed_output = routed_output.reshape(&orig_shape)?;
        let shared_output = self.shared_expert.forward(xs)?;
        shared_output + routed_output
    }
}

// ---------------------------------------------------------------------------
// Llama4 Decoder Layer — supports both dense MLP and MoE per-layer,
// alternating RoPE / no-RoPE (iRoPE) with temperature tuning,
// and chunked vs full causal attention masks.
// ---------------------------------------------------------------------------

enum FeedForward {
    Dense(MLP),
    Moe(Llama4TextMoe),
}

pub struct LLama4DecoderLayer {
    self_attn: Attention,
    ff: FeedForward,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Option<Arc<ScalingRotaryEmbedding>>,
    use_chunked_attention: bool,
}

impl LLama4DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        text_cfg: &TextConfig,
        layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let use_rope = (layer_idx + 1) % 4 == 0;
        let use_chunked_attention = !use_rope;

        let self_attn = Attention::new(
            vb.pp("self_attn"),
            comm.clone(),
            config,
            None,
            config.sliding_window,
            dtype,
        )?;

        let moe_layers = text_cfg.moe_layers();
        let is_moe_layer = moe_layers.contains(&layer_idx);

        let ff = if is_moe_layer {
            FeedForward::Moe(Llama4TextMoe::new(
                vb.pp("feed_forward"),
                comm.clone(),
                config,
                text_cfg,
                dtype,
            )?)
        } else {
            FeedForward::Dense(MLP::new(
                vb.pp("feed_forward"),
                comm.clone(),
                config.hidden_size,
                text_cfg.mlp_intermediate_size(),
                &config.hidden_act,
                &config.quantization_config,
                &config.quant,
                false,
                dtype,
                "",
            )?)
        };

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
            dtype,
            false,
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            dtype,
            false,
        )?;

        Ok(Self {
            self_attn,
            ff,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb: if use_rope { Some(rotary_emb) } else { None },
            use_chunked_attention,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        chunked_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        let mask = if self.use_chunked_attention {
            chunked_mask
        } else {
            attention_mask
        };

        let rope: Option<Arc<dyn ApplyRotaryEmbedding>> = self
            .rotary_emb
            .clone()
            .map(|r| r as Arc<dyn ApplyRotaryEmbedding>);

        let attn_output =
            self.self_attn
                .forward(&xs, &rope, mask, positions, cache, input_metadata)?;

        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;

        let ff_output = match &self.ff {
            FeedForward::Dense(mlp) => mlp.forward(&xs)?,
            FeedForward::Moe(moe) => moe.forward(&xs)?,
        };

        residual + ff_output
    }
}

// ---------------------------------------------------------------------------
// Multimodal Projector
// ---------------------------------------------------------------------------

struct Llama4MultiModalProjector {
    linear: ReplicatedLinear,
}

impl Llama4MultiModalProjector {
    fn new(
        vision_output_dim: usize,
        hidden_size: usize,
        vb: VarBuilderX,
        dtype: DType,
    ) -> Result<Self> {
        let linear = ReplicatedLinear::load_no_bias(
            vision_output_dim,
            hidden_size,
            vb.pp("linear_1"),
            &None,
            &None,
            dtype,
        )?;
        Ok(Self { linear })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// LLama4ForConditionalGeneration
// ---------------------------------------------------------------------------

pub struct LLama4ForConditionalGeneration {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<LLama4DecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    vision_model: Option<Llama4VisionModel>,
    multi_modal_projector: Option<Llama4MultiModalProjector>,
    image_token_index: u32,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
    attention_chunk_size: usize,
}

impl LLama4ForConditionalGeneration {
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

        // Parse the Llama4-specific config from extra_config_json
        let llama4_cfg: Llama4Config = if let Some(extra) = &config.extra_config_json {
            serde_json::from_str(extra).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to parse Llama4 config: {e}"))
            })?
        } else {
            candle_core::bail!(
                "Llama4 requires extra_config_json with text_config and vision_config"
            );
        };
        let text_cfg = &llama4_cfg.text_config;

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("language_model.model.embed_tokens"),
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

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = LLama4DecoderLayer::new(
                vb.pp(format!("language_model.model.layers.{}", i).as_str()),
                comm.clone(),
                rotary_emb.clone(),
                config,
                text_cfg,
                i,
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("language_model.model.norm"),
            dtype,
            false,
        )?;

        let lm_head = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if config.tie_word_embeddings.is_some_and(|x| x) {
                vb.pp("language_model.model.embed_tokens")
            } else {
                vb.pp("language_model.lm_head")
            },
            &None,
            &None,
            dtype,
        )?;

        // Vision model (optional — gracefully skip if weights missing)
        let vision_model = match Llama4VisionModel::new(
            &llama4_cfg.vision_config,
            vb.pp("vision_model"),
            dtype,
            device,
        ) {
            Ok(vm) => {
                crate::log_info!("Llama4 vision model loaded successfully.");
                Some(vm)
            }
            Err(e) => {
                crate::log_warn!("Llama4 vision model not loaded: {e}. Running text-only.");
                None
            }
        };

        let multi_modal_projector = if vision_model.is_some() {
            match Llama4MultiModalProjector::new(
                llama4_cfg.vision_config.vision_output_dim,
                config.hidden_size,
                vb.pp("multi_modal_projector"),
                dtype,
            ) {
                Ok(proj) => Some(proj),
                Err(e) => {
                    crate::log_warn!("Llama4 projector not loaded: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vision_model,
            multi_modal_projector,
            image_token_index: llama4_cfg.image_token_index as u32,
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            is_qvar_builder,
            attention_chunk_size: text_cfg.attention_chunk_size,
        })
    }

    fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
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
        images: Option<&ImageData>,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();

        let mut xs = self.embed_forward(input_ids)?;

        // Vision fusion: replace image token embeddings with projected vision features
        if let (Some(images), Some(vision_model), Some(projector)) =
            (images, &self.vision_model, &self.multi_modal_projector)
        {
            let image_mask = input_ids.eq(self.image_token_index)?;
            let image_mask = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(xs.shape().clone())?
                .to_dtype(DType::U32)?;

            let mut image_tensor = images.to_tensor_f32(&xs.device())?.to_dtype(self.dtype)?;
            let num_images = image_tensor.dim(0)?;
            if images.image_idx > 0 && (images.image_idx as usize) < num_images {
                image_tensor = image_tensor.narrow(
                    0,
                    images.image_idx as usize,
                    num_images - images.image_idx as usize,
                )?;
            }

            let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            if indices.shape().dim(0)? > 0 {
                let image_features = vision_model.forward(&image_tensor)?;
                let image_features =
                    image_features.reshape(((), image_features.dim(D::Minus1)?))?;
                let image_features = projector.forward(&image_features)?;

                let hidden = xs.dim(D::Minus1)?;
                let indices_len = indices.shape().dim(0)?;
                if indices_len % hidden != 0 {
                    candle_core::bail!(
                        "image indices length {} not divisible by hidden size {}",
                        indices_len,
                        hidden
                    );
                }
                let tokens_in_chunk = indices_len / hidden;
                let total_tokens = image_features.dim(0)?;
                let start = images.image_token_offset.min(total_tokens);
                let end = start + tokens_in_chunk;
                if end > total_tokens {
                    candle_core::bail!(
                        "image token slice out of range: start {}, len {}, total {}",
                        start,
                        tokens_in_chunk,
                        total_tokens
                    );
                }
                let image_features = if start > 0 || end < total_tokens {
                    image_features.narrow(0, start, tokens_in_chunk)?
                } else {
                    image_features
                };

                let mut x_flat = xs.flatten_all()?;
                let image_flat = image_features.flatten_all()?.to_dtype(xs.dtype())?;
                x_flat = x_flat.scatter_add(
                    &indices,
                    &(image_flat - x_flat.gather(&indices, 0)?)?,
                    0,
                )?;
                xs = x_flat.reshape(xs.shape())?;
            }
        }

        // Full causal mask
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            None,
            input_metadata.is_prefill,
        );

        // Chunked causal mask for iRoPE layers
        let chunked_mask = if self.attention_chunk_size > 0 {
            get_attention_causal_mask(
                &self.device,
                self.dtype,
                positions,
                seqlens.clone(),
                Some(self.attention_chunk_size),
                input_metadata.is_prefill,
            )
        } else {
            attention_mask.clone()
        };

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    chunked_mask.as_ref(),
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
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            images,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            images,
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
