// src/models/gemma3.rs

use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{conv2d_no_bias, embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::ScalingRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use parking_lot::RwLock;
pub mod config;
use attention_rs::ops::NonZeroOp;
use config::Gemma3Config;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

// =========================================================================
//  Vision Components (SigLIP-like)
// =========================================================================

struct Gemma3VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: candle_nn::Embedding,
    num_patches: usize,
    embed_dim: usize,
}

impl Gemma3VisionEmbeddings {
    fn new(vb: VarBuilderX, config: &Gemma3Config, dtype: DType) -> Result<Self> {
        let embed_dim = config.vision_config.hidden_size;
        let patch_size = config.vision_config.patch_size;
        let image_size = config.vision_config.image_size;

        let patch_embedding = conv2d_no_bias(
            config.vision_config.num_channels,
            embed_dim,
            patch_size,
            Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;

        let num_patches = (image_size / patch_size).pow(2);
        let (position_embedding, _) = embedding(
            Some(num_patches),
            embed_dim,
            vb.pp("position_embedding"),
            dtype,
        )?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            num_patches,
            embed_dim,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // pixel_values: [Batch, Channels, Height, Width]
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;
        // Flatten: [Batch, EmbedDim, H', W'] -> [Batch, EmbedDim, NumPatches]
        let (b, c, h, w) = patch_embeds.dims4()?;
        let patch_embeds = patch_embeds.flatten_from(2)?.transpose(1, 2)?;

        // Create position ids
        let position_ids = Tensor::arange(0, self.num_patches as u32, &pixel_values.device())?
            .unsqueeze(0)?
            .broadcast_as((b, self.num_patches))?;

        let embeddings = (patch_embeds + self.position_embedding.forward(&position_ids)?)?;
        Ok(embeddings)
    }
}

struct Gemma3VisionEncoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl Gemma3VisionEncoderLayer {
    fn new(vb: VarBuilderX, comm: Rc<Comm>, config: &Gemma3Config, dtype: DType) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        // Vision Attention usually doesn't need rotary embeddings, just absolute pos provided in embeddings
        // We reuse your Attention struct but might need to disable RoPE for vision if the struct enforces it.
        // For this snippet, assuming Attention can handle "None" for RoPE or we pass a dummy.
        // Note: Real implementations often use a separate VisionAttention struct,
        // but for style consistency we try to reuse.
        let self_attn = Attention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            Arc::new(ScalingRotaryEmbedding::new_dummy(dtype, &vb.device())?), // Dummy RoPE
            &config.to_generic_config(), // Helper to map vision config to generic
            dtype,
        )?;

        let mlp = MLP::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("mlp").clone()
            },
            comm.clone(),
            config.vision_config.hidden_size,
            config.vision_config.intermediate_size,
            &None, // Vision usually not quantized in the same way
            &None,
            false,
            dtype,
            "gelu_pytorch_tanh", // Gemma often uses approximation
        )?;

        let input_layernorm = rms_norm(
            config.vision_config.hidden_size,
            config.vision_config.layer_norm_eps,
            vb.pp("input_layernorm"),
            dtype,
        )?;

        let post_attention_layernorm = rms_norm(
            config.vision_config.hidden_size,
            config.vision_config.layer_norm_eps,
            vb.pp("post_attention_layernorm"),
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        // Vision encoder is usually bidirectional, no causal mask
        // We pass None for mask to imply full attention
        let attn_output = self.self_attn.forward(&xs, None)?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
    }
}

struct Gemma3VisionTransformer {
    embeddings: Gemma3VisionEmbeddings,
    layers: Vec<Gemma3VisionEncoderLayer>,
    norm: NormX,
}

impl Gemma3VisionTransformer {
    fn new(vb: VarBuilderX, comm: Rc<Comm>, config: &Gemma3Config, dtype: DType) -> Result<Self> {
        let embeddings = Gemma3VisionEmbeddings::new(vb.pp("embeddings"), config, dtype)?;

        let mut layers = Vec::new();
        for i in 0..config.vision_config.num_hidden_layers {
            layers.push(Gemma3VisionEncoderLayer::new(
                vb.pp(&format!("encoder.layers.{}", i)),
                comm.clone(),
                config,
                dtype,
            )?);
        }

        let norm = rms_norm(
            config.vision_config.hidden_size,
            config.vision_config.layer_norm_eps,
            vb.pp("post_layernorm"),
            dtype,
        )?;

        Ok(Self {
            embeddings,
            layers,
            norm,
        })
    }

    fn forward(&self, pixel_values: &Tensor, _: Vec<(u32, u32)>) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(pixel_values)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.norm.forward(&xs)
    }
}

struct Gemma3MultiModalProjector {
    projector: ReplicatedLinear,
    norm: NormX,
    pool: AvgPool2d,
    patches: usize,
}

impl Gemma3MultiModalProjector {
    fn new(vb: VarBuilderX, config: &Gemma3Config, dtype: DType) -> Result<Self> {
        let projector = ReplicatedLinear::load_no_bias(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("linear"),
            &None,
            &None,
            dtype,
        )?;
        let norm = rms_norm(
            config.vision_config.hidden_size,
            config.vision_config.layer_norm_eps,
            vb.pp("mm_soft_emb_norm"),
            dtype,
        )?;

        let patches = config.vision_config.image_size / config.vision_config.patch_size;
        let kernel_size = patches / config.mm_tokens_per_image.isqrt();
        let avg_pool = AvgPool2d::new(kernel_size, kernel_size);

        Ok(Self {
            projector,
            norm,
            pool,
            patches,
        })
    }

    fn forward(&self, xs: &Tensor, image_sizes: Vec<(u32, u32)>) -> Result<Tensor> {
        let (bs, _, seqlen) = xs.dims3()?;
        let mut out = xs.transpose(1, 2)?;
        out = out
            .reshape((bs, seqlen, self.patches, self.patches))?
            .contiguous()?;
        out = self
            .avg_pool
            .forward(&out)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        out = self.norm.forward(&out)?;
        self.projector.forward(&out)
    }
}

// =========================================================================
//  Text Components (Gemma 3 Specifics)
// =========================================================================

pub struct Gemma3DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    pre_feedforward_layernorm: NormX,
    post_feedforward_layernorm: NormX,
    sliding_window: Option<usize>,
}

impl Gemma3DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Gemma3Config,
        layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let cfg = &config.text_config;

        // Logic for Alternating Sliding Window
        let sliding_window = if (layer_idx + 1) % cfg.sliding_window_pattern != 0 {
            Some(cfg.sliding_window)
        } else {
            None
        };

        // Note: You must ensure your `Attention` struct supports logit_softcapping
        // via a config injection or a dedicated field.
        let self_attn = Attention::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("self_attn").clone()
            },
            comm.clone(),
            rotary_emb,
            &config.to_generic_config_with_softcap(cfg.attn_logit_softcapping),
            dtype,
        )?;

        let mlp = MLP::new(
            if is_qvar_builder {
                vb.clone()
            } else {
                vb.pp("mlp").clone()
            },
            comm.clone(),
            cfg.hidden_size,
            cfg.intermediate_size,
            &None,
            &None,
            false,
            dtype,
            "gelu_pytorch_tanh", // Gemma standard
        )?;

        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
            dtype,
        )?;

        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            dtype,
        )?;

        let pre_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
            dtype,
        )?;
        let post_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            sliding_window,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        sliding_attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        // Select appropriate mask based on layer type
        let mask = if self.sliding_window.is_some() {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let attn_output = self
            .self_attn
            .forward(&xs, mask, positions, cache, input_metadata)?;

        let mut xs = self.post_attention_layernorm.forward(&attn_output)?;
        xs = (xs + residual)?;

        let residual = &xs;

        let mlp_input = self.pre_feedforward_layernorm.forward(&xs)?;

        let mut mlp_output = self.mlp.forward(&mlp_input)?;
        mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;

        residual + mlp_output
    }
}

// =========================================================================
//  Main Model
// =========================================================================

pub struct Gemma3ForConditionalGeneration {
    // Vision
    vision_tower: Option<Gemma3VisionTransformer>,
    multi_modal_projector: Option<Gemma3MultiModalProjector>,

    // Text
    embed_tokens: candle_nn::Embedding, // ScaledEmbedding logic needed inside
    layers: Vec<Gemma3DecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,

    // Metadata
    device: Device,
    config: Gemma3Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
}

impl Gemma3ForConditionalGeneration {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Gemma3Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let reporter = progress_reporter.clone();
        let is_qvar_builder = vb.is_qvar_builder();
        let text_cfg = &config.text_config;

        // 1. Text Embeddings
        // Gemma scales embeddings by sqrt(dim)
        let (embed_tokens, vocab_size) = embedding(
            Some(text_cfg.vocab_size),
            text_cfg.hidden_size,
            if is_qvar_builder {
                vb.pp("model.embed_tokens")
            } else {
                vb.pp("model.embed_tokens")
            },
            if is_qvar_builder || text_cfg.quant.is_some() {
                DType::F32
            } else {
                dtype
            },
        )?;

        // 2. Vision & Projector (Optional load based on config)
        let vision_tower = if config.has_vision {
            Some(Gemma3VisionTransformer::new(
                vb.pp("model.vision_tower"),
                comm.clone(),
                config,
                dtype,
            )?)
        } else {
            None
        };

        let multi_modal_projector = if config.has_vision {
            Some(Gemma3MultiModalProjector::new(
                vb.pp("model.multi_modal_projector"),
                config,
                dtype,
            )?)
        } else {
            None
        };

        // 3. RoPE
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder || text_cfg.quant.is_some() {
                DType::F32
            } else {
                dtype
            },
            &config.to_generic_config(),
            &vb.device(),
            is_rope_i,
        )?);

        // 4. Layers
        let mut layers = Vec::new();
        for i in 0..text_cfg.num_hidden_layers {
            let layer = Gemma3DecoderLayer::new(
                vb.pp(&format!("model.layers.{}", i)),
                comm.clone(),
                rotary_emb.clone(),
                config,
                i,
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        // 5. Final Norm & Head
        let norm = rms_norm(
            text_cfg.hidden_size,
            text_cfg.rms_norm_eps,
            vb.pp("model.norm"),
            dtype,
        )?;

        let lm_head = ReplicatedLinear::load_no_bias(
            text_cfg.hidden_size,
            vocab_size,
            if text_cfg.tie_word_embeddings {
                vb.pp("model.embed_tokens")
            } else {
                vb.pp("lm_head")
            },
            &None,
            &None,
            dtype,
        )?;

        Ok(Self {
            vision_tower,
            multi_modal_projector,
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

    fn vision_tower(
        &self,
        image_features: &Tensor,
        image_sizes: Vec<(u32, u32)>,
    ) -> Result<Tensor> {
        let image_outputs = self
            .vision_tower
            .as_ref()
            .unwrap()
            .forward(image_features, image_sizes.clone())?;
        let selected_image_feature = image_outputs;
        self.multi_modal_projector
            .as_ref()
            .unwrap()
            .forward(&selected_image_feature.squeeze(0)?, image_sizes)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<Tensor>,
        image_sizes: Option<Vec<(u32, u32)>>, // No needed since gemma3 uses fixed image input size
    ) -> Result<Tensor> {
        let text_cfg = &self.config.text_config;
        // 1. Prepare Text Embeddings (Scaled)
        let scale = (text_cfg.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids)? * scale)?;

        // vision projection and embedding
        if let Some(image_tensor) = &images {
            let image_mask = input_ids.eq(self.config.image_token_index as u32)?;
            let image_mask = image_mask
                .unsqueeze(candle_core::D::Minus1)?
                .broadcast_as(xs.shape().clone())?
                .to_dtype(DType::U32)?;

            let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            let image_features =
                self.vision_tower(&image_tensor.to_dtype(self.dtype)?, image_sizes.unwrap())?;

            let mut x_flat = xs.flatten_all()?;
            let image_flat = image_features.flatten_all()?;

            x_flat =
                x_flat.scatter_add(&indices, &(image_flat - x_flat.gather(&indices, 0)?)?, 0)?;
            xs = x_flat.reshape(xs.shape())?;
        }

        // Followings are language model logics
        // 3. Prepare Masks
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

        // Full Global Mask
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            None, // No window for global
            input_metadata.is_prefill,
        );

        // Sliding Window Mask
        let sliding_attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            Some(text_cfg.sliding_window),
            input_metadata.is_prefill,
        );

        // 4. Pass through layers
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    sliding_attention_mask.as_ref(),
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
                    sliding_attention_mask.as_ref(),
                    positions,
                    None,
                    input_metadata,
                )?
            }
        }

        // 5. Final Norm & Projection
        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }

        let mut xs = self.norm.forward(&xs)?;

        // Final Logit Softcapping (Gemma Specific)
        let logits = if self.is_qvar_builder {
            self.lm_head.forward(&xs)?
        } else {
            self.lm_head
                .forward(&xs.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)?
        };

        if let Some(cap) = text_cfg.final_logit_softcapping {
            // tanh(logits / cap) * cap
            let scaled = (logits / cap)?;
            let tanh = scaled.tanh()?;
            tanh * cap
        } else {
            Ok(logits)
        }
    }
}
