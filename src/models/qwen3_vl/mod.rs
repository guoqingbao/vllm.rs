use parking_lot::RwLock;
use std::rc::Rc;
use std::sync::Arc;
pub mod config;
pub mod input;
pub mod vision;

use crate::models::layers::VarBuilderX;
use crate::models::qwen3::Qwen3ForCausalLM;
use crate::models::qwen3_5::Qwen3_5ForCausalLM;
use crate::models::qwen3_5_moe::Qwen3_5MoEForCausalLM;
use crate::models::qwen3_moe::Qwen3MoEForCausalLM;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use crate::{models::layers::distributed::Comm, utils::image::ImageData};
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use config::Qwen3VLConfig;
use vision::Qwen3VLVisionModel;

pub enum Qwen3TextModel {
    Dense(Qwen3ForCausalLM),
    MoE(Qwen3MoEForCausalLM),
    Dense35(Qwen3_5ForCausalLM),
    MoE35(Qwen3_5MoEForCausalLM),
}

#[allow(dead_code)]
pub struct Qwen3VLForConditionalGeneration {
    text_model: Qwen3TextModel,
    vision_model: Qwen3VLVisionModel,
    spatial_merge_size: usize,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

impl Qwen3VLForConditionalGeneration {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        assert!(
            config.extra_config_json.is_some(),
            "Invalid multimodel config file!"
        );
        let mut cfg: Qwen3VLConfig =
            serde_json::from_str(config.extra_config_json.as_ref().unwrap())
                .map_err(candle_core::Error::wrap)?;
        cfg.text_config = config.clone();

        crate::log_info!("Loading vision tower...");

        let vision_model =
            Qwen3VLVisionModel::new(&cfg.vision_config, &vb.pp("model.visual"), dtype, device)?;

        if cfg.quantization_config.is_some() {
            cfg.text_config.quantization_config = cfg.quantization_config.clone();
        }

        let arch = cfg
            .architectures
            .unwrap_or(vec!["Qwen3VLForConditionalGeneration".to_string()]);
        let arch = arch[0].as_str();
        crate::log_info!("Loading language model...");

        let mut config_text = config.clone();
        if cfg.quantization_config.is_some() {
            config_text.quantization_config = cfg.quantization_config.clone();
        }
        let next_is_moe = config_text
            .moe_cfg
            .as_ref()
            .and_then(|m| m.num_experts)
            .unwrap_or(0)
            > 0;

        let text_model = match arch {
            "Qwen3VLMoeForConditionalGeneration" => {
                Qwen3TextModel::MoE(Qwen3MoEForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    Some("model.language_model.".to_string()),
                )?)
            }
            "Qwen3_5MoeForConditionalGeneration" => {
                Qwen3TextModel::MoE35(Qwen3_5MoEForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    Some("model.language_model.".to_string()),
                )?)
            }
            "Qwen3_5ForConditionalGeneration" => {
                Qwen3TextModel::Dense35(Qwen3_5ForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    Some("model.language_model.".to_string()),
                )?)
            }
            "Qwen3NextForConditionalGeneration" if next_is_moe => {
                Qwen3TextModel::MoE35(Qwen3_5MoEForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    Some("model.language_model.".to_string()),
                )?)
            }
            "Qwen3NextForConditionalGeneration" => {
                Qwen3TextModel::Dense35(Qwen3_5ForCausalLM::new_with_prefix(
                    &vb,
                    comm.clone(),
                    &config_text,
                    dtype,
                    is_rope_i,
                    device,
                    progress_reporter,
                    Some("model.language_model.".to_string()),
                )?)
            }
            _ => Qwen3TextModel::Dense(Qwen3ForCausalLM::new_with_prefix(
                &vb,
                comm.clone(),
                &config_text,
                dtype,
                is_rope_i,
                device,
                progress_reporter,
                Some("model.language_model.".to_string()),
            )?),
        };

        Ok(Self {
            text_model,
            vision_model,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
            image_token_id: cfg.image_token_id,
            vision_start_token_id: cfg.vision_start_token_id,
            vision_end_token_id: cfg.vision_end_token_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        images: Option<&ImageData>,
    ) -> Result<Tensor> {
        let (mut input_embeds, dtype) = match &self.text_model {
            Qwen3TextModel::Dense(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::MoE(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::Dense35(m) => (m.embed_forward(input_ids)?, m.dtype()),
            Qwen3TextModel::MoE35(m) => (m.embed_forward(input_ids)?, m.dtype()),
        };
        let device = input_embeds.device().clone();
        let mut visual_pos_masks: Option<Tensor> = None;
        let mut deepstack_visual_embeds: Option<Vec<Tensor>> = None;

        if let Some(images) = images {
            let mut pixel_values = images.to_tensor_f32(&device)?.to_dtype(dtype)?;
            let mut patches = Vec::new();
            for (h, w) in &images.patches {
                patches.extend(vec![1, *h as u32, *w as u32]);
            }
            let mut image_grid_thw = Tensor::from_vec(patches, (images.patches.len(), 3), &device)?;
            let num_images = pixel_values.dim(0)?;
            assert!(
                num_images == image_grid_thw.dim(0)?,
                "Input image and patch dim mismatch!"
            );
            if images.image_idx > 0 && (images.image_idx as usize) < num_images {
                pixel_values = pixel_values.narrow(
                    0,
                    images.image_idx as usize,
                    num_images - images.image_idx as usize,
                )?;
                image_grid_thw = image_grid_thw.narrow(
                    0,
                    images.image_idx as usize,
                    num_images - images.image_idx as usize,
                )?;
                crate::log_warn!(
                    "Slicing images: start idx {} -> {:?}",
                    images.image_idx,
                    pixel_values.shape()
                );
            }

            let dims = pixel_values.dims();
            if dims.len() == 3 {
                pixel_values = pixel_values.reshape((dims[0] * dims[1], dims[2]))?;
            }
            let (image_embeds, deepstack_image_embeds) =
                self.vision_model.forward(&pixel_values, &image_grid_thw)?;

            let image_embeds = image_embeds
                .to_device(&device)?
                .to_dtype(input_embeds.dtype())?;
            let deepstack_image_embeds = deepstack_image_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(input_embeds.dtype()))
                .collect::<Result<Vec<_>>>()?;

            let image_mask = input_ids.eq(self.image_token_id as u32)?;
            visual_pos_masks = Some(image_mask.to_dtype(DType::U8)?);

            let image_mask = image_mask
                .unsqueeze(candle_core::D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;
            use attention_rs::ops::NonZeroOp;
            let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;
            if indices.shape().dim(0)? > 0 {
                let hidden = input_embeds.dim(D::Minus1)?;
                let indices_len = indices.shape().dim(0)?;
                if indices_len % hidden != 0 {
                    candle_core::bail!(
                        "image indices length {} not divisible by hidden size {}",
                        indices_len,
                        hidden
                    );
                }
                let tokens_in_chunk = indices_len / hidden;
                let total_tokens = image_embeds.dim(0)?;
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
                let image_embeds = if start > 0 || end < total_tokens {
                    image_embeds.narrow(0, start, tokens_in_chunk)?
                } else {
                    image_embeds
                };
                let deepstack_image_embeds = deepstack_image_embeds
                    .into_iter()
                    .map(|t| {
                        if start > 0 || end < total_tokens {
                            t.narrow(0, start, tokens_in_chunk)
                        } else {
                            Ok(t)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                let mut x_flat = input_embeds.flatten_all()?;
                let image_flat = image_embeds.flatten_all()?;

                x_flat = x_flat.scatter_add(
                    &indices,
                    &(image_flat - x_flat.gather(&indices, 0)?)?,
                    0,
                )?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
                deepstack_visual_embeds = Some(deepstack_image_embeds);
            } else {
                crate::log_info!(
                    "Skip image embedding because no image tokens found in this chunk!"
                );
            }
        }

        match &self.text_model {
            Qwen3TextModel::Dense(m) => m.forward_with_deepstack(
                &input_embeds,
                &positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::MoE(m) => m.forward_with_deepstack(
                &input_embeds,
                &positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::Dense35(m) => m.forward_with_deepstack(
                &input_embeds,
                &positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
            Qwen3TextModel::MoE35(m) => m.forward_with_deepstack(
                &input_embeds,
                &positions,
                kv_caches,
                input_metadata,
                true,
                &visual_pos_masks,
                &deepstack_visual_embeds,
            ),
        }
    }

    pub fn get_vocab_size(&self) -> usize {
        match &self.text_model {
            Qwen3TextModel::Dense(m) => m.get_vocab_size(),
            Qwen3TextModel::MoE(m) => m.get_vocab_size(),
            Qwen3TextModel::Dense35(m) => m.get_vocab_size(),
            Qwen3TextModel::MoE35(m) => m.get_vocab_size(),
        }
    }
}
