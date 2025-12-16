use parking_lot::RwLock;
use std::rc::Rc;
use std::sync::Arc;
pub mod config;
pub mod input;
pub mod vision;
use crate::models::layers::distributed::Comm;
use crate::models::layers::VarBuilderX;
use crate::models::qwen3::Qwen3ForCausalLM;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use config::Qwen3VLConfig;
use vision::Qwen3VLVisionModel;

pub struct Qwen3VLForConditionalGeneration {
    text_model: Qwen3ForCausalLM,
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

        let vision_model = Qwen3VLVisionModel::new(
            &cfg.vision_config,
            vb.pp("model.visual"),
            &vb.device(),
            dtype,
        )?;

        if cfg.quantization_config.is_some() {
            cfg.text_config.quantization_config = cfg.quantization_config.clone();
        }
        let text_model = Qwen3ForCausalLM::new_with_prefix(
            &vb,
            comm.clone(),
            config,
            dtype,
            is_rope_i,
            device,
            progress_reporter,
            Some("model.language_model".to_string()),
        )?;
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
        images: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        continuous_img_pad: Option<Vec<(u32, u32)>>,
    ) -> Result<Tensor> {
        let mut input_embeds = self.text_model.embed_forward(input_ids)?;
        let (seq_len, hidden_dim) = input_embeds.dims2()?;
        let device = input_embeds.device().clone();

        let mut visual_pos_masks: Option<Tensor> = None;
        let mut deepstack_visual_embeds: Option<Vec<Tensor>> = None;

        if let (Some(pixel_values), Some(image_pad)) = (&images, &continuous_img_pad) {
            let Some(image_grid_thw_ref) = image_grid_thw.as_ref() else {
                candle_core::bail!("pixel_values require image_grid_thw");
            };
            let mut pixel_values = pixel_values.clone();
            let dims = pixel_values.dims();
            if dims.len() == 3 {
                pixel_values = pixel_values.reshape((dims[0] * dims[1], dims[2]))?;
            }
            let (image_embeds, deepstack_image_embeds) = self
                .vision_model
                .forward(&pixel_values, image_grid_thw_ref)?;
            let image_embeds = image_embeds
                .to_device(&device)?
                .to_dtype(self.text_model.dtype())?;
            let deepstack_image_embeds = deepstack_image_embeds
                .into_iter()
                .map(|t| t.to_device(&device)?.to_dtype(self.text_model.dtype()))
                .collect::<Result<Vec<_>>>()?;

            let mut offset = 0usize;
            let mut image_mask = Tensor::zeros((1, seq_len), DType::F32, input_ids.device())?;
            let total_expected: usize = image_pad
                .iter()
                .map(|(s, e)| *e as usize - *s as usize)
                .sum();
            if image_embeds.dim(0)? != total_expected {
                candle_core::bail!(
                    "Image embedding length {} does not match placeholder tokens {}",
                    image_embeds.dim(0)?,
                    total_expected
                );
            }

            for &(start, end) in image_pad {
                let (start, end) = (start as usize, end as usize);
                let len = end - start;
                let chunk = image_embeds.narrow(0, offset, len)?;
                offset += len;
                input_embeds = input_embeds
                    .slice_assign(&[0..1, start..end, 0..hidden_dim], &chunk.unsqueeze(0)?)?;
                let ones = Tensor::ones((1, len), DType::F32, input_ids.device())?;
                image_mask = image_mask.slice_assign(&[0..1, start..end], &ones)?;
            }
            visual_pos_masks = Some(image_mask.to_dtype(DType::U8)?);
            deepstack_visual_embeds = Some(deepstack_image_embeds);
        }

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

        let position_ids = if images.is_some() && visual_pos_masks.is_some() {
            let mut ropeidx_attn_mask_bs = Vec::new();
            let max_seqlens = *seqlens.iter().max().unwrap() as usize;
            for len in &seqlens {
                let len = *len as usize;
                ropeidx_attn_mask_bs.push(Tensor::new(
                    [vec![1f32; len], vec![0f32; max_seqlens - len]].concat(),
                    input_ids.device(),
                )?);
            }
            let ropeidx_attn_mask = Tensor::stack(&ropeidx_attn_mask_bs, 0)?;
            if input_metadata.is_prefill {
                use crate::models::layers::deepstack::ApplyRopeIndex;
                input_ids
                    .apply_rope_index(
                        image_grid_thw.as_ref(),
                        Some(&ropeidx_attn_mask),
                        self.spatial_merge_size,
                        self.image_token_id,
                        self.vision_start_token_id,
                        self.vision_end_token_id,
                    )?
                    .0
            } else {
                positions.to_owned()
            }
        } else {
            positions.to_owned()
        };

        let out = self.text_model.forward_with_deepstack(
            &input_embeds,
            &position_ids,
            kv_caches,
            input_metadata,
            true,
            &visual_pos_masks,
            &deepstack_visual_embeds,
        )?;
        Ok(out)
    }

    pub fn get_vocab_size(&self) -> usize {
        todo!()
    }
}
