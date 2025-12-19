use crate::models::glm4::GLM4ForCausalLM;
use crate::models::layers::distributed::Comm;
use crate::models::layers::VarBuilderX;
pub use crate::models::qwen3_vl::vision::Qwen3VLVisionModel as Glm4VVisionModel;
use crate::utils::config::Config;
use crate::utils::image::ImageData;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use parking_lot::RwLock;
use std::rc::Rc;
use std::sync::Arc;
pub mod config;
use config::Glm4VConfig;

pub struct Glm4VForConditionalGeneration {
    text_model: GLM4ForCausalLM,
    vision_model: Glm4VVisionModel,
    image_token_id: u32,
}

impl Glm4VForConditionalGeneration {
    #[allow(clippy::too_many_arguments)]
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
        let mut cfg: Glm4VConfig = serde_json::from_str(config.extra_config_json.as_ref().unwrap())
            .map_err(candle_core::Error::wrap)?;
        cfg.text_config = config.clone();

        let vision_model =
            Glm4VVisionModel::new(&cfg.vision_config, &vb.pp("model.visual"), dtype, device)?;

        if cfg.quantization_config.is_some() {
            cfg.text_config.quantization_config = cfg.quantization_config.clone();
        }

        let text_model = GLM4ForCausalLM::new_with_prefix(
            &vb,
            comm.clone(),
            &cfg.text_config,
            dtype,
            is_rope_i,
            device,
            progress_reporter,
            Some("model.language_model".to_string()),
        )?;

        Ok(Self {
            text_model,
            vision_model,
            image_token_id: cfg.image_token_id,
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
        let (mut input_embeds, dtype) = (
            self.text_model.embed_forward(input_ids)?,
            self.text_model.dtype(),
        );
        let device = input_embeds.device().clone();

        if let Some(images) = &images {
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
            let (image_embeds, _) = self.vision_model.forward(&pixel_values, &image_grid_thw)?;

            let image_embeds = image_embeds
                .to_device(&device)?
                .to_dtype(input_embeds.dtype())?;

            let image_mask = input_ids.eq(self.image_token_id as u32)?;
            let image_mask = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;
            use attention_rs::ops::NonZeroOp;
            let indices = image_mask.flatten_all()?.nonzero()?.squeeze(1)?;

            let mut x_flat = input_embeds.flatten_all()?;
            let image_flat = image_embeds.flatten_all()?;

            x_flat =
                x_flat.scatter_add(&indices, &(image_flat - x_flat.gather(&indices, 0)?)?, 0)?;
            input_embeds = x_flat.reshape(input_embeds.shape())?;
        }

        self.text_model
            .forward(&input_embeds, positions, kv_caches, input_metadata, true)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.text_model.get_vocab_size()
    }
}
