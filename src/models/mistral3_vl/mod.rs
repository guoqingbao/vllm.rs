use crate::models::layers::distributed::{
    Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::others::{rms_norm, NormX};
use crate::models::layers::VarBuilderX;
use crate::models::{self, llama::LLaMaForCausalLM};
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor, D};
use parking_lot::RwLock;
use std::rc::Rc;
use std::sync::Arc;
mod config;
mod vision;
pub use config::Mistral3Config;
use vision::VisionModel;

struct PatchMerger {
    merging_layer: ReplicatedLinear,
    spatial_merge_size: usize,
    patch_size: usize,
}

impl PatchMerger {
    fn new(cfg: &Mistral3Config, vb: VarBuilderX, dtype: DType) -> Result<Self> {
        Ok(Self {
            merging_layer: ReplicatedLinear::load_no_bias(
                cfg.vision_config.hidden_size * cfg.spatial_merge_size.pow(2),
                cfg.vision_config.hidden_size,
                vb.pp("merging_layer"),
                &None,
                &None,
                dtype,
            )?,
            spatial_merge_size: cfg.spatial_merge_size,
            patch_size: cfg.vision_config.patch_size,
        })
    }

    fn forward(&self, image_features: &Tensor, image_sizes: Vec<(u32, u32)>) -> Result<Tensor> {
        let image_sizes = image_sizes
            .iter()
            .map(|&(h, w)| (h as usize / self.patch_size, w as usize / self.patch_size))
            .collect::<Vec<_>>();

        let tokens_per_image = image_sizes.iter().map(|&(h, w)| h * w).collect::<Vec<_>>();
        let d = image_features.dim(D::Minus1)?;

        let mut permuted_tensor = Vec::new();

        for (image_index, image_tokens) in image_features
            .split(&tokens_per_image, 0)?
            .iter()
            .enumerate()
        {
            let (h, w) = image_sizes[image_index];
            let image_grid = image_tokens
                .reshape((h, w, d))?
                .permute((2, 0, 1))?
                .unsqueeze(0)?;
            let grid = {
                let patches = image_grid
                    .unfold(2, self.spatial_merge_size, self.spatial_merge_size)?
                    .unfold(3, self.spatial_merge_size, self.spatial_merge_size)?;

                let patches = patches.permute((0, 1, 4, 5, 2, 3))?;
                patches.contiguous()?.reshape((
                    1,
                    d * self.spatial_merge_size * self.spatial_merge_size,
                    (),
                ))?
            };
            let grid = grid
                .reshape((d * self.spatial_merge_size.pow(2), ()))?
                .t()?;
            permuted_tensor.push(grid);
        }

        let image_features = Tensor::cat(&permuted_tensor, 0)?;

        self.merging_layer.forward(&image_features)
    }
}

struct MultiModalProjector {
    norm: NormX,
    linear_1: TensorParallelColumnLinear,
    linear_2: TensorParallelRowLinear,
    act: candle_nn::Activation,
    patch_merger: PatchMerger,
}

impl MultiModalProjector {
    fn new(cfg: &Mistral3Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        let norm = rms_norm(
            cfg.vision_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("norm"),
            if is_qvar_builder { DType::F32 } else { dtype },
        )?;
        let num_feature_layers = 1;
        let linear_1 = TensorParallelColumnLinear::load_with_hints(
            cfg.vision_config.hidden_size * num_feature_layers,
            cfg.text_config.hidden_size,
            cfg.multimodal_projector_bias,
            if is_qvar_builder {
                vb.pp("ln1")
            } else {
                vb.pp("linear_1")
            },
            comm.clone(),
            &cfg.text_config.quantization_config,
            &cfg.text_config.quant,
            dtype,
        )?;

        let linear_2 = TensorParallelRowLinear::load_with_hints(
            cfg.vision_config.hidden_size,
            cfg.text_config.hidden_size,
            if is_qvar_builder {
                vb.pp("ln2")
            } else {
                vb.pp("linear_2")
            },
            comm.clone(),
            &cfg.text_config.quantization_config,
            &cfg.text_config.quant,
            dtype,
        )?;

        let patch_merger = PatchMerger::new(cfg, vb.pp("patch_merger"), dtype)?;
        Ok(Self {
            norm,
            linear_1,
            linear_2,
            act: cfg.projector_hidden_act,
            patch_merger,
        })
    }

    fn forward(&self, image_features: &Tensor, image_sizes: Vec<(u32, u32)>) -> Result<Tensor> {
        let mut hidden_states = self.norm.forward(image_features)?;
        hidden_states = self.patch_merger.forward(&hidden_states, image_sizes)?;
        hidden_states = self.linear_1.forward(&hidden_states)?.apply(&self.act)?;
        self.linear_2.forward(&hidden_states)
    }
}

pub struct Mistral3ForConditionalGeneration {
    text_model: LLaMaForCausalLM,
    vision_model: VisionModel,
    mmproj: MultiModalProjector,
    cfg: Mistral3Config,
    dtype: DType,
}

impl Mistral3ForConditionalGeneration {
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
        let mut cfg: Mistral3Config =
            serde_json::from_str(config.extra_config_json.as_ref().unwrap())
                .map_err(candle_core::Error::wrap)?;
        cfg.text_config = config.clone();
        let vision_model = VisionModel::new(
            &cfg.vision_config,
            vb.pp("vision_tower"),
            comm.clone(),
            dtype,
        )?;
        let mmproj =
            MultiModalProjector::new(&cfg, vb.pp("multi_modal_projector"), comm.clone(), dtype)?;

        let text_model = LLaMaForCausalLM::new(
            &vb.pp("language_model"),
            comm.clone(),
            &cfg.text_config,
            dtype,
            is_rope_i,
            device,
            progress_reporter,
        )?;

        assert_eq!(cfg.vision_feature_layer, -1);

        Ok(Self {
            vision_model,
            text_model,
            mmproj,
            cfg: cfg.clone(),
            dtype,
        })
    }

    fn get_image_features(
        &self,
        image_features: &Tensor,
        image_sizes: Vec<(u32, u32)>,
    ) -> Result<Tensor> {
        let image_outputs = self
            .vision_model
            .forward(image_features, image_sizes.clone())?;
        let selected_image_feature = image_outputs;
        self.mmproj
            .forward(&selected_image_feature.squeeze(0)?, image_sizes)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        pixel_values: Option<Tensor>,
        image_sizes: Option<Vec<(u32, u32)>>,
    ) -> Result<Tensor> {
        let input_embeds = self.text_model.embed_forward(input_ids)?;

        if let Some(pixel_values) = pixel_values {
            let special_image_mask = input_ids
                .eq(self.cfg.image_token_index as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;
            let mask_flat = special_image_mask.flatten_all()?;
            // Nonzero before vision model to allow async processing all the way through logits.
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            let image_sizes = image_sizes.unwrap();
            let image_features =
                self.get_image_features(&pixel_values.to_dtype(self.dtype)?, image_sizes)?;

            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = image_features.flatten_all()?;

            let current_vals = x_flat.gather(&indices, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&indices, &diff, 0)?;

            input_embeds = x_flat.reshape(input_embeds.shape())?;
        }

        self.text_model.forward(
            &input_embeds,
            positions,
            kv_caches,
            input_metadata,
            pixel_values.is_some(),
        )
    }

    pub fn get_vocab_size(&self) -> usize {
        panic!("not impl")
    }
}
