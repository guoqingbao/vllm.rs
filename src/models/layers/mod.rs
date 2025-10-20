pub mod attention;
pub mod distributed;
pub mod linear;
pub mod mask;
pub mod mlp;
pub mod moe;
pub mod others;
pub mod rotary_emb;
pub mod wna16;
use crate::utils::downloader::ModelPaths;
use crate::utils::gguf_varbuilder::VarBuilder as QVarBuilder;
use candle_core::DType;
use candle_core::{Device, Result};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use either::Either;

#[derive(Clone)]
pub struct VarBuilderX<'a>(pub Either<VarBuilder<'a>, QVarBuilder>);

impl VarBuilderX<'_> {
    pub fn new(
        model_pathes: &ModelPaths,
        is_gguf: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        assert!(
            !model_pathes.get_weight_filenames().is_empty(),
            "No weight files found!"
        );
        let weight_files = model_pathes.get_weight_filenames();
        if is_gguf {
            let vb = crate::utils::gguf_varbuilder::VarBuilder::from_gguf(
                weight_files[0].clone(),
                device,
            )?;
            Ok(Self(Either::Right(vb)))
        } else {
            let vb = unsafe {
                candle_nn::var_builder::ShardedSafeTensors::var_builder(
                    &weight_files,
                    dtype,
                    device,
                )?
            };
            Ok(Self(Either::Left(vb)))
        }
    }

    pub fn is_var_builder(&self) -> bool {
        matches!(self.0, Either::Left(_))
    }

    pub fn is_qvar_builder(&self) -> bool {
        matches!(self.0, Either::Right(_))
    }

    pub fn device(&self) -> Device {
        match &self.0 {
            Either::Left(vb) => vb.device().clone(),
            Either::Right(vb) => vb.device().clone(),
        }
    }

    pub fn pp(&self, name: &str) -> VarBuilderX {
        match &self.0 {
            Either::Left(vb) => VarBuilderX(Either::Left(vb.pp(name))),
            Either::Right(vb) => VarBuilderX(Either::Right(vb.pp(name))),
        }
    }
}
