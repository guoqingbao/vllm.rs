pub mod attention;
pub mod distributed;
pub mod linear;
pub mod mask;
pub mod mlp;
pub mod others;
pub mod rotary_emb;
use crate::utils::gguf_varbuilder::VarBuilder as QVarBuilder;
use candle_core::DType;
use candle_core::{Device, Result};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use either::Either;
use std::path::Path;
#[derive(Clone)]
pub struct VarBuilderX<'a>(Either<VarBuilder<'a>, QVarBuilder>);
use crate::utils::hub_load_local_safetensors;

impl VarBuilderX<'_> {
    pub fn new(model_path: &String, dtype: DType, device: &Device) -> Result<Self> {
        let mut is_gguf: bool = false;
        let model_path = model_path.clone();
        let weight_files = if Path::new(&model_path)
            .join("model.safetensors.index.json")
            .exists()
        {
            hub_load_local_safetensors(&model_path, "model.safetensors.index.json")?
        } else if Path::new(&model_path).join("model.safetensors").exists() {
            vec![Path::new(&model_path).join("model.safetensors")]
        } else if Path::new(&model_path).exists() {
            is_gguf = true;
            vec![Path::new(&model_path).into()]
        } else {
            candle_core::bail!("Safetensors files not found in path {}", model_path);
        };

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
