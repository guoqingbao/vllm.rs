use candle_core::{DType, Result};
use candle_nn::var_builder::Shard;
use either::Either;
// use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use crate::models::layers::VarBuilderX;
use candle_nn::{Embedding, LayerNorm, RmsNorm};

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilderX, dtype: DType) -> Result<RmsNorm> {
    let weight = match &vb.0 {
        Either::Left(vb) => vb.get_with_hints(size, "weight", Shard::default())?,
        Either::Right(vb) => vb.get(size, "weight")?.dequantize(vb.device())?,
    };
    Ok(RmsNorm::new(weight.to_dtype(dtype)?, eps))
}

pub fn layer_norm(
    size: usize,
    eps: f64,
    affine: bool,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<LayerNorm> {
    let weight = match &vb.0 {
        Either::Left(vb) => vb.get_with_hints(size, "weight", Shard::default())?,
        Either::Right(vb) => vb.get(size, "weight")?.dequantize(vb.device())?,
    };
    if affine {
        let bias = match &vb.0 {
            Either::Left(vb) => vb.get(size, "bias")?,
            Either::Right(vb) => vb.get(size, "bias")?.dequantize(vb.device())?,
        };
        Ok(LayerNorm::new(
            weight.to_dtype(dtype)?,
            bias.to_dtype(dtype)?,
            eps,
        ))
    } else {
        Ok(LayerNorm::new_no_bias(weight.to_dtype(dtype)?, eps))
    }
}

pub fn embedding(
    vocab_size: Option<usize>,
    hidden_size: usize,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<(Embedding, usize)> {
    let (embeddings, vocab_size) = match &vb.0 {
        Either::Left(vb) => {
            assert!(
                vocab_size.is_some(),
                "vocab_size must be specified for safetensor models"
            );
            (
                vb.get((vocab_size.unwrap(), hidden_size), "weight")?,
                vocab_size.unwrap(),
            )
        }
        Either::Right(vb) => {
            let weight = if vocab_size.is_some() {
                vb.get((vocab_size.unwrap(), hidden_size), "weight")?
            } else {
                vb.get_no_shape("weight")?
            }
            .dequantize(vb.device())?;
            let vocab_size = vocab_size.unwrap_or(weight.dim(0)?);
            (weight.to_dtype(dtype)?, vocab_size)
        }
    };
    Ok((Embedding::new(embeddings, hidden_size), vocab_size))
}
