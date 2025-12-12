use candle_core::{DType, Result, Tensor};
use candle_nn::{var_builder::Shard, Module};
use either::Either;
// use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use crate::models::layers::VarBuilderX;
use candle_nn::{Embedding, LayerNorm, RmsNorm};

pub struct NormX {
    norm: Either<RmsNorm, LayerNorm>,
    dtype: DType,
}
impl NormX {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let xs = if xs.dtype() != self.dtype {
            xs.to_dtype(self.dtype)?
        } else {
            xs.to_owned()
        };
        let xs = match &self.norm {
            Either::Left(norm) => norm.forward(&xs)?,
            Either::Right(norm) => norm.forward(&xs)?,
        };
        if xs.dtype() != in_dtype {
            xs.to_dtype(in_dtype)
        } else {
            Ok(xs)
        }
    }
}

pub fn rms_norm(
    size: usize,
    eps: f64,
    vb: VarBuilderX,
    dtype: DType,
    is_gemma: bool,
) -> Result<NormX> {
    let (weight, dtype) = match &vb.0 {
        Either::Left(vb) => {
            let ws = vb.get_with_hints(size, "weight", Shard::default())?;
            if ws.dtype() != dtype {
                (ws.to_dtype(dtype)?, dtype)
            } else {
                (ws, dtype)
            }
        }
        Either::Right(vb) => (vb.get(size, "weight")?.dequantize(vb.device())?, DType::F32),
    };

    let weight = if is_gemma { (weight + 1.0)? } else { weight };
    Ok(NormX {
        norm: Either::Left(RmsNorm::new(weight, eps)),
        dtype,
    })
}

pub fn layer_norm(
    size: usize,
    eps: f64,
    affine: bool,
    vb: VarBuilderX,
    dtype: DType,
) -> Result<NormX> {
    let (weight, dtype) = match &vb.0 {
        Either::Left(vb) => (
            vb.get_with_hints(size, "weight", Shard::default())?
                .to_dtype(dtype)?,
            dtype,
        ),
        Either::Right(vb) => (vb.get(size, "weight")?.dequantize(vb.device())?, DType::F32),
    };
    if affine {
        let bias = match &vb.0 {
            Either::Left(vb) => vb.get(size, "bias")?.to_dtype(dtype)?,
            Either::Right(vb) => vb.get(size, "bias")?.dequantize(vb.device())?,
        };
        Ok(NormX {
            norm: Either::Right(LayerNorm::new(weight, bias, eps)),
            dtype,
        })
    } else {
        Ok(NormX {
            norm: Either::Right(LayerNorm::new_no_bias(weight, eps)),
            dtype,
        })
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
                vb.get((vocab_size.unwrap(), hidden_size), "weight")?
                    .to_dtype(dtype)?,
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
            (weight, vocab_size)
        }
    };
    Ok((Embedding::new(embeddings, hidden_size), vocab_size))
}

pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vb: VarBuilderX,
) -> Result<candle_nn::Conv2d> {
    let ws = match vb.0 {
        Either::Left(v) => v.get(
            (
                out_channels,
                in_channels / cfg.groups,
                kernel_size,
                kernel_size,
            ),
            "weight",
        )?,
        _ => {
            todo!()
        }
    };
    Ok(candle_nn::Conv2d::new(ws, None, cfg))
}

pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.avg_pool2d_with_stride(self.kernel_size, self.stride)
    }
}
