use crate::models::layers::VarBuilderX;
use candle_core::{DType, IndexOp, Result, Tensor, WithDType};
use candle_nn::{var_builder::Shard, Module};
use candle_nn::{Embedding, LayerNorm, RmsNorm};
use either::Either;

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

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vb: VarBuilderX,
    bias: bool,
) -> Result<candle_nn::Conv2d> {
    let (ws, bs) = match vb.0 {
        Either::Left(v) => {
            let ws = v.get(
                (
                    out_channels,
                    in_channels / cfg.groups,
                    kernel_size,
                    kernel_size,
                ),
                "weight",
            )?;
            let bs = if bias {
                Some(v.get(out_channels, "bias")?)
            } else {
                None
            };
            (ws, bs)
        }
        _ => {
            todo!()
        }
    };

    Ok(candle_nn::Conv2d::new(ws, bs, cfg))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

pub struct Conv3dNoBias {
    conv2d_1: candle_nn::Conv2d,
    conv2d_2: candle_nn::Conv2d,
}

impl Conv3dNoBias {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: VarBuilderX,
    ) -> Result<Self> {
        use candle_nn::Conv2dConfig;
        let expected_shape = (
            out_channels,
            in_channels / cfg.groups,
            kernel_sizes[0],
            kernel_sizes[1],
            kernel_sizes[2],
        );
        let ws = match vb.0 {
            Either::Left(v) => v.get(expected_shape, "weight")?,
            _ => {
                panic!("Unsupported quantized format for conv3d")
            }
        };

        let w1 = ws.i((.., .., 0, .., ..))?;
        let w2 = ws.i((.., .., 1, .., ..))?;

        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
        };

        Ok(Self {
            conv2d_1: candle_nn::Conv2d::new(w1.contiguous()?, None, cfg),
            conv2d_2: candle_nn::Conv2d::new(w2.contiguous()?, None, cfg),
        })
    }

    pub fn weight(&self) -> Result<Tensor> {
        let w1 = self.conv2d_1.weight().clone().unsqueeze(2)?;
        let w2 = self.conv2d_2.weight().clone().unsqueeze(2)?;
        Tensor::cat(&[w1, w2], 2)
    }
}

impl Module for Conv3dNoBias {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs1 = xs.i((.., .., 0, .., ..))?;
        let xs2 = xs.i((.., .., 1, .., ..))?;

        (self.conv2d_1.forward(&xs1)? + self.conv2d_2.forward(&xs2)?)?.unsqueeze(2)
    }
}

pub fn masked_fill<D: WithDType>(xs: &Tensor, mask: &Tensor, value: D) -> Result<Tensor> {
    let on_true = Tensor::full(value, xs.shape(), xs.device())?.to_dtype(xs.dtype())?;
    let on_false = xs;
    let res = mask
        .broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)?;
    Ok(res)
}
