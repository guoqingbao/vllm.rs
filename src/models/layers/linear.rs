//! Linear layer (GGUF and unquantized safetensors)
use crate::models::layers::VarBuilderX;
use candle_core::quantized;
use candle_core::Module;
use candle_core::{
    quantized::{QMatMul, QTensor},
    DType, Result, Tensor,
};
use candle_nn::var_builder::Shard;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use either::Either;
use std::sync::Arc;

pub fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match *x.dims() {
            [b1, seq_len, _, _] => {
                if seq_len > 1 {
                    self.weight.broadcast_left((b1, seq_len))?.t()?
                } else {
                    self.weight.t()?
                }
            }
            [bsize, seq_len, _] => {
                if seq_len > 1 {
                    self.weight.broadcast_left(bsize)?.t()?
                } else {
                    self.weight.t()?
                }
            }
            _ => self.weight.t()?,
        };
        let x = match *x.dims() {
            [bsize, seq_len, dim1, dim2] => {
                if seq_len > 1 {
                    x.matmul(&w)?
                } else {
                    let wdim = w.dims()[w.dims().len() - 1];
                    x.reshape((bsize * seq_len, dim1, dim2))?
                        .matmul(&w)?
                        .reshape((bsize, seq_len, dim1, wdim))?
                }
            }
            [bsize, seq_len, dim] => {
                if seq_len > 1 {
                    x.matmul(&w)?
                } else {
                    let wdim = w.dims()[w.dims().len() - 1];
                    x.reshape((bsize * seq_len, dim))?
                        .matmul(&w)?
                        .reshape((bsize, seq_len, wdim))?
                }
            }
            _ => x.matmul(&w)?,
        };

        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
    let weight = if weight.dtype() != dtype {
        weight.to_dtype(dtype)?
    } else {
        weight
    };
    Ok(Linear::new(weight, None))
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
    let ws = if ws.dtype() != dtype {
        ws.to_dtype(dtype)?
    } else {
        ws
    };
    let bs = vb.get((out_dim,), "bias")?;
    let bs = if shard.world_size > 1 {
        let dim_size = bs.dim(0)?;
        let start = shard.rank * (dim_size / shard.world_size);
        bs.narrow(0, start, dim_size / shard.world_size)?
            .contiguous()?
    } else {
        bs
    };

    let bs = if bs.dtype() != dtype {
        bs.to_dtype(dtype)?
    } else {
        bs
    };

    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb, shard, dtype)
    } else {
        linear_no_bias(in_dim, out_dim, vb, shard, dtype)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    dtype: DType,
}

impl QLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: crate::utils::gguf_varbuilder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((out_dim, in_dim), "weight")?;
        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let b = vb.get(out_dim, "bias");
        let bias = if b.is_ok() {
            Some(b.unwrap().dequantize(vb.device())?)
        } else {
            None
        };
        Ok(Self {
            inner,
            bias,
            dtype: DType::F32,
        })
    }

    pub fn from_qparts_x(w: QTensor, b: Option<Tensor>, dtype: DType) -> Self {
        let bx = match b {
            Some(b_) => {
                if b_.dtype() != DType::F32 {
                    Some(b_.to_dtype(DType::F32).unwrap())
                } else {
                    Some(b_)
                }
            }
            _ => None,
        };

        Self {
            inner: QMatMul::QTensor(Arc::new(w)),
            bias: bx,
            dtype,
        }
    }

    //in-situ quantization
    pub fn from_linear_x(linear: Linear, quant: String) -> Self {
        use quantized::GgmlDType;
        let ggml_dtype = match quant.as_str() {
            "q4_0" => GgmlDType::Q4_0,
            "q4_1" => GgmlDType::Q4_1,
            "q5_0" => GgmlDType::Q5_0,
            "q5_1" => GgmlDType::Q5_1,
            "q8_0" => GgmlDType::Q8_0,
            "q2k" => GgmlDType::Q2K,
            "q3k" => GgmlDType::Q3K,
            "q4k" => GgmlDType::Q4K,
            "q5k" => GgmlDType::Q5K,
            "q6k" => GgmlDType::Q6K,
            _ => panic!("Unsupported GGML data type!"),
        };
        let weight = linear.weight();
        let qbias = linear.bias().cloned();
        let dtype = weight.dtype();
        let qtensor = QTensor::quantize(weight, ggml_dtype).unwrap();
        QLinear::from_qparts_x(qtensor, qbias, dtype)
    }

    pub fn inner(&mut self) -> &mut QMatMul {
        &mut self.inner
    }

    pub fn inner_ref(&self) -> &QMatMul {
        &self.inner
    }

    pub fn is_quant(&self) -> bool {
        matches!(self.inner, QMatMul::QTensor(_))
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xs = match *x.dims() {
            [bsize, seq_len, dim1, dim2] => {
                if seq_len > 1 {
                    x.to_dtype(DType::F32)?
                } else {
                    x.reshape((bsize, dim1, dim2))?.to_dtype(DType::F32)?
                }
            }
            [bsize, seq_len, dim] => {
                if seq_len > 1 {
                    x.to_dtype(DType::F32)?
                } else {
                    x.reshape((bsize, dim))?.to_dtype(DType::F32)?
                }
            }
            _ => x.to_dtype(DType::F32)?,
        };
        let xs = match *x.dims() {
            [bsize, seq_len, dim1, _] => {
                if seq_len > 1 {
                    QMatMul::forward(&self.inner, &xs)?
                } else {
                    QMatMul::forward(&self.inner, &xs)?.reshape((bsize, seq_len, dim1, ()))?
                }
            }
            [bsize, seq_len, _] => {
                if seq_len > 1 {
                    QMatMul::forward(&self.inner, &xs)?
                } else {
                    QMatMul::forward(&self.inner, &xs)?.reshape((bsize, seq_len, ()))?
                }
            }
            _ => QMatMul::forward(&self.inner, &xs)?,
        };

        if let Some(bias) = &self.bias {
            xs.broadcast_add(bias)?.to_dtype(self.dtype)
        } else {
            xs.to_dtype(self.dtype)
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearX(Either<Linear, QLinear>);

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.0 {
            Either::Left(ln) => ln.forward(x),
            Either::Right(ln) => ln.forward(x),
        }
    }
}

impl LinearX {
    pub fn new(weight: Tensor, bias: Option<Tensor>, quant: &Option<String>) -> Self {
        let ln = Linear::new(weight, bias);
        if let Some(quantized_type) = quant {
            LinearX(Either::Right(QLinear::from_linear_x(
                ln,
                quantized_type.clone(),
            )))
        } else {
            LinearX(Either::Left(ln))
        }
    }
}

pub fn linear_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shard: Shard,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    match &vb.0 {
        Either::Left(vb) => {
            let ln = linear(in_dim, out_dim, vb.clone(), shard, dtype)?;
            if let Some(quantized_type) = quant {
                Ok(LinearX(Either::Right(QLinear::from_linear_x(
                    ln,
                    quantized_type.clone(),
                ))))
            } else {
                Ok(LinearX(Either::Left(ln)))
            }
        }
        Either::Right(vb) => Ok(LinearX(Either::Right(QLinear::new(
            in_dim,
            out_dim,
            vb.clone(),
        )?))),
    }
}

pub fn linear_no_bias_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    match &vb.0 {
        Either::Left(vb) => {
            let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
            if let Some(quantized_type) = quant {
                Ok(LinearX(Either::Right(QLinear::from_linear_x(
                    ln,
                    quantized_type.clone(),
                ))))
            } else {
                Ok(LinearX(Either::Left(ln)))
            }
        }
        Either::Right(vb) => Ok(LinearX(Either::Right(QLinear::new(
            in_dim,
            out_dim,
            vb.clone(),
        )?))),
    }
}

pub fn linear_b_x(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilderX,
    shard: Shard,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    if bias {
        linear_x(in_dim, out_dim, vb, shard, quant, dtype)
    } else {
        linear_no_bias_x(in_dim, out_dim, vb, shard, quant, dtype)
    }
}
