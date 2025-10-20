//! Linear layer (GGUF and unquantized safetensors)
use super::wna16::WNA16;
use crate::models::layers::VarBuilderX;
use crate::utils::config::QuantConfig;
use candle_core::quantized;
use candle_core::quantized::GgmlDType;
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

pub fn linear_no_bias_merged(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shards: Shard,
    dtype: DType,
) -> Result<Linear> {
    let sd = shard(shards.dim + 1, shards.rank, shards.world_size);
    let weight = vb.get_with_hints((num_experts, out_dim, in_dim), "weight", sd)?;
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
    let bs = vb.get((out_dim,), "bias");
    let bs = if bs.is_ok() {
        let bs = bs.unwrap();
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
        Some(bs)
    } else {
        None
    };

    Ok(Linear::new(ws, bs))
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
    pub inner: Option<QMatMul>,
    pub bias: Option<Tensor>,
    pub wna16: Option<WNA16>,
    pub dtype: DType,
}

impl QLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: crate::utils::gguf_varbuilder::VarBuilder,
        shards: Shard,
        dtype: DType,
    ) -> Result<Self> {
        let ws = vb.get((out_dim, in_dim), "weight")?;
        let mut wdtype = ws.dtype();
        let ws = if shards.world_size > 1 {
            let ws = ws.dequantize_f16(&vb.device())?;
            let chunk_size = ws.shape().dims()[shards.dim] / shards.world_size;
            if chunk_size % wdtype.block_size() != 0 {
                // crate::log_warn!(
                //     "Invalid dim_size to chunk {} (start {}, size {}) for block_size {}, switching to Q8_0 format!",
                //     ws.shape().dims()[shards.dim],
                //     shards.rank * chunk_size,
                //     chunk_size,
                //     wdtype.block_size()
                // );
                wdtype = GgmlDType::Q8_0;
            }

            let ws = ws
                .narrow(shards.dim, shards.rank * chunk_size, chunk_size)?
                .contiguous()?;
            let qtensor = QTensor::quantize(&ws, wdtype)?;
            Arc::new(qtensor)
        } else {
            ws.to_owned()
        };
        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let b = vb.get(out_dim, "bias");
        let bias = if b.is_ok() {
            let bw = b.unwrap().dequantize(vb.device())?;
            if shards.world_size > 1 {
                let bw_chunk = bw.dim(0)? / shards.world_size;
                Some(
                    bw.narrow(0, shards.rank * bw_chunk, bw_chunk)?
                        .contiguous()?,
                )
            } else {
                Some(bw)
            }
        } else {
            None
        };
        Ok(Self {
            inner: Some(inner),
            bias,
            wna16: None,
            dtype,
        })
    }

    pub fn new_fused(
        num_experts: usize,
        in_dim: usize,
        out_dim: usize,
        vb: crate::utils::gguf_varbuilder::VarBuilder,
        shards: Shard,
        dtype: DType,
    ) -> Result<Self> {
        let ws = vb.get((num_experts, out_dim, in_dim), "weight")?;
        let wdtype = ws.dtype();
        let ws = if shards.world_size > 1 {
            let ws = ws.dequantize_f16(&vb.device())?;
            let chunk_size = ws.shape().dims()[shards.dim + 1] / shards.world_size;
            assert!(
                chunk_size % wdtype.block_size() == 0,
                "chunk position invalid dim {}, start {}, size {}",
                ws.shape().dims()[shards.dim],
                shards.rank * chunk_size,
                chunk_size
            );
            let ws = ws
                .narrow(shards.dim + 1, shards.rank * chunk_size, chunk_size)?
                .contiguous()?;
            let qtensor = QTensor::quantize(&ws, wdtype)?;
            Arc::new(qtensor)
        } else {
            ws.to_owned()
        };

        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let b = vb.get(out_dim, "bias");
        let bias = if b.is_ok() {
            let bw = b.unwrap().dequantize(vb.device())?;
            if shards.world_size > 1 {
                let bw_chunk = bw.dim(0)? / shards.world_size;
                Some(
                    bw.narrow(0, shards.rank * bw_chunk, bw_chunk)?
                        .contiguous()?,
                )
            } else {
                Some(bw)
            }
        } else {
            None
        };
        Ok(Self {
            inner: Some(inner),
            bias,
            wna16: None,
            dtype,
        })
    }

    pub fn from_qparts_x(w: QTensor, b: Option<Tensor>, dtype: DType) -> Self {
        let bx = match b {
            Some(b_) => Some(b_.to_dtype(DType::F32).unwrap()),
            _ => None,
        };

        Self {
            inner: Some(QMatMul::QTensor(Arc::new(w))),
            bias: bx,
            wna16: None,
            dtype,
        }
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        match &self.inner {
            Some(QMatMul::QTensor(t)) => t.dequantize(&t.device()),
            _ => {
                panic!("Not supported!");
            }
        }
    }
    //in-situ quantization
    pub fn from_linear_x(linear: Linear, quant: String, dtype: DType) -> Result<Self> {
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
        if weight.dim(candle_core::D::Minus1)? % ggml_dtype.block_size() != 0 {
            crate::log_error!(
                "Unable to quantize weight {:?} into gguf dtype {:?} \
                because the last dim is not divisible to block size {}! \
                \n\n\t***Tips: use '--isq q8_0' instead since it has smaller block size of 32!",
                weight.shape(),
                ggml_dtype,
                ggml_dtype.block_size()
            );
        }
        let qtensor = QTensor::quantize(weight, ggml_dtype)?;
        Ok(QLinear::from_qparts_x(qtensor, qbias, dtype))
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
        if self.wna16.is_some() {
            self.wna16_forward(x)
        } else if let Some(inner) = &self.inner {
            let xs = if x.dtype() != DType::F32 {
                x.to_dtype(DType::F32)?
            } else {
                x.to_owned()
            };
            let xs = QMatMul::forward(inner, &xs)?;

            if let Some(bias) = &self.bias {
                xs.broadcast_add(bias)
            } else {
                Ok(xs)
            }
        } else {
            candle_core::bail!("Invalid quantization type!")
        }
    }
}

impl QLinear {
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        if let Some(inner) = &self.inner {
            let xs = inner.indexed_moe_forward(&x.to_dtype(DType::F32)?, ids)?;
            if let Some(bias) = &self.bias {
                xs.broadcast_add(bias)?.to_dtype(self.dtype)
            } else {
                xs.to_dtype(self.dtype)
            }
        } else {
            candle_core::bail!("Invalid quantization type!")
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearX(pub Either<Linear, QLinear>);

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.0 {
            Either::Left(ln) => ln.forward(x),
            Either::Right(ln) => ln.forward(x),
        }
    }
}

impl LinearX {
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match &self.0 {
            Either::Left(_) => {
                panic!("No supported!")
            }
            Either::Right(ln) => ln.indexed_moe_forward(x, ids),
        }
    }
}

impl LinearX {
    pub fn new(weight: Tensor, bias: Option<Tensor>, quant: &Option<String>) -> Self {
        let dtype = weight.dtype();
        let ln = Linear::new(weight, bias);
        if let Some(quantized_type) = quant {
            LinearX(Either::Right(
                QLinear::from_linear_x(ln, quantized_type.clone(), dtype).unwrap(),
            ))
        } else {
            LinearX(Either::Left(ln))
        }
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        match &self.0 {
            Either::Left(_) => {
                panic!("Unquantized tensor unable to be dequantized!")
            }
            Either::Right(ln) => ln.dequantize(),
        }
    }
}

pub fn linear_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    match &vb.0 {
        Either::Left(vb) => {
            if quant_cfg.is_some() {
                let wna16 = QLinear::new_w4a16(
                    in_dim,
                    out_dim,
                    vb.clone(),
                    shards,
                    quant_cfg,
                    true,
                    dtype,
                )?;
                let ln = QLinear {
                    inner: None,
                    wna16: Some(wna16),
                    bias: None,
                    dtype,
                };
                Ok(LinearX(Either::Right(ln)))
            } else {
                let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                if let Some(quantized_type) = quant {
                    Ok(LinearX(Either::Right(QLinear::from_linear_x(
                        ln,
                        quantized_type.clone(),
                        dtype,
                    )?)))
                } else {
                    Ok(LinearX(Either::Left(ln)))
                }
            }
        }
        Either::Right(vb) => Ok(LinearX(Either::Right(QLinear::new(
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?))),
    }
}

pub fn linear_no_bias_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    match &vb.0 {
        Either::Left(vb) => {
            if quant_cfg.is_some() {
                let wna16 = QLinear::new_w4a16(
                    in_dim,
                    out_dim,
                    vb.clone(),
                    shards,
                    quant_cfg,
                    false,
                    dtype,
                )?;
                let ln = QLinear {
                    inner: None,
                    wna16: Some(wna16),
                    bias: None,
                    dtype,
                };
                Ok(LinearX(Either::Right(ln)))
            } else {
                let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                if let Some(quantized_type) = quant {
                    Ok(LinearX(Either::Right(QLinear::from_linear_x(
                        ln,
                        quantized_type.clone(),
                        dtype,
                    )?)))
                } else {
                    Ok(LinearX(Either::Left(ln)))
                }
            }
        }
        Either::Right(vb) => Ok(LinearX(Either::Right(QLinear::new(
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?))),
    }
}

pub fn linear_no_bias_merged_x(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilderX,
    shards: Shard,
    _: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    match &vb.0 {
        Either::Left(vb) => {
            let ln =
                linear_no_bias_merged(num_experts, in_dim, out_dim, vb.clone(), shards, dtype)?;
            if let Some(quantized_type) = quant {
                Ok(LinearX(Either::Right(QLinear::from_linear_x(
                    ln,
                    quantized_type.clone(),
                    dtype,
                )?)))
            } else {
                Ok(LinearX(Either::Left(ln)))
            }
        }
        Either::Right(vb) => Ok(LinearX(Either::Right(QLinear::new_fused(
            num_experts,
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?))),
    }
}

pub fn linear_b_x(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilderX,
    shard: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    if bias {
        linear_x(in_dim, out_dim, vb, shard, quant_cfg, quant, dtype)
    } else {
        linear_no_bias_x(in_dim, out_dim, vb, shard, quant_cfg, quant, dtype)
    }
}
