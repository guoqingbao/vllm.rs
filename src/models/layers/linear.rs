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

    pub fn from_qparts_x(w: QTensor, b: Option<Tensor>, dtype: DType) -> Result<Self> {
        let bx = match b {
            Some(b_) => Some(b_.to_dtype(DType::F32)?),
            _ => None,
        };

        Ok(Self {
            inner: Some(QMatMul::QTensor(Arc::new(w))),
            bias: bx,
            wna16: None,
            dtype,
        })
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
            "q40" | "q4_0" => GgmlDType::Q4_0,
            "q4" | "q41" | "q4_1" => GgmlDType::Q4_1,
            "q50" | "q5_0" => GgmlDType::Q5_0,
            "q5" | "q51" | "q5_1" => GgmlDType::Q5_1,
            "q8" | "q80" | "q8_0" => GgmlDType::Q8_0,
            "q2k" | "q2_k" => GgmlDType::Q2K,
            "q3k" | "q3_k" => GgmlDType::Q3K,
            "q4k" | "q4_k" => GgmlDType::Q4K,
            "q5k" | "q5_k" => GgmlDType::Q5K,
            "q6k" | "q6_k" => GgmlDType::Q6K,
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
        QLinear::from_qparts_x(qtensor, qbias, dtype)
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
        if let Some(wna16) = &self.wna16 {
            wna16.forward(x)
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
pub enum LinearX {
    Linear(Linear),
    QLinear(QLinear),
    LnFp8(LnFp8),
}

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(ln) => ln.forward(x),
            Self::QLinear(ln) => ln.forward(x),
            Self::LnFp8(ln) => ln.forward(x),
        }
    }
}

impl LinearX {
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(_) => {
                panic!("No supported!")
            }
            Self::QLinear(ln) => ln.indexed_moe_forward(x, ids),
            Self::LnFp8(_) => panic!("LnFp8 does not support indexed_moe_forward yet"),
        }
    }
}

impl LinearX {
    pub fn new(weight: Tensor, bias: Option<Tensor>, quant: &Option<String>) -> Result<Self> {
        let dtype = weight.dtype();
        let ln = Linear::new(weight, bias);
        if let Some(quantized_type) = quant {
            Ok(Self::QLinear(QLinear::from_linear_x(
                ln,
                quantized_type.clone(),
                dtype,
            )?))
        } else {
            Ok(Self::Linear(ln))
        }
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        match self {
            Self::Linear(_) => {
                panic!("Unquantized tensor unable to be dequantized!")
            }
            Self::QLinear(ln) => ln.dequantize(),
            Self::LnFp8(_) => panic!("LnFp8 unable to be dequantized"),
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
            if let Some(cfg) = quant_cfg {
                if cfg.quant_method == "fp8" {
                    let ln = LnFp8::new(in_dim, out_dim, vb.clone(), shards, cfg)?;
                    return Ok(LinearX::LnFp8(ln));
                }

                let wna16 = WNA16::new(
                    in_dim,
                    out_dim,
                    vb.clone(),
                    shards,
                    quant_cfg,
                    true,
                    dtype,
                    true,
                )?;
                let ln = QLinear {
                    inner: None,
                    wna16: Some(wna16),
                    bias: None,
                    dtype,
                };
                Ok(LinearX::QLinear(ln))
            } else {
                let ln = linear(in_dim, out_dim, vb.clone(), shards, dtype)?;
                if let Some(quantized_type) = quant {
                    Ok(LinearX::QLinear(QLinear::from_linear_x(
                        ln,
                        quantized_type.clone(),
                        dtype,
                    )?))
                } else {
                    Ok(LinearX::Linear(ln))
                }
            }
        }
        Either::Right(vb) => Ok(LinearX::QLinear(QLinear::new(
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?)),
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
            if let Some(cfg) = quant_cfg {
                if cfg.quant_method == "fp8" {
                    let ln = LnFp8::new(in_dim, out_dim, vb.clone(), shards, cfg)?;
                    return Ok(LinearX::LnFp8(ln));
                }

                let wna16 = WNA16::new(
                    in_dim,
                    out_dim,
                    vb.clone(),
                    shards,
                    quant_cfg,
                    false,
                    dtype,
                    true,
                )?;
                let ln = QLinear {
                    inner: None,
                    wna16: Some(wna16),
                    bias: None,
                    dtype,
                };
                Ok(LinearX::QLinear(ln))
            } else {
                let ln = linear_no_bias(in_dim, out_dim, vb.clone(), shards, dtype)?;
                if let Some(quantized_type) = quant {
                    Ok(LinearX::QLinear(QLinear::from_linear_x(
                        ln,
                        quantized_type.clone(),
                        dtype,
                    )?))
                } else {
                    Ok(LinearX::Linear(ln))
                }
            }
        }
        Either::Right(vb) => Ok(LinearX::QLinear(QLinear::new(
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?)),
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
                Ok(LinearX::QLinear(QLinear::from_linear_x(
                    ln,
                    quantized_type.clone(),
                    dtype,
                )?))
            } else {
                Ok(LinearX::Linear(ln))
            }
        }
        Either::Right(vb) => Ok(LinearX::QLinear(QLinear::new_fused(
            num_experts,
            in_dim,
            out_dim,
            vb.clone(),
            shards,
            dtype,
        )?)),
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

#[derive(Debug, Clone)]
pub struct LnFp8 {
    pub weight: Tensor,
    pub weight_scale: Tensor,
    pub bias: Option<Tensor>,
    pub weight_block_size: Vec<usize>,
}

impl LnFp8 {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        shard: Shard,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        // Expected format:
        // weight: [out_dim, in_dim]
        // weight_scale: [out_dim, in_dim // block_size[1]]  (assuming block_size_y = 1)
        // Or weight_scale: [out_dim // block_size_y, in_dim // block_size_x]

        let block_size = quant_cfg
            .weight_block_size
            .clone()
            .unwrap_or(vec![128, 128]);
        if block_size.len() != 2 {
            candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
        }

        let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
        let weight = if weight.dtype() != DType::U8 {
            weight.to_dtype(DType::U8)?
        } else {
            weight
        };

        let by = block_size[0];
        let bx = block_size[1];

        let scale_dim0 = (out_dim + by - 1) / by;
        let scale_dim1 = (in_dim + bx - 1) / bx;

        let weight_scale = match vb.get_with_hints((scale_dim0, scale_dim1), "weight_scale", shard)
        {
            Ok(s) => s,
            Err(_) => vb
                .get_with_hints((scale_dim0, scale_dim1), "weight_scale_inv", shard)
                .map_err(|_| {
                    candle_core::Error::Msg(
                        "LnFp8: Missing weight_scale or weight_scale_inv".into(),
                    )
                })?,
        }
        .to_dtype(DType::F32)?;

        #[cfg(feature = "cutlass")]
        let weight_scale = weight_scale.t()?.contiguous()?;

        // Load bias if present
        let bias = vb.get((out_dim,), "bias");
        let bias = if bias.is_ok() {
            let bs = bias.unwrap();
            let bs = if shard.world_size > 1 {
                let dim_size = bs.dim(0)?;
                let start = shard.rank * (dim_size / shard.world_size);
                bs.narrow(0, start, dim_size / shard.world_size)?
                    .contiguous()?
            } else {
                bs
            };
            Some(bs)
        } else {
            None
        };

        Ok(Self {
            weight,
            weight_scale,
            bias,
            weight_block_size: block_size,
        })
    }
}

impl Module for LnFp8 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [Batch, Seq, InDim] or [Batch, InDim]
        // Flatten inputs to [M, K]
        let (b_sz, seq_len, in_dim) = match x.dims() {
            [b, s, d] => (*b, *s, *d),
            [b, d] => (*b, 1, *d),
            _ => candle_core::bail!("LnFp8: Input should be 2D or 3D"),
        };

        let m = b_sz * seq_len;
        let k = in_dim;

        let x_2d = x.reshape((m, k))?;

        // Call FP8 matmul
        #[cfg(feature = "cutlass")]
        let out = attention_rs::fp8_linear::fp8_matmul_cutlass(
            &x_2d,
            &self.weight.t()?,
            &self.weight_scale,
            &self.weight_block_size,
        )?;
        
        #[cfg(not(feature = "cutlass"))]
        let out = attention_rs::fp8_linear::fp8_matmul(
            &x_2d,
            &self.weight,
            &self.weight_scale,
            &self.weight_block_size,
        )?;

        // Reshape output back
        let (_, out_dim) = out.dims2()?;
        let out = if seq_len > 1 {
            out.reshape((b_sz, seq_len, out_dim))?
        } else {
            out
        };

        // Add bias
        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}
