use crate::models::layers::linear::{
    linear_b_x as linear_b, linear_no_bias_x as linear, LinearX as Linear,
};
use crate::models::layers::VarBuilderX;
use crate::utils::config::QuantConfig;
#[cfg(feature = "nccl")]
pub use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, Id};
use candle_core::CustomOp1;
use candle_core::{CpuStorage, DType, Layout, Module, Result, Shape, Tensor};
use candle_nn::var_builder::Shard;
#[cfg(not(feature = "nccl"))]
pub struct Comm {}
#[cfg(not(feature = "nccl"))]
impl Comm {
    //dummy Comm
    pub fn default() -> Self {
        Self {}
    }
    pub fn dim(&self) -> usize {
        0
    }

    pub fn rank(&self) -> usize {
        0
    }
    pub fn world_size(&self) -> usize {
        1
    }
}

#[cfg(not(feature = "nccl"))]
#[derive(Debug, Clone, Copy)]
pub struct Id {
    pub internal: [::core::ffi::c_char; 128usize],
}

#[cfg(not(feature = "nccl"))]
impl Id {
    pub fn as_bytes(&self) -> &[u8] {
        // Safe reinterpretation of `c_char` as `u8`
        unsafe { std::slice::from_raw_parts(self.internal.as_ptr() as *const u8, 128) }
    }
}

pub use std::rc::Rc;

pub struct ReplicatedLinear {
    linear: Linear,
}

pub struct TensorParallelColumnLinear {
    linear: Linear,
}

impl TensorParallelColumnLinear {
    pub fn new(linear: Linear) -> Self {
        Self { linear }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

pub struct MergedParallelColumnLinear {
    linears: Vec<TensorParallelColumnLinear>,
    biases: Vec<Option<Tensor>>,
}

impl MergedParallelColumnLinear {
    pub fn new(linears: Vec<TensorParallelColumnLinear>) -> Self {
        Self {
            linears,
            biases: Vec::new(),
        }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut xss = Vec::<Tensor>::new();
        for i in 0..self.linears.len() {
            let mut xs = self.linears[i].forward(x)?;
            if let Some(bias) = &self.biases[i] {
                xs = xs.broadcast_add(bias)?;
            }
            xss.push(xs);
        }
        Ok(xss)
    }
}

#[allow(dead_code)]
pub struct TensorParallelRowLinear {
    linear: Linear,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    bias: Option<Tensor>,
    dtype: DType,
}

#[allow(dead_code)]
pub struct AllReduce {
    comm: Rc<Comm>,
}

unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl AllReduce {
    pub fn new(comm: Rc<Comm>) -> Self {
        Self { comm: comm.clone() }
    }
    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply_op1_no_bwd(self)
    }
}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is never used on cpu")
    }

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DeviceSlice;
        use candle_core::cuda_backend::cudarc::nccl::safe::ReduceOp;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::DType;
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_cuda_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

impl TensorParallelRowLinear {
    #[allow(unused_variables)]
    pub fn new(linear: Linear, comm: Rc<Comm>, dtype: DType) -> Self {
        #[cfg(feature = "nccl")]
        let all_reduce = if comm.world_size() > 1 {
            Some(AllReduce { comm })
        } else {
            None
        };
        Self {
            linear,
            #[cfg(feature = "nccl")]
            all_reduce,
            bias: None,
            dtype,
        }
    }

    #[allow(unused_variables)]
    pub fn new_with_bias(
        linear: Linear,
        bias: Option<Tensor>,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Self {
        #[cfg(feature = "nccl")]
        let all_reduce = if comm.world_size() > 1 {
            Some(AllReduce { comm })
        } else {
            None
        };
        Self {
            linear,
            #[cfg(feature = "nccl")]
            all_reduce,
            bias,
            dtype,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut xs = self.linear.forward(x)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            if xs.dtype() != self.dtype {
                //only bf16/fp16 supported in all reduce
                let xs_reduce = xs.to_dtype(self.dtype)?.apply_op1_no_bwd(all_reduce)?;
                xs = xs_reduce.to_dtype(xs.dtype())?
            } else {
                xs = xs.apply_op1_no_bwd(all_reduce)?;
            }
        }
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(&bias)?;
        }
        Ok(xs)
    }
}

pub fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

impl TensorParallelColumnLinear {
    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let linear = linear_b(
            in_dim,
            out_dim,
            bias,
            vb,
            shard(0, comm.rank(), comm.world_size()),
            quant_cfg,
            quant,
            dtype,
        )?;
        Ok(Self { linear })
    }

    pub fn load_with_shard(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        shard: Shard,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let linear = linear_b(in_dim, out_dim, bias, vb, shard, quant_cfg, quant, dtype)?;
        Ok(Self { linear })
    }
}

impl MergedParallelColumnLinear {
    pub fn load_merged_with_hints(
        in_dim: usize,
        out_dim: usize,
        chunk_dim: usize,
        chunks: usize,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        if quant.is_some() {
            candle_core::bail!("Merged quantized weight is not supported at the moment!");
        }
        let mut vec_linear = Vec::<TensorParallelColumnLinear>::new();
        for chunk_idx in 0..chunks {
            let linear = linear(
                in_dim,
                out_dim,
                vb.clone(),
                shard(
                    chunk_dim,
                    chunk_idx * comm.world_size() + comm.rank(),
                    comm.world_size() * chunks,
                ),
                quant_cfg,
                quant,
                dtype,
            )?;

            let ln = TensorParallelColumnLinear { linear };
            vec_linear.push(ln);
        }
        Ok(Self {
            linears: vec_linear,
            biases: vec![None; chunks],
        })
    }
}

impl TensorParallelRowLinear {
    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let linear = linear(
            in_dim,
            out_dim,
            vb,
            shard(1, comm.rank(), comm.world_size()),
            quant_cfg,
            quant,
            dtype,
        )?;
        Ok(Self::new(linear, comm, dtype))
    }
}

// pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
//     let weight = vb.get_with_hints(size, "weight", shard(0, 0, 1))?;
//     Ok(RmsNorm::new(weight, eps))
// }

// pub fn layer_norm(size: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<LayerNorm> {
//     let weight = vb.get_with_hints(size, "weight", Shard::default())?;
//     if affine {
//         Ok(LayerNorm::new(weight, vb.get(size, "bias")?, eps))
//     } else {
//         Ok(LayerNorm::new_no_bias(weight, eps))
//     }
// }

// pub fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
//     let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
//     Ok(Embedding::new(embeddings, hidden_size))
// }

impl ReplicatedLinear {
    pub fn from(linear: Linear) -> Result<Self> {
        Ok(Self { linear })
    }

    pub fn from_weight_bias(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let linear = Linear::new(weight, bias, &None);
        Ok(Self { linear })
    }

    pub fn load_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilderX,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let linear = linear(in_dim, out_dim, vb, shard(0, 0, 1), quant_cfg, quant, dtype)?;
        Ok(Self { linear })
    }

    pub fn load_b(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        if !bias {
            ReplicatedLinear::load_no_bias(in_dim, out_dim, vb, quant_cfg, quant, dtype)
        } else {
            let linear = linear_b(
                in_dim,
                out_dim,
                bias,
                vb,
                shard(0, 0, 1),
                quant_cfg,
                quant,
                dtype,
            )?;
            Ok(Self { linear })
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}
