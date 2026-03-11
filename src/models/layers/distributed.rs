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
use either::Either;
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
    output_splits: Option<Vec<usize>>,
}

impl MergedParallelColumnLinear {
    pub fn new(linears: Vec<TensorParallelColumnLinear>) -> Self {
        Self {
            linears,
            biases: Vec::new(),
            output_splits: None,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        if let Some(output_splits) = &self.output_splits {
            if self.linears.len() != 1 {
                candle_core::bail!(
                    "MergedParallelColumnLinear expected exactly 1 linear for split outputs, got {}",
                    self.linears.len()
                );
            }
            let mut ys = self.linears[0].forward(x)?;
            if let Some(Some(bias)) = self.biases.first() {
                ys = ys.broadcast_add(bias)?;
            }
            let split_dim = ys.dims().len().saturating_sub(1);
            let total_dim = ys.dim(split_dim)?;
            let expected_dim: usize = output_splits.iter().sum();
            if total_dim != expected_dim {
                candle_core::bail!(
                    "MergedParallelColumnLinear split mismatch: output dim {} vs expected {}",
                    total_dim,
                    expected_dim
                );
            }
            let mut outputs = Vec::with_capacity(output_splits.len());
            let mut start = 0usize;
            for split_size in output_splits {
                outputs.push(ys.narrow(split_dim, start, *split_size)?.contiguous()?);
                start += *split_size;
            }
            return Ok(outputs);
        }

        let mut xss = Vec::<Tensor>::new();
        for i in 0..self.linears.len() {
            let mut xs = self.linears[i].forward(x)?;
            if self.biases.len() > 0 && i < self.biases.len() {
                if let Some(bias) = &self.biases[i] {
                    xs = xs.broadcast_add(bias)?;
                }
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
        let start_offset = l.start_offset();

        let dst = match s.dtype() {
            DType::BF16 => {
                let full_slice = s.as_cuda_slice::<bf16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_reduce BF16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                // Slice to only the valid elements (handles narrow/view tensors)
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(&src_slice, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let full_slice = s.as_cuda_slice::<f16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_reduce F16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                // Slice to only the valid elements (handles narrow/view tensors)
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(&src_slice, &mut dst, &ReduceOp::Sum)
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

/// Determine local KV heads and shard mapping for tensor-parallel attention.
///
/// In replicated KV mode (`total_num_kv_heads < world_size`), query heads are sharded in
/// contiguous rank ranges, so KV head replication must follow the same contiguous grouping.
pub fn kv_head_shard(
    total_num_kv_heads: usize,
    rank: usize,
    world_size: usize,
) -> Result<(usize, Shard)> {
    if total_num_kv_heads == 0 {
        candle_core::bail!("num_kv_heads must be > 0");
    }
    if world_size == 0 {
        candle_core::bail!("tensor parallel world_size must be > 0");
    }
    if rank >= world_size {
        candle_core::bail!(
            "rank out of bounds for tensor parallel group (rank={}, world_size={})",
            rank,
            world_size
        );
    }

    if total_num_kv_heads >= world_size {
        if total_num_kv_heads % world_size != 0 {
            candle_core::bail!(
                "kv heads must be divisible by tensor parallel world_size when partitioned (num_kv_heads={}, world_size={})",
                total_num_kv_heads,
                world_size
            );
        }
        Ok((total_num_kv_heads / world_size, shard(0, rank, world_size)))
    } else {
        if world_size % total_num_kv_heads != 0 {
            candle_core::bail!(
                "tensor parallel world_size must be divisible by kv heads when kv heads are replicated (num_kv_heads={}, world_size={})",
                total_num_kv_heads,
                world_size
            );
        }
        let ranks_per_kv_head = world_size / total_num_kv_heads;
        let kv_head_rank = rank / ranks_per_kv_head;
        Ok((1, shard(0, kv_head_rank, total_num_kv_heads)))
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
            output_splits: None,
        })
    }

    pub fn load_merged_chunks(
        in_dim: usize,
        out_dim: usize,
        chunk_dim: usize,
        chunks: Vec<usize>,
        chunk_shards: Option<Vec<Shard>>,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let is_fp8_quant = if let Some(cfg) = quant_cfg {
            cfg.quant_method == "fp8"
        } else {
            false
        };

        if vb.is_qvar_builder() {
            candle_core::bail!(
                "Merged quantized weight is not supported for GGUF varbuilder at the moment!"
            );
        }
        if quant_cfg.is_some() && !is_fp8_quant {
            candle_core::bail!(
                "Merged quantized weight is not supported at the moment, using ISQ instead!"
            );
        }
        if let Some(chunk_shards) = &chunk_shards {
            if chunk_shards.len() != chunks.len() {
                candle_core::bail!(
                    "chunk_shards length mismatch: expected {}, got {}",
                    chunks.len(),
                    chunk_shards.len()
                );
            }
        }
        let mut vec_linear = Vec::<TensorParallelColumnLinear>::new();
        let mut output_splits: Option<Vec<usize>> = None;
        use crate::models::layers::linear::{LinearX, LnFp8, QLinear};
        match vb.0 {
            Either::Left(v) => {
                if is_fp8_quant {
                    if chunk_dim != 0 {
                        candle_core::bail!(
                            "FP8 merged chunk loading currently supports chunk_dim=0 only, got {}",
                            chunk_dim
                        );
                    }
                    let quant_cfg = quant_cfg.as_ref().ok_or_else(|| {
                        candle_core::Error::Msg(
                            "FP8 merged chunk loading requires quantization config".to_string(),
                        )
                    })?;

                    let block_size = quant_cfg
                        .weight_block_size
                        .clone()
                        .unwrap_or(vec![128, 128]);
                    if block_size.len() != 2 {
                        candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
                    }
                    let by = block_size[0];
                    let bx = block_size[1];
                    if by == 0 || bx == 0 {
                        candle_core::bail!("LnFp8: invalid zero in weight_block_size");
                    }

                    let weight = v.get_with_hints_dtype(
                        (out_dim, in_dim),
                        "weight",
                        Shard::default(),
                        DType::U8,
                    )?;
                    let scale_dim0 = (out_dim + by - 1) / by;
                    let scale_dim1 = (in_dim + bx - 1) / bx;
                    let weight_scale = match v.get_with_hints_dtype(
                        (scale_dim0, scale_dim1),
                        "weight_scale",
                        Shard::default(),
                        DType::F32,
                    ) {
                        Ok(s) => s,
                        Err(_) => v
                            .get_with_hints_dtype(
                                (scale_dim0, scale_dim1),
                                "weight_scale_inv",
                                Shard::default(),
                                DType::F32,
                            )
                            .map_err(|_| {
                                candle_core::Error::Msg(
                                    "LnFp8: Missing weight_scale or weight_scale_inv".into(),
                                )
                            })?,
                    };

                    #[cfg(feature = "cuda")]
                    let sm_version =
                        attention_rs::cuda_utils::sm_version(v.device().as_cuda_device()?)
                            .unwrap_or(0) as usize;

                    #[cfg(not(feature = "cuda"))]
                    let sm_version = 0;

                    let mut chunk_start = 0;
                    let mut local_weight_chunks = Vec::<Tensor>::with_capacity(chunks.len());
                    let mut local_scale_chunks = Vec::<Tensor>::with_capacity(chunks.len());
                    let mut local_output_splits = Vec::<usize>::with_capacity(chunks.len());
                    for chunk_idx in 0..chunks.len() {
                        let chunk_size = chunks[chunk_idx];
                        let ws = weight.narrow(0, chunk_start, chunk_size)?;
                        let chunk_shard = if let Some(chunk_shards) = &chunk_shards {
                            chunk_shards[chunk_idx]
                        } else {
                            shard(0, comm.rank(), comm.world_size())
                        };
                        if chunk_shard.dim != 0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} shard dim {} is unsupported, expected 0",
                                chunk_idx,
                                chunk_shard.dim
                            );
                        }
                        if ws.dim(0)? % chunk_shard.world_size != 0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} dim {} is not divisible by shard world_size {}",
                                chunk_idx,
                                ws.dim(0)?,
                                chunk_shard.world_size
                            );
                        }
                        let local_out = ws.dim(0)? / chunk_shard.world_size;
                        if local_out == 0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} produced empty shard",
                                chunk_idx
                            );
                        }
                        let local_out_start = chunk_start + chunk_shard.rank * local_out;
                        if local_out_start % by != 0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} local start {} is not aligned to block_size_y {}",
                                chunk_idx,
                                local_out_start,
                                by
                            );
                        }
                        let ws_chunk = ws
                            .narrow(0, chunk_shard.rank * local_out, local_out)?
                            .contiguous()?;
                        local_output_splits.push(local_out);

                        let scale_row_start = local_out_start / by;
                        let scale_rows = (local_out + by - 1) / by;
                        if scale_row_start + scale_rows > scale_dim0 {
                            candle_core::bail!(
                                "FP8 merged chunk {} scale slice out of bounds: start={}, rows={}, total={}",
                                chunk_idx,
                                scale_row_start,
                                scale_rows,
                                scale_dim0
                            );
                        }
                        let ws_scale = weight_scale
                            .narrow(0, scale_row_start, scale_rows)?
                            .contiguous()?;
                        local_weight_chunks.push(ws_chunk);
                        local_scale_chunks.push(ws_scale);
                        chunk_start += chunk_size;
                    }

                    let merged_weight = if local_weight_chunks.len() == 1 {
                        local_weight_chunks.remove(0)
                    } else {
                        let weight_refs = local_weight_chunks.iter().collect::<Vec<_>>();
                        Tensor::cat(&weight_refs, 0)?
                    };

                    let merged_scale = if local_scale_chunks.len() == 1 {
                        local_scale_chunks.remove(0)
                    } else {
                        let scale_refs = local_scale_chunks.iter().collect::<Vec<_>>();
                        Tensor::cat(&scale_refs, 0)?
                    };

                    #[cfg(feature = "cutlass")]
                    let merged_scale_cutlass = if sm_version >= 100 {
                        Some(merged_scale.t()?)
                    } else if sm_version >= 90 {
                        Some(merged_scale.t()?.contiguous()?)
                    } else {
                        None
                    };

                    #[cfg(not(feature = "cutlass"))]
                    let merged_scale_cutlass = None;

                    let linear = LinearX::LnFp8(LnFp8 {
                        weight: merged_weight,
                        weight_scale: merged_scale,
                        weight_scale_cutlass: merged_scale_cutlass,
                        bias: None,
                        weight_block_size: block_size.clone(),
                        sm_version,
                    });
                    let ln = TensorParallelColumnLinear { linear };
                    vec_linear.push(ln);
                    output_splits = Some(local_output_splits);
                } else {
                    let weight = v.get((out_dim, in_dim), "weight")?;
                    let weight = if weight.dtype() != dtype {
                        weight.to_dtype(dtype)?
                    } else {
                        weight
                    };
                    let mut chunk_start = 0;
                    for chunk_idx in 0..chunks.len() {
                        let chunk_size = chunks[chunk_idx];
                        let ws = weight.narrow(chunk_dim, chunk_start, chunk_size)?;
                        let ws_chunk = if let Some(chunk_shards) = &chunk_shards {
                            let chunk_shard = &chunk_shards[chunk_idx];
                            if ws.dim(0)? % chunk_shard.world_size != 0 {
                                candle_core::bail!(
                                    "merged chunk {} dim {} is not divisible by shard world_size {}",
                                    chunk_idx,
                                    ws.dim(0)?,
                                    chunk_shard.world_size
                                );
                            }
                            let c_chunk_size = ws.dim(0)? / chunk_shard.world_size;
                            ws.narrow(0, chunk_shard.rank * c_chunk_size, c_chunk_size)?
                                .contiguous()?
                        } else {
                            if ws.dim(0)? % comm.world_size() != 0 {
                                candle_core::bail!(
                                    "merged chunk {} dim {} is not divisible by comm world_size {}",
                                    chunk_idx,
                                    ws.dim(0)?,
                                    comm.world_size()
                                );
                            }
                            let c_chunk_size = ws.dim(0)? / comm.world_size();
                            ws.narrow(0, comm.rank() * c_chunk_size, c_chunk_size)?
                                .contiguous()?
                        };
                        chunk_start += chunk_size;

                        let ln = crate::models::layers::linear::Linear::new(ws_chunk, None);
                        let linear = if let Some(quantized_type) = quant {
                            let quantized_type = if chunk_idx == chunks.len() - 1 {
                                "q8_0".to_string()
                            } else {
                                quantized_type.clone()
                            };
                            LinearX::QLinear(QLinear::from_linear_x(ln, quantized_type, dtype)?)
                        } else {
                            LinearX::Linear(ln)
                        };
                        let ln = TensorParallelColumnLinear { linear };
                        vec_linear.push(ln);
                    }
                }
            }
            _ => {
                candle_core::bail!("Quantized qkv weight not supported!")
            }
        }

        let linear_count = vec_linear.len();
        Ok(Self {
            linears: vec_linear,
            biases: vec![None; linear_count],
            output_splits,
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

impl ReplicatedLinear {
    pub fn from(linear: Linear) -> Result<Self> {
        Ok(Self { linear })
    }

    pub fn from_weight_bias(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let linear = Linear::new(weight, bias, &None)?;
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
