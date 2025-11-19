// src/models/layers/moe.rs
use crate::models::layers::distributed::{shard, AllReduce, Comm};
use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use attention_rs::moe;
use candle_core::Module;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Result, Tensor, D,
};
use candle_nn::var_builder::Shard;
use either::Either;
use std::rc::Rc;
use std::sync::Arc;

#[allow(dead_code)]
pub struct FusedMoe {
    gate: Linear,
    gate_up_w: Tensor,
    down_w: Tensor,
    w_size_n: usize,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoe {
    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        assert!(
            cfg.quantization_config.is_none(),
            "Invalid quantization format!"
        );
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &None,
            &None,
            dtype,
        )?;

        let experts_vb = vb.pp("experts");
        let mut gate_up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        //pack experts
        for i in 0..num_experts {
            let experts_vb = experts_vb.pp(format!("{}", i).as_str());

            let (gate_up_expert, down_expert) = match &experts_vb.0 {
                Either::Left(vb) => {
                    // n x k format
                    let gate_expert = vb.pp("gate_proj").get_with_hints(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                        shard(0, comm.rank(), comm.world_size()),
                    )?;
                    let up_expert = vb.pp("up_proj").get_with_hints(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                        shard(0, comm.rank(), comm.world_size()),
                    )?;
                    let down_expert = vb.pp("down_proj").get_with_hints(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                        "weight",
                        shard(1, comm.rank(), comm.world_size()),
                    )?;
                    //pack gate_proj and up_proj
                    let gate_up_expert = Tensor::cat(&[&gate_expert, &up_expert], 0)?;

                    (gate_up_expert, down_expert)
                }
                _ => candle_core::bail!("invalid varbuild or quant config!"),
            };

            gate_up_experts.push(gate_up_expert);
            down_experts.push(down_expert);
        }

        let gate_up_w = Tensor::stack(&gate_up_experts, 0)?;
        let down_w = Tensor::stack(&down_experts, 0)?;
        let world_size = comm.world_size();
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_w,
            down_w,
            w_size_n,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) = attention_rs::topk::topk_softmax(
            &router_logits.to_dtype(DType::F32)?,
            self.num_experts_per_tok,
        )?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        //out (M, top_k, N)
        let gate_up = moe::moe_gemm(
            &xs,
            &self.gate_up_w,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?;

        let gate = gate_up
            .narrow(candle_core::D::Minus1, 0, self.w_size_n)?
            .contiguous()?;
        let up = gate_up
            .narrow(candle_core::D::Minus1, self.w_size_n, self.w_size_n)?
            .contiguous()?;

        //(M * top_k, N // 2)
        let down_inputs = (up * gate.apply(&self.act)?)?.reshape(((), self.w_size_n))?;

        //view(M, top_k, K) -> sum -> (M, K)
        let mut ys = moe::moe_gemm(
            &down_inputs,
            &self.down_w,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}

pub struct FusedMoeGGUF {
    gate: Linear,
    gate_experts: Arc<QTensor>,
    up_experts: Arc<QTensor>,
    down_experts: Arc<QTensor>,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeGGUF {
    pub fn new_repack(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("ffn_gate_inp"),
            Shard::default(),
            &None,
            &None,
            DType::F32,
        )?;

        let (gate_experts, up_experts, down_experts) = match &vb.0 {
            Either::Right(v) => (
                v.pp("ffn_gate_exps")
                    .get(
                        (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                    )?
                    .dequantize_f16(&v.device())?,
                v.pp("ffn_up_exps")
                    .get(
                        (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                    )?
                    .dequantize_f16(&v.device())?,
                v.pp("ffn_down_exps")
                    .get(
                        (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                        "weight",
                    )?
                    .dequantize_f16(&v.device())?,
            ),
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        let (ggml_dtype, block_size) = (GgmlDType::Q4K, GgmlDType::Q4K.block_size());

        let moe_intermediate_chunk =
            if moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0 {
                ((moe_cfg.moe_intermediate_size / comm.world_size() + block_size - 1) / block_size)
                    * block_size
            } else {
                moe_cfg.moe_intermediate_size / comm.world_size()
            };

        let cur_chunk_size = if comm.rank() * moe_intermediate_chunk + moe_intermediate_chunk
            < moe_cfg.moe_intermediate_size
        {
            moe_intermediate_chunk
        } else {
            moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk
        };

        assert!(cur_chunk_size > 0 && cur_chunk_size % block_size == 0,
            "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}! \n \
            \t*****Tips: you may try these gglm types: `q8_0` (recommend), `q4_0`, `q4_1`, `q5_0`, `q5_1` (with smaller block_size 32)",
            moe_cfg.moe_intermediate_size,
            comm.world_size(),
            block_size
        );

        let (gate_experts, up_experts, down_experts) = (
            gate_experts
                .narrow(1, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
            up_experts
                .narrow(1, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
            down_experts
                .narrow(2, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
        );
        let gate_experts = Arc::new(QTensor::quantize(&gate_experts, ggml_dtype)?);
        let up_experts = Arc::new(QTensor::quantize(&up_experts, ggml_dtype)?);
        let down_experts = Arc::new(QTensor::quantize(&down_experts, GgmlDType::Q8_0)?);

        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        if comm.world_size() > 1 {
            return Self::new_repack(cfg, vb, comm.clone(), dtype);
        }
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("ffn_gate_inp"),
            Shard::default(),
            &None,
            &None,
            DType::F32,
        )?;

        let (gate_experts, up_experts, down_experts) = match &vb.0 {
            Either::Right(v) => (
                v.pp("ffn_gate_exps").get(
                    (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                )?,
                v.pp("ffn_up_exps").get(
                    (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                )?,
                v.pp("ffn_down_exps").get(
                    (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                )?,
            ),
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size: 1,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) =
            attention_rs::topk::topk_softmax(&router_logits, self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        let ys = {
            let gate = moe::moe_gemm_gguf(
                &xs,
                &self.gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let up = moe::moe_gemm_gguf(
                &xs,
                &self.up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;

            let down_inputs = (up * gate.apply(&self.act)?)?;
            moe::moe_gemm_gguf(
                &down_inputs,
                &self.down_experts,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?
        };
        let mut ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}

pub struct FusedMoeISQ {
    gate: Linear,
    gate_up_experts: QTensor,
    w_size_n: usize,
    down_experts: QTensor,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeISQ {
    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let mut quant_type = match cfg.quant.as_ref().unwrap().as_str() {
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

        let get_moe_intermediate_chunk = |blk_size: usize| -> usize {
            let base = moe_cfg.moe_intermediate_size / comm.world_size();
            if base % blk_size != 0 {
                ((base + blk_size - 1) / blk_size) * blk_size
            } else {
                base
            }
        };

        let mut block_size = quant_type.block_size();
        if comm.world_size() > 1
            && moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0
        {
            //in case of the experts unable to be split under qkk format,
            //and asymetric split also not workable, switch to q8_0
            let chunk = get_moe_intermediate_chunk(block_size);
            if (moe_cfg.moe_intermediate_size - chunk) % (comm.world_size() - 1) != 0 {
                quant_type = GgmlDType::Q8_0;
                block_size = quant_type.block_size();
            }
        }
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &None,
            &None,
            DType::F32,
        )?;

        let experts_vb = vb.pp("experts");
        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        let moe_intermediate_chunk = get_moe_intermediate_chunk(block_size);

        //pack experts
        for i in 0..num_experts {
            let experts_vb = experts_vb.pp(format!("{}", i).as_str());
            let (gate_expert, up_expert, down_expert) = if moe_cfg.moe_intermediate_size
                / comm.world_size()
                % block_size
                != 0
            {
                let (gate_expert, up_expert, down_expert) = match &experts_vb.0 {
                    Either::Left(vb) => {
                        let gate_expert = vb.pp("gate_proj").get_with_hints(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                            Shard::default(),
                        )?;
                        let up_expert = vb.pp("up_proj").get_with_hints(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                            Shard::default(),
                        )?;
                        let down_expert = vb.pp("down_proj").get_with_hints(
                            (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                            "weight",
                            Shard::default(),
                        )?;
                        (gate_expert, up_expert, down_expert)
                    }
                    Either::Right(_) => panic!("invalid varbuild!"),
                };
                let (gate_expert, up_expert, down_expert) = if comm.rank() * moe_intermediate_chunk
                    + moe_intermediate_chunk
                    < moe_cfg.moe_intermediate_size
                {
                    (
                        gate_expert.narrow(
                            0,
                            comm.rank() * moe_intermediate_chunk,
                            moe_intermediate_chunk,
                        )?,
                        up_expert.narrow(
                            0,
                            comm.rank() * moe_intermediate_chunk,
                            moe_intermediate_chunk,
                        )?,
                        down_expert.narrow(
                            1,
                            comm.rank() * moe_intermediate_chunk,
                            moe_intermediate_chunk,
                        )?,
                    )
                } else {
                    let last_remain_size =
                        moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk;
                    assert!(last_remain_size > 0 && last_remain_size % block_size == 0,
                        "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}! \n \
                        \t*****Tips: you may try these gglm types: `q8_0` (recommend), `q4_0`, `q4_1`, `q5_0`, `q5_1` (with smaller block_size 32)",
                        moe_cfg.moe_intermediate_size,
                        comm.world_size(),
                        block_size
                    );
                    let gate_expert = gate_expert.narrow(
                        0,
                        comm.rank() * moe_intermediate_chunk,
                        last_remain_size,
                    )?;
                    let up_expert = up_expert.narrow(
                        0,
                        comm.rank() * moe_intermediate_chunk,
                        last_remain_size,
                    )?;
                    let down_expert = down_expert.narrow(
                        1,
                        comm.rank() * moe_intermediate_chunk,
                        last_remain_size,
                    )?;
                    (gate_expert, up_expert, down_expert)
                };
                (gate_expert, up_expert, down_expert)
            } else {
                match &experts_vb.0 {
                    Either::Left(vb) => {
                        let gate_expert = vb.pp("gate_proj").get_with_hints(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                            shard(0, comm.rank(), comm.world_size()),
                        )?;
                        let up_expert = vb.pp("up_proj").get_with_hints(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                            shard(0, comm.rank(), comm.world_size()),
                        )?;
                        let down_expert = vb.pp("down_proj").get_with_hints(
                            (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                            "weight",
                            shard(1, comm.rank(), comm.world_size()),
                        )?;
                        (gate_expert, up_expert, down_expert)
                    }
                    Either::Right(_) => panic!("invalid varbuild!"),
                }
            };

            gate_experts.push(gate_expert);
            up_experts.push(up_expert);
            down_experts.push(down_expert);
        }
        let gate_experts = Tensor::stack(&gate_experts, 0)?;
        let up_experts = Tensor::stack(&up_experts, 0)?;
        let down_experts = Tensor::stack(&down_experts, 0)?;

        // pack gate_proj and up_proj
        let gate_up_experts = Tensor::cat(&[gate_experts, up_experts], candle_core::D::Minus2)?;
        let w_size_n = gate_up_experts.dim(1)? / 2;

        let gate_up_experts = QTensor::quantize(&gate_up_experts, quant_type)?;

        let down_experts = QTensor::quantize(&down_experts, GgmlDType::Q8_0)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_up_experts,
            w_size_n,
            down_experts,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) =
            attention_rs::topk::topk_softmax(&router_logits, self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        let ys = {
            let gate_up = moe::moe_gemm_gguf(
                &xs,
                &self.gate_up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let gate = gate_up
                .narrow(candle_core::D::Minus1, 0, self.w_size_n)?
                .contiguous()?;
            let up = gate_up
                .narrow(candle_core::D::Minus1, self.w_size_n, self.w_size_n)?
                .contiguous()?;
            let down_inputs = (up * gate.apply(&self.act)?)?;
            moe::moe_gemm_gguf(
                &down_inputs,
                &self.down_experts,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?
        };
        let mut ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}
