// src/models/layers/moe.rs
use crate::models::layers::distributed::{shard, AllReduce, Comm};
use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::models::layers::mlp::MLP;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::Module;
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Result, Tensor, D,
};
use candle_nn::var_builder::Shard;
use either::Either;
use std::rc::Rc;
use std::sync::Arc;

pub struct MoeNaive {
    gate: Linear,
    experts: Vec<MLP>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
}

impl MoeNaive {
    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &None,
            DType::F32,
        )?;

        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(MLP::new(
                experts_vb.pp(format!("{}", i).as_str()).clone(),
                comm.clone(),
                cfg,
                moe_cfg.moe_intermediate_size,
                false,
                dtype,
                "",
            )?);
        }
        let world_size = comm.world_size();
        Ok(Self {
            gate,
            experts,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let experts_per_tok = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let routing_weights = routing_weights.gather(&experts_per_tok, D::Minus1)?;

        let routing_weights = routing_weights.to_vec2::<f32>()?;
        let experts_per_tok = experts_per_tok.to_vec2::<u32>()?;
        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_experts = vec![vec![]; self.experts.len()];
        for (row_idx, (rw, expert_idxs)) in routing_weights
            .iter()
            .zip(experts_per_tok.iter())
            .enumerate()
        {
            let sum_rw = rw.iter().sum::<f32>();
            for (&rw, &expert_idx) in rw.iter().zip(expert_idxs.iter()) {
                top_x[expert_idx as usize].push(row_idx as u32);
                let rw = if self.norm_topk_prob { rw / sum_rw } else { rw };
                selected_experts[expert_idx as usize].push(rw)
            }
        }

        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }
            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_experts =
                Tensor::new(selected_experts[expert_idx].as_slice(), xs.device())?
                    .reshape(((), 1))?
                    .to_dtype(xs.dtype())?;
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
            let current_hidden_states = expert_layer
                .forward(&current_state.unsqueeze(0)?)?
                .squeeze(0)?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&selected_experts)?;
            ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}

pub struct FusedMoeGGUF {
    gate: Linear,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
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
        let qtensor = QTensor::quantize(&gate_experts, ggml_dtype)?;
        let gate_experts = QMatMul::QTensor(Arc::new(qtensor));

        let qtensor = QTensor::quantize(&up_experts, ggml_dtype)?;
        let up_experts = QMatMul::QTensor(Arc::new(qtensor));

        let qtensor = QTensor::quantize(&down_experts, GgmlDType::Q8_0)?;
        let down_experts = QMatMul::QTensor(Arc::new(qtensor));

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

        let gate_experts = QMatMul::QTensor(gate_experts);
        let up_experts = QMatMul::QTensor(up_experts);
        let down_experts = QMatMul::QTensor(down_experts);

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

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };
        let router_logits = self.gate.forward(&xs)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut scores = routing_weights.gather(&indices, D::Minus1)?;

        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            let xs = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self.gate_experts.indexed_moe_forward(&xs, &indices)?;
            let up = self.up_experts.indexed_moe_forward(&xs, &indices)?;
            self.down_experts
                .indexed_moe_forward(&(up * gate.apply(&self.act)?)?, &indices)?
        };

        let mut ys = ys
            .broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((num_tokens, hidden_dim))?
            .to_dtype(self.dtype)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}

pub struct FusedMoeISQ {
    gate: Linear,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
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

        let mut block_size = quant_type.block_size();
        //in case the experts unable to split under qkk format, switch to q8_0
        if moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0 {
            quant_type = GgmlDType::Q8_0;
            block_size = quant_type.block_size();
        }
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &None,
            DType::F32,
        )?;

        let experts_vb = vb.pp("experts");
        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        let moe_intermediate_chunk =
            if moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0 {
                ((moe_cfg.moe_intermediate_size / comm.world_size() + block_size - 1) / block_size)
                    * block_size
            } else {
                moe_cfg.moe_intermediate_size / comm.world_size()
            };

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
        // in-situ quantization for using fused moe kernel
        let qtensor = QTensor::quantize(&gate_experts, quant_type).unwrap();
        let gate_experts = QMatMul::QTensor(Arc::new(qtensor));

        let qtensor = QTensor::quantize(&up_experts, quant_type).unwrap();
        let up_experts = QMatMul::QTensor(Arc::new(qtensor));

        let qtensor = QTensor::quantize(&down_experts, GgmlDType::Q8_0).unwrap();
        let down_experts = QMatMul::QTensor(Arc::new(qtensor));
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };
        let router_logits = self.gate.forward(&xs)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut scores = routing_weights.gather(&indices, D::Minus1)?;

        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            let xs = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self.gate_experts.indexed_moe_forward(&xs, &indices)?;
            let up = self.up_experts.indexed_moe_forward(&xs, &indices)?;
            let down_inputs = (up * gate.apply(&self.act)?)?;
            self.down_experts
                .indexed_moe_forward(&down_inputs, &indices)?
        };
        let mut ys = ys
            .broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((num_tokens, hidden_dim))?
            .to_dtype(self.dtype)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}
