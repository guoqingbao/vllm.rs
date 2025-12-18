use crate::models::layers::distributed::{
    shard, Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::VarBuilderX;
use crate::utils::config::QuantConfig;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Activation, Module};
use std::collections::HashMap;
use std::rc::Rc;

pub struct MLP {
    gate_proj: TensorParallelColumnLinear,
    up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    activation: Activation,
}

impl MLP {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        hidden_size: usize,
        intermediate_size: usize,
        activation: &Activation,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        gate_up_merged: bool,
        dtype: DType,
        suffix: &str,
    ) -> Result<Self> {
        let key_map: HashMap<&str, &str> = [
            ("gate_proj", "ffn_gate"),
            ("up_proj", "ffn_up"),
            ("gate_up_proj", "ffn_up"),
            ("down_proj", "ffn_down"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();

        let gate_proj = TensorParallelColumnLinear::load_with_shard(
            hidden_size,
            if gate_up_merged {
                intermediate_size * 2
            } else {
                intermediate_size
            },
            false,
            if is_qvar_builder {
                vb.pp((key_map[if gate_up_merged {
                    "up_proj"
                } else {
                    "gate_proj"
                }]
                .to_string()
                    + suffix)
                    .as_str())
            } else {
                vb.pp(if gate_up_merged {
                    "gate_up_proj"
                } else {
                    "gate_proj"
                })
            },
            if gate_up_merged {
                shard(0, comm.rank(), comm.world_size() * 2)
            } else {
                shard(0, comm.rank(), comm.world_size())
            },
            quant_cfg,
            quant,
            dtype,
        )?;

        let up_proj = TensorParallelColumnLinear::load_with_shard(
            hidden_size,
            if gate_up_merged {
                intermediate_size * 2
            } else {
                intermediate_size
            },
            false,
            if is_qvar_builder {
                vb.pp((key_map["up_proj"].to_string() + suffix).as_str())
            } else {
                vb.pp(if gate_up_merged {
                    "gate_up_proj"
                } else {
                    "up_proj"
                })
            },
            if gate_up_merged {
                shard(0, comm.world_size() + comm.rank(), comm.world_size() * 2)
            } else {
                shard(0, comm.rank(), comm.world_size())
            },
            quant_cfg,
            quant,
            dtype,
        )?;

        let down_proj = TensorParallelRowLinear::load_with_hints(
            intermediate_size,
            hidden_size,
            if is_qvar_builder {
                vb.pp((key_map["down_proj"].to_string() + suffix).as_str())
            } else {
                vb.pp("down_proj")
            },
            comm.clone(),
            quant_cfg,
            quant,
            dtype,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            activation: activation.clone(),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj
            .forward(&(self.activation.forward(&gate)? * up)?)
    }
}

pub struct NaiveMLP {
    fc1: ReplicatedLinear,
    fc2: ReplicatedLinear,
    act: Activation,
}

impl NaiveMLP {
    pub fn new(
        vb: VarBuilderX,
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        names: &[&str],
        hidden_act: Activation,
        dtype: DType,
    ) -> Result<Self> {
        let fc1 = ReplicatedLinear::load_b(
            hidden_size,
            intermediate_size,
            bias,
            vb.pp(names[0]),
            &None,
            &None,
            dtype,
        )?;

        let fc2 = ReplicatedLinear::load_b(
            intermediate_size,
            hidden_size,
            bias,
            vb.pp(names[1]),
            &None,
            &None,
            dtype,
        )?;

        Ok(Self {
            fc1,
            fc2,
            act: hidden_act,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.fc1.forward(xs)?;
        let down = self.act.forward(&gate_up)?;
        self.fc2.forward(&down)
    }
}
