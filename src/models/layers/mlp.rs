use crate::models::layers::distributed::{
    shard, Comm, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use candle_core::{DType, Result, Tensor};
use std::collections::HashMap;
use std::rc::Rc;

pub struct MLP {
    gate_proj: TensorParallelColumnLinear,
    up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
}

impl MLP {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        intermediate_size: usize,
        gate_up_merged: bool,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        // let intermediate_size = config.intermediate_size;
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
                vb.pp(key_map[if gate_up_merged {
                    "up_proj"
                } else {
                    "gate_proj"
                }])
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
            &config.quant,
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
                vb.pp(key_map["up_proj"])
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
            &config.quant,
            dtype,
        )?;

        let down_proj = TensorParallelRowLinear::load_with_hints(
            intermediate_size,
            hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["down_proj"])
            } else {
                vb.pp("down_proj")
            },
            comm.clone(),
            &config.quant,
            dtype,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj
            .forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }
}
