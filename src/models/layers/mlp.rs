use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::utils::config::Config;
use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::var_builder::Shard;
// use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use crate::models::layers::VarBuilderX;
use std::collections::HashMap;
pub struct MLP {
    gate_proj: Option<Linear>,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(
        vb: VarBuilderX,
        config: &Config,
        gate_up_merged: bool,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let sd = Shard::default();
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

        let gate_proj = if !gate_up_merged {
            Some(linear_no_bias(
                hidden_size,
                intermediate_size,
                if is_qvar_builder {
                    vb.pp(key_map["gate_proj"])
                } else {
                    vb.pp("gate_proj")
                },
                sd,
                &config.quant,
                dtype,
            )?)
        } else {
            None
        };

        let up_proj = linear_no_bias(
            hidden_size,
            if gate_up_merged {
                intermediate_size * 2
            } else {
                intermediate_size
            },
            if is_qvar_builder {
                vb.pp(key_map["up_proj"])
            } else {
                vb.pp(if gate_up_merged {
                    "gate_up_proj"
                } else {
                    "up_proj"
                })
            },
            sd,
            &config.quant,
            dtype,
        )?;
        let down_proj = linear_no_bias(
            intermediate_size,
            hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["down_proj"])
            } else {
                vb.pp("down_proj")
            },
            sd,
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
        let (gate, up) = if self.gate_proj.is_some() {
            let gate = self.gate_proj.as_ref().unwrap().forward(xs)?;
            let up = self.up_proj.forward(xs)?;
            (gate, up)
        } else {
            //gate and up merged
            let w = self.up_proj.forward(xs)?;
            let chunk_size = w.dim(D::Minus1)? / 2;
            let gate = w.narrow(D::Minus1, 0, chunk_size)?.contiguous()?;
            let up = w.narrow(D::Minus1, chunk_size, chunk_size)?.contiguous()?;
            (gate, up)
        };
        self.down_proj
            .forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }
}
