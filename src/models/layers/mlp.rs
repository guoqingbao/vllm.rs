use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::utils::config::Config;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::var_builder::Shard;
// use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use crate::models::layers::VarBuilderX;
use std::collections::HashMap;
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(vb: VarBuilderX, config: &Config, dtype: DType) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let sd = Shard::default();
        let key_map: HashMap<&str, &str> = [
            ("gate_proj", "ffn_gate"),
            ("up_proj", "ffn_up"),
            ("down_proj", "ffn_down"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();

        let gate_proj = linear_no_bias(
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
        )?;
        let up_proj = linear_no_bias(
            hidden_size,
            intermediate_size,
            if is_qvar_builder {
                vb.pp(key_map["up_proj"])
            } else {
                vb.pp("up_proj")
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
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}
