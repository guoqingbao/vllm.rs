use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::utils::config::Config;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::var_builder::Shard;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;

pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(vb: VarBuilder, config: &Config, dtype: DType) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let sd = Shard::default();
        let gate_proj = linear_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("gate_proj"),
            sd,
            &config.quant,
            dtype,
        )?;
        let up_proj = linear_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp("up_proj"),
            sd,
            &config.quant,
            dtype,
        )?;
        let down_proj = linear_no_bias(
            intermediate_size,
            hidden_size,
            vb.pp("down_proj"),
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
