use crate::utils::config::Config;
use candle_core::{DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let max_seq_len = cfg.max_model_len.unwrap_or(cfg.max_position_embeddings);
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // let input_positions: Vec<i64> = positions.to_vec1::<i64>()?;
        let cos = self.cos.index_select(&positions, 0)?;
        let sin = self.sin.index_select(&positions, 0)?;
        let q_embed = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}
