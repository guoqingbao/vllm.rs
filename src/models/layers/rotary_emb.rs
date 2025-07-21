use crate::utils::config::Config;
use candle_core::{DType, Device, Result, Tensor, D};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    is_rope_i: bool,
    rotary_dim: Option<usize>,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device, is_rope_i: bool) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|factor| (factor * dim as f32) as usize)
            .unwrap_or(dim);
        let max_seq_len = cfg.max_model_len.unwrap_or(cfg.max_position_embeddings);
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
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
            is_rope_i,
            rotary_dim: if cfg.partial_rotary_factor.is_some() {
                Some(rotary_dim)
            } else {
                None
            },
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // let input_positions: Vec<i64> = positions.to_vec1::<i64>()?;
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        let func = if self.is_rope_i {
            candle_nn::rotary_emb::rope_i
        } else {
            candle_nn::rotary_emb::rope
        };

        let (q_embed, k_embed) = if let Some(rotary_dim) = self.rotary_dim {
            let q_rot = q.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?;
            let q_pass = q
                .narrow(D::Minus1, rotary_dim, q.dim(D::Minus1)? - rotary_dim)?
                .contiguous()?;
            let q_rot = func(&q_rot, &cos, &sin)?;
            let q_embed = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?.contiguous()?;

            let k_rot = k.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?;
            let k_pass = k
                .narrow(D::Minus1, rotary_dim, k.dim(D::Minus1)? - rotary_dim)?
                .contiguous()?;
            let k_rot = func(&k_rot, &cos, &sin)?;
            let k_embed = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?.contiguous()?;
            (q_embed, k_embed)
        } else {
            let q_embed = func(&q, &cos, &sin)?;
            let k_embed = func(&k, &cos, &sin)?;
            (q_embed, k_embed)
        };

        Ok((q_embed, k_embed))
    }
}
