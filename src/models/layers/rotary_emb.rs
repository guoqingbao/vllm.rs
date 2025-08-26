use crate::utils::config::Config;
use crate::utils::config::{RopeScaling, ScalingValue};
use candle_core::{DType, Device, Result, Tensor, D};
use either::Either;
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
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
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

fn calculate_default_inv_freq(base: f64, dim: usize) -> Vec<f32> {
    (0..dim)
        .step_by(2)
        .map(|i| 1f32 / base.powf(i as f64 / dim as f64) as f32)
        .collect()
}

#[derive(Debug, Clone)]
pub struct ScalingRotaryEmbedding(RotaryEmbedding);

impl ScalingRotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device, is_rope_i: bool) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|factor| (factor * dim as f32) as usize)
            .unwrap_or(dim);
        if let Some(rope_scaling) = &cfg.rope_scaling {
            let mut rope_scaling = rope_scaling.clone();
            if !rope_scaling.contains_key("rope_type") && rope_scaling.contains_key("type") {
                //for non-standard models that use "type" instead of "rope_type"
                let value = rope_scaling.remove("type").unwrap();
                rope_scaling.insert("rope_type".to_string(), value);
            }

            let original_max_position_embeddings = if let RopeScaling(Either::Left(ScalingValue(
                Either::Left(original_max_position_embeddings),
            ))) =
                &rope_scaling["original_max_position_embeddings"]
            {
                *original_max_position_embeddings
            } else if let RopeScaling(Either::Left(ScalingValue(Either::Left(factor)))) =
                &rope_scaling["factor"]
            {
                //for missing original_max_position_embeddings, we assume the original was max_position_embeddings / factor
                cfg.max_position_embeddings as f64 / *factor
            } else {
                candle_core::bail!(
                    "original_max_position_embeddings must be set in rope_scaling or cfg"
                );
            };

            if let RopeScaling(Either::Right(rope_type)) = &rope_scaling["rope_type"] {
                let rope_result = if rope_type == "linear" {
                    if let RopeScaling(Either::Left(ScalingValue(Either::Left(factor)))) =
                        &rope_scaling["factor"]
                    {
                        let inv_freq: Vec<_> =
                            calculate_default_inv_freq(cfg.rope_theta, rotary_dim);
                        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?;
                        let idx_theta = (Tensor::arange(
                            0,
                            (original_max_position_embeddings * *factor) as u32,
                            &dev,
                        )?
                        .to_dtype(DType::F32)?
                        .reshape(((original_max_position_embeddings * *factor) as usize, 1))?
                            / (*factor as f64))?
                            .matmul(&inv_freq.reshape((1, inv_freq.elem_count()))?)?;
                        let cos = idx_theta.cos()?.to_dtype(dtype)?;
                        let sin = idx_theta.sin()?.to_dtype(dtype)?;
                        Self(RotaryEmbedding {
                            sin,
                            cos,
                            is_rope_i,
                            rotary_dim: if cfg.partial_rotary_factor.is_some() {
                                Some(rotary_dim)
                            } else {
                                None
                            },
                        })
                    } else {
                        candle_core::bail!("Linear rope_type requires factor to be set");
                    }
                } else if rope_type == "llama3" {
                    match (
                        &rope_scaling["factor"],
                        &rope_scaling["low_freq_factor"],
                        &rope_scaling["high_freq_factor"],
                    ) {
                        (
                            RopeScaling(Either::Left(ScalingValue(Either::Left(factor)))),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(low_freq_factor)))),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(high_freq_factor)))),
                        ) => {
                            let low_freq_wavelen =
                                (original_max_position_embeddings / low_freq_factor) as f32;
                            let high_freq_wavelen =
                                (original_max_position_embeddings / high_freq_factor) as f32;

                            let inv_freq = calculate_default_inv_freq(cfg.rope_theta, rotary_dim)
                                .into_iter()
                                .map(|freq| {
                                    let wavelen = 2. * std::f32::consts::PI / freq;
                                    if wavelen < high_freq_wavelen {
                                        freq
                                    } else if wavelen > low_freq_wavelen {
                                        freq / *factor as f32
                                    } else {
                                        let smooth = (original_max_position_embeddings as f32
                                            / wavelen
                                            - *low_freq_factor as f32)
                                            / (*high_freq_factor - *low_freq_factor) as f32;
                                        (1. - smooth) * freq / *factor as f32 + smooth * freq
                                    }
                                })
                                .collect::<Vec<_>>();
                            let inv_freq_len = inv_freq.len();
                            let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
                            let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                                .to_dtype(DType::F32)?
                                .reshape((cfg.max_position_embeddings, 1))?;
                            let freqs = t.matmul(&inv_freq)?;
                            let sin = freqs.sin()?.to_dtype(dtype)?;
                            let cos = freqs.cos()?.to_dtype(dtype)?;
                            Self(RotaryEmbedding {
                                sin,
                                cos,
                                is_rope_i,
                                rotary_dim: if cfg.partial_rotary_factor.is_some() {
                                    Some(rotary_dim)
                                } else {
                                    None
                                },
                            })
                        }
                        _ => {
                            candle_core::bail!("Llama3 rope_type requires factor, low_freq_factor, high_freq_factor and original_max_position_embeddings to be set");
                        }
                    }
                } else if rope_type == "default" {
                    Self(RotaryEmbedding::new(dtype, cfg, dev, is_rope_i)?)
                } else if rope_type == "dynamic" {
                    let scaling_factor = if rope_scaling.contains_key("alpha") {
                        if let RopeScaling(Either::Left(ScalingValue(Either::Left(factor)))) =
                            &rope_scaling["alpha"]
                        {
                            *factor as f32
                        } else {
                            candle_core::bail!("Dynamic rope_type requires alpha to be set")
                        }
                    } else if rope_scaling.contains_key("factor") {
                        if let RopeScaling(Either::Left(ScalingValue(Either::Left(factor)))) =
                            &rope_scaling["factor"]
                        {
                            *factor as f32
                        } else {
                            candle_core::bail!("Dynamic rope_type requires factor to be set")
                        }
                    } else {
                        candle_core::bail!(
                            "Dynamic rope_type requires either alpha or factor to be set"
                        )
                    };

                    let (rope_theta, max_seq_len) = if rope_scaling.contains_key("alpha") {
                        let max_len = cfg.max_position_embeddings as u32;
                        let rope_theta = (cfg.rope_theta * scaling_factor as f64)
                            .powf(rotary_dim as f64 / (rotary_dim - 2) as f64);
                        (rope_theta, max_len)
                    } else {
                        let max_len =
                            (original_max_position_embeddings as f32 * scaling_factor) as u32;
                        let rope_theta = (cfg.rope_theta
                            * ((scaling_factor as f64 * max_len as f64
                                / original_max_position_embeddings)
                                - (scaling_factor as f64 - 1f64)))
                            .powf(rotary_dim as f64 / (rotary_dim - 2) as f64);
                        (rope_theta, max_len)
                    };

                    let inv_freq: Vec<_> = calculate_default_inv_freq(rope_theta, rotary_dim);
                    let inv_freq_len = inv_freq.len();
                    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
                    let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
                        .to_dtype(DType::F32)?
                        .reshape((max_seq_len as usize, 1))?;
                    let freqs = t.matmul(&inv_freq)?;
                    let sin = freqs.sin()?.to_dtype(dtype)?;
                    let cos = freqs.cos()?.to_dtype(dtype)?;
                    Self(RotaryEmbedding {
                        sin,
                        cos,
                        is_rope_i,
                        rotary_dim: if cfg.partial_rotary_factor.is_some() {
                            Some(rotary_dim)
                        } else {
                            None
                        },
                    })
                } else if rope_type == "yarn" {
                    let mut ropescaling = rope_scaling.clone();
                    if !ropescaling.contains_key("extrapolation_factor") {
                        ropescaling.insert(
                            "extrapolation_factor".to_string(),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(1.0)))),
                        );
                    }
                    if !ropescaling.contains_key("attn_factor") {
                        ropescaling.insert(
                            "attn_factor".to_string(),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(1.0)))),
                        );
                    }
                    if !ropescaling.contains_key("beta_fast") {
                        ropescaling.insert(
                            "beta_fast".to_string(),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(32.0)))),
                        );
                    }
                    if !ropescaling.contains_key("beta_slow") {
                        ropescaling.insert(
                            "beta_slow".to_string(),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(1.0)))),
                        );
                    }

                    match (
                        &ropescaling["extrapolation_factor"],
                        &ropescaling["attn_factor"],
                        &ropescaling["beta_fast"],
                        &ropescaling["beta_slow"],
                        &ropescaling["factor"],
                    ) {
                        (
                            RopeScaling(Either::Left(ScalingValue(Either::Left(
                                extrapolation_factor,
                            )))),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(attn_factor)))),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(beta_fast)))),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(beta_slow)))),
                            RopeScaling(Either::Left(ScalingValue(Either::Left(factor)))),
                        ) => {
                            let embed = YarnRotaryEmbedding::new_yarn(
                                dtype,
                                dev,
                                cfg.rope_theta as f32,
                                rotary_dim,
                                cfg.max_position_embeddings,
                                original_max_position_embeddings as usize,
                                *beta_fast as f32,
                                *beta_slow as f32,
                                *attn_factor as f32,
                                *extrapolation_factor as f32,
                                *factor as f32,
                            )?;
                            Self(RotaryEmbedding {
                                sin: embed.sin,
                                cos: embed.cos,
                                is_rope_i,
                                rotary_dim: if cfg.partial_rotary_factor.is_some() {
                                    Some(rotary_dim)
                                } else {
                                    None
                                },
                            })
                        }
                        _ => {
                            candle_core::bail!("Llama3 rope_type requires factor, low_freq_factor, high_freq_factor and original_max_position_embeddings to be set");
                        }
                    }
                } else {
                    candle_core::bail!("Unknown rope_type: {rope_type}");
                };
                Ok(rope_result)
            } else {
                candle_core::bail!("rope_type must be a string");
            }
        } else {
            Ok(Self(RotaryEmbedding::new(dtype, cfg, dev, is_rope_i)?))
        }
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.0.apply_rotary_emb_qkv(q, k, positions)
    }
}

pub struct YarnRotaryEmbedding {
    pub sin: Tensor,
    pub cos: Tensor,
}

impl YarnRotaryEmbedding {
    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * std::f32::consts::PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1.)
    }

    pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        if scale <= 1. {
            return 1.;
        }
        0.1 * mscale * scale.ln() + 1.
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        dtype: DType,
        dev: &Device,
        rope_theta: f32,
        dim: usize,
        max_position_embeddings: usize,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        attn_factor: f32,
        extrapolation_factor: f32,
        factor: f32,
    ) -> Result<Self> {
        let freq_extra: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, &Device::Cpu)?;
        let freq_inter: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * rope_theta.powf(i as f32 / dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), &Device::Cpu)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            rope_theta,
            original_max_position_embeddings,
        );
        let inv_freq_mask = ((1.
            - Self::yarn_linear_ramp_mask(low, high, dim / 2, &Device::Cpu)?)?
            * extrapolation_factor as f64)?;
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(
            0u32,
            (max_position_embeddings as f32 * factor) as u32,
            &Device::Cpu,
        )?
        .to_dtype(DType::F32)?
        .reshape(((max_position_embeddings as f32 * factor) as usize, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale = Self::yarn_get_mscale(factor, 1.0f32) * attn_factor;
        let sin = (freqs.sin()? * mscale as f64)?
            .to_device(dev)?
            .to_dtype(dtype)?;
        let cos = (freqs.cos()? * mscale as f64)?
            .to_device(dev)?
            .to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }
}
