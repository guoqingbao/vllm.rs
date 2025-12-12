use super::config::VisionConfig;
use crate::models::layers::attention::NaiveAttention;
use crate::models::layers::distributed::Comm;
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{conv2d, rms_norm, NormX};
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

struct AttentionLayer {
    norm: NormX,
    mlp: MLP,
    attention: NaiveAttention,
    post_norm: NormX,
}

impl AttentionLayer {
    fn new(vb: VarBuilderX, comm: Rc<Comm>, cfg: &VisionConfig, dtype: DType) -> Result<Self> {
        let key_map: HashMap<&str, &str> =
            [("attention_norm", "attn_norm"), ("ffn_norm", "ffn_norm")]
                .iter()
                .cloned()
                .collect();
        let is_qvar_builder = vb.is_qvar_builder();

        let norm = rms_norm(
            cfg.hidden_size,
            1e-5,
            if is_qvar_builder {
                vb.pp(key_map["attention_norm"])
            } else {
                vb.pp("attention_norm")
            },
            if is_qvar_builder { DType::F32 } else { dtype },
            false,
        )?;
        let mlp = MLP::new(
            vb.pp("feed_forward"),
            comm.clone(),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.hidden_act,
            &None,
            &None,
            false,
            dtype,
            "",
        )?;

        let attention = NaiveAttention::new(
            vb.pp("attention"),
            cfg.num_attention_heads,
            cfg.hidden_size,
            cfg.head_dim(),
            None,
            dtype,
            HashMap::new(),
        )?;
        let post_norm = rms_norm(
            cfg.hidden_size,
            1e-5,
            if is_qvar_builder {
                vb.pp(key_map["ffn_norm"])
            } else {
                vb.pp("ffn_norm")
            },
            if is_qvar_builder { DType::F32 } else { dtype },
            false,
        )?;
        Ok(Self {
            norm,
            mlp,
            attention,
            post_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rotary_embed: &Arc<dyn ApplyRotaryEmbedding>,
        subsampled_positions: &Option<Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention.forward(
            &self.norm.forward(&xs)?,
            rotary_embed,
            subsampled_positions,
            mask,
        )?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = self.mlp.forward(&self.post_norm.forward(&xs)?)?;
        xs + residual
    }
}

struct Transformer {
    layers: Vec<AttentionLayer>,
}

impl Transformer {
    fn new(vb: VarBuilderX, comm: Rc<Comm>, cfg: &VisionConfig, dtype: DType) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer =
                AttentionLayer::new(vb.pp(&format!("layers.{}", idx)), comm.clone(), cfg, dtype)?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Arc<dyn ApplyRotaryEmbedding>,
        subsampled_positions: &Option<Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, rotary_emb, subsampled_positions, mask)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &VisionConfig, vb: VarBuilderX, dtype: DType) -> Result<Self> {
        let dev = vb.device();
        let dim = cfg.head_dim();
        let rope_theta = cfg.rope_theta as f32;
        let max_patches_per_side = cfg.image_size / cfg.patch_size;
        let freqs: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let freqs_h = freqs.iter().step_by(2).copied().collect::<Vec<_>>();
        let freqs_h = Tensor::new(freqs_h, &dev)?;
        let freqs_w = freqs.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();
        let freqs_w = Tensor::new(freqs_w, &dev)?;
        let h = Tensor::arange(0u32, max_patches_per_side as u32, &dev)?.to_dtype(DType::F32)?;
        let w = Tensor::arange(0u32, max_patches_per_side as u32, &dev)?.to_dtype(DType::F32)?;
        let freqs_h = h.unsqueeze(1)?.matmul(&freqs_h.unsqueeze(0)?)?;
        let freqs_w = w.unsqueeze(1)?.matmul(&freqs_w.unsqueeze(0)?)?;
        let inv_freq = Tensor::cat(
            &[
                freqs_h.unsqueeze(1)?.repeat((1, max_patches_per_side, 1))?,
                freqs_w.unsqueeze(0)?.repeat((max_patches_per_side, 1, 1))?,
            ],
            D::Minus1,
        )?
        .reshape(((), dim / 2))?;
        let cos = inv_freq.cos()?.to_dtype(dtype)?;
        let sin = inv_freq.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }
}

impl ApplyRotaryEmbedding for RotaryEmbedding {
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, _seq_len, _n_embd) = q.dims4()?;
        let (cos, sin) = (
            &self.cos.index_select(positions, 0)?,
            &self.sin.index_select(positions, 0)?,
        );
        let q_embed = candle_nn::rotary_emb::rope(q, cos, sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, cos, sin)?;
        Ok((q_embed, k_embed))
    }

    fn get_original_max_position_embeddings(&self) -> Option<usize> {
        None
    }

    fn get_llama_4_scaling_beta(&self) -> Option<f64> {
        None
    }
}

unsafe impl Send for RotaryEmbedding {}
unsafe impl Sync for RotaryEmbedding {}

#[allow(unused)]
pub struct VisionModel {
    patch_conv: candle_nn::Conv2d,
    ln_pre: NormX,
    transformer: Transformer,
    patch_positional_embedding: Arc<RotaryEmbedding>,
    max_image_width: u32,
    patch_size: usize,
    dtype: DType,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let conv2d_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_conv = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv2d_cfg,
            vb.pp("patch_conv"), // qvar todo
            false,
        )?;
        let ln_pre = rms_norm(
            cfg.hidden_size,
            1e-5,
            vb.pp("ln_pre"), // qvar todo
            dtype,
            false,
        )?;
        let transformer = Transformer::new(vb.pp("transformer"), comm, cfg, dtype)?;
        let patch_positional_embedding = Arc::new(RotaryEmbedding::new(
            cfg,
            vb.pp("patch_positional_embedding"),
            dtype,
        )?);
        let max_image_width = (cfg.image_size / cfg.patch_size) as u32;
        Ok(Self {
            patch_conv,
            ln_pre,
            transformer,
            patch_positional_embedding,
            max_image_width,
            patch_size: cfg.patch_size,
            dtype,
        })
    }

    fn position_ids_in_meshgrid(
        &self,
        patch_embeds_list: &Vec<Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let mut positions = Vec::new();
        for patch in patch_embeds_list {
            let (height, width) = (patch.dim(D::Minus2)? as u32, patch.dim(D::Minus1)? as u32);
            let idx = Tensor::arange(0, height, device)?;
            let idy = Tensor::arange(0, width, device)?;
            let mesh = Tensor::meshgrid(&[idx, idy], false)?;
            let ids = (&mesh[0] * (self.max_image_width as f64) + &mesh[1])?.flatten_all()?;
            positions.push(ids);
        }
        Tensor::cat(&positions, 0)
    }

    fn generate_block_attention_mask(
        &self,
        patch_embeds_list: Vec<usize>,
        patch_embeds: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = patch_embeds.dim(1)?;
        let mut causal_mask = Tensor::ones(
            (seq_len, seq_len),
            patch_embeds.dtype(),
            patch_embeds.device(),
        )?;

        use half::{bf16, f16};
        let min_value = match patch_embeds.dtype() {
            DType::F32 => f32::MIN as f64,
            DType::F16 => f16::MIN.to_f64(),
            DType::BF16 => bf16::MIN.to_f64(),
            _ => candle_core::bail!("Not supported dtype!"),
        };

        causal_mask = (causal_mask * min_value)?;

        let block_end_idx: Vec<usize> = patch_embeds_list.iter().fold(Vec::new(), |mut acc, &x| {
            let new_sum = x + acc.last().copied().unwrap_or(0);
            acc.push(new_sum);
            acc
        });
        let block_start_idx: Vec<usize> = {
            let mut extended = vec![0];
            extended.extend_from_slice(&patch_embeds_list[..patch_embeds_list.len() - 1]);
            extended.into_iter().fold(Vec::new(), |mut acc, x| {
                let new_sum = x + acc.last().copied().unwrap_or(0);
                acc.push(new_sum);
                acc
            })
        };
        for (start, end) in block_start_idx.into_iter().zip(block_end_idx) {
            causal_mask = causal_mask.slice_assign(
                &[start..end, start..end],
                &Tensor::zeros(
                    (end - start, end - start),
                    causal_mask.dtype(),
                    causal_mask.device(),
                )?,
            )?;
        }

        causal_mask
            .reshape((1, 1, causal_mask.dim(0)?, causal_mask.dim(1)?))?
            .repeat((patch_embeds.dim(0)?, 1, 1, 1))
    }

    pub fn forward(&self, xs: &Tensor, image_sizes: Vec<(u32, u32)>) -> Result<Tensor> {
        let patch_embeds = self.patch_conv.forward(&xs)?;
        let patch_embeds_list = image_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let patches_h = size.0 as usize / self.patch_size;
                let patches_w = size.1 as usize / self.patch_size;
                patch_embeds
                    .i(i)?
                    .narrow(D::Minus2, 0, patches_h)?
                    .narrow(D::Minus1, 0, patches_w)
            })
            .collect::<Result<Vec<Tensor>>>()?;
        let patch_embeds = Tensor::cat(
            &patch_embeds_list
                .iter()
                .map(|p| p.flatten_from(1)?.t())
                .collect::<Result<Vec<Tensor>>>()?,
            0,
        )?
        .unsqueeze(0)?;

        let patch_embeds = self.ln_pre.forward(&patch_embeds)?;
        let subsampled_positions =
            Some(self.position_ids_in_meshgrid(&patch_embeds_list, patch_embeds.device())?);

        let attention_mask = self.generate_block_attention_mask(
            patch_embeds_list
                .iter()
                .map(|p| Ok(p.dim(D::Minus2)? * p.dim(D::Minus1)?))
                .collect::<Result<Vec<usize>>>()?,
            &patch_embeds,
        )?;

        let trait_obj: Arc<dyn ApplyRotaryEmbedding> = self.patch_positional_embedding.clone();
        self.transformer.forward(
            &patch_embeds,
            &trait_obj,
            &subsampled_positions,
            Some(&attention_mask),
        )
    }
}
