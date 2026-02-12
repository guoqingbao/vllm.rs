// src/models/layers/deltanet.rs
// Shared Qwen3.5/Qwen3Next GatedDeltaNet linear-attention layer.

use crate::models::layers::distributed::{
    Comm, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::resolve_qwen3_hybrid_config;
use attention_rs::gdn;
use attention_rs::mamba_cache::MambaCache;
use attention_rs::InputMetadata;
use candle_core::{DType, IndexOp, Result, Tensor};
use std::rc::Rc;

enum GdnProjection {
    // Qwen3Next: in_proj_qkvz + in_proj_ba
    FusedQkvzBa {
        in_proj_qkvz: TensorParallelColumnLinear,
        in_proj_ba: TensorParallelColumnLinear,
    },
    // Qwen3.5: in_proj_qkv + in_proj_z + in_proj_ba + in_proj_a
    SplitQkvZa {
        in_proj_qkv: TensorParallelColumnLinear,
        in_proj_z: TensorParallelColumnLinear,
        in_proj_b: TensorParallelColumnLinear,
        in_proj_a: TensorParallelColumnLinear,
    },
}

pub struct GatedDeltaNet {
    projection: GdnProjection,
    out_proj: TensorParallelRowLinear,
    conv_weight: Tensor,
    conv_bias: Option<Tensor>,
    a_log: Tensor,
    dt_bias: Tensor,
    gdn_norm_weight: Tensor,
    gdn_norm_bias: Option<Tensor>,
    gdn_norm_per_head: bool,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    kv_group_size: usize,
    gdn_layer_idx: usize,
    rms_norm_eps: f64,
}

impl GatedDeltaNet {
    fn load_projection(
        vb: &VarBuilderX,
        hidden_size: usize,
        key_dim_global: usize,
        value_dim_global: usize,
        num_v_heads_global: usize,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
    ) -> Result<GdnProjection> {
        // linear_attn.in_proj_qkvz(2)
        // linear_attn.in_proj_qkvz.weight	[12 288, 2 048]	F8_E4M3
        // linear_attn.in_proj_qkvz.weight_scale_inv	[96, 16] BF16

        // Qwen3Next format: fused qkvz + fused ba
        let projection_size_qkvz = key_dim_global * 2 + value_dim_global * 2;
        let projection_size_ba = num_v_heads_global * 2;
        let fused_qkvz = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            projection_size_qkvz,
            false,
            vb.pp("in_proj_qkvz"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        );

        let fused_ba = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            projection_size_ba,
            false,
            vb.pp("in_proj_ba"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        );

        if let (Ok(in_proj_qkvz), Ok(in_proj_ba)) = (fused_qkvz, fused_ba) {
            // Qwen3 Next projection
            return Ok(GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            });
        };

        // Qwen3.5 format: split qkv, z, b, a
        let split_qkv = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            key_dim_global * 2 + value_dim_global,
            false,
            vb.pp("in_proj_qkv"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        );

        let split_z = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            value_dim_global,
            false,
            vb.pp("in_proj_z"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        );

        let split_b = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            num_v_heads_global,
            false,
            vb.pp("in_proj_ba"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )
        .or_else(|_| {
            TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                num_v_heads_global,
                false,
                vb.pp("in_proj_b"),
                comm.clone(),
                &config.quantization_config,
                &config.quant,
                dtype,
            )
        });
        let split_a = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            num_v_heads_global,
            false,
            vb.pp("in_proj_a"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        );

        if let (Ok(in_proj_qkv), Ok(in_proj_z), Ok(in_proj_b), Ok(in_proj_a)) =
            (split_qkv, split_z, split_b, split_a)
        {
            return Ok(GdnProjection::SplitQkvZa {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            });
        }

        candle_core::bail!("Unable to load Qwen3.5/Qwen3Next linear attention projection weights",)
    }

    fn fix_qwen3next_projection_order(
        &self,
        mixed_qkvz: &Tensor,
        mixed_ba: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let seq_len = mixed_qkvz.dim(0)?;
        let qkvz_group_dim =
            self.head_k_dim + self.head_k_dim + self.kv_group_size * self.head_v_dim * 2;
        let ba_group_dim = 2 * self.kv_group_size;

        let mixed_qkvz = mixed_qkvz.reshape((seq_len, self.num_k_heads, qkvz_group_dim))?;
        let mixed_ba = mixed_ba.reshape((seq_len, self.num_k_heads, ba_group_dim))?;

        let mut offset = 0usize;
        let query = mixed_qkvz.narrow(2, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let key = mixed_qkvz.narrow(2, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let value = mixed_qkvz.narrow(2, offset, self.kv_group_size * self.head_v_dim)?;
        offset += self.kv_group_size * self.head_v_dim;
        let z = mixed_qkvz.narrow(2, offset, self.kv_group_size * self.head_v_dim)?;

        let b = mixed_ba.narrow(2, 0, self.kv_group_size)?;
        let a = mixed_ba.narrow(2, self.kv_group_size, self.kv_group_size)?;

        Ok((
            query.reshape((seq_len, self.key_dim))?,
            key.reshape((seq_len, self.key_dim))?,
            value.reshape((seq_len, self.value_dim))?,
            z.reshape((seq_len, self.value_dim))?,
            b.reshape((seq_len, self.num_v_heads))?,
            a.reshape((seq_len, self.num_v_heads))?,
        ))
    }

    fn project_inputs(
        &self,
        xs: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        match &self.projection {
            GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            } => {
                let mixed_qkvz = in_proj_qkvz.forward(xs)?;
                let mixed_ba = in_proj_ba.forward(xs)?;
                self.fix_qwen3next_projection_order(&mixed_qkvz, &mixed_ba)
            }
            GdnProjection::SplitQkvZa {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let proj_qkv = in_proj_qkv.forward(xs)?;
                let q = proj_qkv.narrow(1, 0, self.key_dim)?;
                let k = proj_qkv.narrow(1, self.key_dim, self.key_dim)?;
                let v = proj_qkv.narrow(1, self.key_dim * 2, self.value_dim)?;
                let z = in_proj_z.forward(xs)?;
                let b = in_proj_b.forward(xs)?;
                let a = in_proj_a.forward(xs)?;
                Ok((q, k, v, z, b, a))
            }
        }
    }

    fn repeat_kv_heads(&self, x: Tensor) -> Result<Tensor> {
        if self.num_k_heads == self.num_v_heads {
            return Ok(x);
        }
        let (seq_len, _h, _d) = x.dims3()?;
        x.unsqueeze(2)?
            .broadcast_as((
                seq_len,
                self.num_k_heads,
                self.kv_group_size,
                self.head_k_dim,
            ))?
            .reshape((seq_len, self.num_v_heads, self.head_k_dim))
    }

    fn prefill_lengths(input_metadata: &InputMetadata, token_count: usize) -> Result<Vec<usize>> {
        if let Some(cumulative) = &input_metadata.seqlens {
            if cumulative.is_empty() {
                candle_core::bail!("Missing prefill sequence lengths for linear attention batch")
            }
            let mut prev = 0usize;
            let mut lens = Vec::with_capacity(cumulative.len());
            for &cur_end in cumulative {
                let cur = cur_end as usize;
                if cur < prev {
                    candle_core::bail!("Invalid cumulative sequence lengths for linear attention")
                }
                lens.push(cur - prev);
                prev = cur;
            }
            if prev != token_count {
                candle_core::bail!(
                    "Linear attention token mismatch: cumulative {} vs token count {}",
                    prev,
                    token_count
                )
            }
            return Ok(lens);
        }
        Ok(vec![token_count])
    }

    fn l2_norm_last_dim(x: &Tensor, eps: f64) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let inv_norm = (&x_f32 * &x_f32)?
            .sum_keepdim(2)?
            .broadcast_add(&Tensor::new(eps as f32, x.device())?)?
            .sqrt()?
            .recip()?;
        x_f32.broadcast_mul(&inv_norm)?.to_dtype(x.dtype())
    }

    fn recurrent_delta_rule_single_sequence(
        &self,
        q_seq: &Tensor,         // [seq, heads, k_dim], l2-normalized
        k_seq: &Tensor,         // [seq, heads, k_dim], l2-normalized
        v_seq: &Tensor,         // [seq, heads, v_dim]
        g_seq: &Tensor,         // [seq, heads]
        beta_seq: &Tensor,      // [seq, heads]
        state_seq: &mut Tensor, // [heads, k_dim, v_dim]
    ) -> Result<Tensor> {
        let (_seq_len, heads, k_dim) = q_seq.dims3()?;
        let v_dim = v_seq.dim(2)?;
        let scale = 1.0f64 / (k_dim as f64).sqrt();

        let q_bh = (q_seq.transpose(0, 1)?.contiguous()?.to_dtype(DType::F32)? * scale)?;
        let k_bh = k_seq.transpose(0, 1)?.contiguous()?.to_dtype(DType::F32)?;
        let v_bh = v_seq.transpose(0, 1)?.contiguous()?.to_dtype(DType::F32)?;
        let g_bh = g_seq.transpose(0, 1)?.contiguous()?.to_dtype(DType::F32)?;
        let beta_bh = beta_seq
            .transpose(0, 1)?
            .contiguous()?
            .to_dtype(DType::F32)?;

        let state_dtype = state_seq.dtype();
        let mut state_bh = state_seq.to_dtype(DType::F32)?.contiguous()?;
        let out_bh =
            gdn::gated_delta_rule_recurrence(&q_bh, &k_bh, &v_bh, &g_bh, &beta_bh, &mut state_bh)?;
        *state_seq = state_bh.to_dtype(state_dtype)?;

        out_bh
            .reshape((heads, q_seq.dim(0)?, v_dim))?
            .transpose(0, 1)?
            .contiguous()?
            .to_dtype(q_seq.dtype())
    }

    fn recurrent_delta_rule_decode_batch(
        &self,
        q: &Tensor,                   // [batch, heads, k_dim], l2-normalized
        k: &Tensor,                   // [batch, heads, k_dim], l2-normalized
        v: &Tensor,                   // [batch, heads, v_dim]
        g: &Tensor,                   // [batch, heads]
        beta: &Tensor,                // [batch, heads]
        recurrent_state: &mut Tensor, // [batch, heads, k_dim, v_dim]
    ) -> Result<Tensor> {
        let (batch, heads, k_dim) = q.dims3()?;
        let v_dim = v.dim(2)?;
        let scale = 1.0f64 / (k_dim as f64).sqrt();

        let q_bh = (q
            .contiguous()?
            .reshape((batch * heads, 1, k_dim))?
            .to_dtype(DType::F32)?
            * scale)?;
        let k_bh = k
            .contiguous()?
            .reshape((batch * heads, 1, k_dim))?
            .to_dtype(DType::F32)?;
        let v_bh = v
            .contiguous()?
            .reshape((batch * heads, 1, v_dim))?
            .to_dtype(DType::F32)?;
        let g_bh = g
            .contiguous()?
            .reshape((batch * heads, 1))?
            .to_dtype(DType::F32)?;
        let beta_bh = beta
            .contiguous()?
            .reshape((batch * heads, 1))?
            .to_dtype(DType::F32)?;

        let state_dtype = recurrent_state.dtype();
        let mut state_bh = recurrent_state
            .to_dtype(DType::F32)?
            .reshape((batch * heads, k_dim, v_dim))?
            .contiguous()?;

        let out_bh =
            gdn::gated_delta_rule_recurrence(&q_bh, &k_bh, &v_bh, &g_bh, &beta_bh, &mut state_bh)?;
        *recurrent_state = state_bh
            .reshape((batch, heads, k_dim, v_dim))?
            .to_dtype(state_dtype)?;

        out_bh
            .reshape((batch, heads, 1, v_dim))?
            .transpose(1, 2)?
            .squeeze(1)?
            .to_dtype(q.dtype())
    }

    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        gdn_layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let hybrid = resolve_qwen3_hybrid_config(config);
        let world_size = comm.world_size();
        let rank = comm.rank();

        let num_v_heads_global = hybrid.num_v_heads;
        let num_k_heads_global = hybrid.num_k_heads;
        if num_v_heads_global % num_k_heads_global != 0 {
            candle_core::bail!(
                "linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
                num_v_heads_global,
                num_k_heads_global
            );
        }
        if num_v_heads_global % world_size != 0 || num_k_heads_global % world_size != 0 {
            candle_core::bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                num_v_heads_global,
                num_k_heads_global,
                world_size
            );
        }

        let num_v_heads = num_v_heads_global / world_size;
        let num_k_heads = num_k_heads_global / world_size;
        let head_k_dim = hybrid.key_head_dim;
        let head_v_dim = hybrid.value_head_dim;
        let key_dim_global = num_k_heads_global * head_k_dim;
        let value_dim_global = num_v_heads_global * head_v_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let kv_group_size = num_v_heads / num_k_heads;
        let conv_kernel_size = hybrid.conv_kernel_size;
        let conv_dim_global = key_dim_global * 2 + value_dim_global;
        // linear_attn.A_log	[32]	BF16

        // linear_attn.conv1d.weight	[8 192, 1, 4]	BF16
        // linear_attn.dt_bias	[32]	BF16
        // linear_attn.in_proj_ba.weight	[64, 2 048]	BF16
        // Learned GDN parameters
        let a_log = vb
            .get((num_v_heads_global,), "A_log")?
            .narrow(0, rank * num_v_heads, num_v_heads)?
            .contiguous()?;
        let dt_bias = vb
            .get((num_v_heads_global,), "dt_bias")?
            .narrow(0, rank * num_v_heads, num_v_heads)?
            .contiguous()?;

        let projection = Self::load_projection(
            &vb,
            hidden_size,
            key_dim_global,
            value_dim_global,
            num_v_heads_global,
            comm.clone(),
            config,
            dtype,
        )?;

        // Conv1D weights are stored global; slice rank-local q/k/v channel blocks.
        let conv_weight = vb.get((conv_dim_global, 1, conv_kernel_size), "conv1d.weight")?;
        let q_start = rank * key_dim;
        let k_start = key_dim_global + rank * key_dim;
        let v_start = key_dim_global * 2 + rank * value_dim;
        let q_w = conv_weight.narrow(0, q_start, key_dim)?;
        let k_w = conv_weight.narrow(0, k_start, key_dim)?;
        let v_w = conv_weight.narrow(0, v_start, value_dim)?;
        let conv_weight = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;

        let conv_bias = vb.get((conv_dim_global,), "conv1d.bias").ok();
        let conv_bias = if let Some(cb) = conv_bias {
            let q_b = cb.narrow(0, q_start, key_dim)?;
            let k_b = cb.narrow(0, k_start, key_dim)?;
            let v_b = cb.narrow(0, v_start, value_dim)?;
            Some(Tensor::cat(&[&q_b, &k_b, &v_b], 0)?)
        } else {
            None
        };

        // linear_attn.norm.weight	[128]	BF16

        // linear_attn.out_proj(2)
        // linear_attn.out_proj.weight	[2 048, 4 096]	F8_E4M3
        // linear_attn.out_proj.weight_scale_inv	[16, 32]	BF16

        // Output projection
        let out_proj = TensorParallelRowLinear::load_with_hints(
            value_dim_global,
            hidden_size,
            vb.pp("out_proj"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        // GDN output norm (gated RMSNorm)
        // Qwen3.5 checkpoints typically store [value_dim], while Qwen3Next can store per-head [head_v_dim].
        let (gdn_norm_weight, gdn_norm_bias, gdn_norm_per_head) = match vb
            .get((value_dim_global,), "norm.weight")
        {
            Ok(weight) => {
                let weight = weight
                    .narrow(0, rank * value_dim, value_dim)?
                    .contiguous()?;
                let bias = vb
                    .get((value_dim_global,), "norm.bias")
                    .ok()
                    .map(|b| {
                        b.narrow(0, rank * value_dim, value_dim)
                            .and_then(|x| x.contiguous())
                    })
                    .transpose()?;
                (weight, bias, false)
            }
            Err(full_err) => match vb.get((head_v_dim,), "norm.weight") {
                Ok(weight) => {
                    let bias = vb.get((head_v_dim,), "norm.bias").ok();
                    (weight, bias, true)
                }
                Err(head_err) => {
                    candle_core::bail!(
                            "Unable to load linear_attn.norm.weight: expected [{value_dim_global}] or [{head_v_dim}], full={full_err}, per_head={head_err}"
                        )
                }
            },
        };

        Ok(Self {
            projection,
            out_proj,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            gdn_norm_weight,
            gdn_norm_bias,
            gdn_norm_per_head,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            kv_group_size,
            gdn_layer_idx,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mamba_cache: &mut MambaCache,
        input_metadata: &InputMetadata,
        seq_slots: &[usize],
    ) -> Result<Tensor> {
        if seq_slots.is_empty() {
            candle_core::bail!("Linear attention requires non-empty sequence slots");
        }
        let (token_count, _hidden) = xs.dims2()?;
        let is_prefill = input_metadata.is_prefill;

        let (q, k, v, z, b, a) = self.project_inputs(xs)?;
        let mixed_qkv = Tensor::cat(&[&q, &k, &v], 1)?;

        let mut conv_state = mamba_cache.get_batch_conv_state(self.gdn_layer_idx, seq_slots)?;
        let kv_conv = if is_prefill {
            let lengths = Self::prefill_lengths(input_metadata, token_count)?;
            if lengths.len() != seq_slots.len() {
                candle_core::bail!(
                    "Linear attention prefill mismatch: {} sequences in metadata vs {} sequence slots",
                    lengths.len(),
                    seq_slots.len()
                );
            }
            let mut cu = Vec::with_capacity(lengths.len() + 1);
            cu.push(0u32);
            let mut acc = 0u32;
            for &len in &lengths {
                acc += len as u32;
                cu.push(acc);
            }
            let cu_len = cu.len();
            let cu_seqlens = Tensor::from_vec(cu, (cu_len,), xs.device())?;
            gdn::causal_conv1d_fwd(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_state,
                Some(&cu_seqlens),
                true, // SiLU activation
            )?
        } else {
            if token_count != seq_slots.len() {
                candle_core::bail!(
                    "Linear attention decode mismatch: {} tokens vs {} sequence slots",
                    token_count,
                    seq_slots.len()
                );
            }
            gdn::causal_conv1d_update(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_state,
                true,
            )?
        };
        mamba_cache.set_batch_conv_state(self.gdn_layer_idx, seq_slots, &conv_state)?;

        // Split convolved output back into q', k', v'
        let q_conv = kv_conv.narrow(1, 0, self.key_dim)?;
        let k_conv = kv_conv.narrow(1, self.key_dim, self.key_dim)?;
        let v_conv = kv_conv.narrow(1, self.key_dim * 2, self.value_dim)?;

        // Fused GDN gating
        let a_expanded = a.unsqueeze(0)?; // [1, seq_len, num_heads]
        let b_expanded = b.unsqueeze(0)?;
        let (g, beta) =
            gdn::fused_gdn_gating(&self.a_log, &a_expanded, &b_expanded, &self.dt_bias)?;
        let g = g.squeeze(0)?;
        let beta = beta.squeeze(0)?;

        let q = q_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let k = k_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let v = v_conv.reshape((token_count, self.num_v_heads, self.head_v_dim))?;
        let q = self.repeat_kv_heads(q)?;
        let k = self.repeat_kv_heads(k)?;
        let q = Self::l2_norm_last_dim(&q, 1e-6)?;
        let k = Self::l2_norm_last_dim(&k, 1e-6)?;

        let mut recurrent_state =
            mamba_cache.get_batch_recurrent_state(self.gdn_layer_idx, seq_slots)?;
        let output = if is_prefill {
            let lengths = Self::prefill_lengths(input_metadata, token_count)?;
            let mut seq_outputs = Vec::with_capacity(lengths.len());
            let mut start = 0usize;
            for (batch_idx, &len) in lengths.iter().enumerate() {
                let q_seq = q.narrow(0, start, len)?;
                let k_seq = k.narrow(0, start, len)?;
                let v_seq = v.narrow(0, start, len)?;
                let g_seq = g.narrow(0, start, len)?;
                let beta_seq = beta.narrow(0, start, len)?;

                let mut state_seq = recurrent_state.i(batch_idx)?;
                let out_seq = self.recurrent_delta_rule_single_sequence(
                    &q_seq,
                    &k_seq,
                    &v_seq,
                    &g_seq,
                    &beta_seq,
                    &mut state_seq,
                )?;
                recurrent_state = recurrent_state.slice_assign(
                    &[
                        batch_idx..batch_idx + 1,
                        0..self.num_v_heads,
                        0..self.head_k_dim,
                        0..self.head_v_dim,
                    ],
                    &state_seq.unsqueeze(0)?,
                )?;
                seq_outputs.push(out_seq);
                start += len;
            }
            let seq_output_refs = seq_outputs.iter().collect::<Vec<_>>();
            Tensor::cat(&seq_output_refs, 0)?
        } else {
            let batch = seq_slots.len();
            let q_b = q.reshape((batch, self.num_v_heads, self.head_k_dim))?;
            let k_b = k.reshape((batch, self.num_v_heads, self.head_k_dim))?;
            let v_b = v.reshape((batch, self.num_v_heads, self.head_v_dim))?;
            let g_b = g.reshape((batch, self.num_v_heads))?;
            let beta_b = beta.reshape((batch, self.num_v_heads))?;
            self.recurrent_delta_rule_decode_batch(
                &q_b,
                &k_b,
                &v_b,
                &g_b,
                &beta_b,
                &mut recurrent_state,
            )?
        };
        mamba_cache.set_batch_recurrent_state(self.gdn_layer_idx, seq_slots, &recurrent_state)?;

        // output: [seq_len, num_v_heads, head_v_dim] -> [seq_len, value_dim]
        let output = output.reshape((token_count, self.value_dim))?;

        // Gated RMSNorm: norm(output) * silu(z)
        let z_gate = candle_nn::ops::silu(&z)?;
        let normed = self.gated_rms_norm(&output)?;
        let gated_output = (normed * z_gate)?;

        // Output projection
        self.out_proj.forward(&gated_output.to_dtype(xs.dtype())?)
    }

    fn gated_rms_norm(&self, x: &Tensor) -> Result<Tensor> {
        if self.gdn_norm_per_head {
            let (token_count, _) = x.dims2()?;
            let x_f32 = x.to_dtype(DType::F32)?.reshape((
                token_count,
                self.num_v_heads,
                self.head_v_dim,
            ))?;
            let variance = (&x_f32 * &x_f32)?.mean_keepdim(2)?;
            let x_normed = x_f32.broadcast_div(&(variance + self.rms_norm_eps)?.sqrt()?)?;
            let x_normed = x_normed.broadcast_mul(&self.gdn_norm_weight.to_dtype(DType::F32)?)?;
            let x_normed = if let Some(ref bias) = self.gdn_norm_bias {
                x_normed.broadcast_add(&bias.to_dtype(DType::F32)?)?
            } else {
                x_normed
            };
            x_normed
                .reshape((token_count, self.value_dim))?
                .to_dtype(x.dtype())
        } else {
            // Simple RMSNorm with learnable weight (and optional bias) across full value dim.
            let x_f32 = x.to_dtype(DType::F32)?;
            let variance = (&x_f32 * &x_f32)?.mean_keepdim(1)?;
            let x_normed = x_f32.broadcast_div(&(variance + self.rms_norm_eps)?.sqrt()?)?;
            let x_normed = x_normed.broadcast_mul(&self.gdn_norm_weight.to_dtype(DType::F32)?)?;
            let x_normed = if let Some(ref bias) = self.gdn_norm_bias {
                x_normed.broadcast_add(&bias.to_dtype(DType::F32)?)?
            } else {
                x_normed
            };
            x_normed.to_dtype(x.dtype())
        }
    }
}
