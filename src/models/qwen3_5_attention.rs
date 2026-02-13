// src/models/qwen3_5_attention.rs
// Model-specific full attention for Qwen3.5/Qwen3Next (supports attn_output_gate).
use crate::models::layers::distributed::{
    kv_head_shard, Comm, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use crate::models::layers::others::{rms_norm, NormX};
use crate::models::layers::rotary_emb::ApplyRotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use attention_rs::{InputMetadata, PagedAttention};
use candle_core::{DType, Result, Tensor};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

pub struct Qwen3_5Attention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    q_norm: NormX,
    k_norm: NormX,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_output_gate: bool,
    attn: PagedAttention,
    softcapping: Option<f64>,
    dtype: DType,
}

impl Qwen3_5Attention {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let total_num_heads = config.num_attention_heads;
        let total_num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim.unwrap_or(hidden_size / total_num_heads);
        let qkv_bias = config.qkv_bias.or(config.attention_bias).unwrap_or(false);
        let attn_output_gate = config.attn_output_gate.unwrap_or(true);
        let world_size = comm.world_size();

        if total_num_heads % world_size != 0 {
            candle_core::bail!(
                "attention heads must be divisible by tensor parallel world_size (num_heads={}, world_size={})",
                total_num_heads,
                world_size
            );
        }

        let key_map: HashMap<&str, &str> = [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
            ("q_norm", "attn_q_norm"),
            ("k_norm", "attn_k_norm"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();

        let q_out_dim = total_num_heads * head_dim * if attn_output_gate { 2 } else { 1 };
        let kv_out_dim = total_num_kv_heads * head_dim;
        let num_heads = total_num_heads / world_size;
        let (num_kv_heads, kv_shard) = kv_head_shard(total_num_kv_heads, comm.rank(), world_size)?;

        let q_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            q_out_dim,
            qkv_bias,
            if is_qvar_builder {
                vb.pp(key_map["q_proj"])
            } else {
                vb.pp("q_proj")
            },
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        let k_proj = TensorParallelColumnLinear::load_with_shard(
            hidden_size,
            kv_out_dim,
            qkv_bias,
            if is_qvar_builder {
                vb.pp(key_map["k_proj"])
            } else {
                vb.pp("k_proj")
            },
            kv_shard,
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        // Keep v_proj at higher precision when ISQ is used (same behavior as generic attention).
        let q8_0_quant = Some("q8_0".to_string());
        let v_proj = TensorParallelColumnLinear::load_with_shard(
            hidden_size,
            kv_out_dim,
            qkv_bias,
            if is_qvar_builder {
                vb.pp(key_map["v_proj"])
            } else {
                vb.pp("v_proj")
            },
            kv_shard,
            &config.quantization_config,
            if config.quant.is_some() && config.quantization_config.is_none() {
                &q8_0_quant
            } else {
                &None
            },
            dtype,
        )?;
        let o_proj = TensorParallelRowLinear::load_with_hints(
            total_num_heads * head_dim,
            hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["o_proj"])
            } else {
                vb.pp("o_proj")
            },
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        let q_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["q_norm"])
            } else {
                vb.pp("q_norm")
            },
            if is_qvar_builder || config.quant.is_some() || config.quantization_config.is_some() {
                DType::F32
            } else {
                dtype
            },
            true,
        )?;
        let k_norm = rms_norm(
            head_dim,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["k_norm"])
            } else {
                vb.pp("k_norm")
            },
            if is_qvar_builder || config.quant.is_some() || config.quantization_config.is_some() {
                DType::F32
            } else {
                dtype
            },
            true,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            attn_output_gate,
            attn: PagedAttention::new(
                num_heads,
                head_dim,
                attention_scale.unwrap_or(1. / (head_dim as f32).sqrt()),
                Some(num_kv_heads),
                sliding_window,
                vb.device().clone(),
                None,
                config.fp8_kvcache.unwrap_or(false),
            )?,
            softcapping: config.attn_logit_softcapping,
            dtype,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        let q_raw = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let local_q_dim = self.num_heads * self.head_dim;
        let (q_linear, gate) = if self.attn_output_gate {
            let q_dim = q_raw.dim(1)?;
            if q_dim != local_q_dim * 2 {
                candle_core::bail!(
                    "q_proj output dim mismatch for gated attention, expected {}, got {}",
                    local_q_dim * 2,
                    q_dim
                );
            }
            // Split per-head as [q, gate] along head_dim*2, matching reference layout.
            let q_gate = q_raw.reshape((seq_len, self.num_heads, self.head_dim * 2))?;
            let q = q_gate.narrow(2, 0, self.head_dim)?;
            let gate = q_gate.narrow(2, self.head_dim, self.head_dim)?;
            (
                q.reshape((seq_len, local_q_dim))?,
                Some(gate.reshape((seq_len, local_q_dim))?),
            )
        } else {
            (q_raw, None)
        };

        let q = q_linear
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Per-head RMSNorm used by Qwen3.5/Qwen3Next.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (q, k) = if let Some(rotary_emb) = rotary_emb {
            match rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)? {
                Some((q_new, k_new)) => (q_new, k_new),
                None => (q, k),
            }
        } else {
            (q, k)
        };

        let (q, k) = if q.dtype() != self.dtype {
            (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?)
        } else {
            (q, k)
        };
        let v = if v.dtype() != self.dtype {
            v.to_dtype(self.dtype)?
        } else {
            v
        };

        let mut y = self
            .attn
            .forward(
                &q,
                &k,
                &v,
                attention_mask,
                cache.map(|(k_, _)| k_.clone()),
                cache.map(|(_, v_)| v_.clone()),
                input_metadata,
                self.softcapping,
            )?
            .reshape((seq_len, ()))?;

        if let Some(gate) = gate {
            let gate = if gate.dtype() != y.dtype() {
                gate.to_dtype(y.dtype())?
            } else {
                gate
            };
            let gate = candle_nn::ops::sigmoid(&gate)?;
            y = y.broadcast_mul(&gate)?;
        }

        self.o_proj.forward(&y.to_dtype(xs.dtype())?)
    }
}
