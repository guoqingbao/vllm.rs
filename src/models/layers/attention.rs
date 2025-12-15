use crate::models::layers::distributed::{
    Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
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

pub struct Attention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    q_norm: Option<NormX>,
    k_norm: Option<NormX>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn: PagedAttention,
    softcapping: Option<f64>,
    dtype: DType,
    is_gemma: bool,
}

impl Attention {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        attention_scale: Option<f32>,
        sliding_window: Option<usize>,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_heads);
        let attention_bias = config.attention_bias.unwrap_or(true);
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

        let arch = config.architectures.as_ref().unwrap()[0].clone();
        let is_gemma = arch == "Gemma3ForConditionalGeneration".to_string()
            || arch == "Gemma3ForCausalLM".to_string();

        let q_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            num_heads * head_dim,
            attention_bias,
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
        let k_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            num_kv_heads * head_dim,
            attention_bias,
            if is_qvar_builder {
                vb.pp(key_map["k_proj"])
            } else {
                vb.pp("k_proj")
            },
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;
        //v_proj requires higher precision format
        let q8_0_qunat = Some("q8_0".to_string());
        let v_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            num_kv_heads * head_dim,
            attention_bias,
            if is_qvar_builder {
                vb.pp(key_map["v_proj"])
            } else {
                vb.pp("v_proj")
            },
            comm.clone(),
            &config.quantization_config,
            if config.quant.is_some() && config.quantization_config.is_none() {
                &q8_0_qunat
            } else {
                &None
            },
            dtype,
        )?;

        let o_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * head_dim,
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
            is_gemma,
        );
        let q_norm = if q_norm.is_ok() {
            Some(q_norm.unwrap())
        } else {
            None
        };

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
            is_gemma,
        );
        let k_norm = if k_norm.is_ok() {
            Some(k_norm.unwrap())
        } else {
            None
        };

        let attention_heads = num_heads / comm.world_size();
        let kv_heads = num_kv_heads / comm.world_size();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: attention_heads,
            num_kv_heads: kv_heads,
            head_dim,
            attn: PagedAttention::new(
                attention_heads,
                head_dim,
                attention_scale.unwrap_or(1. / (head_dim as f32).sqrt()),
                Some(kv_heads),
                sliding_window,
                vb.device().clone(),
                None,
                config.fp8_kvcache.unwrap_or(false),
            )?,
            softcapping: config.attn_logit_softcapping,
            dtype,
            is_gemma,
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

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
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

        let (q, k) = if self.q_norm.is_some() && self.k_norm.is_some() {
            //Per‑head RMSNorm in qwen3
            if self.is_gemma {
                let q = self.q_norm.as_ref().unwrap().forward(&q)?;
                let k = self.k_norm.as_ref().unwrap().forward(&k)?;
                (q, k)
            } else {
                let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> (BHL, D) after transpose later
                let k_flat = k.flatten(0, 2)?;
                let q_flat = self.q_norm.as_ref().unwrap().forward(&q_flat)?;
                let k_flat = self.k_norm.as_ref().unwrap().forward(&k_flat)?;
                let q = q_flat.reshape((1, self.num_heads, seq_len, self.head_dim))?;
                let k = k_flat.reshape((1, self.num_kv_heads, seq_len, self.head_dim))?;
                (q, k)
            }
        } else {
            (q, k)
        };

        // Apply rotary embeddings
        let (q, k) = if let Some(rotary_emb) = &rotary_emb {
            rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)?
        } else {
            (q, k)
        };

        let (mut q, k) = if q.dtype() != self.dtype {
            let q = q.to_dtype(self.dtype)?;
            let k = k.to_dtype(self.dtype)?;
            (q, k)
        } else {
            (q, k)
        };

        let v = if v.dtype() != self.dtype {
            v.to_dtype(self.dtype)?
        } else {
            v
        };

        if let Some(rotary_emb) = &rotary_emb {
            if let (Some(original_max_position_embeddings), Some(llama_4_scaling_beta)) = (
                rotary_emb.get_original_max_position_embeddings(),
                rotary_emb.get_llama_4_scaling_beta(),
            ) {
                use crate::utils::get_llama4_attn_scale;
                let scale = get_llama4_attn_scale(
                    &positions,
                    llama_4_scaling_beta,
                    original_max_position_embeddings as f64,
                )?
                .to_dtype(q.dtype())?;
                q = q.broadcast_mul(&scale)?;
            }
        }

        let y = self
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

        self.o_proj.forward(&y.to_dtype(xs.dtype())?)
    }
}

pub struct NaiveAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    scale: f64,
    num_heads: usize,
    head_dim: usize,
    softcapping: Option<f64>,
}

impl NaiveAttention {
    pub fn new(
        vb: VarBuilderX,
        num_heads: usize,
        hidden_size: usize,
        head_dim: usize,
        softcapping: Option<f64>,
        dtype: DType,
        key_mappings: HashMap<String, String>,
    ) -> Result<Self> {
        let key_map: HashMap<&str, &str> = [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();

        let q_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["q_proj"])
            } else {
                vb.pp("q_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let k_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["k_proj"])
            } else {
                vb.pp("k_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let v_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["v_proj"])
            } else {
                vb.pp("v_proj")
            },
            &None,
            &None,
            dtype,
        )?;

        let o_proj = ReplicatedLinear::load_b(
            hidden_size,
            hidden_size,
            true,
            if is_qvar_builder {
                vb.pp(key_map["o_proj"])
            } else {
                if key_mappings.contains_key("o_proj") {
                    vb.pp(&key_mappings["o_proj"])
                } else {
                    vb.pp("o_proj")
                }
            },
            &None,
            &None,
            dtype,
        )?;

        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            scale,
            num_heads,
            head_dim,
            softcapping,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        emb: &Arc<dyn ApplyRotaryEmbedding>,
        positions: &Option<Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let shape = (b, seq_len, self.num_heads, self.head_dim);
        let q = q.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let (q, k) = if let Some(positions) = positions {
            emb.apply_rotary_emb_qkv(&q, &k, positions)?
        } else {
            (q, k)
        };
        let mut attn = (q.matmul(&k.t()?)? * self.scale)?;

        if let Some(sc) = self.softcapping {
            attn = ((attn / sc)?.tanh()? * sc)?;
        }

        let attn = match mask {
            None => attn,
            Some(mask) => attn.broadcast_add(mask)?,
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        self.o_proj.forward(
            &attn
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, seq_len, ()))?,
        )
    }
}
