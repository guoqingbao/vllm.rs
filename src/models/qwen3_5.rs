// src/models/qwen3_5.rs
// Qwen3.5 dense model with hybrid attention (full attention + GatedDeltaNet layers)
use crate::models::layers::attention::Attention;
use crate::models::layers::distributed::{Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear};
use crate::models::layers::mask::get_attention_causal_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{embedding, rms_norm, NormX};
use crate::models::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::ProgressLike;
use attention_rs::gdn;
use attention_rs::mamba_cache::MambaCache;
use attention_rs::InputMetadata;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use parking_lot::RwLock;
use std::rc::Rc;
use std::sync::Arc;

// =============================================================================
// GatedDeltaNet layer (linear attention with gated delta rule)
// =============================================================================

pub struct GatedDeltaNet {
    // Input projection: hidden_size -> 2 * (num_heads * key_head_dim) + 2 * (num_heads * value_head_dim) + num_heads
    in_proj: TensorParallelColumnLinear,
    // Beta/alpha projection: hidden_size -> 2 * num_heads
    in_proj_ba: TensorParallelColumnLinear,
    // Output projection: num_heads * value_head_dim -> hidden_size
    out_proj: TensorParallelRowLinear,
    // Conv1d weight: [d_conv, 1, conv_kernel_size] where d_conv = num_heads * (key_head_dim + value_head_dim)
    conv_weight: Tensor,
    conv_bias: Option<Tensor>,
    // Learned parameters
    a_log: Tensor,    // [num_heads] - log-space decay
    dt_bias: Tensor,  // [num_heads] - dt bias for gating
    // GDN normalization (gated rms norm on output)
    gdn_norm_weight: Tensor,
    gdn_norm_bias: Option<Tensor>,
    // Layer dimensions
    num_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_size: usize,
    gdn_layer_idx: usize, // index into MambaCache (0-based among GDN layers)
    rms_norm_eps: f64,
    dtype: DType,
}

impl GatedDeltaNet {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        gdn_layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.linear_num_heads.unwrap_or(config.num_attention_heads);
        let key_head_dim = config.linear_key_head_dim.unwrap_or(
            config.head_dim.unwrap_or(hidden_size / config.num_attention_heads),
        );
        let value_head_dim = config.linear_value_head_dim.unwrap_or(key_head_dim);
        let conv_kernel_size = config.conv_kernel_size.unwrap_or(4);

        // Projection sizes
        // in_proj outputs: q(num_heads*key_head_dim) + k(num_heads*key_head_dim) + v(num_heads*value_head_dim) + z(num_heads*value_head_dim) + dt(num_heads)
        let qkv_z_dt_size = num_heads * (2 * key_head_dim + 2 * value_head_dim + 1);
        let in_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            qkv_z_dt_size,
            false, // no bias
            vb.pp("in_proj"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        // Beta/alpha projection: hidden_size -> 2 * num_heads
        let in_proj_ba = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            2 * num_heads,
            false,
            vb.pp("in_proj_ba"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        // Output projection
        let out_proj = TensorParallelRowLinear::load_with_hints(
            num_heads * value_head_dim,
            hidden_size,
            vb.pp("out_proj"),
            comm.clone(),
            &config.quantization_config,
            &config.quant,
            dtype,
        )?;

        // Conv1d parameters
        let d_conv = num_heads * (key_head_dim + value_head_dim);
        let conv_weight = vb.get((d_conv, 1, conv_kernel_size), "conv1d.weight")?;
        let conv_bias = vb.get((d_conv,), "conv1d.bias").ok();

        // Learned GDN parameters
        let a_log = vb.get((num_heads,), "A_log")?;
        let dt_bias = vb.get((num_heads,), "dt_bias")?;

        // GDN output norm (gated RMSNorm)
        let gdn_norm_weight = vb.get((num_heads * value_head_dim,), "norm.weight")?;
        let gdn_norm_bias = vb.get((num_heads * value_head_dim,), "norm.bias").ok();

        Ok(Self {
            in_proj,
            in_proj_ba,
            out_proj,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            gdn_norm_weight,
            gdn_norm_bias,
            num_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            gdn_layer_idx,
            rms_norm_eps: config.rms_norm_eps,
            dtype,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mamba_cache: &mut MambaCache,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _hidden) = xs.dims2()?;
        let is_prefill = input_metadata.is_prefill;

        // Project input -> q, k, v, z, dt
        let proj = self.in_proj.forward(xs)?; // [seq_len, qkv_z_dt_size]

        // Split into components
        let q_size = self.num_heads * self.key_head_dim;
        let k_size = self.num_heads * self.key_head_dim;
        let v_size = self.num_heads * self.value_head_dim;
        let z_size = self.num_heads * self.value_head_dim;
        let dt_size = self.num_heads;

        let mut offset = 0;
        let q = proj.narrow(1, offset, q_size)?;
        offset += q_size;
        let k = proj.narrow(1, offset, k_size)?;
        offset += k_size;
        let v = proj.narrow(1, offset, v_size)?;
        offset += v_size;
        let z = proj.narrow(1, offset, z_size)?;
        offset += z_size;
        let _dt = proj.narrow(1, offset, dt_size)?;

        // Project alpha/beta
        let ba = self.in_proj_ba.forward(xs)?; // [seq_len, 2 * num_heads]
        let a = ba.narrow(1, 0, self.num_heads)?;
        let b = ba.narrow(1, self.num_heads, self.num_heads)?;

        // Causal conv1d on concatenated [k, v]
        let d_conv = self.num_heads * (self.key_head_dim + self.value_head_dim);
        let kv_cat = Tensor::cat(&[&k, &v], 1)?; // [seq_len, d_conv]

        let mut conv_state = mamba_cache.conv_state(self.gdn_layer_idx).clone();
        let kv_conv = if is_prefill {
            gdn::causal_conv1d_fwd(
                &kv_cat,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_state,
                None, // cu_seqlens
                true, // SiLU activation
            )?
        } else {
            gdn::causal_conv1d_update(
                &kv_cat.unsqueeze(0)?,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_state,
                true,
            )?.squeeze(0)?
        };
        *mamba_cache.conv_state_mut(self.gdn_layer_idx) = conv_state;

        // Split convolved output back into k', v'
        let k_conv = kv_conv.narrow(1, 0, self.num_heads * self.key_head_dim)?;
        let v_conv = kv_conv.narrow(1, self.num_heads * self.key_head_dim, self.num_heads * self.value_head_dim)?;

        // Fused GDN gating
        let a_expanded = a.unsqueeze(0)?; // [1, seq_len, num_heads]
        let b_expanded = b.unsqueeze(0)?;
        let (g, beta) = gdn::fused_gdn_gating(
            &self.a_log,
            &a_expanded,
            &b_expanded,
            &self.dt_bias,
        )?;
        let g = g.squeeze(0)?; // [seq_len, num_heads]
        let beta = beta.squeeze(0)?;

        // Reshape for chunked delta rule
        let q_reshaped = q.reshape((1, seq_len, self.num_heads, self.key_head_dim))?;
        let k_reshaped = k_conv.reshape((1, seq_len, self.num_heads, self.key_head_dim))?;
        let v_reshaped = v_conv.reshape((1, seq_len, self.num_heads, self.value_head_dim))?;
        let g_reshaped = g.reshape((1, seq_len, self.num_heads))?;
        let beta_reshaped = beta.reshape((1, seq_len, self.num_heads))?;

        // Delta rule recurrence
        let mut rec_state = mamba_cache.recurrent_state(self.gdn_layer_idx).clone();
        let output = gdn::chunk_gated_delta_rule(
            &q_reshaped,
            &k_reshaped,
            &v_reshaped,
            &g_reshaped,
            &beta_reshaped,
            &mut rec_state,
            is_prefill,
        )?;
        *mamba_cache.recurrent_state_mut(self.gdn_layer_idx) = rec_state;

        // output: [1, seq_len, num_heads, value_head_dim] -> [seq_len, num_heads * value_head_dim]
        let output = output.squeeze(0)?.reshape((seq_len, self.num_heads * self.value_head_dim))?;

        // Gated RMSNorm: norm(output) * silu(z)
        let z_gate = candle_nn::ops::silu(&z)?;
        let normed = self.gated_rms_norm(&output)?;
        let gated_output = (normed * z_gate)?;

        // Output projection
        self.out_proj.forward(&gated_output.to_dtype(xs.dtype())?)
    }

    fn gated_rms_norm(&self, x: &Tensor) -> Result<Tensor> {
        // Simple RMSNorm with learnable weight (and optional bias)
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

// =============================================================================
// Hybrid decoder layer: either full attention or GatedDeltaNet
// =============================================================================

pub enum Qwen3_5AttnType {
    FullAttention(Attention),
    LinearAttention(GatedDeltaNet),
}

pub struct Qwen3_5DecoderLayer {
    attn: Qwen3_5AttnType,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Option<Arc<ScalingRotaryEmbedding>>,
}

impl Qwen3_5DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        layer_type: &str,
        gdn_layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        let attn = if layer_type == "full_attention" {
            Qwen3_5AttnType::FullAttention(Attention::new(
                if is_qvar_builder { vb.clone() } else { vb.pp("self_attn").clone() },
                comm.clone(),
                config,
                None,
                config.sliding_window,
                dtype,
            )?)
        } else {
            Qwen3_5AttnType::LinearAttention(GatedDeltaNet::new(
                if is_qvar_builder { vb.clone() } else { vb.pp("linear_attn").clone() },
                comm.clone(),
                config,
                gdn_layer_idx,
                dtype,
            )?)
        };

        let mlp = MLP::new(
            if is_qvar_builder { vb.clone() } else { vb.pp("mlp").clone() },
            comm.clone(),
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
            &config.quantization_config,
            &config.quant,
            false,
            dtype,
            "",
        )?;

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder { vb.pp("attn_norm").clone() } else { vb.pp("input_layernorm").clone() },
            DType::F32,
            false,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder { vb.pp("ffn_norm").clone() } else { vb.pp("post_attention_layernorm").clone() },
            DType::F32,
            false,
        )?;

        let rotary = if layer_type == "full_attention" {
            Some(rotary_emb)
        } else {
            None
        };

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb: rotary,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        mamba_cache: &mut MambaCache,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        let attn_output = match &self.attn {
            Qwen3_5AttnType::FullAttention(attn) => {
                let rope: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.as_ref().unwrap().clone();
                attn.forward(&xs, &Some(rope), attention_mask, positions, cache, input_metadata)?
            }
            Qwen3_5AttnType::LinearAttention(gdn) => {
                gdn.forward(&xs, mamba_cache, input_metadata)?
            }
        };

        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
    }

    pub fn is_full_attention(&self) -> bool {
        matches!(&self.attn, Qwen3_5AttnType::FullAttention(_))
    }
}

// =============================================================================
// Qwen3.5 causal LM (dense variant)
// =============================================================================

pub struct Qwen3_5ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    mamba_cache: RwLock<MambaCache>,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
}

impl Qwen3_5ForCausalLM {
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<Box<dyn ProgressLike>>>,
    ) -> Result<Self> {
        let prefix = "model.".to_string();
        let is_qvar_builder = vb.is_qvar_builder();

        let tie_word_embeddings = config.tie_word_embeddings;

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp("token_embd")
            } else {
                vb.pp(&format!("{}embed_tokens", prefix))
            },
            if is_qvar_builder || config.quant.is_some() {
                DType::F32
            } else {
                dtype
            },
        )?;

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder || config.quant.is_some() {
                DType::F32
            } else {
                dtype
            },
            config,
            &vb.device(),
            is_rope_i,
            config.rope_theta,
        )?);

        // Get layer block types
        let default_layer_types: Vec<String> = (0..config.num_hidden_layers)
            .map(|_| "full_attention".to_string())
            .collect();
        let layer_types = config.layers_block_type.as_ref().unwrap_or(&default_layer_types);

        // Build layers, tracking GDN layer index separately
        let mut layers = Vec::new();
        let mut gdn_layer_idx = 0usize;
        let reporter = progress_reporter.clone();

        for i in 0..config.num_hidden_layers {
            let layer_type = &layer_types[i];
            let current_gdn_idx = if layer_type == "linear_attention" {
                let idx = gdn_layer_idx;
                gdn_layer_idx += 1;
                idx
            } else {
                0 // unused for full attention
            };

            let layer = Qwen3_5DecoderLayer::new(
                vb.pp(format!(
                    "{}.{}",
                    if is_qvar_builder {
                        "blk".to_string()
                    } else {
                        format!("{}layers", prefix)
                    },
                    i
                ).as_str()),
                comm.clone(),
                rotary_emb.clone(),
                config,
                layer_type,
                current_gdn_idx,
                dtype,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let num_gdn_layers = gdn_layer_idx;

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("output_norm")
            } else {
                vb.pp(&format!("{}norm", prefix))
            },
            DType::F32,
            false,
        )?;

        let lm_head = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            vocab_size,
            if tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp("token_embd")
                } else {
                    vb.pp(&format!("{}embed_tokens", prefix))
                }
            } else {
                if is_qvar_builder {
                    vb.pp("output")
                } else {
                    vb.pp("lm_head")
                }
            },
            &None,
            &None,
            dtype,
        )?;

        // Initialize MambaCache for GDN layers
        let num_heads = config.linear_num_heads.unwrap_or(config.num_attention_heads);
        let key_head_dim = config.linear_key_head_dim.unwrap_or(
            config.head_dim.unwrap_or(config.hidden_size / config.num_attention_heads),
        );
        let value_head_dim = config.linear_value_head_dim.unwrap_or(key_head_dim);
        let conv_kernel_size = config.conv_kernel_size.unwrap_or(4);
        let d_conv = num_heads * (key_head_dim + value_head_dim);

        // max_batch_size: use a reasonable default; will be updated at runtime
        let max_batch_size = 8; // Conservative default
        let mamba_cache = if num_gdn_layers > 0 {
            MambaCache::new(
                num_gdn_layers,
                max_batch_size,
                d_conv,
                conv_kernel_size,
                num_heads,
                key_head_dim, // recurrent state uses key_head_dim for both dims
                dtype,
                device,
            )?
        } else {
            // No GDN layers, create minimal cache
            MambaCache::new(0, 1, 1, 2, 1, 1, dtype, device)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mamba_cache: RwLock::new(mamba_cache),
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            is_qvar_builder,
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(xs)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();

        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );

        let mut xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_tokens.forward(input_ids)?
        };

        // Key insight: kv_caches only has entries for full-attention layers
        // We track a separate kv_cache_idx to index into kv_caches correctly
        let mut kv_cache_idx = 0usize;
        let mut mamba_cache = self.mamba_cache.write();

        for (_i, layer) in self.layers.iter().enumerate() {
            let cache = if layer.is_full_attention() {
                if let Some(kv_caches) = kv_caches {
                    let c = &kv_caches[kv_cache_idx];
                    kv_cache_idx += 1;
                    Some((&c.0, &c.1))
                } else {
                    None
                }
            } else {
                None // GDN layers don't use KV cache
            };

            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                positions,
                cache,
                input_metadata,
                &mut mamba_cache,
            )?;
        }

        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }

        let xs = self.norm.forward(&xs)?;
        if self.is_qvar_builder {
            self.lm_head.forward(&xs)
        } else {
            self.lm_head
                .forward(&xs.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)
        }
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        // For embedding models, return hidden states
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.clone(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );

        let mut xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_tokens.forward(input_ids)?
        };

        let mut kv_cache_idx = 0usize;
        let mut mamba_cache = self.mamba_cache.write();
        for (_i, layer) in self.layers.iter().enumerate() {
            let cache = if layer.is_full_attention() {
                if let Some(kv_caches) = kv_caches {
                    let c = &kv_caches[kv_cache_idx];
                    kv_cache_idx += 1;
                    Some((&c.0, &c.1))
                } else {
                    None
                }
            } else {
                None
            };
            xs = layer.forward(&xs, attention_mask.as_ref(), positions, cache, input_metadata, &mut mamba_cache)?;
        }

        let xs = self.norm.forward(&xs)?;
        xs.to_dtype(DType::F32)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
