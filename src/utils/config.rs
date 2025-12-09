// src/utils/config.rs
use crate::transfer::PdConfig;
use either::Either;
#[cfg(feature = "python")]
use pyo3::pyclass;
use serde::de::value::SeqAccessDeserializer;
use serde::de::{Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl<'de> Deserialize<'de> for EosTokenId {
    fn deserialize<D>(deserializer: D) -> Result<EosTokenId, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            // For JSON: deserialize as "untagged" using a visitor
            struct EosTokenIdVisitor;

            impl<'de> Visitor<'de> for EosTokenIdVisitor {
                type Value = EosTokenId;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("a u32 or a sequence of u32s")
                }

                // Handle a single number
                fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                    Ok(EosTokenId::Single(v as u32))
                }

                // Handle an array of numbers
                fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
                where
                    A: serde::de::SeqAccess<'de>,
                {
                    let vals = Vec::<u32>::deserialize(SeqAccessDeserializer::new(seq))?;
                    Ok(EosTokenId::Multiple(vals))
                }
            }

            deserializer.deserialize_any(EosTokenIdVisitor)
        } else {
            // For Bincode: deserialize as "tagged"
            let bincode_id = BincodeEosTokenId::deserialize(deserializer)?;
            let id = match bincode_id {
                BincodeEosTokenId::Single(v) => EosTokenId::Single(v),
                BincodeEosTokenId::Multiple(v) => EosTokenId::Multiple(v),
            };
            Ok(id)
        }
    }
}

impl Serialize for EosTokenId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            // For JSON: serialize as "untagged"
            match self {
                EosTokenId::Single(v) => v.serialize(serializer),
                EosTokenId::Multiple(v) => v.serialize(serializer),
            }
        } else {
            // For Bincode: serialize as "tagged"
            let bincode_id = match self {
                EosTokenId::Single(v) => BincodeEosTokenId::Single(*v),
                EosTokenId::Multiple(v) => BincodeEosTokenId::Multiple(v.clone()),
            };
            bincode_id.serialize(serializer)
        }
    }
}

// To make the "tagged" logic work for bincode, we need a separate
// definition of the enum with derived traits. We keep it private inside this module.
#[derive(Serialize, Deserialize)]
enum BincodeEosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoEConfig {
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ScalingValue(#[serde(with = "either::serde_untagged")] pub Either<f64, Vec<f64>>);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RopeScaling(#[serde(with = "either::serde_untagged")] pub Either<ScalingValue, String>);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub architectures: Option<Vec<String>>,
    pub head_dim: Option<usize>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub max_model_len: Option<usize>,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: Option<usize>,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<EosTokenId>,
    pub use_sliding_window: Option<bool>,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub partial_rotary_factor: Option<f32>,
    pub hidden_act: candle_nn::Activation,
    #[serde(alias = "rope_parameters")]
    pub rope_scaling: Option<HashMap<String, RopeScaling>>,
    pub quant: Option<String>,
    pub moe_cfg: Option<MoEConfig>,
    pub fp8_kvcache: Option<bool>,
    pub quantization_config: Option<QuantConfig>,
    pub is_multi_model: Option<bool>,
    pub extra_config_json: Option<String>,
}

#[cfg(not(feature = "python"))]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EngineConfig {
    pub model_id: Option<String>,
    pub weight_path: Option<String>,
    pub weight_file: Option<String>,
    pub hf_token: Option<String>,
    pub hf_token_path: Option<String>,
    pub num_blocks: usize,
    pub kv_fraction: Option<f32>, // After loading the model, the remaining percent of gpu used for kvcache
    pub cpu_mem_fold: Option<f32>, // the percentage of gpu kvcache: 0.1x to 10x, default 1.0x
    pub kvcache_memory_bytes: usize,
    pub block_size: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub config_model_len: Option<usize>,
    pub max_model_len: Option<usize>,
    pub max_tokens: Option<usize>,
    pub isq: Option<String>,
    pub num_shards: Option<usize>,
    pub device_ids: Option<Vec<usize>>,
    pub generation_cfg: Option<GenerationConfig>,
    pub seed: Option<u64>,
    pub flash_context: Option<bool>,
    pub fp8_kvcache: Option<bool>,
    pub server_mode: Option<bool>,
    pub pd_config: Option<PdConfig>,
}

#[cfg(feature = "python")]
#[pyclass]
#[allow(unused_variables)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EngineConfig {
    #[pyo3(get, set)]
    pub model_id: Option<String>,
    #[pyo3(get, set)]
    pub weight_path: Option<String>,
    #[pyo3(get, set)]
    pub weight_file: Option<String>,
    #[pyo3(get, set)]
    pub hf_token: Option<String>,
    #[pyo3(get, set)]
    pub hf_token_path: Option<String>,
    #[pyo3(get, set)]
    pub num_blocks: usize,
    #[pyo3(get, set)]
    pub cpu_mem_fold: Option<f32>,
    #[pyo3(get, set)]
    pub kv_fraction: Option<f32>,
    pub block_size: usize,
    pub kvcache_memory_bytes: usize,
    #[pyo3(get, set)]
    pub max_num_seqs: usize,
    #[pyo3(get, set)]
    pub max_num_batched_tokens: usize,
    #[pyo3(get, set)]
    pub max_model_len: Option<usize>,
    #[pyo3(get, set)]
    pub config_model_len: Option<usize>,
    #[pyo3(get, set)]
    pub max_tokens: Option<usize>,
    #[pyo3(get, set)]
    pub isq: Option<String>,
    #[pyo3(get, set)]
    pub num_shards: Option<usize>,
    #[pyo3(get, set)]
    pub device_ids: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub generation_cfg: Option<GenerationConfig>,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub flash_context: Option<bool>,
    #[pyo3(get, set)]
    pub fp8_kvcache: Option<bool>,
    #[pyo3(get, set)]
    pub server_mode: Option<bool>,
    #[pyo3(get, set)]
    pub pd_config: Option<PdConfig>,
}

#[cfg(not(feature = "python"))]
impl EngineConfig {
    pub fn new(
        model_id: Option<String>,
        weight_path: Option<String>,
        weight_file: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
        max_num_seqs: Option<usize>,
        config_model_len: Option<usize>,
        max_model_len: Option<usize>,
        max_tokens: Option<usize>,
        isq: Option<String>,
        num_shards: Option<usize>,
        device_ids: Option<Vec<usize>>,
        generation_cfg: Option<GenerationConfig>,
        seed: Option<u64>,
        flash_context: Option<bool>,
        fp8_kvcache: Option<bool>,
        server_mode: Option<bool>,
        cpu_mem_fold: Option<f32>,
        kv_fraction: Option<f32>,
        pd_config: Option<PdConfig>,
    ) -> Self {
        let mut device_ids = device_ids.unwrap_or_default();
        if device_ids.is_empty() {
            device_ids.push(0);
        }

        if flash_context.unwrap_or(false)
            && fp8_kvcache.unwrap_or(false)
            && (cfg!(feature = "metal") || cfg!(feature = "flash-context"))
        {
            panic!("Error: context-cache and fp8 kvcache are not compatible under the current settings!\n\t***Tips: use only one of the two features (`--fp8-kvcache` or `--flash-context` (`--context-cache`)).");
        }

        Self {
            model_id,
            weight_path,
            weight_file,
            hf_token,
            hf_token_path,
            num_blocks: 128, //placeholder
            cpu_mem_fold,
            kv_fraction,
            kvcache_memory_bytes: 0, //placeholder
            block_size: 64,
            max_num_seqs: max_num_seqs.unwrap_or(32),
            max_num_batched_tokens: max_num_seqs.unwrap_or(32) * 1024, //placeholder
            config_model_len,
            max_model_len, //placeholder
            max_tokens,
            isq,
            num_shards,
            device_ids: Some(device_ids),
            generation_cfg,
            seed,
            flash_context,
            fp8_kvcache,
            server_mode,
            pd_config,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct TokenizerConfig {
    pub model_max_length: Option<f64>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

#[cfg(not(feature = "python"))]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub ignore_eos: bool,
    pub top_k: Option<isize>,
    pub top_p: Option<f32>,
    pub session_id: Option<String>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SamplingParams {
    #[pyo3(get, set)]
    pub temperature: Option<f32>,
    #[pyo3(get, set)]
    pub max_tokens: Option<usize>,
    #[pyo3(get, set)]
    pub ignore_eos: bool,
    #[pyo3(get, set)]
    pub top_k: Option<isize>,
    #[pyo3(get, set)]
    pub top_p: Option<f32>,
    #[pyo3(get, set)]
    pub session_id: Option<String>,
    #[pyo3(get, set)]
    pub frequency_penalty: Option<f32>,
    #[pyo3(get, set)]
    pub presence_penalty: Option<f32>,
}

#[cfg(not(feature = "python"))]
impl SamplingParams {
    pub fn new(
        temperature: Option<f32>,
        max_tokens: Option<usize>,
        ignore_eos: Option<bool>,
        top_k: Option<isize>,
        top_p: Option<f32>,
        session_id: Option<String>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            ignore_eos: ignore_eos.unwrap_or(false),
            top_k,
            top_p,
            session_id,
            frequency_penalty,
            presence_penalty,
        }
    }

    pub fn new_with_max_tokens(max_tokens: usize) -> Self {
        Self {
            temperature: None,
            max_tokens: Some(max_tokens),
            ignore_eos: false,
            top_k: None,
            top_p: None,
            session_id: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: Some(16384),
            ignore_eos: false,
            top_k: None,
            top_p: None,
            session_id: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ModelType {
    Qwen3,
    Qwen3MoE,
    LLaMa,
    Gemma,
    Phi,
    Mistral,
    GLM4,
    Yi,
    StableLM,
    DeepSeek,
    Mistral3VL,
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Randomness of sampling.
    /// rec. default = 1
    pub temperature: Option<f32>,
    /// Cumulative prob of the top tokens to consider, must be in (0, 1]. Set 1 to consider all toks.  
    /// rec. default = 1    
    pub top_p: Option<f32>,
    /// Control the number of top tokens to consider, set -1 to consider all.
    /// rec. default = -1
    pub top_k: Option<isize>,

    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,

    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<EosTokenId>,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct QuantConfig {
    pub quant_method: String,
    #[serde(default)]
    pub bits: usize,
    #[serde(default)]
    pub group_size: i32,
    pub sym: Option<bool>,
    pub desc_act: Option<bool>,
    pub checkpoint_format: Option<String>,
}
