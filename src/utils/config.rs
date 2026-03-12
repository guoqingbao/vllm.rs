// src/utils/config.rs
use crate::transfer::PdConfig;
#[cfg(not(feature = "python"))]
use crate::utils::guidance::ReasoningEffort;
use llguidance::api::TopLevelGrammar;
#[cfg(feature = "python")]
use pyo3::pyclass;
use serde::de::Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(not(feature = "python"))]
impl SamplingParams {
    /// Convert grammar to constraint for GuidanceState construction
    /// Prioritizes constraint field, falls back to grammar field
    pub fn to_constraint(&self) -> Option<TopLevelGrammar> {
        self.grammar.clone()
    }
}

#[cfg(feature = "python")]
impl SamplingParams {
    /// Convert grammar to constraint for GuidanceState construction
    pub fn to_constraint(&self) -> Option<TopLevelGrammar> {
        self.grammar.clone()
    }
}

// EosTokenId enum has been replaced with direct Vec<u32> for simplicity

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoEConfig {
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(alias = "n_routed_experts")]
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: Option<f64>,
    pub first_k_dense_replace: Option<usize>,
    pub n_shared_experts: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RopeScalingValue {
    Bool(bool),
    Number(f64),
    NumberArray(Vec<f64>),
    String(String),
}

impl RopeScalingValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            RopeScalingValue::Number(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            RopeScalingValue::String(v) => Some(v),
            _ => None,
        }
    }
}

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
    #[serde(default, alias = "ffn_hidden_size", alias = "feed_forward_length")]
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: Option<usize>,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    pub qkv_bias: Option<bool>,
    pub attn_output_gate: Option<bool>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<usize>,
    #[serde(deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Option<Vec<u32>>,
    pub use_sliding_window: Option<bool>,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub partial_rotary_factor: Option<f32>,
    #[serde(alias = "hidden_activation")]
    pub hidden_act: candle_nn::Activation,
    #[serde(alias = "rope_parameters")]
    pub rope_scaling: Option<HashMap<String, RopeScalingValue>>,
    pub quant: Option<String>,
    pub moe_cfg: Option<MoEConfig>,
    pub fp8_kvcache: Option<bool>,
    pub quantization_config: Option<QuantConfig>,
    pub is_multi_model: Option<bool>,
    pub extra_config_json: Option<String>,
}

impl Config {
    pub fn apply_generation_cfg(&mut self, generation_cfg: Option<&GenerationConfig>) {
        let Some(gcfg) = generation_cfg else { return };

        // BOS merge (fill if missing; config wins)
        if self.bos_token_id.is_none() {
            self.bos_token_id = gcfg.bos_token_id;
        }

        // EOS merge (combine if both present)
        self.eos_token_id = match (self.eos_token_id.take(), gcfg.eos_token_id.as_ref()) {
            (None, None) => None,
            (None, Some(e)) => Some(e.clone()),
            (Some(e), None) => Some(e),
            (Some(e), Some(other)) => {
                let mut merged = e.clone();
                merged.extend(other.clone());
                Some(merged)
            }
        };
    }
}

// Custom deserializer for eos_token_id to handle both integer and array formats
fn deserialize_eos_token_id<'de, D>(deserializer: D) -> Result<Option<Vec<u32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        Some(serde_json::Value::Number(n)) => {
            if let Some(id) = n.as_u64() {
                Ok(Some(vec![id as u32]))
            } else {
                Err(serde::de::Error::custom(
                    "eos_token_id must be a positive integer",
                ))
            }
        }
        Some(serde_json::Value::Array(arr)) => {
            let ids: Result<Vec<u32>, D::Error> = arr
                .into_iter()
                .map(|v| {
                    if let Some(id) = v.as_u64() {
                        Ok(id as u32)
                    } else {
                        Err(D::Error::custom(
                            "eos_token_id array must contain only unsigned integers",
                        ))
                    }
                })
                .collect();
            Ok(Some(ids?))
        }
        Some(serde_json::Value::Null) => Ok(None),
        Some(v) => Err(serde::de::Error::custom(format!(
            "Expected integer or array for eos_token_id, got {:?}",
            v
        ))),
        None => Ok(None),
    }
}

#[cfg(not(feature = "python"))]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EngineConfig {
    pub model_id: Option<String>,
    pub weight_path: Option<String>,
    pub weight_file: Option<String>,
    pub enforce_parser: Option<String>,
    pub hf_token: Option<String>,
    pub hf_token_path: Option<String>,
    pub num_blocks: usize,
    pub kv_fraction: Option<f32>, // After loading the model, the remaining percent of gpu used for kvcache
    pub mamba_fraction: Option<f32>, // Percent of cache budget reserved for hybrid mamba states
    pub cpu_mem_fold: Option<f32>, // the percentage of gpu kvcache: 0.1x to 10x, default 1.0x
    pub kvcache_memory_bytes: usize,
    #[serde(default)]
    pub mamba_memory_bytes: usize,
    #[serde(default)]
    pub mamba_slot_bytes: usize,
    #[serde(default)]
    pub mamba_cache_capacity: Option<usize>,
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
    pub prefix_cache: Option<bool>,
    pub prefix_cache_max_tokens: Option<usize>,
    pub fp8_kvcache: Option<bool>,
    pub server_mode: Option<bool>,
    pub pd_config: Option<PdConfig>,
    pub mcp_command: Option<String>,
    pub mcp_config: Option<String>,
    pub mcp_args: Option<Vec<String>>,
    pub tool_prompt_template: Option<String>,
    pub pd_server_prefix_cache_ratio: Option<f32>,
    pub pd_client_prefix_cache_ratio: Option<f32>,
    /// Allow client-submitted constraints via HTTP API
    pub allow_constraint_api: bool,
    /// Whether to automatically build LLG grammar from tools
    pub enable_tool_grammar: bool,
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
    pub enforce_parser: Option<String>,
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
    #[pyo3(get, set)]
    pub mamba_fraction: Option<f32>,
    pub block_size: usize,
    pub kvcache_memory_bytes: usize,
    #[serde(default)]
    pub mamba_memory_bytes: usize,
    #[serde(default)]
    pub mamba_slot_bytes: usize,
    #[pyo3(get, set)]
    pub mamba_cache_capacity: Option<usize>,
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
    pub prefix_cache: Option<bool>,
    #[pyo3(get, set)]
    pub prefix_cache_max_tokens: Option<usize>,
    #[pyo3(get, set)]
    pub fp8_kvcache: Option<bool>,
    #[pyo3(get, set)]
    pub server_mode: Option<bool>,
    #[pyo3(get, set)]
    pub pd_config: Option<PdConfig>,
    #[pyo3(get, set)]
    pub mcp_command: Option<String>,
    #[pyo3(get, set)]
    pub mcp_config: Option<String>,
    #[pyo3(get, set)]
    pub mcp_args: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub tool_prompt_template: Option<String>,
    #[pyo3(get, set)]
    pub pd_server_prefix_cache_ratio: Option<f32>,
    #[pyo3(get, set)]
    pub pd_client_prefix_cache_ratio: Option<f32>,
    #[pyo3(get, set)]
    pub allow_constraint_api: bool,
    #[pyo3(get, set)]
    pub enable_tool_grammar: bool,
}

#[cfg(not(feature = "python"))]
impl EngineConfig {
    pub fn new(
        model_id: Option<String>,
        weight_path: Option<String>,
        weight_file: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
        enforce_parser: Option<String>,
        max_num_seqs: Option<usize>,
        config_model_len: Option<usize>,
        max_model_len: Option<usize>,
        max_tokens: Option<usize>,
        isq: Option<String>,
        num_shards: Option<usize>,
        device_ids: Option<Vec<usize>>,
        generation_cfg: Option<GenerationConfig>,
        seed: Option<u64>,
        prefix_cache: Option<bool>,
        prefix_cache_max_tokens: Option<usize>,
        fp8_kvcache: Option<bool>,
        server_mode: Option<bool>,
        cpu_mem_fold: Option<f32>,
        kv_fraction: Option<f32>,
        mamba_fraction: Option<f32>,
        pd_config: Option<PdConfig>,
        mcp_command: Option<String>,
        mcp_config: Option<String>,
        mcp_args: Option<Vec<String>>,
        tool_prompt_template: Option<String>,
        pd_server_prefix_cache_ratio: Option<f32>,
        pd_client_prefix_cache_ratio: Option<f32>,
        allow_constraint_api: bool,
        enable_tool_grammar: bool,
    ) -> Self {
        let mut device_ids = device_ids.unwrap_or_default();
        if device_ids.is_empty() {
            device_ids.push(0);
        }

        if prefix_cache.unwrap_or(false)
            && fp8_kvcache.unwrap_or(false)
            && (cfg!(feature = "flashinfer") || cfg!(feature = "flashattn"))
        {
            panic!("Error: prefix-cache and fp8 kvcache are not compatible under the current settings!\n\t***Tips: use only one of the two features (`--fp8-kvcache` or `--prefix-cache`).");
        }

        Self {
            model_id,
            weight_path,
            weight_file,
            hf_token,
            hf_token_path,
            enforce_parser,
            num_blocks: 128, //placeholder
            cpu_mem_fold,
            kv_fraction,
            mamba_fraction,
            kvcache_memory_bytes: 0, //placeholder
            mamba_memory_bytes: 0,
            mamba_slot_bytes: 0,
            mamba_cache_capacity: None,
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
            prefix_cache,
            prefix_cache_max_tokens,
            fp8_kvcache,
            server_mode,
            pd_config,
            mcp_command,
            mcp_config,
            mcp_args,
            tool_prompt_template,
            pd_server_prefix_cache_ratio,
            pd_client_prefix_cache_ratio,
            allow_constraint_api,
            enable_tool_grammar,
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
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    // stop_token_ids removed - use SpecialTokens for stop detection
    #[serde(alias = "enable_thinking")]
    pub thinking: Option<bool>, // enable reasoning
    /// Tool mode for tool call handling.
    /// If Some(true), external tools are enabled and stream finishes at </tool_call>.
    #[serde(default)]
    pub mcp_mode: Option<bool>,
    /// Grammar constraint as TopLevelGrammar for RPC serialization
    #[serde(default)]
    pub grammar: Option<TopLevelGrammar>,
    /// Reasoning effort level for OpenAI-compatible reasoning API
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
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
    #[pyo3(get, set)]
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    // stop_token_ids removed - use SpecialTokens for stop detection
    /// Tool mode for tool call handling.
    /// If Some(true), external tools are enabled and stream finishes at </tool_call>.
    #[pyo3(get, set)]
    pub mcp_mode: Option<bool>,
    #[pyo3(get, set)]
    pub thinking: Option<bool>,
    /// Grammar constraint as TopLevelGrammar for RPC serialization
    #[serde(default)]
    pub grammar: Option<TopLevelGrammar>,
    /// Grammar constraint as JSON string for Python API
    #[pyo3(get, set)]
    pub grammar_json: Option<String>,
    /// Reasoning effort level for OpenAI-compatible reasoning API
    #[pyo3(get, set)]
    pub reasoning_effort: Option<String>,
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
        thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
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
            mcp_mode: None,
            stop_sequences: None,
            grammar: None,
            thinking,
            reasoning_effort,
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
            mcp_mode: None,
            stop_sequences: None,
            grammar: None,
            thinking: None,
            reasoning_effort: None,
        }
    }
}

#[cfg(not(feature = "python"))]
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
            mcp_mode: None,
            stop_sequences: None,
            grammar: None,
            thinking: None,
            reasoning_effort: None,
        }
    }
}

#[cfg(feature = "python")]
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
            mcp_mode: None,
            stop_sequences: None,
            thinking: None,
            grammar: None,
            grammar_json: None,
            reasoning_effort: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ModelType {
    Qwen3,
    Qwen3MoE,
    Qwen3_5,
    Qwen3_5MoE,
    LLaMa,
    Gemma,
    Gemma3,
    Phi,
    Phi4,
    Mistral,
    GLM4,
    GLM4MoE,
    Yi,
    StableLM,
    DeepSeek,
    Mistral3VL,
    Qwen3VL,
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
    pub eos_token_id: Option<Vec<u32>>,
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
    pub fmt: Option<String>,
    pub weight_block_size: Option<Vec<usize>>,
    #[serde(default, alias = "ignore")]
    pub modules_to_not_convert: Vec<String>,
}
