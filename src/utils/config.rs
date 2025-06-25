// src/utils/config.rs
use either::Either;
use serde::Deserialize;

//   "hidden_act": "silu",
//   "max_window_layers": 36,
//   "rope_scaling": null,

#[derive(Deserialize, Debug, Clone)]
pub struct EosToken(
    #[serde(with = "either::serde_untagged")] pub Either<Option<u32>, Option<Vec<u32>>>,
);

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub architectures: Vec<String>,
    pub head_dim: Option<usize>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub max_model_len: Option<usize>,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub attention_bias: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: EosToken,
    pub use_sliding_window: Option<bool>,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub hidden_act: candle_nn::Activation,
    pub quant: Option<String>,
}

#[derive(Clone)]
pub struct EngineConfig {
    pub model_path: String,
    pub tokenizer: Option<String>,
    pub tokenizer_config: Option<String>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub temperature: f32,
    pub max_model_len: usize,
    pub quant: Option<String>,
    pub num_shards: Option<usize>,
    pub kvcache_mem_gpu: Option<usize>,
    pub device_id: Option<usize>,
}

impl EngineConfig {
    pub fn new(
        model_path: String,
        block_size: usize,
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
        temperature: f32,
        quant: Option<String>,
        num_shards: Option<usize>,
        kvcache_mem_gpu: Option<usize>,
        device_ids: Option<Vec<usize>>,
    ) -> Self {
        let mut device_ids = device_ids.unwrap_or_default();
        if device_ids.is_empty() {
            device_ids.push(0);
        }
        Self {
            model_path,
            tokenizer: None,
            tokenizer_config: None,
            num_blocks: 128, //placeholder
            block_size,
            max_num_seqs,
            max_num_batched_tokens,
            temperature,
            max_model_len: 32768, //placeholder
            quant,
            num_shards,
            kvcache_mem_gpu,
            device_id: Some(device_ids[0]),
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

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl SamplingParams {
    pub fn new(
        temperature: f32,
        max_tokens: usize,
        ignore_eos: bool,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            ignore_eos,
            top_k,
            top_p,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_tokens: 100,
            ignore_eos: false,
            top_k: None,
            top_p: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ModelType {
    Qwen3,
    LLaMa,
    Gemma,
    Phi,
    Mistral,
    GLM4,
    Yi,
    StableLM,
    DeepSeek,
}
