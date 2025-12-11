use crate::utils::config::EosTokenId;
use crate::utils::config::QuantConfig;
use crate::utils::config::RopeScaling;
use candle_nn::Activation;
use std::collections::HashMap;
#[doc(hidden)]
#[macro_export]
macro_rules! serde_default {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

serde_default!(usize, hidden_size, 768);
serde_default!(usize, intermediate_size, 3072);
serde_default!(usize, num_hidden_layers, 12);
serde_default!(usize, num_attention_heads_vision, 12);
serde_default!(usize, num_channels, 3);
serde_default!(usize, image_size, 224);
serde_default!(usize, patch_size, 16);
serde_default!(Activation, hidden_act, Activation::GeluPytorchTanh);
serde_default!(f64, layer_norm_eps, 1e-6);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    #[serde(default = "hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "num_attention_heads_vision")]
    pub num_attention_heads: usize,
    #[serde(default = "num_channels")]
    pub num_channels: usize,
    #[serde(default = "image_size")]
    pub image_size: usize,
    #[serde(default = "patch_size")]
    pub patch_size: usize,
    #[serde(default = "hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "layer_norm_eps")]
    pub layer_norm_eps: f64,
}

serde_default!(bool, attention_bias, false);
serde_default!(usize, head_dim, 256);
serde_default!(Activation, hidden_activation, Activation::GeluPytorchTanh);
serde_default!(f64, rms_norm_eps, 1e-6);
serde_default!(f64, rope_theta, 1000000.);
serde_default!(usize, vocab_size, 262208);
serde_default!(bool, tie_word_embeddings, true);
serde_default!(usize, query_pre_attn_scalar, 256);
serde_default!(usize, max_position_embeddings, 131072);
serde_default!(f64, rope_local_base_freq, 10000.);
serde_default!(usize, sliding_window_pattern, 6);
serde_default!(usize, num_attention_heads_text, 8);
serde_default!(usize, num_key_value_heads, 4);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextConfig {
    #[serde(default = "attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "head_dim")]
    pub head_dim: usize,
    #[serde(default = "hidden_activation")]
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "num_attention_heads_text")]
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "vocab_size")]
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default = "max_position_embeddings")]
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "rope_local_base_freq")]
    pub rope_local_base_freq: f64,
    #[serde(default = "sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub rope_scaling: Option<HashMap<String, RopeScaling>>,
    pub quant: Option<String>,
}

fn has_vision() -> bool {
    true
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_index: usize,
    pub mm_tokens_per_image: usize,
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default = "has_vision")]
    pub has_vision: bool,
}
