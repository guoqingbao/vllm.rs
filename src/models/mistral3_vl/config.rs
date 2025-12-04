use crate::utils::config::Config;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct VisionConfig {
    // #[serde(default = 1024)]
    pub hidden_size: usize,
    // #[serde(default = 3)]
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    // #[serde(default = 4096)]
    pub intermediate_size: usize,
    // #[serde(default = 24)]
    pub num_hidden_layers: usize,
    pub head_dim: Option<usize>,
    // #[serde(default = 16)]
    pub num_attention_heads: usize,
    // #[serde(default = candle_nn::Activation::Silu)]
    pub hidden_act: candle_nn::Activation,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Mistral3Config {
    pub image_token_index: usize,
    pub multimodal_projector_bias: bool,
    pub projector_hidden_act: candle_nn::Activation,
    pub spatial_merge_size: usize,
    pub vision_feature_layer: isize,
    pub text_config: Config,
    pub vision_config: VisionConfig,
}
