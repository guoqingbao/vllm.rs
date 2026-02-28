pub use crate::models::qwen3_vl::config::VisionConfig;
use crate::{
    serde_default,
    utils::config::{Config, QuantConfig},
};

serde_default!(Vec<String>, default_architectures, Vec::new());

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Glm4VConfig {
    #[serde(default = "default_architectures")]
    pub architectures: Vec<String>,
    pub text_config: Config,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantConfig>,
}
