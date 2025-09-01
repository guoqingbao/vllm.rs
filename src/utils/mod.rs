pub mod chat_template;
pub mod config;
pub mod downloader;
pub mod gguf_helper;
pub mod gguf_varbuilder;
#[cfg(all(feature = "cuda", feature = "graph"))]
pub mod graph;
pub mod logits_processor;
pub mod progress;
use crate::utils::config::MoEConfig;
use crate::utils::config::ModelType;
use crate::utils::config::{RopeScaling, ScalingValue};
use crate::utils::downloader::ModelPaths;
use crate::utils::gguf_helper::{get_gguf_info, GGUFInfo};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result};
use config::{Config, EngineConfig, EosTokenId, GenerationConfig, TokenizerConfig};
use either::Either;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub fn hub_load_local_safetensors(path: &String, json_file: &str) -> Result<Vec<PathBuf>> {
    crate::log_info!("{:}", Path::new(path).join(json_file).display());
    let jsfile = std::fs::File::open(Path::new(path).join(json_file))?;
    let json: serde_json::Value =
        serde_json::from_reader(&jsfile).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files
        .into_iter()
        .map(|v| Path::new(path).join(v))
        .collect();
    Ok(safetensors_files)
}

pub fn new_device(ordinal: usize) -> Result<Device> {
    if cuda_is_available() {
        use candle_core::CudaDevice;
        let device = Device::Cuda(CudaDevice::new_with_stream(ordinal).unwrap());
        Ok(device)
    } else if metal_is_available() {
        Ok(Device::new_metal(ordinal)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            crate::log_info!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            crate::log_info!(
                "Running on CPU, to run on GPU, build this example with `--features cuda`"
            );
        }
        Ok(Device::Cpu)
    }
}

pub fn get_kvcache_blocks(
    max_num_seqs: usize,
    max_model_len: usize,
    block_size: usize,
    config: &config::Config,
    num_shards: usize,
    dtype: DType,
) -> usize {
    const SIZE_IN_MB: usize = 1024 * 1024;

    let dsize = dtype.size_in_bytes();
    let head_dim = config
        .head_dim
        .unwrap_or(config.hidden_size / config.num_attention_heads);

    let num_gpu_blocks = max_num_seqs * max_model_len / block_size;

    let total_memory_bytes = num_gpu_blocks
        * block_size
        * (config.num_key_value_heads / num_shards)
        * head_dim
        * dsize
        * 2
        * config.num_hidden_layers;
    crate::log_info!(
        "Allocating {} KV blocks ({:2} MB) for [{} seqs x {} tokens]",
        num_gpu_blocks,
        total_memory_bytes as f32 / SIZE_IN_MB as f32,
        max_num_seqs,
        max_model_len,
    );

    num_gpu_blocks
}

pub fn config_from_gguf<R: std::io::Seek + std::io::Read>(
    ct: &candle_core::quantized::gguf_file::Content,
    reader: &mut R,
) -> Result<Config> {
    let md_get = |s: &str| match ct.metadata.get(s) {
        None => candle_core::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };
    let arch = md_get("general.architecture")?.to_string()?;

    let head_count = md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
    let head_count_kv =
        md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;

    let head_dim = md_get(format!("{arch}.attention.key_length").as_str());
    let head_dim = if head_dim.is_ok() {
        Some(head_dim.unwrap().to_u32()? as usize)
    } else {
        None
    };
    let embedding_length = md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
    let feed_forward_length =
        md_get(format!("{arch}.feed_forward_length").as_str())?.to_u32()? as usize;
    let context_length = md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
    let block_count = md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
    let rms_norm_eps =
        md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
    let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
        .and_then(|m| m.to_f32())
        .unwrap_or(10000f32);
    let vocab_size = md_get(format!("{arch}.vocab_size").as_str());

    let vocab_size = if vocab_size.is_ok() {
        Some(vocab_size.unwrap().to_u32()? as usize)
    } else {
        let vocab_size = md_get("tokenizer.ggml.tokens");
        if vocab_size.is_ok() {
            let size = vocab_size.unwrap().to_vec()?.len();
            crate::log_info!(
                "No vocab_size in metadata, using tokenizer.ggml.tokens with size {}",
                size
            );
            Some(size)
        } else {
            None
        }
    };

    let bos_token_id = md_get("tokenizer.ggml.bos_token_id");

    let bos_token_id = if bos_token_id.is_ok() {
        Some(bos_token_id.unwrap().to_u32()? as usize)
    } else {
        None
    };

    let eos_token_id = md_get("tokenizer.ggml.eos_token_id");

    let eos_token_id = if eos_token_id.is_ok() {
        EosTokenId::Single(eos_token_id.unwrap().to_u32()?)
    } else {
        EosTokenId::Multiple(vec![])
    };

    let rope_scaling_type = md_get(format!("{arch}.rope.scaling.type").as_str());
    let rope_scaling = if rope_scaling_type.is_ok() {
        let scaling_type = rope_scaling_type.unwrap().to_string()?;
        crate::log_info!("Rope scaling type: {}", scaling_type);
        let mut scaling_map = HashMap::<String, RopeScaling>::new();
        let scaling_factor = md_get(format!("{arch}.rope.scaling.alpha").as_str());
        if scaling_factor.is_ok() {
            scaling_map.insert(
                "alpha".to_string(),
                RopeScaling(Either::Left(ScalingValue(Either::Left(
                    scaling_factor.unwrap().to_f32()? as f64,
                )))),
            );
        } else {
            let factor = md_get(format!("{arch}.rope.scaling.factor").as_str());
            if factor.is_ok() {
                scaling_map.insert(
                    "factor".to_string(),
                    RopeScaling(Either::Left(ScalingValue(Either::Left(
                        factor.unwrap().to_f32()? as f64,
                    )))),
                );
            }
        };
        let original_max_position_embeddings =
            md_get(format!("{arch}.rope.scaling.original_context_length").as_str());
        if original_max_position_embeddings.is_ok() {
            scaling_map.insert(
                "original_max_position_embeddings".to_string(),
                RopeScaling(Either::Left(ScalingValue(Either::Left(
                    original_max_position_embeddings.unwrap().to_u32()? as f64,
                )))),
            );
        }

        if scaling_type == "llama3" {
            let low_freq_factor = md_get(format!("{arch}.rope.scaling.low_freq_factor").as_str());
            if low_freq_factor.is_ok() {
                scaling_map.insert(
                    "low_freq_factor".to_string(),
                    RopeScaling(Either::Left(ScalingValue(Either::Left(
                        low_freq_factor.unwrap().to_f32()? as f64,
                    )))),
                );
            }
            let high_freq_factor = md_get(format!("{arch}.rope.scaling.high_freq_factor").as_str());
            if high_freq_factor.is_ok() {
                scaling_map.insert(
                    "high_freq_factor".to_string(),
                    RopeScaling(Either::Left(ScalingValue(Either::Left(
                        high_freq_factor.unwrap().to_f32()? as f64,
                    )))),
                );
            }
        }

        if scaling_type == "yarn" {
            let extrapolation_factor =
                md_get(format!("{arch}.rope.scaling.extrapolation_factor").as_str());
            if extrapolation_factor.is_ok() {
                scaling_map.insert(
                    "extrapolation_factor".to_string(),
                    RopeScaling(Either::Left(ScalingValue(Either::Left(
                        extrapolation_factor.unwrap().to_f32()? as f64,
                    )))),
                );
            }

            let attn_factor = md_get(format!("{arch}.rope.scaling.attn_factor").as_str());
            if attn_factor.is_ok() {
                scaling_map.insert(
                    "attn_factor".to_string(),
                    RopeScaling(Either::Left(ScalingValue(Either::Left(
                        attn_factor.unwrap().to_f32()? as f64,
                    )))),
                );
            }

            let beta_fast = md_get(format!("{arch}.rope.scaling.beta_fast").as_str());
            if beta_fast.is_ok() {
                scaling_map.insert(
                    "beta_fast".to_string(),
                    RopeScaling(Either::Left(ScalingValue(Either::Left(
                        beta_fast.unwrap().to_f32()? as f64,
                    )))),
                );
            }

            let beta_slow = md_get(format!("{arch}.rope.scaling.beta_slow").as_str());
            if beta_slow.is_ok() {
                scaling_map.insert(
                    "beta_slow".to_string(),
                    RopeScaling(Either::Left(ScalingValue(Either::Left(
                        beta_slow.unwrap().to_f32()? as f64,
                    )))),
                );
            }
        }

        scaling_map.insert(
            "rope_type".to_string(),
            RopeScaling(Either::Right(scaling_type.clone())),
        );
        crate::log_info!("Rope scaling map: {:?}", scaling_map);
        Some(scaling_map)
    } else {
        None
    };

    let head_dim = head_dim.unwrap_or(embedding_length / head_count);

    let has_output_weight = ct.tensor(reader, "output.weight", &Device::Cpu).is_ok();

    let rope_dim = md_get(format!("{arch}.rope.dimension_count").as_str());
    let partial_rotary_factor = if rope_dim.is_ok() {
        let rope_dim = rope_dim.unwrap().to_u32()? as usize;
        if rope_dim != head_dim {
            Some(rope_dim as f32 / head_dim as f32)
        } else {
            None
        }
    } else {
        None
    };

    let mod_cfg = if arch.to_string() == "qwen3moe" || arch.to_string() == "qwen2moe" {
        let shared_expert_intermediate_size = if arch.to_string() == "qwen2moe" {
            Some(
                md_get(format!("{arch}.expert_shared_feed_forward_length").as_str())?.to_u32()?
                    as usize,
            )
        } else {
            None //qwen3 moe has no shared experts
        };
        Some(MoEConfig {
            moe_intermediate_size: md_get(format!("{arch}.expert_feed_forward_length").as_str())?
                .to_u32()? as usize,
            shared_expert_intermediate_size,
            num_experts: Some(md_get(format!("{arch}.expert_count").as_str())?.to_u32()? as usize),
            mlp_only_layers: Some(vec![]),
            decoder_sparse_step: Some(1),
            norm_topk_prob: shared_expert_intermediate_size.is_none(),
            num_experts_per_tok: md_get(format!("{arch}.expert_used_count").as_str())?.to_u32()?
                as usize,
        })
    } else {
        None
    };

    let cfg = Config {
        architectures: vec![arch.clone()],
        head_dim: Some(head_dim),
        num_attention_heads: head_count,
        num_key_value_heads: head_count_kv,
        max_position_embeddings: context_length,
        hidden_size: embedding_length,
        num_hidden_layers: block_count,
        max_model_len: Some(context_length),
        intermediate_size: feed_forward_length,
        rms_norm_eps,
        vocab_size,
        rope_theta: rope_freq_base as f64,
        attention_bias: None,
        tie_word_embeddings: Some(!has_output_weight),
        bos_token_id,
        eos_token_id,
        use_sliding_window: None,
        sliding_window: None,
        max_window_layers: None,
        partial_rotary_factor,
        hidden_act: candle_nn::Activation::Silu,
        rope_scaling,
        quant: None,
        moe_cfg: mod_cfg,
    };

    Ok(cfg)
}

pub fn init_config_tokenizer(
    econfig: &EngineConfig,
) -> Result<(
    ModelPaths,
    bool,
    Config,
    TokenizerConfig,
    Tokenizer,
    Option<GenerationConfig>,
)> {
    let loader = crate::utils::downloader::Downloader::new(
        econfig.model_id.clone(),
        econfig.weight_path.clone(),
        econfig.weight_file.clone(),
    );
    let (model_pathes, is_gguf) =
        loader.prepare_model_weights(econfig.hf_token.clone(), econfig.hf_token_path.clone())?;
    if !is_gguf {
        let config_path = model_pathes.get_config_filename();
        let mut config: Config =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(candle_core::Error::wrap)?)
                .map_err(candle_core::Error::wrap)?;
        if matches!(
            config.architectures[0].as_str(),
            "Qwen2MoeForCausalLM" | "Qwen3MoeForCausalLM"
        ) {
            let moe_cfg: MoEConfig = serde_json::from_slice(
                &std::fs::read(&config_path).map_err(candle_core::Error::wrap)?,
            )
            .map_err(candle_core::Error::wrap)?;
            config.moe_cfg = Some(moe_cfg);
        }
        config.quant = econfig.isq.clone();
        let tokenizer_config_path = model_pathes.get_tokenizer_config_filename();
        let config_tokenizer: TokenizerConfig = {
            match std::fs::read(tokenizer_config_path).map_err(candle_core::Error::wrap) {
                Ok(f) => serde_json::from_slice(&f).map_err(candle_core::Error::wrap)?,
                _ => {
                    crate::log_error!(
                        "Missing tokenizer_config.json file, chat template may not correct!"
                    );
                    TokenizerConfig {
                        model_max_length: None,
                        add_bos_token: None,
                        add_eos_token: None,
                        chat_template: None,
                        bos_token: None,
                        eos_token: None,
                    }
                }
            }
        };
        let tokenizer_file = model_pathes.get_tokenizer_filename();

        let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(candle_core::Error::wrap)?;

        let generation_config_path = model_pathes.get_generation_config_filename();
        let generation_cfg = if generation_config_path.display().to_string() != ""
            && Path::new(&generation_config_path).exists()
        {
            let str_cfg: Option<String> = std::fs::read_to_string(generation_config_path).ok();
            let cfg: GenerationConfig = serde_json::from_str(str_cfg.unwrap().as_str()).unwrap();
            Some(cfg)
        } else {
            None
        };

        Ok((
            model_pathes,
            is_gguf,
            config,
            config_tokenizer,
            tokenizer,
            generation_cfg,
        ))
    } else if !model_pathes.get_weight_filenames().is_empty()
        && model_pathes.get_weight_filenames()[0].exists()
    {
        assert!(econfig.isq.is_none(), "GGUF model does not support ISQ! \n\t***Tips: use `--w` to specify safetensors model path!***");
        let GGUFInfo {
            tokenizer,
            bos,
            eos,
            unk: _,
            context_length,
            chat_template,
        } = {
            let file = std::fs::File::open(&model_pathes.get_weight_filenames()[0]).unwrap();
            let mut readers = vec![file];
            let mut readers = readers.iter_mut().collect::<Vec<_>>();
            let content = crate::utils::gguf_helper::Content::from_readers(&mut readers).unwrap();
            get_gguf_info(&content).map_err(candle_core::Error::wrap)?
        };

        let config = {
            let mut file = std::fs::File::open(&model_pathes.get_weight_filenames()[0]).unwrap();
            let content = candle_core::quantized::gguf_file::Content::read(&mut file).unwrap();
            config_from_gguf(&content, &mut file)?
        };
        let config_tokenizer = TokenizerConfig {
            model_max_length: Some(context_length.unwrap_or(config.max_position_embeddings) as f64),
            add_bos_token: Some(bos.is_some()),
            add_eos_token: Some(eos.is_some()),
            chat_template: chat_template.clone(),
            bos_token: bos,
            eos_token: eos,
        };

        Ok((
            model_pathes,
            is_gguf,
            config,
            config_tokenizer,
            tokenizer,
            None,
        ))
    } else {
        candle_core::bail!("Model file(s) not found!");
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModule;

pub fn get_runner_path() -> Result<PathBuf> {
    #[cfg(feature = "python")]
    {
        Python::with_gil(|py| {
            let module = PyModule::import(py, "vllm_rs").map_err(candle_core::Error::wrap)?;
            let file: String = module
                .getattr("__file__")
                .map_err(candle_core::Error::wrap)?
                .extract()
                .map_err(candle_core::Error::wrap)?;
            let module_path = Path::new(&file).parent().unwrap().join("runner");
            Ok(module_path)
        })
    }

    #[cfg(not(feature = "python"))]
    {
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path
            .parent()
            .ok_or("Failed to get exe directory")
            .map_err(candle_core::Error::wrap)?;
        Ok(exe_dir.join("runner"))
    }
}

pub fn spawn_runner(
    #[cfg(feature = "python")] py: Python,
    runner_path: &str,
    sock_name: &str,
) -> Result<()> {
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::{PyDict, PyString, PyTuple};
        crate::log_info!("Spawning runner at: {}", runner_path);
        let subprocess = py.import("subprocess").map_err(candle_core::Error::wrap)?;

        let args = PyTuple::new(
            py,
            &[
                PyString::new(py, runner_path),
                PyString::new(py, "--sock"),
                PyString::new(py, sock_name),
            ],
        )
        .map_err(candle_core::Error::wrap)?;

        let kwargs = PyDict::new(py);
        kwargs
            .set_item("shell", false)
            .map_err(candle_core::Error::wrap)?;
        kwargs
            .set_item("text", true)
            .map_err(candle_core::Error::wrap)?;

        let result = subprocess
            .call_method("Popen", (args,), Some(&kwargs))
            .map_err(candle_core::Error::wrap)?;
        crate::log_info!("Runner spawned {:?}", result);
        Ok(())
    }
    #[cfg(not(feature = "python"))]
    {
        use std::process::Command;

        Command::new(runner_path)
            .arg("--sock")
            .arg(sock_name)
            .spawn()
            .map_err(|e| e.into())
            .map(|_child| ())
    }
}

pub fn get_arch_rope(
    tokenizer: &Tokenizer,
    generation_cfg: &mut Option<GenerationConfig>,
    architectures: String,
) -> Result<(ModelType, String, bool)> {
    let rope_key_map: HashMap<&str, bool> = [
        ("Qwen2ForCausalLM", false),
        ("Qwen3ForCausalLM", false),
        ("MistralForCausalLM", false),
        ("LlamaForCausalLM", false),
        ("Glm4ForCausalLM", true),
        ("glm4", true),
        ("qwen2", false),
        ("qwen3", false),
        ("llama", true),
        ("mistral", true),
    ]
    .iter()
    .cloned()
    .collect();

    let arch = architectures.as_str();
    let (model_type, default_chat_template) = match arch {
        "Qwen2ForCausalLM"
        | "Qwen2ForConditionalGeneration"
        | "Qwen3ForCausalLM"
        | "Qwen3ForConditionalGeneration"
        | "qwen2"
        | "qwen3" => (
            ModelType::Qwen3,
            "<|im_start|>user\n {} <|im_end|>".to_string(),
        ),
        "qwen2moe" | "Qwen2MoeForCausalLM" | "qwen3moe" | "Qwen3MoeForCausalLM" => (
            ModelType::Qwen3MoE,
            "<|im_start|>user\n {} <|im_end|>".to_string(),
        ),
        "LlamaForCausalLM"
        | "MistralForCausalLM"
        | "LlamaForConditionalGeneration"
        | "llama"
        | "mistral"
        | "llama2"
        | "llama3" => {
            if let Some(_) = tokenizer
                .get_vocab(true)
                .get("<|start_header_id|>")
                .copied()
            {
                //llama3
                (
                    ModelType::LLaMa,
                    "<|start_header_id|>user<|end_header_id|>\n\n {} <|eot_id|>".to_string(),
                )
            } else {
                //llama2
                (ModelType::LLaMa, "[INST] {} [/INST]".to_string())
            }
        }
        "Glm4ForCausalLM" | "Glm4ForConditionalGeneration" | "glm4" => {
            if let Some(ref mut gen_cfg) = generation_cfg {
                if gen_cfg.penalty.is_none() {
                    gen_cfg.penalty = Some(1.2); //default repetition penalty for glm4 models
                }
            }
            (
                ModelType::GLM4,
                "[gMASK]<sop><|user|>{}<|assistant|>".to_string(),
            )
        }
        _ => candle_core::bail!("Unsupported architecture: {}", architectures),
    };

    let is_rope_i = if rope_key_map.contains_key(arch) {
        rope_key_map[arch]
    } else {
        false
    };
    Ok((model_type, default_chat_template, is_rope_i))
}
