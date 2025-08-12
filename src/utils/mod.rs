pub mod chat_template;
pub mod config;
pub mod gguf_helper;
pub mod gguf_varbuilder;
#[cfg(all(feature = "cuda", feature = "graph"))]
pub mod graph;
pub mod progress;
use crate::utils::config::MoEConfig;
use crate::utils::gguf_helper::{get_gguf_info, GGUFInfo};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result};
use config::{Config, EngineConfig, EosTokenId, TokenizerConfig};
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
    let architecture = md_get("general.architecture")?.to_string()?;

    let head_count =
        md_get(format!("{}.attention.head_count", architecture).as_str())?.to_u32()? as usize;
    let head_count_kv =
        md_get(format!("{}.attention.head_count_kv", architecture).as_str())?.to_u32()? as usize;

    let head_dim = md_get(format!("{}.attention.key_length", architecture).as_str());
    let head_dim = if head_dim.is_ok() {
        Some(head_dim.unwrap().to_u32()? as usize)
    } else {
        None
    };
    let embedding_length =
        md_get(format!("{}.embedding_length", architecture).as_str())?.to_u32()? as usize;
    let feed_forward_length =
        md_get(format!("{}.feed_forward_length", architecture).as_str())?.to_u32()? as usize;
    let context_length =
        md_get(format!("{}.context_length", architecture).as_str())?.to_u32()? as usize;
    let block_count = md_get(format!("{}.block_count", architecture).as_str())?.to_u32()? as usize;
    let rms_norm_eps =
        md_get(format!("{}.attention.layer_norm_rms_epsilon", architecture).as_str())?.to_f32()?
            as f64;
    let rope_freq_base = md_get(format!("{}.rope.freq_base", architecture).as_str())
        .and_then(|m| m.to_f32())
        .unwrap_or(10000f32);
    let vocab_size = md_get(format!("{}.vocab_size", architecture).as_str());

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

    let head_dim = head_dim.unwrap_or(embedding_length / head_count);

    let has_output_weight = ct.tensor(reader, "output.weight", &Device::Cpu).is_ok();

    //no partial rotary factor info in gguf file
    let partial_rot_arch_map: HashMap<String, bool> =
        [("glm4".to_string(), true)].iter().cloned().collect();

    let arch = architecture.to_string();

    let mod_cfg = if arch == "qwen3moe" {
        Some(MoEConfig {
            moe_intermediate_size: md_get(
                format!("{}.expert_feed_forward_length", architecture).as_str(),
            )?
            .to_u32()? as usize,
            num_experts: Some(
                md_get(format!("{}.expert_count", architecture).as_str())?.to_u32()? as usize,
            ),
            mlp_only_layers: Some(vec![]),
            decoder_sparse_step: Some(1),
            norm_topk_prob: true,
            num_experts_per_tok: md_get(format!("{}.expert_used_count", architecture).as_str())?
                .to_u32()? as usize,
        })
    } else {
        None
    };

    let extra_loading_len = ct.tensor_infos.keys().len();

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
        extra_loading_len: Some(extra_loading_len),
        partial_rotary_factor: if partial_rot_arch_map.contains_key(&arch) {
            Some(0.5f32)
        } else {
            None
        },
        hidden_act: candle_nn::Activation::Silu,
        quant: None,
        moe_cfg: mod_cfg,
    };

    Ok(cfg)
}

pub fn init_config_tokenizer(
    econfig: &EngineConfig,
) -> Result<(Config, TokenizerConfig, Tokenizer)> {
    let config_path = format!("{}/config.json", econfig.model_path);
    if Path::new(&config_path).exists() {
        let mut config: Config =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(candle_core::Error::wrap)?)
                .map_err(candle_core::Error::wrap)?;
        if config.architectures[0] == "Qwen3MoeForCausalLM" {
            let moe_cfg: MoEConfig = serde_json::from_slice(
                &std::fs::read(&config_path).map_err(candle_core::Error::wrap)?,
            )
            .map_err(candle_core::Error::wrap)?;
            config.moe_cfg = Some(moe_cfg);
        }
        config.quant = econfig.isq.clone();
        let tokenizer_config_path = format!("{}/tokenizer_config.json", econfig.model_path);
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
        let tokenizer_file = if econfig.tokenizer.is_some() {
            econfig.tokenizer.clone().unwrap()
        } else {
            econfig.model_path.clone() + "/tokenizer.json"
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(candle_core::Error::wrap)?;
        Ok((config, config_tokenizer, tokenizer))
    } else if Path::new(&econfig.model_path).exists() {
        assert!(econfig.isq.is_none(), "GGUF model does not support ISQ!");
        let GGUFInfo {
            tokenizer,
            bos,
            eos,
            unk: _,
            context_length,
            chat_template,
        } = {
            let file = std::fs::File::open(&econfig.model_path.clone()).unwrap();
            let mut readers = vec![file];
            let mut readers = readers.iter_mut().collect::<Vec<_>>();
            let content = crate::utils::gguf_helper::Content::from_readers(&mut readers).unwrap();
            get_gguf_info(&content).map_err(candle_core::Error::wrap)?
        };

        let config = {
            let mut file = std::fs::File::open(econfig.model_path.clone()).unwrap();
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

        Ok((config, config_tokenizer, tokenizer))
    } else {
        candle_core::bail!("No valid config found");
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
