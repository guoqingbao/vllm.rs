pub mod chat_template;
pub mod config;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result};
use std::path::Path;

pub fn hub_load_local_safetensors(
    path: &String,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    tracing::info!("{:}", Path::new(path).join(json_file).display());
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
            tracing::warn!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            tracing::warn!(
                "Running on CPU, to run on GPU, build this example with `--features cuda`"
            );
        }
        Ok(Device::Cpu)
    }
}

pub fn get_kvcache_blocks(
    kvcache_mem_gpu: usize,
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

    let num_gpu_blocks = kvcache_mem_gpu * SIZE_IN_MB
        / dsize
        / block_size
        / (config.num_key_value_heads / num_shards)
        / head_dim
        / config.num_hidden_layers
        / 2;
    num_gpu_blocks
}
