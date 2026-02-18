// src/utils/kvcache_allocator.rs
//!
//! KVCache Allocation Module
//!
//! This module provides a centralized, robust entry point for determining
//! available GPU memory, calculating KVCache blocks, and allocating KV cache tensors.
//!
//! # Usage
//!
//! ```ignore
//! // After model loading, create allocator and plan allocation
//! let allocator = KVCacheAllocator::new(&econfig, &config, dtype);
//! let available_memory = allocator.get_available_memory(&device_ids)?;
//! let allocation = allocator.plan_allocation(available_memory)?;
//! // Allocate tensors
//! let (gpu_cache, cpu_cache) = allocator.init_kv_cache(&allocation, dtype, &device)?;
//! ```

use super::{qwen3_hybrid_layer_types, resolve_qwen3_hybrid_config};
use crate::utils::config::{Config, EngineConfig};
use candle_core::{DType, Device, Result, Tensor};
use std::fmt;

/// Reserved memory constants - used for post-allocation warnings
const CUDA_RESERVED_BYTES: u64 = 512 * 1024 * 1024; // 512 MB recommended minimum remaining memory
const SIZE_IN_MB: f64 = (1024 * 1024) as f64;
const SIZE_IN_GB: f64 = 1024.0 * 1024.0 * 1024.0;
const DEFAULT_HYBRID_MAMBA_FRACTION: f64 = 0.1;
const MAX_HYBRID_MAMBA_FRACTION: f64 = 0.3;

/// Represents the result of KVCache allocation planning
#[derive(Debug, Clone)]
pub struct KVCacheAllocation {
    /// Number of GPU blocks for KVCache
    pub num_gpu_blocks: usize,
    /// Number of CPU blocks for KVCache swap
    pub num_cpu_blocks: usize,
    /// Maximum number of concurrent sequences
    pub max_num_seqs: usize,
    /// Maximum model context length
    pub max_model_len: usize,
    /// Total GPU memory allocated for KVCache in bytes
    pub kvcache_memory_bytes: usize,
    /// Maximum number of batched tokens
    pub max_num_batched_tokens: usize,
}

/// Error types for KVCache allocation
#[derive(Debug, Clone)]
pub enum KVCacheError {
    /// Not enough GPU memory to allocate KVCache
    InsufficientGpuMemory {
        available_mb: f64,
        required_mb: f64,
        reserved_mb: f64,
    },
    /// Invalid configuration parameters
    InvalidConfiguration { message: String },
    /// Platform-specific error (e.g., CUDA/Metal not available)
    PlatformError { message: String },
}

impl fmt::Display for KVCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KVCacheError::InsufficientGpuMemory {
                available_mb,
                required_mb,
                reserved_mb,
            } => {
                write!(
                    f,
                    "Insufficient GPU memory for KVCache allocation.\n\
                     Available: {:.2} MB, Required: {:.2} MB, Reserved: {:.2} MB.\n\
                     Tips: Try reducing --max-model-len or --max-num-seqs, \
                     or free GPU resources.",
                    available_mb, required_mb, reserved_mb
                )
            }
            KVCacheError::InvalidConfiguration { message } => {
                write!(f, "Invalid KVCache configuration: {}", message)
            }
            KVCacheError::PlatformError { message } => {
                write!(f, "Platform error: {}", message)
            }
        }
    }
}

impl std::error::Error for KVCacheError {}

/// Main allocator struct for platform-aware KVCache memory planning
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct KVCacheAllocator {
    // Model parameters
    num_hidden_layers: usize,
    /// Number of layers that need KV cache (excludes GDN/Mamba layers)
    num_kv_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_shards: usize,
    block_size: usize,
    // User constraints (None = auto-decide)
    user_max_model_len: Option<usize>,
    user_max_num_seqs: Option<usize>,
    config_model_len: usize,
    kv_fraction: f64,
    // Flags
    fp8_kvcache: bool,
    cpu_mem_fold: f32,
    dtype_size: usize,
    model_dtype_size: usize,
    hybrid_mamba_slot_bytes: Option<usize>,
    hybrid_num_gdn_layers: usize,
}

impl KVCacheAllocator {
    fn kv_heads_per_shard(&self) -> usize {
        if self.num_shards == 0 {
            // Defensive fallback: constructor normalizes this to >=1.
            return 1;
        }

        if self.num_kv_heads >= self.num_shards {
            self.num_kv_heads / self.num_shards
        } else {
            // For MQA/GQA configs where TP shards exceed KV heads, each shard still stores
            // one replicated KV head, matching runtime attention behavior.
            1
        }
    }

    /// Create a new KVCacheAllocator from engine and model configs
    pub fn new(econfig: &EngineConfig, config: &Config, dtype: DType) -> Self {
        let configured_num_shards = econfig.num_shards.unwrap_or(1);
        if configured_num_shards == 0 {
            crate::log_warn!(
                "EngineConfig.num_shards=0 is invalid; defaulting to 1 for KVCache allocation"
            );
        }
        let num_shards = configured_num_shards.max(1);
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        let fp8_kvcache = econfig.fp8_kvcache.unwrap_or(false);
        let dtype_size = if fp8_kvcache {
            1
        } else {
            dtype.size_in_bytes()
        };
        let model_dtype_size = dtype.size_in_bytes();

        let kv_fraction = econfig
            .kv_fraction
            .unwrap_or(if cfg!(feature = "flash-attn") {
                0.7
            } else {
                0.5
            }) as f64;

        let config_model_len = econfig
            .config_model_len
            .unwrap_or(config.max_position_embeddings);

        // For hybrid models (e.g., Qwen3.5), only count full-attention layers
        let num_kv_layers = if let Some(block_types) = qwen3_hybrid_layer_types(config) {
            block_types
                .iter()
                .filter(|t| t.as_str() == "full_attention")
                .count()
        } else {
            config.num_hidden_layers
        };

        let (hybrid_mamba_slot_bytes, hybrid_num_gdn_layers) = if let Some(block_types) =
            qwen3_hybrid_layer_types(config)
        {
            let num_gdn_layers = block_types
                .iter()
                .filter(|t| t.as_str() == "linear_attention")
                .count();
            if num_gdn_layers == 0 {
                (None, 0)
            } else {
                let hybrid = resolve_qwen3_hybrid_config(config);
                let shard_count = num_shards.max(1);
                if hybrid.num_v_heads % shard_count != 0 || hybrid.num_k_heads % shard_count != 0 {
                    crate::log_warn!(
                            "Hybrid mamba heads are not divisible by num_shards (v_heads={}, k_heads={}, shards={}); memory estimate uses floor division.",
                            hybrid.num_v_heads,
                            hybrid.num_k_heads,
                            shard_count
                        );
                }
                let num_v_heads = std::cmp::max(1, hybrid.num_v_heads / shard_count);
                let num_k_heads = std::cmp::max(1, hybrid.num_k_heads / shard_count);
                let conv_window = hybrid.conv_kernel_size.saturating_sub(1);
                let d_conv = num_k_heads
                    .saturating_mul(hybrid.key_head_dim)
                    .saturating_mul(2)
                    .saturating_add(num_v_heads.saturating_mul(hybrid.value_head_dim));
                let per_layer_conv_bytes = d_conv
                    .saturating_mul(conv_window)
                    .saturating_mul(model_dtype_size);
                let per_layer_recurrent_bytes = num_v_heads
                    .saturating_mul(hybrid.key_head_dim)
                    .saturating_mul(hybrid.value_head_dim)
                    .saturating_mul(DType::F32.size_in_bytes());
                let per_slot_bytes = num_gdn_layers
                    .saturating_mul(per_layer_conv_bytes.saturating_add(per_layer_recurrent_bytes));
                if per_slot_bytes == 0 {
                    (None, num_gdn_layers)
                } else {
                    (Some(per_slot_bytes), num_gdn_layers)
                }
            }
        } else {
            (None, 0)
        };

        Self {
            num_hidden_layers: config.num_hidden_layers,
            num_kv_layers,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_shards,
            block_size: econfig.block_size,
            user_max_model_len: econfig.max_model_len,
            user_max_num_seqs: if econfig.max_model_len.is_some() {
                Some(econfig.max_num_seqs)
            } else {
                None // Auto-decide if max_model_len not specified
            },
            config_model_len,
            kv_fraction: if econfig.max_model_len.is_some() && econfig.kv_fraction.is_none() {
                0.95
            } else {
                kv_fraction
            },
            fp8_kvcache,
            cpu_mem_fold: econfig.cpu_mem_fold.unwrap_or(0.2),
            dtype_size,
            model_dtype_size,
            hybrid_mamba_slot_bytes,
            hybrid_num_gdn_layers,
        }
    }

    pub fn plan(&self, device_ids: &[usize], econfig: &mut EngineConfig) -> Result<()> {
        match self.get_available_memory(device_ids) {
            Ok(min_available) => {
                let mut kv_budget = min_available;
                let mut mamba_budget = 0u64;
                let mut mamba_capacity = 0usize;
                let mut mamba_budget_enabled = false;

                if let Some(slot_bytes) = self.hybrid_mamba_slot_bytes {
                    let requested_fraction = econfig
                        .mamba_fraction
                        .map(|f| f as f64)
                        .unwrap_or(DEFAULT_HYBRID_MAMBA_FRACTION)
                        .clamp(0.0, MAX_HYBRID_MAMBA_FRACTION);
                    if requested_fraction > 0.0 {
                        mamba_budget_enabled = true;

                        let mut target_budget =
                            ((min_available as f64) * requested_fraction) as u64;
                        let min_one_slot = slot_bytes as u64;
                        if target_budget > 0 && target_budget < min_one_slot {
                            crate::log_warn!(
                                "Hybrid mamba budget {:.2} MB is smaller than one slot {:.2} MB; bumping to one slot.",
                                target_budget as f64 / SIZE_IN_MB,
                                min_one_slot as f64 / SIZE_IN_MB
                            );
                            target_budget = min_one_slot;
                        }
                        if target_budget >= min_available {
                            candle_core::bail!(
                                "Hybrid mamba budget ({:.2} GB) leaves no memory for KV cache. Reduce mamba_fraction or max_model_len.",
                                target_budget as f64 / SIZE_IN_GB
                            );
                        }

                        mamba_budget = target_budget;
                        kv_budget = min_available.saturating_sub(mamba_budget);
                        mamba_capacity = if slot_bytes == 0 {
                            0
                        } else {
                            (mamba_budget as usize / slot_bytes).max(1)
                        };
                    }
                }

                match self.plan_allocation(kv_budget, device_ids.len()) {
                    Ok(allocation) => {
                        self.apply_to_config(&allocation, econfig);

                        if let Some(slot_bytes) = self.hybrid_mamba_slot_bytes {
                            if !mamba_budget_enabled {
                                econfig.mamba_slot_bytes = 0;
                                econfig.mamba_memory_bytes = 0;
                                econfig.mamba_cache_capacity = None;
                                return Ok(());
                            }
                            if allocation.max_num_seqs > mamba_capacity {
                                crate::log_warn!(
                                    "Clamping max_num_seqs from {} to {} due to hybrid mamba slot capacity.",
                                    allocation.max_num_seqs,
                                    mamba_capacity
                                );
                                econfig.max_num_seqs = mamba_capacity;
                            }
                            econfig.mamba_slot_bytes = slot_bytes;
                            econfig.mamba_memory_bytes = mamba_budget as usize;
                            econfig.mamba_cache_capacity = Some(mamba_capacity);
                            crate::log_warn!(
                                "Hybrid Mamba Allocation: {} slot(s), {:.2} GB budget, {:.2} MB/slot, {} linear-attention layer(s), model dtype {} bytes",
                                mamba_capacity,
                                mamba_budget as f64 / SIZE_IN_GB,
                                slot_bytes as f64 / SIZE_IN_MB,
                                self.hybrid_num_gdn_layers,
                                self.model_dtype_size
                            );
                        } else {
                            econfig.mamba_slot_bytes = 0;
                            econfig.mamba_memory_bytes = 0;
                            econfig.mamba_cache_capacity = None;
                        }
                        Ok(())
                    }
                    Err(e) => {
                        crate::log_error!("KVCache allocation failed: {}", e);
                        candle_core::bail!("KVCache allocation failed: {}", e)
                    }
                }
            }
            Err(e) => {
                crate::log_error!("Failed to get available memory: {:?}", e);
                Err(e)
            }
        }?;

        Ok(())
    }
    /// Calculate per-block memory size in bytes
    pub fn per_block_bytes(&self) -> usize {
        self.block_size
            * self.kv_heads_per_shard()
            * self.head_dim
            * self.dtype_size
            * 2 // K and V
            * self.num_kv_layers
    }

    /// Calculate required memory for given parameters
    pub fn calculate_required_memory(&self, num_seqs: usize, model_len: usize) -> usize {
        let blocks_per_seq = (model_len + self.block_size - 1) / self.block_size;
        let num_blocks = num_seqs * blocks_per_seq;
        num_blocks * self.per_block_bytes()
    }

    /// Query available GPU memory for a single device (platform-aware)
    /// Returns usable memory AFTER applying kv_fraction (reserved memory check is done post-allocation)
    #[cfg(feature = "cuda")]
    pub fn get_rank_available_memory(&self, device_id: usize) -> Result<u64> {
        use candle_core::backend::BackendDevice;
        use candle_core::cuda_backend::cudarc::driver::sys;
        use candle_core::cuda_backend::CudaDevice;

        // Create a CUDA context for that device
        let _ = CudaDevice::new(device_id)?;

        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;

            sys::lib()
                .cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuMemGetInfo_v2 failed: {e:?}")))?;

            // Apply kv_fraction only - reserved memory check is done post-allocation
            let usable = (free as f64 * self.kv_fraction) as u64;

            crate::log_warn!(
                "GPU {}: total {:.2} GB, free {:.2} GB, kv_fraction {:.0}%, Max usable cache budget {:.2} GB",
                device_id,
                total as f64 / SIZE_IN_GB,
                free as f64 / SIZE_IN_GB,
                self.kv_fraction * 100.0,
                usable as f64 / SIZE_IN_GB
            );

            Ok(usable)
        }
    }

    /// Query available GPU memory for a single device (non-CUDA platforms)
    #[cfg(not(feature = "cuda"))]
    pub fn get_rank_available_memory(&self, _device_id: usize) -> Result<u64> {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_all();

        #[cfg(feature = "metal")]
        let avail_mem = {
            let device = metal::Device::system_default().expect("No Metal device found");
            let max_mem = device.recommended_max_working_set_size();
            let alloc_mem = device.current_allocated_size();
            std::cmp::max(max_mem.saturating_sub(alloc_mem), sys.available_memory())
        };

        #[cfg(not(feature = "metal"))]
        let avail_mem = sys.available_memory();

        // Apply kv_fraction only - reserved memory check is done post-allocation
        let usable = (avail_mem as f64 * self.kv_fraction) as u64;

        crate::log_warn!(
            "Memory: available {:.2} GB, kv_fraction {:.0}%, Max usable cache budget {:.2} GB",
            avail_mem as f64 / SIZE_IN_GB,
            self.kv_fraction * 100.0,
            usable as f64 / SIZE_IN_GB
        );

        Ok(usable)
    }

    /// Query available GPU memory across all given device_ids
    /// Returns the MINIMUM usable memory across all devices
    pub fn get_available_memory(&self, device_ids: &[usize]) -> Result<u64> {
        let mut min_memory: Option<u64> = None;

        for &device_id in device_ids {
            let mem = self.get_rank_available_memory(device_id)?;
            min_memory = Some(min_memory.map_or(mem, |m| std::cmp::min(m, mem)));
        }

        min_memory.ok_or_else(|| candle_core::Error::msg("No device IDs provided"))
    }

    /// Auto-decide optimal (max_num_seqs, max_model_len) within memory budget
    fn auto_decide_params(
        &self,
        available_memory: u64,
    ) -> std::result::Result<(usize, usize), KVCacheError> {
        let per_block = self.per_block_bytes();
        let total_blocks = available_memory as usize / per_block;

        // Descending candidate model lengths
        let candidates = [
            self.config_model_len,
            self.config_model_len / 2,
            self.config_model_len / 4,
            self.config_model_len / 8,
            16 * 1024,
            8 * 1024,
            4 * 1024,
            1024,
        ];

        // Find the first (max_seqs, max_len) that fits in memory
        for &max_len in candidates.iter() {
            if max_len == 0 {
                continue;
            }
            let blocks_per_seq = (max_len + self.block_size - 1) / self.block_size;
            let max_possible_seqs = total_blocks / blocks_per_seq;

            if max_possible_seqs > 0 {
                // Cap at 8 sequences to avoid memory overuse on small models
                let max_seqs = std::cmp::min(max_possible_seqs, 8);

                // Verify this actually fits in memory
                let required_blocks = max_seqs * blocks_per_seq;
                let required_bytes = required_blocks * per_block;

                if required_bytes as u64 <= available_memory {
                    return Ok((max_seqs, max_len));
                }
            }
        }

        Err(KVCacheError::InsufficientGpuMemory {
            available_mb: available_memory as f64 / SIZE_IN_MB,
            required_mb: (candidates.last().unwrap_or(&1024) * per_block / self.block_size) as f64
                / SIZE_IN_MB,
            reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
        })
    }

    /// Calculate allocation plan given the minimum available memory across ranks
    /// This is the main entry point - call AFTER collecting all rank memory reports
    pub fn plan_allocation(
        &self,
        min_available_memory: u64,
        num_shards: usize,
    ) -> std::result::Result<KVCacheAllocation, KVCacheError> {
        let per_block = self.per_block_bytes();
        if per_block == 0 {
            return Err(KVCacheError::InvalidConfiguration {
                message: format!(
                    "Invalid KVCache block size: per_block=0 (num_kv_heads={}, num_shards={}, head_dim={}, block_size={}, dtype_size={}, num_kv_layers={})",
                    self.num_kv_heads,
                    self.num_shards,
                    self.head_dim,
                    self.block_size,
                    self.dtype_size,
                    self.num_kv_layers
                ),
            });
        }

        let mut available_memory_for_kvcache = min_available_memory;
        let (max_num_seqs, max_model_len) = if let (Some(max_num_seqs), Some(max_model_len)) =
            (self.user_max_num_seqs, self.user_max_model_len)
        {
            let required_bytes = self.calculate_required_memory(max_num_seqs, max_model_len);
            if required_bytes as u64 > min_available_memory {
                return Err(KVCacheError::InsufficientGpuMemory {
                    available_mb: min_available_memory as f64 / SIZE_IN_MB,
                    required_mb: required_bytes as f64 / SIZE_IN_MB,
                    reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
                });
            }
            available_memory_for_kvcache = required_bytes as u64;
            (max_num_seqs, max_model_len)
        } else {
            // Auto-decide based on available capacity
            self.auto_decide_params(min_available_memory)?
        };

        // Calculate number of GPU blocks based on ALL available memory
        // This maximizes KVCache capacity - the scheduler will use max_num_seqs/max_model_len as limits
        let num_gpu_blocks = available_memory_for_kvcache as usize / per_block;

        if num_gpu_blocks == 0 {
            return Err(KVCacheError::InsufficientGpuMemory {
                available_mb: available_memory_for_kvcache as f64 / SIZE_IN_MB,
                required_mb: per_block as f64 / SIZE_IN_MB,
                reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
            });
        }

        // Max usable KVCache tokens = num_blocks * block_size
        let max_num_batched_tokens = num_gpu_blocks * self.block_size;
        let kvcache_memory_bytes = num_gpu_blocks * per_block;

        // CPU blocks for swap
        #[cfg(feature = "cuda")]
        let num_cpu_blocks = (num_gpu_blocks as f32 * self.cpu_mem_fold) as usize;
        #[cfg(not(feature = "cuda"))]
        let num_cpu_blocks = 1;

        // Final validation
        if num_gpu_blocks == 0 || max_num_seqs == 0 {
            return Err(KVCacheError::InsufficientGpuMemory {
                available_mb: min_available_memory as f64 / SIZE_IN_MB,
                required_mb: per_block as f64 / SIZE_IN_MB,
                reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
            });
        }

        let allocation = KVCacheAllocation {
            num_gpu_blocks,
            num_cpu_blocks,
            max_num_seqs,
            max_model_len,
            kvcache_memory_bytes,
            max_num_batched_tokens,
        };

        crate::log_warn!(
            "KVCache Allocation: {} GPU blocks ({:.2} GB x {}), max usable kvcache tokens {} ({}k bytes per token), scheduling limits [{} seqs x {} tokens]",
            num_gpu_blocks,
            kvcache_memory_bytes as f64 / SIZE_IN_GB,
            num_shards,
            max_num_batched_tokens,
            per_block / 1024 / self.block_size,
            max_num_seqs,
            max_model_len
        );

        Ok(allocation)
    }

    /// Apply allocation result to EngineConfig
    pub fn apply_to_config(&self, allocation: &KVCacheAllocation, econfig: &mut EngineConfig) {
        econfig.num_blocks = allocation.num_gpu_blocks;
        econfig.max_num_seqs = allocation.max_num_seqs;
        econfig.max_model_len = Some(allocation.max_model_len);
        econfig.kvcache_memory_bytes = allocation.kvcache_memory_bytes;
        econfig.max_num_batched_tokens = allocation.max_num_batched_tokens;
    }

    /// Check if auto-decide mode is needed
    pub fn needs_auto_decide(&self) -> bool {
        self.user_max_model_len.is_none()
    }

    //==========================================================================
    // Tensor Allocation Methods
    //==========================================================================

    /// Calculate flash-context KV block shape: [num_blocks, block_size, num_kv_heads, head_size]
    fn calculate_flash_key_value_block_shape(&self) -> (usize, usize, usize) {
        (self.block_size, self.kv_heads_per_shard(), self.head_dim)
    }

    /// Calculate key block shape for paged attention
    fn calculate_key_block_shape(&self, cache_dtype: DType) -> (usize, usize, usize, usize) {
        let element_size = cache_dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            self.kv_heads_per_shard(),
            self.head_dim / x,
            self.block_size,
            x,
        )
    }

    /// Calculate value block shape for paged attention
    fn calculate_value_block_shape(&self) -> (usize, usize, usize) {
        (self.kv_heads_per_shard(), self.head_dim, self.block_size)
    }

    /// Initialize KV cache tensors on GPU and CPU
    ///
    /// # Arguments
    /// * `allocation` - The allocation plan from `plan_allocation()`
    /// * `dtype` - Data type for the cache (will use U8 for FP8)
    /// * `device` - The GPU device to allocate on
    /// * `pd_config` - Optional P/D config for sync allocation
    ///
    /// # Returns
    /// Tuple of (GPU KV cache, CPU KV cache) - each is a Vec of (key_tensor, value_tensor) per layer
    pub fn init_kv_cache(
        &self,
        allocation: &KVCacheAllocation,
        dtype: DType,
        device: &Device,
        pd_config: Option<&crate::transfer::PdConfig>,
    ) -> Result<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
        let num_gpu_blocks = allocation.num_gpu_blocks;
        let num_cpu_blocks = allocation.num_cpu_blocks;

        #[cfg(not(feature = "cuda"))]
        let sync_alloc = true;

        #[allow(unused)]
        #[cfg(feature = "cuda")]
        let sync_alloc = if let Some(p_cfg) = pd_config {
            matches!(p_cfg.role, crate::transfer::PdRole::Server)
        } else {
            false
        };

        #[cfg(not(feature = "cuda"))]
        let _ = pd_config;

        let cache_dtype = if self.fp8_kvcache { DType::U8 } else { dtype };
        crate::log_warn!(
            "Using FP8 KV Cache? {}, cache dtype {:?}",
            self.fp8_kvcache,
            cache_dtype
        );

        if cfg!(feature = "flashinfer") || cfg!(feature = "flash-context") {
            assert!(
                !self.fp8_kvcache,
                "fp8 kvcache is not compatible with flashinfer or flash-context feature!"
            );

            let kv_shape = self.calculate_flash_key_value_block_shape();

            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for _ in 0..self.num_kv_layers {
                let key_blocks = Tensor::empty(
                    (num_gpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                let value_blocks = Tensor::empty(
                    (num_gpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                gpu_cache.push((key_blocks, value_blocks));
            }
            for _ in 0..self.num_kv_layers {
                let key_blocks = Tensor::zeros(
                    (num_cpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_cpu_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                cpu_cache.push((key_blocks, value_blocks));
            }
            Ok((gpu_cache, cpu_cache))
        } else {
            let kshape = self.calculate_key_block_shape(cache_dtype);
            let vshape = self.calculate_value_block_shape();

            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for _ in 0..self.num_kv_layers {
                let key_blocks = Tensor::empty(
                    (num_gpu_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                let value_blocks = Tensor::empty(
                    (num_gpu_blocks, vshape.0, vshape.1, vshape.2),
                    cache_dtype,
                    device,
                    Some(sync_alloc),
                )?;
                gpu_cache.push((key_blocks, value_blocks));
            }
            for _ in 0..self.num_kv_layers {
                let key_blocks = Tensor::zeros(
                    (num_cpu_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_cpu_blocks, vshape.0, vshape.1, vshape.2),
                    cache_dtype,
                    &Device::Cpu,
                )?;
                cpu_cache.push((key_blocks, value_blocks));
            }
            Ok((gpu_cache, cpu_cache))
        }
    }
}
