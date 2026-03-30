use std::env;

pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "VLLM_RS_MAMBA_SNAPSHOT_STRIDE_BLOCKS";
pub const DEFAULT_MAMBA_SNAPSHOT_BLOCK_STRIDE: usize = 1;

pub fn mamba_snapshot_block_stride_blocks() -> usize {
    let default = DEFAULT_MAMBA_SNAPSHOT_BLOCK_STRIDE;
    let Ok(raw) = env::var(MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV) else {
        return default;
    };
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            crate::log_warn!(
                "{} must be >= 1, got 0. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                default
            );
            default
        }
        Ok(v) => v,
        Err(_) => {
            crate::log_warn!(
                "Invalid {}='{}'. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                raw,
                default
            );
            default
        }
    }
}
