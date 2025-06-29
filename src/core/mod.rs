pub mod block_manager;
pub mod engine;
pub mod runner;
pub mod scheduler;
pub mod sequence;
#[cfg(feature = "python")]
use pyo3::pyclass;

#[cfg(feature = "python")]
#[pyclass]
#[derive(Debug)]
pub struct GenerationOutput {
    #[pyo3(get)]
    pub seq_id: usize,
    #[pyo3(get)]
    pub prompt_length: usize,
    #[pyo3(get)]
    pub decode_start_time: usize,
    #[pyo3(get)]
    pub decoded_length: usize,
    #[pyo3(get)]
    pub decode_output: String,
}

#[cfg(not(feature = "python"))]
#[derive(Debug)]
pub struct GenerationOutput {
    pub seq_id: usize,
    pub prompt_length: usize,
    pub decode_start_time: usize,
    pub decoded_length: usize,
    pub decode_output: String,
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "python")]
            {
                println!($($arg)*);
            }
            #[cfg(not(feature = "python"))]
            {
                tracing::info!($($arg)*);
            }
        }
    };
}
