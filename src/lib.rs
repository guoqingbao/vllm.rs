#[cfg(feature = "python")]
use pyo3::prelude::*;
pub mod core;
pub mod models;
#[cfg(feature = "python")]
pub mod py;
pub mod runner;
pub mod server;
pub mod utils;
#[cfg(feature = "python")]
use crate::core::GenerationOutput;
#[cfg(feature = "python")]
use crate::py::Engine;
#[cfg(feature = "python")]
use crate::utils::chat_template::Message;
#[cfg(feature = "python")]
use crate::utils::config::{EngineConfig, GenerationConfig, SamplingParams};
/// A Python module implemented in Rust.
#[cfg(feature = "python")]
#[pymodule]
fn vllm_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Engine>()?;
    m.add_class::<EngineConfig>()?;
    m.add_class::<SamplingParams>()?;
    m.add_class::<GenerationConfig>()?;
    m.add_class::<GenerationOutput>()?;
    m.add_class::<Message>()?;
    Ok(())
}
