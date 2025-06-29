use crate::core::engine::LLMEngine;
use crate::core::GenerationOutput;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, SamplingParams};
use candle_core::DType;
use either::Either;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

/// Python wrapper
#[pyclass]
pub struct Engine {
    engine: LLMEngine,
    #[pyo3(get, set)]
    econfig: EngineConfig,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(text_signature = "(econfig, dtype)")]
    pub fn new(econfig: EngineConfig, dtype: String) -> PyResult<Self> {
        let dtype_parsed = match dtype.as_str() {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "f32" => DType::F32,
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid data type (only f16, bf16 and f32 are supported)",
                ))
            }
        };

        match LLMEngine::new(&econfig, dtype_parsed) {
            Ok(engine) => Ok(Engine { engine, econfig }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to create engine ({:?})",
                e
            ))),
        }
    }

    #[pyo3(text_signature = "($self, params, prompt)")]
    pub fn add_request(
        &mut self,
        params: SamplingParams,
        prompt: &str,
    ) -> PyResult<(usize, usize)> {
        self.engine
            .add_request(params, prompt)
            .map_err(|e| PyValueError::new_err(format!("Failed to add request ({:?})", e)))
    }

    #[pyo3(text_signature = "($self)")]
    pub fn step(&mut self, py: Python<'_>) -> PyResult<Vec<(usize, PyObject)>> {
        let result = self
            .engine
            .step()
            .map_err(|e| PyValueError::new_err(format!("Failed to perform step ({:?})", e)))?;

        // Convert Either<u32, Vec<u32>> into Python-compatible objects
        let converted = result
            .into_iter()
            .map(|(id, val)| {
                let obj = match val {
                    Either::Left(x) => x.into_py_any(py).unwrap(),
                    Either::Right(vs) => vs.into_py_any(py).unwrap(),
                };
                (id, obj)
            })
            .collect();
        Ok(converted)
    }

    #[pyo3(text_signature = "($self, messages, log)")]
    pub fn apply_chat_template(&self, messages: Vec<Message>, log: bool) -> String {
        self.engine.apply_chat_template(&messages, log)
    }

    #[pyo3(text_signature = "($self, params, prompts, log)")]
    pub fn generate(
        &mut self,
        params: SamplingParams,
        prompts: Vec<String>,
    ) -> PyResult<Vec<GenerationOutput>> {
        self.engine
            .generate(&params, &prompts)
            .map_err(|e| PyValueError::new_err(format!("Failed to generate ({:?})", e)))
    }
}

#[pymethods]
impl Message {
    #[new]
    fn new(role: String, content: String) -> Self {
        Message { role, content }
    }
}

#[pymethods]
impl EngineConfig {
    #[new]
    pub fn new(
        model_path: String,
        block_size: usize,
        max_num_seqs: usize,
        quant: Option<String>,
        num_shards: Option<usize>,
        kvcache_mem_gpu: Option<usize>,
        device_ids: Option<Vec<usize>>,
    ) -> Self {
        let mut device_ids = device_ids.unwrap_or_default();
        if device_ids.is_empty() {
            device_ids.push(0);
        }
        Self {
            model_path,
            tokenizer: None,
            tokenizer_config: None,
            num_blocks: 128, //placeholder
            block_size,
            max_num_seqs,
            max_num_batched_tokens: 32768, //placeholder
            max_model_len: 32768,          //placeholder
            quant,
            num_shards,
            kvcache_mem_gpu,
            device_id: Some(device_ids[0]),
        }
    }
}

#[pymethods]
impl SamplingParams {
    #[new]
    pub fn new(
        temperature: f32,
        max_tokens: usize,
        ignore_eos: bool,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            ignore_eos,
            top_k,
            top_p,
        }
    }
}
