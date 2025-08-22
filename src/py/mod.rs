use crate::core::engine::LLMEngine;
use crate::core::engine::StreamItem;
use crate::core::engine::GLOBAL_RT;
use crate::core::GenerationOutput;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, GenerationConfig, SamplingParams};
use candle_core::DType;
use parking_lot::RwLock;
use pyo3::exceptions::PyStopIteration;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::mpsc;
/// Python wrapper
#[pyclass]
pub struct Engine {
    engine: Arc<RwLock<LLMEngine>>,
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
            Ok(engine) => Ok(Self { engine, econfig }),
            Err(e) => Err(PyValueError::new_err(format!("Engine init failed: {e:?}"))),
        }
    }

    #[pyo3(text_signature = "($self, messages, log)")]
    pub fn apply_chat_template(
        &self,
        params: SamplingParams,
        messages: Vec<Message>,
        log: bool,
    ) -> String {
        self.engine
            .read()
            .apply_chat_template(&params, &messages, log)
    }

    #[pyo3(name = "generate_sync", text_signature = "($self, params, prompts)")]
    pub fn generate_sync(
        &mut self,
        params: Vec<SamplingParams>,
        prompts: Vec<String>,
    ) -> PyResult<Vec<GenerationOutput>> {
        tokio::task::block_in_place(|| {
            GLOBAL_RT.block_on(async {
                let (receivers, tokenizer) = {
                    let mut engine = self.engine.write();
                    (
                        engine.generate_sync(&params, prompts).map_err(|e| {
                            PyValueError::new_err(format!("generate_sync failed: {:?}", e))
                        })?,
                        Arc::new(engine.tokenizer.clone()),
                    )
                };

                let results = LLMEngine::collect_sync_results(receivers, tokenizer)
                    .await
                    .map_err(|e| {
                        PyValueError::new_err(format!("collect_sync_results failed: {:?}", e))
                    })?;

                Ok(results)
            })
        })
    }

    #[pyo3(name = "generate_stream", text_signature = "($self)")]
    pub fn generate_stream(
        &mut self,
        params: SamplingParams,
        prompt: String,
    ) -> PyResult<(usize, usize, EngineStream)> {
        let (seq_id, prompt_length, stream) = {
            let mut engine = self.engine.write();
            engine
                .generate_stream(&params, prompt)
                .map_err(|e| PyValueError::new_err(format!("stream error: {:?}", e)))?
        };

        Ok((
            seq_id,
            prompt_length,
            EngineStream {
                engine: self.engine.clone(),
                finished: false,
                seq_id,
                prompt_length,
                cancelled: false,
                rx: std::sync::Mutex::new(stream),
            },
        ))
    }

    #[pyo3(name = "get_num_cached_tokens", text_signature = "($self)")]
    pub fn get_num_cached_tokens(&mut self) -> PyResult<usize> {
        let engine = self.engine.read();
        Ok(engine.get_num_cached_tokens())
    }
}

#[pyclass]
#[allow(unused_variables)]
pub struct EngineStream {
    engine: Arc<RwLock<LLMEngine>>,
    rx: std::sync::Mutex<mpsc::Receiver<StreamItem>>,
    #[pyo3(get, set)]
    finished: bool,
    #[pyo3(get, set)]
    seq_id: usize,
    #[pyo3(get, set)]
    prompt_length: usize,
    #[pyo3(get, set)]
    cancelled: bool, // User cancellation flag
}

#[pymethods]
impl EngineStream {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn cancel(mut slf: PyRefMut<Self>) {
        slf.cancelled = true;
        let mut engine_guard = slf.engine.write();
        engine_guard.cancel(slf.seq_id);
    }

    fn __next__(&self) -> PyResult<String> {
        let mut rx = self.rx.lock().unwrap();

        match GLOBAL_RT.block_on(rx.recv()) {
            Some(StreamItem::Token(token)) => Ok(token),
            Some(StreamItem::Done(_)) | None => Err(PyStopIteration::new_err("[DONE]")),
            Some(StreamItem::Error(e)) => Err(PyValueError::new_err(e)),
            Some(StreamItem::TokenID(_)) => Err(PyValueError::new_err(
                "We should not receive raw token id (used for completion) during streaming!",
            )),
            Some(_) => self.__next__(), // Skip over Completion/etc
        }
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
    #[pyo3(signature = (model_path, max_num_seqs=Some(32), max_model_len=Some(1024), isq=None, num_shards=Some(1), device_ids=None, generation_cfg=None, seed=None))]
    pub fn new(
        model_path: String,
        max_num_seqs: Option<usize>,
        max_model_len: Option<usize>,
        isq: Option<String>,
        num_shards: Option<usize>,
        device_ids: Option<Vec<usize>>,
        generation_cfg: Option<GenerationConfig>,
        seed: Option<u64>,
    ) -> Self {
        let mut device_ids = device_ids.unwrap_or_default();
        if device_ids.is_empty() {
            device_ids.push(0);
        }
        #[cfg(any(feature = "flash-decoding", feature = "flash-context"))]
        let block_size = 256;
        #[cfg(not(any(feature = "flash-decoding", feature = "flash-context")))]
        let block_size = 32;
        Self {
            model_path,
            tokenizer: None,
            tokenizer_config: None,
            num_blocks: 128, //placeholder
            block_size,
            max_num_seqs: max_num_seqs.unwrap_or(32),
            max_num_batched_tokens: 32768, //placeholder
            max_model_len,                 //placeholder
            isq,
            num_shards,
            device_ids: Some(device_ids),
            generation_cfg,
            seed,
        }
    }
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (temperature=None, max_tokens=Some(4096), ignore_eos=Some(false), top_k=None, top_p=None, session_id=None))]
    pub fn new(
        temperature: Option<f32>,
        max_tokens: Option<usize>,
        ignore_eos: Option<bool>,
        top_k: Option<isize>,
        top_p: Option<f32>,
        session_id: Option<String>,
    ) -> Self {
        Self {
            temperature,
            max_tokens: max_tokens.unwrap_or(4096),
            ignore_eos: ignore_eos.unwrap_or(false),
            top_k,
            top_p,
            session_id,
        }
    }
}

#[pymethods]
impl GenerationConfig {
    #[new]
    #[pyo3(signature = (temperature=None, top_p=None, top_k=None, penalty=None))]
    pub fn new(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<isize>,
        penalty: Option<f32>,
    ) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            penalty,
        }
    }
}
