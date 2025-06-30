use crate::core::engine::LLMEngine;
use crate::core::GenerationOutput;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, SamplingParams};
use candle_core::DType;
use either::Either;
use parking_lot::RwLock;
use pyo3::exceptions::PyStopIteration;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use std::sync::Arc;
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
            Ok(engine) => Ok(Engine {
                engine: Arc::new(RwLock::new(engine)),
                econfig,
            }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to create engine ({:?})",
                e
            ))),
        }
    }

    #[pyo3(text_signature = "($self, messages, log)")]
    pub fn apply_chat_template(&self, messages: Vec<Message>, log: bool) -> String {
        self.engine.read().apply_chat_template(&messages, log)
    }

    #[pyo3(name = "generate_sync", text_signature = "($self, params, prompts)")]
    pub fn generate_sync(
        &mut self,
        params: SamplingParams,
        prompts: Vec<String>,
    ) -> PyResult<Vec<GenerationOutput>> {
        self.engine
            .write()
            .generate(&params, &prompts)
            .map_err(|e| PyValueError::new_err(format!("Failed to generate ({:?})", e)))
    }

    #[pyo3(name = "generate_stream", text_signature = "($self)")]
    pub fn generate_stream(
        &mut self,
        params: SamplingParams,
        prompt: String,
    ) -> PyResult<EngineStream> {
        let (seq_id, _) = {
            self.engine
                .write()
                .add_request(params, &prompt)
                .map_err(|e| PyValueError::new_err(format!("Failed to add request ({:?})", e)))?
        };

        let tokenizer = self.engine.read().tokenizer.clone();

        let leaked: &'static _ = Box::leak(Box::new(tokenizer));
        let decoder = leaked.decode_stream(false);
        let wrapped = StreamWithTokenizer {
            _tokenizer: unsafe { Box::from_raw(leaked as *const _ as *mut _) },
            stream: decoder,
        };
        let boxed_decoder: Box<dyn DecodeStreamTrait + Send + Sync> = Box::new(wrapped);

        Ok(EngineStream {
            engine: self.engine.clone(),
            finished: false,
            seq_id,
            stream_decoder: boxed_decoder,
            cancelled: false,
        })
    }
}

pub trait DecodeStreamTrait: Send + Sync {
    fn step(&mut self, id: u32) -> Option<String>;
}

struct StreamWithTokenizer<M, N, PT, PP, D>
where
    M: tokenizers::Model + Send + Sync + 'static,
    N: tokenizers::Normalizer + Send + Sync + 'static,
    PT: tokenizers::PreTokenizer + Send + Sync + 'static,
    PP: tokenizers::PostProcessor + Send + Sync + 'static,
    D: tokenizers::Decoder + Send + Sync + 'static,
{
    _tokenizer: Box<tokenizers::TokenizerImpl<M, N, PT, PP, D>>,
    stream: tokenizers::DecodeStream<'static, M, N, PT, PP, D>,
}

impl<M, N, PT, PP, D> DecodeStreamTrait for StreamWithTokenizer<M, N, PT, PP, D>
where
    M: tokenizers::Model + Send + Sync + 'static,
    N: tokenizers::Normalizer + Send + Sync + 'static,
    PT: tokenizers::PreTokenizer + Send + Sync + 'static,
    PP: tokenizers::PostProcessor + Send + Sync + 'static,
    D: tokenizers::Decoder + Send + Sync + 'static,
{
    fn step(&mut self, id: u32) -> Option<String> {
        self.stream.step(id).ok().flatten()
    }
}

type DecodeStreamType = Box<dyn DecodeStreamTrait + Send + Sync>;

#[pyclass]
#[allow(unused_variables)]
pub struct EngineStream {
    engine: Arc<RwLock<LLMEngine>>,
    finished: bool,
    seq_id: usize,
    stream_decoder: DecodeStreamType,
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

    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<PyObject> {
        if slf.finished {
            return Err(PyStopIteration::new_err("Finished!"));
        }

        let step_output = {
            let mut engine_guard = slf.engine.write();
            match engine_guard.step() {
                Ok(output) => output,
                Err(e) => {
                    return Err(PyValueError::new_err(format!("step error: {:?}", e)));
                }
            }
        };

        let is_finished = {
            let engine_guard = slf.engine.read();
            engine_guard.scheduler.is_finished()
        };
        slf.finished = is_finished;

        Python::with_gil(|py| {
            let py_list: Vec<_> = step_output
                .into_iter()
                .map(|(id, tok)| {
                    let obj = match tok {
                        Either::Left(token_id) => {
                            if let Some(output) = slf.stream_decoder.step(token_id) {
                                output.into_py_any(py).unwrap()
                            } else {
                                "".into_py_any(py).unwrap()
                            }
                        }
                        Either::Right(_) => "[DONE]".into_py_any(py).unwrap(),
                    };
                    (id, obj)
                })
                .collect();
            py_list.into_py_any(py)
        })
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
    #[pyo3(signature = (model_path, block_size=Some(32), max_num_seqs=Some(32), quant=None, num_shards=Some(1), kvcache_mem_gpu=Some(4096), device_ids=None))]
    pub fn new(
        model_path: String,
        block_size: Option<usize>,
        max_num_seqs: Option<usize>,
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
            block_size: block_size.unwrap_or(32),
            max_num_seqs: max_num_seqs.unwrap_or(32),
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
    #[pyo3(signature = (temperature=None, max_tokens=Some(4096), ignore_eos=Some(false), top_k=None, top_p=None))]
    pub fn new(
        temperature: Option<f32>,
        max_tokens: Option<usize>,
        ignore_eos: Option<bool>,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Self {
        Self {
            temperature: temperature.unwrap_or(1.0),
            max_tokens: max_tokens.unwrap_or(4096),
            ignore_eos: ignore_eos.unwrap_or(false),
            top_k,
            top_p,
        }
    }
}
