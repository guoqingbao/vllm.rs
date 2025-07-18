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
    pub prompt_start_time: usize,
    #[pyo3(get)]
    pub decode_start_time: usize,
    #[pyo3(get)]
    pub decode_finish_time: usize,
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
    pub prompt_start_time: usize,
    pub decode_start_time: usize,
    pub decode_finish_time: usize,
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

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "python")]
            {
                eprintln!($($arg)*);
            }
            #[cfg(not(feature = "python"))]
            {
                tracing::error!($($arg)*);
            }
        }
    };
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
// type StreamDecoderMap = HashMap<usize, DecodeStreamType>;
