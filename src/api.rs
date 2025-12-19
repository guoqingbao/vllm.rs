use crate::core::engine::{LLMEngine, StreamItem, GLOBAL_RT};
use crate::core::GenerationOutput;
use crate::server::{build_messages_and_images, run_server, ChatMessage};
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, SamplingParams};
use crate::utils::get_dtype;
use candle_core::{DType, Result};
use parking_lot::RwLock;
use std::borrow::Cow;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub enum ModelRepo {
    /// (model_id, filename) -- when filename is None, treat as safetensor model id.
    /// When filename is Some, treat as GGUF model id + GGUF filename.
    ModelID((Cow<'static, str>, Option<Cow<'static, str>>)),
    /// Safetensor local path.
    ModelPath(Cow<'static, str>),
    /// GGUF file(s). Only the first file is used today.
    ModelFile(Vec<Cow<'static, str>>),
}

#[derive(Clone, Debug)]
pub struct EngineBuilder {
    repo: ModelRepo,
    isq: Option<String>,
    dtype: Option<DType>,
    flash_attn: Option<bool>,
    fp8_kvcache: Option<bool>,
    context_cache: Option<bool>,
    device_ids: Option<Vec<usize>>,
}

impl EngineBuilder {
    pub fn new(repo: ModelRepo) -> Self {
        Self {
            repo,
            isq: None,
            dtype: None,
            flash_attn: None,
            fp8_kvcache: None,
            context_cache: None,
            device_ids: None,
        }
    }

    pub fn with_isq(mut self, isq: impl Into<String>) -> Self {
        self.isq = Some(isq.into());
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn with_flash_attn(mut self) -> Self {
        self.flash_attn = Some(true);
        self
    }

    pub fn without_flash_attn(mut self) -> Self {
        self.flash_attn = Some(false);
        self
    }

    pub fn with_fp8_kvcache(mut self) -> Self {
        self.fp8_kvcache = Some(true);
        self
    }

    pub fn with_context_cache(mut self, enabled: bool) -> Self {
        self.context_cache = Some(enabled);
        self
    }

    pub fn with_multirank(mut self, device_ids: &str) -> Result<Self> {
        self.device_ids = Some(parse_device_ids(device_ids)?);
        Ok(self)
    }

    pub fn build(self) -> Result<Engine> {
        let mut econfig = EngineConfig::new(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.context_cache,
            self.fp8_kvcache,
            Some(true),
            None,
            None,
            None,
        );

        match self.repo {
            ModelRepo::ModelID((model_id, filename)) => {
                econfig.model_id = Some(model_id.into_owned());
                econfig.weight_file = filename.map(|f| f.into_owned());
            }
            ModelRepo::ModelPath(path) => {
                econfig.weight_path = Some(path.into_owned());
            }
            ModelRepo::ModelFile(files) => {
                if files.len() > 1 {
                    crate::log_warn!("Multiple GGUF files provided, using the first one.");
                }
                econfig.weight_file = files.into_iter().next().map(|f| f.into_owned());
            }
        }

        econfig.isq = self.isq;
        econfig.device_ids = self.device_ids;

        if let Some(enable_flash_attn) = self.flash_attn {
            econfig.disable_flash_attn = Some(!enable_flash_attn);
        }

        let dtype = self.dtype.clone().map(dtype_to_str);
        let dtype = get_dtype(dtype);

        let engine = LLMEngine::new(&econfig, dtype)?;
        Ok(Engine {
            engine,
            econfig,
            dtype,
        })
    }
}

pub struct Engine {
    engine: Arc<RwLock<LLMEngine>>,
    econfig: EngineConfig,
    dtype: DType,
}

impl Engine {
    pub fn multirank(mut self, device_ids: &str) -> Result<Self> {
        self.econfig.device_ids = Some(parse_device_ids(device_ids)?);
        self.rebuild()?;
        Ok(self)
    }

    pub fn start_server(
        &mut self,
        port: usize,
        with_ui_server: bool,
        context_cache: bool,
    ) -> Result<()> {
        if self.econfig.flash_context != Some(context_cache) {
            self.econfig.flash_context = Some(context_cache);
            self.rebuild()?;
        }

        GLOBAL_RT.block_on(async {
            run_server(
                self.engine.clone(),
                self.econfig.clone(),
                port,
                with_ui_server,
                false,
            )
            .await
        })
    }

    pub fn generate(
        &mut self,
        params: SamplingParams,
        messages: Vec<ChatMessage>,
    ) -> Result<GenerationOutput> {
        let img_cfg = { self.engine.read().img_cfg.clone() };
        let (messages, image_data) = build_messages_and_images(&messages, img_cfg.as_ref())?;
        self.generate_messages(params, messages, image_data)
    }

    pub fn generate_messages(
        &mut self,
        params: SamplingParams,
        messages: Vec<Message>,
        images: Option<crate::utils::image::ImageData>,
    ) -> Result<GenerationOutput> {
        let (receivers, tokenizer) = {
            let mut engine = self.engine.write();
            (
                engine.generate_sync(&vec![params], &vec![messages], images)?,
                Arc::new(engine.tokenizer.clone()),
            )
        };

        let mut results = GLOBAL_RT
            .block_on(async { LLMEngine::collect_sync_results(receivers, tokenizer).await })?;

        results
            .pop()
            .ok_or_else(|| candle_core::Error::msg("No generation output returned"))
    }

    pub fn generate_stream(
        &mut self,
        params: SamplingParams,
        messages: Vec<ChatMessage>,
    ) -> Result<EngineStream> {
        let img_cfg = { self.engine.read().img_cfg.clone() };
        let (messages, image_data) = build_messages_and_images(&messages, img_cfg.as_ref())?;

        let (seq_id, prompt_length, stream) = {
            let mut engine = self.engine.write();
            engine.generate_stream(&params, &messages, image_data)?
        };

        Ok(EngineStream {
            engine: self.engine.clone(),
            rx: stream,
            finished: false,
            seq_id,
            prompt_length,
            cancelled: false,
        })
    }

    pub fn get_num_cached_tokens(&self) -> usize {
        let engine = self.engine.read();
        engine.get_num_cached_tokens()
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let engine = self.engine.read();
        engine.get_available_kv_tokens()
    }

    fn rebuild(&mut self) -> Result<()> {
        self.engine = LLMEngine::new(&self.econfig, self.dtype)?;
        Ok(())
    }
}

pub struct EngineStream {
    engine: Arc<RwLock<LLMEngine>>,
    rx: mpsc::Receiver<StreamItem>,
    finished: bool,
    pub seq_id: usize,
    pub prompt_length: usize,
    cancelled: bool,
}

impl EngineStream {
    pub fn cancel(&mut self) {
        self.cancelled = true;
        let mut engine_guard = self.engine.write();
        engine_guard.cancel(self.seq_id);
    }

    pub async fn recv(&mut self) -> Option<StreamItem> {
        if self.finished {
            return None;
        }
        let item = self.rx.recv().await;
        if matches!(item, Some(StreamItem::Done(_) | StreamItem::Error(_))) {
            self.finished = true;
        }
        item
    }

    pub fn recv_blocking(&mut self) -> Option<StreamItem> {
        if self.finished {
            return None;
        }
        let item = GLOBAL_RT.block_on(self.rx.recv());
        if matches!(item, Some(StreamItem::Done(_) | StreamItem::Error(_))) {
            self.finished = true;
        }
        item
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

fn parse_device_ids(device_ids: &str) -> Result<Vec<usize>> {
    device_ids
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            usize::from_str(s.trim())
                .map_err(|e| candle_core::Error::msg(format!("Invalid device id '{s}': {e}")))
        })
        .collect()
}

fn dtype_to_str(dtype: DType) -> String {
    match dtype {
        DType::F16 => "f16".to_string(),
        DType::BF16 => "bf16".to_string(),
        DType::F32 => "f32".to_string(),
        _ => "bf16".to_string(),
    }
}
