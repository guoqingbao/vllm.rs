use candle::quantized::QTensor;
use candle::{Device, Result, Shape};
use candle_core as candle;
use std::fs::File;
use std::sync::Arc;
use std::sync::Mutex;
// light-cached qvarbuilder

#[derive(Clone)]
pub struct VarBuilder {
    content: Arc<candle_core::quantized::gguf_file::Content>,
    file: Arc<std::sync::Mutex<File>>, // Keep file open for lazy loading
    cache: Arc<Mutex<Option<(String, Arc<QTensor>)>>>, // last cached tensor
    path: Vec<String>,
    device: Device,
}

impl VarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let mut file = File::open(&p)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        Ok(Self {
            content: Arc::new(content),
            file: Arc::new(std::sync::Mutex::new(file)),
            cache: Arc::new(Mutex::new(None)),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            content: self.content.clone(),
            file: self.file.clone(),
            cache: self.cache.clone(),
            path,
            device: self.device.clone(),
        }
    }

    pub fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);

        // Check cache
        {
            let cache_guard = self.cache.lock().unwrap();
            if let Some((ref cached_name, ref cached_tensor)) = *cache_guard {
                if cached_name == &path {
                    // Return cached tensor
                    let shape = s.into();
                    if cached_tensor.shape() != &shape {
                        candle::bail!(
                            "shape mismatch for {name}, got {:?}, expected {shape:?}",
                            cached_tensor.shape()
                        );
                    }
                    return Ok(cached_tensor.clone());
                }
            }
        }

        let mut file = self.file.lock().unwrap();
        let tensor = self.content.tensor(&mut *file, &path, &self.device)?;
        let tensor = Arc::new(tensor);
        // Update cache
        *self.cache.lock().unwrap() = Some((path.clone(), tensor.clone()));

        let shape = s.into();
        if tensor.shape() != &shape {
            candle::bail!(
                "shape mismatch for {name}, got {:?}, expected {shape:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let mut file = self.file.lock().unwrap();
        let tensor = self.content.tensor(&mut *file, name, &self.device)?;
        Ok(Arc::new(tensor))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.content.tensor_infos.contains_key(key)
    }
}
