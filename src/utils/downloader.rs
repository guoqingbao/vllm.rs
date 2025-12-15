use candle_core::Result;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
pub struct Downloader {
    model_id: Option<String>,
    weight_path: Option<String>,
    weight_file: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelPaths {
    pub tokenizer_filename: PathBuf,
    pub tokenizer_config_filename: PathBuf,
    pub config_filename: PathBuf,
    pub generation_config_filename: PathBuf,
    pub filenames: Vec<PathBuf>,
    pub chat_template_filename: Option<PathBuf>,
}

impl ModelPaths {
    pub fn get_config_filename(&self) -> PathBuf {
        self.config_filename.clone()
    }
    pub fn get_tokenizer_filename(&self) -> PathBuf {
        self.tokenizer_filename.clone()
    }
    pub fn get_tokenizer_config_filename(&self) -> PathBuf {
        self.tokenizer_config_filename.clone()
    }
    pub fn get_weight_filenames(&self) -> Vec<PathBuf> {
        self.filenames.clone()
    }
    pub fn get_generation_config_filename(&self) -> PathBuf {
        self.generation_config_filename.clone()
    }
    pub fn get_chat_template_filename(&self) -> Option<PathBuf> {
        self.chat_template_filename.clone()
    }
}

impl Downloader {
    pub fn new(
        model_id: Option<String>,
        weight_path: Option<String>,
        weight_file: Option<String>,
    ) -> Self {
        Self {
            model_id,
            weight_path,
            weight_file,
        }
    }
}

pub(crate) fn get_token(hf_token: Option<String>, hf_token_path: Option<String>) -> Result<String> {
    Ok(match (hf_token, hf_token_path) {
        (Some(envvar), None) => env::var(envvar)
            .map_err(candle_core::Error::wrap)?
            .trim()
            .to_string(),
        (None, Some(path)) => fs::read_to_string(path)
            .map_err(candle_core::Error::wrap)?
            .trim()
            .to_string(),
        (None, None) => fs::read_to_string(format!(
            "{}/.cache/huggingface/token",
            dirs::home_dir().unwrap().display()
        ))
        .map_err(candle_core::Error::wrap)?
        .trim()
        .to_string(),
        (Some(_), Some(path)) => fs::read_to_string(path)
            .map_err(candle_core::Error::wrap)?
            .trim()
            .to_string(),
    })
}

impl Downloader {
    pub fn prepare_model_weights(
        &self,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<(ModelPaths, bool)> {
        let (paths, gguf): (ModelPaths, bool) = match (
            &self.model_id,
            &self.weight_path,
            &self.weight_file,
        ) {
            //model in a folder (safetensor format, huggingface folder structure)
            (None, Some(path), None) => {
                if !Path::new(path).is_dir() {
                    candle_core::bail!("Safetensor weight path must be a directory! \n\t***Tips: use `--f` to specify gguf model file!***");
                } else {
                    (
                        ModelPaths {
                            tokenizer_filename: Path::new(path).join("tokenizer.json"),
                            tokenizer_config_filename: Path::new(path)
                                .join("tokenizer_config.json"),
                            config_filename: Path::new(path).join("config.json"),
                            filenames: if Path::new(path)
                                .join("model.safetensors.index.json")
                                .exists()
                            {
                                super::hub_load_local_safetensors(
                                    path,
                                    "model.safetensors.index.json",
                                )?
                            } else {
                                //a single weight file case
                                let mut safetensors_files = Vec::<std::path::PathBuf>::new();
                                safetensors_files
                                    .insert(0, Path::new(path).join("model.safetensors"));
                                safetensors_files
                            },
                            generation_config_filename: if Path::new(path)
                                .join("generation_config.json")
                                .exists()
                            {
                                Path::new(path).join("generation_config.json")
                            } else {
                                "".into()
                            },
                            chat_template_filename: if Path::new(path)
                                .join("chat_template.json")
                                .exists()
                            {
                                Some(Path::new(path).join("chat_template.json"))
                            } else {
                                None
                            },
                        },
                        false,
                    )
                }
            }
            //model in a quantized file (gguf/ggml format)
            (None, path, Some(file)) => (
                ModelPaths {
                    tokenizer_filename: PathBuf::new(),
                    tokenizer_config_filename: PathBuf::new(),
                    config_filename: PathBuf::new(),
                    filenames: {
                        let path = path.clone().unwrap_or("".to_string());
                        if Path::new(&path).join(file).exists() {
                            vec![Path::new(&path).join(file)]
                        } else {
                            panic!("Model file not found {file}");
                        }
                    },
                    generation_config_filename: "".into(),
                    chat_template_filename: None,
                },
                true,
            ),
            (Some(_), None, Some(_)) => (self.download_gguf_model(None)?, true),
            (Some(_), None, None) => {
                //try download model anonymously
                let loaded = self.download_model(None, hf_token.clone(), hf_token_path.clone());
                // crate::log_warn!("Model pathes {:?}", loaded);
                if loaded.is_ok() {
                    (loaded.unwrap(), false)
                } else {
                    //if it's failed, try using huggingface token
                    crate::log_info!("Try request model using cached huggingface token...");
                    if hf_token.is_none() && hf_token_path.is_none() {
                        //no token provided
                        let token_path = format!(
                            "{}/.cache/huggingface/token",
                            dirs::home_dir().unwrap().display()
                        );
                        if !Path::new(&token_path).exists() {
                            //also no token cache
                            use std::io::Write;
                            let mut input_token = String::new();
                            crate::log_warn!("Unable to request model, please provide your huggingface token to download model:\n");
                            std::io::stdin()
                                .read_line(&mut input_token)
                                .expect("Failed to read token!");
                            std::fs::create_dir_all(Path::new(&token_path).parent().unwrap())
                                .unwrap();
                            let mut output = std::fs::File::create(token_path).unwrap();
                            write!(output, "{}", input_token.trim())
                                .expect("Failed to save token!");
                        }
                    }
                    (
                        self.download_model(None, hf_token.clone(), hf_token_path.clone())?,
                        false,
                    )
                }
            }
            _ => {
                candle_core::bail!("No model id or weight_path/weight_file provided!\n***Tips***: \n \t For local model weights, \
                    `--w <path/to/folder>` for safetensors models or gguf models.\n \
                    \t For remote safetensor models, `--m <model_id>` to download from HuggingFace hub. \
                    \n \t For remote gguf models, `--m <model_id> --f <weight_file>` to download from HuggingFace hub.");
            }
        };

        Ok((paths, gguf))
    }

    pub fn check_cache(&self) -> Option<PathBuf> {
        let sanitized_id = std::path::Path::new(self.model_id.as_ref().unwrap())
            .display()
            .to_string()
            .replace("/", "--");

        let home_folder = if dirs::home_dir().is_some() {
            let mut path = dirs::home_dir().unwrap();
            path.push(".cache/huggingface/hub/");
            if !path.exists() {
                let _ = std::fs::create_dir_all(&path);
            }
            path
        } else {
            "./".into()
        };

        let cache_dir: std::path::PathBuf = std::env::var("HF_HUB_CACHE")
            .map(std::path::PathBuf::from)
            .unwrap_or(home_folder.into());
        let cache_path = cache_dir.join(format!("models--{sanitized_id}/"));
        if !cache_path.join("refs/main").exists() {
            return None;
        }
        let cache_id = std::fs::read_to_string(&cache_path.join("refs/main")).ok()?;
        let cache_path = cache_path.join(format!("snapshots/{}/", cache_id));

        if !cache_path.exists() {
            return None;
        }

        crate::log_warn!("Cache found {:?}", cache_path);
        Some(cache_path)
    }

    pub fn download_model(
        &self,
        revision: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<ModelPaths> {
        assert!(self.model_id.is_some(), "No model id provided!");
        let mut filenames = vec![];

        if let Some(cache_path) = self.check_cache() {
            let tokenizer_filename = cache_path.join("tokenizer.json");
            let config_filename = cache_path.join("config.json");
            let tokenizer_config_filename = cache_path.join("tokenizer_config.json");
            let generation_config_filename = cache_path.join("generation_config.json");
            let mut chat_template_filename = cache_path.join("chat_template.json");
            if !chat_template_filename.exists() {
                chat_template_filename = cache_path.join("chat_template.jinja");
            }
            let chat_template_filename = if !chat_template_filename.exists() {
                Some(chat_template_filename)
            } else {
                None
            };
            for entry in std::fs::read_dir(&cache_path)? {
                let path = entry?.path();
                if path.extension() == Some("safetensors".as_ref()) {
                    crate::log_warn!("Found cache: {}", path.display());
                    filenames.push(path);
                }
            }
            return Ok(ModelPaths {
                tokenizer_filename,
                tokenizer_config_filename,
                config_filename,
                filenames,
                generation_config_filename,
                chat_template_filename,
            });
        }

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(get_token(hf_token, hf_token_path)?))
            .build()
            .map_err(candle_core::Error::wrap)?;
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            self.model_id.clone().unwrap(),
            RepoType::Model,
            revision.clone(),
        ));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .map_err(candle_core::Error::wrap)?;

        let config_filename = api.get("config.json").map_err(candle_core::Error::wrap)?;

        let tokenizer_config_filename = match api.get("tokenizer_config.json") {
            Ok(f) => f,
            _ => "".into(),
        };

        let generation_config_filename = match api.get("generation_config.json") {
            Ok(f) => f,
            _ => "".into(),
        };

        for rfilename in api
            .info()
            .map_err(candle_core::Error::wrap)?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| x.ends_with(".safetensors"))
        {
            let filename = api.get(&rfilename).map_err(candle_core::Error::wrap)?;
            filenames.push(filename);
        }

        Ok(ModelPaths {
            tokenizer_filename,
            tokenizer_config_filename,
            config_filename,
            filenames,
            generation_config_filename,
            chat_template_filename: None,
        })
    }

    pub fn download_gguf_model(&self, revision: Option<String>) -> Result<ModelPaths> {
        assert!(self.model_id.is_some(), "No model id provided!");
        crate::log_info!(
            "Downloading GGUF file {} from repo {}",
            self.weight_file.as_ref().unwrap(),
            self.model_id.as_ref().unwrap(),
        );
        let filename = self.weight_file.clone().unwrap();
        let mut filenames = vec![];
        if let Some(cache_path) = self.check_cache() {
            let cached_gguf_file = cache_path.join(&filename);
            if cached_gguf_file.exists() {
                crate::log_warn!("Found cache: {}", cached_gguf_file.display());
                filenames.push(cached_gguf_file.clone());
                return Ok(ModelPaths {
                    tokenizer_filename: "".into(),
                    tokenizer_config_filename: "".into(),
                    config_filename: "".into(),
                    filenames,
                    generation_config_filename: "".into(),
                    chat_template_filename: None,
                });
            }
        }
        let api = hf_hub::api::sync::Api::new().unwrap();
        let revision = revision.unwrap_or("main".to_string());
        let filename = api
            .repo(hf_hub::Repo::with_revision(
                self.model_id.clone().unwrap(),
                hf_hub::RepoType::Model,
                revision.to_string(),
            ))
            .get(filename.as_str())
            .map_err(candle_core::Error::wrap)?;
        filenames.push(filename);

        Ok(ModelPaths {
            tokenizer_filename: "".into(),
            tokenizer_config_filename: "".into(),
            config_filename: "".into(),
            filenames,
            generation_config_filename: "".into(),
            chat_template_filename: None,
        })
    }
}
