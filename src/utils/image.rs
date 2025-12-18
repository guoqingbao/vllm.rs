use crate::utils::config::ModelType;
use crate::utils::Config;
use candle_core::{DType, Device, Result, Storage, Tensor};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, Pixel};
use serde::{Deserialize, Serialize};
pub const IMAGE_PLACEHOLDER: &str = "<|VLLM-RS-IMAGE|>";
pub const PLACEHOLDER: &str = "<|VLLM-RS-PLACEHOLDER|>";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageData {
    pub raw: Vec<u8>,
    pub shape: Vec<usize>,
    pub patches: Vec<(usize, usize)>,
}

impl ImageData {
    pub fn to_tensor_f32(&self, device: &Device) -> Result<Tensor> {
        let floats: &[f32] = bytemuck::cast_slice(&self.raw);
        Tensor::from_slice(floats, self.shape.clone(), device)
    }
}
// load from url
pub fn load_image_from_url(url: &str) -> Result<DynamicImage> {
    let bytes = reqwest::blocking::get(url)
        .map_err(candle_core::Error::wrap)?
        .bytes()
        .map_err(candle_core::Error::wrap)?;
    let img = image::load_from_memory(&bytes).map_err(candle_core::Error::wrap)?;
    Ok(img)
}

// load from "data:image/jpeg;base64,XXXXX"
pub fn load_image_from_base64(data: &str) -> Result<DynamicImage> {
    use base64::prelude::{Engine as _, BASE64_STANDARD};
    let base64_part = data.split(",").last().unwrap_or(data);
    let bytes = BASE64_STANDARD
        .decode(base64_part)
        .map_err(candle_core::Error::wrap)?;
    let img = image::load_from_memory(&bytes).map_err(candle_core::Error::wrap)?;
    Ok(img)
}

pub fn get_tensor_raw_data(t: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape = t.dims().to_vec();
    let (storage, _) = t.storage_and_layout();
    let storage = match &*storage {
        Storage::Cpu(p) => p,
        _ => candle_core::bail!("t must be a cpu tensor"),
    };
    let bytes: Vec<u8> = match t.dtype() {
        DType::F32 => {
            let slice = storage.as_slice::<f32>()?;
            bytemuck::cast_slice(slice).to_vec()
        }
        _ => {
            return Err(candle_core::Error::Msg(
                "unsupported dtype for tensor_raw".into(),
            ));
        }
    };

    Ok((bytes, shape))
}

fn image_resize(
    image: &DynamicImage,
    mut height: usize,
    mut width: usize,
    max_height: usize,
    max_width: usize,
    patch_size: usize,
    filter: FilterType,
) -> DynamicImage {
    let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
    if ratio > 1. {
        height = (height as f64 / ratio).floor() as usize;
        width = (width as f64 / ratio).floor() as usize;
    }

    let num_height_tokens = (height - 1) / patch_size + 1;
    let num_width_tokens = (width - 1) / patch_size + 1;

    let resize_height = num_height_tokens * patch_size;
    let resize_width = num_width_tokens * patch_size;

    image.resize_exact(resize_width as u32, resize_height as u32, filter)
}

pub fn to_tensor(images: &Vec<DynamicImage>) -> Result<(Tensor, Vec<(usize, usize)>)> {
    let mut image_sizes = Vec::new();
    let mut pixel_values = Vec::new();
    for image in images.iter() {
        let (width, height) = image.dimensions();
        image_sizes.push((height as usize, width as usize));

        let rgb = image.to_rgb32f(); // HWC f32
        let (w, h) = rgb.dimensions();
        let data = rgb.into_raw(); // Vec<f32> in HWC layout, FAST

        // Build tensor from slice
        let t = Tensor::from_vec(data, (h as usize, w as usize, 3), &Device::Cpu)?;

        // Convert HWC → CHW without copying data manually
        let t = t.permute((2, 0, 1))?.contiguous()?; // [H, W, C] -> [C, H, W]

        // let t = image_to_pixels(image, &Device::Cpu)?;
        pixel_values.push(t.unsqueeze(0)?);
    }
    Ok((Tensor::cat(&pixel_values, 0)?, image_sizes))
}

pub fn normalize(
    images: &Vec<DynamicImage>,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
) -> Vec<DynamicImage> {
    images
        .into_iter()
        .map(|img| {
            let mut rgb = img.to_rgb32f();
            if let (Some(m), Some(s)) = (mean, std) {
                for px in rgb.pixels_mut() {
                    let c = px.channels_mut();
                    c[0] = (c[0] - m[0]) / s[0];
                    c[1] = (c[1] - m[1]) / s[1];
                    c[2] = (c[2] - m[2]) / s[2];
                }
            }

            DynamicImage::ImageRgb32F(rgb)
        })
        .collect()
}

pub fn rescale(images: &Vec<DynamicImage>, factor: Option<f32>) -> Vec<DynamicImage> {
    images
        .into_iter()
        .map(|img| {
            let mut rgb = img.to_rgb32f();

            if let Some(scale) = factor {
                for px in rgb.pixels_mut() {
                    let c = px.channels_mut();
                    c[0] *= scale;
                    c[1] *= scale;
                    c[2] *= scale;
                }
            }
            DynamicImage::ImageRgb32F(rgb)
        })
        .collect()
}
/// Preprocess input DynamicImages:
/// 1. Resize w/ custom rule (padding to uniform size)
/// 2. Optional rescale
/// 3. Optional normalize
pub fn preprocess_images(
    images: &Vec<DynamicImage>,
    max_height: usize,
    max_width: usize,
    patch_size: usize,
    scale_factor: Option<f32>,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
    abolute_resize: bool,
) -> Vec<DynamicImage> {
    // First pass: determine final height/width after custom resize for each image
    let mut resized_imgs: Vec<DynamicImage> = Vec::new();
    let mut padded_imgs: Vec<DynamicImage> = Vec::new();
    let filter = FilterType::Triangle;

    if abolute_resize {
        for img in images.iter() {
            let resized = img.resize_exact(max_width as u32, max_height as u32, filter);
            padded_imgs.push(resized);
        }
    } else {
        // resize in ratio and padding
        let mut final_h = 0;
        let mut final_w = 0;

        for img in images.iter() {
            let (h, w) = img.dimensions();
            let resized = image_resize(
                img, h as usize, w as usize, max_height, max_width, patch_size, filter,
            );
            final_h = final_h.max(resized.height());
            final_w = final_w.max(resized.width());
            resized_imgs.push(resized);
        }

        // Second pass: pad to final_h × final_w
        for img in resized_imgs {
            if img.height() == final_h && img.width() == final_w {
                padded_imgs.push(img);
            } else {
                // Pad with black (zeros)
                let mut canvas = DynamicImage::new_rgb8(final_w, final_h).to_rgb8();
                let rgb = img.to_rgb8();
                image::imageops::overlay(&mut canvas, &rgb, 0, 0);
                padded_imgs.push(DynamicImage::ImageRgb8(canvas));
            }
        }
    }

    normalize(&rescale(&padded_imgs, scale_factor), mean, std)
}

#[derive(Clone)]
pub struct ImageProcessConfig {
    image_start_token: Option<String>,
    image_token: String,
    image_break_token: Option<String>,
    image_end_token: String,
    pub spatial_merge_size: usize,
    pub do_normalize: Option<bool>,
    pub do_resize: Option<bool>,
    pub image_mean: Option<[f32; 3]>,
    pub image_std: Option<[f32; 3]>,
    pub max_height: usize,
    pub max_width: usize,
    pub patch_size: usize,
    pub temporal_patch_size: Option<usize>,
    pub abolute_resize: bool,
    pub scale_factor: Option<f32>,
    pub resampling: Option<usize>,
    pub model_type: ModelType,
}

impl ImageProcessConfig {
    pub fn default(
        image_start_token: Option<String>,
        image_token: String,
        image_break_token: Option<String>,
        image_end_token: String,
        spatial_merge_size: usize,
        temporal_patch_size: Option<usize>,
        patch_size: usize,
        image_size: usize,
        abolute_resize: bool,
    ) -> Self {
        ImageProcessConfig {
            image_start_token,
            image_token,
            image_break_token,
            image_end_token,
            spatial_merge_size,
            image_mean: None,
            image_std: None,
            do_resize: Some(true),
            do_normalize: Some(true),
            max_height: image_size,
            max_width: image_size,
            patch_size,
            abolute_resize,
            resampling: None,
            scale_factor: None,
            temporal_patch_size: temporal_patch_size,
            model_type: ModelType::Mistral3VL,
        }
    }
}
pub struct ImageProcessor {
    cfg: ImageProcessConfig,
}

impl ImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    pub fn new(cfg: &ImageProcessConfig) -> Self {
        Self { cfg: cfg.clone() }
    }

    fn preprocess(&self, images: &Vec<DynamicImage>) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let do_normalize = self.cfg.do_normalize.unwrap_or(false);
        let image_mean = if do_normalize {
            Some(self.cfg.image_mean.unwrap_or(Self::DEFAULT_MEAN))
        } else {
            None
        };

        let image_std = if do_normalize {
            Some(self.cfg.image_std.unwrap_or(Self::DEFAULT_STD))
        } else {
            None
        };

        let images = preprocess_images(
            images,
            self.cfg.max_height,
            self.cfg.max_width,
            self.cfg.patch_size,
            self.cfg.scale_factor,
            image_mean,
            image_std,
            self.cfg.abolute_resize,
        );
        to_tensor(&images)
    }

    pub fn process_inputs(
        &self,
        prompt: &mut String,
        images: &mut Vec<DynamicImage>,
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let (pixel_values, image_sizes_all) =
            self.preprocess(images).expect("Preprocessing failed");

        crate::log_info!(
            "pixel_values tensor shape {:?}, image_sizes_all {:?}",
            pixel_values.shape(),
            image_sizes_all
        );
        let mut image_sizes_all_iter = image_sizes_all.clone().into_iter();
        let mut replace_strings = Vec::new();
        while prompt.contains(IMAGE_PLACEHOLDER) {
            let (height, width) = image_sizes_all_iter.next().unwrap();
            let num_height_tokens =
                (height as usize) / (self.cfg.patch_size * self.cfg.spatial_merge_size);
            let num_width_tokens =
                (width as usize) / (self.cfg.patch_size * self.cfg.spatial_merge_size);

            crate::log_info!(
                "num_height_tokens {num_height_tokens}, num_width_tokens {num_width_tokens}"
            );
            let mut replace_tokens = vec![
                [
                    vec![self.cfg.image_token.clone(); num_width_tokens],
                    if self.cfg.image_break_token.is_some() {
                        vec![self.cfg.image_break_token.as_ref().unwrap().clone()]
                    } else {
                        vec![]
                    }
                ]
                .concat();
                num_height_tokens
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

            if self.cfg.image_break_token.is_some() {
                *replace_tokens.last_mut().unwrap() = self.cfg.image_end_token.clone();
            } else {
                replace_tokens.push(self.cfg.image_end_token.clone());
            }

            if let Some(start_token) = &self.cfg.image_start_token {
                let mut vec_final = vec![start_token.clone()];
                vec_final.extend(replace_tokens);
                replace_tokens = vec_final.clone();
            }
            replace_strings.push(replace_tokens.join(""));
            *prompt = prompt.replace(IMAGE_PLACEHOLDER, PLACEHOLDER);
        }

        while prompt.contains(PLACEHOLDER) {
            let replace_str = replace_strings.pop().unwrap();
            *prompt = prompt.replace(PLACEHOLDER, &replace_str);
        }

        Ok((pixel_values, image_sizes_all))
    }
}

pub fn get_image_config(
    model_type: ModelType,
    config: &Config,
) -> Result<Option<ImageProcessConfig>> {
    let img_cfg = match model_type {
        ModelType::Mistral3VL => {
            use crate::models::mistral3_vl::Mistral3Config;
            assert!(
                config.extra_config_json.is_some(),
                "Multimodel missing vision config!"
            );
            let cfg: Mistral3Config =
                serde_json::from_str(config.extra_config_json.as_ref().unwrap())
                    .map_err(candle_core::Error::wrap)?;

            let mut img_cfg = ImageProcessConfig::default(
                None,
                "[IMG]".to_string(),
                Some("[IMG_BREAK]".to_string()),
                "[IMG_END]".to_string(),
                cfg.spatial_merge_size,
                None,
                cfg.vision_config.patch_size,
                cfg.vision_config.image_size,
                false,
            );
            img_cfg.model_type = ModelType::Mistral3VL;
            Some(img_cfg)
        }
        ModelType::Gemma3 => {
            use crate::models::gemma3::config::Gemma3Config;
            assert!(
                config.extra_config_json.is_some(),
                "Multimodel missing vision config!"
            );
            let cfg: Gemma3Config =
                serde_json::from_str(config.extra_config_json.as_ref().unwrap())
                    .map_err(candle_core::Error::wrap)?;

            let mut img_cfg = ImageProcessConfig::default(
                Some("<start_of_image>".to_string()),
                "<image_soft_token>".to_string(),
                None,
                "<end_of_image>".to_string(),
                4,
                None,
                cfg.vision_config.patch_size,
                cfg.vision_config.image_size,
                true,
            );
            img_cfg.model_type = ModelType::Gemma3;
            img_cfg.scale_factor = Some(0.003921567);
            img_cfg.image_mean = Some([0.5, 0.5, 0.5]);
            img_cfg.image_std = Some([0.5, 0.5, 0.5]);
            Some(img_cfg)
        }
        ModelType::Qwen3VL => {
            use crate::models::qwen3_vl::config::Qwen3VLConfig;
            assert!(
                config.extra_config_json.is_some(),
                "Multimodel missing vision config!"
            );
            let cfg: Qwen3VLConfig =
                serde_json::from_str(config.extra_config_json.as_ref().unwrap())
                    .map_err(candle_core::Error::wrap)?;

            let mut img_cfg = ImageProcessConfig::default(
                Some("<|vision_start|>".to_string()),
                "<|image_pad|>".to_string(),
                None,
                "<|vision_end|>".to_string(),
                cfg.vision_config.spatial_merge_size,
                Some(cfg.vision_config.temporal_patch_size),
                cfg.vision_config.patch_size,
                1536,
                false,
            );
            img_cfg.model_type = ModelType::Qwen3VL;
            img_cfg.image_mean = Some([0.5, 0.5, 0.5]);
            img_cfg.image_std = Some([0.5, 0.5, 0.5]);
            Some(img_cfg)
        }
        _ => None,
    };
    Ok(img_cfg)
}

#[allow(dead_code)]
pub trait ToFilter {
    fn to_filter(self) -> Result<FilterType>;
}

impl ToFilter for Option<usize> {
    fn to_filter(self) -> Result<FilterType> {
        match self {
            Some(0) => Ok(FilterType::Nearest),
            Some(1) => Ok(FilterType::Lanczos3),
            Some(2) | None => Ok(FilterType::Triangle), // BiLinear
            Some(3) => Ok(FilterType::CatmullRom),      // BiCubic
            Some(4) => Ok(FilterType::Nearest),
            Some(x) => candle_core::bail!("Filter number {x} not supported"),
        }
    }
}
