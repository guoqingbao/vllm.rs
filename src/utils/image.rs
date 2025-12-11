use candle_core::{DType, Device, Result, Storage, Tensor};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, Pixel};
pub const IMAGE_PLACEHOLDER: &str = "<|VLLM-RS-IMAGE|>";
const PLACEHOLDER: &str = "<|VLLM-RS-PLACEHOLDER|>";

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
    // println!("tensor_raw shape {:?}, dtype {:?}", shape, t.dtype());
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

pub fn bytes_to_tensor_f32(bytes: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    Tensor::from_slice(floats, shape, device)
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
) -> Vec<DynamicImage> {
    // First pass: determine final height/width after custom resize for each image
    let mut resized_imgs: Vec<DynamicImage> = Vec::new();
    let filter = FilterType::Nearest;

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
    let mut padded_imgs: Vec<DynamicImage> = Vec::new();
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

    // Step 3: convert to f32, rescale + normalize if required
    padded_imgs
        .into_iter()
        .map(|img| {
            let mut rgb = img.to_rgb32f();

            if let Some(scale) = scale_factor {
                for px in rgb.pixels_mut() {
                    let c = px.channels_mut();
                    c[0] *= scale;
                    c[1] *= scale;
                    c[2] *= scale;
                }
            }

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

#[derive(Clone)]
pub struct ImageProcessConfig {
    image_token: String,
    image_break_token: String,
    image_end_token: String,
    spatial_merge_size: usize,
    do_normalize: Option<bool>,
    image_mean: Option<[f32; 3]>,
    image_std: Option<[f32; 3]>,
    max_height: usize,
    max_width: usize,
    patch_size: usize,
    scale_factor: Option<f32>,
}

impl ImageProcessConfig {
    pub fn default(
        image_token: String,
        image_break_token: String,
        image_end_token: String,
        spatial_merge_size: usize,
        patch_size: usize,
        image_size: usize,
    ) -> Self {
        ImageProcessConfig {
            image_token,
            image_break_token,
            image_end_token,
            spatial_merge_size,
            image_mean: None,
            image_std: None,
            do_normalize: Some(true),
            max_height: image_size,
            max_width: image_size,
            patch_size,
            scale_factor: None,
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

        let mut images = preprocess_images(
            images,
            self.cfg.max_height,
            self.cfg.max_width,
            self.cfg.patch_size,
            self.cfg.scale_factor,
            image_mean,
            image_std,
        );
        let mut pixel_values = Vec::new();
        let mut image_sizes = Vec::new();

        for image in images.iter_mut() {
            let (width, height) = image.dimensions();
            image_sizes.push((height as usize, width as usize));

            let rgb = image.to_rgb32f(); // HWC f32
            let (w, h) = rgb.dimensions();
            let data = rgb.into_raw(); // Vec<f32> in HWC layout, FAST

            // Build tensor from slice
            let t = Tensor::from_vec(data, (h as usize, w as usize, 3), &Device::Cpu)?;

            // Convert HWC → CHW without copying data manually
            let t = t.permute((2, 0, 1))?.contiguous()?; // [H, W, C] -> [C, H, W]

            pixel_values.push(t.unsqueeze(0)?);
        }

        Ok((Tensor::cat(&pixel_values, 0)?, image_sizes))
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
                    vec![self.cfg.image_break_token.clone()],
                ]
                .concat();
                num_height_tokens
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

            *replace_tokens.last_mut().unwrap() = self.cfg.image_end_token.clone();

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
