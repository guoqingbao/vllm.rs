use candle_core::{Device, IndexOp, Result, Tensor};
use image::{DynamicImage, GenericImageView};

use crate::utils::image::{normalize, to_tensor, ImageProcessConfig, ToFilter};

/// Replace first occurrence helper (unchanged)
fn replace_first_occurrence(text: &str, from: &str, to: &str) -> String {
    if let Some(pos) = text.find(from) {
        let mut s = text.to_string();
        s.replace_range(pos..pos + from.len(), to);
        s
    } else {
        text.to_string()
    }
}

/// Find continuous sequences of a value (kept for downstream token logic)
pub fn find_sequences(nums: &[u32], needle: u32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;

    for (i, &v) in nums.iter().enumerate() {
        if v == needle {
            start.get_or_insert(i);
        } else if let Some(s) = start.take() {
            out.push((s, i));
        }
    }

    if let Some(s) = start {
        out.push((s, nums.len()));
    }

    out
}

/// Qwen3-VL Image + Prompt Processor
#[derive(Clone)]
pub struct Qwen3VLImageProcessor {
    pub cfg: ImageProcessConfig,

    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl Qwen3VLImageProcessor {
    #[allow(dead_code)]
    fn default(cfg: ImageProcessConfig) -> Self {
        Self {
            cfg,
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 256 * 256,
            max_pixels: 1536 * 1536,
        }
    }
}

impl Qwen3VLImageProcessor {
    pub const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    pub const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    pub const VISION_START: &str = "<|vision_start|>";
    pub const VISION_END: &str = "<|vision_end|>";
    pub const IMAGE_PAD: &str = "<|image_pad|>";
    pub const PLACEHOLDER: &str = "<|placeholder|>";

    /// Resize respecting patch constraints
    fn smart_resize(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let factor = self.patch_size * self.merge_size;

        let mut nh = (h as f64 / factor as f64).round() as usize * factor;
        let mut nw = (w as f64 / factor as f64).round() as usize * factor;

        let pixels = nh * nw;

        if pixels > self.max_pixels {
            let beta = (pixels as f64 / self.max_pixels as f64).sqrt();
            nh = ((nh as f64 / beta) as usize / factor) * factor;
            nw = ((nw as f64 / beta) as usize / factor) * factor;
        } else if pixels < self.min_pixels {
            let beta = (self.min_pixels as f64 / pixels as f64).sqrt();
            nh = ((nh as f64 * beta) as usize / factor) * factor;
            nw = ((nw as f64 * beta) as usize / factor) * factor;
        }

        Ok((nh, nw))
    }

    fn preprocess_inner(
        &self,
        image: &DynamicImage,
        target_hw: (u32, u32),
    ) -> Result<(Tensor, (u32, u32, u32))> {
        let (th, tw) = target_hw;

        let (nh, nw) = self.smart_resize(th as usize, tw as usize)?;

        let image = image
            .resize_exact(nw as u32, nh as u32, self.cfg.resampling.to_filter()?)
            .to_rgb8();

        let images = normalize(
            &vec![DynamicImage::ImageRgb8(image)],
            Some(self.cfg.image_mean.unwrap_or(Self::DEFAULT_MEAN)),
            Some(self.cfg.image_std.unwrap_or(Self::DEFAULT_STD)),
        );

        let (mut patches, _) = to_tensor(&images)?;

        if patches.dim(0)? == 1 {
            patches = patches.repeat((self.temporal_patch_size, 1, 1, 1))?;
        }

        let c = patches.dim(1)?;
        let grid_t = patches.dim(0)? / self.temporal_patch_size;
        let grid_h = nh / self.patch_size;
        let grid_w = nw / self.patch_size;

        patches = patches.reshape(&[
            grid_t,
            self.temporal_patch_size,
            c,
            grid_h / self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w / self.merge_size,
            self.merge_size,
            self.patch_size,
        ])?;

        patches = patches.permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?;

        let patches = patches.reshape((
            grid_t * grid_h * grid_w,
            c * self.temporal_patch_size * self.patch_size * self.patch_size,
        ))?;

        Ok((patches, (grid_t as u32, grid_h as u32, grid_w as u32)))
    }

    /// 🔹 Main entry: processes prompt + images together
    pub fn process_inputs(
        &self,
        prompt: &mut String,
        images: &[DynamicImage],
    ) -> Result<(Tensor, Tensor)> {
        let (max_w, max_h) = images
            .iter()
            .map(|i| i.dimensions())
            .fold((0, 0), |(mw, mh), (w, h)| (mw.max(w), mh.max(h)));

        let mut pixel_values = Vec::new();
        let mut grid_thw = Vec::new();

        for image in images {
            let (patches, (t, h, w)) = self.preprocess_inner(image, (max_h, max_w))?;

            pixel_values.push(patches);
            grid_thw.push(Tensor::new(&[t, h, w], &Device::Cpu)?);
        }

        let pixel_values = Tensor::stack(&pixel_values, 0)?;
        let grid_thw = Tensor::stack(&grid_thw, 0)?;

        // ===== Prompt expansion logic (preserved & fixed) =====
        let merge_len = self.merge_size * self.merge_size;
        let mut image_idx = 0;

        while prompt.contains(Self::IMAGE_PAD) {
            let grid = grid_thw.i(image_idx)?;
            let num_patches: usize =
                grid.to_vec1::<u32>()?.iter().product::<u32>() as usize / merge_len;

            *prompt = replace_first_occurrence(
                prompt,
                Self::IMAGE_PAD,
                &Self::PLACEHOLDER.repeat(num_patches),
            );

            image_idx += 1;
        }

        *prompt = prompt.replace(Self::PLACEHOLDER, Self::IMAGE_PAD);
        // ======================================================

        Ok((pixel_values, grid_thw))
    }
}
