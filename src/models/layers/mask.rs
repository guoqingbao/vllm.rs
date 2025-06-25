use candle_core::{DType, Device, Tensor};

#[cfg(feature = "cuda")] // If CUDA is enabled, we don't implement this function.
                         // The actual implementation would be embedded in the flash attention kernel.
pub fn get_attention_casual_mask(
    _: &Device,
    _: DType,
    _: usize,
    _: &Tensor,
    _: Option<usize>,
) -> Option<Tensor> {
    None
}

#[cfg(not(feature = "cuda"))]
pub fn get_attention_casual_mask(
    device: &Device,
    dtype: DType,
    tgt_len: usize,
    positions: &Tensor,
    sliding_window: Option<usize>,
) -> Option<Tensor> {
    if tgt_len <= 1 {
        return None;
    }
    let vec_positions = positions.to_vec1::<i64>().unwrap();
    let seqlen_offset = vec_positions[0] as usize;
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device).ok();
    let mask = if seqlen_offset > 0 && mask.is_some() {
        match Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device) {
            Ok(mask0) => Tensor::cat(&[&mask0, &mask.unwrap()], candle_core::D::Minus1).ok(),
            Err(_) => {
                return None;
            }
        }
    } else {
        mask
    };
    match mask {
        Some(m) => m
            .expand((1, 1, tgt_len, tgt_len + seqlen_offset))
            .unwrap()
            .to_dtype(dtype)
            .ok(),
        _ => {
            return None;
        }
    }
}
