use candle_core::{DType, Device, Result, Tensor, D};
pub fn get_attention_casual_mask(
    device: &Device,
    dtype: DType,
    tgt_len: usize,
    positions: &Tensor,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let vec_positions = positions.to_vec1::<i64>()?;

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
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((1, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}
