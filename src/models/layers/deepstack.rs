use crate::models::layers::others::masked_fill;
use candle_core::{DType, Result, Tensor, D};
pub trait ApplyDeepStack {
    fn apply_deep_stack(&self, visual_pos_masks: &Tensor, visual_embeds: &Tensor)
        -> Result<Tensor>;
}

impl ApplyDeepStack for Tensor {
    fn apply_deep_stack(
        &self,
        visual_pos_masks: &Tensor,
        visual_embeds: &Tensor,
    ) -> Result<Tensor> {
        deepstack_process(&self, visual_pos_masks, visual_embeds)
    }
}

// Reference: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-core/src/vision_models/qwen3_vl_moe/text.rs#698
fn deepstack_process(
    hidden_states: &Tensor,
    visual_pos_masks: &Tensor,
    visual_embeds: &Tensor,
) -> Result<Tensor> {
    let device = hidden_states.device();
    let dtype = hidden_states.dtype();

    let mask = visual_pos_masks.to_device(device)?.to_dtype(DType::F32)?;
    let mask_flat = mask.flatten_all()?;

    let masked_count = mask_flat.sum_all()?.to_scalar::<f32>()? as usize;
    let visual_embeds = visual_embeds.to_device(device)?.to_dtype(dtype)?;

    if masked_count == 0 {
        if visual_embeds.dim(0)? != 0 {
            candle_core::bail!(
                "DeepStack visual embeds ({}) provided but mask is empty",
                visual_embeds.dim(0)?
            );
        }
        return Ok(hidden_states.clone());
    }

    if visual_embeds.dim(0)? != masked_count {
        candle_core::bail!(
            "Mismatch between DeepStack visual embeds ({}) and mask positions ({})",
            visual_embeds.dim(0)?,
            masked_count
        );
    }

    let (batch, seq, hidden) = hidden_states.dims3()?;
    let total_positions = batch * seq;
    let mut hidden_flat = hidden_states.reshape((total_positions, hidden))?;

    let prefix = mask_flat.cumsum(0)?;
    let rank = (prefix - &mask_flat)?.mul(&mask_flat)?;
    let rank_u32 = rank.to_dtype(DType::U32)?;

    let positions = Tensor::arange(0u32, total_positions as u32, device)?;
    let positions_f32 = positions.to_dtype(DType::F32)?;
    let masked_positions = positions_f32.mul(&mask_flat)?;

    let mut position_per_rank = Tensor::zeros((masked_count,), DType::F32, device)?;
    position_per_rank = position_per_rank.scatter_add(&rank_u32, &masked_positions, 0)?;
    let position_per_rank = position_per_rank.to_dtype(DType::U32)?;

    let linear_index = position_per_rank.unsqueeze(1)?.repeat((1, hidden))?;

    hidden_flat = hidden_flat.scatter_add(&linear_index, &visual_embeds, 0)?;
    hidden_flat.reshape((batch, seq, hidden))
}

pub trait ApplyRopeIndex {
    fn apply_rope_index(
        &self,
        image_grid_thw: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        spatial_merge_size: usize,
        image_token_id: u32,
        vision_start_token_id: u32,
        vision_end_token_id: u32,
    ) -> Result<(Tensor, Tensor)>;
}

impl ApplyRopeIndex for Tensor {
    fn apply_rope_index(
        &self,
        image_grid_thw: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        spatial_merge_size: usize,
        image_token_id: u32,
        vision_start_token_id: u32,
        vision_end_token_id: u32,
    ) -> Result<(Tensor, Tensor)> {
        get_rope_index(
            &self,
            image_grid_thw,
            attention_mask,
            spatial_merge_size,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        )
    }
}

// Reference: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-core/src/vision_models/qwen3_vl_moe/mod.rs#L82
fn get_rope_index(
    input_ids: &Tensor,
    image_grid_thw: Option<&Tensor>,
    attention_mask: Option<&Tensor>,
    spatial_merge_size: usize,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
) -> Result<(Tensor, Tensor)> {
    if image_grid_thw.is_some() {
        let batch = input_ids.dim(0)?;
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device().clone();

        let attention_mask_tensor = match attention_mask {
            Some(mask) => mask.clone(),
            None => Tensor::ones((batch, seq_len), DType::F32, &device)?,
        };
        let attention_mask_vec = attention_mask_tensor.to_vec2::<f32>()?;
        let input_ids_vec = input_ids.to_vec2::<u32>()?;

        let image_grid_data = if let Some(grid) = image_grid_thw {
            let raw = grid.to_vec2::<u32>()?;
            let mut data = Vec::with_capacity(raw.len());
            for row in raw {
                if row.len() != 3 {
                    candle_core::bail!("image_grid_thw entries must have length 3");
                }
                data.push([row[0], row[1], row[2]]);
            }
            Some(data)
        } else {
            None
        };

        let mut image_index = 0usize;
        let merge_size = spatial_merge_size as u32;

        let mut position_ids_data = vec![vec![vec![1i64; seq_len]; batch]; 3];
        let mut mrope_position_deltas = Vec::with_capacity(batch);

        for batch_idx in 0..batch {
            let mask_row = &attention_mask_vec[batch_idx];
            let input_row = &input_ids_vec[batch_idx];

            let mut valid_indices = Vec::new();
            let mut filtered_tokens = Vec::new();
            for (idx, (&token, &mask_val)) in input_row.iter().zip(mask_row.iter()).enumerate() {
                if mask_val != 0.0 {
                    valid_indices.push(idx);
                    filtered_tokens.push(token);
                }
            }

            let mut positions_for_valid: Vec<[i64; 3]> = Vec::with_capacity(valid_indices.len());
            let mut max_position_value: Option<i64> = None;

            let mut spans = Vec::new();
            let mut span_idx = 0usize;
            while span_idx < filtered_tokens.len() {
                if filtered_tokens[span_idx] == vision_start_token_id {
                    let mut end_idx = span_idx + 1;
                    while end_idx < filtered_tokens.len()
                        && filtered_tokens[end_idx] != vision_end_token_id
                    {
                        end_idx += 1;
                    }
                    if end_idx == filtered_tokens.len() {
                        candle_core::bail!(
                            "vision_start_token_id without matching vision_end_token_id"
                        );
                    }
                    spans.push((span_idx, end_idx));
                    span_idx = end_idx + 1;
                } else {
                    span_idx += 1;
                }
            }

            let mut max_last_llm_pos_ids: Option<i64> = None;
            let mut cursor = 0usize;

            for (start_idx, end_idx) in spans {
                if start_idx + 1 > end_idx {
                    continue;
                }

                let placeholder_start = filtered_tokens[start_idx + 1..end_idx]
                    .iter()
                    .enumerate()
                    .find_map(|(offset, &tok)| {
                        (tok == image_token_id).then_some(offset + start_idx + 1)
                    });
                let placeholder_start = match placeholder_start {
                    Some(pos) => pos,
                    None => {
                        candle_core::bail!("vision span missing image/video placeholder tokens");
                    }
                };

                let text_len = placeholder_start.saturating_sub(cursor);
                let st_idx = max_last_llm_pos_ids.unwrap_or(0);
                for offset in 0..text_len {
                    let pos_val = st_idx + offset as i64;
                    positions_for_valid.push([pos_val, pos_val, pos_val]);
                    max_position_value = Some(match max_position_value {
                        Some(current) => current.max(pos_val),
                        None => pos_val,
                    });
                }

                let placeholder_token_id = filtered_tokens[placeholder_start];
                let placeholder_slice = &filtered_tokens[placeholder_start..end_idx];
                if placeholder_slice.is_empty() {
                    candle_core::bail!("vision span placeholder slice is empty");
                }
                if !placeholder_slice
                    .iter()
                    .all(|&tok| tok == placeholder_token_id)
                {
                    candle_core::bail!("Mixed placeholder tokens found within a vision span");
                }
                let placeholder_len = placeholder_slice.len();

                let (grid_t, grid_h, grid_w) = match placeholder_token_id {
                    id if id == image_token_id => {
                        let Some(ref img_grid) = image_grid_data else {
                            candle_core::bail!("image_grid_thw required for image placeholders");
                        };
                        if image_index >= img_grid.len() {
                            candle_core::bail!(
                                "Not enough image_grid_thw entries for placeholders"
                            );
                        }
                        let grid = img_grid[image_index];
                        image_index += 1;
                        if merge_size == 0 || grid[1] % merge_size != 0 || grid[2] % merge_size != 0
                        {
                            candle_core::bail!(
                                "image grid dimensions must be divisible by spatial_merge_size"
                            );
                        }
                        (
                            grid[0] as usize,
                            (grid[1] / merge_size) as usize,
                            (grid[2] / merge_size) as usize,
                        )
                    }
                    other => {
                        candle_core::bail!("Unexpected placeholder token id {other}");
                    }
                };

                if grid_t == 0 || grid_h == 0 || grid_w == 0 {
                    candle_core::bail!("Zero-sized grid encountered in vision span");
                }

                let expected_len = grid_t * grid_h * grid_w;
                if placeholder_len != expected_len {
                    candle_core::bail!(
                            "Placeholder token count {placeholder_len} does not match expected {expected_len}"
                        );
                }

                let base_offset = st_idx + text_len as i64;
                for t in 0..grid_t {
                    for h in 0..grid_h {
                        for w in 0..grid_w {
                            let t_pos = base_offset + t as i64;
                            let h_pos = base_offset + h as i64;
                            let w_pos = base_offset + w as i64;
                            positions_for_valid.push([t_pos, h_pos, w_pos]);
                            max_position_value = Some(match max_position_value {
                                Some(current) => current.max(t_pos).max(h_pos).max(w_pos),
                                None => t_pos.max(h_pos).max(w_pos),
                            });
                        }
                    }
                }

                let max_dim = std::cmp::max(grid_t, std::cmp::max(grid_h, grid_w)) as i64;
                max_last_llm_pos_ids = Some(base_offset + max_dim);
                cursor = placeholder_start + placeholder_len;
            }

            if cursor < filtered_tokens.len() {
                let text_len = filtered_tokens.len() - cursor;
                let st_idx = max_last_llm_pos_ids.unwrap_or(0);
                for offset in 0..text_len {
                    let pos_val = st_idx + offset as i64;
                    positions_for_valid.push([pos_val, pos_val, pos_val]);
                    max_position_value = Some(match max_position_value {
                        Some(current) => current.max(pos_val),
                        None => pos_val,
                    });
                }
            }

            if positions_for_valid.len() != valid_indices.len() {
                candle_core::bail!(
                    "Mismatch between computed positions ({}) and valid tokens ({})",
                    positions_for_valid.len(),
                    valid_indices.len()
                );
            }

            for (pos_idx, &seq_idx) in valid_indices.iter().enumerate() {
                let [p0, p1, p2] = positions_for_valid[pos_idx];
                position_ids_data[0][batch_idx][seq_idx] = p0;
                position_ids_data[1][batch_idx][seq_idx] = p1;
                position_ids_data[2][batch_idx][seq_idx] = p2;
            }

            let seq_total_len = input_row.len() as i64;
            let max_position_value = max_position_value.unwrap_or(0);
            mrope_position_deltas.push(max_position_value + 1 - seq_total_len);
        }

        let mut flat_positions = Vec::with_capacity(3 * batch * seq_len);
        for plane in position_ids_data.iter().take(3) {
            for row in plane.iter().take(batch) {
                flat_positions.extend_from_slice(row);
            }
        }
        let position_ids = Tensor::from_vec(flat_positions, (3, batch, seq_len), &device)?;
        let mrope_position_deltas = Tensor::from_vec(mrope_position_deltas, (batch, 1), &device)?;

        Ok((position_ids, mrope_position_deltas))
    } else if let Some(attention_mask) = attention_mask {
        let position_ids = (attention_mask.to_dtype(DType::F32)?.cumsum(D::Minus1)? - 1f64)?;
        let position_ids = masked_fill(&position_ids, &attention_mask.eq(0f64)?, 1i64)?;
        let position_ids = position_ids.unsqueeze(0)?.repeat((3, 1, 1))?;

        let max_position_ids = position_ids.max(0)?.max_keepdim(D::Minus1)?;
        let mrope_position_deltas =
            ((max_position_ids + 1.)? - attention_mask.dim(D::Minus1)? as f64)?;

        Ok((
            position_ids.to_dtype(DType::I64)?,
            mrope_position_deltas.to_dtype(DType::I64)?,
        ))
    } else {
        let position_ids = Tensor::arange(0i64, input_ids.dim(1)? as i64, input_ids.device())?
            .reshape((1, 1, ()))?
            .repeat((3, input_ids.dim(0)?, 1))?;
        let mrope_position_deltas =
            Tensor::zeros((input_ids.dim(0)?, 1), DType::I64, input_ids.device())?;

        Ok((position_ids, mrope_position_deltas))
    }
}
