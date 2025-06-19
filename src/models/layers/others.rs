use candle_core::Result;
use candle_nn::var_builder::Shard;
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, LayerNorm, RmsNorm};

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get_with_hints(size, "weight", Shard::default())?;
    Ok(RmsNorm::new(weight, eps))
}

pub fn layer_norm(size: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get_with_hints(size, "weight", Shard::default())?;
    if affine {
        Ok(LayerNorm::new(weight, vb.get(size, "bias")?, eps))
    } else {
        Ok(LayerNorm::new_no_bias(weight, eps))
    }
}

pub fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}
