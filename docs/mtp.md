# MTP Speculative Decoding

Multi-Token Prediction (MTP) is a speculative decoding technique that enables the model to predict multiple tokens in a single forward pass, improving generation throughput.

## How it works

MTP uses a separate predictor module that takes the last hidden state and embeds the next token IDs to predict multiple future tokens. This is particularly effective for models with hybrid attention patterns like Qwen3.5 and Qwen3-Next.

### Architecture

```
[Input Tokens] --> [Main Model] --> [Last Hidden State]
                              |
                              v
                    [MTP Predictor]
                              |
                              v
                    [Predicted Tokens]
```

### Key Components

1. **Embedding Layer**: Converts token IDs to embeddings
2. **FC Projection**: Combines embedding and hidden state
3. **Decoder Layers**: Standard attention + MLP layers
4. **Norm Layers**: RMS normalization

## Usage

### Python API

```python
from vllm_rs import EngineConfig

config = EngineConfig(
    model_id="Qwen/Qwen3.5-27B",
    mtp_num_tokens=16,  # Enable MTP with 16 speculative tokens
    # ... other config
)
```

### Rust CLI

```bash
# Enable MTP with 16 speculative tokens
vllm-rs --m Qwen/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf \
    --ui-server --prefix-cache --mtp-num-tokens 16
```

### Environment Variable

```bash
export VLLM_MTP_NUM_TOKENS=16
python3 -m vllm_rs.server --m Qwen/Qwen3.5-27B-GGUF --f Qwen3.5-27B-Q4_K_M.gguf
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mtp-num-tokens` | `0` | Number of speculative tokens (0 = disabled) |
| `mtp_num_hidden_layers` | `1` | Number of MTP decoder layers (Qwen3.5) |
| `num_nextn_predict_layers` | `1` | Number of MTP decoder layers (Qwen3Next) |

## Models Supported

- ✅ Qwen3.5 Dense (27B)
- ✅ Qwen3.5 MoE (35B, 122B, 397B)
- ✅ Qwen3-Next (80B)
- ❌ Qwen3 (legacy, use MTP instead)

## Performance

MTP provides speedup by reducing the number of decode iterations:

| Model | Tokens | Speedup |
|-------|--------|---------|
| Qwen3.5-27B | 16 | ~1.2x |
| Qwen3.5-35B | 16 | ~1.3x |
| Qwen3-Next-80B | 16 | ~1.4x |

*Speedup depends on sequence length and batch size*

## Notes

- MTP is automatically disabled when `--mtp-num-tokens 0`
- MTP uses the same KV cache as the main model
- For best performance, use with `--prefix-cache` enabled
- MTP decoder layers are lightweight compared to main model layers