# MTP Speculative Decoding

Multi-Token Prediction (MTP) is a speculative decoding technique that uses
lightweight predictor heads built into the model checkpoint to draft multiple
tokens per decode step. The main model then verifies the drafts in a single
batched forward pass, accepting tokens greedily until the first mismatch.

## How it works

```
Decode step N:
  1. Main model forward  → logits + hidden state
  2. Sample base token from logits
  3. MTP predictor drafts K tokens autoregressively
     (each step: embed(token) ⊕ hidden → FC → decoder layer → norm → lm_head → argmax)
  4. Verify: run main model on [base, draft_0, ..., draft_{K-1}]
  5. Greedy-accept matching prefix; take bonus token at first mismatch
  → emit 1..K+1 tokens in one engine step
```

### Key Components

1. **Embedding Layer** — converts token IDs to embeddings (shared with main model)
2. **Pre-FC Norms** — RMSNorm on both the embedding and the hidden state
3. **FC Projection** — linear `[hidden*2 → hidden]` fusing embedding + hidden
4. **Decoder Layer(s)** — standard attention + MLP (or MoE) block, cycled per step
5. **Final Norm** — RMSNorm before lm_head projection
6. **LM Head** — shared with main model or separate per-MTP weights

## Usage

### Rust CLI

```bash
vllm-rs --m Qwen/Qwen3.5-122B-A10B --d 0,1,2,3 \
    --ui-server --prefix-cache --mtp-num-tokens 3
```

### Python API

```python
from vllm_rs import EngineConfig

config = EngineConfig(
    model_id="Qwen/Qwen3.5-122B-A10B",
    mtp_num_tokens=3,
)
```

### Rust Builder API

```rust
let engine = EngineBuilder::new(ModelRepo::ModelID(("Qwen/Qwen3.5-122B-A10B", None)))
    .with_mtp_num_tokens(3)
    .build()?;
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mtp-num-tokens` | `0` | Number of speculative draft tokens (0 = disabled) |
| `mtp_num_hidden_layers` | from config | MTP decoder layers in the checkpoint (Qwen3.5) |
| `num_nextn_predict_layers` | from config | MTP decoder layers (Qwen3Next) |

## Supported Models

- Qwen3.5 Dense (27B) — `mtp_num_hidden_layers: 1`
- Qwen3.5 MoE (35B, 122B-A10B, 397B) — `mtp_num_hidden_layers: 1`, MoE experts in MTP layer
- Qwen3-Next variants — `num_nextn_predict_layers`

## Performance Notes

- Start with `--mtp-num-tokens 1` and increase; diminishing returns beyond 3-4.
- MTP is most effective at low batch sizes (latency-bound workloads).
- At high QPS / large batches, the verification overhead may reduce gains.
- MTP is automatically disabled when `--mtp-num-tokens 0` (default).
- Combine with `--prefix-cache` for best multi-turn performance.
- CUDA graph replay is used for the base decode step; the MTP draft and
  verify passes currently run without graph capture.

## Architecture Details

The MTP predictor is a lightweight module attached to the main transformer:

```
[Main Model Hidden State] ──┐
                             ├─ concat ─→ FC ─→ Decoder Layer ─→ Norm ─→ LM Head ─→ draft token
[Embedding(last_token)]  ───┘
```

For MoE models (e.g. Qwen3.5-122B-A10B), the MTP decoder layer itself
contains a full MoE block with the same expert count as the main model,
plus optional shared experts. The tensor layout matches:

```
mtp.pre_fc_norm_embedding.weight   [hidden_size]
mtp.pre_fc_norm_hidden.weight      [hidden_size]
mtp.fc.weight                      [hidden_size, hidden_size*2]
mtp.layers.0.self_attn.*           (attention weights)
mtp.layers.0.mlp.gate.weight       [num_experts, hidden_size]
mtp.layers.0.mlp.experts.*         (per-expert gate/up/down)
mtp.layers.0.mlp.shared_expert.*   (shared expert weights)
mtp.layers.0.mlp.shared_expert_gate.weight
mtp.norm.weight                    [hidden_size]
```
