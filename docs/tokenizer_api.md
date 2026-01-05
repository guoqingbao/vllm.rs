# Tokenizer API

The Tokenizer API provides direct access to the model's tokenizer for encoding/decoding text without running inference. This is useful for:

- Pre-computing token counts for cost estimation
- Validating inputs before sending to the model
- Debugging tokenization issues
- Building custom tooling around the tokenizer

## Endpoints

### POST /tokenize

Convert text or chat messages to token IDs.

**Request (plain text):**
```json
{
  "prompt": "Hello, world!"
}
```

**Request (chat messages - applies chat template):**
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ]
}
```

**Optional parameters:**
- `model`: Model name (optional, uses loaded model)
- `add_special_tokens`: Whether to add special tokens (default: `true`)

**Response:**
```json
{
  "tokens": [1, 2, 3, 4],
  "count": 4,
  "max_model_len": 4096
}
```

### POST /detokenize

Convert token IDs back to text.

**Request:**
```json
{
  "tokens": [1, 2, 3, 4]
}
```

**Optional parameters:**
- `model`: Model name (optional, uses loaded model)
- `skip_special_tokens`: Whether to skip special tokens in output (default: `true`)

**Response:**
```json
{
  "prompt": "Hello, world!"
}
```

## Examples

### Python

```python
import requests

# Tokenize text
response = requests.post("http://localhost:8000/tokenize", json={
    "prompt": "Hello, world!"
})
print(response.json())
# {"tokens": [9906, 11, 1917, 0], "count": 4, "max_model_len": 4096}

# Detokenize
response = requests.post("http://localhost:8000/detokenize", json={
    "tokens": [9906, 11, 1917, 0]
})
print(response.json())
# {"prompt": "Hello, world!"}
```

### cURL

```bash
# Tokenize
curl -X POST http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'

# Detokenize
curl -X POST http://localhost:8000/detokenize \
  -H "Content-Type: application/json" \
  -d '{"tokens": [9906, 11, 1917, 0]}'
```

## Demo Scripts

- **Python**: `example/tokenize.py` - Interactive demo with multiple examples
- **Rust**: `example/rust-demo-tokenize/` - Rust client demo

Run the Python demo:
```bash
# Start the server first
cargo run --release -- --m Qwen/Qwen2.5-0.5B-Instruct --server

# In another terminal
python example/tokenize.py
```
