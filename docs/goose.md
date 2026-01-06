# Goose + vLLM.rs (OpenAI-compatible endpoint)

This guide connects Goose (Rust `AI Agent`) directly to vLLM.rs using the built-in OpenAI-compatible `/v1/chat/completions` API. No proxy required.

```
Goose -> vLLM.rs (OpenAI-compatible)
```

## 1) Start vLLM.rs on port 8000

```bash
# Rust
./target/release/vllm-rs --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --server --prefix-cache
# Or
./run.sh --features cuda,nccl,graph,flash-attn,flash-context --release --w /path/Qwen3-30B-A3B-Instruct-2507 --d 0,1 --ui-server --prefix-cache
# Python
python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --server --prefix-cache
```

## 2) Configure Goose

### Download and install Goose: https://block.github.io/goose/docs/getting-started/installation/

```shell
# For non-UI system,
export GOOSE_DISABLE_KEYRING=1
```

Export empty API KEY

```shell
export VLLM_API_KEY="empty"
```


### Configure goose with `Custom Providers` and API key `empty`

```shell
goose configure

┌   goose-configure 
│
◇  What would you like to configure?
│  Custom Providers 
│
◇  What would you like to do?
│  Add A Custom Provider 
│
◇  What type of API is this?
│  OpenAI Compatible 
│
◇  What should we call this provider?
│  vllm-rs
│
◇  Provider API URL:
│  http://10.9.112.41:8000/v1/
│
◇  API key:
│  ▪▪▪▪▪
│
◇  Available models (separate with commas):
│  default
│
◇  Does this provider support streaming responses?
│  Yes 
│
◇  Does this provider require custom headers?
│  No 
│
└  Custom provider added: vllm-rs
└  Configuration saved successfully to /root/.config/goose/config.yaml
```

### Run `goose` at any folder to work with your local server

```shell
goose
```