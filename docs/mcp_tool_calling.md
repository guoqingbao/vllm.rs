# MCP Integration and Tool Calling

vLLM.rs supports **Model Context Protocol (MCP)** integration and **tool calling**, enabling LLM models to interact with external tools and services during inference. This feature supports both streaming and non-streaming modes with automatic tool execution, KV cache preservation for multi-turn conversations, and robust error handling.

## Overview

### What is Tool Calling?

Tool calling allows LLMs to invoke external functions/tools to gather information or perform actions. vLLM.rs supports:

- **OpenAI-compatible API**: Full support for `tools` and `tool_choice` parameters in `/v1/chat/completions`
- **Multiple tool formats**: Auto-detection for Qwen, Llama/Mistral, and Generic formats
- **Tool call parsing**: Robust XML-based parser for extracting tool calls from model output

### What is MCP?

The **Model Context Protocol (MCP)** is a standardized protocol for connecting LLMs to external tools and services. vLLM.rs supports:

- **Stdio transport**: Connect to MCP servers via command-line processes
- **Multi-server support**: Configure multiple MCP servers with automatic tool name prefixing
- **Automatic tool injection**: MCP tools are injected into the prompt when configured
- **Internal tool execution**: Tool calls are executed automatically with 60-second timeout

---

## Quick Start

### 1. Single MCP Server (CLI)

```bash
# CUDA
./run.sh --release --features cuda --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --context-cache \
  --mcp-command npx \
  --mcp-args=-y,@modelcontextprotocol/server-filesystem,~/

# Metal/macOS
cargo run --release --features metal -- --m Qwen/Qwen3-8B-GGUF --f Qwen3-8B-Q4_K_M.gguf --ui-server --context-cache \
  --mcp-command npx \
  --mcp-args=-y,@modelcontextprotocol/server-filesystem,~/
```

### 2. Multiple MCP Servers (Config File)

Create `mcp.json`:

```json
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "~/"
            ]
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
            }
        }
    }
}
```

Start the server:

```bash
# pip install vllm_rs
python3 -m vllm_rs.server --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --context-cache --mcp-config ./mcp.json
```

or Rust:

```bash
# CUDA
./build.sh --release --features cuda,nccl,graph,flash-attn,flash-context

target/release/vllm-rs --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --context-cache \
  --mcp-config ./mcp.json

# Metal/macOS
cargo run --release --features metal -- --m Qwen/Qwen3-8B-GGUF --f Qwen3-8B-Q4_K_M.gguf --ui-server --context-cache \
  --mcp-config ./mcp.json
```

---

## Tool Calling Modes

vLLM.rs supports two distinct modes for handling tool calls:

### Mode 1: Internal MCP Execution (Automatic)

When MCP is configured and no user tools are provided, vLLM.rs automatically:
1. Injects MCP tools into the prompt
2. Detects tool calls in model output
3. Executes tools via MCP servers
4. Resumes generation with tool results

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

# No tools provided → MCP tools are auto-injected and executed
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List files in current directory (./)"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Mode 2: External Tool Handling (User-provided)

When user provides their own tools, vLLM.rs:
1. Uses user's tool definitions
2. Detects tool calls in model output
3. **Finishes the stream** at `</tool_call>` for client-side execution

```python
# User provides tools → stream finishes at tool call for external handling
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }],
    stream=True
)

# Stream will have finish_reason when tool call detected
# Parse tool calls and execute externally
```

---

## Behavior Matrix

| Has Request Tools | MCP Configured | Tool Mode | Streaming Behavior |
|-------------------|----------------|-----------|-------------------|
| ❌ | ❌ | `None` | Normal streaming, `</tool_call>` treated as text |
| ✅ | ❌ | `Some(false)` | Stream finishes at `</tool_call>` for external handling |
| ❌ | ✅ | `Some(true)` | Stream pauses, MCP executes (60s timeout), stream resumes |
| ✅ | ✅ | `Some(false)` | User tools take precedence, external handling |

---

## Streaming Tool Execution

When MCP is configured and a tool call is detected during streaming:

```
User Request
     │
     ▼
┌─────────────────┐
│ Inject MCP tools│
│ Start streaming │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model generates │
│ <tool_call>...  │
│ </tool_call>    │
└────────┬────────┘
         │ ToolCallPause
         ▼
┌─────────────────┐
│ Execute MCP tool│
│ (60s timeout)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Resume with     │
│ cached KV +     │
│ tool result     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stream final    │
│ response        │
└─────────────────┘
```

### Key Features

- **Token buffering**: Tool call XML is buffered and NOT streamed to client
- **KV cache preservation**: Context is cached to avoid re-processing prompt tokens on resume
- **Multi-turn support**: Multiple sequential tool calls work seamlessly
- **60-second timeout**: Each tool call has a configurable timeout to prevent hanging
- **Client disconnect detection**: Tool execution aborts if client disconnects

---

## Configuration Options

### CLI Options

| Option | Description |
|--------|-------------|
| `--mcp-command` | Path to single MCP server executable |
| `--mcp-args` | Comma-separated arguments for MCP server |
| `--mcp-config` | Path to JSON config file for multiple servers |

### Config File Format

```json
{
    "mcpServers": {
        "server_name": {
            "command": "path/to/executable",
            "args": ["arg1", "arg2"],
            "env": {
                "ENV_VAR": "value"
            }
        }
    }
}
```

### Tool Name Prefixing

When using multiple MCP servers, tool names are prefixed with the server name to avoid conflicts:



- Server `filesystem` with tool `list_directory` → `filesystem_list_directory`
- Server `github` with tool `search_repos` → `github_search_repos`

---

## API Reference

### Chat Completion with Tools

```python
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Your message"}
    ],
    tools=[  # Optional: provide your own tools
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...}
                }
            }
        }
    ],
    tool_choice="auto",  # "auto", "none", or specific function
    stream=True  # or False
)
```

### Tool Choice Options

| Value | Description |
|-------|-------------|
| `"auto"` | Model decides when to call tools |
| `"none"` | Disable tool calling |
| `{"type": "function", "function": {"name": "..."}}` | Force specific tool |

---

## Supported Tool Formats

vLLM.rs automatically detects the appropriate tool format based on the model:

| Model Family | Tool Format |
|--------------|-------------|
| Qwen, Qwen2, Qwen3 | Qwen format |
| Llama, Llama2, Llama3 | Llama format |
| Mistral | Llama format |
| Others | Generic format |

---

## Examples

### Example 1: Filesystem Operations

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Please list all Python files in the current directory (./) and show me the content of the main.py file if it exists."}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Example 2: Multi-turn Tool Calls

```python
messages = [
    {"role": "user", "content": "First, list files in ./, then read the first .txt file you find."}
]

response = client.chat.completions.create(
    model="default",
    messages=messages,
    stream=True
)

# The model will automatically make multiple tool calls:
# 1. list_directory for ./
# 2. read_file for the first .txt file
```

---

## Troubleshooting

### Common Issues

**Tool calls not being executed:**
- Ensure MCP server is properly configured and accessible
- Check server logs for MCP connection errors
- Verify tool names match expected format

**Timeout errors:**
- Tool execution has a 60-second timeout
- For long-running operations, consider breaking into smaller steps

**Tool results not appearing:**
- Ensure the MCP server returns valid JSON responses
- Check for errors in MCP server output

### Debug Logging

The server logs detailed information about tool execution:

```
[Seq 0] Executing 1 tool call(s) via MCP (with 60s timeout)
Executing tool call: filesystem_list_directory with args {"path": "."}
Tool 'filesystem_list_directory' completed in 0.01s
```

---

## Performance Notes

- **Tool list refresh**: MCP tools are refreshed automatically every 60 seconds
- **KV Cache**: Context is cached between tool calls for efficient multi-turn conversations
- **Timeout**: Each tool call has a 60-second timeout to prevent hanging
- **Client disconnect**: If client disconnects during tool execution, the request is aborted

---

## Popular MCP Servers

| Server | Package | Description |
|--------|---------|-------------|
| Filesystem | `@modelcontextprotocol/server-filesystem` | Read/write files, list directories |
| GitHub | `@modelcontextprotocol/server-github` | GitHub API operations |
| Brave Search | `@anthropic/mcp-brave-search` | Web search via Brave |
| Puppeteer | `@anthropic/mcp-puppeteer` | Browser automation |

Install with npx:
```bash
npx -y @modelcontextprotocol/server-filesystem ~/
```

---

## See Also

- [Context Cache](./context-cache.md) - How KV caching works for multi-turn conversations
- [Get Started](./get_started.md) - Basic setup and installation
