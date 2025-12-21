# Tool Calling and MCP Support for vLLM.rs

This document describes how to use tool calling and the Model Context Protocol (MCP) features in vLLM.rs.

## Overview

vLLM.rs supports:
- **OpenAI-compatible Tool Calling**: Define tools and let the model decide when to call them
- **MCP Client Integration**: Connect to external MCP servers to use their tools
- **Streaming Tool Execution**: Automatic pause-and-resume for tool calls during streaming

## Tool Calling Modes

vLLM.rs supports two distinct modes for tool calling:

### Mode 1: External Tool Handling (User-Provided Tools)

When you provide tools via the `tools` parameter in your request, vLLM.rs outputs tool calls for **external handling**. You're responsible for executing the tools and continuing the conversation.

```python
# User provides tools → External handling
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{  # User-provided tools
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {...}
        }
    }]
)

# Model outputs tool_calls → You execute them externally
if response.choices[0].message.tool_calls:
    # Execute tool yourself
    result = my_weather_api(...)
    # Continue conversation with result
```

### Mode 2: Internal Tool Execution (MCP-Injected Tools)

When MCP servers are configured and you **don't** provide `tools` in your request, vLLM.rs:
1. Injects MCP tools into the system prompt
2. Detects tool calls in model output
3. **Automatically executes** tools via MCP
4. Resumes generation with tool results

```python
# No tools provided + MCP configured → Internal execution
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Read the README.md file"}],
    stream=True
    # No 'tools' parameter → MCP tools are auto-injected
)

# Tool execution happens automatically inside vLLM.rs
# You receive the final response with tool results incorporated
```

## Behavior Matrix

| Request Has `tools`? | MCP Configured? | Tool Mode | Stream on `</tool_call>` |
|----------------------|-----------------|-----------|--------------------------|
| ❌ No | ❌ No | None | **Continues streaming** (no special handling) |
| ✅ Yes | ❌ No | External | **Stream finishes** |
| ✅ Yes | ✅ Yes | External | **Stream finishes** |
| ❌ No | ✅ Yes | MCP Internal | **Stream pauses**, executes MCP, resumes |

> **Key Insight**: When the model outputs `</tool_call>`:
> - **No tools**: Token is treated as normal output, streaming continues
> - **External tools**: Stream finishes immediately so you can execute tools yourself
> - **MCP internal**: Stream pauses, tool executes internally, generation resumes with result

## Streaming Tool Execution

When using streaming mode (`stream=true`) with MCP-injected tools, vLLM.rs implements a **pause-and-resume** pattern:

```
User: "What files are in the current directory?"
     ↓
Model generates: "I'll list the files...<tool_call>{"name": "list_directory", ...}</tool_call>"
     ↓
[PAUSE] vLLM.rs detects </tool_call>
     ↓
[EXECUTE] MCP tool call via tools/call
     ↓
[RESUME] Continue generation with tool result
     ↓
Model generates: "The directory contains: README.md, src/, Cargo.toml..."
     ↓
Client receives all tokens seamlessly
```

### How It Works Internally

1. **Detection**: Scheduler detects `</tool_call>` token
2. **Cache**: KV cache is preserved (even for non-session requests)
3. **Signal**: Engine sends `ToolCallPause` event to server
4. **Execute**: Server executes tool via MCP client manager
5. **Resume**: New generation request with tool result appended
6. **Continue**: Streaming resumes from cached context

### Multi-Tool Support

When the model makes multiple tool calls, the process repeats:

```
Model: <tool_call>{"name": "tool_1"}</tool_call>
  → Execute tool_1, resume
Model: Based on that, <tool_call>{"name": "tool_2"}</tool_call>
  → Execute tool_2, resume
Model: Here's the final answer...
  → Stream complete
```

## Quick Start: Tool Calling

### 1. Define Tools

Tools are defined using JSON Schema following the OpenAI format:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]
```

### 2. Send Request with Tools

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    "tools": tools
})

result = response.json()
```

### 3. Handle Tool Calls

When the model wants to use a tool, it returns `tool_calls`:

```python
choice = result["choices"][0]

if choice["message"].get("tool_calls"):
    for call in choice["message"]["tool_calls"]:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])
        
        # Execute your tool
        result = my_tool_function(name, args)
        
        # Continue conversation with result
        messages.append({"role": "assistant", "tool_calls": [call]})
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(result)
        })
```

## MCP Configuration

### Single MCP Server (CLI)

```bash
cargo run --release --features metal -- --w /path/Qwen3-8B-Q2_K.gguf --ui-server --context-cache \
  --mcp-command /path/to/mcp-server \
  --mcp-args=--arg1,arg2
```

### Multiple MCP Servers (Config File)

Create `mcp.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/allowed_dir"]
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

Start the server with MCP config (it starts api server, ui server and mcp servers):

```bash
./run.sh --release --features cuda --m unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --f Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --ui-server --context-cache \
  --mcp-config ./mcp.json
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mcp-command` | Path to a single MCP server executable | - |
| `--mcp-args` | Comma-separated arguments for the MCP server | - |
| `--mcp-config` | Path to JSON config file for multiple servers | - |

> **Note**: Tool lists are refreshed automatically every 30 seconds.

## Tool Name Prefixing

When multiple MCP servers are configured, tool names are prefixed with the server ID:

| Server ID | Original Tool | Prefixed Name |
|-----------|---------------|---------------|
| `filesystem` | `read_file` | `filesystem_read_file` |
| `github` | `read_file` | `github_read_file` |

This prevents name collisions. Tool call routing automatically strips the prefix.

## Rust API

### Defining Tools

```rust
use vllm_rs::tools::{Tool, ToolCall, ToolResult};

// Builder pattern
let weather_tool = Tool::function("get_weather", "Get weather for a location")
    .param("location", "string", "City name", true)
    .param("unit", "string", "Temperature unit", false)
    .build();
```

### Parsing Tool Calls

```rust
use vllm_rs::tools::parser::ToolParser;

let parser = ToolParser::new();
let output = "I'll check the weather.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>";

let calls = parser.parse(output);
for call in calls {
    println!("Calling: {} with {}", call.function.name, call.function.arguments);
}
```

## Guided Decoding

vLLM.rs can use `llguidance` to enforce JSON outputs. There are two ways to enable it:

### 1) Force a Tool Call via `tool_choice`

If you pass `tool_choice` with a specific function, vLLM.rs applies a JSON schema constraint so the model must emit a tool call.

### 2) Provide an Explicit JSON Schema

You can pass a schema directly:

```json
{
  "guided_json_schema": {
    "type": "object",
    "properties": {
      "name": { "type": "string" },
      "arguments": { "type": "object" }
    },
    "required": ["name", "arguments"]
  }
}
```

Guided decoding is applied during the sampling loop, masking out invalid tokens at each step.

> **Note**: Guided decoding is temporarily disabled while we migrate to a new API. See release notes for updates.

## Supported Tool Call Formats

The parser recognizes:

1. **Qwen Format**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
2. **JSON Objects**: `{"name": "...", "arguments": {...}}`
3. **Code Blocks**: `` ```json {"name": "...", "arguments": {...}} ``` ``

## API Reference

### Request Fields

| Field | Type | Description |
|-------|------|-------------|
| `tools` | `Tool[]` | List of available tools (if provided, disables MCP auto-injection) |
| `tool_choice` | `string` or `object` | "auto", "none", or specific tool |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_calls` | `ToolCall[]` | Tools the model wants to call |
| `finish_reason` | `string` | "stop", "length", or "tool_calls" |

### ToolCall Structure

```json
{
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"Tokyo\"}"
    }
}
```

## Examples

See the `example/` directory:
- `tool_calling.py` - Python example with weather, calculator, and search tools
- `rust-demo-tools/` - Rust API example

### Streaming with MCP (Python)

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Don't provide tools → MCP tools are auto-injected and executed
stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "List files in the current directory"}],
    stream=True
)

# Receive the final answer (tool execution happens internally)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### External Tool Handling (Python)

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Provide tools → External handling
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{"type": "function", "function": {"name": "get_weather", ...}}]
)

# Handle tool calls yourself
if response.choices[0].message.tool_calls:
    for call in response.choices[0].message.tool_calls:
        print(f"Execute: {call.function.name}({call.function.arguments})")
```

## Best Practices

1. **Clear Descriptions**: Write detailed tool descriptions
2. **Type Hints**: Use proper JSON Schema types
3. **Required Fields**: Mark essential parameters as required
4. **Error Handling**: Always handle tool execution errors gracefully
5. **Stateless Tools**: Design tools to be stateless when possible
6. **Choose the Right Mode**:
   - Use MCP for server-side tools you control
   - Provide `tools` when clients should execute them

## Troubleshooting

### Tools Not Being Called

- Check if the model supports tool calling (Qwen3, Llama3, etc.)
- Verify tool descriptions are clear and relevant to the prompt
- Try adding "Use the available tools" to your prompt

### MCP Tools Not Injected

- Verify MCP server is running: check server logs
- Ensure `tools` parameter is not provided in the request
- Check `--mcp-tool-refresh-seconds` is reasonable

### Streaming Pauses Forever

- Check MCP server logs for tool execution errors
- Verify network connectivity to external services
- Add timeouts to your MCP server implementations

## Limitations

- Tool calling depends on the model's training for function calling
- Some models may not reliably generate tool calls
- Complex arguments may require additional validation
- Guided decoding is temporarily disabled (API migration in progress)

## See Also

- [MCP Integration Guide](mcp.md)
- [Context Cache for Sessions](context-cache.md)
- [MCP Specification](https://modelcontextprotocol.io/)
