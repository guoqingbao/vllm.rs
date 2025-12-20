# Tool Calling and MCP Support for vLLM.rs

This document describes how to use tool calling and the Model Context Protocol (MCP) features in vLLM.rs.

## Overview

vLLM.rs supports:
- **OpenAI-compatible Tool Calling**: Define tools and let the model decide when to call them
- **MCP Server**: Expose your tools to AI agents via the Model Context Protocol
- **MCP Client**: Connect to external MCP servers to use their tools

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

## MCP Tool Execution (Server)

If you start the server with an MCP client manager, tool calls are executed automatically:

```bash
vllm-rs --server --mcp-command ./my-mcp-server --mcp-args=--port,0
```

When MCP is enabled:
- The cached MCP tool list is injected into the system prompt when no `tools` are provided.
- Tool calls are routed to `tools/call`, and the tool results are fed back into the model for a follow-up completion.

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

## Supported Tool Call Formats

The parser recognizes:

1. **Qwen Format**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
2. **JSON Objects**: `{"name": "...", "arguments": {...}}`
3. **Code Blocks**: `` ```json {"name": "...", "arguments": {...}} ``` ``

## API Reference

### Request Fields

| Field | Type | Description |
|-------|------|-------------|
| `tools` | `Tool[]` | List of available tools |
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

## Best Practices

1. **Clear Descriptions**: Write detailed tool descriptions
2. **Type Hints**: Use proper JSON Schema types
3. **Required Fields**: Mark essential parameters as required
4. **Error Handling**: Always handle tool execution errors gracefully
5. **Stateless Tools**: Design tools to be stateless when possible

## Limitations

- Tool calling depends on the model's training for function calling
- Some models may not reliably generate tool calls
- Complex arguments may require additional validation
