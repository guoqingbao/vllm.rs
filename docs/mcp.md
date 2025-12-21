# Model Context Protocol (MCP) Integration

vLLM.rs supports the Model Context Protocol (MCP) for AI agent integration.

## What is MCP?

MCP is an open protocol that enables seamless integration between LLM applications and external data sources and tools. It provides:
- Standardized tool discovery and invocation
- Resource management for prompts and templates
- Transport-agnostic communication (stdio, HTTP, WebSocket)

## MCP Client Manager

vLLM.rs can run an MCP client manager alongside the chat server. When configured:

1. **Tool Auto-Injection**: MCP tools are automatically injected into the system prompt when no `tools` are provided in the request
2. **Automatic Execution**: Tool calls are detected and executed via `tools/call`
3. **Streaming Support**: Pause-and-resume pattern for seamless streaming tool execution
4. **Multi-Server**: Connect to multiple MCP servers with tool name prefixing

## Quick Start

### Single MCP Server

```bash
vllm-rs --server --mcp-command ./my-mcp-server --mcp-args=--port,0
```

### Multiple MCP Servers

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

Start the server:

```bash
vllm-rs --server --mcp-config ./mcp.json --mcp-tool-refresh-seconds 30
```

## How Tool Execution Works

### Non-Streaming Mode

```
User Request → Generate → Parse Tool Calls → Execute MCP → Generate Follow-up → Response
```

### Streaming Mode

vLLM.rs implements an efficient **pause-and-resume** pattern:

```
User Request
    ↓
Start Streaming Tokens → Client receives tokens
    ↓
Model outputs: "<tool_call>{"name": "read_file", ...}</tool_call>"
    ↓
[PAUSE] Scheduler detects </tool_call> token
    ↓
[CACHE] KV cache is preserved (auto-generated session_id)
    ↓
[EXECUTE] MCP tools/call
    ↓
[RESUME] Continue generation with tool result
    ↓
Continue Streaming → Client receives more tokens
    ↓
Complete
```

**Key benefits:**
- Zero client-side complexity for tool execution
- KV cache is preserved for efficiency  
- Multiple tool calls are handled automatically
- Works with any OpenAI-compatible client

### When Tool Execution Happens

| Request has `tools`? | MCP Configured? | Tool Mode | Stream on `</tool_call>` |
|----------------------|-----------------|-----------|--------------------------|
| ❌ No | ❌ No | None | **Continues streaming** |
| ✅ Yes | ❌ No | External | **Stream finishes** |
| ✅ Yes | ✅ Yes | External | **Stream finishes** |
| ❌ No | ✅ Yes | MCP Internal | **Stream pauses**, resumes after tool |

> **Important**: When the model outputs `</tool_call>`:
> - **No tools**: Token is treated as normal output, streaming continues
> - **User-provided tools**: Stream finishes so you can handle tools externally
> - **MCP internal**: Stream pauses, tool executes, generation resumes

## Configuration Reference

### CLI Options

| Option | Description | Example |
|--------|-------------|---------|
| `--mcp-command` | Path to single MCP server | `--mcp-command ./mcp-server` |
| `--mcp-args` | Comma-separated server args | `--mcp-args=--port,0` |
| `--mcp-config` | Path to JSON config file | `--mcp-config ./mcp.json` |

> **Note**: Tool lists are refreshed automatically every 30 seconds.

### Config File Format

```json
{
  "mcpServers": {
    "<server-id>": {
      "command": "<executable>",
      "args": ["<arg1>", "<arg2>"],
      "env": {
        "KEY": "value"
      }
    }
  }
}
```

### Environment Variables

MCP servers can receive environment variables:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_...",
        "GITHUB_ORG": "my-org"
      }
    }
  }
}
```

## Tool Name Prefixing

When multiple servers are configured, tool names are prefixed with the server ID:

| Server ID | Original Tool | Prefixed Name |
|-----------|---------------|---------------|
| `filesystem` | `read_file` | `filesystem_read_file` |
| `github` | `read_file` | `github_read_file` |

This prevents name collisions. Tool call routing automatically strips the prefix and forwards to the correct server.

## Using vLLM.rs as an MCP Server

In addition to being an MCP client, vLLM.rs can also act as an MCP server:

```rust
use vllm_rs::mcp::{McpServer, McpTool, ToolContent, CallToolResult};
use serde_json::json;

let mut server = McpServer::new("my-llm-tools", "1.0.0");

// Register a tool
server.register_tool(
    McpTool {
        name: "generate_text".to_string(),
        description: Some("Generate text using the LLM".to_string()),
        input_schema: json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The prompt to complete"}
            },
            "required": ["prompt"]
        }),
        output_schema: None,
    },
    Some(Box::new(|args| {
        let prompt = args.get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        // Your LLM generation logic here
        Ok(CallToolResult {
            content: vec![ToolContent::text("Generated response...")],
            is_error: false,
        })
    }))
);
```

## MCP Client API

Connect to external MCP servers programmatically:

```rust
use vllm_rs::mcp::{McpClient, StdioTransport};

// Connect to an MCP server process
let transport = StdioTransport::spawn("mcp-server", &[])?;
let mut client = McpClient::new(transport, "vllm-rs", "0.6.0");

// Initialize connection
client.initialize()?;

// List available tools
let tools = client.list_tools()?;
for tool in tools {
    println!("Tool: {} - {:?}", tool.name, tool.description);
}

// Call a tool
let result = client.call_tool("search", [
    ("query".to_string(), json!("rust programming"))
].into_iter().collect())?;
```

## Rust Demo: Load MCP Config

```rust
use vllm_rs::mcp::{McpClientManager, McpManagerConfig};

let cfg = McpManagerConfig::from_file("mcp.json")?;
let manager = McpClientManager::new(cfg)?;

let tools = manager.cached_tools();
println!(
    "Tools: {:?}",
    tools.iter().map(|t| &t.function.name).collect::<Vec<_>>()
);
```

## Python Demo: Call a Tool via vLLM.rs

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1/",
    api_key="EMPTY"
)

# Don't provide tools → MCP tools are auto-injected and executed
response = client.chat.completions.create(
    model="",
    messages=[
        {"role": "user", "content": "Use filesystem_read_file to read README.md"}
    ],
)
print(response.choices[0].message.content)
```

## Protocol Support

vLLM.rs implements MCP version `2024-11-05` with support for:

| Feature | Supported |
|---------|-----------|
| Tools | ✅ |
| Resources | ✅ |
| Prompts | ✅ |
| Logging | ⚠️ Partial |

## Transport Layers

### Stdio Transport

For local process communication:
```rust
let transport = StdioTransport::spawn("./my-mcp-server", &["--port", "0"])?;
```

## JSON-RPC Methods

| Method | Description |
|--------|-------------|
| `initialize` | Initialize the connection |
| `tools/list` | List available tools |
| `tools/call` | Invoke a tool |
| `resources/list` | List available resources |
| `prompts/list` | List available prompts |

## Troubleshooting

### MCP Server Won't Start

- Check the command path is correct and executable
- Verify required environment variables are set
- Check server logs in stderr

### Tools Not Being Executed

- Ensure you're NOT providing `tools` in your request (this disables auto-execution)
- Verify MCP server is responding to `tools/list`
- Check vLLM.rs logs for tool call parsing

### Streaming Hangs

- MCP server may be slow or unresponsive
- Add timeouts to your MCP server implementations
- Check for network issues with external services

### Tool Name Conflicts

- Use unique server IDs in your config
- Remember tool names are prefixed with server ID
- Check the actual tool names with `tools/list`

## See Also

- [Tool Calling Guide](tool_calling.md)
- [Context Cache for Sessions](context-cache.md)
- [MCP Specification](https://modelcontextprotocol.io/)
