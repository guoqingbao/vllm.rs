# Model Context Protocol (MCP) Integration

vLLM.rs supports the Model Context Protocol (MCP) for AI agent integration.

## What is MCP?

MCP is an open protocol that enables seamless integration between LLM applications and external data sources and tools. It provides:
- Standardized tool discovery and invocation
- Resource management for prompts and templates
- Transport-agnostic communication (stdio, HTTP, WebSocket)

## Using vLLM.rs as an MCP Server

### Basic Setup

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

## MCP Client

Connect to external MCP servers:

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

## MCP Client Manager (Server Integration)

vLLM.rs can run an MCP client manager alongside the chat server. The manager:
- Spawns MCP server processes on startup
- Runs `initialize` + `tools/list` on each server
- Caches the combined tool list for automatic prompt injection
- Prefixes tool names with the server ID (e.g. `filesystem_read_file`) to avoid collisions

Enable it via CLI:

```bash
vllm-rs --server --mcp-command ./my-mcp-server --mcp-args=--port,0 --mcp-tool-refresh-seconds 30
```

### Multi-Server Configuration

You can define multiple MCP servers via a JSON config file:

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

Start the server with:

```bash
vllm-rs --server --mcp-config ./mcp.json --mcp-tool-refresh-seconds 30
```

When configured, MCP tools are injected into the system prompt if the request does not provide its own `tools` list.

Tool calls detected in model output are executed automatically via `tools/call`, and the results are fed back into the model for a follow-up response.

### Tool Name Prefixing

When multiple servers are configured, tool names are prefixed with the server ID:

- `read_file` from `filesystem` becomes `filesystem_read_file`
- `read_file` from `github` becomes `github_read_file`

Tool call routing strips the prefix and forwards the call to the matching server.

### Rust Demo: Load MCP Config

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

### Python Demo: Call a Tool via vLLM.rs

```python
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

response = openai.chat.completions.create(
    model="",
    messages=[
        {"role": "user", "content": "Use filesystem_read_file to read README.md"}
    ],
)
print(response.choices[0].message)
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

## See Also

- [Tool Calling Guide](tool_calling.md)
- [MCP Specification](https://modelcontextprotocol.io/)
