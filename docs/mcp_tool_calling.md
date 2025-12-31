# MCP Integration and Tool Calling

vLLM.rs supports **Model Context Protocol (MCP)** integration, allowing you to easily expose tools from MCP servers to your LLM.

**Important:** vLLM.rs follows the standard OpenAI tool calling specification. This means the server **only handles tool definitions and prompt injection**. It does **not** execute tools internally. When a model calls a tool, the generation stops, and the tool call details are returned to the client. The client is responsible for executing the tool and submitting the results back to the server.

## Overview

### What is MCP?

The **Model Context Protocol (MCP)** is a standardized protocol for connecting LLMs to external tools and services. vLLM.rs supports:

- **Stdio transport**: Connect to local MCP servers via command-line processes
- **HTTP/SSE transport**: Connect to remote MCP servers via HTTP
- **Multi-server support**: Configure multiple MCP servers with automatic tool name prefixing
- **Automatic tool injection**: Tool definitions from configured MCP servers are automatically injected into the model's system prompt

---

## Tool Calling Workflow

1. **Configuration**: You configure MCP servers in vLLM.rs (via CLI or `mcp.json`).
2. **Injection**: vLLM.rs fetches tool definitions from these servers and appends them to the system prompt of your request.
3. **Generation**: The model generates a tool call.
4. **Completion**: The stream (or request) finishes with `finish_reason="tool_calls"`.
5. **Execution (Client-side)**: Your client code receives the tool call, executes the corresponding function (which you must implement or bridge to an MCP client), and sends the result back in a new request.

### Example Flow

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="empty")

# Request triggers tool injection from configured MCP servers
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List files in current directory"}],
    stream=True
)

# ... Handle stream ...
# If model calls a tool, response will contain tool_calls
```

---

## Configuration Options

### CLI Options

| Option | Description |
|--------|-------------|
| `--mcp-command` | Path to single MCP server executable |
| `--mcp-args` | Comma-separated arguments for MCP server |
| `--mcp-config` | Path to JSON config file for multiple servers |

### Config File Format

#### Local Servers (Stdio Transport)

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

#### Remote Servers (HTTP/SSE Transport)

```json
{
    "mcpServers": {
        "server_name": {
            "url": "https://mcp.example.com/api/",
            "headers": {
                "Authorization": "Bearer YOUR_TOKEN",
                "Accept": "text/event-stream"
            }
        }
    }
}
```

> **Note:** Each server must have either `command` (local) or `url` (remote), not both. You can mix local and remote servers in the same config file.

### Tool Name Prefixing

When using multiple MCP servers, tool names are prefixed with the server name to avoid conflicts:

- Server `filesystem` with tool `list_directory` → `filesystem_list_directory`
- Server `github` with tool `search_repos` → `github_search_repos`

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

## Behavior Matrix

| Has Request Tools | MCP Configured | Behavior |
|-------------------|----------------|----------|
| ❌ | ❌ | Normal generation |
| ✅ | ❌ | User tools used. Stream finishes at tool call. |
| ❌ | ✅ | MCP tools injected. Stream finishes at tool call. |
| ✅ | ✅ | Both User and MCP tools available. Stream finishes at tool call. |
