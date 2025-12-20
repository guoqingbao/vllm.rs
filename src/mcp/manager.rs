// src/mcp/manager.rs
//! MCP client manager for vLLM.rs
//!
//! Manages a background MCP client thread and cached tool list.

use super::client::{McpClient, McpClientError};
use super::transport::StdioTransport;
use crate::tools::Tool;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct McpToolConfig {
    pub command: String,
    pub args: Vec<String>,
    pub tool_refresh_interval: Duration,
}

impl McpToolConfig {
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            tool_refresh_interval: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCache {
    tools: Vec<Tool>,
}

impl ToolCache {
    fn new() -> Self {
        Self { tools: Vec::new() }
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }

    pub fn tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[derive(Debug)]
pub struct McpClientManager {
    tool_cache: Arc<RwLock<ToolCache>>,
    available: Arc<AtomicBool>,
    stop_flag: Arc<AtomicBool>,
}

impl McpClientManager {
    pub fn new(config: McpToolConfig) -> Result<Self, McpClientError> {
        let tool_cache = Arc::new(RwLock::new(ToolCache::new()));
        let available = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::new(AtomicBool::new(false));

        let cache_clone = Arc::clone(&tool_cache);
        let available_clone = Arc::clone(&available);
        let stop_clone = Arc::clone(&stop_flag);

        thread::Builder::new()
            .name("mcp-client-manager".to_string())
            .spawn(move || {
                if let Err(err) = run_manager_loop(config, cache_clone, available_clone, stop_clone)
                {
                    crate::log_error!("MCP manager loop failed: {:?}", err);
                }
            })
            .map_err(|e| McpClientError::Transport(super::transport::TransportError::Process(
                format!("Failed to spawn MCP manager thread: {e}"),
            )))?;

        Ok(Self {
            tool_cache,
            available,
            stop_flag,
        })
    }

    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Relaxed)
    }

    pub fn cached_tools(&self) -> Vec<Tool> {
        self.tool_cache.read().tools()
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }
}

fn run_manager_loop(
    config: McpToolConfig,
    tool_cache: Arc<RwLock<ToolCache>>,
    available: Arc<AtomicBool>,
    stop_flag: Arc<AtomicBool>,
) -> Result<(), McpClientError> {
    let mut last_refresh = Instant::now() - config.tool_refresh_interval;
    let args: Vec<&str> = config.args.iter().map(|s| s.as_str()).collect();
    let transport = StdioTransport::spawn(&config.command, &args)?;
    let mut client = McpClient::new(transport, "vllm-rs", "0.6.0");
    client.initialize()?;

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }
        if last_refresh.elapsed() >= config.tool_refresh_interval {
            match client.list_tools() {
                Ok(tools) => {
                    let mapped = tools
                        .into_iter()
                        .map(|tool| Tool {
                            tool_type: "function".to_string(),
                            function: crate::tools::FunctionDefinition {
                                name: tool.name,
                                description: tool.description.unwrap_or_default(),
                                parameters: tool.input_schema,
                                strict: None,
                            },
                        })
                        .collect::<Vec<_>>();
                    tool_cache.write().set_tools(mapped);
                    available.store(true, Ordering::Relaxed);
                    last_refresh = Instant::now();
                }
                Err(err) => {
                    available.store(false, Ordering::Relaxed);
                    crate::log_error!("Failed to refresh MCP tools: {:?}", err);
                }
            }
        }
        thread::sleep(Duration::from_millis(200));
    }

    Ok(())
}

pub fn call_mcp_tool(
    config: &McpToolConfig,
    name: &str,
    arguments: HashMap<String, serde_json::Value>,
) -> Result<super::types::CallToolResult, McpClientError> {
    let args: Vec<&str> = config.args.iter().map(|s| s.as_str()).collect();
    let transport = StdioTransport::spawn(&config.command, &args)?;
    let mut client = McpClient::new(transport, "vllm-rs", "0.6.0");
    client.initialize()?;
    client.call_tool(name, arguments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::transport::MemoryTransport;
    use crate::mcp::types::*;
    use serde_json::json;

    #[test]
    fn tool_cache_roundtrip() {
        let cache = ToolCache::new();
        assert!(cache.is_empty());
    }

    #[test]
    fn map_mcp_tool_to_openai_tool() {
        let mcp_tool = McpTool {
            name: "search".to_string(),
            description: Some("Search docs".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
            output_schema: None,
        };

        let tools = vec![mcp_tool];
        let mapped = tools
            .into_iter()
            .map(|tool| Tool {
                tool_type: "function".to_string(),
                function: crate::tools::FunctionDefinition {
                    name: tool.name,
                    description: tool.description.unwrap_or_default(),
                    parameters: tool.input_schema,
                    strict: None,
                },
            })
            .collect::<Vec<_>>();

        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].function.name, "search");
    }

    #[test]
    fn memory_transport_client_roundtrip() {
        let (mut client_transport, mut server_transport) = MemoryTransport::pair();
        let server = thread::spawn(move || {
            let mut server = crate::mcp::server::McpServer::new("test", "0.1");
            server.register_tool(
                McpTool {
                    name: "echo".to_string(),
                    description: None,
                    input_schema: json!({"type": "object"}),
                    output_schema: None,
                },
                Some(Box::new(|args| {
                    Ok(CallToolResult {
                        content: vec![ToolContent::text(format!(
                            "echo: {}",
                            args.get("message")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                        ))],
                        is_error: false,
                    })
                })),
            );
            let mut handled = 0;
            while handled < 3 {
                let line = server_transport.receive().unwrap();
                let msg = crate::mcp::transport::framing::parse_message(&line).unwrap();
                match msg {
                    crate::mcp::transport::McpMessage::Request(req) => {
                        let response = server.handle_request(&req);
                        let response_str =
                            crate::mcp::transport::framing::encode_line(&response).unwrap();
                        server_transport.send(&response_str).unwrap();
                        handled += 1;
                    }
                    crate::mcp::transport::McpMessage::Notification(_) => {}
                    crate::mcp::transport::McpMessage::Response(_) => {}
                }
            }
        });

        let mut client = McpClient::new(client_transport, "test-client", "0.1");
        let _ = client.initialize().unwrap();
        let tools = client.list_tools().unwrap();
        assert_eq!(tools.len(), 1);

        let result = client
            .call_tool(
                "echo",
                [("message".to_string(), json!("hello"))]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        match &result.content[0] {
            ToolContent::Text { text } => assert_eq!(text, "echo: hello"),
            _ => panic!("unexpected tool content"),
        }
        server.join().unwrap();
    }
}
