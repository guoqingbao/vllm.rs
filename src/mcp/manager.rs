// src/mcp/manager.rs
//! MCP client manager for vLLM.rs
//!
//! Manages a background MCP client thread and cached tool list.

use super::client::{McpClient, McpClientError};
use super::transport::StdioTransport;
use crate::tools::Tool;
use parking_lot::{Mutex, RwLock};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
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

#[derive(Debug, Clone, Deserialize)]
pub struct McpConfigFile {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerConfigFile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpServerConfigFile {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct McpServerDefinition {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct McpManagerConfig {
    pub servers: Vec<McpServerDefinition>,
    pub tool_refresh_interval: Duration,
}

impl McpManagerConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, McpClientError> {
        let contents = std::fs::read_to_string(path.as_ref())
            .map_err(|err| McpClientError::Config(format!("Failed to read MCP config: {err}")))?;
        let config: McpConfigFile =
            serde_json::from_str(&contents).map_err(McpClientError::Serialization)?;
        let servers = config
            .mcp_servers
            .into_iter()
            .map(|(id, server)| McpServerDefinition {
                id,
                command: server.command,
                args: server.args,
                env: server.env,
            })
            .collect();
        Ok(Self {
            servers,
            tool_refresh_interval: Duration::from_secs(30),
        })
    }

    pub fn from_single(config: McpToolConfig) -> Self {
        Self {
            servers: vec![McpServerDefinition {
                id: "default".to_string(),
                command: config.command,
                args: config.args,
                env: HashMap::new(),
            }],
            tool_refresh_interval: config.tool_refresh_interval,
        }
    }

    pub fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.tool_refresh_interval = interval;
        self
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

pub struct McpClientManager {
    tool_cache: Arc<RwLock<ToolCache>>,
    routing_table: Arc<RwLock<HashMap<String, ToolRouting>>>,
    clients: Arc<RwLock<HashMap<String, Arc<Mutex<McpClient<StdioTransport>>>>>>,
    available: Arc<AtomicBool>,
    stop_flag: Arc<AtomicBool>,
}

impl McpClientManager {
    pub fn new(config: McpManagerConfig) -> Result<Self, McpClientError> {
        if config.servers.is_empty() {
            return Err(McpClientError::Config(
                "MCP manager requires at least one server".to_string(),
            ));
        }

        let tool_cache = Arc::new(RwLock::new(ToolCache::new()));
        let routing_table = Arc::new(RwLock::new(HashMap::new()));
        let clients = Arc::new(RwLock::new(HashMap::new()));
        let available = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::new(AtomicBool::new(false));

        let mut any_client = false;
        for server in &config.servers {
            let args: Vec<&str> = server.args.iter().map(|s| s.as_str()).collect();
            match StdioTransport::spawn_with_env(&server.command, &args, &server.env) {
                Ok(transport) => {
                    let mut client = McpClient::new(transport, "vllm-rs", "0.6.0");
                    if let Err(err) = client.initialize() {
                        crate::log_error!(
                            "Failed to initialize MCP server {}: {:?}",
                            server.id,
                            err
                        );
                        continue;
                    }
                    clients
                        .write()
                        .insert(server.id.clone(), Arc::new(Mutex::new(client)));
                    any_client = true;
                }
                Err(err) => {
                    crate::log_error!("Failed to spawn MCP server {}: {:?}", server.id, err);
                }
            }
        }

        if !any_client {
            return Err(McpClientError::Config(
                "Failed to initialize any MCP servers".to_string(),
            ));
        }

        let cache_clone = Arc::clone(&tool_cache);
        let routing_clone = Arc::clone(&routing_table);
        let clients_clone = Arc::clone(&clients);
        let available_clone = Arc::clone(&available);
        let stop_clone = Arc::clone(&stop_flag);
        let refresh_interval = config.tool_refresh_interval;

        thread::Builder::new()
            .name("mcp-client-manager".to_string())
            .spawn(move || {
                if let Err(err) = run_manager_loop(
                    refresh_interval,
                    cache_clone,
                    routing_clone,
                    clients_clone,
                    available_clone,
                    stop_clone,
                ) {
                    crate::log_error!("MCP manager loop failed: {:?}", err);
                }
            })
            .map_err(|e| {
                McpClientError::Transport(super::transport::TransportError::Process(format!(
                    "Failed to spawn MCP manager thread: {e}"
                )))
            })?;

        Ok(Self {
            tool_cache,
            routing_table,
            clients,
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

    pub fn call_tool(
        &self,
        name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> Result<super::types::CallToolResult, McpClientError> {
        let routing = self
            .routing_table
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| McpClientError::ToolNotFound(name.to_string()))?;

        let client = self
            .clients
            .read()
            .get(&routing.server_id)
            .cloned()
            .ok_or_else(|| McpClientError::ToolNotFound(name.to_string()))?;

        let mut client = client.lock();
        client.call_tool(&routing.original_name, arguments)
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
struct ToolRouting {
    server_id: String,
    original_name: String,
}

fn run_manager_loop(
    refresh_interval: Duration,
    tool_cache: Arc<RwLock<ToolCache>>,
    routing_table: Arc<RwLock<HashMap<String, ToolRouting>>>,
    clients: Arc<RwLock<HashMap<String, Arc<Mutex<McpClient<StdioTransport>>>>>>,
    available: Arc<AtomicBool>,
    stop_flag: Arc<AtomicBool>,
) -> Result<(), McpClientError> {
    let mut last_refresh = Instant::now() - refresh_interval;

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }
        if last_refresh.elapsed() >= refresh_interval {
            let mut mapped_tools = Vec::new();
            let mut routing = HashMap::new();
            let mut any_success = false;

            for (server_id, client) in clients.read().iter() {
                let mut client = client.lock();
                match client.list_tools() {
                    Ok(tools) => {
                        any_success = true;
                        mapped_tools.extend(map_mcp_tools(server_id, tools, &mut routing));
                    }
                    Err(err) => {
                        crate::log_error!(
                            "Failed to refresh MCP tools for {}: {:?}",
                            server_id,
                            err
                        );
                        break;
                    }
                }
            }

            if any_success {
                tool_cache.write().set_tools(mapped_tools);
                *routing_table.write() = routing;
                available.store(true, Ordering::Relaxed);
                last_refresh = Instant::now();
            } else {
                available.store(false, Ordering::Relaxed);
            }
        }
        thread::sleep(Duration::from_millis(200));
    }

    Ok(())
}

fn map_mcp_tools(
    server_id: &str,
    tools: Vec<super::types::McpTool>,
    routing: &mut HashMap<String, ToolRouting>,
) -> Vec<Tool> {
    tools
        .into_iter()
        .map(|tool| {
            let prefixed_name = format!("{server_id}_{}", tool.name);
            routing.insert(
                prefixed_name.clone(),
                ToolRouting {
                    server_id: server_id.to_string(),
                    original_name: tool.name,
                },
            );
            Tool {
                tool_type: "function".to_string(),
                function: crate::tools::FunctionDefinition {
                    name: prefixed_name,
                    description: tool.description.unwrap_or_default(),
                    parameters: tool.input_schema,
                    strict: None,
                },
            }
        })
        .collect()
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
        let mut routing = HashMap::new();
        let mapped = map_mcp_tools("filesystem", tools, &mut routing);

        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].function.name, "filesystem_search");
        let routing = routing.get("filesystem_search").unwrap();
        assert_eq!(routing.server_id, "filesystem");
        assert_eq!(routing.original_name, "search");
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
                            args.get("message").and_then(|v| v.as_str()).unwrap_or("")
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

    #[test]
    fn parse_mcp_config_file() {
        let json = r#"{
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                },
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_example"
                    }
                }
            }
        }"#;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("mcp_config_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();

        let config = McpManagerConfig::from_file(&path).unwrap();
        assert_eq!(config.servers.len(), 2);
        let filesystem = config
            .servers
            .iter()
            .find(|server| server.id == "filesystem")
            .unwrap();
        assert_eq!(filesystem.command, "npx");
        assert_eq!(
            filesystem.args,
            vec!["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        );
        let github = config
            .servers
            .iter()
            .find(|server| server.id == "github")
            .unwrap();
        assert_eq!(
            github.env.get("GITHUB_PERSONAL_ACCESS_TOKEN").unwrap(),
            "ghp_example"
        );

        let _ = std::fs::remove_file(&path);
    }
}
