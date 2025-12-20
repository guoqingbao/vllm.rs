// src/mcp/mod.rs
//! Model Context Protocol (MCP) support for vLLM.rs
//!
//! MCP enables AI applications to connect with external data sources and tools
//! through a standardized protocol.

pub mod client;
pub mod server;
pub mod transport;
pub mod types;

pub use client::McpClient;
pub use server::McpServer;
pub use types::*;
