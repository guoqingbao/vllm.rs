// src/mcp/transport.rs
//! MCP transport layer implementations
//!
//! Supports stdio (for local processes) and HTTP/SSE (for remote servers)

use super::types::*;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Transport trait for sending and receiving MCP messages
pub trait Transport: Send + Sync {
    /// Send a message
    fn send(&mut self, message: &str) -> Result<(), TransportError>;

    /// Receive a message (blocking)
    fn receive(&mut self) -> Result<String, TransportError>;

    /// Close the transport
    fn close(&mut self) -> Result<(), TransportError>;
}

/// Transport errors
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Process error: {0}")]
    Process(String),

    #[error("Connection closed")]
    Closed,

    #[error("Timeout")]
    Timeout,

    #[error("Parse error: {0}")]
    Parse(String),
}

/// Stdio transport for communicating with local MCP server processes
pub struct StdioTransport {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout_reader: Option<BufReader<ChildStdout>>,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning a process
    pub fn spawn(command: &str, args: &[&str]) -> Result<Self, TransportError> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stdout_reader = stdout.map(BufReader::new);

        Ok(Self {
            child,
            stdin,
            stdout_reader,
        })
    }
}

impl Transport for StdioTransport {
    fn send(&mut self, message: &str) -> Result<(), TransportError> {
        if let Some(ref mut stdin) = self.stdin {
            writeln!(stdin, "{}", message)?;
            stdin.flush()?;
            Ok(())
        } else {
            Err(TransportError::Closed)
        }
    }

    fn receive(&mut self) -> Result<String, TransportError> {
        if let Some(ref mut reader) = self.stdout_reader {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(TransportError::Closed);
            }
            Ok(line.trim().to_string())
        } else {
            Err(TransportError::Closed)
        }
    }

    fn close(&mut self) -> Result<(), TransportError> {
        self.stdin = None;
        self.stdout_reader = None;
        let _ = self.child.kill();
        Ok(())
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// In-memory transport for testing (not thread-safe, for single-threaded tests only)
pub struct MemoryTransport {
    tx: std::sync::mpsc::Sender<String>,
    rx: std::sync::mpsc::Receiver<String>,
}

impl MemoryTransport {
    /// Create a pair of connected transports for testing
    pub fn pair() -> (Self, Self) {
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();

        (Self { tx: tx1, rx: rx2 }, Self { tx: tx2, rx: rx1 })
    }

    pub fn send(&mut self, message: &str) -> Result<(), TransportError> {
        self.tx
            .send(message.to_string())
            .map_err(|_| TransportError::Closed)
    }

    pub fn receive(&mut self) -> Result<String, TransportError> {
        self.rx.recv().map_err(|_| TransportError::Closed)
    }

    #[allow(dead_code)]
    pub fn close(&mut self) -> Result<(), TransportError> {
        Ok(())
    }
}

/// Message framing utilities for MCP over different transports
pub mod framing {
    use super::*;

    /// Encode a JSON-RPC message for line-based transport
    pub fn encode_line(message: &impl serde::Serialize) -> Result<String, TransportError> {
        serde_json::to_string(message).map_err(|e| TransportError::Parse(e.to_string()))
    }

    /// Decode a JSON-RPC message from line-based transport
    pub fn decode_line<T: serde::de::DeserializeOwned>(line: &str) -> Result<T, TransportError> {
        serde_json::from_str(line).map_err(|e| TransportError::Parse(e.to_string()))
    }

    /// Parse any JSON-RPC message (request, response, or notification)
    pub fn parse_message(line: &str) -> Result<McpMessage, TransportError> {
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|e| TransportError::Parse(e.to_string()))?;

        // Check if it's a response (has result or error)
        if value.get("result").is_some() || value.get("error").is_some() {
            let response: JsonRpcResponse =
                serde_json::from_value(value).map_err(|e| TransportError::Parse(e.to_string()))?;
            return Ok(McpMessage::Response(response));
        }

        // Check if it's a notification (no id)
        if value.get("id").is_none() {
            let notification: JsonRpcNotification =
                serde_json::from_value(value).map_err(|e| TransportError::Parse(e.to_string()))?;
            return Ok(McpMessage::Notification(notification));
        }

        // It's a request
        let request: JsonRpcRequest =
            serde_json::from_value(value).map_err(|e| TransportError::Parse(e.to_string()))?;
        Ok(McpMessage::Request(request))
    }
}

/// Parsed MCP message types
#[derive(Debug)]
pub enum McpMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_transport() {
        let (mut t1, mut t2) = MemoryTransport::pair();

        t1.send("hello").unwrap();
        assert_eq!(t2.receive().unwrap(), "hello");

        t2.send("world").unwrap();
        assert_eq!(t1.receive().unwrap(), "world");
    }

    #[test]
    fn test_framing() {
        let req = JsonRpcRequest::new(1i64, "test", None);
        let encoded = framing::encode_line(&req).unwrap();
        let decoded: JsonRpcRequest = framing::decode_line(&encoded).unwrap();

        assert_eq!(decoded.method, "test");
    }

    #[test]
    fn test_parse_message() {
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"test"}"#;
        let response = r#"{"jsonrpc":"2.0","id":1,"result":{}}"#;
        let notification = r#"{"jsonrpc":"2.0","method":"notify"}"#;

        assert!(matches!(
            framing::parse_message(request).unwrap(),
            McpMessage::Request(_)
        ));
        assert!(matches!(
            framing::parse_message(response).unwrap(),
            McpMessage::Response(_)
        ));
        assert!(matches!(
            framing::parse_message(notification).unwrap(),
            McpMessage::Notification(_)
        ));
    }
}
