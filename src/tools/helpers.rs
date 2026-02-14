// src/tools/helpers.rs
//! Helper functions for tool call processing.
//!
//! These functions handle tool resolution, schema mapping, and tool call validation.

use super::{FunctionCall, Tool, ToolCall};
use serde_json::Value;
use std::collections::HashMap;

/// Resolve tools from request or MCP fallback
pub fn resolve_tools(request_tools: Option<&[Tool]>, mcp_tools: &[Tool]) -> Vec<Tool> {
    if let Some(tools) = request_tools {
        if !tools.is_empty() {
            return tools.to_vec();
        }
    }
    mcp_tools.to_vec()
}

/// Build a map of tool names to their parameter schemas
pub fn build_tool_schema_map(tools: &[Tool]) -> HashMap<String, Value> {
    tools
        .iter()
        .map(|tool| (tool.function.name.clone(), tool.function.parameters.clone()))
        .collect()
}

/// Filter tool calls into valid and invalid based on schema validation
pub fn filter_tool_calls(
    tool_calls: &[ToolCall],
    schemas: &HashMap<String, Value>,
) -> (Vec<ToolCall>, Vec<ToolCall>) {
    let mut valid = Vec::new();
    let mut invalid = Vec::new();

    for call in tool_calls {
        let schema = match schemas.get(&call.function.name) {
            Some(schema) => schema,
            None => {
                crate::log_warn!(
                    "Tool '{}' not found in schema map. Available tools: {:?}",
                    call.function.name,
                    schemas.keys().collect::<Vec<_>>()
                );
                invalid.push(call.clone());
                continue;
            }
        };

        let args_str = call.function.arguments.as_deref().unwrap_or("{}");
        let mut parsed_args = match serde_json::from_str::<Value>(args_str) {
            Ok(value) => value,
            Err(e) => {
                crate::log_warn!(
                    "Failed to parse arguments for tool '{}': {}. Args: {}",
                    call.function.name,
                    e,
                    args_str
                );
                invalid.push(call.clone());
                continue;
            }
        };

        if let Value::String(inner) = &parsed_args {
            if let Ok(decoded) = serde_json::from_str::<Value>(inner) {
                parsed_args = decoded;
            }
        }

        if !parsed_args.is_object() {
            crate::log_warn!(
                "Arguments for tool '{}' must be a JSON object. Got: {:?}",
                call.function.name,
                parsed_args
            );
            invalid.push(call.clone());
            continue;
        }

        let filtered_args =
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                let mut filtered = serde_json::Map::new();
                if let Some(args_obj) = parsed_args.as_object() {
                    for (key, value) in args_obj {
                        if props.contains_key(key) {
                            filtered.insert(key.clone(), value.clone());
                        }
                    }
                }
                Value::Object(filtered)
            } else {
                parsed_args.clone()
            };

        let normalized_args =
            serde_json::to_string(&filtered_args).unwrap_or_else(|_| args_str.to_string());
        valid.push(ToolCall {
            id: call.id.clone(),
            tool_type: call.tool_type.clone(),
            function: FunctionCall {
                name: call.function.name.clone(),
                arguments: Some(normalized_args),
            },
        });
    }

    (valid, invalid)
}

/// Format tool calls for logging - returns a summary string
pub fn format_tool_calls_summary(tool_calls: &[ToolCall]) -> String {
    if tool_calls.is_empty() {
        return String::new();
    }
    tool_calls
        .iter()
        .map(|call| {
            let args = call
                .function
                .arguments
                .as_deref()
                .unwrap_or("")
                .replace('\n', " ");
            let truncated = if args.len() > 160 {
                let snippet: String = args.chars().take(160).collect();
                format!("{}...", snippet)
            } else {
                args
            };
            format!("{}(args={})", call.function.name, truncated)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Log tool calls with a label (uses crate logging)
pub fn log_tool_calls(label: &str, tool_calls: &[ToolCall]) {
    if tool_calls.is_empty() {
        return;
    }
    let summary = format_tool_calls_summary(tool_calls);
    crate::log_info!("{} tool call(s): {}", label, summary);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_tools_prefers_request() {
        let request_tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mcp_tools = vec![crate::tools::function_tool("mcp", "mcp desc").build()];

        let resolved = resolve_tools(Some(&request_tools), &mcp_tools);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].function.name, "test");
    }

    #[test]
    fn test_resolve_tools_falls_back_to_mcp() {
        let mcp_tools = vec![crate::tools::function_tool("mcp", "mcp desc").build()];
        let resolved = resolve_tools(None, &mcp_tools);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].function.name, "mcp");
    }

    #[test]
    fn test_build_tool_schema_map() {
        let tools = vec![crate::tools::function_tool("test", "desc")
            .param("arg1", "string", "desc", true)
            .build()];
        let map = build_tool_schema_map(&tools);
        assert!(map.contains_key("test"));
    }
}
