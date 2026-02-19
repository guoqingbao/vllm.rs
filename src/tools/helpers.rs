// src/tools/helpers.rs
//! Helper functions for tool call processing.
//!
//! These functions handle tool resolution, schema mapping, and tool call validation.

use super::{FunctionCall, Tool, ToolCall};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

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
        let args_str = call.function.arguments.as_deref().unwrap_or("{}");
        let mut parsed_args = match serde_json::from_str::<Value>(args_str) {
            Ok(value) => value,
            Err(e) => {
                crate::log_error!(
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
            crate::log_error!(
                "Arguments for tool '{}' must be a JSON object. Got: {:?}",
                call.function.name,
                parsed_args
            );
            invalid.push(call.clone());
            continue;
        }

        let args_obj = match parsed_args.as_object() {
            Some(obj) => obj,
            None => {
                invalid.push(call.clone());
                continue;
            }
        };

        let schema = match schemas.get(&call.function.name) {
            Some(schema) => schema,
            None => {
                if call.function.name == "TodoWrite" {
                    if let Some(converted_calls) =
                        convert_todowrite_to_task_create_calls(call, args_obj, schemas)
                    {
                        crate::log_info!(
                            "Converted TodoWrite into {} TaskCreate call(s)",
                            converted_calls.len()
                        );
                        valid.extend(converted_calls);
                        continue;
                    }
                    crate::log_error!(
                        "Unable to convert TodoWrite: TaskCreate unavailable or TodoWrite payload malformed. Args: {}",
                        args_str
                    );
                    invalid.push(call.clone());
                    continue;
                }
                crate::log_error!(
                    "Tool '{}' not found in schema map. Available tools: {:?}",
                    call.function.name,
                    schemas.keys().collect::<Vec<_>>()
                );
                invalid.push(call.clone());
                continue;
            }
        };

        let repaired_args_obj = repair_embedded_parameter_blocks(args_obj, schema);
        let normalized_args_obj = normalize_argument_keys(&repaired_args_obj, schema);

        if let Some(missing) = missing_required_keys(&normalized_args_obj, schema) {
            crate::log_error!(
                "Missing required argument(s) for tool '{}': {:?}. Args: {}",
                call.function.name,
                missing,
                args_str
            );
            invalid.push(call.clone());
            continue;
        }

        let filtered_args = Value::Object(normalized_args_obj);

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

fn convert_todowrite_to_task_create_calls(
    source_call: &ToolCall,
    args_obj: &serde_json::Map<String, Value>,
    schemas: &HashMap<String, Value>,
) -> Option<Vec<ToolCall>> {
    let _task_create_schema = schemas.get("TaskCreate")?;

    let mut tasks: Vec<&serde_json::Map<String, Value>> = Vec::new();

    if let Some(values) = args_obj
        .get("tasks")
        .or_else(|| args_obj.get("todos"))
        .and_then(Value::as_array)
    {
        for value in values {
            if let Some(task) = value.as_object() {
                tasks.push(task);
            }
        }
    }

    if tasks.is_empty()
        && (args_obj.get("title").is_some()
            || args_obj.get("subject").is_some()
            || args_obj.get("content").is_some()
            || args_obj.get("description").is_some())
    {
        tasks.push(args_obj);
    }

    if tasks.is_empty() {
        return None;
    }

    let mut converted = Vec::new();
    for (index, task) in tasks.iter().enumerate() {
        let subject = task
            .get("subject")
            .and_then(Value::as_str)
            .or_else(|| task.get("title").and_then(Value::as_str))
            .or_else(|| task.get("content").and_then(Value::as_str))
            .map(str::trim)
            .filter(|v| !v.is_empty());

        let Some(subject) = subject else {
            continue;
        };

        let description = task
            .get("description")
            .and_then(Value::as_str)
            .or_else(|| task.get("content").and_then(Value::as_str))
            .or_else(|| task.get("title").and_then(Value::as_str))
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or(subject);

        let active_form = task
            .get("activeForm")
            .and_then(Value::as_str)
            .or_else(|| task.get("active_form").and_then(Value::as_str))
            .map(str::trim)
            .filter(|v| !v.is_empty());

        let mut mapped = serde_json::Map::new();
        mapped.insert("subject".to_string(), Value::String(subject.to_string()));
        mapped.insert(
            "description".to_string(),
            Value::String(description.to_string()),
        );
        if let Some(active_form) = active_form {
            mapped.insert(
                "activeForm".to_string(),
                Value::String(active_form.to_string()),
            );
        }
        if let Some(metadata) = task.get("metadata").and_then(Value::as_object) {
            mapped.insert("metadata".to_string(), Value::Object(metadata.clone()));
        }

        let call_id = if index == 0 {
            source_call.id.clone()
        } else {
            format!("{}_{}", source_call.id, index)
        };

        let arguments =
            serde_json::to_string(&Value::Object(mapped)).unwrap_or_else(|_| "{}".to_string());
        converted.push(ToolCall {
            id: call_id,
            tool_type: source_call.tool_type.clone(),
            function: FunctionCall {
                name: "TaskCreate".to_string(),
                arguments: Some(arguments),
            },
        });
    }

    if converted.is_empty() {
        None
    } else {
        Some(converted)
    }
}

fn normalize_argument_keys(
    args_obj: &serde_json::Map<String, Value>,
    schema: &Value,
) -> serde_json::Map<String, Value> {
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return args_obj.clone();
    };

    let mut canonical_props: HashMap<String, Vec<String>> = HashMap::new();
    for prop in props.keys() {
        canonical_props
            .entry(canonicalize_key(prop))
            .or_default()
            .push(prop.clone());
    }

    let mut normalized = serde_json::Map::new();
    for (key, value) in args_obj {
        for candidate_key in normalized_key_candidates(key) {
            if props.contains_key(&candidate_key) {
                normalized
                    .entry(candidate_key.clone())
                    .or_insert_with(|| value.clone());
                continue;
            }

            let canonical = canonicalize_key(&candidate_key);
            if let Some(candidates) = canonical_props.get(&canonical) {
                if candidates.len() == 1 {
                    let target = &candidates[0];
                    normalized
                        .entry(target.clone())
                        .or_insert_with(|| value.clone());
                    continue;
                }
            }

            // Common fallback for editor/file tools where models often emit "file".
            if candidate_key == "file"
                && props.contains_key("filePath")
                && !props.contains_key("file")
            {
                normalized
                    .entry("filePath".to_string())
                    .or_insert_with(|| value.clone());
            }
        }
    }

    normalized
}

fn normalized_key_candidates(key: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    let trimmed = key.trim();
    if !trimmed.is_empty() {
        candidates.push(trimmed.to_string());
    }

    let stripped = strip_parameter_artifacts(trimmed);
    if !stripped.is_empty() && stripped != trimmed {
        candidates.push(stripped.to_string());
    }

    if let Some(name) = extract_parameter_name(trimmed) {
        if !name.is_empty() && !candidates.iter().any(|c| c == &name) {
            candidates.push(name);
        }
    }

    candidates
}

fn strip_parameter_artifacts(key: &str) -> &str {
    let end = key
        .char_indices()
        .find_map(|(idx, ch)| matches!(ch, '\n' | '\r' | '<').then_some(idx))
        .unwrap_or(key.len());
    key[..end].trim()
}

fn canonicalize_key(key: &str) -> String {
    key.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn parameter_start_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?is)<\s*parameter\s*=\s*([^>\n\r]+)")
            .expect("parameter start regex must compile")
    })
}

fn missing_required_keys(
    args_obj: &serde_json::Map<String, Value>,
    schema: &Value,
) -> Option<Vec<String>> {
    let required = schema.get("required").and_then(|r| r.as_array())?;
    let mut missing = Vec::new();
    for key in required {
        let Some(name) = key.as_str() else {
            continue;
        };
        if !args_obj.get(name).is_some_and(|value| !value.is_null()) {
            missing.push(name.to_string());
        }
    }
    if missing.is_empty() {
        None
    } else {
        Some(missing)
    }
}

fn qwen_parameter_block_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?is)<\s*parameter\s*=\s*([^>\n\r]+)\s*>(.*?)</\s*parameter\s*>")
            .expect("qwen parameter regex must compile")
    })
}

fn parse_loose_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        v
    } else {
        Value::String(trimmed.to_string())
    }
}

fn parse_unclosed_parameter_block(text: &str) -> Option<(usize, String, Value)> {
    let caps = parameter_start_regex().captures(text)?;
    let (Some(full), Some(name_m)) = (caps.get(0), caps.get(1)) else {
        return None;
    };
    let start = full.start();
    let mut value_start = full.end();
    let name = name_m.as_str().trim();
    if name.is_empty() {
        return None;
    }

    if text.as_bytes().get(value_start) == Some(&b'>') {
        value_start += 1;
    }

    let value = parse_loose_value(&text[value_start..]);
    Some((start, name.to_string(), value))
}

fn repair_embedded_parameter_blocks(
    args_obj: &serde_json::Map<String, Value>,
    schema: &Value,
) -> serde_json::Map<String, Value> {
    let re = qwen_parameter_block_regex();
    let mut repaired = args_obj.clone();
    let mut extracted_raw = serde_json::Map::new();

    for (key, value) in args_obj {
        let Some(text) = value.as_str() else {
            continue;
        };
        if !parameter_start_regex().is_match(text) {
            if let Some(name) = extract_parameter_name(key) {
                extracted_raw.insert(name, parse_loose_value(text));
                repaired.remove(key);
            }
            continue;
        }

        let mut first_marker_start = None;
        let mut matched_any = false;
        for caps in re.captures_iter(text) {
            let (Some(full), Some(name_m), Some(value_m)) = (caps.get(0), caps.get(1), caps.get(2))
            else {
                continue;
            };
            matched_any = true;
            first_marker_start =
                Some(first_marker_start.map_or(full.start(), |s: usize| s.min(full.start())));
            let name = name_m.as_str().trim();
            if name.is_empty() {
                continue;
            }
            extracted_raw.insert(name.to_string(), parse_loose_value(value_m.as_str()));
        }

        if !matched_any {
            if let Some((marker_start, name, value)) = parse_unclosed_parameter_block(text) {
                matched_any = true;
                first_marker_start = Some(marker_start);
                extracted_raw.insert(name, value);
            }
        }

        if !matched_any {
            continue;
        }

        if let Some(marker_start) = first_marker_start {
            let prefix = text[..marker_start].trim();
            if prefix.is_empty() {
                // The value is only malformed nested parameter blocks.
                repaired.remove(key);
            } else {
                repaired.insert(key.clone(), Value::String(prefix.to_string()));
            }
        }
    }

    if extracted_raw.is_empty() {
        return repaired;
    }

    // Canonicalize recovered keys against tool schema and merge.
    let extracted_normalized = normalize_argument_keys(&extracted_raw, schema);
    for (k, v) in extracted_normalized {
        repaired.insert(k, v);
    }
    repaired
}

fn extract_parameter_name(text: &str) -> Option<String> {
    let caps = parameter_start_regex().captures(text)?;
    let name = caps.get(1)?.as_str().trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
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

    #[test]
    fn converts_todowrite_to_taskcreate_calls() {
        let schemas = HashMap::from([(
            "TaskCreate".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "subject": {"type":"string"},
                    "description": {"type":"string"},
                    "activeForm": {"type":"string"}
                },
                "required": ["subject", "description"],
                "additionalProperties": false
            }),
        )]);
        let args = serde_json::json!({
            "tasks": [
                {"id": "1", "title": "Do thing", "status": "in_progress", "activeForm": "Doing thing"},
                {"id": "2", "title": "Do other thing", "status": "pending"}
            ]
        })
        .to_string();
        let call = crate::tools::new_tool_call("call_1", "TodoWrite", args);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(invalid.is_empty());
        assert_eq!(valid.len(), 2);
        assert!(valid.iter().all(|call| call.function.name == "TaskCreate"));

        let first_args = valid[0].function.arguments.as_ref().unwrap();
        let first: Value = serde_json::from_str(first_args).unwrap();
        assert_eq!(first["subject"], "Do thing");
        assert_eq!(first["description"], "Do thing");
        assert_eq!(first["activeForm"], "Doing thing");

        let second_args = valid[1].function.arguments.as_ref().unwrap();
        let second: Value = serde_json::from_str(second_args).unwrap();
        assert_eq!(second["subject"], "Do other thing");
        assert_eq!(second["description"], "Do other thing");
    }

    #[test]
    fn converts_todowrite_todos_content_shape_to_taskcreate_calls() {
        let schemas = HashMap::from([(
            "TaskCreate".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "subject": {"type":"string"},
                    "description": {"type":"string"},
                    "activeForm": {"type":"string"}
                },
                "required": ["subject", "description"],
                "additionalProperties": false
            }),
        )]);

        let args = serde_json::json!({
            "todos": [
                {
                    "content": "Fix Bug 1: Add Mistral3 variant to ModelType enum",
                    "status": "in_progress",
                    "activeForm": "Adding Mistral3 variant to ModelType enum"
                },
                {
                    "content": "Fix Bug 2: Fix shared_gate weight loading shape in Qwen3DecoderLayer",
                    "status": "pending",
                    "activeForm": "Fixing shared_gate weight loading shape"
                }
            ]
        })
        .to_string();
        let call = crate::tools::new_tool_call("call_1", "TodoWrite", args);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(invalid.is_empty());
        assert_eq!(valid.len(), 2);
        assert!(valid.iter().all(|call| call.function.name == "TaskCreate"));

        let first: Value =
            serde_json::from_str(valid[0].function.arguments.as_ref().unwrap()).unwrap();
        assert_eq!(
            first["subject"],
            "Fix Bug 1: Add Mistral3 variant to ModelType enum"
        );
        assert_eq!(
            first["description"],
            "Fix Bug 1: Add Mistral3 variant to ModelType enum"
        );
        assert_eq!(
            first["activeForm"],
            "Adding Mistral3 variant to ModelType enum"
        );
    }

    #[test]
    fn rejects_todowrite_when_taskcreate_not_available() {
        let schemas = HashMap::new();
        let call =
            crate::tools::new_tool_call("call_1", "TodoWrite", r#"{"tasks":[{"title":"x"}]}"#);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(valid.is_empty());
        assert_eq!(invalid.len(), 1);
    }

    #[test]
    fn rejects_unknown_tool_without_schema() {
        let schemas = HashMap::new();
        let call = crate::tools::new_tool_call("call_1", "UnknownTool", r#"{"x":1}"#);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(valid.is_empty());
        assert_eq!(invalid.len(), 1);
    }

    #[test]
    fn repairs_embedded_qwen_parameter_blocks_for_write_tool() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": {"type":"string"},
                "content": {"type":"string"}
            },
            "required": ["filePath", "content"]
        });
        let schemas = HashMap::from([("write".to_string(), schema)]);

        let args = serde_json::json!({
            "file": "/root/vllm.rs/AGENTS.md\n<parameter=content>\n# Title\n</parameter>"
        })
        .to_string();
        let call = crate::tools::new_tool_call("call_1", "write", args);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(invalid.is_empty());
        assert_eq!(valid.len(), 1);

        let args = valid[0].function.arguments.as_ref().unwrap();
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["filePath"], "/root/vllm.rs/AGENTS.md");
        assert_eq!(parsed["content"], "# Title");
    }

    #[test]
    fn repairs_embedded_parameter_blocks_when_filepath_already_present() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": {"type":"string"},
                "content": {"type":"string"}
            },
            "required": ["filePath", "content"]
        });
        let schemas = HashMap::from([("write".to_string(), schema)]);

        let args = serde_json::json!({
            "filePath": "/root/vllm.rs/AGENTS.md",
            "file": "<parameter=content>\n## Body\n</parameter>"
        })
        .to_string();
        let call = crate::tools::new_tool_call("call_1", "write", args);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(invalid.is_empty());
        assert_eq!(valid.len(), 1);

        let args = valid[0].function.arguments.as_ref().unwrap();
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["filePath"], "/root/vllm.rs/AGENTS.md");
        assert_eq!(parsed["content"], "## Body");
    }

    #[test]
    fn repairs_unclosed_parameter_block_for_content() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "content": {"type":"string"}
            },
            "required": ["content"]
        });
        let schemas = HashMap::from([("write".to_string(), schema)]);

        let args = serde_json::json!({
            "file\n</parameter": "<parameter=content>\n# vLLM.rs - AGENTS.md\n\n## Overview"
        })
        .to_string();
        let call = crate::tools::new_tool_call("call_1", "write", args);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(invalid.is_empty());
        assert_eq!(valid.len(), 1);

        let args = valid[0].function.arguments.as_ref().unwrap();
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["content"], "# vLLM.rs - AGENTS.md\n\n## Overview");
    }

    #[test]
    fn repairs_unclosed_content_when_filepath_exists() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": {"type":"string"},
                "content": {"type":"string"}
            },
            "required": ["filePath", "content"]
        });
        let schemas = HashMap::from([("write".to_string(), schema)]);

        let args = serde_json::json!({
            "filePath": "/root/vllm.rs/AGENTS.md",
            "file\n</parameter": "<parameter=content>\n# vLLM.rs - AGENTS.md\n\n## Overview"
        })
        .to_string();
        let call = crate::tools::new_tool_call("call_1", "write", args);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(invalid.is_empty());
        assert_eq!(valid.len(), 1);

        let args = valid[0].function.arguments.as_ref().unwrap();
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["filePath"], "/root/vllm.rs/AGENTS.md");
        assert_eq!(parsed["content"], "# vLLM.rs - AGENTS.md\n\n## Overview");
    }

    #[test]
    fn normalizes_malformed_file_key_to_filepath() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": {"type":"string"},
                "content": {"type":"string"}
            },
            "required": ["filePath", "content"]
        });
        let schemas = HashMap::from([("write".to_string(), schema)]);

        let args = serde_json::json!({
            "file\n</parameter": "/root/vllm.rs/AGENTS.md",
            "content": "hello"
        })
        .to_string();
        let call = crate::tools::new_tool_call("call_1", "write", args);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(invalid.is_empty());
        assert_eq!(valid.len(), 1);

        let args = valid[0].function.arguments.as_ref().unwrap();
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["filePath"], "/root/vllm.rs/AGENTS.md");
        assert_eq!(parsed["content"], "hello");
    }

    #[test]
    fn repairs_spaced_parameter_start_tag() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": {"type":"string"},
                "content": {"type":"string"}
            },
            "required": ["filePath", "content"]
        });
        let schemas = HashMap::from([("write".to_string(), schema)]);

        let args = serde_json::json!({
            "filePath": "/root/vllm.rs/AGENTS.md",
            "file": "<parameter = content>\nhello world"
        })
        .to_string();
        let call = crate::tools::new_tool_call("call_1", "write", args);

        let (valid, invalid) = filter_tool_calls(&[call], &schemas);
        assert!(invalid.is_empty());
        assert_eq!(valid.len(), 1);

        let args = valid[0].function.arguments.as_ref().unwrap();
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["filePath"], "/root/vllm.rs/AGENTS.md");
        assert_eq!(parsed["content"], "hello world");
    }
}
