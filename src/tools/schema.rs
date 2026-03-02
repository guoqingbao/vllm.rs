// src/tools/schema.rs
//! JSON Schema utilities for tool parameters
//!
//! Provides helpers for working with JSON Schema in tool definitions.

use crate::tools::Tool;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Remove JSON Schema features that llguidance doesn't support.
/// Currently strips all "format" fields recursively.
pub fn sanitize_schema_for_llguidance(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, value) in map {
                if key == "format" {
                    continue;
                }
                out.insert(key.clone(), sanitize_schema_for_llguidance(value));
            }
            Value::Object(out)
        }
        Value::Array(items) => {
            Value::Array(items.iter().map(sanitize_schema_for_llguidance).collect())
        }
        _ => schema.clone(),
    }
}

/// Lark grammar helper functions for llguidance constraint building
fn lark_quote(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{}\"", escaped)
}

fn lark_literal(value: &str, is_special: bool) -> String {
    if is_special && value.starts_with('<') && value.ends_with('>') {
        value.to_string()
    } else {
        lark_quote(value)
    }
}

/// Build a Lark grammar that wraps a tool call JSON schema between start/end markers.
pub fn build_tool_call_lark_grammar(
    schema: &Value,
    start: &str,
    end: &str,
    start_is_special: bool,
    end_is_special: bool,
) -> String {
    let schema_json = serde_json::to_string(schema).unwrap_or_else(|_| "{}".to_string());

    if start.is_empty() || end.is_empty() {
        return format!("start: tool\ntool: %json {schema_json}\n");
    }

    let start_lit = lark_literal(start, start_is_special);
    let end_lit = lark_literal(end, end_is_special);

    format!(
        "start: {start_lit} _WS? tool _WS? {end_lit}\n\
         tool: %json {schema_json}\n\
         _WS: /[ \\t\\r\\n]+/\n"
    )
}

/// Build a Lark grammar for QwenCoder-style function/parameter tags with JSON values.
pub fn build_function_tag_lark_grammar(
    tools: &[Tool],
    start: &str,
    end: &str,
    start_is_special: bool,
    end_is_special: bool,
) -> String {
    let mut rules: Vec<String> = Vec::new();
    let start_tag = if start.is_empty() {
        None
    } else {
        Some(lark_literal(start, start_is_special))
    };
    let end_tag = if end.is_empty() {
        None
    } else {
        Some(lark_literal(end, end_is_special))
    };

    let tool_rule_names: Vec<String> = (0..tools.len()).map(|i| format!("tool_{i}")).collect();
    let toolcall_rule = if tool_rule_names.is_empty() {
        "toolcall:".to_string()
    } else {
        format!("toolcall: {}", tool_rule_names.join(" | "))
    };

    if let (Some(start_lit), Some(end_lit)) = (start_tag.as_ref(), end_tag.as_ref()) {
        rules.push(format!("start: {start_lit} _WS? toolcall _WS? {end_lit}"));
    } else {
        rules.push("start: toolcall".to_string());
    }

    rules.push(toolcall_rule);

    for (tool_idx, tool) in tools.iter().enumerate() {
        let func_start = lark_quote(&format!("<‌function={}>", tool.function.name));
        let func_end = lark_quote("<‌/function>");

        let params_schema = &tool.function.parameters;
        let props = params_schema.get("properties").and_then(|p| p.as_object());
        let defs = params_schema.get("$defs").cloned();
        let definitions = params_schema.get("definitions").cloned();

        let mut param_rule_names = Vec::new();

        if let Some(props) = props {
            for (param_idx, (param_name, schema)) in props.iter().enumerate() {
                let param_tag = lark_quote(&format!("<‌parameter={}>", param_name));
                let param_end = lark_quote("<‌/parameter>");
                let value_rule = format!("value_{tool_idx}_{param_idx}");
                let param_rule = format!("param_{tool_idx}_{param_idx}");
                let schema_with_defs = if defs.is_some() || definitions.is_some() {
                    if let Some(obj) = schema.as_object() {
                        let mut merged = obj.clone();
                        if let Some(ref defs_val) = defs {
                            merged
                                .entry("$defs".to_string())
                                .or_insert_with(|| defs_val.clone());
                        }
                        if let Some(ref defs_val) = definitions {
                            merged
                                .entry("definitions".to_string())
                                .or_insert_with(|| defs_val.clone());
                        }
                        serde_json::Value::Object(merged)
                    } else {
                        let mut merged = serde_json::Map::new();
                        merged.insert("allOf".to_string(), json!([schema.clone()]));
                        if let Some(ref defs_val) = defs {
                            merged.insert("$defs".to_string(), defs_val.clone());
                        }
                        if let Some(ref defs_val) = definitions {
                            merged.insert("definitions".to_string(), defs_val.clone());
                        }
                        serde_json::Value::Object(merged)
                    }
                } else {
                    schema.clone()
                };

                let schema_json =
                    serde_json::to_string(&schema_with_defs).unwrap_or_else(|_| "{}".to_string());

                rules.push(format!("{value_rule}: %json {schema_json}"));
                rules.push(format!(
                    "{param_rule}: {param_tag} {value_rule} {param_end}"
                ));
                param_rule_names.push(param_rule);
            }
        }

        let params_expr = if param_rule_names.is_empty() {
            String::new()
        } else {
            format!("({})*", param_rule_names.join(" | "))
        };

        if params_expr.is_empty() {
            rules.push(format!("tool_{tool_idx}: {func_start} _WS? {func_end}"));
        } else {
            rules.push(format!(
                "tool_{tool_idx}: {func_start} _WS? {params_expr} _WS? {func_end}"
            ));
        }
    }

    rules.push("_WS: /[ \\t\\r\\n]+/".to_string());
    rules.join("\n")
}

/// Builder for creating JSON Schema objects
#[derive(Debug, Clone, Default)]
pub struct SchemaBuilder {
    schema_type: String,
    properties: HashMap<String, Value>,
    required: Vec<String>,
    description: Option<String>,
    additional_properties: Option<bool>,
}

impl SchemaBuilder {
    /// Create a new object schema builder
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
            description: None,
            additional_properties: None,
        }
    }

    /// Add a description to the schema
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a string property
    pub fn string_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "string",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a number property
    pub fn number_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "number",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an integer property
    pub fn integer_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "integer",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a boolean property
    pub fn boolean_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "boolean",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an array property
    pub fn array_prop(
        mut self,
        name: impl Into<String>,
        items_type: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "array",
                "items": { "type": items_type.into() },
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an enum property
    pub fn enum_prop(
        mut self,
        name: impl Into<String>,
        values: Vec<&str>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "string",
                "enum": values,
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a custom property with full schema
    pub fn custom_prop(mut self, name: impl Into<String>, schema: Value, required: bool) -> Self {
        let name = name.into();
        self.properties.insert(name.clone(), schema);
        if required {
            self.required.push(name);
        }
        self
    }

    /// Disallow additional properties
    pub fn no_additional_properties(mut self) -> Self {
        self.additional_properties = Some(false);
        self
    }

    /// Build the final JSON Schema
    pub fn build(self) -> Value {
        let mut schema = json!({
            "type": self.schema_type,
            "properties": self.properties,
            "required": self.required
        });

        if let Some(desc) = self.description {
            schema["description"] = json!(desc);
        }

        if let Some(additional) = self.additional_properties {
            schema["additionalProperties"] = json!(additional);
        }

        schema
    }
}

/// Common tool schemas for built-in tools
pub mod common {
    use super::*;

    /// Calculator tool schema
    pub fn calculator_schema() -> Value {
        SchemaBuilder::object()
            .description("Evaluate a mathematical expression")
            .string_prop(
                "expression",
                "The mathematical expression to evaluate",
                true,
            )
            .build()
    }

    /// Web search tool schema
    pub fn web_search_schema() -> Value {
        SchemaBuilder::object()
            .description("Search the web for information")
            .string_prop("query", "The search query", true)
            .integer_prop("max_results", "Maximum number of results to return", false)
            .build()
    }

    /// Get current time tool schema
    pub fn get_time_schema() -> Value {
        SchemaBuilder::object()
            .description("Get current date and time")
            .string_prop(
                "timezone",
                "Timezone (e.g., 'UTC', 'America/New_York')",
                false,
            )
            .build()
    }

    /// Code execution tool schema
    pub fn code_execution_schema() -> Value {
        SchemaBuilder::object()
            .description("Execute code in a sandboxed environment")
            .enum_prop(
                "language",
                vec!["python", "javascript", "rust"],
                "Programming language",
                true,
            )
            .string_prop("code", "The code to execute", true)
            .build()
    }
}

/// Build a Lark grammar for choice constraints (structured outputs choice field)
pub fn build_choice_lark_grammar(choices: &[String]) -> Result<String, String> {
    if choices.is_empty() {
        return Err("structured_outputs.choice must include at least one option".to_string());
    }

    let mut parts = Vec::with_capacity(choices.len());
    for choice in choices {
        if choice.is_empty() {
            return Err("structured_outputs.choice cannot contain empty strings".to_string());
        }
        parts.push(lark_quote(choice));
    }

    Ok(format!("start: {}\\n", parts.join(" | ")))
}

/// Normalize a tag string for structural_tag parsing
fn normalize_tag_pair(tag: &str) -> Result<(String, String), String> {
    let trimmed = tag.trim();
    if trimmed.is_empty() {
        return Err("structured_outputs.structural_tag.tag cannot be empty".to_string());
    }

    if trimmed.starts_with('<') && trimmed.ends_with('>') {
        let inner = trimmed
            .trim_start_matches('<')
            .trim_end_matches('>')
            .trim_start_matches('/');
        if inner.is_empty() {
            return Err("structured_outputs.structural_tag.tag is invalid".to_string());
        }
        let start = if trimmed.starts_with("</") {
            format!("<{}>", inner)
        } else {
            trimmed.to_string()
        };
        let end = format!("</{}>", inner);
        Ok((start, end))
    } else {
        Ok((format!("<{}>", trimmed), format!("</{}>", trimmed)))
    }
}

/// Parse structural_tag for structured outputs
pub fn parse_structural_tag(value: &Value) -> Result<(String, String, Value), String> {
    let obj = value.as_object().ok_or_else(|| {
        "structured_outputs.structural_tag must be an object".to_string()
    })?;

    let schema = obj.get("schema").cloned().ok_or_else(|| {
        "structured_outputs.structural_tag.schema is required".to_string()
    })?;

    let start = obj.get("start_tag").or_else(|| obj.get("start")).or_else(|| obj.get("tag"));
    let end = obj.get("end_tag").or_else(|| obj.get("end"));

    let (start_tag, end_tag) = match (start, end) {
        (Some(start_val), Some(end_val)) => {
            let start = start_val.as_str().ok_or_else(|| {
                "structured_outputs.structural_tag.start_tag must be a string".to_string()
            })?;
            let end = end_val.as_str().ok_or_else(|| {
                "structured_outputs.structural_tag.end_tag must be a string".to_string()
            })?;
            (start.to_string(), end.to_string())
        }
        (Some(tag), None) if obj.contains_key("tag") => normalize_tag_pair(tag.as_str().ok_or_else(|| "structured_outputs.structural_tag.tag must be a string".to_string())?)?,
        _ => {
            return Err("structured_outputs.structural_tag requires tag or start_tag/end_tag".to_string());
        }
    };

    Ok((start_tag, end_tag, schema))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_schema_for_llguidance_strips_format() {
        let schema = json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "nested": {"type": "object", "properties": {"id": {"type": "string", "format": "uuid"}}}
            }
        });
        let sanitized = sanitize_schema_for_llguidance(&schema);
        assert!(sanitized["properties"]["url"].get("format").is_none());
        assert!(sanitized["properties"]["nested"]["properties"]["id"].get("format").is_none());
    }

    #[test]
    fn test_build_choice_lark_grammar_empty() {
        let result = build_choice_lark_grammar(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_choice_lark_grammar_single() {
        let result = build_choice_lark_grammar(&["yes".to_string()]).unwrap();
        assert!(result.contains("\"yes\""));
    }

    #[test]
    fn test_build_choice_lark_grammar_multiple() {
        let result = build_choice_lark_grammar(&["option1".to_string(), "option2".to_string()]).unwrap();
        assert!(result.contains("\"option1\""));
        assert!(result.contains("\"option2\""));
    }

    #[test]
    fn test_build_choice_lark_grammar_empty_string() {
        let result = build_choice_lark_grammar(&["".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_structural_tag_missing_schema() {
        let value = json!({});
        let result = parse_structural_tag(&value);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_structural_tag_start_end() {
        let value = json!({
            "start_tag": "<tool>",
            "end_tag": "</tool>",
            "schema": {"type": "object"}
        });
        let result = parse_structural_tag(&value);
        assert!(result.is_ok());
        let (start, end, schema) = result.unwrap();
        assert_eq!(start, "<tool>");
        assert_eq!(end, "</tool>");
        assert_eq!(schema, json!({"type": "object"}));
    }

    #[test]
    fn test_parse_structural_tag_tag() {
        let value = json!({
            "tag": "<tool>",
            "schema": {"type": "object"}
        });
        let result = parse_structural_tag(&value);
        assert!(result.is_ok());
        let (start, end, _) = result.unwrap();
        assert_eq!(start, "<tool>");
        assert_eq!(end, "</tool>");
    }

    #[test]
    fn test_parse_structural_tag_invalid() {
        let value = json!({
            "schema": {"type": "object"}
        });
        let result = parse_structural_tag(&value);
        assert!(result.is_err());
    }

    #[test]
    fn test_lark_quote_escapes_special_chars() {
        let result = lark_quote("test\"value");
        assert!(result.contains("test\\\"value"));
    }

    #[test]
    fn test_lark_literal_special_tags() {
        let result = lark_literal("<tool>", true);
        assert_eq!(result, "<tool>");
    }

    #[test]
    fn test_lark_literal_regular_string() {
        let result = lark_literal("regular", false);
        assert!(result.contains("\"regular\""));
    }
}
