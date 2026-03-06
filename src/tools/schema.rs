// src/tools/schema.rs
//! JSON Schema utilities for tool parameters
//!
//! Provides helpers for working with JSON Schema in tool definitions.

use crate::tools::Tool;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use crate::utils::guidance::{lark_ws_regex, TopLevelGrammarExt, GrammarError, GrammarResult};
use llguidance::api::TopLevelGrammar;

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
/// Sanitize string for Lark grammar - only allow ASCII characters
fn lark_quote(value: &str) -> String {
    // Strip non-ASCII characters to prevent grammar parser errors
    let sanitized: String = value
        .chars()
        .filter(|c| c.is_ascii())
        .collect();
    let escaped = sanitized.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{}\"", escaped)
}

/// Convert token IDs to Lark special token syntax <[token_id]>
/// This is used when the tokenizer has canonical tokenization for the tag
fn lark_special_token(token_ids: &HashSet<u32>) -> String {
    if token_ids.is_empty() {
        return String::new();
    }
    // Join multiple token IDs with |
    let ids: Vec<String> = token_ids.iter().map(|id| format!("[{}]", id)).collect();
    format!("<{}>", ids.join(","))
}

fn lark_literal(value: &str, is_special: bool) -> String {
    if is_special && value.starts_with('<') && value.ends_with('>') {
        // Only allow ASCII special tags
        let sanitized: String = value
            .chars()
            .filter(|c| c.is_ascii())
            .collect();
        sanitized
    } else {
        lark_quote(value)
    }
}

/// Builder for constructing tool call grammars
pub struct ToolGrammarBuilder {
    tools: Vec<Tool>,
    start_tag: String,
    end_tag: String,
    start_is_special: bool,
    end_is_special: bool,
    start_token_ids: Option<HashSet<u32>>,
    end_token_ids: Option<HashSet<u32>>,
}

impl ToolGrammarBuilder {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            start_tag: String::new(),
            end_tag: String::new(),
            start_is_special: false,
            end_is_special: false,
            start_token_ids: None,
            end_token_ids: None,
        }
    }

    pub fn tools(mut self, tools: &[Tool]) -> Self {
        self.tools.extend(tools.iter().cloned());
        self
    }

    pub fn start_tag(mut self, tag: impl Into<String>) -> Self {
        self.start_tag = tag.into();
        self
    }

    pub fn end_tag(mut self, tag: impl Into<String>) -> Self {
        self.end_tag = tag.into();
        self
    }

    pub fn start_is_special(mut self, special: bool) -> Self {
        self.start_is_special = special;
        self
    }

    pub fn end_is_special(mut self, special: bool) -> Self {
        self.end_is_special = special;
        self
    }

    pub fn start_token_ids(mut self, ids: Option<HashSet<u32>>) -> Self {
        self.start_token_ids = ids;
        self
    }

    pub fn end_token_ids(mut self, ids: Option<HashSet<u32>>) -> Self {
        self.end_token_ids = ids;
        self
    }

    pub fn build_json(self) -> llguidance::api::TopLevelGrammar {
        let mut rules = Vec::new();

        for tool in &self.tools {
            let tool_name = tool.function.name.replace("-", "_");
            let schema_str = serde_json::to_string(&tool.function.parameters).unwrap_or_default();
            rules.push(format!("obj_{tool_name}: %json {schema_str}"));
        }

        let start_tag = self.get_tag_or_token_id(&self.start_tag, &self.start_token_ids, self.start_is_special);
        let end_tag = self.get_tag_or_token_id(&self.end_tag, &self.end_token_ids, self.end_is_special);

        rules.push("start: tool_call".to_string());
        rules.push(format!("tool_call: {} tool_obj {}", start_tag, end_tag));
        rules.push("tool_obj: %json {\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"arguments\":{\"type\":\"object\"}},\"required\":[\"name\",\"arguments\"]}".to_string());
        rules.push("json_array: \"[\" obj (\",\" obj)* \"]\"".to_string());

        if rules.len() <= 4 {
            rules.push("obj: %json {\"type\": \"object\"}".to_string());
        } else {
            rules.extend(self.tools.iter().enumerate().map(|(i, t)| {
                let name = t.function.name.replace("-", "_");
                format!("obj_{name}: %json {}", serde_json::to_string(&t.function.parameters).unwrap_or_default())
            }));

            let obj_names = self.tools.iter().map(|t| {
                format!("obj_{}", t.function.name.replace("-", "_"))
            }).collect::<Vec<_>>().join(" | ");
            rules.push(format!("obj: {}", obj_names));
        }

        rules.push(format!("ws: {}", lark_ws_regex()));

        let lark = rules.join("\n") + "\n";
        llguidance::api::TopLevelGrammar::from_lark_utf8(&lark)
    }

    /// Build Lark expression for valid XML parameter content
    /// This replaces the broken %json approach with proper text-based patterns
    // Fixed build_xml_value_expression - returns proper regex patterns without leading colon
    fn build_xml_value_expression(schema: &serde_json::Value) -> String {
        let param_type = schema.get("type").and_then(|t| t.as_str()).unwrap_or("string");
        
        match param_type {
            "string" => {
                // Match any text content except the closing tag
                r"/(?s:[^<]|<(?!\/parameter>))+/".to_string()
            }
            "integer" => r"/-?\d+/".to_string(),
            "number" => r"/-?\d+(\.\d+)?/".to_string(),
            "boolean" => r"/^(true|false)$/".to_string(),
            "array" => r"/\[[^\]]*\]/".to_string(),
            "object" => r"/\{[^\}]*\}/".to_string(),
            _ => r"/(?s:.*)/".to_string(),
        }
    }

    // Fixed build_xml method - pure XML text patterns, no JSON
    pub fn build_xml(self) -> llguidance::api::TopLevelGrammar {
        let mut rules: Vec<String> = Vec::new();

        // Build envelope tag using token IDs when available
        let envelope_start_tag = self.get_envelope_tag(&self.start_tag, &self.start_token_ids, self.start_is_special);
        let envelope_end_tag = self.get_envelope_tag(&self.end_tag, &self.end_token_ids, self.end_is_special);

        let tool_rule_names: Vec<String> = (0..self.tools.len()).map(|i| format!("tool_{i}")).collect();
        rules.push("start: tool_call".to_string());
        rules.push(format!("tool_call: {} tool_content {}", envelope_start_tag, envelope_end_tag));
        
        // Get required params from schema
        let get_required_params = |params_schema: &serde_json::Value| -> Vec<String> {
            params_schema.get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default()
        };

        for (tool_idx, tool) in self.tools.iter().enumerate() {
            let tool_name_ascii: String = tool.function.name.chars().filter(|c| c.is_ascii()).collect();
            let func_start = lark_quote(&format!("<function=\"{}\">", tool_name_ascii));
            let func_end = lark_quote("</function>");
            let params_schema = &tool.function.parameters;
            let props = params_schema.get("properties").and_then(|p| p.as_object());
            let required_params = get_required_params(params_schema);

            if let Some(props) = props {
                let mut param_rules_vec: Vec<String> = Vec::new();
                
                for (param_idx, (param_name, schema)) in props.iter().enumerate() {
                    let param_name_ascii: String = param_name.chars().filter(|c| c.is_ascii()).collect();
                    let param_tag = lark_quote(&format!("<parameter=\"{}\">", param_name_ascii));
                    let param_end = lark_quote("</parameter>");
                    let value_rule = format!("value_{tool_idx}_{param_idx}");
                    let param_rule = format!("param_{tool_idx}_{param_idx}");

                    // Determine the Lark expression for valid XML content based on schema type
                    let value_expr = Self::build_xml_value_expression(schema);
                    rules.push(format!("{value_rule}: {value_expr}"));
                    rules.push(format!("{param_rule}: {param_tag} {value_rule} {param_end}"));
                    
                    // Add to param_rules_vec with ? for optional, bare for required
                    if required_params.contains(param_name) {
                        param_rules_vec.push(param_rule.clone());
                    } else {
                        param_rules_vec.push(format!("({param_rule})?"));
                    }
                }
                
                let params_expr = param_rules_vec.join(" ");
                rules.push(format!("tool_{tool_idx}: {func_start} {params_expr} {func_end}"));
            } else {
                // No parameters - just function tags
                rules.push(format!("tool_{tool_idx}: {func_start} {func_end}"));
            }
        }

        // Build tool_content with alternation of all tools
        let tool_variants = tool_rule_names.join(" | ");
        rules.push(format!("tool_content: {tool_variants}"));
        // rules.push(format!("_WS: {}", lark_ws_regex()));
        
        let lark = rules.join("\n") + "\n";
        TopLevelGrammar::from_lark_utf8(&lark)
    }


    pub fn old_build_xml(self) -> llguidance::api::TopLevelGrammar {
        let mut rules: Vec<String> = Vec::new();

        // Build envelope tag using token IDs when available
        let envelope_start_tag = self.get_envelope_tag(&self.start_tag, &self.start_token_ids, self.start_is_special);
        let envelope_end_tag = self.get_envelope_tag(&self.end_tag, &self.end_token_ids, self.end_is_special);

        let tool_rule_names: Vec<String> = (0..self.tools.len()).map(|i| format!("tool_{i}")).collect();
        rules.push("start: tool_call".to_string());
        rules.push(format!("tool_call: {} tool_content {}", envelope_start_tag, envelope_end_tag));

        for (tool_idx, tool) in self.tools.iter().enumerate() {
            let tool_name_ascii: String = tool.function.name.chars().filter(|c| c.is_ascii()).collect();
            let func_start = lark_quote(&format!("<function={}>", tool_name_ascii));
            let func_end = lark_quote("</function>");

            let params_schema = &tool.function.parameters;
            let props = params_schema.get("properties").and_then(|p| p.as_object());
            let defs = params_schema.get("$defs").cloned();
            let definitions = params_schema.get("definitions").cloned();

            let mut param_rule_names = Vec::new();

            if let Some(props) = props {
                for (param_idx, (param_name, schema)) in props.iter().enumerate() {
                    let param_name_ascii: String = param_name.chars().filter(|c| c.is_ascii()).collect();
                    let param_tag = lark_quote(&format!("<parameter={}>", param_name_ascii));
                    let param_end = lark_quote("</parameter>");
                    let value_rule = format!("value_{tool_idx}_{param_idx}");
                    let param_rule = format!("param_{tool_idx}_{param_idx}");
                    let schema_with_defs = if defs.is_some() || definitions.is_some() {
                        if let Some(obj) = schema.as_object() {
                            let mut merged = obj.clone();
                            if let Some(ref defs_val) = defs {
                                merged.entry("$defs".to_string()).or_insert_with(|| defs_val.clone());
                            }
                            if let Some(ref defs_val) = definitions {
                                merged.entry("definitions".to_string()).or_insert_with(|| defs_val.clone());
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

                    let schema_json = serde_json::to_string(&schema_with_defs).unwrap_or_else(|_| "{}".to_string());

                    rules.push(format!("{value_rule}: %json {schema_json}"));
                    rules.push(format!("{param_rule}: {param_tag} {value_rule} {param_end}"));
                    param_rule_names.push(param_rule);
                }
            }

            // Determine which parameters are required from the schema
            let required_params: Vec<&str> = params_schema.get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();

            // Build required parameters first (no *, must be present)
            let required_expr: String = if required_params.is_empty() {
                String::new()
            } else {
                let required_rules: Vec<String> = param_rule_names.iter()
                    .filter(|r| {
                        let param_idx = r.split('_').nth(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        if let Some(props) = props {
                            props.keys().nth(param_idx).map(|k| required_params.contains(&k.as_str())).unwrap_or(false)
                        } else {
                            false
                        }
                    })
                    .cloned()
                    .collect();
                if required_rules.is_empty() {
                    String::new()
                } else {
                    format!("{} ", required_rules.join(" "))
                }
            };

            // Build optional parameters (with *, can be omitted)
            let optional_expr: String = {
                let optional_rules: Vec<String> = param_rule_names.iter()
                    .filter(|r| {
                        let param_idx = r.split('_').nth(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        if let Some(props) = props {
                            props.keys().nth(param_idx).map(|k| !required_params.contains(&k.as_str())).unwrap_or(true)
                        } else {
                            true
                        }
                    })
                    .cloned()
                    .collect();
                if optional_rules.is_empty() {
                    String::new()
                } else {
                    format!("({})* ", optional_rules.join(" | "))
                }
            };

            let params_expr = format!("{}{}", required_expr, optional_expr).trim().to_string();

            if params_expr.is_empty() {
                rules.push(format!("tool_{tool_idx}: {func_start} _WS? {func_end}"));
            } else {
                rules.push(format!("tool_{tool_idx}: {func_start} _WS? {params_expr} _WS? {func_end}"));
            }
        }

        // Add the tool_content rule that references all tool variants
        let tool_variants = tool_rule_names.join(" | ");
        rules.push(format!("tool_content: {}", tool_variants));

        rules.push(format!("_WS: {}", lark_ws_regex()));
        let lark = rules.join("\n") + "\n";
        TopLevelGrammar::from_lark_utf8(&lark)
    }

    /// Get envelope tag (start/end) using token IDs when available, falling back to string literals
    fn get_envelope_tag(&self, tag: &str, token_ids: &Option<HashSet<u32>>, is_special: bool) -> String {
        if let Some(ids) = token_ids {
            if !ids.is_empty() {
                return lark_special_token(ids);
            }
        }

        if is_special && tag.starts_with('<') && tag.ends_with('>') {
            // Only allow ASCII special tags
            let sanitized: String = tag.chars().filter(|c| c.is_ascii()).collect();
            sanitized
        } else {
            lark_quote(tag)
        }
    }

    fn get_tag_or_token_id(&self, tag: &str, token_ids: &Option<HashSet<u32>>, is_special: bool) -> String {
        if let Some(ids) = token_ids {
            if !ids.is_empty() {
                return format!("<{}>", ids.iter().map(|id| format!("[{}]", id)).collect::<Vec<_>>().join(","));
            }
        }

        if is_special && tag.starts_with('<') && tag.ends_with('>') {
            tag.to_string()
        } else {
            lark_quote(tag)
        }
    }
}

/// Build a Lark grammar for QwenCoder-style function/parameter tags with JSON values.
/// Used for models like Qwen3-Coder that use XML-style tool call envelopes.
pub fn build_xml_tool_lark_grammar(
    tools: &[Tool],
    start: &str,
    end: &str,
    start_is_special: bool,
    end_is_special: bool,
    start_token_ids: Option<&HashSet<u32>>,
    end_token_ids: Option<&HashSet<u32>>,
) -> llguidance::api::TopLevelGrammar {
    ToolGrammarBuilder::new()
        .tools(tools)
        .start_tag(start)
        .end_tag(end)
        .start_is_special(start_is_special)
        .end_is_special(end_is_special)
        .start_token_ids(start_token_ids.cloned())
        .end_token_ids(end_token_ids.cloned())
        .build_xml()
}

fn build_xml_tool_lark_string(
    tools: &[Tool],
    start: &str,
    end: &str,
    start_is_special: bool,
    end_is_special: bool,
    start_token_ids: Option<&HashSet<u32>>,
    end_token_ids: Option<&HashSet<u32>>,
) -> String {
    let mut rules: Vec<String> = Vec::new();

    // Use token IDs for XML envelope tags when available
    let _xml_start_tag = if let Some(ids) = start_token_ids {
        if ids.is_empty() {
            lark_literal(start, start_is_special)
        } else {
            lark_special_token(ids)
        }
    } else {
        lark_literal(start, start_is_special)
    };
    let _xml_end_tag = if let Some(ids) = end_token_ids {
        if ids.is_empty() {
            lark_literal(end, end_is_special)
        } else {
            lark_special_token(ids)
        }
    } else {
        lark_literal(end, end_is_special)
    };

    let tool_rule_names: Vec<String> = (0..tools.len()).map(|i| format!("tool_{i}")).collect();
    rules.push("start: tool_call".to_string());
    let toolcall_rule = if tool_rule_names.is_empty() {
        "tool_call:".to_string()
    } else {
        format!("tool_call: {}", tool_rule_names.join(" | "))
    };

    rules.push(toolcall_rule);

    for (tool_idx, tool) in tools.iter().enumerate() {
        // Sanitize tool name to ASCII-only for grammar safety
        let tool_name_ascii: String = tool.function.name.chars().filter(|c| c.is_ascii()).collect();
        let func_start = lark_quote(&format!("<function={}>", tool_name_ascii));
        let func_end = lark_quote("</function>");

        let params_schema = &tool.function.parameters;
        let props = params_schema.get("properties").and_then(|p| p.as_object());
        let defs = params_schema.get("$defs").cloned();
        let definitions = params_schema.get("definitions").cloned();

        let mut param_rule_names = Vec::new();

        if let Some(props) = props {
            for (param_idx, (param_name, schema)) in props.iter().enumerate() {
                // Sanitize parameter name to ASCII-only for grammar safety
                let param_name_ascii: String = param_name.chars().filter(|c| c.is_ascii()).collect();
                let param_tag = lark_quote(&format!("<parameter={}>", param_name_ascii));
                let param_end = lark_quote("</parameter>");
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

    rules.push(format!("_WS: {}", lark_ws_regex()).to_string());
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
pub fn build_choice_lark_grammar(choices: &[String]) -> GrammarResult<llguidance::api::TopLevelGrammar> {
    if choices.is_empty() {
        return Err(GrammarError::InvalidGrammar("structured_outputs.choice must include at least one option".to_string()));
    }

    let mut parts = Vec::with_capacity(choices.len());
    for choice in choices {
        if choice.is_empty() {
            return Err(GrammarError::InvalidGrammar("structured_outputs.choice cannot contain empty strings".to_string()));
        }
        parts.push(lark_quote(choice));
    }

    let body = parts.join(" | ");
    let lark_string = format!("start: {}\n", body);
    Ok(llguidance::api::TopLevelGrammar::from_lark_utf8(&lark_string))
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

/// Convert a Value schema to a Vec of Tool objects using ToolBuilder
/// The schema should be an object where keys are tool names and values are tool schemas
pub fn schema_to_tools(schema: &Value) -> Vec<Tool> {
    let mut tools = Vec::new();
    if let Value::Object(obj) = schema {
        for (name, tool_schema) in obj {
            if let Value::Object(props) = tool_schema {
                if let Some(params) = props.get("parameters") {
                    let builder = crate::tools::ToolBuilder::new(name.clone(), "".to_string())
                        .parameters_schema(params.clone());
                    tools.push(builder.build());
                }
            }
        }
    }
    tools
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::guidance::{chat_text_expression_with_eos, get_lark_from_top_level_grammar, merge_top_level_grammars};
    use llguidance::api::TopLevelGrammar;

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

    #[test]
    fn test_lark_special_token_single_id() {
        let mut ids = HashSet::new();
        ids.insert(151657);
        let result = lark_special_token(&ids);
        assert_eq!(result, "<[151657]>");
    }

    #[test]
    fn test_lark_special_token_multiple_ids() {
        let mut ids = HashSet::new();
        ids.insert(151657);
        ids.insert(151658);
        let result = lark_special_token(&ids);
        assert!(result.contains("[151657]"));
        assert!(result.contains("[151658]"));
    }

    #[test]
    fn test_lark_special_token_empty() {
        let ids = HashSet::new();
        let result = lark_special_token(&ids);
        assert_eq!(result, "");
    }

    #[test]
    fn test_build_xml_tool_lark_grammar_qwen3_coder_required_only() {
        // Test Qwen3-Coder XML tool format with required attributes only
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
        ];
        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        println!("{}", &lark_str);

        // Qwen3Coder uses XML format with start: tool_call
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("<function=search>"), "Should contain function tag");
        assert!(lark_str.contains("tool_0:"), "Should contain tool_0 rule");
    }

    #[test]
    fn test_build_xml_tool_lark_grammar_qwen3_coder_optional() {
        // Test Qwen3-Coder XML tool format with optional attributes
        let tools = vec![
            crate::tools::ToolBuilder::new("get_weather".to_string(), "Get weather".to_string())
                .param("city", "string", "City name", true)
                .param("units", "string", "Temperature units (optional)", false)
                .build(),
        ];
        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);

        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("<function=get_weather>"), "Should contain function tag");
        assert!(lark_str.contains("city"), "Should contain city parameter");
        assert!(lark_str.contains("units"), "Should contain optional units parameter");
    }

    #[test]
    fn test_merge_top_level_grammars_qwen3_coder_text_and_xml_tool() {
        // Test tool_call | TEXT merging for Qwen3-Coder (XML format)
        // Use chat_text_expression_with_eos with empty EOS to get proper text pattern
        let text_gram = TopLevelGrammar::from_lark(chat_text_expression_with_eos(&[21, 43]));
        let tool_gram = build_xml_tool_lark_grammar(
            &vec![crate::tools::ToolBuilder::new("search".to_string(), "Search".to_string())
                .param("query", "string", "Query", true)
                .build()],
            "", "", false, false, None, None
        );

        // Use None for default separator (|)
        let result = merge_top_level_grammars(vec![text_gram, tool_gram], None, None);
        let lark_str = get_lark_from_top_level_grammar(&result);

        assert!(lark_str.contains("start: ( text_with_eos | tool_call )+"), "Expected ( text_with_eos | tool_call )+ but got {}", &lark_str);
        assert!(!lark_str.contains("rule_0:"), "Should not contain rule_0 indirection");
        assert!(!lark_str.contains("rule_1:"), "Should not contain rule_1 indirection");
    }
 
    #[test]
    fn test_build_xml_tool_lark_grammar_qwen3_coder_deep_parameters() {
        // Test Qwen3-Coder XML tool format with nested/complex parameters
        let tools = vec![
            crate::tools::ToolBuilder::new("edit_file".to_string(), "Edit a file with complex parameters".to_string())
                .param("file_path", "string", "Path to the file", true)
                .param("old_string", "string", "String to replace", true)
                .param("new_string", "string", "Replacement string", true)
                .param("replace_all", "boolean", "Replace all occurrences", false)
                .build(),
        ];
        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);
        println!("XML Grammar:\n{}", &lark_str);

        // Verify the grammar contains XML structure
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        // Note: <function=...> uses U+200C (zero-width non-joiner) which is invisible
        assert!(lark_str.contains("function="), "Should contain function tag with attribute");

        // Verify all parameter tags are present
        // Note: <parameter=...> uses U+200C (zero-width non-joiner) which is invisible
        assert!(lark_str.contains("parameter=file_path"), "Should contain file_path parameter tag");
        assert!(lark_str.contains("parameter=old_string"), "Should contain old_string parameter tag");
        assert!(lark_str.contains("parameter=new_string"), "Should contain new_string parameter tag");
        assert!(lark_str.contains("parameter=replace_all"), "Should contain replace_all parameter tag");

        // Verify parameter rules reference the correct types
        assert!(lark_str.contains("param_0_0:"), "Should have param_0_0 rule for first param");
        assert!(lark_str.contains("param_0_1:"), "Should have param_0_1 rule for second param");
        assert!(lark_str.contains("param_0_2:"), "Should have param_0_2 rule for third param");
        assert!(lark_str.contains("param_0_3:"), "Should have param_0_3 rule for fourth param");

        // Verify tool rule has all parameters
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
    }

    #[test]
    fn test_xml_grammar_required_params_no_wrapper() {
        // Test that XML grammar puts required params directly without (...) * wrapper
        let tools = vec![crate::tools::ToolBuilder::new("search_tool".to_string(), "Search tool".to_string())
            .param("query", "string", "Search query", true)      // REQUIRED - should appear as bare rule reference
            .build()];

        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);

        // Required param rule should appear directly in tool_0 (no parentheses/asterisk around it)
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
        assert!(lark_str.contains("param_0"), "Should have parameter rules");

        // The required param should NOT be wrapped in (...) * pattern
        // Look for the pattern where required params appear as direct references: "param_X Y" not "(param_X | ...)*"
    }

    #[test]
    fn test_xml_grammar_optional_params_wrapped() {
        // Test that XML grammar wraps optional params with (...) * syntax
        let tools = vec![crate::tools::ToolBuilder::new("mixed_tool".to_string(), "Mixed params".to_string())
            .param("required_param", "string", "Required", true)      // REQUIRED
            .param("optional_param", "string", "Optional", false)     // OPTIONAL
            .build()];

        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);
        let lark_str = get_lark_from_top_level_grammar(&grammar);

        println!("XML Grammar for mixed tool:\n{}", lark_str);

        // Optional parameters should appear in a (...) * pattern when there are multiple options
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
    }

    #[test]
    fn test_xml_tool_call_structure_validates() {
        // Full end-to-end: verify XML grammar produces valid llguidance TopLevelGrammar structure
        let tools = vec![crate::tools::ToolBuilder::new("formatter".to_string(), "Formatter".to_string())
            .param("text", "string", "Text to format", true)
            .build()];

        let grammar = build_xml_tool_lark_grammar(&tools, "", "", false, false, None, None);

        // Grammar should have at least one sub-grammar (the tool rules)
        assert!(grammar.grammars.len() > 0, "Should have generated grammars");
    }

    // === ToolGrammarBuilder JSON Mode Tests ===

    #[test]
    fn test_tool_grammar_builder_build_json_single_tool() {
        // Test ToolGrammarBuilder.build_json() with a single tool
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify basic structure
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("obj_search:"), "Should contain obj_search rule");
        assert!(lark_str.contains("query"), "Should contain query parameter");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_multiple_tools() {
        // Test ToolGrammarBuilder.build_json() with multiple tools
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
            crate::tools::ToolBuilder::new("weather".to_string(), "Get weather".to_string())
                .param("city", "string", "City name", true)
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify all tools are present
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("obj_search:"), "Should contain obj_search rule");
        assert!(lark_str.contains("obj_weather:"), "Should contain obj_weather rule");
        // Verify obj alternation includes both tools
        assert!(lark_str.contains("obj: obj_search | obj_weather"), "Should have obj alternation");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_with_token_ids() {
        // Test ToolGrammarBuilder.build_json() with token IDs
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
        ];
        let mut start_ids = HashSet::new();
        start_ids.insert(151657);
        let mut end_ids = HashSet::new();
        end_ids.insert(151658);
        
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("")
            .end_tag("")
            .start_is_special(false)
            .end_is_special(false)
            .start_token_ids(Some(start_ids))
            .end_token_ids(Some(end_ids))
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify token IDs are used
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("<[151657]>"), "Should contain start token ID");
        assert!(lark_str.contains("<[151658]>"), "Should contain end token ID");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_with_special_tags() {
        // Test ToolGrammarBuilder.build_json() with special tags
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(true)
            .end_is_special(true)
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify special tags are used as-is
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("<tool>"), "Should contain special start tag");
        assert!(lark_str.contains("</tool>"), "Should contain special end tag");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_required_optional() {
        // Test ToolGrammarBuilder.build_json() with mix of required/optional params
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .param("max_results", "integer", "Max results", false)
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify both params are in schema, and required array is correct
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("obj_search:"), "Should contain obj_search rule");
        assert!(lark_str.contains("query"), "Should contain query parameter");
        assert!(lark_str.contains("max_results"), "Should contain max_results parameter");
        assert!(lark_str.contains("\"required\""), "Should have required array");
    }

    // === ToolGrammarBuilder XML Mode Tests ===

    #[test]
    fn test_tool_grammar_builder_build_xml_single_tool() {
        // Test ToolGrammarBuilder.build_xml() with a single tool
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify XML structure
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("tool_call:"), "Should have tool_call rule");
        assert!(lark_str.contains("function=search"), "Should contain function tag");
        assert!(lark_str.contains("parameter=query"), "Should contain parameter tag");
        assert!(lark_str.contains("param_0_0:"), "Should have param_0_0 rule");
    }

    #[test]
    fn test_tool_grammar_builder_build_xml_multiple_tools() {
        // Test ToolGrammarBuilder.build_xml() with multiple tools
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
            crate::tools::ToolBuilder::new("weather".to_string(), "Get weather".to_string())
                .param("city", "string", "City name", true)
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify all tools are present
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("tool_0:"), "Should contain tool_0 rule");
        assert!(lark_str.contains("tool_1:"), "Should contain tool_1 rule");
        assert!(lark_str.contains("tool_content:"), "Should have tool_content rule");
        // Verify tool_content has alternation
        assert!(lark_str.contains("tool_content: tool_0 | tool_1"), "Should have tool alternation");
    }

    #[test]
    fn test_tool_grammar_builder_build_xml_with_token_ids() {
        // Test ToolGrammarBuilder.build_xml() with token IDs
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
        ];
        let mut start_ids = HashSet::new();
        start_ids.insert(151657);
        let mut end_ids = HashSet::new();
        end_ids.insert(151658);
        
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("")
            .end_tag("")
            .start_is_special(false)
            .end_is_special(false)
            .start_token_ids(Some(start_ids))
            .end_token_ids(Some(end_ids))
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify token IDs are used for envelope tags
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("<[151657]>"), "Should contain start token ID");
        assert!(lark_str.contains("<[151658]>"), "Should contain end token ID");
    }

    #[test]
    fn test_tool_grammar_builder_build_xml_with_special_tags() {
        // Test ToolGrammarBuilder.build_xml() with special tags
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(true)
            .end_is_special(true)
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify special tags are used as-is
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("<tool>"), "Should contain special start tag");
        assert!(lark_str.contains("</tool>"), "Should contain special end tag");
    }

    #[test]
    fn test_tool_grammar_builder_build_xml_required_optional() {
        // Test ToolGrammarBuilder.build_xml() with mix of required/optional params
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .param("max_results", "integer", "Max results", false)
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify both params are present
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("param_0_0:"), "Should have param_0_0 rule (query - required)");
        assert!(lark_str.contains("param_0_1:"), "Should have param_0_1 rule (max_results - optional)");
        assert!(lark_str.contains("parameter=query"), "Should contain query parameter tag");
        assert!(lark_str.contains("parameter=max_results"), "Should contain max_results parameter tag");
    }

    #[test]
    fn test_tool_grammar_builder_build_xml_no_parameters() {
        // Test ToolGrammarBuilder.build_xml() with tool that has no parameters
        let tools = vec![
            crate::tools::ToolBuilder::new("hello".to_string(), "Say hello".to_string())
                .param("query", "string", "Search query", true)
                .parameters_schema(serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }))
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify tool with no parameters still generates valid grammar
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("function=hello"), "Should contain function tag");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_no_parameters() {
        // Test ToolGrammarBuilder.build_json() with tool that has no parameters
        let tools = vec![
            crate::tools::ToolBuilder::new("hello".to_string(), "Say hello".to_string())
                .parameters_schema(serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }))
                .build(),
        ];
        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify tool with no parameters still generates valid grammar
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("obj_hello:"), "Should contain obj_hello rule");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_empty_tools() {
        // Test ToolGrammarBuilder.build_json() with empty tools list
        let grammar = ToolGrammarBuilder::new()
            .tools(&[])
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify grammar is still valid with no tools
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        // With no tools, obj should be a generic object
        assert!(lark_str.contains("obj: %json"), "Should have obj rule with generic schema");
    }

    #[test]
    fn test_tool_grammar_builder_build_xml_empty_tools() {
        // Test ToolGrammarBuilder.build_xml() with empty tools list
        let grammar = ToolGrammarBuilder::new()
            .tools(&[])
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        
        // Verify grammar is still valid with no tools
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("tool_content:"), "Should have tool_content rule");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_structure_validates() {
        // Full end-to-end: verify JSON grammar produces valid llguidance TopLevelGrammar structure
        let tools = vec![crate::tools::ToolBuilder::new("calculator".to_string(), "Calculator".to_string())
            .param("expression", "string", "Math expression", true)
            .build()];

        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_json();

        // Grammar should have at least one sub-grammar
        assert!(grammar.grammars.len() > 0, "Should have generated grammars");
    }

    #[test]
    fn test_tool_grammar_builder_build_xml_structure_validates() {
        // Full end-to-end: verify XML grammar produces valid llguidance TopLevelGrammar structure
        let tools = vec![crate::tools::ToolBuilder::new("formatter".to_string(), "Formatter".to_string())
            .param("text", "string", "Text to format", true)
            .build()];

        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("<tool>")
            .end_tag("</tool>")
            .start_is_special(false)
            .end_is_special(false)
            .build_xml();

        // Grammar should have at least one sub-grammar
        assert!(grammar.grammars.len() > 0, "Should have generated grammars");
    }

    // === Comprehensive ToolGrammarBuilder Tests ===

    #[test]
    fn test_tool_grammar_builder_build_xml_complex_full_schema() {
        // Test ToolGrammarBuilder.build_xml() with complex nested schema
        // and model-specific envelope tags with token IDs
        let tools = vec![
            crate::tools::ToolBuilder::new("edit_file".to_string(), "Edit a file".to_string())
                .param("file_path", "string", "Path to the file", true)
                .param("old_string", "string", "String to replace", true)
                .param("new_string", "string", "Replacement string", true)
                .param("max_replacements", "integer", "Maximum replacements", false)
                .param("context", "object", "Context object", false)
                .param("tags", "array", "Optional tags array", false)
                .build(),
        ];

        // Build XML grammar with token IDs for envelope tags
        let mut start_ids = HashSet::new();
        start_ids.insert(151657);
        let mut end_ids = HashSet::new();
        end_ids.insert(151658);

        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("")
            .end_tag("")
            .start_is_special(false)
            .end_is_special(false)
            .start_token_ids(Some(start_ids))
            .end_token_ids(Some(end_ids))
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);
        // println!("{}", &lark_str);

        // Validate envelope layer (token IDs)
        assert!(lark_str.contains("<[151657]>"), "Should have start token ID envelope");
        assert!(lark_str.contains("<[151658]>"), "Should have end token ID envelope");

        // Validate tool_call structure
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("tool_call:"), "Should have tool_call rule");

        // Validate tool_content alternation
        assert!(lark_str.contains("tool_content: tool_0"), "Should have tool_content with tool_0");

        // Validate function tag layer
        assert!(lark_str.contains("function=\\\"edit_file\\\""), "Should have function tag");
        assert!(lark_str.contains("function="), "Should have function tag pattern");

        // Validate parameter tags and rules
        assert!(lark_str.contains("parameter=\\\"file_path\\\""), "Should have file_path parameter tag");
        assert!(lark_str.contains("parameter=\\\"old_string\\\""), "Should have old_string parameter tag");
        assert!(lark_str.contains("parameter=\\\"new_string\\\""), "Should have new_string parameter tag");
        assert!(lark_str.contains("parameter=\\\"max_replacements\\\""), "Should have max_replacements parameter tag");
        assert!(lark_str.contains("parameter=\\\"context\\\""), "Should have context parameter tag");
        assert!(lark_str.contains("parameter=\\\"tags\\\""), "Should have tags parameter tag");

        // Validate param rules with correct types
        assert!(lark_str.contains("param_0_0:"), "Should have param_0_0 rule (file_path - required)");
        assert!(lark_str.contains("param_0_1:"), "Should have param_0_1 rule (old_string - required)");
        assert!(lark_str.contains("param_0_2:"), "Should have param_0_2 rule (new_string - required)");
        assert!(lark_str.contains("param_0_3:"), "Should have param_0_3 rule (max_replacements - optional)");
        assert!(lark_str.contains("param_0_4:"), "Should have param_0_4 rule (context - optional)");
        assert!(lark_str.contains("param_0_5:"), "Should have param_0_5 rule (tags - optional)");

        // Validate value rules with regex patterns for each type
        assert!(lark_str.contains("value_0_0:"), "Should have value_0_0 rule for file_path");
        assert!(lark_str.contains("value_0_1:"), "Should have value_0_1 rule for old_string");
        assert!(lark_str.contains("value_0_2:"), "Should have value_0_2 rule for new_string");
        assert!(lark_str.contains("value_0_3:"), "Should have value_0_3 rule for max_replacements");
        assert!(lark_str.contains("value_0_4:"), "Should have value_0_4 rule for context");
        assert!(lark_str.contains("value_0_5:"), "Should have value_0_5 rule for tags");

        // Validate required params are bare (no ? wrapper)
        assert!(lark_str.contains("param_0_0 "), "file_path should be bare (required)");
        assert!(lark_str.contains("param_0_1 "), "old_string should be bare (required)");
        assert!(lark_str.contains("param_0_2 "), "new_string should be bare (required)");

        // Validate optional params have ? wrapper
        assert!(lark_str.contains("(param_0_3)?"), "max_replacements should be optional");
        assert!(lark_str.contains("(param_0_4)?"), "context should be optional");
        assert!(lark_str.contains("(param_0_5)?"), "tags should be optional");

        // Validate tool rule structure
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
        assert!(lark_str.contains("tool_0: \"<function=\\\"edit_file\\\">\""), "Should have tool_0 with function tags");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_complex_full_schema() {
        // Test ToolGrammarBuilder.build_json() with complex nested schema
        // and model-specific envelope tags with token IDs
        let tools = vec![
            crate::tools::ToolBuilder::new("edit_file".to_string(), "Edit a file".to_string())
                .param("file_path", "string", "Path to the file", true)
                .param("old_string", "string", "String to replace", true)
                .param("new_string", "string", "Replacement string", true)
                .param("max_replacements", "integer", "Maximum replacements", false)
                .param("context", "object", "Context object", false)
                .param("tags", "array", "Optional tags array", false)
                .build(),
        ];

        // Build JSON grammar with token IDs for envelope tags
        let mut start_ids = HashSet::new();
        start_ids.insert(151657);
        let mut end_ids = HashSet::new();
        end_ids.insert(151658);

        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("")
            .end_tag("")
            .start_is_special(false)
            .end_is_special(false)
            .start_token_ids(Some(start_ids))
            .end_token_ids(Some(end_ids))
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);

        // Validate envelope layer (token IDs)
        assert!(lark_str.contains("<[151657]>"), "Should have start token ID envelope");
        assert!(lark_str.contains("<[151658]>"), "Should have end token ID envelope");

        // Validate tool_call structure
        assert!(lark_str.contains("start: tool_call"), "Should have start: tool_call");
        assert!(lark_str.contains("tool_call:"), "Should have tool_call rule");

        // Validate tool_obj structure with name and arguments
        assert!(lark_str.contains("tool_obj:"), "Should have tool_obj rule");
        assert!(lark_str.contains("\"name\""), "Should have name in tool_obj");
        assert!(lark_str.contains("\"arguments\""), "Should have arguments in tool_obj");

        // Validate obj rule references the tool
        assert!(lark_str.contains("obj_edit_file:"), "Should have obj_edit_file rule");
        assert!(lark_str.contains("obj: obj_edit_file"), "Should have obj alternation");

        // Validate JSON schema contains all parameters
        assert!(lark_str.contains("file_path"), "Should contain file_path in schema");
        assert!(lark_str.contains("old_string"), "Should contain old_string in schema");
        assert!(lark_str.contains("new_string"), "Should contain new_string in schema");
        assert!(lark_str.contains("max_replacements"), "Should contain max_replacements in schema");
        assert!(lark_str.contains("context"), "Should contain context in schema");
        assert!(lark_str.contains("tags"), "Should contain tags in schema");

        // Validate required parameters in JSON schema
        assert!(lark_str.contains("\"required\""), "Should have required array");
        assert!(lark_str.contains("file_path"), "Should have file_path in required");
        assert!(lark_str.contains("old_string"), "Should have old_string in required");
        assert!(lark_str.contains("new_string"), "Should have new_string in required");
    }

    #[test]
    fn test_tool_grammar_builder_build_xml_multiple_tools_full_validation() {
        // Test ToolGrammarBuilder.build_xml() with multiple tools and full validation
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .param("max_results", "integer", "Max results", false)
                .build(),
            crate::tools::ToolBuilder::new("weather".to_string(), "Get weather".to_string())
                .param("city", "string", "City name", true)
                .param("units", "string", "Units", false)
                .build(),
        ];

        // Build XML grammar with token IDs for envelope tags
        let mut start_ids = HashSet::new();
        start_ids.insert(151657);
        let mut end_ids = HashSet::new();
        end_ids.insert(151658);

        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("")
            .end_tag("")
            .start_is_special(false)
            .end_is_special(false)
            .start_token_ids(Some(start_ids))
            .end_token_ids(Some(end_ids))
            .build_xml();

        let lark_str = get_lark_from_top_level_grammar(&grammar);

        // Validate envelope layer
        assert!(lark_str.contains("<[151657]>"), "Should have start token ID envelope");
        assert!(lark_str.contains("<[151658]>"), "Should have end token ID envelope");

        // Validate tool_content alternation with both tools
        assert!(lark_str.contains("tool_content: tool_0 | tool_1"), "Should have tool alternation");

        // Validate tool_0 (search) structure
        assert!(lark_str.contains("tool_0:"), "Should have tool_0 rule");
        assert!(lark_str.contains("function=search"), "Should have search function tag");
        assert!(lark_str.contains("parameter=query"), "Should have query parameter");
        assert!(lark_str.contains("parameter=max_results"), "Should have max_results parameter");
        assert!(lark_str.contains("param_0_0:"), "Should have param_0_0 (query - required)");
        assert!(lark_str.contains("param_0_1:"), "Should have param_0_1 (max_results - optional)");

        // Validate tool_1 (weather) structure
        assert!(lark_str.contains("tool_1:"), "Should have tool_1 rule");
        assert!(lark_str.contains("function=weather"), "Should have weather function tag");
        assert!(lark_str.contains("parameter=city"), "Should have city parameter");
        assert!(lark_str.contains("parameter=units"), "Should have units parameter");
        assert!(lark_str.contains("param_1_0:"), "Should have param_1_0 (city - required)");
        assert!(lark_str.contains("param_1_1:"), "Should have param_1_1 (units - optional)");
    }

    #[test]
    fn test_tool_grammar_builder_build_json_multiple_tools_full_validation() {
        // Test ToolGrammarBuilder.build_json() with multiple tools and full validation
        let tools = vec![
            crate::tools::ToolBuilder::new("search".to_string(), "Search the web".to_string())
                .param("query", "string", "Search query", true)
                .param("max_results", "integer", "Max results", false)
                .build(),
            crate::tools::ToolBuilder::new("weather".to_string(), "Get weather".to_string())
                .param("city", "string", "City name", true)
                .param("units", "string", "Units", false)
                .build(),
        ];

        // Build JSON grammar with token IDs for envelope tags
        let mut start_ids = HashSet::new();
        start_ids.insert(151657);
        let mut end_ids = HashSet::new();
        end_ids.insert(151658);

        let grammar = ToolGrammarBuilder::new()
            .tools(&tools)
            .start_tag("")
            .end_tag("")
            .start_is_special(false)
            .end_is_special(false)
            .start_token_ids(Some(start_ids))
            .end_token_ids(Some(end_ids))
            .build_json();

        let lark_str = get_lark_from_top_level_grammar(&grammar);

        // Validate envelope layer
        assert!(lark_str.contains("<[151657]>"), "Should have start token ID envelope");
        assert!(lark_str.contains("<[151658]>"), "Should have end token ID envelope");

        // Validate obj alternation with both tools
        assert!(lark_str.contains("obj: obj_search | obj_weather"), "Should have obj alternation");

        // Validate obj_search structure
        assert!(lark_str.contains("obj_search:"), "Should have obj_search rule");
        assert!(lark_str.contains("query"), "Should have query in obj_search");
        assert!(lark_str.contains("max_results"), "Should have max_results in obj_search");

        // Validate obj_weather structure
        assert!(lark_str.contains("obj_weather:"), "Should have obj_weather rule");
        assert!(lark_str.contains("city"), "Should have city in obj_weather");
        assert!(lark_str.contains("units"), "Should have units in obj_weather");

        // Validate required parameters in both schemas
        assert!(lark_str.contains("\"required\":[\"query\"]"), "Should have query in required for search");
        assert!(lark_str.contains("\"required\":[\"city\"]"), "Should have city in required for weather");
    }
}
