// src/tools/mod.rs
//! Tool calling support for vLLM.rs
//!
//! This module provides OpenAI-compatible tool calling functionality,
//! allowing LLMs to invoke external functions and tools.

pub mod helpers;
pub mod parser;
pub mod schema;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub use openai_protocol::common::{Function, FunctionCallResponse as FunctionCall, Tool, ToolCall};

/// Builder for creating Tool definitions
pub struct ToolBuilder {
    name: String,
    description: String,
    parameters: Value,
    strict: Option<bool>,
}

impl ToolBuilder {
    fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            strict: None,
        }
    }

    /// Add a parameter to the function
    pub fn param(
        mut self,
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        if let Some(props) = self.parameters.get_mut("properties") {
            props[&name] = serde_json::json!({
                "type": param_type.into(),
                "description": description.into()
            });
        }
        if required {
            if let Some(req) = self.parameters.get_mut("required") {
                if let Some(arr) = req.as_array_mut() {
                    arr.push(Value::String(name));
                }
            }
        }
        self
    }

    /// Set custom parameters schema
    pub fn parameters_schema(mut self, schema: Value) -> Self {
        self.parameters = schema;
        self
    }

    /// Enable strict mode
    pub fn strict(mut self, value: bool) -> Self {
        self.strict = Some(value);
        self
    }

    /// Build the final Tool
    pub fn build(self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: self.name,
                description: Some(self.description),
                parameters: self.parameters,
                strict: self.strict,
            },
        }
    }
}

/// Create a new function tool builder (replacement for Tool::function).
pub fn function_tool(name: impl Into<String>, description: impl Into<String>) -> ToolBuilder {
    ToolBuilder::new(name.into(), description.into())
}

/// Tool choice configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String modes: "auto" | "none" | "required"
    Mode(ToolChoiceMode),
    /// Force a specific tool
    Function {
        #[serde(rename = "type")]
        choice_type: ToolChoiceType,
        function: ToolChoiceFunction,
    },
}

impl ToolChoice {
    pub fn auto() -> Self {
        ToolChoice::Mode(ToolChoiceMode::Auto)
    }

    pub fn none() -> Self {
        ToolChoice::Mode(ToolChoiceMode::None)
    }

    pub fn required() -> Self {
        ToolChoice::Mode(ToolChoiceMode::Required)
    }

    pub fn function(name: impl Into<String>) -> Self {
        ToolChoice::Function {
            choice_type: ToolChoiceType::Function,
            function: ToolChoiceFunction { name: name.into() },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceType {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// Build a ToolCall from name/arguments with a provided ID.
pub fn new_tool_call(
    id: impl Into<String>,
    name: impl Into<String>,
    arguments: impl Into<String>,
) -> ToolCall {
    ToolCall {
        id: id.into(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.into(),
            arguments: Some(arguments.into()),
        },
    }
}

/// Generate a compact tool call ID with required `call_` prefix.
/// Uses 16 hex chars (64 bits) from UUIDv4 for low collision risk and shorter payloads.
pub fn generate_tool_call_id() -> String {
    let raw = Uuid::new_v4().simple().to_string();
    format!("call_{}", &raw[..16])
}

/// Convert a parsed tool call into an OpenAI-compatible ToolCall.
pub fn tool_call_from_parser(parsed: tool_parser::ToolCall) -> ToolCall {
    ToolCall {
        id: generate_tool_call_id(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: parsed.function.name,
            arguments: Some(parsed.function.arguments),
        },
    }
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The tool call ID this result corresponds to
    pub tool_call_id: String,
    /// The result content (typically JSON or text)
    pub content: String,
    /// Whether the tool execution was successful
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            is_error: None,
        }
    }

    /// Create an error tool result
    pub fn error(tool_call_id: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: error_message.into(),
            is_error: Some(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_choice_deserializes_string_modes() {
        let auto: ToolChoice = serde_json::from_str(r#""auto""#).unwrap();
        let none: ToolChoice = serde_json::from_str(r#""none""#).unwrap();
        let required: ToolChoice = serde_json::from_str(r#""required""#).unwrap();

        assert!(matches!(auto, ToolChoice::Mode(ToolChoiceMode::Auto)));
        assert!(matches!(none, ToolChoice::Mode(ToolChoiceMode::None)));
        assert!(matches!(
            required,
            ToolChoice::Mode(ToolChoiceMode::Required)
        ));
    }

    #[test]
    fn tool_choice_deserializes_function_mode() {
        let choice: ToolChoice =
            serde_json::from_str(r#"{"type":"function","function":{"name":"read_file"}}"#).unwrap();
        match choice {
            ToolChoice::Function {
                choice_type,
                function,
            } => {
                assert_eq!(choice_type, ToolChoiceType::Function);
                assert_eq!(function.name, "read_file");
            }
            _ => panic!("expected function tool choice"),
        }
    }
}

/// Format tool definitions for injection into the prompt
#[derive(Debug, Clone)]
pub struct ToolFormat {}

impl ToolFormat {
    /// Get tool prompt for a specific model type
    pub fn get_tool_prompt(model_type: &crate::utils::config::ModelType) -> String {
        use crate::server::parser::ToolConfig;
        let config = ToolConfig::for_model_type(model_type);
        let start_tag = &config.start_token_str;
        let end_tag = &config.end_token_str;
        let rule = format!(
            "MOST IMPORTANT INSTRUCTION, **MUST** FOLLOW: For each function call, you MUST wrap function name and arguments in {start_tag}{end_tag} tags.\n\n\
            Do NOT USE ANY code blocks. Required format:\n\
            {start_tag}\n\
            {{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}}\n\
            {end_tag}\n\n\
            Rules:\n\
            - Wrap function name and arguments with {start_tag} and {end_tag} tags\n\
            - Always use the exact {start_tag}{end_tag} format shown above\n\
            - Do NOT USE ANY code blocks\n\
            - Tool-use must be placed **at the end** of your response (**AFTER REASONING**), **top-level**, and not nested within other tags.\n\
            - Always adhere to this format for the tool use to ensure proper parsing and execution.\n\
            - The \"name\" and \"arguments\" are necessary fields\n\
            - DO NOT call ANY functions that DOES NOT defined between <tool> and </tool>\n\
            - MUST FOLLOW the above instruction when using tool call!",
        );
        rule
    }
}
