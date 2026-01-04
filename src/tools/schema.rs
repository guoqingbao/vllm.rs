// src/tools/schema.rs
//! JSON Schema utilities for tool parameters
//!
//! Provides helpers for working with JSON Schema in tool definitions.

use serde_json::{json, Value};
use std::collections::HashMap;

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

/// Validate arguments against a JSON Schema
pub fn validate_arguments(schema: &Value, arguments: &Value) -> Result<(), String> {
    // Basic validation - check required fields and types
    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for req in required {
            if let Some(field_name) = req.as_str() {
                if !arguments.get(field_name).map_or(false, |v| !v.is_null()) {
                    return Err(format!("Missing required field: {}", field_name));
                }
            }
        }
    }

    if let Some(additional) = schema.get("additionalProperties") {
        if additional == &Value::Bool(false) {
            if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
                if let Some(args_obj) = arguments.as_object() {
                    for key in args_obj.keys() {
                        if !properties.contains_key(key) {
                            return Err(format!("Unexpected field: {}", key));
                        }
                    }
                }
            }
        }
    }

    if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
        if let Some(args_obj) = arguments.as_object() {
            for (key, value) in args_obj {
                if let Some(prop_schema) = properties.get(key) {
                    validate_type(prop_schema, value, key)?;
                }
            }
        }
    }

    Ok(())
}

fn validate_type(schema: &Value, value: &Value, field_name: &str) -> Result<(), String> {
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        if !enum_values.iter().any(|v| v == value) {
            return Err(format!("Field '{}' must be one of enum values", field_name));
        }
    }

    let expected_type = schema.get("type");
    if let Some(type_list) = expected_type.and_then(|t| t.as_array()) {
        let mut matches = false;
        for entry in type_list {
            if let Some(kind) = entry.as_str() {
                if type_matches(kind, value) {
                    matches = true;
                    break;
                }
            }
        }
        if !matches {
            return Err(format!("Field '{}' has invalid type", field_name));
        }
    } else if let Some(kind) = expected_type.and_then(|t| t.as_str()) {
        if !type_matches(kind, value) {
            return Err(format!("Field '{}' has invalid type", field_name));
        }
    }

    match expected_type.and_then(|t| t.as_str()) {
        Some("string") => {
            if let Some(min_len) = schema.get("minLength").and_then(|v| v.as_u64()) {
                if value.as_str().map_or(0, |s| s.len() as u64) < min_len {
                    return Err(format!("Field '{}' is too short", field_name));
                }
            }
            if let Some(max_len) = schema.get("maxLength").and_then(|v| v.as_u64()) {
                if value.as_str().map_or(0, |s| s.len() as u64) > max_len {
                    return Err(format!("Field '{}' is too long", field_name));
                }
            }
        }
        Some("number") | Some("integer") => {
            if let Some(min) = schema.get("minimum").and_then(|v| v.as_f64()) {
                if value.as_f64().map_or(true, |n| n < min) {
                    return Err(format!("Field '{}' is below minimum", field_name));
                }
            }
            if let Some(max) = schema.get("maximum").and_then(|v| v.as_f64()) {
                if value.as_f64().map_or(true, |n| n > max) {
                    return Err(format!("Field '{}' is above maximum", field_name));
                }
            }
        }
        Some("array") => {
            if let Some(min) = schema.get("minItems").and_then(|v| v.as_u64()) {
                if value.as_array().map_or(0, |a| a.len() as u64) < min {
                    return Err(format!("Field '{}' has too few items", field_name));
                }
            }
            if let Some(max) = schema.get("maxItems").and_then(|v| v.as_u64()) {
                if value.as_array().map_or(0, |a| a.len() as u64) > max {
                    return Err(format!("Field '{}' has too many items", field_name));
                }
            }
            if let Some(items) = schema.get("items") {
                if let Some(array) = value.as_array() {
                    for item in array {
                        validate_type(items, item, field_name)?;
                    }
                }
            }
        }
        Some("object") => {
            if let Some(min) = schema.get("minProperties").and_then(|v| v.as_u64()) {
                if value.as_object().map_or(0, |o| o.len() as u64) < min {
                    return Err(format!("Field '{}' has too few properties", field_name));
                }
            }
            if let Some(max) = schema.get("maxProperties").and_then(|v| v.as_u64()) {
                if value.as_object().map_or(0, |o| o.len() as u64) > max {
                    return Err(format!("Field '{}' has too many properties", field_name));
                }
            }
            if let Some(schema_obj) = schema.as_object() {
                if schema_obj.contains_key("properties") || schema_obj.contains_key("required") {
                    return validate_arguments(schema, value);
                }
            }
        }
        _ => {}
    }

    Ok(())
}

fn type_matches(expected: &str, value: &Value) -> bool {
    match expected {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builder() {
        let schema = SchemaBuilder::object()
            .description("Get weather information")
            .string_prop("location", "City name", true)
            .enum_prop(
                "unit",
                vec!["celsius", "fahrenheit"],
                "Temperature unit",
                false,
            )
            .build();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["location"].is_object());
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("location")));
    }

    #[test]
    fn test_validate_required() {
        let schema = SchemaBuilder::object()
            .string_prop("name", "Name", true)
            .build();

        let valid = json!({"name": "test"});
        let invalid = json!({});

        assert!(validate_arguments(&schema, &valid).is_ok());
        assert!(validate_arguments(&schema, &invalid).is_err());
    }

    #[test]
    fn test_validate_types() {
        let schema = SchemaBuilder::object()
            .string_prop("name", "Name", true)
            .integer_prop("age", "Age", false)
            .build();

        let valid = json!({"name": "test", "age": 25});
        let invalid = json!({"name": "test", "age": "twenty-five"});

        assert!(validate_arguments(&schema, &valid).is_ok());
        assert!(validate_arguments(&schema, &invalid).is_err());
    }

    #[test]
    fn test_common_schemas() {
        let calc = common::calculator_schema();
        assert!(calc["properties"]["expression"].is_object());

        let search = common::web_search_schema();
        assert!(search["properties"]["query"].is_object());
    }

    #[test]
    fn test_additional_properties() {
        let schema = SchemaBuilder::object()
            .string_prop("path", "Path", true)
            .no_additional_properties()
            .build();

        let valid = json!({"path": "."});
        let invalid = json!({"path": ".", "extra": "nope"});

        assert!(validate_arguments(&schema, &valid).is_ok());
        assert!(validate_arguments(&schema, &invalid).is_err());
    }
}
