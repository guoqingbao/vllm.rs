// src/tools/parser.rs
//! Tool call parsing from model output
//!
//! Supports multiple formats used by different models.

use super::ToolCall;
use regex::Regex;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::{Map, Value};
use std::fmt;
use std::sync::OnceLock;

/// Parser for extracting tool calls from model output text
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ToolParser {
    /// Regex patterns for different formats
    #[allow(dead_code)]
    patterns: Vec<(String, Regex)>,
}

impl Default for ToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolParser {
    /// Create a new parser with default patterns
    pub fn new() -> Self {
        let patterns = vec![
            // Qwen format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
            (
                "qwen".to_string(),
                Regex::new(r#"<tool_call>\s*(\{[^}]+\})\s*</tool_call>"#).unwrap()
            ),
            // Generic JSON object with name and arguments
            (
                "json".to_string(),
                Regex::new(r#"\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\}|\[[^\]]*\]|"[^"]*"|\d+|true|false|null)\s*\}"#).unwrap()
            ),
            // Function call format in code blocks
            (
                "func".to_string(),
                Regex::new(r#"```(?:json)?\s*\{[^}]*"name"[^}]*\}\s*```"#).unwrap()
            ),
        ];
        Self { patterns }
    }

    /// Parse tool calls from model output
    /// Only parses tool calls from the final answer (after reasoning end markers)
    pub fn parse(&self, text: &str) -> Vec<ToolCall> {
        let mut call_id = 0;

        // Extract only the final answer portion (after reasoning ends)
        let final_answer = Self::extract_final_answer(text);

        // Mistral-style parsing: strip wrappers and parse JSON or JSON array.
        let mut calls = parse_tool_calls_from_text(&final_answer, &mut call_id);

        if !calls.is_empty() {
            return calls;
        }

        // Try Qwen format first
        if let Some(qwen_calls) = self.parse_qwen_format(&final_answer, &mut call_id) {
            calls.extend(qwen_calls);
        }

        // Try generic JSON format
        if calls.is_empty() {
            if let Some(json_calls) = self.parse_json_format(&final_answer, &mut call_id) {
                calls.extend(json_calls);
            }
        }

        // Try code block format
        if calls.is_empty() {
            if let Some(block_calls) = self.parse_code_block_format(&final_answer, &mut call_id) {
                calls.extend(block_calls);
            }
        }

        calls
    }

    /// Extract the final answer portion from model output, skipping reasoning blocks.
    /// Returns the text after reasoning end markers, or the full text if no reasoning found.
    pub fn extract_final_answer(text: &str) -> String {
        // Reasoning end markers used by different models
        let reasoning_end_markers = [
            "</think>",     // Common thinking format
            "</thought>",   // Alternative thinking format
            "<|/think|>",   // Qwen-style special tokens
            "[/THINK]",     // Bracket format
            "</reasoning>", // Reasoning tag
        ];

        // Find the last occurrence of any reasoning end marker
        let mut last_end_pos = None;
        for marker in &reasoning_end_markers {
            if let Some(pos) = text.rfind(marker) {
                let end_pos = pos + marker.len();
                if last_end_pos.is_none() || end_pos > last_end_pos.unwrap() {
                    last_end_pos = Some(end_pos);
                }
            }
        }

        // Return content after the last reasoning end marker, or full text if none found
        if let Some(pos) = last_end_pos {
            text[pos..].to_string()
        } else {
            text.to_string()
        }
    }

    /// Parse XML-wrapped tool call formats (<tool_call>)
    fn parse_qwen_format(&self, text: &str, call_id: &mut usize) -> Option<Vec<ToolCall>> {
        let mut calls = Vec::new();

        // Try both <tool_call> formats
        // Use a more flexible regex that allows for missing closing > if at end of string
        // This handles cases where generation stops exactly on </tool_call
        // Pattern: <tool_call> ... </tool_call>?
        // Note: We use a single regex with optional > to avoid duplicate matches
        let pattern = r"(?s)<tool_call>\s*(.*?)\s*</tool_call>?";

        if let Ok(re) = Regex::new(pattern) {
            for cap in re.captures_iter(text) {
                if let Some(json_str) = cap.get(1) {
                    // Validate that whatever we captured looks like JSON before parsing
                    // This prevents matching random text if </tool_call> is missing entirely and we match to end of string
                    let trimmed = json_str.as_str().trim();
                    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
                        continue;
                    }

                    if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                        calls.extend(self.value_to_tool_calls(&parsed, call_id));
                    }
                }
            }
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    /// Parse generic JSON format with name and arguments
    fn parse_json_format(&self, text: &str, call_id: &mut usize) -> Option<Vec<ToolCall>> {
        // Try to find JSON objects that look like tool calls
        let mut calls = Vec::new();

        // Simple approach: try to parse the entire text as JSON first
        if let Ok(parsed) = serde_json::from_str::<Value>(text.trim()) {
            let parsed_calls = self.value_to_tool_calls(&parsed, call_id);
            if !parsed_calls.is_empty() {
                return Some(parsed_calls);
            }
        }

        // Look for JSON blocks in the text
        let mut depth = 0;
        let mut start = None;

        for (i, c) in text.char_indices() {
            match c {
                '{' => {
                    if depth == 0 {
                        start = Some(i);
                    }
                    depth += 1;
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s) = start {
                            let json_str = &text[s..=i];
                            if let Ok(parsed) = serde_json::from_str::<Value>(json_str) {
                                calls.extend(self.value_to_tool_calls(&parsed, call_id));
                            }
                        }
                        start = None;
                    }
                }
                _ => {}
            }
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    /// Parse tool calls from markdown code blocks
    fn parse_code_block_format(&self, text: &str, call_id: &mut usize) -> Option<Vec<ToolCall>> {
        let re = Regex::new(r"```(?:json)?\s*([\s\S]*?)\s*```").ok()?;
        let mut calls = Vec::new();

        for cap in re.captures_iter(text) {
            if let Some(content) = cap.get(1) {
                if let Ok(parsed) = serde_json::from_str::<Value>(content.as_str().trim()) {
                    calls.extend(self.value_to_tool_calls(&parsed, call_id));
                }
            }
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    /// Convert a JSON Value to ToolCall(s) if it has the right structure
    fn value_to_tool_calls(&self, value: &Value, call_id: &mut usize) -> Vec<ToolCall> {
        match value {
            Value::Array(items) => items
                .iter()
                .flat_map(|item| self.value_to_tool_calls(item, call_id))
                .collect(),
            Value::Object(_) => {
                if let Some(call) = json_value_to_tool_call(value, call_id) {
                    vec![call]
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }

    /// Check if text contains any tool calls (JSON or model-specific wrappers)
    pub fn has_tool_calls(&self, text: &str) -> bool {
        let final_answer = Self::extract_final_answer(text);
        let mut call_id = 0;
        if !parse_tool_calls_from_text(&final_answer, &mut call_id).is_empty() {
            return true;
        }
        contains_tool_call_prefix(&final_answer)
    }

    /// Check if text contains a complete, parseable tool call
    /// Returns true only if the tool call has valid structure with both tags and valid JSON
    pub fn has_complete_tool_call(&self, text: &str) -> bool {
        let final_answer = Self::extract_final_answer(text);

        // Must have both opening and closing tags
        if !final_answer.contains("<tool_call>") || !final_answer.contains("</tool_call>") {
            return false;
        }

        // Try to parse - if successful, it's complete
        !self.parse(&final_answer).is_empty()
    }

    /// Check if text could be a partial tool call tag (for lookback detection)
    /// Used to detect when we might be in the middle of receiving "<tool_call>"
    pub fn could_be_partial_tag(text: &str) -> bool {
        const TAG: &str = "<tool_call>";
        // Check if end of text matches any prefix of tag (length 1 to len-1)
        for i in 1..TAG.len() {
            if text.ends_with(&TAG[..i]) {
                return true;
            }
        }
        false
    }
}

// --- Mistral-style tool parsing helpers ---

// Accept either `{...}` **or** a `"stringified { ... }"`
fn flexible_args<'de, D>(d: D) -> std::result::Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    struct ArgVisitor;

    impl<'de> Visitor<'de> for ArgVisitor {
        type Value = Value;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("an object or a JSON-encoded string containing an object")
        }

        fn visit_map<M>(self, mut m: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut map = Map::new();
            while let Some((k, v)) = m.next_entry()? {
                map.insert(k, v);
            }
            Ok(Value::Object(map))
        }

        fn visit_str<E>(self, s: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            serde_json::from_str(s).map_err(|e| E::custom(format!("inner JSON error: {e}")))
        }
    }

    d.deserialize_any(ArgVisitor)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CalledFunctionParameters {
    #[serde(alias = "function")]
    name: String,
    #[serde(alias = "arguments", deserialize_with = "flexible_args")]
    parameters: Value,
}

fn contains_tool_call_prefix(prefix: &str) -> bool {
    prefix.contains("<tool_call>")
        || prefix.contains("<｜tool▁call▁begin｜>")
        || prefix.contains("<|python_tag|>")
        || prefix.contains("[TOOL_CALLS]")
}

fn process_model_specific_message(message: &str) -> String {
    static DEEPSEEK_REGEX: OnceLock<Regex> = OnceLock::new();
    static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();

    let deepseek_regex = DEEPSEEK_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<name>[^\n]+)\n```json\n(?P<json>.+?)\n```<｜tool▁call▁end｜>",
        )
        .unwrap()
    });
    let qwen_regex = QWEN_REGEX
        .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

    if let Some(message) = message.strip_prefix("<|python_tag|>") {
        message
            .strip_suffix("<|eom_id|>")
            .unwrap_or(message)
            .to_string()
    } else if qwen_regex.is_match(message) {
        if let Some(caps) = qwen_regex.captures(message) {
            let inner = caps.name("inner").unwrap().as_str();
            return inner.trim().to_string();
        }
        message.to_string()
    } else if let Some(message) = message
        .strip_prefix("[TOOL_CALLS][")
        .and_then(|s| s.strip_suffix("]"))
    {
        message.to_string()
    } else if deepseek_regex.find(message).is_some() {
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct ToolCall {
            name: String,
            arguments: Value,
        }
        let mut calls = Vec::new();
        for caps in deepseek_regex.captures_iter(message) {
            let name = caps
                .name("name")
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            let json_str = caps.name("json").map(|m| m.as_str().trim()).unwrap_or("{}");
            let arguments: Value =
                serde_json::from_str(json_str).unwrap_or_else(|_| Value::Object(Map::new()));
            calls.push(ToolCall { name, arguments });
        }
        serde_json::to_string(&calls).unwrap_or_else(|_| message.to_string())
    } else {
        message.to_string()
    }
}

fn fix_broken_json(raw: &str) -> String {
    if raw.contains(r#""arguments":"{"#) {
        let tmp = raw.replacen(r#""arguments":"{"#, r#""arguments":{"#, 1);
        tmp.replacen(r#"}"}"#, r#"}}"#, 1)
    } else {
        raw.to_string()
    }
}

fn json_value_to_tool_call(value: &Value, call_id: &mut usize) -> Option<ToolCall> {
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value.get("arguments")?;
    let args_str = if arguments.is_string() {
        arguments.as_str().unwrap_or("{}").to_string()
    } else {
        serde_json::to_string(arguments).ok()?
    };

    let call = ToolCall {
        index: Some(*call_id),
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: super::FunctionCall {
            name,
            arguments: args_str,
        },
    };
    *call_id += 1;
    Some(call)
}

/// Parse tool calls from a raw message string (handles model-specific wrappers).
pub fn parse_tool_calls_from_text(text: &str, call_id: &mut usize) -> Vec<ToolCall> {
    // First, handle explicit <tool_call> wrappers (may appear multiple times)
    if text.contains("<tool_call>") {
        let mut calls = Vec::new();
        if let Ok(re) = Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>") {
            for cap in re.captures_iter(text) {
                if let Some(inner) = cap.get(1) {
                    let inner = inner.as_str().trim();
                    if let Ok(parsed) = serde_json::from_str::<Value>(inner) {
                        if let Some(call) = json_value_to_tool_call(&parsed, call_id) {
                            calls.push(call);
                        }
                        continue;
                    }

                    if let Some(call) = parse_function_tag_tool_call(inner, call_id) {
                        calls.push(call);
                    }
                }
            }
        }
        if !calls.is_empty() {
            return calls;
        }
    }

    let processed = process_model_specific_message(text);
    let processed = fix_broken_json(&processed);

    if let Ok(deser) = serde_json::from_str::<CalledFunctionParameters>(&processed) {
        let args = serde_json::to_string(&deser.parameters).unwrap_or_else(|_| "{}".to_string());
        let call = ToolCall {
            index: Some(*call_id),
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            call_type: "function".to_string(),
            function: super::FunctionCall {
                name: deser.name,
                arguments: args,
            },
        };
        *call_id += 1;
        return vec![call];
    }

    if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(&processed) {
        let mut out = Vec::new();
        for item in deser {
            let args = serde_json::to_string(&item.parameters).unwrap_or_else(|_| "{}".to_string());
            out.push(ToolCall {
                index: Some(*call_id),
                id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                call_type: "function".to_string(),
                function: super::FunctionCall {
                    name: item.name,
                    arguments: args,
                },
            });
            *call_id += 1;
        }
        return out;
    }

    Vec::new()
}

/// Checks if the given prefix could be the start of, or the entire JSON serialization of a tool call.
/// Returns (could_be_tool, is_complete_tool).
pub fn prefix_could_be_tool(prefix: &str) -> (bool, bool) {
    if prefix.trim().is_empty() {
        return (false, false);
    }

    // If we already have a full <tool_call>...</tool_call>, attempt to parse directly.
    if prefix.contains("</tool_call>") {
        let mut call_id = 0;
        if !parse_tool_calls_from_text(prefix, &mut call_id).is_empty() {
            return (false, true);
        }
    }

    // If we see a start tag, it's at least a potential tool call.
    if prefix.contains("<tool_call>") {
        return (true, false);
    }

    let processed = process_model_specific_message(prefix);
    let processed = fix_broken_json(&processed);

    let checks = [
        could_be_json::<CalledFunctionParameters>,
        could_be_json::<Vec<CalledFunctionParameters>>,
    ];

    for check in checks {
        let (could_be, complete) = check(&processed);
        if could_be || complete {
            return (could_be, complete);
        }
    }

    (
        contains_tool_call_prefix(prefix) || contains_tool_call_prefix(&processed),
        false,
    )
}

fn could_be_json<T>(text_prefix: &str) -> (bool, bool)
where
    T: serde::de::DeserializeOwned,
{
    if text_prefix.trim().is_empty() {
        return (false, false);
    }
    match serde_json::from_str::<T>(text_prefix) {
        Ok(_) => (false, true),
        Err(e) if e.is_eof() => (true, false),
        _ => (false, false),
    }
}

fn parse_function_tag_tool_call(inner: &str, call_id: &mut usize) -> Option<ToolCall> {
    let func_tag = "<function=";
    let func_start = inner.find(func_tag)?;
    let name_start = func_start + func_tag.len();
    let name_end = inner[name_start..].find('>')? + name_start;
    if name_end <= name_start {
        return None;
    }
    let func_name = inner[name_start..name_end].trim();
    if func_name.is_empty() {
        return None;
    }

    let mut params = Map::new();
    let mut pos = name_end + 1;
    while let Some(param_tag_pos) = inner[pos..].find("<parameter=") {
        let param_tag_pos = pos + param_tag_pos;
        let key_start = param_tag_pos + "<parameter=".len();
        let key_end = inner[key_start..].find('>')? + key_start;
        if key_end <= key_start {
            break;
        }
        let key = inner[key_start..key_end].trim();
        let value_start = key_end + 1;
        let value_end = inner[value_start..]
            .find("</parameter>")
            .map(|v| v + value_start)?;
        if value_end <= value_start {
            break;
        }
        let value_raw = inner[value_start..value_end].trim();
        let value = serde_json::from_str::<Value>(value_raw)
            .unwrap_or_else(|_| Value::String(value_raw.to_string()));
        params.insert(key.to_string(), value);
        pos = value_end + "</parameter>".len();
    }

    let args = Value::Object(params);
    let args_str = serde_json::to_string(&args).ok()?;

    let call = ToolCall {
        index: Some(*call_id),
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: super::FunctionCall {
            name: func_name.to_string(),
            arguments: args_str,
        },
    };
    *call_id += 1;
    Some(call)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_qwen_format() {
        let parser = ToolParser::new();
        let text = r#"I'll help you with the weather.
<tool_call>
{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}
</tool_call>"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Tokyo"));
    }

    #[test]
    fn test_parse_json_format() {
        let parser = ToolParser::new();
        let text = r#"{"name": "calculate", "arguments": {"expression": "2+2"}}"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calculate");
    }

    #[test]
    fn test_parse_code_block() {
        let parser = ToolParser::new();
        let text = r#"Let me search for that:

```json
{"name": "search", "arguments": {"query": "rust programming"}}
```"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_multiple_tool_calls() {
        let parser = ToolParser::new();
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "London"}}
</tool_call>"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_has_tool_calls() {
        let parser = ToolParser::new();

        assert!(parser.has_tool_calls("<tool_call>{}</tool_call>"));
        assert!(parser.has_tool_calls(r#"{"name": "foo", "arguments": {}}"#));
        assert!(!parser.has_tool_calls("Just a normal response"));
    }

    #[test]
    fn test_parse_function_tag_format() {
        let parser = ToolParser::new();
        let text = r#"<tool_call>
<function=my_tool>
<parameter=foo>
{"bar": 1}
</parameter>
<parameter=baz>
qux
</parameter>
</function>
</tool_call>"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "my_tool");
        assert!(calls[0].function.arguments.contains("\"foo\""));
        assert!(calls[0].function.arguments.contains("\"baz\""));
    }
}
