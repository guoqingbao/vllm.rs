// src/server/parser.rs
//! Streaming tool call parser for detecting and buffering tool calls during streaming.
//! Handles model-specific tool call tokens and formats.

use crate::server::{ChatChoiceChunk, ChatCompletionChunk, Delta};
use crate::tools::{FunctionCall, ToolCall};
use crate::utils::config::ModelType;
use serde_json::Value;
use std::collections::HashSet;
use tokenizers::Tokenizer;

/// Parser state for streaming tool call detection
#[derive(Debug, Clone, PartialEq)]
pub enum ParserState {
    /// Normal streaming mode - tokens pass through
    Normal,
    /// Potential start tag detected (partial match)
    MaybeStart,
    /// Buffering mode - accumulating confirmed tool call content
    Buffering,
}

/// Result of processing a token in the stream
#[derive(Debug, Clone)]
pub enum StreamResult {
    /// Normal content - send to client
    Content(String),
    /// Buffering - don't send anything yet
    Buffering,
    /// Tool calls parsed - return tool calls for deferred emission
    ToolCalls(Vec<ToolCall>),
    /// False positive - flush accumulated buffer as content
    FlushBuffer(String),
}

/// Configuration for model-specific tool call detection
#[derive(Clone, Debug)]
pub struct ToolConfig {
    pub start_token_ids: HashSet<u32>,
    pub end_token_ids: HashSet<u32>,
    pub start_token_str: String,
    pub end_token_str: String,
}

impl ToolConfig {
    /// Create tool config for a specific model type
    pub fn for_model_type(model_type: &ModelType) -> Self {
        let mut start_ids = HashSet::new();
        let mut end_ids = HashSet::new();

        match model_type {
            ModelType::LLaMa => {
                // Llama 3/3.1
                start_ids.insert(128010); // <|python_tag|>
                end_ids.insert(128008); // <|eom_id|>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<|python_tag|>".to_string(),
                    end_token_str: "<|eom_id|>".to_string(),
                }
            }
            ModelType::Qwen3 | ModelType::Qwen3MoE | ModelType::Qwen3VL => {
                // Qwen 2.5 / 3
                start_ids.insert(151657); // <tool_call>
                end_ids.insert(151658); // </tool_call>
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<tool_call>".to_string(),
                    end_token_str: "</tool_call>".to_string(),
                }
            }
            ModelType::Mistral | ModelType::Mistral3VL => {
                // Mistral v3
                start_ids.insert(9); // [TOOL_CALLS]
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "[TOOL_CALLS]".to_string(),
                    end_token_str: "]".to_string(),
                }
            }
            ModelType::Gemma | ModelType::Gemma3 => {
                // Gemma 2/3 - uses text-only matching
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<start_function_call>".to_string(),
                    end_token_str: "<end_function_call>".to_string(),
                }
            }
            // Phi, GLM, Yi, StableLM, DeepSeek - use Qwen format (text-only)
            ModelType::Phi
            | ModelType::Phi4
            | ModelType::GLM4
            | ModelType::GLM4MoE
            | ModelType::Yi
            | ModelType::StableLM
            | ModelType::DeepSeek => ToolConfig {
                start_token_ids: HashSet::new(),
                end_token_ids: HashSet::new(),
                start_token_str: "<tool_call>".to_string(),
                end_token_str: "</tool_call>".to_string(),
            },
        }
    }

    /// Returns true if this config has special token IDs for detection
    pub fn has_special_tokens(&self) -> bool {
        self.has_start_tokens()
    }

    /// Returns true if start token IDs are available
    pub fn has_start_tokens(&self) -> bool {
        !self.start_token_ids.is_empty()
    }

    /// Returns true if end token IDs are available
    pub fn has_end_tokens(&self) -> bool {
        !self.end_token_ids.is_empty()
    }

    /// Validate special token IDs against the tokenizer, falling back to text-only matching if needed.
    pub fn validate_with_tokenizer(&mut self, tokenizer: &Tokenizer, model_type: &ModelType) {
        if self.has_start_tokens()
            && !Self::matches_single_token(tokenizer, &self.start_token_str, &self.start_token_ids)
        {
            crate::log_warn!(
                "Tool start token IDs not supported by tokenizer for model {:?}, falling back to text matching",
                model_type
            );
            self.start_token_ids.clear();
        }

        if self.has_end_tokens()
            && !Self::matches_single_token(tokenizer, &self.end_token_str, &self.end_token_ids)
        {
            crate::log_error!(
                "Tool end token IDs not supported by tokenizer for model {:?}, falling back to text matching",
                model_type
            );
            self.end_token_ids.clear();
        }
    }

    /// Resolve tool call end token IDs using tokenizer and the validated config.
    pub fn tool_call_end_ids(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        let mut tool_call_end_ids: Vec<u32> = Vec::new();

        let mut used_special = false;
        if self.has_end_tokens() {
            let mut use_special = true;
            if !self.end_token_str.is_empty() {
                if let Ok(encoded) = tokenizer.encode(self.end_token_str.as_str(), false) {
                    let ids = encoded.get_ids();
                    if ids.len() != 1 || !self.end_token_ids.contains(&ids[0]) {
                        use_special = false;
                    }
                } else {
                    use_special = false;
                }
            }
            if use_special {
                tool_call_end_ids.extend(self.end_token_ids.iter().copied());
                used_special = true;
            }
        }

        if !used_special && !self.end_token_str.is_empty() && self.end_token_str.starts_with('<') {
            // Only use text tags that look like explicit tool markers to avoid false positives.
            if let Ok(encoded) = tokenizer.encode(self.end_token_str.as_str(), false) {
                let ids = encoded.get_ids();
                if ids.len() == 1 {
                    tool_call_end_ids.push(ids[0]);
                }
            }
        }

        tool_call_end_ids
    }

    fn matches_single_token(tokenizer: &Tokenizer, text: &str, token_ids: &HashSet<u32>) -> bool {
        if text.is_empty() {
            return false;
        }
        match tokenizer.encode(text, false) {
            Ok(encoded) => {
                let ids = encoded.get_ids();
                ids.len() == 1 && token_ids.contains(&ids[0])
            }
            Err(_) => false,
        }
    }
}

/// Streaming tool parser that handles tool call detection and buffering
pub struct StreamToolParser {
    config: ToolConfig,
    state: ParserState,
    buffer: String,
    model_id: String,
    parse_strategy: String,
    // Accumulated output for final parsing
    accumulated_output: String,
    // Reasoning block tracking
    active_reasoning_end: Option<&'static str>,
    // Code block tracking
    in_code_block: bool,
    // Tool call index counter
    tool_call_index: usize,
}

/// Reasoning marker pairs: (start, end)
const REASONING_MARKERS: &[(&str, &str)] = &[
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("[THINK]", "[/THINK]"),
    ("<thought>", "</thought>"),
];

impl StreamToolParser {
    /// Create a new parser for the given model type
    pub fn new(model_type: ModelType, model_id: String) -> Self {
        let config = ToolConfig::for_model_type(&model_type);
        Self::new_with_config(&model_type, model_id, config)
    }

    /// Create a new parser with a pre-validated tool config
    pub fn new_with_config(model_type: &ModelType, model_id: String, config: ToolConfig) -> Self {
        let parse_strategy = match model_type {
            ModelType::Mistral | ModelType::Mistral3VL => "mistral_list",
            _ => "json",
        }
        .to_string();

        Self {
            config,
            state: ParserState::Normal,
            buffer: String::new(),
            model_id,
            parse_strategy,
            accumulated_output: String::new(),
            active_reasoning_end: None,
            in_code_block: false,
            tool_call_index: 0,
        }
    }

    /// Check if currently inside a reasoning block
    pub fn in_reasoning(&self) -> bool {
        self.active_reasoning_end.is_some()
    }

    /// Check if currently inside a code block
    pub fn in_code_block(&self) -> bool {
        self.in_code_block
    }

    /// Get the current parser state
    pub fn state(&self) -> &ParserState {
        &self.state
    }

    /// Get accumulated output for debugging/logging
    pub fn accumulated_output(&self) -> &str {
        &self.accumulated_output
    }

    /// Get the buffered content
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Process a single incoming token.
    /// Returns StreamResult indicating what action to take.
    pub fn process_token(&mut self, token_id: u32, token_text: &str) -> StreamResult {
        // Always accumulate
        self.accumulated_output.push_str(token_text);

        // Measure code block start/end markers in the buffer
        let mut code_block_count = 0;
        for line in  self.accumulated_output.clone().lines() {
            // account for labled code block starts
            if line.trim().starts_with("```") {
                code_block_count += 1;
            }
        }
        // Even number indicates blocks are closed, odd
        self.in_code_block = code_block_count % 2 == 1;

        // Track reasoning blocks
        if self.active_reasoning_end.is_none() {
            for &(start, end) in REASONING_MARKERS {
                if token_text.contains(start) || self.accumulated_output.ends_with(start) {
                    self.active_reasoning_end = Some(end);
                    break;
                }
            }
        } else if let Some(end_marker) = self.active_reasoning_end {
            if token_text.contains(end_marker) || self.accumulated_output.ends_with(end_marker) {
                self.active_reasoning_end = None;
            }
        }

        // Don't detect tool calls inside reasoning or code blocks
        if self.in_reasoning() || self.in_code_block {
            return StreamResult::Content(token_text.to_string());
        }

        match self.state.clone() {
            ParserState::Normal => {
                // Check for start trigger - which should be the only token on the line
                if self.is_start_token(token_id, token_text)
                    && (self.config.start_token_str.trim() != "<tool_call>" // ignore line check for other formats
                        || token_text
                            .lines()
                            .any(|l| l.starts_with(self.config.start_token_str.trim())))
                {
                    let pos = token_text.find(&self.config.start_token_str).unwrap(); // infallible due to check above
                    self.state = ParserState::Buffering;
                    self.buffer.clear();
                    let before = &token_text[..pos];
                    let after = &token_text[pos + self.config.start_token_str.len()..];
                    if !after.is_empty() {
                        self.buffer.push_str(after);
                    }

                    crate::log_info!(
                        "Tool call {} ({}) found, start buffering!",
                        token_text,
                        token_id
                    );
                    if !before.is_empty() {
                        return StreamResult::Content(before.to_string());
                    }

                    return StreamResult::Buffering;
                }

                // Check for partial tag match at end of current token
                if !self.config.has_start_tokens() {
                    if let Some((prefix, partial)) = self.split_partial_start(token_text) {
                        self.state = ParserState::MaybeStart;
                        self.buffer.clear();
                        self.buffer.push_str(&partial);
                        return if prefix.is_empty() {
                            StreamResult::Buffering
                        } else {
                            StreamResult::Content(prefix)
                        };
                    }
                }

                // Normal content
                StreamResult::Content(token_text.to_string())
            }
            ParserState::MaybeStart => {
                self.buffer.push_str(token_text);

                if let Some(tag_pos) = self.buffer.find(&self.config.start_token_str) {
                    let before = self.buffer[..tag_pos].to_string();
                    let after =
                        self.buffer[tag_pos + self.config.start_token_str.len()..].to_string();
                    self.buffer.clear();
                    if !after.is_empty() {
                        self.buffer.push_str(&after);
                    }
                    self.state = ParserState::Buffering;
                    return if before.is_empty() {
                        StreamResult::Buffering
                    } else {
                        StreamResult::Content(before)
                    };
                }

                if self.partial_suffix_len(&self.buffer) > 0 {
                    return StreamResult::Buffering;
                }

                // False alarm - not a tool call tag
                self.state = ParserState::Normal;
                let flushed = self.buffer.clone();
                self.buffer.clear();
                StreamResult::FlushBuffer(flushed)
            }
            ParserState::Buffering => {
                self.buffer.push_str(token_text);

                let end_reached = self.is_end_token(token_id, token_text)
                    || self.buffer_has_end_tag()
                    || self.maybe_complete_mistral_list();
                if end_reached {
                    crate::log_info!(
                        "Tool call buffering end, reached {} ({})",
                        token_text,
                        token_id
                    );

                    let tool_calls = self.parse_buffer();
                    let result = if tool_calls.is_empty() {
                        // Parse failed - return buffered content
                        crate::log_error!(
                            "Unable to parse tool call buffer: {}\n of accumulated buffer: {}",
                            self.buffer, self.accumulated_output
                        );
                        StreamResult::FlushBuffer(self.buffer.clone())
                    } else {
                        StreamResult::ToolCalls(tool_calls)
                    };
                    self.buffer.clear();
                    self.state = ParserState::Normal;
                    return result;
                }

                StreamResult::Buffering
            }
        }
    }

    /// Finalize parsing when stream ends
    pub fn finalize(&mut self) -> Option<Vec<ToolCall>> {
        match self.state {
            ParserState::Buffering => {
                if self.buffer.is_empty() {
                    self.state = ParserState::Normal;
                    return None;
                }
                let tool_calls = self.parse_buffer();
                if !tool_calls.is_empty() {
                    self.buffer.clear();
                    self.state = ParserState::Normal;
                    return Some(tool_calls);
                }
                // Leave buffer intact so caller can flush it.
                self.state = ParserState::Normal;
            }
            ParserState::MaybeStart => {
                self.state = ParserState::Normal;
            }
            ParserState::Normal => {}
        }
        None
    }

    /// Drain the buffer and reset parser state.
    pub fn take_buffer(&mut self) -> String {
        self.state = ParserState::Normal;
        std::mem::take(&mut self.buffer)
    }

    /// Check if token/text matches start trigger
    fn is_start_token(&self, id: u32, text: &str) -> bool {
        // Token ID match (if available)
        if self.config.has_start_tokens() {
            return self.config.start_token_ids.contains(&id);
        }
        // Text match
        text.contains(&self.config.start_token_str)
    }

    /// Check if token/text matches end trigger
    fn is_end_token(&self, id: u32, text: &str) -> bool {
        // Token ID match (if available)
        if self.config.has_end_tokens() {
            return self.config.end_token_ids.contains(&id);
        }
        if self.parse_strategy == "mistral_list" && self.config.end_token_str == "]" {
            return false;
        }
        // Text match
        text.contains(&self.config.end_token_str)
    }

    /// Parse buffered content into tool calls
    fn parse_buffer(&mut self) -> Vec<ToolCall> {
        let mut clean_text = self.buffer.trim().to_string();
        if self.should_strip_end_tag() {
            if let Some(pos) = clean_text.rfind(&self.config.end_token_str) {
                clean_text.truncate(pos);
            }
        }
        let mut calls = Vec::new();

        // Strategy 1: Mistral List [ {...}, {...} ]
        if self.parse_strategy == "mistral_list" && clean_text.starts_with('[') {
            if let Ok(list) = serde_json::from_str::<Vec<Value>>(&clean_text) {
                for item in list.iter() {
                    if let Some(call) = self.json_to_tool_call(item) {
                        calls.push(call);
                    }
                }
            }
        }
        // Strategy 2: Single JSON Object (Qwen, Llama, Phi)
        else if let Ok(item) = serde_json::from_str::<Value>(&clean_text) {
            if let Some(call) = self.json_to_tool_call(&item) {
                calls.push(call);
            }
        }
        // Strategy 3: QwenCoder XML-style function tags
        else if clean_text.starts_with("<function=") && clean_text.contains("</function>") {
            // Extract function name from <function=...> tag
            let func_start = "<function=".len();
            let func_end = clean_text.find('>').unwrap_or(0);
            if func_end > func_start {
                let func_name = &clean_text[func_start..func_end];

                // Find parameter section
                let params_start = clean_text.find("<parameter=");
                let params_end = clean_text.find("</function>");

                let mut params = std::collections::HashMap::new();

                if let (Some(start), Some(end)) = (params_start, params_end) {
                    let param_content = &clean_text[start..end];

                    // Parse all parameter tags
                    let mut pos = 0;
                    while pos < param_content.len() {
                        let param_start_tag = param_content[pos..].find("<parameter=");
                        if param_start_tag.is_none() {
                            break;
                        }
                        let param_start_tag = param_start_tag.unwrap() + pos;

                        let key_start = param_start_tag + "<parameter=".len();
                        let key_end = param_content[key_start..].find(">").unwrap_or(0) + key_start;

                        if key_end > key_start {
                            let key = &param_content[key_start..key_end];

                            let value_start = key_end + 1;
                            let value_end = param_content[value_start..]
                                .find("</parameter>")
                                .unwrap_or(0)
                                + value_start;

                            if value_end > value_start {
                                let value = &param_content[value_start..value_end];
                                params.insert(key.to_string(), value.trim().to_string());
                            }
                        }

                        // Move position past this parameter
                        let next_pos = param_content[pos..].find("</parameter>").unwrap_or(0)
                            + pos
                            + "</parameter>".len();
                        if next_pos <= pos {
                            break;
                        }
                        pos = next_pos;
                    }
                }

                if let Ok(args) = serde_json::to_string(&params) {
                    let call = ToolCall {
                        index: Some(self.tool_call_index),
                        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: func_name.to_string(),
                            arguments: args,
                        },
                    };

                    self.tool_call_index += 1;
                    calls.push(call);
                }
            }
        }
        // Strategy 4: Repair unbalanced JSON and retry
        else if let Some(repaired) = self.repair_unbalanced_json(&clean_text) {
            if repaired != clean_text {
                crate::log_warn!("Tool call JSON missing closing braces; attempting repair");
            }
            if let Ok(item) = serde_json::from_str::<Value>(&repaired) {
                if let Some(call) = self.json_to_tool_call(&item) {
                    calls.push(call);
                }
            }
        }

        calls
    }

    fn split_partial_start(&self, text: &str) -> Option<(String, String)> {
        let tag = &self.config.start_token_str;
        let suffix_len = self.partial_suffix_len(text);

        // If no valid partial match (not less than full length), skip.
        if suffix_len == 0 || suffix_len >= tag.len() {
            return None;
        }

        let prefix = text[..text.len() - suffix_len].to_string();
        let partial = text[text.len() - suffix_len..].to_string();

        // For "<tool_call>" tool call tags only: enforce that the partial match starts on a new line.
        if self.config.start_token_str == "<tool_call>" {
            if prefix.len() == 0 || prefix.ends_with("\n") || prefix.ends_with("\r\n") {
                return Some((prefix, partial));
            } else {
                return None;
            }
        }

        Some((prefix, partial))
    }

    fn partial_suffix_len(&self, text: &str) -> usize {
        let tag = &self.config.start_token_str;
        let max = std::cmp::min(tag.len(), text.len());
        for i in (1..=max).rev() {
            if text.ends_with(&tag[..i]) {
                return i;
            }
        }
        0
    }

    fn buffer_has_end_tag(&self) -> bool {
        if self.config.end_token_str.is_empty() {
            return false;
        }
        if self.config.has_end_tokens() {
            return false;
        }
        if self.parse_strategy == "mistral_list" && self.config.end_token_str == "]" {
            return false;
        }
        self.buffer.contains(&self.config.end_token_str)
    }

    fn maybe_complete_mistral_list(&self) -> bool {
        if self.parse_strategy != "mistral_list" {
            return false;
        }
        let trimmed = self.buffer.trim();
        if !trimmed.ends_with(']') {
            return false;
        }
        serde_json::from_str::<Vec<Value>>(trimmed).is_ok()
    }

    fn should_strip_end_tag(&self) -> bool {
        let end_tag = self.config.end_token_str.as_str();
        if end_tag.is_empty() {
            return false;
        }
        if self.parse_strategy == "mistral_list" && end_tag == "]" {
            return false;
        }
        end_tag.starts_with('<')
    }

    fn repair_unbalanced_json(&self, text: &str) -> Option<String> {
        let trimmed = text.trim();
        if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
            return None;
        }

        let mut in_string = false;
        let mut escape = false;
        let mut open_braces = 0usize;
        let mut close_braces = 0usize;
        let mut open_brackets = 0usize;
        let mut close_brackets = 0usize;

        for ch in trimmed.chars() {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' if in_string => {
                    escape = true;
                }
                '"' => {
                    in_string = !in_string;
                }
                '{' if !in_string => open_braces += 1,
                '}' if !in_string => close_braces += 1,
                '[' if !in_string => open_brackets += 1,
                ']' if !in_string => close_brackets += 1,
                _ => {}
            }
        }

        if in_string {
            return None;
        }
        if close_braces > open_braces || close_brackets > open_brackets {
            return None;
        }

        if open_braces == close_braces && open_brackets == close_brackets {
            return None;
        }

        let mut fixed = trimmed.to_string();
        if open_brackets > close_brackets {
            fixed.push_str(&"]".repeat(open_brackets - close_brackets));
        }
        if open_braces > close_braces {
            fixed.push_str(&"}".repeat(open_braces - close_braces));
        }
        Some(fixed)
    }

    /// Convert JSON value to ToolCall
    fn json_to_tool_call(&mut self, item: &Value) -> Option<ToolCall> {
        let name = item["name"].as_str()?.to_string();
        let arguments = if let Some(args) = item.get("arguments") {
            if args.is_string() {
                args.as_str().unwrap_or("{}").to_string()
            } else {
                args.to_string()
            }
        } else {
            "{}".to_string()
        };

        let call = ToolCall {
            index: Some(self.tool_call_index),
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            call_type: "function".to_string(),
            function: FunctionCall { name, arguments },
        };
        self.tool_call_index += 1;
        Some(call)
    }

    // --- Chunk creation helpers (for use by server.rs) ---

    /// Create a content chunk for streaming
    pub fn create_content_chunk(&self, content: &str) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "chat.completion.chunk",
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: self.model_id.clone(),
            choices: vec![ChatChoiceChunk {
                index: 0,
                delta: Delta {
                    content: Some(content.to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
                error: None,
            }],
            usage: None,
        }
    }

    /// Create a tool call chunk for streaming
    pub fn create_tool_chunk(&self, tools: Vec<ToolCall>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "chat.completion.chunk",
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: self.model_id.clone(),
            choices: vec![ChatChoiceChunk {
                index: 0,
                delta: Delta {
                    content: None,
                    tool_calls: Some(tools),
                },
                finish_reason: None,
                error: None,
            }],
            usage: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::config::ModelType;

    #[test]
    fn test_tool_config_qwen() {
        let config = ToolConfig::for_model_type(&ModelType::Qwen3);
        assert!(config.has_special_tokens());
        assert!(config.start_token_ids.contains(&151657));
        assert_eq!(config.start_token_str, "<tool_call>");
    }

    #[test]
    fn test_tool_config_default() {
        let config = ToolConfig::for_model_type(&ModelType::Phi);
        assert!(!config.has_special_tokens());
        assert_eq!(config.start_token_str, "<tool_call>");
    }

    #[test]
    fn test_parser_normal_content() {
        let mut parser = StreamToolParser::new(ModelType::Qwen3, "qwen3".to_string());
        match parser.process_token(0, "Hello world") {
            StreamResult::Content(s) => assert_eq!(s, "Hello world"),
            _ => panic!("Expected Content"),
        }
    }

    #[test]
    fn test_parser_tool_call_detection() {
        let mut parser = StreamToolParser::new(ModelType::Qwen3, "qwen3".to_string());

        // Start tag triggers buffering
        match parser.process_token(151657, "<tool_call>") {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on start tag"),
        }

        // Content is buffered
        match parser.process_token(0, r#"{"name": "test", "arguments": {}}"#) {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering"),
        }

        // End tag triggers parsing
        match parser.process_token(151658, "</tool_call>") {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_parser_partial_start_text_mode() {
        let mut parser = StreamToolParser::new(ModelType::Phi, "phi".to_string());

        // Partial start tag splits across tokens
        match parser.process_token(0, "<tool_") {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on partial start"),
        }
        match parser.process_token(0, "call>") {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on completed start"),
        }
        match parser.process_token(0, r#"{"name": "test", "arguments": {}}"#) {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering"),
        }
        match parser.process_token(0, "</tool_call>") {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_parser_token_id_strict_match() {
        let mut parser = StreamToolParser::new(ModelType::Qwen3, "qwen3".to_string());

        // Text match should not trigger when token IDs are available
        match parser.process_token(0, "<tool_call>") {
            StreamResult::Content(text) => assert_eq!(text, "<tool_call>"),
            _ => panic!("Expected Content without token ID match"),
        }
    }
}
