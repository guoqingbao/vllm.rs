// src/server/parser.rs
//! Streaming tool call parser for detecting and buffering tool calls during streaming.
//! Handles model-specific tool call tokens and formats.

use crate::server::{ChatChoiceChunk, ChatCompletionChunk, Delta};
use crate::tools::parser::{parse_tool_calls_from_text, prefix_could_be_tool};
use crate::tools::ToolCall;
use crate::utils::config::ModelType;
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
    pub start_is_special: bool,
    pub end_is_special: bool,
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
                    start_is_special: false,
                    end_is_special: false,
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
                    start_is_special: false,
                    end_is_special: false,
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
                    start_is_special: false,
                    end_is_special: false,
                }
            }
            ModelType::Gemma | ModelType::Gemma3 => {
                // Gemma 2/3 - uses text-only matching
                ToolConfig {
                    start_token_ids: start_ids,
                    end_token_ids: end_ids,
                    start_token_str: "<start_function_call>".to_string(),
                    end_token_str: "<end_function_call>".to_string(),
                    start_is_special: false,
                    end_is_special: false,
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
                start_is_special: false,
                end_is_special: false,
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

        self.start_is_special = Self::is_special_token(tokenizer, &self.start_token_str);
        self.end_is_special = Self::is_special_token(tokenizer, &self.end_token_str);
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

    fn is_special_token(tokenizer: &Tokenizer, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }
        let encoded = match tokenizer.encode(text, false) {
            Ok(encoded) => encoded,
            Err(_) => return false,
        };
        let ids = encoded.get_ids();
        if ids.len() != 1 {
            return false;
        }
        let id = ids[0];
        let added = tokenizer.get_added_tokens_decoder();
        if let Some(info) = added.get(&id) {
            if info.content == text
                && (info.special || (info.content.starts_with('<') && info.content.ends_with('>')))
            {
                return true;
            }
        }
        false
    }
}

/// Streaming tool parser that handles tool call detection and buffering
pub struct StreamToolParser {
    #[allow(dead_code)]
    config: ToolConfig,
    state: ParserState,
    buffer: String,
    model_id: String,
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
    pub fn new_with_config(_model_type: &ModelType, model_id: String, config: ToolConfig) -> Self {
        Self {
            config,
            state: ParserState::Normal,
            buffer: String::new(),
            model_id,
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
    pub fn process_token(&mut self, _token_id: u32, token_text: &str) -> StreamResult {
        // Always accumulate
        self.accumulated_output.push_str(token_text);

        // Measure code block start/end markers in the buffer
        let mut code_block_count = 0;
        for line in self.accumulated_output.clone().lines() {
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
            if !self.buffer.is_empty() {
                let flushed = self.take_buffer();
                return StreamResult::FlushBuffer(format!("{}{}", flushed, token_text));
            }
            return StreamResult::Content(token_text.to_string());
        }

        self.buffer.push_str(token_text);

        let (could_be_tool, tool_complete) = prefix_could_be_tool(&self.buffer);
        if could_be_tool || tool_complete {
            self.state = ParserState::Buffering;
            if tool_complete {
                let mut tool_calls =
                    parse_tool_calls_from_text(&self.buffer, &mut self.tool_call_index);
                let result = if tool_calls.is_empty() {
                    crate::log_error!(
                        "Unable to parse tool call buffer: {}\n of accumulated buffer: {}",
                        self.buffer,
                        self.accumulated_output
                    );
                    StreamResult::FlushBuffer(self.buffer.clone())
                } else {
                    StreamResult::ToolCalls(std::mem::take(&mut tool_calls))
                };
                self.buffer.clear();
                self.state = ParserState::Normal;
                return result;
            }
            return StreamResult::Buffering;
        }

        // Not a tool call - flush buffered content
        self.state = ParserState::Normal;
        let flushed = std::mem::take(&mut self.buffer);
        StreamResult::Content(flushed)
    }

    /// Finalize parsing when stream ends
    pub fn finalize(&mut self) -> Option<Vec<ToolCall>> {
        match self.state {
            ParserState::Buffering => {
                if self.buffer.is_empty() {
                    self.state = ParserState::Normal;
                    return None;
                }
                let mut tool_calls =
                    parse_tool_calls_from_text(&self.buffer, &mut self.tool_call_index);
                if !tool_calls.is_empty() {
                    self.buffer.clear();
                    self.state = ParserState::Normal;
                    return Some(std::mem::take(&mut tool_calls));
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

    // legacy parsing helpers removed in favor of mistral-style JSON prefix checks

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

// legacy partial JSON checks moved to tools::parser::prefix_could_be_tool

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
        match parser.process_token(0, "<tool_call>") {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on start tag"),
        }

        // Content is buffered
        match parser.process_token(0, r#"{"name": "test", "arguments": {}}"#) {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering"),
        }

        // End tag triggers parsing
        match parser.process_token(0, "</tool_call>") {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_parser_tool_call_array() {
        let mut parser = StreamToolParser::new(ModelType::Mistral, "mistral".to_string());
        let payload =
            "[TOOL_CALLS][{\"name\":\"a\",\"arguments\":{}},{\"name\":\"b\",\"arguments\":{}}]";
        match parser.process_token(0, payload) {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].function.name, "a");
                assert_eq!(calls[1].function.name, "b");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_parser_function_tag_tool_call() {
        let mut parser = StreamToolParser::new(ModelType::Qwen3, "qwen3".to_string());
        let payload = r#"<tool_call><function=my_tool><parameter=foo>{"bar":1}</parameter></function></tool_call>"#;
        match parser.process_token(0, payload) {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "my_tool");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[test]
    fn test_parser_non_tool_content_flushes() {
        let mut parser = StreamToolParser::new(ModelType::Phi, "phi".to_string());

        match parser.process_token(0, "Hello ") {
            StreamResult::Content(text) => assert_eq!(text, "Hello "),
            _ => panic!("Expected Content without token ID match"),
        }
    }
}
