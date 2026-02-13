// src/server/parser.rs
//! Streaming tool call parser for detecting and buffering tool calls during streaming.
//! Handles model-specific tool call tokens and formats.

use crate::server::{ChatChoiceChunk, ChatCompletionChunk, Delta};
use crate::tools::{Tool, ToolCall};
use crate::utils::config::ModelType;
use serde_json::Map;
use serde_json::Value;
use std::collections::HashSet;
use tokenizers::Tokenizer;
use tool_parser::{
    types::{StreamingParseResult, ToolCallItem},
    ParserFactory, ToolParser as ExternalToolParser,
};
/// Parser state for streaming tool call detection
#[derive(Debug, Clone, PartialEq)]
pub enum ParserState {
    /// Normal streaming mode - tokens pass through
    Normal,
    /// Potential start tag detected (partial match)
    // MaybeStart,
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
            ModelType::Qwen3
            | ModelType::Qwen3MoE
            | ModelType::Qwen3_5
            | ModelType::Qwen3_5MoE
            | ModelType::Qwen3VL => {
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
    parser: Box<dyn ExternalToolParser>,
    tools: Vec<Tool>,
    streaming_calls: Vec<StreamingToolCallState>,
    // Accumulated output for final parsing
    accumulated_output: String,
    // Reasoning block tracking
    active_reasoning_end: Option<&'static str>,
    // Code block tracking
    in_code_block: bool,
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
        Self::new_with_config(&model_type, model_id, config, Vec::new(), None)
    }

    /// Create a new parser with a pre-validated tool config
    pub fn new_with_config(
        model_type: &ModelType,
        model_id: String,
        config: ToolConfig,
        tools: Vec<Tool>,
        enforce_parser: Option<String>,
    ) -> Self {
        let parse_strategy = match model_type {
            ModelType::Mistral | ModelType::Mistral3VL => "mistral_list",
            _ => "json",
        }
        .to_string();

        let factory = ParserFactory::new();
        let parser_name = if let Some(name) = enforce_parser.as_ref().and_then(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }) {
            if !factory.registry().has_parser(name) {
                let valid = factory.list_parsers().join(", ");
                panic!(
                    "Invalid enforce-parser '{}'. Valid parsers: {}",
                    name, valid
                );
            }
            name
        } else {
            Self::parser_name_for_model(model_type, &model_id)
        };
        if !tools.is_empty() {
            crate::log_info!(
                "Tool parser selected: {} (model_id={}, enforce_parser={})",
                parser_name,
                model_id,
                enforce_parser.as_deref().unwrap_or("none")
            );
        }
        let parser = factory
            .registry()
            .create_parser(parser_name)
            .or_else(|| factory.registry().create_for_model(&model_id))
            .or_else(|| factory.registry().create_parser("passthrough"))
            .expect("tool parser available");

        Self {
            config,
            state: ParserState::Normal,
            buffer: String::new(),
            model_id,
            parse_strategy,
            parser,
            tools,
            streaming_calls: Vec::new(),
            accumulated_output: String::new(),
            active_reasoning_end: None,
            in_code_block: false,
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
    pub async fn process_token(&mut self, token_id: u32, token_text: &str) -> StreamResult {
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
            return StreamResult::Content(token_text.to_string());
        }

        match self.state.clone() {
            ParserState::Normal => {
                // Check for start trigger
                if self.is_start_token(token_id, token_text) {
                    self.state = ParserState::Buffering;
                    self.buffer.clear();
                    self.buffer.push_str(token_text);
                    self.streaming_calls.clear();
                    if let Ok(result) = self.parser.parse_incremental(token_text, &self.tools).await
                    {
                        self.apply_streaming_result(&result);
                    }

                    crate::log_info!(
                        "Tool call {} ({}) found, start buffering!",
                        token_text,
                        token_id
                    );
                    return StreamResult::Buffering;
                }
                // Normal content
                StreamResult::Content(token_text.to_string())
            }
            ParserState::Buffering => {
                self.buffer.push_str(token_text);
                if let Ok(result) = self.parser.parse_incremental(token_text, &self.tools).await {
                    if !result.calls.is_empty() {
                        crate::log_info!("Stream parsing: {:?}", result.calls);
                    }
                    self.apply_streaming_result(&result);
                }
                let end_reached = self.is_end_token(token_id, token_text)
                    || self.buffer_has_end_tag()
                    || self.maybe_complete_mistral_list();
                if end_reached {
                    crate::log_info!(
                        "Tool call buffering end, reached {} ({})",
                        token_text,
                        token_id
                    );

                    if let Some(unstreamed) = self.parser.get_unstreamed_tool_args() {
                        self.apply_stream_items(&unstreamed);
                    }
                    let mut tool_calls = self.build_tool_calls_from_streaming();
                    if tool_calls.is_empty() {
                        crate::log_info!(
                            "Fallback to non-stream parsing for buffer: {}",
                            self.buffer
                        );
                        tool_calls = self.parse_complete_with_fallback(&self.buffer).await;
                    }
                    let result = if tool_calls.is_empty() {
                        // Parse failed - return buffered content
                        crate::log_error!("Unable to parse tool call buffer: {}", self.buffer,);
                        StreamResult::FlushBuffer(self.buffer.clone())
                    } else {
                        StreamResult::ToolCalls(tool_calls)
                    };
                    self.parser.reset();
                    self.buffer.clear();
                    self.state = ParserState::Normal;
                    self.streaming_calls.clear();
                    return result;
                }

                StreamResult::Buffering
            }
        }
    }

    /// Drain the buffer and reset parser state.
    pub fn take_buffer(&mut self) -> String {
        self.state = ParserState::Normal;
        std::mem::take(&mut self.buffer)
    }

    /// Check if token/text matches start trigger
    fn is_start_token(&self, id: u32, text: &str) -> bool {
        match self.accumulated_output[..self.accumulated_output.len() - text.len()]
            .chars()
            .last()
        {
            // Empty buffer or newline are valid "start of line" checks for tool calls
            None | Some('\n') => {}
            _ => return false,
        };
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

    fn apply_streaming_result(&mut self, result: &StreamingParseResult) {
        if !result.calls.is_empty() {
            self.apply_stream_items(&result.calls);
        }
    }

    fn apply_stream_items(&mut self, items: &[ToolCallItem]) {
        for item in items {
            if self.streaming_calls.len() <= item.tool_index {
                self.streaming_calls
                    .resize_with(item.tool_index + 1, StreamingToolCallState::default);
            }
            let state = &mut self.streaming_calls[item.tool_index];
            if let Some(name) = &item.name {
                state.name = Some(name.clone());
            }
            if !item.parameters.is_empty() {
                state.arguments.push_str(&item.parameters);
            }
        }
    }

    fn build_tool_calls_from_streaming(&mut self) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        crate::log_info!("Building tool call: {:?}", self.streaming_calls);
        for state in &self.streaming_calls {
            let Some(name) = &state.name else { continue };
            let args = if state.arguments.trim().is_empty() {
                "{}".to_string()
            } else {
                state.arguments.clone()
            };
            calls.push(crate::tools::new_tool_call(
                format!("call_{}", uuid::Uuid::new_v4().simple()),
                name.clone(),
                args,
            ));
        }
        calls
    }

    pub async fn parse_complete_with_fallback(&self, text: &str) -> Vec<ToolCall> {
        let mut parsed_calls = match self.parser.parse_complete(text).await {
            Ok((_normal_text, calls)) => calls,
            Err(err) => {
                crate::log_warn!("Tool parse failed: {:?}", err);
                Vec::new()
            }
        };

        if parsed_calls.is_empty() && text.contains("<function=") {
            let factory = ParserFactory::new();
            if let Some(xml_parser) = factory.registry().create_parser("qwen_coder") {
                if let Ok((_normal_text, calls)) = xml_parser.parse_complete(text).await {
                    parsed_calls = calls;
                }
            }
        }

        if parsed_calls.is_empty() && self.is_qwen3_next_coder_model() {
            parsed_calls = self.parse_qwen3_next_coder_calls(text);
        }

        if parsed_calls.is_empty()
            && self.config.start_token_str.starts_with('<')
            && self.config.end_token_str.starts_with('<')
        {
            let stripped = self.strip_tool_tags(text);
            let factory = ParserFactory::new();
            if let Some(json_parser) = factory.registry().create_parser("json") {
                if let Ok((_normal_text, calls)) = json_parser.parse_complete(&stripped).await {
                    parsed_calls = calls;
                }
            }
        }

        parsed_calls
            .into_iter()
            .map(crate::tools::tool_call_from_parser)
            .collect()
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

    fn parser_name_for_model(model_type: &ModelType, model_id: &str) -> &'static str {
        let model_lower = model_id.to_ascii_lowercase();
        match model_type {
            ModelType::LLaMa => "llama",
            ModelType::Mistral | ModelType::Mistral3VL => "mistral",
            ModelType::Qwen3
            | ModelType::Qwen3MoE
            | ModelType::Qwen3_5
            | ModelType::Qwen3_5MoE
            | ModelType::Qwen3VL => {
                if Self::uses_qwen_coder_parser(&model_lower) {
                    "qwen_coder"
                } else {
                    "qwen"
                }
            }
            ModelType::Gemma | ModelType::Gemma3 => "json",
            ModelType::Phi | ModelType::Phi4 => "qwen",
            ModelType::GLM4 | ModelType::GLM4MoE => "glm47_moe",
            ModelType::Yi | ModelType::StableLM => "qwen",
            ModelType::DeepSeek => "deepseek",
        }
    }

    fn uses_qwen_coder_parser(model_lower: &str) -> bool {
        Self::is_qwen2_5_coder_model(model_lower)
            || Self::is_qwen3_next_coder_model_name(model_lower)
    }

    fn is_qwen2_5_coder_model(model_lower: &str) -> bool {
        model_lower.contains("coder")
            && (model_lower.contains("qwen2.5") || model_lower.contains("qwen2_5"))
    }

    fn is_qwen3_next_coder_model_name(model_lower: &str) -> bool {
        model_lower.contains("coder")
            && model_lower.contains("next")
            && (model_lower.contains("qwen3") || model_lower.contains("qwen-3"))
    }

    fn is_qwen3_next_coder_model(&self) -> bool {
        Self::is_qwen3_next_coder_model_name(&self.model_id.to_ascii_lowercase())
    }

    fn parse_qwen3_next_coder_calls(&self, text: &str) -> Vec<tool_parser::types::ToolCall> {
        let mut parsed_calls = Vec::new();
        let mut tool_index = 0;

        let mut blocks = Vec::new();
        let mut cursor = 0;
        while let Some(start_rel) = text[cursor..].find("<tool_call>") {
            let start = cursor + start_rel + "<tool_call>".len();
            if let Some(end_rel) = text[start..].find("</tool_call>") {
                let end = start + end_rel;
                blocks.push(&text[start..end]);
                cursor = end + "</tool_call>".len();
            } else {
                blocks.push(&text[start..]);
                break;
            }
        }

        if blocks.is_empty() && text.contains("<function=") {
            blocks.push(text);
        }

        for block in blocks {
            let mut pos = 0;
            while let Some(func_rel) = block[pos..].find("<function=") {
                let name_start = pos + func_rel + "<function=".len();
                let rest = &block[name_start..];
                let Some(name_end_rel) = rest.find('>') else {
                    break;
                };
                let func_name = &rest[..name_end_rel];
                let body_start = name_start + name_end_rel + 1;
                let func_end = block[body_start..]
                    .find("</function>")
                    .map(|v| body_start + v)
                    .unwrap_or(block.len());
                let func_body = &block[body_start..func_end];

                let mut params = Map::new();
                let mut ppos = 0;
                while let Some(p_rel) = func_body[ppos..].find("<parameter=") {
                    let p_name_start = ppos + p_rel + "<parameter=".len();
                    let p_rest = &func_body[p_name_start..];
                    let Some(p_name_end_rel) = p_rest.find('>') else {
                        break;
                    };
                    let p_name = &p_rest[..p_name_end_rel];
                    let p_val_start = p_name_start + p_name_end_rel + 1;
                    let p_val_rest = &func_body[p_val_start..];

                    let end_parameter = p_val_rest.find("</parameter>");
                    let next_parameter = p_val_rest.find("<parameter=");

                    let (value_end, consumed) = match (end_parameter, next_parameter) {
                        (Some(a), Some(b)) if b < a => (b, b),
                        (Some(a), _) => (a, a + "</parameter>".len()),
                        (None, Some(b)) => (b, b),
                        (None, None) => (p_val_rest.len(), p_val_rest.len()),
                    };

                    let mut raw_value = p_val_rest[..value_end].to_string();
                    if let Some(stripped) = raw_value.strip_prefix('\n') {
                        raw_value = stripped.to_string();
                    }
                    if let Some(stripped) = raw_value.strip_suffix('\n') {
                        raw_value = stripped.to_string();
                    }
                    let converted = self.convert_qwen3_next_parameter_value(
                        &raw_value,
                        p_name,
                        &self.qwen3_next_tool_param_config(func_name),
                        func_name,
                    );
                    params.insert(p_name.to_string(), converted);
                    ppos = p_val_start + consumed;
                }

                parsed_calls.push(tool_parser::types::ToolCall {
                    tool_index,
                    name: func_name.to_string(),
                    parameters: Value::Object(params).to_string(),
                });
                tool_index += 1;
                pos = if func_end < block.len() {
                    func_end + "</function>".len()
                } else {
                    block.len()
                };
            }
        }

        parsed_calls
    }

    fn qwen3_next_tool_param_config(&self, func_name: &str) -> Map<String, Value> {
        for tool in &self.tools {
            if tool.tool_type == "function" && tool.function.name == func_name {
                if let Some(obj) = tool
                    .function
                    .parameters
                    .get("properties")
                    .and_then(Value::as_object)
                {
                    return obj.clone();
                }
                if let Some(obj) = tool.function.parameters.as_object() {
                    return obj.clone();
                }
                return Map::new();
            }
        }
        Map::new()
    }

    fn convert_qwen3_next_parameter_value(
        &self,
        raw: &str,
        param_name: &str,
        param_config: &Map<String, Value>,
        _func_name: &str,
    ) -> Value {
        if raw.eq_ignore_ascii_case("null") {
            return Value::Null;
        }
        let param_type = param_config
            .get(param_name)
            .and_then(Value::as_object)
            .and_then(|v| v.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("string")
            .to_ascii_lowercase();

        if ["string", "str", "text", "varchar", "char", "enum"].contains(&param_type.as_str()) {
            return Value::String(raw.to_string());
        }
        if param_type.starts_with("int")
            || param_type.starts_with("uint")
            || param_type.starts_with("long")
            || param_type.starts_with("short")
            || param_type.starts_with("unsigned")
        {
            if let Ok(v) = raw.parse::<i64>() {
                return Value::Number(v.into());
            }
            return Value::String(raw.to_string());
        }
        if param_type.starts_with("num") || param_type.starts_with("float") {
            if let Ok(v) = raw.parse::<f64>() {
                if let Some(n) = serde_json::Number::from_f64(v) {
                    return Value::Number(n);
                }
            }
            return Value::String(raw.to_string());
        }
        if ["boolean", "bool", "binary"].contains(&param_type.as_str()) {
            return Value::Bool(raw.eq_ignore_ascii_case("true"));
        }
        if param_type == "object"
            || param_type == "array"
            || param_type == "arr"
            || param_type.starts_with("dict")
            || param_type.starts_with("list")
        {
            if let Ok(v) = serde_json::from_str::<Value>(raw) {
                return v;
            }
        }
        if let Ok(v) = serde_json::from_str::<Value>(raw) {
            return v;
        }
        Value::String(raw.to_string())
    }

    fn strip_tool_tags(&self, text: &str) -> String {
        let mut output = text.to_string();
        if !self.config.start_token_str.is_empty() {
            output = output.replace(&self.config.start_token_str, "");
        }
        if !self.config.end_token_str.is_empty() {
            output = output.replace(&self.config.end_token_str, "");
        }
        output
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
                    tool_calls: Some(
                        tools
                            .into_iter()
                            .enumerate()
                            .map(|(i, tc)| crate::server::PublicToolCall {
                                index: Some(i),
                                id: tc.id,
                                type_: tc.tool_type,
                                function: tc.function,
                            })
                            .collect(),
                    ),
                },
                finish_reason: None,
                error: None,
            }],
            usage: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct StreamingToolCallState {
    name: Option<String>,
    arguments: String,
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

    #[tokio::test]
    async fn test_parser_normal_content() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );
        match parser.process_token(0, "Hello world").await {
            StreamResult::Content(s) => assert_eq!(s, "Hello world"),
            _ => panic!("Expected Content"),
        }
    }

    #[tokio::test]
    async fn test_parser_tool_call_detection() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        // Start tag triggers buffering
        match parser.process_token(151657, "<tool_call>").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on start tag"),
        }

        // Content is buffered
        match parser
            .process_token(0, r#"{"name": "test", "arguments": {}}"#)
            .await
        {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering"),
        }

        // End tag triggers parsing
        match parser.process_token(151658, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[tokio::test]
    async fn test_parser_partial_start_text_mode() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Phi,
            "phi".to_string(),
            ToolConfig::for_model_type(&ModelType::Phi),
            tools,
            None,
        );

        // Partial start tag splits across tokens
        match parser.process_token(0, "<tool_").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on partial start"),
        }
        match parser.process_token(0, "call>").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on completed start"),
        }
        match parser
            .process_token(0, r#"{"name": "test", "arguments": {}}"#)
            .await
        {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering"),
        }
        match parser.process_token(0, "</tool_call>").await {
            StreamResult::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "test");
            }
            _ => panic!("Expected ToolCalls"),
        }
    }

    #[tokio::test]
    async fn test_parser_token_id_strict_match() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        // Text match should not trigger when token IDs are available
        match parser.process_token(0, "<tool_call>").await {
            StreamResult::Content(text) => assert_eq!(text, "<tool_call>"),
            _ => panic!("Expected Content without token ID match"),
        }
    }

    #[test]
    fn test_qwen_coder_parser_only_for_qwen_2_5_coder() {
        assert_eq!(
            StreamToolParser::parser_name_for_model(
                &ModelType::Qwen3,
                "Qwen/Qwen2.5-Coder-32B-Instruct"
            ),
            "qwen_coder"
        );
        assert_eq!(
            StreamToolParser::parser_name_for_model(
                &ModelType::Qwen3_5MoE,
                "Qwen/Qwen3-Coder-Next-FP8"
            ),
            "qwen_coder"
        );
        assert_eq!(
            StreamToolParser::parser_name_for_model(
                &ModelType::Qwen3_5,
                "Qwen/Qwen3.5-Coder-32B-Instruct"
            ),
            "qwen"
        );
    }

    #[tokio::test]
    async fn test_qwen3_next_parser_fallback_function_parameter_format() {
        let tools = vec![crate::tools::function_tool("calc", "desc")
            .param("x", "integer", "x", true)
            .param("flag", "boolean", "flag", false)
            .build()];
        let parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3_5MoE,
            "Qwen/Qwen3-Coder-Next-FP8".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3_5MoE),
            tools,
            None,
        );

        let text = "<tool_call><function=calc><parameter=x>42</parameter><parameter=flag>true</parameter></function></tool_call>";
        let calls = parser.parse_complete_with_fallback(text).await;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc");
        assert_eq!(calls[0].function.arguments, r#"{"flag":true,"x":42}"#);
    }
}
