// src/server/parser.rs
//! Streaming tool call parser for detecting and buffering tool calls during streaming.
//! Handles model-specific tool call tokens and formats.

use crate::server::{ChatChoiceChunk, ChatCompletionChunk, Delta};
use crate::tools::{Tool, ToolCall};
use crate::utils::config::ModelType;
use serde_json::{Map, Value};
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

/// Result of finalizing a buffered tool call at end-of-stream.
#[derive(Debug, Clone)]
pub enum BufferedFinalizeResult {
    ToolCalls(Vec<ToolCall>),
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

    /// Resolve tool call start token IDs using tokenizer and the validated config.
    pub fn tool_call_start_ids(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        let mut tool_call_start_ids: Vec<u32> = Vec::new();

        let mut used_special = false;
        if self.has_start_tokens() {
            let mut use_special = true;
            if !self.start_token_str.is_empty() {
                if let Ok(encoded) = tokenizer.encode(self.start_token_str.as_str(), false) {
                    let ids = encoded.get_ids();
                    if ids.len() != 1 || !self.start_token_ids.contains(&ids[0]) {
                        use_special = false;
                    }
                } else {
                    use_special = false;
                }
            }
            if use_special {
                tool_call_start_ids.extend(self.start_token_ids.iter().copied());
                used_special = true;
            }
        }

        if !used_special
            && !self.start_token_str.is_empty()
            && self.start_token_str.starts_with('<')
        {
            // Only use text tags that look like explicit tool markers to avoid false positives.
            if let Ok(encoded) = tokenizer.encode(self.start_token_str.as_str(), false) {
                let ids = encoded.get_ids();
                if ids.len() == 1 {
                    tool_call_start_ids.push(ids[0]);
                }
            }
        }

        tool_call_start_ids
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
    // Set when incremental parsing found ToolCallItem(s) for the latest processed token.
    saw_buffer_parse_activity: bool,
    // Set when any parsing activity occurs during the current buffering window.
    buffer_had_parse_activity: bool,
    // Candidate end marker seen while buffering; used to avoid false end hits inside content.
    pending_end_marker_candidate: bool,
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
            saw_buffer_parse_activity: false,
            buffer_had_parse_activity: false,
            pending_end_marker_candidate: false,
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

    /// Returns true if text contains tool-structure markup that should not be
    /// emitted verbatim as normal assistant text.
    pub fn contains_tool_markup(&self, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }
        for marker in self.display_escape_markers() {
            if text.contains(&marker) || Self::contains_partial_marker_fragment(text, &marker) {
                return true;
            }
        }
        false
    }

    /// Escapes tool-structure markers in plain text so leaked tool payloads do
    /// not become executable-looking tags in later model turns.
    pub fn sanitize_tool_markup_for_display(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let mut out = text.to_string();
        let mut markers = self.display_escape_markers();
        markers.sort_by_key(|m| std::cmp::Reverse(m.len()));
        markers.dedup();

        for marker in markers {
            if marker.is_empty() {
                continue;
            }
            out = out.replace(&marker, &Self::escape_marker_for_display(&marker));
            out = Self::escape_partial_marker_fragments(&out, &marker);
        }
        out
    }

    /// Returns whether the latest processed token produced incremental tool-parse activity.
    /// The flag is reset after being read.
    pub fn take_buffer_parse_activity(&mut self) -> bool {
        std::mem::take(&mut self.saw_buffer_parse_activity)
    }

    /// Process a single incoming token.
    /// Returns StreamResult indicating what action to take.
    pub async fn process_token(&mut self, token_id: u32, token_text: &str) -> StreamResult {
        self.saw_buffer_parse_activity = false;
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

        match self.state.clone() {
            ParserState::Normal => {
                // Don't detect tool-call starts inside reasoning or code blocks.
                // Once buffering starts we must continue buffering even if arguments
                // contain code fences/backticks.
                if self.in_reasoning() || self.in_code_block {
                    return StreamResult::Content(token_text.to_string());
                }
                // Check for start trigger
                if self.is_start_token(token_id, token_text) {
                    self.state = ParserState::Buffering;
                    self.buffer.clear();
                    self.buffer.push_str(token_text);
                    self.streaming_calls.clear();
                    self.buffer_had_parse_activity = false;
                    self.pending_end_marker_candidate = false;
                    match self.parser.parse_incremental(token_text, &self.tools).await {
                        Ok(result) => {
                            if !result.calls.is_empty() {
                                self.saw_buffer_parse_activity = true;
                                self.buffer_had_parse_activity = true;
                            }
                            self.apply_streaming_result(&result);
                        }
                        Err(err) => {
                            crate::log_warn!(
                                "Incremental tool parse failed at start tag: {:?}",
                                err
                            );
                        }
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
                let nested_start_marker = !self.config.start_token_str.is_empty()
                    && token_text.contains(&self.config.start_token_str);
                if nested_start_marker {
                    crate::log_warn!(
                        "Ignoring nested tool-call start marker while buffering: {:?}",
                        token_text
                    );
                } else {
                    match self.parser.parse_incremental(token_text, &self.tools).await {
                        Ok(result) => {
                            if !result.calls.is_empty() {
                                self.saw_buffer_parse_activity = true;
                                self.buffer_had_parse_activity = true;
                                // crate::log_info!("Stream parsing: {:?}", result.calls);
                            }
                            self.apply_streaming_result(&result);
                        }
                        Err(err) => {
                            crate::log_warn!(
                                "Incremental tool parse failed while buffering: {:?}",
                                err
                            );
                        }
                    }
                }
                let end_reached = self.is_end_token(token_id, token_text)
                    || self.buffer_has_end_tag()
                    || self.maybe_complete_mistral_list();
                if !end_reached && self.pending_end_marker_candidate {
                    self.pending_end_marker_candidate = false;
                }
                if end_reached {
                    let strict_complete = self.has_strict_complete_tool_call().await;
                    if !strict_complete && !self.pending_end_marker_candidate {
                        self.pending_end_marker_candidate = true;
                        crate::log_warn!(
                            "Tool-call end marker seen before payload completion; waiting for confirmation"
                        );
                        return StreamResult::Buffering;
                    }
                    self.pending_end_marker_candidate = false;
                    crate::log_info!(
                        "Tool call buffering end, reached {} ({})",
                        token_text,
                        token_id
                    );

                    let had_partial_calls = !self.streaming_calls.is_empty();
                    let tool_calls = self.build_tool_calls_with_fallback().await;
                    let result = if tool_calls.is_empty() {
                        if had_partial_calls {
                            crate::log_warn!(
                                "End marker seen but tool call is still incomplete; continuing buffering"
                            );
                            StreamResult::Buffering
                        } else {
                            // False positive - flush buffered content as normal text.
                            crate::log_error!("Unable to parse tool call buffer: {}", self.buffer,);
                            StreamResult::FlushBuffer(self.buffer.clone())
                        }
                    } else {
                        StreamResult::ToolCalls(tool_calls)
                    };
                    if matches!(result, StreamResult::Buffering) {
                        return result;
                    }
                    self.parser.reset();
                    self.buffer.clear();
                    self.state = ParserState::Normal;
                    self.streaming_calls.clear();
                    self.buffer_had_parse_activity = false;
                    self.pending_end_marker_candidate = false;
                    return result;
                }

                StreamResult::Buffering
            }
        }
    }

    /// Finalize buffered tool-call state at EOS.
    /// Tries to build tool calls first; if unsuccessful, returns buffered text for flushing.
    pub async fn finalize_buffered_tool_calls(&mut self) -> Option<BufferedFinalizeResult> {
        if !matches!(self.state, ParserState::Buffering) {
            return None;
        }

        crate::log_warn!("Stream ended while buffering a tool call; attempting final parse");

        let buffered_text = self.buffer.clone();
        let strict_complete = self.has_strict_complete_tool_call().await;
        let tool_calls = self.build_tool_calls_with_fallback().await;
        let recoverable_incomplete = !strict_complete
            && !tool_calls.is_empty()
            && self.can_recover_incomplete_buffered_tool_calls()
            && !self.has_ambiguous_incomplete_end_marker();

        self.parser.reset();
        self.buffer.clear();
        self.state = ParserState::Normal;
        self.streaming_calls.clear();
        self.buffer_had_parse_activity = false;
        self.pending_end_marker_candidate = false;

        if tool_calls.is_empty() || (!strict_complete && !recoverable_incomplete) {
            crate::log_warn!("Buffered tool call could not be finalized; flushing buffered text");
            Some(BufferedFinalizeResult::FlushBuffer(buffered_text))
        } else {
            if recoverable_incomplete {
                crate::log_warn!(
                    "Recovered buffered tool call(s) from partial envelope using incremental parse state"
                );
            }
            crate::log_warn!(
                "Recovered {} tool call(s) from buffered state at stream end",
                tool_calls.len()
            );
            Some(BufferedFinalizeResult::ToolCalls(tool_calls))
        }
    }

    /// Drain the buffer and reset parser state.
    pub fn take_buffer(&mut self) -> String {
        self.state = ParserState::Normal;
        self.buffer_had_parse_activity = false;
        self.pending_end_marker_candidate = false;
        std::mem::take(&mut self.buffer)
    }

    /// Check if token/text matches start trigger
    fn is_start_token(&self, id: u32, _text: &str) -> bool {
        // Token ID match (if available)
        if self.config.has_start_tokens() {
            return self.config.start_token_ids.contains(&id);
        }

        // Text-only mode: detect on the current line, allowing split tags while
        // avoiding overly eager triggers like a lone "<".
        let current_line = self.accumulated_output.rsplit('\n').next().unwrap_or("");
        let candidate = current_line.trim_start_matches(|c| c == ' ' || c == '\t' || c == '\r');

        if candidate.starts_with(&self.config.start_token_str) {
            return true;
        }

        let min_prefix_len = Self::safe_partial_prefix_len(&self.config.start_token_str);
        !candidate.is_empty()
            && candidate.len() >= min_prefix_len
            && self.config.start_token_str.starts_with(candidate)
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
        if !items.is_empty() {
            self.buffer_had_parse_activity = true;
        }
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
            let args = self.finalize_streamed_arguments(&state.arguments);
            calls.push(crate::tools::new_tool_call(
                crate::tools::generate_tool_call_id(),
                name.clone(),
                args,
            ));
        }
        calls
    }

    async fn build_tool_calls_with_fallback(&mut self) -> Vec<ToolCall> {
        if let Some(unstreamed) = self.parser.get_unstreamed_tool_args() {
            self.apply_stream_items(&unstreamed);
        }
        self.recover_streaming_arguments_from_buffer();
        let streaming_calls = self.build_tool_calls_from_streaming();
        if streaming_calls.is_empty() {
            crate::log_info!("Fallback to non-stream parsing for buffer: {}", self.buffer);
            return self.parse_complete_with_fallback(&self.buffer).await;
        }

        streaming_calls
    }

    async fn has_strict_complete_tool_call(&self) -> bool {
        if !self.has_complete_tool_envelope() {
            return false;
        }
        if self.streaming_calls.iter().any(|call| call.name.is_none()) {
            return false;
        }
        if !self.streaming_calls.is_empty()
            && self
                .streaming_calls
                .iter()
                .all(|call| call.arguments.trim().is_empty())
        {
            return true;
        }
        if !self.streaming_calls.is_empty()
            && self
                .streaming_calls
                .iter()
                .all(|call| serde_json::from_str::<Value>(call.arguments.trim()).is_ok())
        {
            return true;
        }

        !self
            .parse_complete_with_fallback(&self.buffer)
            .await
            .is_empty()
    }

    fn can_recover_incomplete_buffered_tool_calls(&self) -> bool {
        if self.streaming_calls.is_empty() {
            return false;
        }

        if !self.buffer_had_parse_activity
            && self
                .streaming_calls
                .iter()
                .all(|call| call.arguments.trim().is_empty())
        {
            return false;
        }

        if self.streaming_calls.iter().any(|call| call.name.is_none()) {
            return false;
        }

        self.streaming_calls.iter().all(|call| {
            let args = call.arguments.trim();
            args.is_empty()
                || serde_json::from_str::<Value>(&self.finalize_streamed_arguments(args)).is_ok()
        })
    }

    fn has_ambiguous_incomplete_end_marker(&self) -> bool {
        if self.config.end_token_str.is_empty() || !self.config.end_token_str.starts_with('<') {
            return false;
        }

        self.buffer.contains(&self.config.end_token_str) && !self.has_complete_tool_envelope()
    }

    fn has_complete_tool_envelope(&self) -> bool {
        // Non-XML formats should not be gated by XML envelope checks.
        if !self.config.start_token_str.starts_with('<')
            || !self.config.end_token_str.starts_with('<')
        {
            return true;
        }

        let Some(start_idx) = self.buffer.find(&self.config.start_token_str) else {
            // If no explicit start marker is present, keep existing behavior.
            return true;
        };

        let section = &self.buffer[start_idx..];
        let Some(end_rel) = section.rfind(&self.config.end_token_str) else {
            return false;
        };
        let end_idx = start_idx + end_rel + self.config.end_token_str.len();
        if end_idx <= start_idx {
            return false;
        }

        let block = &self.buffer[start_idx..end_idx];
        let inner_start = start_idx + self.config.start_token_str.len();
        let inner_end = end_idx - self.config.end_token_str.len();
        if inner_end < inner_start {
            return false;
        }
        let inner = self.buffer[inner_start..inner_end].trim();

        // Qwen-coder XML style: <tool_call><function=...><parameter=...>...</parameter></function></tool_call>
        if block.contains("<function=") || block.contains("<parameter=") {
            let Some(function_start) = block.find("<function=") else {
                return false;
            };
            let function_section = &block[function_start..];
            // Use the last function closer inside the current tool-call block so
            // literal `</function>` text inside parameter content does not
            // truncate the structural envelope check.
            let Some(function_end_rel) = function_section.rfind("</function>") else {
                return false;
            };
            let function_end = function_start + function_end_rel + "</function>".len();
            let function_block = &block[function_start..function_end];

            // Validate parameter pairing in order and ignore unmatched closing tags.
            // This tolerates malformed tails like:
            //   </function>\n</parameter>\n</function>
            // which should not invalidate an otherwise complete function payload.
            if !Self::has_balanced_parameter_tags(function_block) {
                return false;
            }
            return true;
        }

        // Qwen JSON style: <tool_call>{"name":"...","arguments":{...}}</tool_call>
        // Accept only if the inner payload is complete JSON at this point.
        if inner.is_empty() {
            return false;
        }
        serde_json::from_str::<Value>(inner).is_ok()
    }

    fn has_balanced_parameter_tags(function_block: &str) -> bool {
        let mut idx = 0usize;
        let mut open_count = 0usize;
        const OPEN: &str = "<parameter=";
        const CLOSE: &str = "</parameter>";

        while idx < function_block.len() {
            let open_pos = function_block[idx..].find(OPEN).map(|p| idx + p);
            let close_pos = function_block[idx..].find(CLOSE).map(|p| idx + p);

            match (open_pos, close_pos) {
                (None, None) => break,
                (Some(op), None) => {
                    open_count += 1;
                    idx = op + OPEN.len();
                }
                (None, Some(cp)) => {
                    // Ignore unmatched closing parameter tags.
                    if open_count > 0 {
                        open_count -= 1;
                    }
                    idx = cp + CLOSE.len();
                }
                (Some(op), Some(cp)) => {
                    if op < cp {
                        open_count += 1;
                        idx = op + OPEN.len();
                    } else {
                        if open_count > 0 {
                            open_count -= 1;
                        }
                        idx = cp + CLOSE.len();
                    }
                }
            }
        }

        open_count == 0
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

        if parsed_calls.is_empty()
            && self.config.start_token_str.starts_with('<')
            && self.config.end_token_str.starts_with('<')
            && (text.contains(&self.config.start_token_str)
                || text.contains(&self.config.end_token_str))
        {
            let stripped = self.strip_tool_tags(text);
            let factory = ParserFactory::new();
            if let Some(json_parser) = factory.registry().create_parser("json") {
                if let Ok((_normal_text, calls)) = json_parser.parse_complete(&stripped).await {
                    parsed_calls = calls;
                }
            }
        }

        // Final fallback: only for JSON-native parsing strategy and only when the
        // completion itself looks like JSON. Avoid scanning arbitrary mixed text
        // to prevent false positives from example snippets.
        if parsed_calls.is_empty()
            && self.parse_strategy == "json"
            && self.parser.has_tool_markers(text)
        {
            let factory = ParserFactory::new();
            if let Some(json_parser) = factory.registry().create_parser("json") {
                if let Ok((normal_text, calls)) = json_parser.parse_complete(text).await {
                    if normal_text.trim().is_empty() {
                        parsed_calls = calls;
                    }
                }
            }
        }

        parsed_calls
            .into_iter()
            .map(crate::tools::tool_call_from_parser)
            .collect()
    }

    fn finalize_streamed_arguments(&self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return "{}".to_string();
        }
        if serde_json::from_str::<Value>(trimmed).is_ok() {
            return trimmed.to_string();
        }

        let repaired = repair_streamed_json_arguments(trimmed);
        if repaired != trimmed {
            crate::log_warn!("Applied structural JSON repair to streamed tool arguments");
        }
        repaired
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
                if model_lower.contains("coder") {
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

    fn safe_partial_prefix_len(start_tag: &str) -> usize {
        if let Some(idx) = start_tag.find('_') {
            // E.g. "<tool_call>" => require at least "<tool"
            return idx.max(2);
        }
        // Default minimum for tags without underscore.
        start_tag.find('>').map_or(6, |idx| idx).clamp(2, 6)
    }

    fn escape_marker_for_display(marker: &str) -> String {
        if let Some(rest) = marker.strip_prefix('<') {
            format!("<\u{200C}{}", rest)
        } else {
            format!("{}\u{200C}", marker)
        }
    }

    fn contains_partial_marker_fragment(text: &str, marker: &str) -> bool {
        if marker.is_empty()
            || !Self::should_escape_marker_for_display(marker)
            || !marker.starts_with('<')
        {
            return false;
        }

        let marker_len = marker.len();
        if marker_len < 4 {
            return false;
        }

        let min_prefix_len =
            Self::safe_partial_prefix_len(marker).min(marker_len.saturating_sub(1));
        if min_prefix_len >= marker_len {
            return false;
        }

        (min_prefix_len..marker_len).rev().any(|len| {
            let prefix = &marker[..len];
            text.contains(prefix)
        })
    }

    fn escape_partial_marker_fragments(text: &str, marker: &str) -> String {
        if marker.is_empty()
            || !Self::should_escape_marker_for_display(marker)
            || !marker.starts_with('<')
        {
            return text.to_string();
        }

        let marker_len = marker.len();
        if marker_len < 4 {
            return text.to_string();
        }

        let min_prefix_len =
            Self::safe_partial_prefix_len(marker).min(marker_len.saturating_sub(1));
        if min_prefix_len >= marker_len {
            return text.to_string();
        }

        let mut out = text.to_string();
        for len in (min_prefix_len..marker_len).rev() {
            let prefix = &marker[..len];
            out = out.replace(prefix, &Self::escape_marker_for_display(prefix));
        }
        out
    }

    fn should_escape_marker_for_display(marker: &str) -> bool {
        if marker.is_empty() || marker.len() < 3 {
            return false;
        }
        let Some(first) = marker.chars().next() else {
            return false;
        };
        matches!(first, '<' | '[' | '{' | '(') || marker.contains('|')
    }

    fn display_escape_markers(&self) -> Vec<String> {
        let mut markers = Vec::new();
        for marker in [&self.config.start_token_str, &self.config.end_token_str] {
            if Self::should_escape_marker_for_display(marker) {
                markers.push(marker.to_string());
            }
        }
        // XML-style nested tool markers commonly appear in qwen-coder payloads.
        if self.config.start_token_str.contains("tool_call")
            && self.config.end_token_str.contains("tool_call")
        {
            markers.extend(
                ["<function=", "</function>", "<parameter=", "</parameter>"]
                    .into_iter()
                    .map(|s| s.to_string()),
            );
        }
        markers
    }

    fn recover_streaming_arguments_from_buffer(&mut self) {
        if self.streaming_calls.is_empty() || !self.buffer.contains("<parameter=") {
            return;
        }

        for state in &mut self.streaming_calls {
            let Some(name) = state.name.as_deref() else {
                continue;
            };

            let recovered = Self::extract_xml_parameters_for_function(&self.buffer, name);
            if recovered.is_empty() {
                continue;
            }

            let mut args_obj = match serde_json::from_str::<Value>(state.arguments.trim()) {
                Ok(Value::Object(map)) => map,
                _ => Map::new(),
            };

            let mut merged_any = false;
            for (key, value) in recovered {
                if !args_obj.contains_key(&key) && !value.is_empty() {
                    args_obj.insert(key, Value::String(value));
                    merged_any = true;
                }
            }

            if merged_any {
                state.arguments = Value::Object(args_obj).to_string();
                crate::log_warn!("Recovered missing parameter(s) from buffered tool-call content");
            }
        }
    }

    fn extract_xml_parameters_for_function(
        buffer: &str,
        function_name: &str,
    ) -> std::collections::HashMap<String, String> {
        let mut recovered = std::collections::HashMap::new();
        let function_tag = format!("<function={}>", function_name);
        let alt_function_tag = format!("<function=\"{}\">", function_name);

        let Some(func_start) = buffer
            .rfind(&function_tag)
            .or_else(|| buffer.rfind(&alt_function_tag))
        else {
            return recovered;
        };

        let section = &buffer[func_start..];
        let mut cursor = 0usize;
        const PARAM_PREFIX: &str = "<parameter=";
        const PARAM_END: &str = "</parameter>";

        while let Some(rel) = section[cursor..].find(PARAM_PREFIX) {
            let tag_start = cursor + rel;
            let name_start = tag_start + PARAM_PREFIX.len();
            let Some(name_end_rel) = section[name_start..].find('>') else {
                break;
            };
            let name_end = name_start + name_end_rel;
            let parameter_name = section[name_start..name_end]
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
            if parameter_name.is_empty() {
                break;
            }

            let value_start = name_end + 1;
            if value_start > section.len() {
                break;
            }

            if let Some(value_end_rel) = section[value_start..].find(PARAM_END) {
                let value_end = value_start + value_end_rel;
                let value = section[value_start..value_end]
                    .trim_matches(|c| c == '\n' || c == '\r')
                    .to_string();
                recovered.insert(parameter_name, value);
                cursor = value_end + PARAM_END.len();
            } else {
                let value = section[value_start..]
                    .trim_matches(|c| c == '\n' || c == '\r')
                    .to_string();
                recovered.insert(parameter_name, value);
                break;
            }
        }

        recovered
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
                    role: None,
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
                    role: None,
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

fn repair_streamed_json_arguments(raw: &str) -> String {
    let mut repaired = raw.trim().to_string();
    if repaired.is_empty() {
        return "{}".to_string();
    }

    let mut in_string = false;
    let mut escaped = false;
    let mut stack: Vec<char> = Vec::new();

    for ch in repaired.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' | '[' => stack.push(ch),
            '}' => {
                if stack.last() == Some(&'{') {
                    stack.pop();
                }
            }
            ']' => {
                if stack.last() == Some(&'[') {
                    stack.pop();
                }
            }
            _ => {}
        }
    }

    if in_string {
        repaired.push('"');
    }

    while repaired
        .chars()
        .last()
        .is_some_and(|c| c.is_whitespace() || c == ',')
    {
        repaired.pop();
    }

    while let Some(open) = stack.pop() {
        repaired.push(match open {
            '{' => '}',
            '[' => ']',
            _ => continue,
        });
    }

    repaired
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

    #[tokio::test]
    async fn test_parser_keeps_buffering_when_args_include_code_fence() {
        let tools = vec![crate::tools::function_tool("test", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        match parser.process_token(151657, "<tool_call>").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering on start tag"),
        }

        // Code-fence-like content inside buffered arguments should not switch the
        // parser back to normal content mode.
        match parser.process_token(0, "\n```markdown\n").await {
            StreamResult::Buffering => {}
            _ => panic!("Expected Buffering while inside tool call arguments"),
        }
    }

    #[test]
    fn test_repair_streamed_json_arguments_balances_only_structural_tokens() {
        let raw = r#"{"file_path":"/tmp/a.rs","new_string":"fn a() { let x = vec![1,2,3]; }","replace_all":false"#;
        let repaired = repair_streamed_json_arguments(raw);
        assert_ne!(repaired, raw);
        let parsed: Value = serde_json::from_str(&repaired).expect("repaired JSON should parse");
        assert_eq!(parsed["file_path"], "/tmp/a.rs");
        assert_eq!(parsed["new_string"], "fn a() { let x = vec![1,2,3]; }");
        assert_eq!(parsed["replace_all"], false);
    }

    #[tokio::test]
    async fn test_finalize_buffered_tool_calls_recovers_calls_on_eos() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call><function=Write>".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.rs","content":"abc""#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "Write");
                let args = calls[0].function.arguments.as_ref().unwrap();
                let parsed: Value = serde_json::from_str(args).unwrap();
                assert_eq!(parsed["file_path"], "/tmp/a.rs");
                assert_eq!(parsed["content"], "abc");
            }
            other => panic!("Expected recovered tool calls, got {:?}", other),
        }

        assert!(matches!(parser.state, ParserState::Normal));
        assert!(parser.buffer.is_empty());
        assert!(parser.streaming_calls.is_empty());
    }

    #[tokio::test]
    async fn test_finalize_buffered_tool_calls_flushes_when_unrecoverable() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call><function=Write><parameter=content>".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: None,
            arguments: String::new(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::FlushBuffer(text)) => {
                assert_eq!(text, "<tool_call><function=Write><parameter=content>");
            }
            other => panic!("Expected FlushBuffer, got {:?}", other),
        }

        assert!(matches!(parser.state, ParserState::Normal));
        assert!(parser.buffer.is_empty());
        assert!(parser.streaming_calls.is_empty());
    }

    #[tokio::test]
    async fn test_fake_end_marker_inside_parameter_keeps_buffering() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/a.md\n</parameter>\n<parameter=content>\n- Qwen format (`<tool_call>..."
            .to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.md","content":"- Qwen format (`<tool_call>..."}"#
                .to_string(),
        }];

        let result = parser.process_token(151658, "</tool_call>").await;
        assert!(matches!(result, StreamResult::Buffering));
        assert!(matches!(parser.state, ParserState::Buffering));
        assert!(parser.pending_end_marker_candidate);
    }

    #[tokio::test]
    async fn test_finalize_rejects_incomplete_xml_envelope_even_if_args_parse() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/a.md\n</parameter>\n<parameter=content>\n- Qwen format (`<tool_call>...</tool_call>"
            .to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.md","content":"- Qwen format (`<tool_call>...</tool_call>"}"#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::FlushBuffer(text)) => {
                assert!(text.contains("<parameter=content>"));
                assert!(text.contains("</tool_call>"));
            }
            other => panic!("Expected FlushBuffer, got {:?}", other),
        }
    }

    #[test]
    fn test_envelope_accepts_stray_parameter_closer_after_function() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        parser.buffer = r#"<tool_call>
<function=edit>
<parameter=filePath>
/root/vllm.rs/src/models/qwen3_5_moe.rs
</parameter>
<parameter=newString>
abc
</parameter>
<parameter=oldString>
def
</parameter>
</function>

</parameter>
</function>
</tool_call>"#
            .to_string();

        assert!(parser.has_complete_tool_envelope());
    }

    #[test]
    fn test_envelope_rejects_unclosed_parameter() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        parser.buffer = r#"<tool_call>
<function=edit>
<parameter=filePath>
/root/vllm.rs/src/models/qwen3_5_moe.rs
</parameter>
<parameter=newString>
abc
</function>
</tool_call>"#
            .to_string();

        assert!(!parser.has_complete_tool_envelope());
    }

    #[tokio::test]
    async fn test_nested_start_marker_is_ignored_while_buffering() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Phi,
            "phi".to_string(),
            ToolConfig::for_model_type(&ModelType::Phi),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call><function=Write>".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: String::new(),
        }];

        let result = parser.process_token(0, "<tool_call>").await;
        assert!(matches!(result, StreamResult::Buffering));
        assert!(matches!(parser.state, ParserState::Buffering));
    }

    #[tokio::test]
    async fn test_false_end_marker_inside_arguments_requires_confirmation() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Phi,
            "phi".to_string(),
            ToolConfig::for_model_type(&ModelType::Phi),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call>".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.rs","content":"text with "#.to_string(),
        }];

        let first = parser.process_token(0, "</tool_call>").await;
        assert!(matches!(first, StreamResult::Buffering));
        assert!(matches!(parser.state, ParserState::Buffering));
        assert!(parser.pending_end_marker_candidate);
    }

    #[tokio::test]
    async fn test_finalize_recovers_unclosed_xml_parameter_content() {
        let tools = vec![crate::tools::function_tool("Write", "desc").build()];
        let mut parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        parser.state = ParserState::Buffering;
        parser.buffer = "<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/a.md\n</parameter>\n<parameter=content>\n# Title\n".to_string();
        parser.streaming_calls = vec![StreamingToolCallState {
            name: Some("Write".to_string()),
            arguments: r#"{"file_path":"/tmp/a.md"}"#.to_string(),
        }];

        let finalized = parser.finalize_buffered_tool_calls().await;
        match finalized {
            Some(BufferedFinalizeResult::ToolCalls(calls)) => {
                assert_eq!(calls.len(), 1);
                let args = calls[0].function.arguments.as_ref().unwrap();
                let parsed: Value = serde_json::from_str(args).unwrap();
                assert_eq!(parsed["file_path"], "/tmp/a.md");
                assert_eq!(parsed["content"], "# Title");
            }
            other => panic!("Expected recovered tool calls, got {:?}", other),
        }
    }

    #[test]
    fn test_sanitize_tool_markup_for_display_escapes_xml_tool_payload() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        let raw = "<tool_call><function=write><parameter=filePath>/tmp/a.md</parameter></function></tool_call>";
        assert!(parser.contains_tool_markup(raw));

        let safe = parser.sanitize_tool_markup_for_display(raw);
        assert!(safe.contains("<\u{200C}tool_call>"));
        assert!(safe.contains("<\u{200C}function=write>"));
        assert!(safe.contains("<\u{200C}parameter=filePath>"));
        assert!(!parser.contains_tool_markup(&safe));
    }

    #[test]
    fn test_sanitize_tool_markup_for_display_keeps_non_xml_models_simple() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ModelType::Mistral,
            "mistral".to_string(),
            ToolConfig::for_model_type(&ModelType::Mistral),
            tools,
            None,
        );

        let raw = "[TOOL_CALLS]";
        assert!(parser.contains_tool_markup(raw));
        let safe = parser.sanitize_tool_markup_for_display(raw);
        assert_eq!(safe, "[TOOL_CALLS]\u{200C}");
    }

    #[test]
    fn test_contains_tool_markup_detects_partial_xml_marker() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        assert!(parser.contains_tool_markup("example <tool_ca"));
    }

    #[test]
    fn test_sanitize_tool_markup_for_display_escapes_partial_xml_marker() {
        let tools = vec![crate::tools::function_tool("write", "desc").build()];
        let parser = StreamToolParser::new_with_config(
            &ModelType::Qwen3,
            "qwen3-coder".to_string(),
            ToolConfig::for_model_type(&ModelType::Qwen3),
            tools,
            None,
        );

        let raw = "example <tool_ca";
        let safe = parser.sanitize_tool_markup_for_display(raw);
        assert!(safe.contains("<\u{200C}tool_ca"));
        assert!(!parser.contains_tool_markup(&safe));
    }
}
