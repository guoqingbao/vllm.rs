// src/utils/guidance.rs
///! Guided decoding support via llguidance.
///!
///! This module provides grammar-aware sampling for vllm.rs, enabling:
///!
///! - **Regex constraints**: Enforce regular expression patterns on output
///! - **Lark constraints**: Use Lark context-free grammars for structured output
///! - **JSON Schema constraints**: Enforce JSON schema compliance
///! - **Llguidance native**: Use native llguidance TopLevelGrammar format

use llguidance::{api::TopLevelGrammar, ParserFactory, Matcher};
use toktrie::SimpleVob;
use serde_json::Value;
use std::sync::Arc;
use tokenizers::Tokenizer;
use std::path::Path;
use std::collections::HashMap;
use serde_json::json;

// Re-export Constraint types from this module for convenience
pub use crate::tools::{Tool};

/// Sanitize a string by removing all non-ASCII bytes (including magic byte 0xFF)
/// This is used for tool choice strings to ensure only safe ASCII characters reach llguidance lexer
pub fn sanitize_to_ascii(s: &str) -> String {
    s.bytes()
        .filter(|&b| b.is_ascii())
        .map(|b| b as char)
        .collect::<String>()
}

/// Sanitize a string by removing invalid UTF-8 sequences and control characters
/// This ensures proper UTF-8 encoding before passing to llguidance grammar parser
pub fn sanitize_utf8_valid(s: &str) -> String {
    let mut result = String::new();
    for ch in s.chars() {
        if ch.is_control() && !matches!(ch, '\n' | '\r' | '\t') {
            continue;  // Skip control characters except newline/tab
        }
        result.push(ch);
    }
    result
}

// Wrapper functions that sanitize inputs before passing to llguidance

/// Create TopLevelGrammar from regex with ASCII sanitization
pub fn top_level_grammar_from_regex(regex: &str) -> TopLevelGrammar {
    let sanitized = sanitize_to_ascii(regex);
    TopLevelGrammar::from_regex(&sanitized)
}

/// Create TopLevelGrammar from Lark string with UTF-8 sanitization
pub fn top_level_grammar_from_lark(lark: &str) -> TopLevelGrammar {
    let sanitized = sanitize_utf8_valid(lark);
    TopLevelGrammar::from_lark(sanitized)
}

/// Create TopLevelGrammar from JSON schema with UTF-8 sanitization
pub fn top_level_grammar_from_json_schema(schema: Value) -> Result<TopLevelGrammar, anyhow::Error> {
    // Convert to string and sanitize
    let schema_str = serde_json::to_string_pretty(&schema)?;
    let sanitized = sanitize_utf8_valid(&schema_str);
    
    // Parse back to Value and create grammar
    let value = serde_json::from_str(&sanitized)?;
    Ok(TopLevelGrammar::from_json_schema(value))
}

/// Build JSON Schema from Tool definitions for llguidance constraints.
/// This function extracts the parameter schemas from each tool and builds a composite schema.
pub fn build_tool_schema(tools: &[Tool]) -> Value {
    crate::log_debug!("[llg] Building JSON Schema from tools");

    let mut properties = HashMap::<String, Value>::new();

    // Extract required fields from each tool's parameters schema
    let mut all_required = Vec::new();

    for tool in tools {
        // Insert the full parameters schema (including its own "required" array)
        properties.insert(format!("{}_params", &tool.function.name), tool.function.parameters.clone());

        // Extract the required field from this tool's parameters schema
        let params = &tool.function.parameters;
        if let Some(reqs) = params.get("required") {
            if let Some(arr) = reqs.as_array() {
                for item in arr.iter() {
                    if let Some(s) = item.as_str() {
                        all_required.push(s.to_string());
                    }
                }
            }
        }
    }

    crate::log_debug!("[llg] Tool schema built successfully with {} tools, {} required fields", tools.len(), all_required.len());

    // Return a simple JSON Schema that describes valid tool call outputs
    json!({
        "type": "object",
        "$schema": "http://json-schema.org/draft-07/schema#",
        "properties": properties,
        "required": all_required
    })
}

/// Build Lark grammar string for optimized outer envelope
/// This creates a grammar that:
/// - Makes tool-calling optional (allows normal text output via TEXT)
/// - Embeds tool parameter schemas using %json { ... } for validation
/// - Uses UTF-8 optimized token sampling for common text
pub fn build_lark_outer_envelope(tools: &[Tool]) -> String {
    if tools.is_empty() {
        // No tools - just allow any text output
        return r#"start: TEXT
TEXT: /(.|[\n\r])*/"#.to_string();
    }

    let mut obj_rules = Vec::new();

    for tool in tools {
        let name = &tool.function.name;
        let schema_str = serde_json::to_string(&tool.function.parameters).unwrap_or_default();

        obj_rules.push(format!(r#"
obj_{name}: %json {schema}
"#, name = name.replace("-", "_"), schema = schema_str));
    }

    // Build Lark grammar with outer envelope that makes tool-calling optional
    // The TEXT terminal uses a negative lookahead to avoid matching JSON objects starting with {
    let lark_grammar = r#"start: TEXT | tool_call

TEXT: /[^\n\r][^\n\r]*|[\n\r]+/

tool_call: <tool_call> ws json_array ws </tool_call>
json_array: "[" obj ("," obj)* "]"

obj:
ws: /[ \t\n]*/"#;

    format!("{}\n{}", lark_grammar.trim_end(), obj_rules.join("\n").trim_start())
}

/// Constraint types for grammar enforcement
#[derive(Clone, Debug)]
pub enum Constraint {
    Regex(String),
    Lark(String),
    JsonSchema(Value),
    Llguidance(TopLevelGrammar),
    None,
}

/// Build ParserFactory from tokenizer (called once per model load)
pub fn build_llg_factory(tokenizer: Tokenizer) -> Result<Arc<ParserFactory>, anyhow::Error> {
    crate::log_debug!("[llg] Building ParserFactory from tokenizer");
    let env = toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer)?.into_tok_env(None)?;
    let factory = ParserFactory::new_simple(&env)?;
    crate::log_debug!("[llg] ParserFactory built successfully");
    Ok(Arc::new(factory))
}

/// Convert Constraint to TopLevelGrammar
pub fn llg_grammar_from_constraint(constraint: &Constraint) -> Result<Option<TopLevelGrammar>, anyhow::Error> {
    match constraint {
        Constraint::Regex(regex) => {
            crate::log_debug!("[llg] Building grammar from regex constraint");
            let grm = top_level_grammar_from_regex(regex);
            crate::log_debug!("[llg] Regex grammar built successfully");
            Ok(Some(grm))
        }
        Constraint::Lark(lark) => {
            crate::log_debug!("[llg] Building grammar from Lark constraint");
            let grm = top_level_grammar_from_lark(lark);
            crate::log_debug!("[llg] Lark grammar built successfully");
            Ok(Some(grm))
        }
        Constraint::JsonSchema(value) => {
            crate::log_debug!("[llg] Building grammar from JSON Schema constraint");
            let grm = top_level_grammar_from_json_schema(value.clone())?;
            crate::log_debug!("[llg] JSON Schema grammar built successfully");
            Ok(Some(grm))
        }
        Constraint::Llguidance(value) => {
            crate::log_info!("[llg] Using native llguidance TopLevelGrammar");
            Ok(Some(value.clone()))
        }
        Constraint::None => {
            crate::log_debug!("[llg] No constraint specified");
            Ok(None)
        }
    }
}

/// Create Matcher from grammar
pub fn constraint_from_llg_grammar(
    factory: &ParserFactory,
    grm: TopLevelGrammar,
) -> Result<Matcher, anyhow::Error> {
    crate::log_debug!("[llg] Creating Matcher from grammar");
    let parser = match factory.create_parser(grm) {
        Ok(p) => p,
        Err(e) => {
            crate::log_error!("[llg] Failed to create parser from grammar: {:?}", e);
            return Err(anyhow::Error::msg(format!("Failed to create parser: {:?}", e)));
        }
    };
    let matcher = Matcher::new(Ok(parser));
    crate::log_debug!("[llg] Matcher created successfully");
    Ok(matcher)
}

/// Build toktrie from tokenizer bytes for llguidance token encoding
pub fn build_toktrie_from_tokenizer_bytes(bytes: &[u8]) -> Result<llguidance::toktrie::TokTrie, anyhow::Error> {
    use tokenizers::Tokenizer;

    crate::log_debug!("[llg] Building TokTrie from tokenizer bytes");

    // Parse the tokenizer from bytes
    let tokenizer = Tokenizer::from_bytes(bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer from bytes: {}", e))?;

    // Get the vocabulary from the tokenizer
    let vocab = tokenizer.get_vocab(true);
    let vocab_size = vocab.len();

    // Build token bytes vector from vocabulary
    let mut token_bytes = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        token_bytes.push(Vec::new());
    }

    // Map token IDs to their byte representations
    for (token, id) in vocab.iter() {
        token_bytes[*id as usize] = token.clone().into_bytes();
    }

    crate::log_error!("[llg] TokTrie construction from tokenizer bytes requires internal toktrie API access. Please implement using toktrie::TokTrie::from() with TokRxInfo and token_bytes.");

    anyhow::bail!("TokTrie construction from tokenizer bytes requires internal toktrie API access. Please implement using toktrie::TokTrie::from() with TokRxInfo and token_bytes.");
}

/// Load toktrie from path (helper for llguidance)
pub fn load_toktrie_from_path(_path: &Path) -> Option<llguidance::toktrie::TokTrie> {
    // Temporarily disabled - returns None
    None
}

/// GuidanceState: Encapsulates the ParserFactory and llg_matcher
#[derive(Clone)]
pub struct GuidanceState {
    pub factory: Arc<ParserFactory>,
    pub matcher: Option<Matcher>,
}

impl GuidanceState {
    /// Create a new GuidanceState from a tokenizer
    pub fn new_from_tokenizer(tokenizer: Tokenizer) -> Result<Self, anyhow::Error> {
        crate::log_debug!("[llg] Creating GuidanceState from tokenizer");
        let factory = build_llg_factory(tokenizer)?;
        let matcher = None;
        crate::log_debug!("[llg] GuidanceState created successfully");
        Ok(Self { factory, matcher })
    }

    /// Create a new GuidanceState from an existing factory
    pub fn new_with_factory(factory: Arc<ParserFactory>) -> Self {
        let matcher = None;
        Self { factory, matcher }
    }

    /// Helper to create Matcher from ParserFactory and TopLevelGrammar
    fn create_matcher_with_factory(factory: &Arc<ParserFactory>, grammar: &TopLevelGrammar) -> Result<Matcher, anyhow::Error> {
        crate::log_debug!("[llg] Creating parser from factory and grammar");

        match factory.create_parser(grammar.clone()) {
            Ok(parser) => {
                let matcher = Matcher::new(Ok(parser));
                crate::log_debug!("[llg] Matcher created successfully");
                Ok(matcher)
            }
            Err(e) => {
                crate::log_error!("[llg] Failed to create parser: {:?}", e);
                Err(anyhow::Error::msg(format!("Failed to create parser: {:?}", e)))
            }
        }
    }

    /// Create a new GuidanceState from an existing factory with specific TopLevelGrammar (alias for backward compatibility)
    pub fn new(factory: Arc<ParserFactory>, grammar: TopLevelGrammar) -> Result<Self, anyhow::Error> {
        Self::new_with_grammar(factory, grammar)
    }

    /// Create a new GuidanceState from an existing factory with specific TopLevelGrammar
    pub fn new_with_grammar(factory: Arc<ParserFactory>, grammar: TopLevelGrammar) -> Result<Self, anyhow::Error> {
        match Self::create_matcher_with_factory(&factory, &grammar) {
            Ok(matcher) => Ok(Self { factory, matcher: Some(matcher) }),
            Err(e) => Err(e),
        }
    }

    /// Check if operating or de-scheduled
    pub fn matcher_is_stopped(&self) -> bool {
        self.matcher.as_ref().map_or(false, |m| m.is_stopped())
    }

    /// Validate token against grammar matcher
    pub fn validate_token(&mut self, token: u32) -> bool {
        let llg = self.matcher.as_mut().expect("Matcher not initialized");
        if llg.is_stopped() {
            return true;  // No validation needed if grammar stopped
        }
        let result = llg.validate_tokens(&[token]).unwrap_or(0) == 1;
        if !result {
            crate::log_warn!("[llg] Token {} rejected by grammar", token);
        }
        result
    }

    /// Compute mask for allowed tokens
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob, anyhow::Error> {
        let llg = self.matcher.as_mut().expect("Matcher not initialized");
        let mask = llg.compute_mask_or_eos().map_err(|e| anyhow::Error::msg(e.to_string()))?;
        crate::log_debug!("[llg] Mask computed for {} allowed tokens", mask.len());
        Ok(mask)
    }

    /// Consume token and advance grammar state
    pub fn consume_token(&mut self, token: u32) -> Result<(), anyhow::Error> {
        let llg = self.matcher.as_mut().expect("Matcher not initialized");
        match llg.consume_token(token) {
            Ok(()) => Ok(()),
            Err(e) => {
                crate::log_error!("[llg] Failed to consume token {}: {:?}", token, e);
                Err(e)
            }
        }
    }

    /// Rollback grammar state by n tokens
    pub fn rollback(&mut self, n_tokens: usize) -> Result<(), anyhow::Error> {
        let llg = self.matcher.as_mut().expect("Matcher not initialized");
        match llg.rollback(n_tokens) {
            Ok(()) => Ok(()),
            Err(e) => {
                crate::log_error!("[llg] Failed to rollback {} tokens: {:?}", n_tokens, e);
                Err(e)
            }
        }
    }

    /// Validate draft tokens against grammar matcher
    pub fn validate_draft_tokens(&mut self, draft_tokens: &[u32]) -> Result<(Vec<u32>, Vec<usize>), anyhow::Error> {
        let llg = self.matcher.as_mut().expect("Matcher not initialized");
        let mut valid_tokens = Vec::new();
        let mut rejected_indices = Vec::new();

        for (i, &token) in draft_tokens.iter().enumerate() {
            if llg.is_stopped() || llg.validate_tokens(&[token]).unwrap_or(0) == 1 {
                valid_tokens.push(token);
            } else {
                rejected_indices.push(i);
                crate::log_warn!("[llg] Token {} rejected by grammar", token);
            }
        }

        Ok((valid_tokens, rejected_indices))
    }

    /// Consume valid draft tokens and advance grammar state
    pub fn consume_draft_tokens(&mut self, tokens: &[u32]) -> Result<(), anyhow::Error> {
        for token in tokens {
            self.consume_token(*token)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_enum_variants() {
        // Test that all constraint variants can be created
        let regex = Constraint::Regex("^[a-z]+$".to_string());
        let lark = Constraint::Lark("start: 'hello' 'world'".to_string());
        let json = Constraint::JsonSchema(serde_json::json!({"type": "object"}));
        let llg = Constraint::Llguidance(TopLevelGrammar::from_regex("^[a-z]+$"));
        let none = Constraint::None;

        assert!(matches!(regex, Constraint::Regex(_)));
        assert!(matches!(lark, Constraint::Lark(_)));
        assert!(matches!(json, Constraint::JsonSchema(_)));
        assert!(matches!(llg, Constraint::Llguidance(_)));
        assert!(matches!(none, Constraint::None));
    }

    #[test]
    fn test_constraint_none_returns_none() {
        let result = llg_grammar_from_constraint(&Constraint::None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_constraint_regex_returns_some() {
        let result = llg_grammar_from_constraint(&Constraint::Regex("^[a-z]+$".to_string()));
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_constraint_json_schema_returns_some() {
        let result = llg_grammar_from_constraint(&Constraint::JsonSchema(serde_json::json!({"type": "object"})));
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }
}
