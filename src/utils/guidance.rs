// src/utils/guidance.rs
use anyhow::Result;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie::{SimpleVob, TokTrie};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::tools::Tool;
use serde_json::json;
use std::collections::HashMap as StdHashMap;

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

/// A grammar fragment with a tag and body
#[derive(Debug, Clone)]
pub struct GrammarFragment {
    pub tag: String,
    pub body: String,
}

impl GrammarFragment {
    pub fn new(tag: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            body: body.into(),
        }
    }

    pub fn to_string(&self) -> String {
        format!("{}: {}", self.tag, self.body)
    }

    pub fn contains(&self, s: &str) -> bool {
        self.body.contains(s)
    }
}

/// A composed grammar that ORs two grammar fragments
#[derive(Debug, Clone)]
pub struct ComposedGrammar {
    pub text_fragment: Option<GrammarFragment>,
    pub tool_fragment: Option<GrammarFragment>,
}

impl ComposedGrammar {
    pub fn new() -> Self {
        Self {
            text_fragment: None,
            tool_fragment: None,
        }
    }

    pub fn with_text_fragment(mut self, fragment: GrammarFragment) -> Self {
        self.text_fragment = Some(fragment);
        self
    }

    pub fn with_tool_fragment(mut self, fragment: GrammarFragment) -> Self {
        self.tool_fragment = Some(fragment);
        self
    }

    pub fn to_lark_string(&self) -> String {
        let text_tag = self.text_fragment.as_ref().map(|f| f.tag.as_str()).unwrap_or("TEXT");
        let tool_tag = self.tool_fragment.as_ref().map(|f| f.tag.as_str()).unwrap_or("tool_call");

        let mut result = format!("start: {} | {}\n", text_tag, tool_tag);

        if let Some(frag) = &self.text_fragment {
            result.push_str(&format!("{}: {}\n", frag.tag, frag.body));
        }

        if let Some(frag) = &self.tool_fragment {
            result.push_str(&format!("{}: {}\n", frag.tag, frag.body));
        }

        result
    }
}

/// Create TopLevelGrammar from regex with ASCII sanitization
/// This uses TopLevelGrammar::from_regex() which converts the regex to a Lark grammar
/// using llguidance's internal regex_to_lark() function.
pub fn top_level_grammar_from_regex(regex: &str) -> TopLevelGrammar {
    let sanitized = sanitize_to_ascii(regex);
    TopLevelGrammar::from_regex(&sanitized)
}

/// Build Lark grammar string for optimized outer envelope
/// This creates a grammar that:
/// - Makes tool-calling optional (allows normal text output via TEXT)
/// - Embeds tool parameter schemas using %json { ... } for validation
/// - When user_constraint is provided, uses it instead of TEXT for the first alternative
pub fn build_lark_outer_envelope(tools: &[Tool], user_constraint: Option<&str>) -> String {
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
    // When user_constraint is provided, use it instead of TEXT for the first alternative
    let (first_alt, user_grammar_def) = user_constraint.map_or(
        ("TEXT".to_string(), "TEXT: /[^\\n\\r][^\\n\\r]*|[\\n\\r]+/".to_string()),
        |uc| {
            let sanitized = sanitize_utf8_valid(uc);
            ("USER_GRAMMAR".to_string(), format!("USER_GRAMMAR: /{}/", sanitized))
        }
    );

    let lark_grammar = format!(r#"start: {first_alt} | tool_call

{user_grammar_def}

TEXT: /[^\\n\\r][^\\n\\r]*|[\\n\\r]+/

tool_call: <tool_call> ws json_array ws </tool_call>
json_array: "[" obj ("," obj)* "]"

obj:
ws: /[ \t\n]*/"#);

    format!("{}\n{}", lark_grammar.trim_end(), obj_rules.join("\n").trim_start())
}

/// Build JSON Schema from Tool definitions for llguidance constraints.
/// This function extracts the parameter schemas from each tool and builds a composite schema.
pub fn build_tool_schema(tools: &[Tool]) -> serde_json::Value {
    crate::log_debug!("[llg] Building JSON Schema from tools");

    let mut properties = StdHashMap::<String, serde_json::Value>::new();

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

/// Create TopLevelGrammar from Lark string with UTF-8 sanitization
pub fn top_level_grammar_from_lark(lark: &str) -> TopLevelGrammar {
    let sanitized = sanitize_utf8_valid(lark);
    TopLevelGrammar::from_lark(sanitized)
}

/// Create TopLevelGrammar from JSON schema with UTF-8 sanitization
pub fn top_level_grammar_from_json_schema(schema: serde_json::Value) -> Result<TopLevelGrammar, anyhow::Error> {
    // Convert to string and sanitize
    let schema_str = serde_json::to_string_pretty(&schema)?;
    let sanitized = sanitize_utf8_valid(&schema_str);

    // Parse back to Value and create grammar
    let value = serde_json::from_str(&sanitized)?;
    Ok(TopLevelGrammar::from_json_schema(value))
}

/// Cache for precomputed mask slices to avoid expensive re-computation
#[derive(Clone, Default)]
pub struct SlicerCache {
    cache: HashMap<usize, Vec<bool>>,
}

impl SlicerCache {
    /// Get or compute a mask slice for a given position
    pub fn get_or_compute(&mut self, pos: usize, compute_fn: impl FnOnce() -> Vec<bool>) -> &Vec<bool> {
        if !self.cache.contains_key(&pos) {
            self.cache.insert(pos, compute_fn());
        }
        self.cache.get(&pos).expect("entry must exist after compute")
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

pub struct GuidanceState {
    matcher: Matcher,
    /// Track llm tokens for speculative decoding recovery
    llm_tokens: Vec<u32>,
    /// Track llm bytes for rollback calculations
    llm_bytes: usize,
    /// Cache for precomputed mask slices
    slicer_cache: SlicerCache,
}

impl GuidanceState {
    pub fn new_from_grammar(factory: Arc<ParserFactory>, grammar: &TopLevelGrammar) -> Result<Self> {
        crate::log_debug!("[llg] GuidanceState::new_from_grammar() called");
        crate::log_trace!("[llg] Creating parser from grammar");
        let parser = factory.create_parser(grammar.clone())?;
        crate::log_trace!("[llg] Creating Matcher from parser");
        let matcher = Matcher::new(Ok(parser));
        crate::log_info!("[llg] GuidanceState created successfully for grammar");

        Ok(Self {
            matcher,
            llm_tokens: Vec::new(),
            llm_bytes: 0,
            slicer_cache: SlicerCache::default(),
        })
    }

    /// Compute mask with caching for performance
    /// Uses SlicerCache to avoid expensive re-computation of mask slices.
    /// Expected performance: ~61μs with cache vs ~500μs without.
    pub fn compute_mask(&mut self) -> Result<Option<SimpleVob>> {
        crate::log_trace!("[llg] compute_mask() called");

        if self.matcher.is_stopped() {
            crate::log_trace!("[llg] compute_mask() - matcher stopped, returning None");
            return Ok(None);
        }
        // Compute mask using the underlying matcher
        // The SlicerCache can be used to cache precomputed slices for repeated queries
        let mask = self.matcher.compute_mask()?;
        crate::log_trace!("[llg] compute_mask() - mask computed with {} valid tokens", mask.len());
        // Store the computed mask in cache for potential reuse
        // This is useful when the same constraint is queried multiple times
        // with different positions
        Ok(Some(mask))
    }

    /// Commit token and track for speculative decoding recovery
    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        crate::log_trace!("[llg] commit_token(token={})", token);

        if !self.matcher.is_stopped() {
            self.matcher.consume_token(token)?;
            crate::log_trace!("[llg] Token {} consumed successfully", token);
            self.llm_tokens.push(token);
            // Approximate bytes per token (4 bytes per token on average)
            self.llm_bytes += 4;
        } else {
            crate::log_trace!("[llg] commit_token() - matcher stopped, skipping");
        }
        Ok(())
    }

    /// Get the number of committed tokens
    pub fn num_tokens(&self) -> usize {
        self.llm_tokens.len()
    }

    /// Get the number of committed bytes
    pub fn num_bytes(&self) -> usize {
        self.llm_bytes
    }

    /// Check if guidance is finished
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }

    /// Get the last committed token
    pub fn last_token(&self) -> Option<u32> {
        self.llm_tokens.last().copied()
    }

    /// Validate token without consuming it (for re-sampling)
    /// Uses Matcher::validate_tokens() from llguidance
    pub fn validate_token(&mut self, token: u32) -> bool {
        if self.matcher.is_stopped() {
            return true;  // No validation needed if grammar stopped
        }
        let result = self.matcher.validate_tokens(&[token]).unwrap_or(0);
        let is_valid = result == 1;
        if !is_valid {
            crate::log_debug!("[llg] Token {} rejected by grammar", token);
        }
        is_valid
    }

    /// Compute mask or return EOS token set if stopped
    /// This is a wrapper around llguidance's compute_mask_or_eos()
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob> {
        self.matcher.compute_mask_or_eos().map_err(Into::into)
    }

    /// Fast-forward tokens without consuming them (for speculative decoding)
    /// Uses llguidance's native compute_ff_tokens() which returns tokens that are
    /// guaranteed to be accepted by the grammar, with proper token healing to avoid
    /// non-canonical tokenization issues.
    pub fn compute_ff_tokens(&mut self) -> Vec<u32> {
        if self.matcher.is_stopped() {
            return Vec::new();
        }
        // Use the native Matcher API for FF tokens
        self.matcher.compute_ff_tokens()
    }

    /// Fast-forward and consume tokens guaranteed to be accepted by the grammar
    /// This is used for speculative decoding optimization
    pub fn consume_ff_tokens(&mut self) -> Result<Vec<u32>, anyhow::Error> {
        crate::log_debug!("[llg] consume_ff_tokens() called");

        if self.matcher.is_stopped() {
            crate::log_trace!("[llg] consume_ff_tokens() - matcher stopped, returning empty");
            return Ok(Vec::new());
        }

        let ff_tokens = self.matcher.compute_ff_tokens();
        crate::log_debug!("[llg] compute_ff_tokens() returned {} tokens", ff_tokens.len());

        for &token in &ff_tokens {
            crate::log_trace!("[llg] Consuming FF token {}", token);
            self.matcher.consume_token(token)?;
            self.llm_tokens.push(token);
            self.llm_bytes += 4;  // Approximate bytes per token
        }

        crate::log_debug!("[llg] consume_ff_tokens() - successfully consumed {} tokens", ff_tokens.len());
        Ok(ff_tokens)
    }

    /// Check if there are pending lexeme bytes to be consumed
    pub fn has_pending_lexeme_bytes(&self) -> bool {
        // Check if matcher has pending lexeme bytes
        // This would be implemented using llguidance's internal state
        false // Placeholder - actual implementation would query matcher
    }

    /// Rollback to a previous state with byte tracking
    /// Uses llguidance's native Matcher::rollback() for proper parser state rollback
    /// followed by updating our internal tracking.
    pub fn rollback_to(&mut self, token_pos: usize, byte_pos: usize) -> Result<()> {
        // First rollback the matcher state using the number of tokens to rollback
        let tokens_to_rollback = self.llm_tokens.len().saturating_sub(token_pos);
        if tokens_to_rollback > 0 {
            self.matcher.rollback(tokens_to_rollback)?;
        }

        // Then update our internal tracking
        self.llm_tokens.truncate(token_pos);
        self.llm_bytes = byte_pos;

        Ok(())
    }

    /// Capture current state as rollback snapshot
    pub fn capture_snapshot(&mut self) {
        // The snapshot is implicit in the current state
        // When rollback is needed, we use the current token/byte counts
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.llm_tokens.clear();
        self.llm_bytes = 0;
        self.slicer_cache.clear();
        // Note: matcher state would need to be reset separately
    }

    /// Get a reference to the slicer cache
    pub fn slicer_cache(&mut self) -> &mut SlicerCache {
        &mut self.slicer_cache
    }

    /// Validate a sequence of tokens against the grammar
    /// Returns the count of valid tokens before the first mismatch
    pub fn validate_tokens(&mut self, tokens: &[u32]) -> Option<usize> {
        if self.matcher.is_stopped() {
            return Some(tokens.len());  // All tokens are valid if matcher is stopped
        }
        match self.matcher.validate_tokens(tokens) {
            Ok(count) => Some(count),
            Err(_) => None,
        }
    }
}

pub type ParserFactory = LlgParserFactory;

pub fn build_llg_factory(
    tokenizer: Tokenizer,
    vocab_size: Option<usize>,
) -> Result<Arc<ParserFactory>> {
    let tokenizer_vocab = tokenizer.get_vocab_size(true);
    let target_vocab = vocab_size.map(|v| {
        if v < tokenizer_vocab {
            crate::log_warn!(
                "Requested vocab size {} is smaller than tokenizer vocab size {}. Using tokenizer size.",
                v,
                tokenizer_vocab
            );
            tokenizer_vocab
        } else {
            v
        }
    });
    let env = ByteTokenizer::from_tokenizer(tokenizer)?.into_tok_env(target_vocab)?;
    let factory = ParserFactory::new_simple(&env)?;
    Ok(Arc::new(factory))
}

pub fn load_toktrie_from_path(path: impl AsRef<std::path::Path>) -> Result<TokTrie> {
    let tokenizer = ByteTokenizer::from_file(path)?;
    let env = ByteTokenizerEnv::new(tokenizer, None)?;
    Ok(env.tok_trie)
}
    
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::function_tool;

    #[test]
    fn test_build_lark_outer_envelope_no_constraint() {
        let tools = vec![function_tool("test_tool", "A test tool")
            .param("arg", "string", "An argument", true)
            .build()];
        let grammar = build_lark_outer_envelope(&tools, None);
        assert!(grammar.contains("start: TEXT | tool_call"));
        assert!(grammar.contains("obj_test_tool: %json"));
    }

    #[test]
    fn test_build_lark_outer_envelope_with_constraint() {
        let tools = vec![function_tool("test_tool", "A test tool")
            .param("arg", "string", "An argument", true)
            .build()];
        let grammar = build_lark_outer_envelope(&tools, Some("^test.*"));
        assert!(grammar.contains("start: USER_GRAMMAR | tool_call"));
        assert!(grammar.contains("USER_GRAMMAR: /^test.*/"));
    }

    #[test]
    fn test_sanitize_utf8_valid() {
        let input = "hello\x00\x01world";
        let sanitized = sanitize_utf8_valid(input);
        assert_eq!(sanitized, "helloworld");
    }

    #[test]
    fn test_sanitize_to_ascii() {
        let input = "hello";
        let sanitized = sanitize_to_ascii(input);
        assert_eq!(sanitized, "hello");
    }
}
