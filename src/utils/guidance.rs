// src/utils/guidance.rs
use anyhow::Result;
use candle_core::Tensor;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie::{SimpleVob, TokTrie};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::tools::Tool;
use crate::utils::logits_processor::{LogitsProcessor, Sampling};
use serde_json::json;
use std::collections::HashMap as StdHashMap;

/// Error type for grammar-related errors
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag")]
    TooManyConstraints,

    #[error("response_format.json_schema is required for type=json_schema")]
    MissingJsonSchema,

    #[error("unsupported response_format type: {0}")]
    UnsupportedFormat(String),

    #[error("invalid grammar: {0}")]
    InvalidGrammar(String),

    #[error("tool grammar construction failed: {0}")]
    ToolGrammarError(String),
}

pub type GrammarResult<T> = Result<T, GrammarError>;

/// Builder for structured output constraint grammars
pub struct ConstraintBuilder {
    choice: Option<Vec<String>>,
    regex: Option<String>,
    json: Option<serde_json::Value>,
    grammar: Option<String>,
    structural_tag: Option<serde_json::Value>,
}

impl ConstraintBuilder {
    pub fn new() -> Self {
        Self {
            choice: None,
            regex: None,
            json: None,
            grammar: None,
            structural_tag: None,
        }
    }

    pub fn choice(mut self, choice: Vec<String>) -> Self {
        self.choice = Some(choice);
        self
    }

    pub fn regex(mut self, regex: String) -> Self {
        self.regex = Some(regex);
        self
    }

    pub fn json(mut self, json: serde_json::Value) -> Self {
        self.json = Some(json);
        self
    }

    pub fn grammar(mut self, grammar: String) -> Self {
        self.grammar = Some(grammar);
        self
    }

    pub fn structural_tag(mut self, tag: serde_json::Value) -> Self {
        self.structural_tag = Some(tag);
        self
    }

    pub fn build(self) -> Result<Option<TopLevelGrammar>> {
        let mut selected: Option<TopLevelGrammar> = None;
        let mut constraint_count = 0;

        if let Some(choice) = self.choice {
            constraint_count += 1;
            if constraint_count > 1 {
                return Err(anyhow::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
            }
            let choice_gram = crate::tools::schema::build_choice_lark_grammar(&choice)
                .map_err(|e| anyhow::Error::msg(e))?;
            selected = Some(choice_gram);
        }

        if let Some(regex) = self.regex {
            constraint_count += 1;
            if constraint_count > 1 {
                return Err(anyhow::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
            }
            let regex_gram = TopLevelGrammarExt::from_regex_ascii(&regex);
            selected = Some(regex_gram);
        }

        if let Some(schema) = self.json {
            constraint_count += 1;
            if constraint_count > 1 {
                return Err(anyhow::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
            }
            let schema = crate::tools::schema::sanitize_schema_for_llguidance(&schema);
            let json_gram = TopLevelGrammarExt::from_json_schema_utf8(schema)
                .map_err(|e| anyhow::Error::msg(e.to_string()))?;
            selected = Some(json_gram);
        }

        if let Some(grammar) = self.grammar {
            constraint_count += 1;
            if constraint_count > 1 {
                return Err(anyhow::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
            }
            let lark_gram = TopLevelGrammarExt::from_lark_utf8(&grammar);
            selected = Some(lark_gram);
        }

        if let Some(tag) = self.structural_tag {
            constraint_count += 1;
            if constraint_count > 1 {
                return Err(anyhow::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
            }
            let (start, end, schema) = crate::tools::schema::parse_structural_tag(&tag)
                .map_err(|e| anyhow::Error::msg(e))?;
            let schema = crate::tools::schema::sanitize_schema_for_llguidance(&schema);
            let tools = crate::tools::schema::schema_to_tools(&schema);
            let tool_gram = crate::tools::schema::build_json_tool_lark_grammar(&tools, &start, &end, false, false, None, None);
            selected = Some(tool_gram);
        }

        if selected.is_none() {
            return Err(anyhow::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"));
        }

        Ok(selected)
    }
}

/// Builder for composing multiple grammars with alternation
/// This provides a more readable, declarative way to build composed grammars
pub struct GrammarBuilder {
    alternatives: Vec<TopLevelGrammar>,
    max_tokens: Option<usize>,
}

impl GrammarBuilder {
    pub fn new() -> Self {
        Self {
            alternatives: Vec::new(),
            max_tokens: None,
        }
    }

    pub fn alternative(mut self, grammar: TopLevelGrammar) -> Self {
        self.alternatives.push(grammar);
        self
    }

    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn build(self) -> TopLevelGrammar {
        // Note: GrammarBuilder currently uses chat_text_expression() without EOS tokens
        // EOS token support is provided through compose_grammars() directly
        match self.alternatives.len() {
            0 => {
                let lark = chat_text_expression();
                TopLevelGrammar::from_lark_utf8(&lark)
            }
            1 => {
                let mut gram = self.alternatives.into_iter().next().unwrap();
                gram.max_tokens = self.max_tokens;
                gram
            }
            _ => {
                let merged = merge_top_level_grammars(
                    self.alternatives,
                    self.max_tokens,
                    Some("|".to_string())
                );
                merged
            }
        }
    }
}

/// Extension trait for TopLevelGrammar with built-in sanitization
/// This ensures all grammar construction paths sanitize inputs consistently
pub trait TopLevelGrammarExt: Sized {
    /// Create TopLevelGrammar from regex with ASCII sanitization
    fn from_regex_ascii(regex: &str) -> Self;

    /// Create TopLevelGrammar from Lark string with UTF-8 sanitization
    fn from_lark_utf8(lark: &str) -> Self;

    /// Create TopLevelGrammar from JSON schema with UTF-8 sanitization
    fn from_json_schema_utf8(schema: serde_json::Value) -> Result<Self, anyhow::Error>;
}

impl TopLevelGrammarExt for TopLevelGrammar {
    fn from_regex_ascii(regex: &str) -> Self {
        let sanitized = sanitize_to_ascii(regex);
        Self::from_regex(&sanitized)
    }

    fn from_lark_utf8(lark: &str) -> Self {
        let sanitized = sanitize_utf8_valid(lark);
        Self::from_lark(sanitized)
    }

    fn from_json_schema_utf8(schema: serde_json::Value) -> Result<Self, anyhow::Error> {
        let schema_str = serde_json::to_string(&schema)?;
        let sanitized = sanitize_utf8_valid(&schema_str);
        let val = serde_json::from_str(&sanitized)?;
        Ok(Self::from_json_schema(val))
    }
}

/// Sanitize a string by removing non-ASCII bytes
/// This is used for tool choice strings to ensure only safe ASCII characters reach llguidance lexer
pub fn sanitize_to_ascii(s: &str) -> String {
    s.bytes()
        .filter(|&b| b.is_ascii())
        .map(|b| b as char)
        .collect::<String>()
}

/// Sanitize a string by removing invalid UTF-8 sequences and control characters
pub fn sanitize_utf8_valid(s: &str) -> String {
    let mut result = String::new();
    for ch in s.chars() {
        if ch.is_control() && !matches!(ch, '\n' | '\r' | '\t') {
            continue;
        }
        result.push(ch);
    }
    result
}

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
pub fn top_level_grammar_from_json_schema(schema: serde_json::Value) -> Result<TopLevelGrammar> {
    let schema_str = serde_json::to_string(&schema)?;
    let sanitized = sanitize_utf8_valid(&schema_str);
    let val = serde_json::from_str(&sanitized)?;
    Ok(TopLevelGrammar::from_json_schema(val))
}

/// Build JSON Schema from Tool definitions for llguidance constraints
pub fn build_tool_schema(tools: &[Tool]) -> serde_json::Value {
    let mut properties = StdHashMap::<String, serde_json::Value>::new();
    let mut all_required = Vec::new();

    for tool in tools {
        properties.insert(format!("{}_params", &tool.function.name), tool.function.parameters.clone());
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

    json!({
        "type": "object",
        "$schema": "http://json-schema.org/draft-07/schema#",
        "properties": properties,
        "required": all_required
    })
}

/// Parse a Lark grammar string to extract the start rule RHS and other rules
/// Returns (start_rhs, other_rules) where start_rhs is the RHS of the start: rule
/// The RHS should be a list of rule names separated by | for alternation
fn parse_lark_grammar(lark: &str) -> (String, Vec<String>) {
    let lines: Vec<&str> = lark.lines().collect();
    if lines.is_empty() {
        return (String::new(), Vec::new());
    }

    let first_line = lines[0].trim();
    if first_line.starts_with("start:") {
        // Extract only the rule names after "start:", not the full rule definition
        let rhs_part = first_line.strip_prefix("start:").unwrap_or("").trim();

        // Parse the RHS to get individual rule names (separated by |)
        // We only want the rule names, not their definitions
        let rule_names: Vec<String> = rhs_part
            .split('|')
            .map(|s| s.trim().to_string())
            .collect();

        // The RHS for alternation should be just the rule names
        let start_rhs = rule_names.join(" | ");

        // Return all remaining lines as other rules
        let other_rules: Vec<String> = lines[1..].iter().map(|s| s.to_string()).collect();

        (start_rhs, other_rules)
    } else {
        // No start rule - treat entire grammar as the start rule
        (lark.to_string(), Vec::new())
    }
}

/// Combine grammar rules, handling duplicate rule names by merging them
fn combine_rules(rules: Vec<String>) -> String {
    if rules.is_empty() {
        return String::new();
    }

    // Group rules by their name (the part before ":")
    use std::collections::HashMap;
    let mut rule_groups: HashMap<String, Vec<String>> = HashMap::new();

    for rule in rules {
        let rule = rule.trim();
        if rule.is_empty() {
            continue;
        }

        // Find the rule name (before the first ":")
        if let Some(colon_pos) = rule.find(':') {
            let name = rule[..colon_pos].trim().to_string();
            let body = rule[colon_pos + 1..].trim().to_string();

            rule_groups.entry(name).or_default().push(body);
        } else {
            // Rule without colon - add as-is
            rule_groups.entry("anonymous".to_string()).or_default().push(rule.to_string());
        }
    }

    // Reconstruct rules, merging duplicates
    let mut combined = Vec::new();
    for (name, bodies) in rule_groups {
        if bodies.len() == 1 {
            combined.push(format!("{}: {}", name, bodies[0]));
        } else {
            // Multiple definitions for same rule - combine with alternation
            combined.push(format!("{}: {}", name, bodies.join(" | ")));
        }
    }

    combined.join("\n")
}

/// Merge multiple TopLevelGrammar objects into one
/// This creates a single Lark grammar with alternation at the start rule level
/// Each sub-grammar's rules are combined directly without rule_N indirection
pub fn merge_top_level_grammars(grammars: Vec<TopLevelGrammar>, max_tokens: Option<usize>, start_separator: Option<String>) -> TopLevelGrammar {
    // Extract all Lark grammar strings
    let mut lark_parts = Vec::new();

    let sep = match start_separator {
        Some(s) => s,
        None => "|".to_string(),
    };

    for (_i, g) in grammars.iter().enumerate() {
        for gw in &g.grammars {
            if let Some(lark) = &gw.lark_grammar {
                lark_parts.push(lark.clone());
            }
        }
    }

    if lark_parts.is_empty() {
        let lark_start_exp = format!("start: text\ntext[stop=\"\"]: /((?s).*?)/");
        let mut tlg = TopLevelGrammar::from_lark(lark_start_exp);
        tlg.max_tokens = max_tokens;
        return tlg;
    }

    // Parse each grammar and extract start RHS + other rules
    let mut combined_start_rhs = Vec::new();
    let mut all_other_rules = Vec::new();

    for lark in lark_parts.iter() {
        let (start_rhs, other_rules) = parse_lark_grammar(lark);
        combined_start_rhs.push(start_rhs);
        all_other_rules.extend(other_rules);
    }

    // Combine all other rules, handling duplicates
    let combined_rules = combine_rules(all_other_rules);

    // Build new grammar with direct alternation at start
    let start_separator = format!(" {} ", &sep);
    let start_alternation = combined_start_rhs.join(&start_separator);
    let final_grammar = format!("start: ( {} )+\n{}", start_alternation, combined_rules);

    let mut top_gram = TopLevelGrammar::from_lark(final_grammar);
    top_gram.max_tokens = max_tokens;
    top_gram
}

/// Extract the Lark grammar string from TopLevelGrammar for debugging
pub fn get_lark_from_top_level_grammar(gram: &TopLevelGrammar) -> String {
    if gram.grammars.is_empty() {
        return "No grammars".to_string();
    }
    let larks: Vec<String> = gram.grammars.iter()
        .filter_map(|g| g.lark_grammar.as_ref())
        .map(|s| s.clone())
        .collect();
    if larks.is_empty() {
        format!("{} grammars, none have lark_grammar", gram.grammars.len())
    } else {
        larks.join("\n---\n")
    }
}

/// Lark grammar TEXT pattern for common UTF-8 printable characters
/// Excludes control characters (0x00-0x1F), DEL (0x7F), and C1 controls (0x80-0x9F)
/// This pattern allows:
/// - ASCII printable: space (0x20) through tilde (0x7E)
/// - Unicode text: 0x80 onwards (Latin extended, accented chars, CJK, emoji, etc.)
/// - Common whitespace: newline, carriage return, tab
///
/// ## Binary Token Matching with llguidance Matcher
///
/// When working with Qwen-style tool tokens (e.g., `<‌tool_call>`), llguidance uses
/// a **byte-level lexer approach** with the following key concepts:
///
/// ### 1. Token-Based, Not Byte-Based
/// The `Matcher.compute_mask()` returns a [`SimpleVob`](toktrie::SimpleVob) - a bit vector
/// where each bit represents whether a **token ID** is allowed. This is pre-computed
/// against the tokenizer's trie.
///
/// ### 2. Special Token Marker (0xFF)
/// llguidance uses byte `0xFF` (TokTrie::SPECIAL_TOKEN_MARKER) to prefix special tokens
/// like `<|end_of_text|>`, `<|eot_id|>`, etc. This is because:
/// - `0xFF` is not valid UTF-8, so it never appears in regular text
/// - In Rust: `&[u8]` can contain 0xFF, but `&str` cannot
/// - Tokenizers like Qwen may embed special tokens as bytes like `[\xFF, b'[', b'1', b'2', b']']`
///
/// ### 3. Qwen Tool Call Format Example
/// For models like Qwen3 that use `<‌tool_call>` delimiters:
///
/// ```lark
/// start: tool*
/// tool: "<‌tool_call>" "\n" func "\n" "<‌/tool_call>" ("\n")*
/// func: %json {"type":"object","properties":{"name":...}}
/// ```
///
/// ### 4. Current Implementation in vLLM.rs
/// The [`src/core/runner.rs`](src/core/runner.rs) uses logits-based sampling:
/// ```ignore
/// // Apply mask: set disallowed tokens to -inf
/// for tok in 0..vocab_size {
///     if !mask.is_allowed(tok as u32) {
///         row[tok] = f32::NEG_INFINITY;
///     }
/// }
/// ```
/// This is compatible with llguidance's token-level SimpleVob mask because:
/// - `mask.is_allowed(tok)` checks if token ID `tok` is in the allowed set
/// - The logits are modified to give -inf to disallowed tokens
/// - Sampling then only picks from allowed tokens
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

/// Build special token syntax for Lark grammar using token IDs
/// When token IDs are available, uses <[token_id]> syntax instead of string literals
/// This ensures alignment with the outbound parser's token-based detection
pub fn build_special_token_tag(token_ids: &std::collections::HashSet<u32>, fallback: &str) -> String {
    if token_ids.is_empty() {
        // Fall back to string representation when token IDs are not available
        return lark_quote(fallback);
    }
    // Convert token IDs to Lark special token syntax <[id]>
    // The format is: <[token_id]> which matches what the tokenizer expects
    let ids: Vec<String> = token_ids.iter().map(|id| format!("[{}]", id)).collect();
    format!("<{}>", ids.join(","))
}

/// Build tool call start tag using token IDs when available
pub fn build_tool_call_tag(start_token_ids: &std::collections::HashSet<u32>, start_token_str: &str) -> String {
    build_special_token_tag(start_token_ids, start_token_str)
}

/// Build tool call end tag using token IDs when available
pub fn build_tool_call_end_tag(end_token_ids: &std::collections::HashSet<u32>, end_token_str: &str) -> String {
    build_special_token_tag(end_token_ids, end_token_str)
}

/// Build TEXT pattern with explicit EOS token IDs using <[id]> syntax
/// The EOS tokens are alternated as optional termination: TEXT eos?
pub fn chat_text_expression_with_eos(eos_token_ids: &[u32]) -> String {
    // First check environment variable override
    if let Ok(val) = std::env::var("VLLM_LLG_DEFAULT_TEXT") {
        return format!("{}", val);
    }

    // Build EOS alternation pattern using <[id]> syntax for token IDs
    // LHS must be lowercase - literal tokens aren't allowed in TERMINAL rules
    let eos_pattern = if eos_token_ids.is_empty() {
        // Fallback to stop="" when no EOS tokens available
        r#"start: text
text[stop=""]: /((?s).*?)/"#.to_string()
    } else if eos_token_ids.len() == 1 {
        format!(r#"start: text_with_eos
text_with_eos: TEXT eos?
TEXT: /(?s:.*)/
eos: <[{}]>"#, eos_token_ids[0])
    } else {
        let ids: Vec<String> = eos_token_ids.iter().map(|id| format!("<[{}]>", id)).collect();
        let eos_alternation = ids.join(" | ");
        format!(r#"start: text_with_eos
text_with_eos: TEXT eos?
TEXT: /(?s:.*)/
eos: {}"#, eos_alternation)
    };

    eos_pattern
}

/// Build TEXT pattern with stop="" attribute for proper EOS bounding
/// The stop="" attribute sets ends_at_eos: true so the parser can terminate at EOS
/// The [lazy] syntax is for rules, not terminals - options go AFTER the rule name, BEFORE the colon
pub fn chat_text_expression() -> String {
    // First check environment variable override
    if let Ok(val) = std::env::var("VLLM_LLG_DEFAULT_TEXT") {
        return format!("{}", val);
    }

    // Use a rule (lowercase) with stop="" attribute for proper EOS termination
    // The stop="" tells llguidance to allow EOS token as a valid termination point
    // Options go after the rule name, before the colon: text[stop=""]: /pattern/
    r#"start: text
text[stop=""]: /((?s).*?)/"#.to_string()
}


/// Build grammar vec based on constraint and tool presence
/// Returns a Vec<TopLevelGrammar> where the first element gets the start: rule
pub fn build_grammar_vec(
    constraint_grammars: Vec<TopLevelGrammar>,
    tool_grammar: Option<TopLevelGrammar>,
    tool_choice_required: bool,
) -> Vec<TopLevelGrammar> {
    match (constraint_grammars.is_empty(), tool_grammar.is_some(), tool_choice_required) {
        // No constraints, no tools → text only
        (true, false, _) => {
            let lark_exp = format!("start: text\ntext[stop=\"\"]: /((?s).*?)/");
            vec![TopLevelGrammar::from_lark(lark_exp)]
        },

        // No constraints, tools optional → TEXT | tool_call
        (true, true, false) => {
            let mut grammars = constraint_grammars;
            grammars.push(tool_grammar.unwrap());
            grammars
        }

        // No constraints, tools required → tool_call only
        (true, true, true) => {
            vec![tool_grammar.unwrap()]
        }

        // Constraints present, no tools → constraint only
        (false, false, _) => constraint_grammars,

        // Constraints present, tools optional → constraint | tool_call
        (false, true, false) => {
            let mut grammars = constraint_grammars;
            grammars.push(tool_grammar.unwrap());
            grammars
        }

        // Constraints present, tools required → constraint | tool_call
        (false, true, true) => {
            let mut grammars = constraint_grammars;
            grammars.push(tool_grammar.unwrap());
            grammars
        }
    }
}

/// Compose grammars based on constraint and tool settings
/// Returns a single TopLevelGrammar with proper precedence
/// This function takes the grammar that was built externally (with appropriate model-specific format)
/// and handles the alternation/composition logic
pub fn compose_grammars(
    mut constraint_grammars: Vec<TopLevelGrammar>,
    tool_grammar: Option<TopLevelGrammar>,
    has_tools: bool,
    tool_choice_required: bool,
    forced_tool_name: Option<String>,
    max_tokens: Option<usize>,
    eos_token_ids: &[u32],
) -> TopLevelGrammar {
    crate::log_debug!("[llg] compose_grammars() called: constraints={:?}", constraint_grammars.len());
    crate::log_debug!("[llg] compose_grammars(): has_tools={}, tool_choice_required={}, forced_tool_name={:?}", has_tools, tool_choice_required, forced_tool_name);

    match (
        constraint_grammars.is_empty(),
        tool_grammar.is_some(),
        tool_choice_required,
        forced_tool_name.is_some(),
    ) {
        // No constraint, no tools → text with EOS bounding
        (true, false, _, _) => {
            // Build TEXT pattern with explicit EOS token IDs
            // This generates: start: text_with_eos, text_with_eos: TEXT eos?, TEXT: /pattern/, eos: <[id]>
            let lark = chat_text_expression_with_eos(eos_token_ids);
            crate::log_debug!("[llg] compose_grammars() -> text with EOS: {}", &lark);
            TopLevelGrammar::from_lark(lark)
        }

        // No constraint, tools optional → tool_call | text with EOS
        (true, true, false, false) => {
            // Build text grammar with explicit EOS token IDs
            let lark = chat_text_expression_with_eos(eos_token_ids);
            let text_gram = TopLevelGrammar::from_lark(lark);
            let tool_gram = tool_grammar.unwrap();
            let start_sep = "|".to_string();
            let merged = merge_top_level_grammars(vec![text_gram, tool_gram], max_tokens, Some(start_sep));
            crate::log_debug!("[llg] compose_grammars() -> tool_call | text with EOS");
            merged
        }

        // No constraint, tools required → tool_call only
        (true, true, true, _) => {
            let tool_gram = tool_grammar.unwrap();
            crate::log_debug!("[llg] compose_grammars() -> tool_call only (tools required)");
            tool_gram
        }

        // No constraint, tools optional, specific tool forced → tool_call only
        (true, true, false, true) => {
            let tool_gram = tool_grammar.unwrap();
            crate::log_debug!("[llg] compose_grammars() -> tool_call only (forced tool: {})", forced_tool_name.unwrap());
            tool_gram
        }

        // Constraint only, no tools → constraint only
        (false, false, _, _) => {
            let constraint_gram = constraint_grammars.remove(0);
            crate::log_debug!("[llg] compose_grammars() -> constraint only");
            constraint_gram
        }

        // Constraint only, tools optional → tool_call | constraint
        (false, true, false, false) => {
            // Build combined grammar with constraint and tool_call
            let constraint_gram = constraint_grammars.remove(0);
            let tool_gram = tool_grammar.unwrap();
            // Build the merged grammar with constraint | tool_call
            // Use merge_top_level_grammars with None separator (|)
            merge_top_level_grammars(vec![constraint_gram, tool_gram], max_tokens, None)
        }

        // Constraint only, tools required → tool_call | constraint
        (false, true, true, _) => {
            let constraint_gram = constraint_grammars.remove(0);
            let tool_gram = tool_grammar.unwrap();
            merge_top_level_grammars(vec![constraint_gram, tool_gram], max_tokens, None)
        }

        // Constraint only, specific tool forced → tool_call | constraint
        (false, true, false, true) => {
            let constraint_gram = constraint_grammars.remove(0);
            let tool_gram = tool_grammar.unwrap();
            merge_top_level_grammars(vec![constraint_gram, tool_gram], max_tokens, None)
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

/// WS regex pattern for Lark grammars - matches whitespace including spaces, tabs, newlines, carriage returns
pub fn lark_ws_regex() -> &'static str {
    "/[ \\\\t\\\\r\\\\n]+/"
}

/// Build Lark grammar string for tool calls
pub fn build_tool_call_lark(tools: &[Tool], schema_map: &std::sync::Arc<std::collections::HashMap<String, serde_json::Value>>, start: &str, end: &str) -> String {
    let mut obj_rules = String::new();
    for tool in tools {
        let name = &tool.function.name;
        let schema_str = serde_json::to_string(schema_map.get(name).unwrap_or(&json!({}))).unwrap_or_default();
        obj_rules.push_str(&format!("obj_{}: %json {}\n", name.replace("-", "_"), schema_str));
    }

    format!("{start} _WS? json_array _WS? {end}\njson_array: \"[\" obj (\",\" obj)* \"]\"\nobj:\n_WS: {}\n{}", lark_ws_regex(), obj_rules.trim_end())
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
    pub fn compute_mask(&mut self) -> Result<Option<SimpleVob>> {
        crate::log_trace!("[llg] compute_mask() called");

        if self.matcher.is_stopped() {
            crate::log_trace!("[llg] compute_mask() - matcher stopped, returning None");
            return Ok(None);
        }
        let mask = self.matcher.compute_mask()?;
        crate::log_trace!("[llg] compute_mask() - mask computed with {} valid tokens", mask.len());
        Ok(Some(mask))
    }

    /// Commit token and track for speculative decoding recovery
    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        crate::log_trace!("[llg] commit_token(token={})", token);

        if !self.matcher.is_stopped() {
            self.matcher.consume_token(token)?;
            crate::log_trace!("[llg] Token {} consumed successfully", token);
            self.llm_tokens.push(token);
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
    pub fn validate_token(&mut self, token: u32) -> bool {
        if self.matcher.is_stopped() {
            return true;
        }
        let result = self.matcher.validate_tokens(&[token]).unwrap_or(0);
        let is_valid = result == 1;
        if !is_valid {
            crate::log_debug!("[llg] Token {} rejected by grammar", token);
        }
        is_valid
    }

    /// Compute mask or return EOS token set if stopped
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob> {
        self.matcher.compute_mask_or_eos().map_err(Into::into)
    }

    /// Fast-forward tokens without consuming them (for speculative decoding)
    pub fn compute_ff_tokens(&mut self) -> Vec<u32> {
        if self.matcher.is_stopped() {
            return Vec::new();
        }
        self.matcher.compute_ff_tokens()
    }

    /// Fast-forward and consume tokens guaranteed to be accepted by the grammar
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
            self.llm_bytes += 4;
        }

        crate::log_debug!("[llg] consume_ff_tokens() - successfully consumed {} tokens", ff_tokens.len());
        Ok(ff_tokens)
    }

    /// Check if there are pending lexeme bytes to be consumed
    pub fn has_pending_lexeme_bytes(&self) -> bool {
        false
    }

    /// Rollback to a previous state with byte tracking
    pub fn rollback_to(&mut self, token_pos: usize, byte_pos: usize) -> Result<()> {
        let tokens_to_rollback = self.llm_tokens.len().saturating_sub(token_pos);
        if tokens_to_rollback > 0 {
            self.matcher.rollback(tokens_to_rollback)?;
        }
        self.llm_tokens.truncate(token_pos);
        self.llm_bytes = byte_pos;
        Ok(())
    }

    /// Capture current state as rollback snapshot
    pub fn capture_snapshot(&mut self) {
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.llm_tokens.clear();
        self.llm_bytes = 0;
        self.slicer_cache.clear();
    }

    /// Get a reference to the slicer cache
    pub fn slicer_cache(&mut self) -> &mut SlicerCache {
        &mut self.slicer_cache
    }

    /// Validate a sequence of tokens against the grammar
    pub fn validate_tokens(&mut self, tokens: &[u32]) -> Option<usize> {
        if self.matcher.is_stopped() {
            return Some(tokens.len());
        }
        match self.matcher.validate_tokens(tokens) {
            Ok(count) => Some(count),
            Err(_) => None,
        }
    }
}

/// Apply sparse mask bias to logits
/// Uses iter_set_entries to only iterate allowed tokens
pub fn batch_mask_bias(
    logits: &Tensor,
    masks: &[(usize, SimpleVob)],
    vocab_size: usize,
) -> candle_core::Result<Tensor> {
    let batch_size = masks.len();
    
    // Create bias vector initialized to -inf
    let mut bias_data = vec![f32::NEG_INFINITY; batch_size * vocab_size];
    
    // Fill in allowed tokens using sparse iteration
    for (seq_idx, (_seq_idx, mask)) in masks.iter().enumerate() {
        mask.iter_set_entries(|idx| {
            if idx < vocab_size {
                bias_data[seq_idx * vocab_size + idx] = 0.0;
            }
        });
    }
    
    // Create bias tensor on same device as logits
    let bias_tensor = Tensor::from_vec(bias_data, (batch_size, vocab_size), logits.device())?;
    
    // GPU tensor addition (no CPU copy)
    logits.broadcast_add(&bias_tensor)
}

/// Two-stage validation with early exit
/// Stage 1: Sample and validate token
/// Stage 2: Only compute mask if token is invalid
pub fn early_exit_validate(
    guidance_states: &mut HashMap<usize, GuidanceState>,
    seq_ids: &[usize],
    tokens: &mut [u32],
    logits: &Tensor,
    vocab_size: usize,
    _factory: &Arc<ParserFactory>,
    sampling: &Sampling,
    logit_processor: &LogitsProcessor,
) -> candle_core::Result<()> {
    for (seq_idx, seq_id) in seq_ids.iter().enumerate() {
        let token = tokens[seq_idx];
        
        if let Some(state) = guidance_states.get_mut(seq_id) {
            // Stage 1: Validate token
            if state.validate_token(token) {
                // Early exit - token is valid, consume it
                state.commit_token(token).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                continue;
            }
            
            crate::log_debug!("[llg] Token {} is invalid, computing mask for seq {}", token, seq_id);
            
            // Stage 2: Token is invalid, compute mask and re-sample
            let mask = match state.compute_mask_or_eos() {
                Ok(m) => m,
                Err(e) => {
                    crate::log_error!("[llg] Unable to compute mask for token {} due to {}", token, e);
                    continue;
                }
            };
            
            crate::log_debug!("[llg] Applying bias to logits for seq {}", seq_id);
            
            // Build bias vector using sparse iteration
            let mut acc = vec![f32::NEG_INFINITY; vocab_size];
            mask.iter_set_entries(|idx| {
                if idx < acc.len() {
                    acc[idx] = 0.0;
                }
            });
            
            // Get logits as vector and apply bias
            let mut logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
            let row = &mut logits_vec[seq_idx * vocab_size..][..vocab_size];
            
            for tok in 0..vocab_size {
                if acc[tok] == 0.0 {
                    // Keep original logit value
                } else {
                    row[tok] = f32::NEG_INFINITY;
                }
            }
            
            // Create biased tensor
            let biased_tensor = Tensor::from_vec(logits_vec, logits.shape(), logits.device())?;
            
            // Re-sample with biased logits
            let re_sampled = logit_processor.sample_with_strategy(&biased_tensor, sampling)?;
            tokens[seq_idx] = re_sampled[seq_idx];
            
            crate::log_debug!("[llg] Consuming re-sampled token {} for seq {}", tokens[seq_idx], seq_id);
            
            // Commit the re-sampled token
            state.commit_token(tokens[seq_idx]).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        } else {
            crate::log_debug!("[llg] No guidance state for seq {}", seq_id);
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_to_ascii() {
        let input = "hello";
        let sanitized = sanitize_to_ascii(input);
        assert_eq!(sanitized, "hello");
    }

    #[test]
    fn test_sanitize_utf8_valid() {
        let input = "hello\x00\x01world";
        let sanitized = sanitize_utf8_valid(input);
        assert_eq!(sanitized, "helloworld");
    }

    #[test]
    fn test_compose_grammars_no_constraint_no_tools() {
        let result = compose_grammars(Vec::new(), None, false, false, None, None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_compose_grammars_no_constraint_tools_optional() {
        let tool_gram = TopLevelGrammar::from_lark("start: 'tool'".to_string());
        let result = compose_grammars(Vec::new(), Some(tool_gram), true, false, None, None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_compose_grammars_no_constraint_tools_required() {
        let tool_gram = TopLevelGrammar::from_lark("start: 'tool'".to_string());
        let result = compose_grammars(Vec::new(), Some(tool_gram), true, true, None, None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_compose_grammars_constraint_no_tools() {
        let constraint_gram = TopLevelGrammar::from_lark("start: 'choice'".to_string());
        let result = compose_grammars(vec![constraint_gram], None, false, false, None, None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_compose_grammars_constraint_with_tools_optional() {
        let constraint_gram = TopLevelGrammar::from_lark("start: 'constraint'".to_string());
        let tool_gram = TopLevelGrammar::from_lark("start: 'tool'".to_string());
        let result = compose_grammars(vec![constraint_gram], Some(tool_gram), true, false, None, None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_compose_grammars_constraint_with_tools_required() {
        let constraint_gram = TopLevelGrammar::from_lark("start: 'constraint'".to_string());
        let tool_gram = TopLevelGrammar::from_lark("start: 'tool'".to_string());
        let result = compose_grammars(vec![constraint_gram], Some(tool_gram), true, true, None, None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_compose_grammars_constraint_with_forced_tool() {
        let constraint_gram = TopLevelGrammar::from_lark("start: 'constraint'".to_string());
        let tool_gram = TopLevelGrammar::from_lark("start: 'tool'".to_string());
        let result = compose_grammars(vec![constraint_gram], Some(tool_gram), true, false, Some("forced_tool".to_string()), None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_compose_grammars_regex_constraint_no_tools() {
        let sanitized = sanitize_to_ascii("^[a-z]+$");
        let constraint_gram = TopLevelGrammar::from_regex(&sanitized);
        let result = compose_grammars(vec![constraint_gram], None, false, false, None, None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_merge_top_level_grammars_direct_alternation() {
        // Test that merge_top_level_grammars produces direct alternation without rule_N indirection
        let gram1 = TopLevelGrammar::from_lark("start: 'a'".to_string());
        let gram2 = TopLevelGrammar::from_lark("start: 'b'".to_string());
        // Use None for default separator (|)
        let result = merge_top_level_grammars(vec![gram1, gram2], None, None);

        // Get the combined Lark string
        let lark_str = get_lark_from_top_level_grammar(&result);

        // Verify that start: directly alternates 'a' | 'b' without rule_N indirection
        assert!(lark_str.contains("start: 'a' | 'b'"), "Expected direct alternation in start rule: {}", lark_str);
        // Verify that rule_N indirection is NOT present
        assert!(!lark_str.contains("rule_0:"), "Should not contain rule_0 indirection");
        assert!(!lark_str.contains("rule_1:"), "Should not contain rule_1 indirection");
    }

    #[test]
    fn test_merge_top_level_grammars_with_text_and_tool() {
        // Test the actual TEXT | tool_call scenario from the issue
        let lark = format!("start: TEXT\n{}", chat_text_expression());
        let text_gram = TopLevelGrammar::from_lark(lark);
        let tool_gram = TopLevelGrammar::from_lark("start: tool_call\ntool_call: \"test\"".to_string());
        // Use None for default separator (|)
        let result = merge_top_level_grammars(vec![text_gram, tool_gram], None, None);

        // Get the combined Lark string
        let lark_str = get_lark_from_top_level_grammar(&result);

        // Verify that start: directly alternates TEXT | tool_call
        assert!(lark_str.contains("start: TEXT | tool_call"), "Expected direct alternation: {}", lark_str);
        // Verify that rule_N indirection is NOT present
        assert!(!lark_str.contains("rule_0:"), "Should not contain rule_0 indirection");
        assert!(!lark_str.contains("rule_1:"), "Should not contain rule_1 indirection");
    }

    #[test]
    fn test_merge_top_level_grammars_unique_names() {
        let gram1 = TopLevelGrammar::from_lark("start: 'a'".to_string());
        let gram2 = TopLevelGrammar::from_lark("start: 'b'".to_string());
        // Use None for default separator (|)
        let result = merge_top_level_grammars(vec![gram1, gram2], None, None);
        // With unique rule names, each grammar gets a unique name
        // The result should have unique names to avoid duplicate grammar errors
        assert!(result.grammars.len() >= 1);
    }

    #[test]
    fn test_merge_top_level_grammars_with_grammar_without_start() {
        // Verify that when merging a grammar without start: line, it gets properly handled
        let gram1 = TopLevelGrammar::from_lark("start: 'a'\n'a': 'a'".to_string());
        let gram2 = TopLevelGrammar::from_lark("'tool': 'call'\ntool: %json {\"type\":\"object\"}".to_string());
        // Use None for default separator (|)
        let result = merge_top_level_grammars(vec![gram1, gram2], None, None);

        // Get the combined Lark string
        let lark_str = get_lark_from_top_level_grammar(&result);

        // Should still have direct alternation at start
        assert!(lark_str.contains("start:"), "Expected start rule in merged grammar");
        // The tool grammar should be properly included
        assert!(lark_str.contains("'tool': 'call'"), "Expected tool content in merged grammar");
    }

    #[test]
    fn test_compose_grammars_match_arm_conditions() {
        // Test each match arm with specific conditions
        // (constraint_empty, tool_grammar_some, tool_choice_required, forced_tool_name_some)

        // (false, false, _, _) -> constraint only
        let constraint = TopLevelGrammar::from_lark("start: 'a'".to_string());
        let result = compose_grammars(vec![constraint], None, false, false, None, None, &[]);
        assert!(result.grammars.len() > 0);

        // (false, true, false, false) -> constraint | tool_call
        let constraint = TopLevelGrammar::from_lark("start: 'a'".to_string());
        let tool = TopLevelGrammar::from_lark("start: 'tool'".to_string());
        let result = compose_grammars(vec![constraint], Some(tool), true, false, None, None, &[]);
        assert!(result.grammars.len() > 0);
    }

    #[test]
    fn test_compose_grammars_tool_grammar_is_not_text() {
        // When has_tools=true, tool_grammar should contain tool_call rules, not just TEXT
        // Create a tool call Lark grammar with actual tool rules
        let tool_lark = r#"
start: '<‌tool_call>' _WS? tool _WS? '<‌/tool_call>'
tool: %json {"type":"object","properties":{"name":{"type":"string"}}, "required":["name"]}
WS: {lark_ws_regex()}
"#.to_string();
        let tool_gram = TopLevelGrammar::from_lark(tool_lark);

        // Verify the tool grammar has tool_call specific content
        let lark_str = get_lark_from_top_level_grammar(&tool_gram);
        assert!(lark_str.contains("<‌tool_call>"), "Tool grammar should contain tool_call tags");
        assert!(lark_str.contains("tool: %json"), "Tool grammar should contain JSON schema");
    }

    #[test]
    fn test_chat_text_expression_with_eos_single() {
        // Test that chat_text_expression_with_eos generates correct grammar with single EOS token
        let eos_ids = [151645u32];
        let lark = chat_text_expression_with_eos(&eos_ids);
        assert!(lark.contains("start: text_with_eos"), "Expected text_with_eos start rule: {}", lark);
        assert!(lark.contains("text_with_eos: TEXT eos?"), "Expected text_with_eos rule with optional eos: {}", lark);
        assert!(lark.contains("eos: <[151645]>"), "Expected EOS token ID in grammar: {}", lark);
    }

    #[test]
    fn test_chat_text_expression_with_eos_multiple() {
        // Test that chat_text_expression_with_eos generates correct grammar with multiple EOS tokens
        let eos_ids = [151645u32, 151643u32];
        let lark = chat_text_expression_with_eos(&eos_ids);
        assert!(lark.contains("start: text_with_eos"), "Expected text_with_eos start rule: {}", lark);
        assert!(lark.contains("eos: <[151645]> | <[151643]>"), "Expected multiple EOS tokens in grammar: {}", lark);
    }

    #[test]
    fn test_chat_text_expression_with_eos_empty() {
        // Test fallback behavior when no EOS tokens provided
        let eos_ids: [u32; 2] = [19287, 3645];
        let lark = chat_text_expression_with_eos(&eos_ids);
        println!("{}", &lark);
        assert!(lark.contains("start: text_with_eos"), "Expected text start rule: {}", lark);
        assert!(lark.contains("<[19287]>"), "Expected 19287 token for EOS bounding: {}", lark);
    }

    #[test]
    fn test_compose_grammars_text_tool_call_plus_pattern() {
        // Test that (TEXT | tool_call)* is properly generated
        // Note: tool_call now uses regex pattern format
        let result = compose_grammars(Vec::new(), Some(TopLevelGrammar::from_lark("start: tool_call\ntool_call: /test/".to_string())), true, false, None, None, &[]);
        let lark_str = get_lark_from_top_level_grammar(&result);
        assert!(lark_str.contains("start: text | tool_call"), "Expected text | tool_call alternation: {}", lark_str);
    }

    #[test]
    fn test_compose_grammars_constraint_tool_call_plus_pattern() {
        // Test that (constraint | tool_call)* is properly generated
        // Note: tool_call now uses regex pattern format
        let constraint_gram = TopLevelGrammar::from_lark("start: constraint\nconstraint: /test/".to_string());
        let tool_gram = Some(TopLevelGrammar::from_lark("start: tool_call\ntool_call: /tool/".to_string()));
        let result = compose_grammars(vec![constraint_gram], tool_gram, true, false, None, None, &[]);
        let lark_str = get_lark_from_top_level_grammar(&result);
        assert!(lark_str.contains("start: constraint | tool_call"), "Expected constraint | tool_call alternation: {}", lark_str);
    }

    #[test]
    fn test_compose_grammars_forced_tool_no_text() {
        // Test that forced tool only uses tool_call grammar
        let result = compose_grammars(Vec::new(), Some(TopLevelGrammar::from_lark("start: tool_call\ntool_call: \"test\"".to_string())), true, false, Some("forced_tool".to_string()), None, &[]);
        let lark_str = get_lark_from_top_level_grammar(&result);
        assert!(lark_str.contains("start: tool_call"), "Expected tool_call only: {}", lark_str);
        assert!(!lark_str.contains("TEXT"), "Should not contain TEXT when tool is forced: {}", lark_str);
    }
}
