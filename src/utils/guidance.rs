// src/utils/guidance.rs
use anyhow::Result;
use candle_core::Tensor;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use crate::utils::special_tokens::SpecialTokens;
use toktrie::{SimpleVob, TokTrie};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::tools::Tool;
use crate::utils::logits_processor::{LogitsProcessor, Sampling};
use serde_json::json;
use crate::tools::schema::ToolGrammarBuilder;

// Re-export reasoning types for convenience
pub use crate::utils::reasoning::{ReasoningEffort, ThinkingGrammarBuilder, build_reasoning_grammar, thinking_grammar_with_reasoning_block};

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
            let tool_gram = ToolGrammarBuilder::new()
                .tools(&tools)
                .start_tag(&start)
                .end_tag(&end)
                .start_is_special(false)
                .end_is_special(false)
                .build_json();
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

/// Grammar composition variant - represents all possible grammar configurations
#[derive(Clone, Debug)]
pub enum GrammarComposers {
    TextWithEos,
    Constraint(TopLevelGrammar),
    Tool(TopLevelGrammar),
    ConstraintOrTool(TopLevelGrammar, TopLevelGrammar),
    ToolOrConstraint(TopLevelGrammar, TopLevelGrammar),
    WithReasoning(TopLevelGrammar, Box<GrammarComposers>),
}

/// Builder for constructing GrammarComposers
pub struct GrammarComposerBuilder {
    constraint_grammars: Vec<TopLevelGrammar>,
    tool_grammar: Option<TopLevelGrammar>,
    has_tools: bool,
    tool_choice_required: bool,
    forced_tool_name: Option<String>,
    reasoning_effort: Option<ReasoningEffort>,
}

impl GrammarComposerBuilder {
    pub fn new() -> Self {
        Self {
            constraint_grammars: Vec::new(),
            tool_grammar: None,
            has_tools: false,
            tool_choice_required: false,
            forced_tool_name: None,
            reasoning_effort: None,
        }
    }

    pub fn constraints(mut self, grammars: Vec<TopLevelGrammar>) -> Self {
        self.constraint_grammars = grammars;
        self
    }

    pub fn tool_grammar(mut self, grammar: Option<TopLevelGrammar>) -> Self {
        self.tool_grammar = grammar;
        self.has_tools = self.tool_grammar.is_some();
        self
    }

    pub fn tool_required(mut self, required: bool) -> Self {
        self.tool_choice_required = required;
        self
    }

    pub fn forced_tool_name(mut self, name: Option<String>) -> Self {
        self.forced_tool_name = name;
        self
    }

    pub fn reasoning_effort(mut self, effort: Option<ReasoningEffort>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    pub fn into_composer(self, special_tokens: &SpecialTokens) -> GrammarComposers {
        let base = self.build_base_composer(special_tokens);
        self.build_with_reasoning(base, special_tokens)
    }

    fn build_base_composer(&self, special_tokens: &SpecialTokens) -> GrammarComposers {
        let tool_required = self.tool_choice_required || self.forced_tool_name.is_some();

        match (self.constraint_grammars.is_empty(), self.tool_grammar.is_some()) {
            (true, false) => GrammarComposers::TextWithEos,
            (true, true) => {
                if tool_required {
                    GrammarComposers::Tool(self.tool_grammar.clone().unwrap())
                } else {
                    let lark = chat_text_expression_with_eos(special_tokens);
                    let text_gram = TopLevelGrammar::from_lark_utf8(&lark);
                    GrammarComposers::ConstraintOrTool(text_gram, self.tool_grammar.clone().unwrap())
                }
            }
            (false, false) => {
                GrammarComposers::Constraint(self.constraint_grammars[0].clone())
            }
            (false, true) => {
                let constraint = self.constraint_grammars[0].clone();
                GrammarComposers::ConstraintOrTool(constraint, self.tool_grammar.clone().unwrap())
            }
        }
    }

    fn build_with_reasoning(
        self,
        base: GrammarComposers,
        special_tokens: &SpecialTokens,
    ) -> GrammarComposers {
        match self.reasoning_effort {
            Some(ReasoningEffort::None) => base,
            Some(effort) => {
                let start_ids = special_tokens.reasoning_start_ids();
                let end_ids = special_tokens.reasoning_end_ids();

                if start_ids.is_empty() || end_ids.is_empty() {
                    crate::log_warn!("[llg] Reasoning effort {:?} set but no reasoning tokens found", effort);
                    base
                } else {
                    let start_id = start_ids[0];
                    let end_id = end_ids[0];
                    let reasoning_lark = thinking_grammar_with_reasoning_block(start_id, end_id, None);
                    let reasoning_gram = TopLevelGrammar::from_lark_utf8(&reasoning_lark);
                    GrammarComposers::WithReasoning(reasoning_gram, Box::new(base))
                }
            }
            None => base,
        }
    }

    pub fn build(self, special_tokens: &SpecialTokens) -> TopLevelGrammar {
        let composer = self.into_composer(special_tokens);
        composer.to_grammar(special_tokens)
    }
}

impl GrammarComposers {
    pub fn to_grammar(&self, special_tokens: &SpecialTokens) -> TopLevelGrammar {
        match self {
            GrammarComposers::TextWithEos => {
                let lark = chat_text_expression_with_eos(special_tokens);
                TopLevelGrammar::from_lark_utf8(&lark)
            }
            GrammarComposers::Constraint(c) => c.clone(),
            GrammarComposers::Tool(t) => t.clone(),
            GrammarComposers::ConstraintOrTool(c, t) => {
                merge_top_level_grammars(vec![c.clone(), t.clone()], None, None)
            }
            GrammarComposers::ToolOrConstraint(t, c) => {
                merge_top_level_grammars(vec![t.clone(), c.clone()], None, None)
            }
            GrammarComposers::WithReasoning(reasoning, inner) => {
                let inner_gram = inner.to_grammar(special_tokens);
                merge_top_level_grammars(vec![reasoning.clone(), inner_gram], None, None)
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
        crate::log_debug!("[llg] parse_lark_grammar() input: {}", &lark);
        let (start_rhs, other_rules) = parse_lark_grammar(lark);
        crate::log_debug!(
            "[llg] parse_lark_grammar() -> start_rhs='{}', other_rules_count={}",
            start_rhs, other_rules.len()
        );
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
pub fn chat_text_expression_with_eos(special_tokens: &SpecialTokens) -> String {
    let eos_token_ids = special_tokens.eos_ids();

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



/// Compose grammars based on constraint and tool settings
/// Returns a single TopLevelGrammar with proper precedence
/// This function takes the grammar that was built externally (with appropriate model-specific format)
/// and handles the alternation/composition logic
pub fn compose_grammars(
    constraint_grammars: Vec<TopLevelGrammar>,
    tool_grammar: Option<TopLevelGrammar>,
    has_tools: bool,
    tool_choice_required: bool,
    forced_tool_name: Option<String>,
    max_tokens: Option<usize>,
    special_tokens: &SpecialTokens,
    reasoning_effort: Option<ReasoningEffort>,
) -> TopLevelGrammar {
    let builder = GrammarComposerBuilder::new()
        .constraints(constraint_grammars)
        .tool_grammar(tool_grammar)
        .tool_required(has_tools && tool_choice_required)
        .forced_tool_name(forced_tool_name)
        .reasoning_effort(reasoning_effort);

    let mut grammar = builder.build(special_tokens);
    grammar.max_tokens = max_tokens;
    grammar
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
pub fn _batch_mask_bias(
    logits: &Tensor,
    masks: &[(usize, SimpleVob)],
    vocab_size: usize,
) -> candle_core::Result<Tensor> {
    let batch_size = masks.len();
    
    // Create bias vector initialized to -inf
    let mut bias_data = vec![f32::NEG_INFINITY; batch_size * vocab_size];
    
    // Fill in allowed tokens using sparse iteration
    // masks is Vec<(batch_idx, SimpleVob)> where batch_idx is the sequence position in the batch
    for (batch_idx, mask) in masks.iter() {
        mask.iter_set_entries(|idx| {
            if idx < vocab_size {
                bias_data[*batch_idx * vocab_size + idx] = 0.0;
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
pub fn _early_exit_validate(
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
            
            // Get current sequence's logits as 1D tensor - MUST CLONE to avoid cross-contamination
            let row_start = seq_idx * vocab_size;
            let row_end = row_start + vocab_size;
            let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
            let mut row_vec = logits_vec.clone();  // Clone to avoid modifying original
            let row = &mut row_vec[row_start..row_end];
            
            // Apply bias directly to this sequence's row
            for tok in 0..vocab_size {
                if acc[tok] != 0.0 {
                    row[tok] = f32::NEG_INFINITY;
                }
            }
            
            // Create 1D tensor for just this sequence
            let biased_row = Tensor::from_vec(row_vec[row_start..row_end].to_vec(), (vocab_size,), logits.device())?;
            
            // Re-sample just this sequence from the biased 1D logits
            let re_sampled = logit_processor.sample_with_strategy(&biased_row, sampling)?;
            tokens[seq_idx] = re_sampled[0];  // 1D output, first (only) element
            
            crate::log_debug!("[llg] Consuming re-sampled token {} for seq {}", tokens[seq_idx], seq_id);
            
            // Commit the re-sampled token
            state.commit_token(tokens[seq_idx]).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        } else {
            crate::log_debug!("[llg] No guidance state for seq {}", seq_id);
        }
    }
    
    Ok(())
}
