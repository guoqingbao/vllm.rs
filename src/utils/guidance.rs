// src/utils/guidance.rs
use crate::utils::special_tokens::SpecialTokens;
use anyhow::Result;
use candle_core::Tensor;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie::{SimpleVob, TokTrie};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::tools::schema::{build_xml_tool_grammar_for_parser, ToolGrammarBuilder};

use crate::tools::Tool;
use crate::utils::logits_processor::{LogitsProcessor, Sampling};
use serde_json::json;

// Import types from server/mod.rs for grammar parsing
use crate::server::parser::{StreamToolParser, ToolConfig};
use crate::server::{ChatCompletionRequest, ResponseFormat, StructuredOutputs};
use crate::tools::{ToolChoice, ToolChoiceMode};
use crate::utils::chat_template::ChatTemplate;

use lazy_static::lazy_static;
use once_cell::sync::Lazy;

// Re-export reasoning types for convenience (without pyclass since it causes compilation issues)
pub use crate::utils::reasoning::{
    build_reasoning_grammar, thinking_grammar_with_reasoning_block, ReasoningEffort,
    ThinkingGrammarBuilder, get_reasoning_token_strings
};

#[derive(Clone, Debug, Default)]
pub struct GuidanceTokens {
    pub bos_token_ids: Vec<u32>,
    pub eos_token_ids: Vec<u32>,
    pub reasoning_start_ids: Vec<u32>,
    pub reasoning_end_ids: Vec<u32>,
    pub tool_call_start_ids: Vec<u32>,
    pub tool_call_end_ids: Vec<u32>,
}

pub fn extract_guidance_tokens(tokenizer: &Tokenizer, eos_token_ids: Vec<u32>, bos_token_ids: Vec<u32>) -> GuidanceTokens {
    let special_tokens = SpecialTokens::new(tokenizer);

    // If no BOS token IDs were provided, scan tokenizer vocab for common BOS tokens
    let bos_token_ids = if bos_token_ids.is_empty() {
        special_tokens.bos_token_ids()
    } else {
        bos_token_ids
    };

    GuidanceTokens {
        bos_token_ids,
        eos_token_ids,
        reasoning_start_ids: special_tokens.reasoning_start_ids(),
        reasoning_end_ids: special_tokens.reasoning_end_ids(),
        tool_call_start_ids: special_tokens.tool_call_start_ids(),
        tool_call_end_ids: special_tokens.tool_call_end_ids(),
    }
}

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
    constraint: Option<UserConstraint>,
}

/// User-supplied constraint types preserved verbatim until final composition
#[derive(Clone, Debug)]
pub enum UserConstraint {
    /// Choice constraint: list of string options
    Choice(Vec<String>),
    /// Regex constraint: raw regex pattern string
    Regex(String),
    /// JSON schema constraint: parsed JSON value
    Json(serde_json::Value),
    /// Lark grammar constraint: raw Lark grammar string
    Grammar(String),
    /// Structural tag constraint: parsed JSON value with start/end tags
    StructuralTag(serde_json::Value),
}

impl ConstraintBuilder {
    pub fn new() -> Self {
        Self { constraint: None }
    }

    pub fn choice(mut self, choice: Vec<String>) -> Self {
        self.constraint = Some(UserConstraint::Choice(choice));
        self
    }

    pub fn regex(mut self, regex: String) -> Self {
        self.constraint = Some(UserConstraint::Regex(regex));
        self
    }

    pub fn json(mut self, json: serde_json::Value) -> Self {
        self.constraint = Some(UserConstraint::Json(json));
        self
    }

    pub fn grammar(mut self, grammar: String) -> Self {
        self.constraint = Some(UserConstraint::Grammar(grammar));
        self
    }

    pub fn structural_tag(mut self, tag: serde_json::Value) -> Self {
        self.constraint = Some(UserConstraint::StructuralTag(tag));
        self
    }

    pub fn build(self) -> Result<Option<TopLevelGrammar>> {
        if let Some(constraint) = self.constraint {
            // Convert UserConstraint to TopLevelGrammar at final composition time
            let grammar = constraint.to_top_level_grammar()?;
            Ok(Some(grammar))
        } else {
            Err(anyhow::Error::msg("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag"))
        }
    }
}

impl UserConstraint {
    /// Convert UserConstraint to TopLevelGrammar at final composition time
    pub fn to_top_level_grammar(&self) -> Result<TopLevelGrammar> {
        match self {
            UserConstraint::Choice(choice) => {
                let choice_gram = crate::tools::schema::build_choice_lark_grammar(choice)
                    .map_err(|e| anyhow::Error::msg(e))?;
                Ok(choice_gram)
            }
            UserConstraint::Regex(regex) => {
                // For regex, use llguidance's Regex type directly via from_regex_ascii
                // This avoids converting to Lark string and re-parsing
                let sanitized = sanitize_ascii_only(regex);
                Ok(TopLevelGrammar::from_regex(&sanitized))
            }
            UserConstraint::Json(schema) => {
                let schema = crate::tools::schema::sanitize_schema_for_llguidance(schema);
                let json_gram = TopLevelGrammarExt::from_json_schema_ascii(schema.clone())
                    .map_err(|e| anyhow::Error::msg(e.to_string()))?;
                Ok(json_gram)
            }
            UserConstraint::Grammar(grammar) => {
                let sanitized = sanitize_ascii_only(grammar);
                Ok(TopLevelGrammar::from_lark(sanitized))
            }
            UserConstraint::StructuralTag(tag) => {
                let (start, end, schema) = crate::tools::schema::parse_structural_tag(tag)
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
                Ok(tool_gram)
            }
        }
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
                let lark = chat_text_expression(false);
                TopLevelGrammar::from_lark_ascii(&lark)
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
                    Some("|".to_string()),
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
    WithReasoning(TopLevelGrammar, TopLevelGrammar),
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

    pub fn into_composer(self, guidance_tokens: &GuidanceTokens) -> GrammarComposers {
        let base = self.build_base_composer(guidance_tokens);
        self.build_with_reasoning(base, guidance_tokens)
    }

    fn build_base_composer(&self, guidance_tokens: &GuidanceTokens) -> GrammarComposers {
        let tool_required = self.tool_choice_required || self.forced_tool_name.is_some();

        match (
            self.constraint_grammars.is_empty(),
            self.tool_grammar.is_some(),
        ) {
            (true, false) => GrammarComposers::TextWithEos,
            (true, true) => {
                if tool_required {
                    GrammarComposers::Tool(self.tool_grammar.clone().unwrap())
                } else {
                    let has_eos = !guidance_tokens.eos_token_ids.is_empty();
                    let lark = chat_text_expression(has_eos);
                    let text_gram = TopLevelGrammar::from_lark_ascii(&lark);
                    GrammarComposers::ConstraintOrTool(
                        text_gram,
                        self.tool_grammar.clone().unwrap(),
                    )
                }
            }
            (false, false) => GrammarComposers::Constraint(self.constraint_grammars[0].clone()),
            (false, true) => {
                let constraint = self.constraint_grammars[0].clone();
                GrammarComposers::ConstraintOrTool(constraint, self.tool_grammar.clone().unwrap())
            }
        }
    }

    fn build_with_reasoning(
        self,
        base: GrammarComposers,
        guidance_tokens: &GuidanceTokens,
    ) -> GrammarComposers {
        match self.reasoning_effort {
            Some(ReasoningEffort::None) => base,
            Some(effort) => {
                let start_ids = &guidance_tokens.reasoning_start_ids;
                let end_ids = &guidance_tokens.reasoning_end_ids;

                if start_ids.is_empty() || end_ids.is_empty() {
                    crate::log_warn!(
                        "[llg] Reasoning effort {:?} set but no reasoning tokens found",
                        effort
                    );
                    base
                } else {
                    let start_id = start_ids[0];
                    let end_id = end_ids[0];
                    let reasoning_lark =
                        thinking_grammar_with_reasoning_block(start_id, end_id, Some(effort));
                    let reasoning_gram = TopLevelGrammar::from_lark_ascii(&reasoning_lark);

                    // When reasoning is enabled and tools are available (but not required),
                    // use ConstraintOrTool as base to allow tool calls within reasoning blocks
                    let base_gram = if self.tool_grammar.is_some() && !self.tool_choice_required && self.forced_tool_name.is_none() {
                        let has_eos = !guidance_tokens.eos_token_ids.is_empty();
                        let lark = chat_text_expression(has_eos);
                        let text_gram = TopLevelGrammar::from_lark_ascii(&lark);
                        let tool_gram = self.tool_grammar.clone().unwrap();
                        GrammarComposers::ConstraintOrTool(text_gram, tool_gram).to_grammar(guidance_tokens)
                    } else {
                        base.to_grammar(guidance_tokens)
                    };

                    GrammarComposers::WithReasoning(reasoning_gram, base_gram)
                }
            }
            None => base,
        }
    }

    pub fn build(self, guidance_tokens: &GuidanceTokens) -> TopLevelGrammar {
        let composer = self.into_composer(guidance_tokens);
        composer.to_grammar(guidance_tokens)
    }
}

impl GrammarComposers {
    pub fn to_grammar(&self, guidance_tokens: &GuidanceTokens) -> TopLevelGrammar {
        let base_grammar = self.build_base_grammar(guidance_tokens);
        // Add eos termination to ensure all grammars can terminate
        // NOTE: BOS priming removed - leave bos wrapper function in place for future use
        add_eos_termination(&base_grammar, &guidance_tokens.eos_token_ids)
    }

    fn build_base_grammar(&self, guidance_tokens: &GuidanceTokens) -> TopLevelGrammar {
        match self {
            GrammarComposers::TextWithEos => {
                let has_eos = !guidance_tokens.eos_token_ids.is_empty();
                let lark = chat_text_expression(has_eos);
                TopLevelGrammar::from_lark_ascii(&lark)
            }
            GrammarComposers::Constraint(c) => c.clone(),
            GrammarComposers::Tool(t) => t.clone(),
            GrammarComposers::ConstraintOrTool(c, t) => {
                build_explicit_constraint_tool_grammar(c, t, guidance_tokens)
            }
            GrammarComposers::ToolOrConstraint(t, c) => {
                build_explicit_constraint_tool_grammar(c, t, guidance_tokens)
            }
            GrammarComposers::WithReasoning(reasoning, inner) => {
                build_explicit_reasoning_grammar(reasoning, inner, guidance_tokens)
            }
        }
    }
}

/// Build text pattern for chat conversations (without EOS)
/// have_eos: when true, generates simple text rule; when false, uses stop="" fallback
pub fn chat_text_expression(have_eos: bool) -> String {
    // First check environment variable override
    if let Ok(val) = std::env::var("VLLM_LLG_DEFAULT_TEXT") {
        return format!("{}", val);
    }

    if have_eos {
        // Text pattern without EOS - just text rule
        r#"start: text
text: /(?s:.+?)/"#
            .to_string()
    } else {
        // Fallback to stop="" when no EOS tokens available
        r#"start: text
text[stop=""]: /((?s).*?)/"#
            .to_string()
    }
}

/// Build EOS pattern for text completion
/// Returns just the EOS rule definition (not combined with text)
pub fn eos_expression(eos_token_ids: &[u32]) -> String {
    if eos_token_ids.is_empty() {
        String::new()
    } else if eos_token_ids.len() == 1 {
        format!("eos: <[{}]>\n", eos_token_ids[0])
    } else {
        let ids: Vec<String> = eos_token_ids
            .iter()
            .map(|id| format!("<[{}]>", id))
            .collect();
        let alternation = ids.join(" | ");
        format!("eos: ( {} )", alternation)
    }
}

/// Build explicit constraint-tool grammar without @ anchors
/// Generates: start: ( text | tool_call )+ with EOS added by add_eos_termination()
/// Constraint and tool rules are merged with alternation at the start level
fn build_explicit_constraint_tool_grammar(
        constraint_gram: &TopLevelGrammar,
        tool_gram: &TopLevelGrammar,
        _guidance_tokens: &GuidanceTokens,
    ) -> TopLevelGrammar {
    // Extract the constraint start rule (text or constraint rule)
    let constraint_lark = get_lark_from_top_level_grammar(constraint_gram);
    let constraint_start = if constraint_lark.contains("grammars, none have lark_grammar") {
        // For JSON schema grammars, use "text" as the default start rule name
        // since JSON schema grammars don't have a lark_grammar field
        "text".to_string()
    } else {
        extract_start_rule_rhs(&constraint_lark)
    };

    // Extract the tool start rule (tool_call)
    let tool_lark = get_lark_from_top_level_grammar(tool_gram);
    let tool_start = if tool_lark.contains("grammars, none have lark_grammar") {
        // For JSON schema grammars, use "tool_call" as the default start rule name
        "tool_call".to_string()
    } else {
        extract_start_rule_rhs(&tool_lark)
    };

    // Build alternation with + repetition
    let start_alternation = format!("{} | {}", constraint_start, tool_start);

    // Build combined rules - get rules from both grammars
    let constraint_rules = if constraint_lark.contains("grammars, none have lark_grammar") {
        // For JSON schema grammars, get rules from the json_schema field
        // Since we can't extract Lark rules, we'll use the tool rules format
        String::new()
    } else {
        extract_rules(&constraint_lark)
    };

    let tool_rules = if tool_lark.contains("grammars, none have lark_grammar") {
        // For JSON schema grammars, get rules from the tool grammar
        String::new()
    } else {
        extract_rules(&tool_lark)
    };

    // Use combine_rules to deduplicate and merge grammar rules
    let combined_rules = if constraint_rules.is_empty() {
        tool_rules
    } else if tool_rules.is_empty() {
        constraint_rules
    } else {
        // Parse rules into vec and combine with deduplication
        let constraint_rule_vec: Vec<String> = constraint_rules.lines().map(|s| s.to_string()).collect();
        let tool_rule_vec: Vec<String> = tool_rules.lines().map(|s| s.to_string()).collect();
        let all_rules = [constraint_rule_vec, tool_rule_vec].concat();
        combine_rules(all_rules)
    };

    // Build the start rule WITHOUT eos - let add_eos_termination() handle it
    let start_rule = format!("start: ( {} )+", start_alternation);

    let final_grammar = format!("{}\n{}", start_rule, combined_rules);

    TopLevelGrammar::from_lark_ascii(&final_grammar)
}

/// Build explicit reasoning grammar without @ anchors
    /// Generates: start: reasoning_block inner_rule+ with EOS added by add_eos_termination()
    /// For regex constraints, uses llguidance's Regex type directly to avoid parsing issues
    fn build_explicit_reasoning_grammar(
        reasoning_gram: &TopLevelGrammar,
        inner_gram: &TopLevelGrammar,
        _guidance_tokens: &GuidanceTokens,
    ) -> TopLevelGrammar {
    // Extract reasoning start rule
    let reasoning_lark = get_lark_from_top_level_grammar(reasoning_gram);
    let reasoning_start = if reasoning_lark.contains("grammars, none have lark_grammar") {
        // For JSON schema grammars, use "reasoning_block" as the default start rule name
        "reasoning_block".to_string()
    } else {
        extract_start_rule_rhs(&reasoning_lark)
    };

    // Extract inner start rule - for regex, use the regex pattern directly
    let inner_lark = get_lark_from_top_level_grammar(inner_gram);
    let inner_start = if inner_lark.contains("grammars, none have lark_grammar") {
        // For JSON schema grammars, use "inner" as the default start rule name
        "inner".to_string()
    } else {
        let extracted = extract_start_rule_rhs(&inner_lark);
        // If the extracted rule is a regex pattern (starts with /), return it directly
        if extracted.starts_with('/') {
            extracted
        } else {
            extracted
        }
    };

    // Build combined rules
    let reasoning_rules = if reasoning_lark.contains("grammars, none have lark_grammar") {
        String::new()
    } else {
        extract_rules(&reasoning_lark)
    };

    let inner_rules = if inner_lark.contains("grammars, none have lark_grammar") {
        String::new()
    } else {
        extract_rules(&inner_lark)
    };

    // Build combined rules with deduplication
    // Extract all rule definitions (lines matching "rule_name: ...")
    let all_rules: Vec<String> = format!("{}\n{}", reasoning_rules, inner_rules)
        .lines()
        .filter(|line| line.contains(':') && !line.trim().starts_with('#'))
        .map(|s| s.trim().to_string())
        .collect();

    // Deduplicate rules - keep only the first occurrence of each rule name
    let mut seen_rules = std::collections::HashSet::new();
    let mut unique_rules = Vec::new();
    for rule in all_rules {
        if let Some(rule_name) = rule.split(':').next() {
            if !seen_rules.contains(rule_name.trim()) {
                seen_rules.insert(rule_name.trim().to_string());
                unique_rules.push(rule);
            }
        }
    }

    let combined_rules = unique_rules.join("\n");

    // Build the start rule WITHOUT eos - let add_eos_termination() handle it
    // inner_start may already contain parentheses from grammar composition, don't double-wrap
    let start_rule = format!("start: {} {}", reasoning_start, inner_start);

    let final_grammar = format!("{}\n{}", start_rule, combined_rules);

    let mut grammar = TopLevelGrammar::from_lark_ascii(&final_grammar);
    grammar.max_tokens = reasoning_gram.max_tokens.or(inner_gram.max_tokens);
    grammar
}

/// Add BOS priming to a grammar, ensuring all grammars start with <[bos_token_id]>"assistant\n"
/// This function prepends the BOS token followed by the assistant role prefix to the start rule
/// NOTE: Function kept in place for future use - BOS wrapping is no longer applied to grammars
fn _add_bos_priming(grammar: &TopLevelGrammar, bos_token_ids: &[u32], role: &str) -> TopLevelGrammar {
    if bos_token_ids.is_empty() {
        return grammar.clone();
    }

    let lark = get_lark_from_top_level_grammar(grammar);
    let lines: Vec<&str> = lark.lines().collect();

    if lines.is_empty() {
        return grammar.clone();
    }

    // Extract the current start RHS (everything after "start:")
    let first_line = lines[0].trim();
    let current_start_rhs = if let Some(rhs) = first_line.strip_prefix("start:") {
        rhs.trim()
    } else {
        return grammar.clone();
    };

    // Build new start rule with BOS priming: start: <[bos_token_id]>"assistant\n" current_start_rhs
    let bos_line = match bos_token_ids.len() {
        // Llama
        2 => format!(r#"bos: <[{}]> "{}" <[{}]> "\n" "#, bos_token_ids[0], role, bos_token_ids[0]),
        // Qwen
        1 => format!(r#"bos: <[{}]> "{}\n" "#, bos_token_ids[0], role),
        // Fall-through
        _ => {
            let ids: Vec<String> = bos_token_ids
                .iter()
                .map(|id| format!("<[{}]>", id))
                .collect();
                let alternation = ids.join(" ? ");
            format!(r#"bos: ( {} ) "{}\n" "#, alternation, role)
        }
    };

    let new_start_line = format!(r#"start: bos {}"#, current_start_rhs);

    // Get existing rules (everything after first line)
    let other_rules = if lines.len() > 1 {
        lines[1..].join("\n")
    } else {
        String::new()
    };

    let final_grammar = format!("{}\n{}\n{}", new_start_line, other_rules, bos_line);

    TopLevelGrammar::from_lark_ascii(&final_grammar)
}

/// Add eos termination to a grammar, ensuring all paths can end with EOS
/// This function modifies the start: rule to append optional EOS token alternation
fn add_eos_termination(grammar: &TopLevelGrammar, eos_token_ids: &[u32]) -> TopLevelGrammar {
    if eos_token_ids.is_empty() {
        return grammar.clone();
    }

    let lark = get_lark_from_top_level_grammar(grammar);

    // Check if grammar already has an eos: definition (prevents duplicate eos rules)
    if lark.contains("eos:") {
        return grammar.clone();
    }

    // Check if start rule already has eos reference
    let lines: Vec<&str> = lark.lines().collect();
    if !lines.is_empty() {
        let first_line = lines[0].trim();
        if first_line.contains("eos") {
            // eos already exists in start rule, return as-is
            return grammar.clone();
        }
    }

    let is_simple_lark = grammar.grammars.len() == 1
        && grammar
            .grammars
            .first()
            .and_then(|g| g.lark_grammar.as_ref())
            .is_some();

    if !is_simple_lark {
        // For non-simple grammars, use explicit EOS termination
        return add_eos_termination_explicit(grammar, &eos_token_ids);
    }

    // For simple grammars, parse lines to find start: rule
    let lines: Vec<&str> = lark.lines().collect();
    if lines.is_empty() {
        return grammar.clone();
    }

    let first_line = lines[0].trim().replace("eos", "");

    // Extract the current start RHS (everything after "start:")
    let current_start_rhs = if let Some(rhs) = first_line.strip_prefix("start:") {
        rhs.trim()
    } else {
        return grammar.clone();
    };

    // Build new start rule with eos termination
    // For multiple EOS tokens, use ( <[id1]> | <[id2]> )? format
    let new_start_line = format!("start: {current_start_rhs} eos");
    let eos_line = if eos_token_ids.len() > 1 {
        let mut sorted_ids: Vec<u32> = eos_token_ids.iter().map(|f| f.clone()).collect();
        sorted_ids.sort_by(|a, b| b.cmp(a));

        let ids: Vec<String> = sorted_ids
            .iter()
            .map(|id| format!("<[{}]>", id))
            .collect();
        let alternation = ids.join(" ");
        // ensure emission of all EOS to prevent run-on
        format!("eos:  ( {} )", alternation)
    } else {
        format!("eos: <[{}]>", eos_token_ids[0])
    };

    // Get existing rules (everything after first line)
    let other_rules = if lines.len() > 1 {
        // Filter out any existing eos: rules to avoid duplication
        let filtered: Vec<_> = lines[1..]
            .iter()
            .filter(|line| !line.trim().starts_with("eos:"))
            .map(|s| *s)
            .collect();
        filtered.join("\n")
    } else {
        String::new()
    };

    let final_grammar = format!("{}\n{}\n{}", new_start_line, other_rules, eos_line);

    TopLevelGrammar::from_lark_ascii(&final_grammar)
}

/// Add eos termination to a grammar with explicit alternation syntax
fn add_eos_termination_explicit(
    grammar: &TopLevelGrammar,
    eos_token_ids: &[u32],
) -> TopLevelGrammar {
    if eos_token_ids.is_empty() {
        return grammar.clone();
    }

    let lark = get_lark_from_top_level_grammar(grammar);
    let lines: Vec<&str> = lark.lines().collect();
    if lines.is_empty() {
        return grammar.clone();
    }

    // Check if already has eos termination in the start rule
    if lines[0].trim().contains("eos") {
        return grammar.clone();
    }

    // Check if eos rule already exists in the grammar (to avoid duplicate rules)
    if lark.contains("eos:") {
        return grammar.clone();
    }

    // Extract current start RHS
    let current_start_rhs = if let Some(rhs) = lines[0].strip_prefix("start:") {
        rhs.trim()
    } else {
        return grammar.clone();
    };

    // Build new start rule with eos termination
    let new_start_line = format!("start: {} eos", current_start_rhs);

    // Build eos rule
    let eos_line = if eos_token_ids.len() > 1 {
        let ids: Vec<String> = eos_token_ids
            .iter()
            .map(|id| format!("<[{}]>", id))
            .collect();
        let alternation = ids.join(" | ");
        format!("eos: ( {} )", alternation)
    } else {
        format!("eos: <[{}]>", eos_token_ids[0])
    };

    let final_grammar = format!("{}\n{}", new_start_line, eos_line);

    let mut new_gram = TopLevelGrammar::from_lark_ascii(&final_grammar);
    new_gram.max_tokens = grammar.max_tokens;
    new_gram
}

/// Extension trait for TopLevelGrammar with built-in sanitization
/// This ensures all grammar construction paths sanitize inputs consistently
pub trait TopLevelGrammarExt: Sized {
    /// Create TopLevelGrammar from regex with ASCII sanitization
    fn from_regex_ascii(regex: &str) -> Self;

    /// Create TopLevelGrammar from Lark string with ASCII sanitization
    fn from_lark_ascii(lark: &str) -> Self;

    /// Create TopLevelGrammar from JSON schema with ASCII sanitization
    fn from_json_schema_ascii(schema: serde_json::Value) -> Result<Self, anyhow::Error>;

    /// Deprecated: use from_lark_ascii instead
    fn from_lark_utf8(lark: &str) -> Self {
        Self::from_lark_ascii(lark)
    }

    /// Deprecated: use from_json_schema_ascii instead
    fn from_json_schema_utf8(schema: serde_json::Value) -> Result<Self, anyhow::Error> {
        Self::from_json_schema_ascii(schema)
    }
}

impl TopLevelGrammarExt for TopLevelGrammar {
    fn from_regex_ascii(regex: &str) -> Self {
        let sanitized = sanitize_ascii_only(regex);
        Self::from_regex(&sanitized)
    }

    fn from_lark_ascii(lark: &str) -> Self {
        let sanitized = sanitize_ascii_only(lark);
        Self::from_lark(sanitized)
    }

    fn from_json_schema_ascii(schema: serde_json::Value) -> Result<Self, anyhow::Error> {
        let schema_str = serde_json::to_string(&schema)?;
        let sanitized = sanitize_ascii_only(&schema_str);
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

/// Sanitize a string by removing invalid ASCII sequences
/// Lark grammars must only contain ASCII characters for llguidance lexer
pub fn sanitize_ascii_only(s: &str) -> String {
    let mut result = String::new();
    for ch in s.chars() {
        if ch.is_ascii() {
            result.push(ch);
        }
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
        let rule_names: Vec<String> = rhs_part.split('|').map(|s| s.trim().to_string()).collect();

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

/// Extract the start rule RHS from a Lark grammar string
/// For "start: text\n...", returns "text"
/// For "start: ( rule1 | rule2 )\n...", returns "( rule1 | rule2 )"
/// For "start: rule1 rule2\n...", returns "rule1 rule2" (sequence preserved)
/// For "start: reasoning_block?\n...", returns "reasoning_block?" (quantifiers preserved)
/// BUT it preserves regex patterns like /^[a-z]+$/
fn extract_start_rule_rhs(lark: &str) -> String {
    let lines: Vec<&str> = lark.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let first_line = lines[0].trim();
    if let Some(rhs) = first_line.strip_prefix("start:") {
        let rhs = rhs.trim();

        // Extract the start rule content, handling:
        // 1. Simple rule names like "text"
        // 2. Alternations like "( rule1 | rule2 )"
        // 3. Sequences like "rule1 rule2"
        // 4. Regex patterns like "/^[a-z]+$/" (preserved as-is)

        // First, identify and temporarily replace regex patterns
        let mut temp_rhs = rhs.to_string();
        let mut regex_patterns: Vec<String> = Vec::new();
        let regex_pattern_regex = regex_pattern_regex();

        for cap in regex_pattern_regex.find_iter(rhs) {
            regex_patterns.push(cap.as_str().to_string());
        }

        for (i, regex_pat) in regex_patterns.iter().enumerate() {
            temp_rhs = temp_rhs.replace(regex_pat, &format!("__REGEX_PATTERN_{}__", i));
        }

        // Return the processed start rule
        let mut result = temp_rhs.trim().to_string();

        // Restore regex patterns
        for (i, regex_pat) in regex_patterns.iter().enumerate() {
            result = result.replace(&format!("__REGEX_PATTERN_{}__", i), regex_pat);
        }

        result
    } else {
        first_line.trim().to_string()
    }
}

/// Extract all rules from a Lark grammar except the start rule
/// Returns the rules as a single string with newlines
fn extract_rules(lark: &str) -> String {
    let lines: Vec<&str> = lark.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    // Skip the first line (start rule) and join remaining lines
    if lines.len() > 1 {
        lines[1..].join("\n")
    } else {
        String::new()
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
            rule_groups
                .entry("anonymous".to_string())
                .or_default()
                .push(rule.to_string());
        }
    }

    // Reconstruct rules, merging duplicates
    let mut combined = Vec::new();
    for (name, bodies) in rule_groups {
        if bodies.len() == 1 {
            combined.push((name.clone(), format!("{}: {}", name, bodies[0])));
        } else {
            // Multiple definitions for same rule - deduplicate bodies before combining with alternation
            // Use a HashSet to remove duplicate rule bodies
            let mut unique_bodies: std::collections::HashSet<String> = std::collections::HashSet::new();
            for body in bodies {
                unique_bodies.insert(body);
            }

            // Convert back to Vec and sort for deterministic output
            let mut unique_bodies_vec: Vec<String> = unique_bodies.into_iter().collect();
            unique_bodies_vec.sort();

            // Only combine with alternation if we have multiple unique bodies
            if unique_bodies_vec.len() == 1 {
                combined.push((name.clone(), format!("{}: {}", name, unique_bodies_vec[0])));
            } else {
                combined.push((name.clone(), format!("{}: {}", name, unique_bodies_vec.join(" | "))));
            }
        }
    }

    // Sort rules: start first, then tool rules (tool_N), then alphabetically
    combined.sort_by(|a, b| {
        let name_a = a.0.as_str();
        let name_b = b.0.as_str();

        // "start" always comes first
        if name_a == "start" {
            return std::cmp::Ordering::Less;
        }
        if name_b == "start" {
            return std::cmp::Ordering::Greater;
        }

        // Tool rules (tool_N) come next, sorted by their numeric index
        if name_a.starts_with("tool_") && name_b.starts_with("tool_") {
            // Extract the numeric part
            let num_a: u32 = name_a[5..].parse().unwrap_or(0);
            let num_b: u32 = name_b[5..].parse().unwrap_or(0);
            return num_a.cmp(&num_b);
        }
        if name_a.starts_with("tool_") {
            return std::cmp::Ordering::Less;
        }
        if name_b.starts_with("tool_") {
            return std::cmp::Ordering::Greater;
        }

        // Other rules sorted alphabetically
        name_a.cmp(name_b)
    });

    combined
        .into_iter()
        .map(|(_, rule)| rule)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Merge multiple TopLevelGrammar objects into one
/// This creates a single Lark grammar with alternation at the start rule level
/// Each sub-grammar's rules are combined directly without rule_N indirection
pub fn merge_top_level_grammars(
    grammars: Vec<TopLevelGrammar>,
    max_tokens: Option<usize>,
    start_separator: Option<String>,
) -> TopLevelGrammar {
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
    let final_grammar = format!("start: ( {} )\n{}", start_alternation, combined_rules);

    let mut top_gram = TopLevelGrammar::from_lark(final_grammar);
    top_gram.max_tokens = max_tokens;
    top_gram
}

/// Extract the Lark grammar string from TopLevelGrammar for debugging
pub fn get_lark_from_top_level_grammar(gram: &TopLevelGrammar) -> String {
    if gram.grammars.is_empty() {
        return "No grammars".to_string();
    }
    let larks: Vec<String> = gram
        .grammars
        .iter()
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
/// When working with Qwen-style tool tokens (e.g., ``), llguidance uses
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
/// For models like Qwen3 that use `` delimiters:
///
/// ```lark
/// start: tool*
/// tool: "" "\n" func "\n" "" ("\n")*
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
/// Sanitize to ASCII only - remove any non-ASCII bytes
pub fn lark_quote(value: &str) -> String {
    let ascii_only: String = value.chars().filter(|c| c.is_ascii()).collect();
    serde_json::to_string(&ascii_only).unwrap_or_else(|_| "\"\"".to_string())
}

/// Build special token syntax for Lark grammar using token IDs
/// When token IDs are available, uses <[token_id]> syntax instead of string literals
/// This ensures alignment with the outbound parser's token-based detection
pub fn lark_special_token(token_ids: &HashSet<u32>) -> String {
    if token_ids.is_empty() {
        return String::new();
    }
    // Join multiple token IDs with | - ensure ASCII only
    let ids: Vec<String> = token_ids.iter().map(|id| format!("[{}]", id)).collect();
    format!("<{}>", ids.join(","))
}

pub fn _lark_literal(value: &str, is_special: bool) -> String {
    if is_special && value.starts_with('<') && value.ends_with('>') {
        let sanitized: String = value.chars().filter(|c| c.is_ascii()).collect();
        sanitized
    } else {
        lark_quote(value)
    }
}

/// Build special token syntax for Lark grammar using token IDs
/// When token IDs are available, uses <[token_id]> syntax instead of string literals
/// This ensures alignment with the outbound parser's token-based detection
pub fn build_special_token_tag(
    token_ids: &std::collections::HashSet<u32>,
    fallback: &str,
) -> String {
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
pub fn build_tool_call_tag(
    start_token_ids: &std::collections::HashSet<u32>,
    start_token_str: &str,
) -> String {
    build_special_token_tag(start_token_ids, start_token_str)
}

/// Build tool call end tag using token IDs when available
pub fn build_tool_call_end_tag(
    end_token_ids: &std::collections::HashSet<u32>,
    end_token_str: &str,
) -> String {
    build_special_token_tag(end_token_ids, end_token_str)
}

/// Build fallback tool envelope grammar when enable_tool_grammar is false
/// This allows tool calls using text tags instead of token-based tags
/// Grammar format: start: (text | tool_call)
/// tool_call: start_tag tool_text end_tag
/// tool_text: %json {"type":"string","description":"Full tool-call syntax..."}
/// where start_tag/end_tag are string literals like "<‌tool_call>" and "<‌/tool_call>"
pub fn build_fallback_tool_envelope_grammar(
    tools: &[Tool],
    start_tag: &str,
    end_tag: &str,
    start_token_ids: &HashSet<u32>,
    end_token_ids: &HashSet<u32>,
) -> TopLevelGrammar {
    // Build the start tag using token IDs if available, otherwise use string literal
    let start_tag = if !start_token_ids.is_empty() {
        // Use token ID format: <[token_id]>
        let ids: Vec<String> = start_token_ids
            .iter()
            .map(|id| format!("[{}]", id))
            .collect();
        format!("<{}>", ids.join(","))
    } else {
        // Fall back to string literal
        lark_quote(start_tag)
    };

    // Build the end tag using token IDs if available, otherwise use string literal
    let end_tag = if !end_token_ids.is_empty() {
        // Use token ID format: <[token_id]>
        let ids: Vec<String> = end_token_ids.iter().map(|id| format!("[{}]", id)).collect();
        format!("<{}>", ids.join(","))
    } else {
        // Fall back to string literal
        lark_quote(end_tag)
    };

    // Validate tool_names before constructing schema
    let tool_names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
    if tool_names.is_empty() {
        return TopLevelGrammar::from_lark_ascii(
            r#"start: text
 text: /(?s:.+?)/
 "#,
        );
    }

    // Permissive tool-call envelope without explicit "\n" anchors as used in XML
    let lark = format!(
        r#"start: tool_call
tool_call: {} text {}
text: /(?s:.+?)/
"#,
        start_tag, end_tag
    );

    TopLevelGrammar::from_lark_ascii(&lark)
}

/// Compose grammars based on constraint and tool settings
/// Returns a single TopLevelGrammar with proper precedence
/// This function takes the grammar that was built externally (with appropriate model-specific format)
/// and handles the alternation/composition logic
pub fn compose_grammars(
    constraint_grammars: Vec<TopLevelGrammar>,
    tool_grammar: Option<TopLevelGrammar>,
    tool_choice_required: bool,
    forced_tool_name: Option<String>,
    max_tokens: Option<usize>,
    guidance_tokens: &GuidanceTokens,
    reasoning_effort: Option<ReasoningEffort>,
) -> TopLevelGrammar {
    let builder = GrammarComposerBuilder::new()
        .constraints(constraint_grammars)
        .tool_grammar(tool_grammar)
        .tool_required(tool_choice_required)
        .forced_tool_name(forced_tool_name)
        .reasoning_effort(reasoning_effort);

    let grammar = builder.build(guidance_tokens);
    let mut grammar = grammar;
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
    "/[ \\\\t\\\
 \\\
 ]+/"
}

/// Regex pattern to find Lark regex patterns like /^[a-z]+$/
static REGEX_PATTERN_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"^/.*?/$"#).expect("regex pattern should be valid at compile time")
});

fn regex_pattern_regex() -> &'static Lazy<Regex> {
    &REGEX_PATTERN_REGEX
}

/// Build Lark grammar string for tool calls
/// This function is for XML-style tool call formats and uses ASCII-only special tokens
pub fn build_tool_call_lark(
    tools: &[Tool],
    schema_map: &std::sync::Arc<std::collections::HashMap<String, serde_json::Value>>,
    start: &str,
    end: &str,
) -> String {
    let mut obj_rules = String::new();
    for tool in tools {
        let name = &tool.function.name;
        let schema_str =
            serde_json::to_string(schema_map.get(name).unwrap_or(&json!({}))).unwrap_or_default();
        obj_rules.push_str(&format!(
            "obj_{}: %json {}\n",
            name.replace("-", "_"),
            schema_str
        ));
    }

    // Sanitize start/end tags to ASCII only
    let start_ascii: String = start.chars().filter(|c| c.is_ascii()).collect();
    let end_ascii: String = end.chars().filter(|c| c.is_ascii()).collect();

    format!(
        "{} _WS? json_array _WS? {}\njson_array: \"[\" obj (\",\" obj)* \"]\"\nobj:\n_WS: {}\n{}",
        start_ascii,
        end_ascii,
        lark_ws_regex(),
        obj_rules.trim_end()
    )
}

/// Cache for precomputed mask slices to avoid expensive re-computation
#[derive(Clone, Default)]
pub struct SlicerCache {
    cache: HashMap<usize, Vec<bool>>,
}

impl SlicerCache {
    /// Get or compute a mask slice for a given position
    pub fn get_or_compute(
        &mut self,
        pos: usize,
        compute_fn: impl FnOnce() -> Vec<bool>,
    ) -> &Vec<bool> {
        if !self.cache.contains_key(&pos) {
            self.cache.insert(pos, compute_fn());
        }
        self.cache
            .get(&pos)
            .expect("entry must exist after compute")
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
    pub fn new_from_grammar(
        factory: Arc<ParserFactory>,
        grammar: &TopLevelGrammar,
    ) -> Result<Self> {
        let lark = get_lark_from_top_level_grammar(grammar);
        crate::log_info!("[llg] Composed Grammar Constraint:\n{}\n", lark);
        let parser = factory.create_parser(grammar.clone())?;
        let matcher = Matcher::new(Ok(parser));

        Ok(Self {
            matcher,
            llm_tokens: Vec::new(),
            llm_bytes: 0,
            slicer_cache: SlicerCache::default(),
        })
    }

    /// Compute mask with caching for performance
    pub fn compute_mask(&mut self) -> Result<Option<SimpleVob>> {
        if self.matcher.is_stopped() {
            return Ok(None);
        }
        let mask = self.matcher.compute_mask()?;
        Ok(Some(mask))
    }

    /// Commit token and track for speculative decoding recovery
    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        if !self.matcher.is_stopped() {
            self.matcher.consume_token(token)?;
            self.llm_tokens.push(token);
            self.llm_bytes += 4;
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
        result == 1
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
        if self.matcher.is_stopped() {
            return Ok(Vec::new());
        }

        let ff_tokens = self.matcher.compute_ff_tokens();

        for &token in &ff_tokens {
            self.matcher.consume_token(token)?;
            self.llm_tokens.push(token);
            self.llm_bytes += 4;
        }

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
    pub fn capture_snapshot(&mut self) {}

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
                state
                    .commit_token(token)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                continue;
            }

            // Stage 2: Token is invalid, compute mask and re-sample
            let mask = match state.compute_mask_or_eos() {
                Ok(m) => m,
                Err(e) => {
                    crate::log_error!(
                        "[llg] Unable to compute mask for token {} due to {}",
                        token,
                        e
                    );
                    continue;
                }
            };

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
            let mut row_vec = logits_vec.clone(); // Clone to avoid modifying original
            let row = &mut row_vec[row_start..row_end];

            // Apply bias directly to this sequence's row
            for tok in 0..vocab_size {
                if acc[tok] != 0.0 {
                    row[tok] = f32::NEG_INFINITY;
                }
            }

            // Create 1D tensor for just this sequence
            let biased_row = Tensor::from_vec(
                row_vec[row_start..row_end].to_vec(),
                (vocab_size,),
                logits.device(),
            )?;

            // Re-sample just this sequence from the biased 1D logits
            let re_sampled = logit_processor.sample_with_strategy(&biased_row, sampling)?;
            tokens[seq_idx] = re_sampled[0]; // 1D output, first (only) element

            // Commit the re-sampled token
            state
                .commit_token(tokens[seq_idx])
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }
    }

    Ok(())
}

/// Parse grammar from ChatCompletionRequest
/// Handles structured_outputs, response_format, constraint fields only.
/// Tool grammars are handled exclusively by generate_grammar_from_request
/// to avoid duplicate tool grammar construction.
/// Returns None if no grammar is specified
///
/// ## Security Guards
/// - `allow_constraint_api`: Must be true to accept client-submitted constraints
pub fn parse_grammar_from_chat_request(
    request: &ChatCompletionRequest,
    allow_constraint_api: bool,
) -> Result<Option<TopLevelGrammar>, String> {
    // First check for structured_outputs
    if let Some(ref so) = request.structured_outputs {
        if so.choice.is_some()
            || so.regex.is_some()
            || so.json.is_some()
            || so.grammar.is_some()
            || so.structural_tag.is_some()
        {
            // Guard 2: Check allow_constraint_api for structured_outputs
            if !allow_constraint_api {
                crate::log_warn!(
                    "[llg] structured_outputs constraint ignored: allow_constraint_api is false"
                );
                return Ok(None);
            }
            return build_grammar_from_structured_outputs(so);
        }
    }

    // Guard 3: Check allow_constraint_api for response_format
    if let Some(ref rf) = request.response_format {
        if !allow_constraint_api {
            crate::log_warn!(
                "[llg] response_format constraint ignored: allow_constraint_api is false"
            );
        } else {
            if let Some(grammar) = build_grammar_from_response_format(rf)? {
                return Ok(Some(grammar));
            }
        }
    }

    // Guard 4: Check allow_constraint_api for legacy constraint field
    if let Some(ref constraint) = request.constraint {
        if !allow_constraint_api {
            crate::log_warn!("[llg] constraint field ignored: allow_constraint_api is false");
        } else {
            let constraint_type = request.constraint_type.as_deref().unwrap_or("regex");
            let grammar = match constraint_type {
                "regex" => TopLevelGrammarExt::from_regex_ascii(constraint),
                "lark" => TopLevelGrammar::from_lark_ascii(constraint),
                "json_schema" | "json" => {
                    let schema: serde_json::Value = serde_json::from_str(constraint)
                        .map_err(|e| format!("Invalid JSON schema: {}", e))?;
                    let schema = crate::tools::schema::sanitize_schema_for_llguidance(&schema);
                    TopLevelGrammarExt::from_json_schema_ascii(schema)
                        .map_err(|e| format!("Invalid JSON schema: {}", e))?
                }
                other => return Err(format!("Unknown constraint_type: {}", other)),
            };
            return Ok(Some(grammar));
        }
    }

    // Tools are NOT handled here - they are handled exclusively by generate_grammar_from_request
    // to avoid duplicate tool grammar construction
    Ok(None)
}

/// Build grammar from StructuredOutputs
fn build_grammar_from_structured_outputs(
    so: &StructuredOutputs,
) -> Result<Option<TopLevelGrammar>, String> {
    // Use ConstraintBuilder from guidance.rs
    let builder = ConstraintBuilder::new();
    let builder = if let Some(ref choice) = so.choice {
        if !choice.is_empty() {
            builder.choice(choice.clone())
        } else {
            return Err("choice must have at least one option".to_string());
        }
    } else if let Some(ref regex) = so.regex {
        builder.regex(regex.clone())
    } else if let Some(ref json) = so.json {
        builder.json(json.clone())
    } else if let Some(ref grammar) = so.grammar {
        builder.grammar(grammar.clone())
    } else if let Some(ref tag) = so.structural_tag {
        builder.structural_tag(tag.clone())
    } else {
        return Ok(None);
    };

    let grammar = builder.build().map_err(|e| e.to_string())?;
    Ok(grammar)
}

/// Build grammar from ResponseFormat
fn build_grammar_from_response_format(
    response_format: &ResponseFormat,
) -> Result<Option<TopLevelGrammar>, String> {
    match response_format.format_type.as_str() {
        "json_schema" => {
            let Some(ref schema) = response_format.json_schema else {
                return Err("json_schema is required for type=json_schema".to_string());
            };
            let schema = crate::tools::schema::sanitize_schema_for_llguidance(&schema.schema);
            let grammar = TopLevelGrammarExt::from_json_schema_ascii(schema)
                .map_err(|e| format!("Invalid JSON schema: {}", e))?;
            Ok(Some(grammar))
        }
        "json_object" => {
            let grammar =
                TopLevelGrammarExt::from_json_schema_ascii(serde_json::json!({"type": "object"}))
                    .map_err(|e| format!("Invalid JSON schema: {}", e))?;
            Ok(Some(grammar))
        }
        other => Err(format!("Unsupported response_format type: {}", other)),
    }
}

/// Build TopLevelGrammar from a GrammarRequest
/// This function handles all grammar types (lark, regex, json_schema, choice)
/// and returns a parsed TopLevelGrammar ready for use in guided decoding.
pub fn build_grammar_from_request(
    grammar_type: &str,
    grammar_content: &str,
    _guidance_tokens: &GuidanceTokens,
) -> Result<TopLevelGrammar> {
    match grammar_type {
        "lark" => Ok(TopLevelGrammar::from_lark_ascii(grammar_content)),
        "regex" => Ok(TopLevelGrammarExt::from_regex_ascii(grammar_content)),
        "json_schema" => {
            let schema: serde_json::Value = serde_json::from_str(grammar_content)
                .map_err(|e| anyhow::Error::msg(format!("Invalid JSON schema: {}", e)))?;
            let schema = crate::tools::schema::sanitize_schema_for_llguidance(&schema);
            TopLevelGrammarExt::from_json_schema_ascii(schema)
                .map_err(|e| anyhow::Error::msg(format!("Invalid JSON schema: {}", e)))
        }
        "choice" => {
            // Parse choice grammar as JSON array
            let choices: Vec<String> = serde_json::from_str(grammar_content)
                .map_err(|e| anyhow::Error::msg(format!("Invalid choices array: {}", e)))?;
            let lark = crate::tools::schema::build_choice_lark_grammar(&choices).map_err(|e| {
                anyhow::Error::msg(format!("Failed to build choice grammar: {}", e))
            })?;
            Ok(lark)
        }
        other => Err(anyhow::Error::msg(format!(
            "Unsupported grammar_type: {}",
            other
        ))),
    }
}

/// Generate complete TopLevelGrammar from ChatCompletionRequest
/// Single call-site function that handles all grammar permutations
/// Returns fully composed grammar with proper <[token_id]> format for tool tags
pub fn generate_grammar_from_request(
    request: &ChatCompletionRequest,
    guidance_tokens: &GuidanceTokens,
    enable_tool_grammar: bool,
    allow_constraint_api: bool,
    model_type: &crate::utils::config::ModelType,
    model_id: &str,
    enforce_parser: Option<&str>,
    tokenizer: &Tokenizer,
    chat_template: Option<&ChatTemplate>,
) -> Option<TopLevelGrammar> {
    // Parse constraint grammar (handles structured_outputs, response_format, constraint)
    let constraint_grammar = match parse_grammar_from_chat_request(
        request,
        allow_constraint_api,
    ) {
        Ok(grammar) => grammar,
        Err(e) => {
            crate::log_warn!("[llg] Failed to parse constraint grammar: {}", e);
            return None;
        }
    };

    // Build tool config for tool grammar generation
    let tool_config = ToolConfig::from_tokenizer(tokenizer, model_type);

    // Determine parser name from enforce_parser, request, or model-specific default
    let parser_name = enforce_parser
        .map(|s| s.to_string())
        .or_else(|| {
            request
                .extra_body
                .as_ref()
                .and_then(|eb| eb.extra.get("enforce_parser"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| Some(StreamToolParser::parser_name_for_model(model_type, model_id).to_string()))
        .unwrap_or_else(|| "json".to_string());

    // Build tool grammar based on enable_tool_grammar and allow_constraint_api flags
    // - enable_tool_grammar=true: Use full XML tool grammar with schema
    // - enable_tool_grammar=false, allow_constraint_api=true: Use fallback text-based tool envelope
    // - both false: No tool grammar (tools will be detected via text parsing)
    let tool_grammar = if let Some(ref tools) = request.tools {
        if enable_tool_grammar {
            // Full tool grammar with proper schema
            let start_ids = if !tool_config.start_token_ids.is_empty() {
                Some(&tool_config.start_token_ids)
            } else {
                None
            };
            let end_ids = if !tool_config.end_token_ids.is_empty() {
                Some(&tool_config.end_token_ids)
            } else {
                None
            };

            Some(build_xml_tool_grammar_for_parser(
                tools,
                &tool_config.start_token_str,
                &tool_config.end_token_str,
                tool_config.start_is_special,
                tool_config.end_is_special,
                start_ids,
                end_ids,
                &parser_name,
            ))
        } else if allow_constraint_api {
            // When enable_tool_grammar=false but allow_constraint_api=true,
            // use fallback tool envelope grammar with text-based tags
            Some(build_fallback_tool_envelope_grammar(
                tools,
                &tool_config.start_token_str,
                &tool_config.end_token_str,
                &tool_config.start_token_ids,
                &tool_config.end_token_ids,
            ))
        } else {
            // Both flags false - no tool grammar
            None
        }
    } else {
        None
    };

    // Build tool choice info
    let tool_choice_required = request
        .tool_choice
        .as_ref()
        .map(|tc| matches!(tc, ToolChoice::Mode(ToolChoiceMode::Required)))
        .unwrap_or(false);

    let forced_tool_name = request.tool_choice.as_ref().and_then(|tc| {
        if let ToolChoice::Function { function, .. } = tc {
            Some(function.name.clone())
        } else {
            None
        }
    });

    // Build reasoning effort only when tool or constraint grammar is enabled
    // This prevents grammar generation when CLI flags are disabled
    let reasoning_effort = if enable_tool_grammar || allow_constraint_api {
        request
            .reasoning_effort
            .as_ref()
            .map(|s| ReasoningEffort::from_str(s.clone()))
    } else {
        None
    };

    // Check if reasoning tokens exist in chat template before enabling reasoning
    // Normalize both template and token strings to ASCII-only for reliable comparison
    let reasoning_tokens_in_template = if let Some((start_str, end_str)) =
        get_reasoning_token_strings(guidance_tokens, tokenizer)
    {
        if let Some(template_str) = chat_template.and_then(|t| t.get_template_string()) {
            // Ensure template is non-zero length for sanity check
            if template_str.trim().is_empty() {
                crate::log_warn!(
                    "[llg] Chat template is empty; cannot verify reasoning token presence"
                );
                true  // Allow added_vocabulary reasoning if template is invalid/empty
            } else {
                // Normalize to ASCII-only for robust comparison (handles emoji, special chars)
                let normalized_template: String = template_str.chars().filter(|c| c.is_ascii()).collect();
                let normalized_start: String = start_str.chars().filter(|c| c.is_ascii()).collect();
                let normalized_end: String = end_str.chars().filter(|c| c.is_ascii()).collect();

                // Check if normalized template contains both normalized token strings
                normalized_template.contains(&normalized_start) && normalized_template.contains(&normalized_end)
            }
        } else {
            true  // No template available, allow added_vocabulary reasoning
        }
    } else {
        true  // No reasoning tokens configured, allow added_vocabulary reasoning
    };

    // When no constraints and no tool grammar, bypass llguidance entirely
    // This is checked before compose_grammars to avoid unnecessary processing
    if constraint_grammar.is_none() && tool_grammar.is_none() {
        let has_reasoning = reasoning_effort
            .as_ref()
            .is_some_and(|effort| *effort != ReasoningEffort::None);
        if !has_reasoning {
            return None;
        }
    }

    // Compose final grammar
    let final_grammar = compose_grammars(
        constraint_grammar.into_iter().collect(),
        tool_grammar,
        tool_choice_required,
        forced_tool_name,
        request.max_tokens,
        guidance_tokens,
        reasoning_effort,
    );

    // Safeguard for models which cannot handle added_vocabulary tokens on which they were not trained
    let provide_thinking_fallback =
        std::env::var("VLLM_RS_PROVIDE_THINKING_FALLBACK")
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);

    if provide_thinking_fallback && !reasoning_tokens_in_template {
        // Models not trained on reasoning blocks have a negligible probability of emitting the end tag which makes
        // subsequent grammar unreachable to include EOS, resulting in garbage tokens or repeat of last output forever
        let lark_str = get_lark_from_top_level_grammar(&final_grammar);
        let reason_start = format!("<[{}]>", guidance_tokens.reasoning_start_ids[0]);
        let reason_end = format!("<[{}]>", guidance_tokens.reasoning_end_ids[0]);
        let lark_str = lark_str
            .replace(&reason_start, "\"<thinking>\"")
            .replace(&reason_end, "\"</thinking>\"");
        let corrected_grammar = TopLevelGrammar::from_lark(lark_str);
        Some(corrected_grammar)
    } else {
        Some(final_grammar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_text_expression_with_eos() {
        // Test that chat_text_expression generates correct Lark grammar
        let lark = chat_text_expression(true);
        assert!(lark.contains("start: text"));
        assert!(lark.contains("text: /"));
    }

    #[test]
    fn test_chat_text_expression_without_eos() {
        let lark = chat_text_expression(false);
        assert!(lark.contains("start: text"));
    }

    #[test]
    fn test_sanitize_ascii_only() {
        // Test that sanitize_ascii_only removes non-ASCII characters
        // Note: This function uses is_ascii() which only keeps ASCII characters
        // Space (0x20) is ASCII, but the input "Hello 世界 🌍" contains non-ASCII
        let input = "Hello 世界 🌍";
        let output = sanitize_ascii_only(input);
        // The function keeps only ASCII characters - space is ASCII
        assert!(output.starts_with("Hello"));
        assert!(!output.contains("世界"));
        assert!(!output.contains("🌍"));
    }

    #[test]
    fn test_sanitize_to_ascii() {
        // Test that sanitize_to_ascii filters to ASCII only
        let input = "Hello 世界 🌍";
        let output = sanitize_to_ascii(input);
        // The function keeps only ASCII bytes - space is ASCII
        assert!(output.starts_with("Hello"));
        assert!(!output.contains("世界"));
        assert!(!output.contains("🌍"));
    }

    #[test]
    fn test_extract_start_rule_rhs_preserves_quantifiers() {
        // Test that quantifiers (?, *, +) are preserved in the start rule RHS
        // This is critical for ReasoningEffort::Low which uses "reasoning_block?"

        // Test case 1: Optional quantifier (?) - ReasoningEffort::Low
        let lark1 = r#"start: reasoning_block?
reasoning_block: <[123]> text <[456]> ("\\n")?
text: /(?s:.+?)/
"#;
        let result1 = extract_start_rule_rhs(lark1);
        assert_eq!(result1, "reasoning_block?", "Optional quantifier should be preserved");

        // Test case 2: Star quantifier (*) - ReasoningEffort::High
        let lark2 = r#"start: reasoning_block* analysis_block*
reasoning_block: <[123]> text <[456]> ("\\n")?
analysis_block: text+
text: /(?s:.+?)/
"#;
        let result2 = extract_start_rule_rhs(lark2);
        assert_eq!(result2, "reasoning_block* analysis_block*", "Star quantifier should be preserved");

        // Test case 3: Plus quantifier (+) in alternation
        let lark3 = r#"start: (text | tool_call)+
text: /(?s:.+?)/
tool_call: "<tool>" text "</tool>"
"#;
        let result3 = extract_start_rule_rhs(lark3);
        assert_eq!(result3, "(text | tool_call)+", "Plus quantifier should be preserved");

        // Test case 4: No quantifier (simple case)
        let lark4 = r#"start: text
text: /(?s:.+?)/
"#;
        let result4 = extract_start_rule_rhs(lark4);
        assert_eq!(result4, "text", "No quantifier should remain as-is");
    }
}
