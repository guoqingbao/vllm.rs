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

use crate::tools::schema::{build_tool_grammar_for_parser, ToolGrammarBuilder};

use crate::tools::Tool;
use crate::utils::logits_processor::{LogitsProcessor, Sampling};
use serde_json::json;

// Import types from server/mod.rs for grammar parsing
use crate::server::parser::{StreamToolParser, ToolConfig};
use crate::server::{ChatCompletionRequest, ResponseFormat, StructuredOutputs};
use crate::tools::{ToolChoice, ToolChoiceMode};
use crate::utils::chat_template::ChatTemplate;

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

/// Configuration for structural tag constraints
#[derive(Debug, Clone)]
pub struct StructuralTagConfig {
    pub start_tag: String,
    pub end_tag: String,
    pub schema: serde_json::Value,
}

/// All constraint/grammar input variants - unified entry point for grammar composition
/// This enum replaces UserConstraint, GrammarComposers, GrammarBuilder, ConstraintBuilder
#[derive(Debug, Clone)]
pub enum GrammarInput {
    /// Choice constraint: list of string options
    Choice(Vec<String>),
    /// Regex constraint: raw regex pattern string
    Regex(String),
    /// JSON schema constraint: parsed JSON value
    Json(serde_json::Value),
    /// Lark grammar constraint: raw Lark grammar string
    Lark(String),
    /// Structural tag constraint: parsed JSON value with start/end tags
    StructuralTag(StructuralTagConfig),

    /// Tool required - all tools available
    ToolRequired,
    /// Tool optional - tools available but not required
    ToolOptional,
    /// Tool with specific name forced
    ToolForced(String),

    /// Combined constraint and tool - sequential composition
    ConstraintThenTool {
        constraint: Box<GrammarInput>,
        tool_name: Option<String>,
    },
    /// Combined tool and constraint - sequential composition
    ToolThenConstraint {
        tool_name: Option<String>,
        constraint: Box<GrammarInput>,
    },

    /// Reasoning with base grammar
    Reasoning {
        effort: ReasoningEffort,
        base: Box<GrammarInput>,
    },
}

impl GrammarInput {
    /// Build base grammar from this input without tool or reasoning composition
    pub fn build_base_grammar(&self, guidance_tokens: &GuidanceTokens) -> GrammarResult<TopLevelGrammar> {
        match self {
            GrammarInput::Choice(choice) => {
                let choice_gram = crate::tools::schema::build_choice_lark_grammar(choice)
                    .map_err(|e| GrammarError::InvalidGrammar(e.to_string()))?;
                Ok(choice_gram)
            }
            GrammarInput::Regex(regex) => {
                let sanitized = sanitize_ascii_only(regex);
                Ok(TopLevelGrammar::from_regex(&sanitized))
            }
            GrammarInput::Json(schema) => {
                let schema = crate::tools::schema::sanitize_schema_for_llguidance(schema);
                TopLevelGrammarExt::from_json_schema_ascii(schema)
                    .map_err(|e| GrammarError::InvalidGrammar(e.to_string()))
            }
            GrammarInput::Lark(grammar) => {
                let sanitized = sanitize_ascii_only(grammar);
                Ok(TopLevelGrammar::from_lark(sanitized))
            }
            GrammarInput::StructuralTag(config) => {
                let schema = crate::tools::schema::sanitize_schema_for_llguidance(&config.schema);
                let tools = crate::tools::schema::schema_to_tools(&schema);
                let start_ids = if guidance_tokens.tool_call_start_ids.is_empty() {
                    None
                } else {
                    Some(&guidance_tokens.tool_call_start_ids)
                };
                let end_ids = if guidance_tokens.tool_call_end_ids.is_empty() {
                    None
                } else {
                    Some(&guidance_tokens.tool_call_end_ids)
                };
                let tool_gram = ToolGrammarBuilder::new()
                    .tools(&tools)
                    .start_tag(&config.start_tag)
                    .end_tag(&config.end_tag)
                    .start_is_special(false)
                    .end_is_special(false)
                    .start_token_ids(start_ids.map(|s| s.iter().copied().collect()))
                    .end_token_ids(end_ids.map(|s| s.iter().copied().collect()))
                    .build_json();
                Ok(tool_gram)
            }
            GrammarInput::ToolRequired | GrammarInput::ToolOptional => {
                // For tool-only cases, return a text grammar that can be composed later
                let lark = chat_text_expression(!guidance_tokens.eos_token_ids.is_empty());
                Ok(TopLevelGrammar::from_lark_ascii(&lark))
            }
            GrammarInput::ToolForced(_name) => {
                let lark = chat_text_expression(!guidance_tokens.eos_token_ids.is_empty());
                Ok(TopLevelGrammar::from_lark_ascii(&lark))
            }
            GrammarInput::ConstraintThenTool { constraint, .. } => {
                constraint.build_base_grammar(guidance_tokens)
            }
            GrammarInput::ToolThenConstraint { constraint, .. } => {
                constraint.build_base_grammar(guidance_tokens)
            }
            GrammarInput::Reasoning { base, .. } => {
                base.build_base_grammar(guidance_tokens)
            }
        }
    }

    /// Compose this grammar input with tools and reasoning
    /// Sequential composition: tool_start constraint_body eos (not alternation)
    pub fn compose(
        &self,
        guidance_tokens: &GuidanceTokens,
        tool_grammar: Option<TopLevelGrammar>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> GrammarResult<TopLevelGrammar> {
        // Build base grammar
        let base_grammar = self.build_base_grammar(guidance_tokens)?;

        // Compose with tool grammar if provided
        let composed = if let Some(tool_gram) = tool_grammar {
            Self::compose_tool(&base_grammar, &tool_gram)?
        } else {
            base_grammar
        };

        // Apply reasoning if specified
        if let Some(effort) = reasoning_effort {
            if effort != ReasoningEffort::None {
                let reasoning_gram = Self::build_reasoning_grammar(&composed, effort, guidance_tokens)?;
                return Self::compose_reasoning(&reasoning_gram, &composed);
            }
        }

        Ok(composed)
    }

    /// Build reasoning grammar from effort level
    fn build_reasoning_grammar(
        _base: &TopLevelGrammar,
        effort: ReasoningEffort,
        guidance_tokens: &GuidanceTokens,
    ) -> GrammarResult<TopLevelGrammar> {
        let start_ids = &guidance_tokens.reasoning_start_ids;
        let end_ids = &guidance_tokens.reasoning_end_ids;

        if start_ids.is_empty() || end_ids.is_empty() {
            return Err(GrammarError::InvalidGrammar(
                "Reasoning tokens not configured".to_string(),
            ));
        }

        let start_id = start_ids[0];
        let end_id = end_ids[0];
        let reasoning_lark = thinking_grammar_with_reasoning_block(start_id, end_id, Some(effort));
        Ok(TopLevelGrammar::from_lark_ascii(&reasoning_lark))
    }

    /// Compose reasoning grammar with base grammar
    /// Sequential composition: reasoning_block followed by base constraint
    pub fn compose_reasoning(
        reasoning: &TopLevelGrammar,
        base: &TopLevelGrammar,
    ) -> GrammarResult<TopLevelGrammar> {
        let reasoning_lark = get_lark_from_top_level_grammar(reasoning);
        let base_lark = get_lark_from_top_level_grammar(base);

        // Extract reasoning start rule
        let reasoning_start = if reasoning_lark.contains("grammars, none have lark_grammar") {
            "reasoning_block".to_string()
        } else {
            extract_start_rule_rhs(&reasoning_lark)
        };

        // Extract base start rule
        let base_start = if base_lark.contains("grammars, none have lark_grammar") {
            "text".to_string()
        } else {
            extract_start_rule_rhs(&base_lark)
        };

        // Extract rules from both grammars
        let reasoning_rules = if reasoning_lark.contains("grammars, none have lark_grammar") {
            String::new()
        } else {
            extract_rules(&reasoning_lark)
        };

        let base_rules = if base_lark.contains("grammars, none have lark_grammar") {
            Self::extract_text_rule_from_json_schema(base)
        } else {
            extract_rules(&base_lark)
        };

        // Merge rules with deduplication
        let all_rules: Vec<String> = format!("{}\n{}", reasoning_rules, base_rules)
            .lines()
            .filter(|line| line.contains(':') && !line.trim().starts_with('#'))
            .map(|s| s.trim().to_string())
            .collect();

        let unique_rules = combine_rules(all_rules);

        // Sequential composition: reasoning_block followed by base constraint
        let start_rule = format!("start: {} {}", reasoning_start, base_start);
        let final_grammar = format!("{}\n{}", start_rule, unique_rules);

        let mut grammar = TopLevelGrammar::from_lark_ascii(&final_grammar);
        grammar.max_tokens = reasoning.max_tokens.or(base.max_tokens);
        Ok(grammar)
    }

    /// Extract the text rule from a JSON schema constraint
    /// Compiles the JSON schema to Lark and extracts the text rule definition
    fn extract_text_rule_from_json_schema(gram: &TopLevelGrammar) -> String {
        if gram.grammars.is_empty() {
            return String::new();
        }

        // Get the JSON schema from the first grammar
        if let Some(json_schema) = &gram.grammars[0].json_schema {
            // The text rule is defined as: text: %json {schema}
            let schema_str = serde_json::to_string(json_schema).unwrap_or_default();
            format!("text: %json {}", schema_str)
        } else {
            String::new()
        }
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

    /// Extract the text: rule definition from a Lark grammar
    fn extract_text_rule(lark: &str) -> Option<String> {
        let lines: Vec<&str> = lark.lines().collect();
        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("text:") {
                return Some(trimmed.to_string());
            }
        }
        None
    }

    /// Extract tool_call related rules from a Lark grammar
    fn extract_tool_rules(lark: &str) -> String {
        let lines: Vec<&str> = lark.lines().collect();
        let mut tool_rules = Vec::new();
        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("tool_call:") ||
                trimmed.starts_with("tool_") ||
                trimmed.starts_with("param_") ||
                trimmed.starts_with("value_") {
                tool_rules.push(trimmed.to_string());
            }
        }
        tool_rules.join("\n")
    }

    /// Compose tool grammar with constraint grammar
    /// Alternation: start: ( text | tool_call )+
    /// The text: rule is replaced with the constraint definition
    pub fn compose_tool(constraint: &TopLevelGrammar, tool: &TopLevelGrammar) -> GrammarResult<TopLevelGrammar> {
        let constraint_lark = get_lark_from_top_level_grammar(constraint);
        let tool_lark = get_lark_from_top_level_grammar(tool);

        // Extract the text: rule from constraint grammar (this will replace the default text rule)
        let constraint_text_rule = if constraint_lark.contains("grammars, none have lark_grammar") {
            // Default text rule for non-lark grammars
            "text: /(?s:.+?)/".to_string()
        } else {
            // Extract the text: rule from constraint grammar
            Self::extract_text_rule(&constraint_lark).unwrap_or("text: /(?s:.+?)/".to_string())
        };

        // Extract tool_call rules from tool grammar
        let tool_rules = Self::extract_tool_rules(&tool_lark);

        // Combine constraint text rule with tool rules
        let all_rules = format!("{}\n{}", constraint_text_rule, tool_rules);
        let combined_rules = combine_rules(all_rules.lines().map(|s| s.to_string()).collect());

        // Keep start: ( text | tool_call )+ eos unchanged
        // This allows alternation between text (constrained) and tool calls
        let start_rule = "start: ( text | tool_call )+".to_string();
        let final_grammar = format!("{}\n{}", start_rule, combined_rules);

        Ok(TopLevelGrammar::from_lark_ascii(&final_grammar))
    }

    /// Create GrammarInput from StructuredOutputs
    pub fn from_structured_outputs(
        choice: Option<Vec<String>>,
        regex: Option<String>,
        json: Option<serde_json::Value>,
        grammar: Option<String>,
        structural_tag: Option<serde_json::Value>,
    ) -> GrammarResult<Self> {
        let constraint_count = [
            choice.is_some(),
            regex.is_some(),
            json.is_some(),
            grammar.is_some(),
            structural_tag.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count();

        if constraint_count > 1 {
            return Err(GrammarError::TooManyConstraints);
        }

        if let Some(choice) = choice {
            if !choice.is_empty() {
                return Ok(GrammarInput::Choice(choice));
            }
            return Err(GrammarError::InvalidGrammar("choice must have at least one option".to_string()));
        }

        if let Some(regex) = regex {
            return Ok(GrammarInput::Regex(regex));
        }

        if let Some(json) = json {
            return Ok(GrammarInput::Json(json));
        }

        if let Some(grammar) = grammar {
            return Ok(GrammarInput::Lark(grammar));
        }

        if let Some(tag) = structural_tag {
            let (start, end, schema) = crate::tools::schema::parse_structural_tag(&tag)
                .map_err(|e| GrammarError::InvalidGrammar(e))?;
            let schema = crate::tools::schema::sanitize_schema_for_llguidance(&schema);
            return Ok(GrammarInput::StructuralTag(StructuralTagConfig {
                start_tag: start,
                end_tag: end,
                schema,
            }));
        }

        Err(GrammarError::InvalidGrammar("No constraint specified".to_string()))
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

/// Simplified EOS insertion function for use in compose()
/// Appends 'eos' to the start rule and adds the EOS rule definition
pub fn eos_insertion(grammar: &TopLevelGrammar, eos_token_ids: &[u32]) -> TopLevelGrammar {
    if eos_token_ids.is_empty() {
        return grammar.clone();
    }

    let lark = get_lark_from_top_level_grammar(grammar);
    
    // Check if eos is already in the start rule
    let lines: Vec<&str> = lark.lines().collect();
    if !lines.is_empty() {
        let first_line = lines[0].trim();
        if first_line.contains("eos") {
            return grammar.clone();
        }
    }

    // Check if eos rule already exists
    if lark.contains("eos:") {
        return grammar.clone();
    }

    // Extract current start RHS (everything after "start:")
    let first_line = lines.first().map(|s| s.trim()).unwrap_or("");
    let current_start_rhs = if let Some(rhs) = first_line.strip_prefix("start:") {
        rhs.trim()
    } else {
        return grammar.clone();
    };

    // Build new start rule with eos appended
    let new_start_line = format!("start: {} eos", current_start_rhs.trim());

    // Build eos rule definition
    let eos_line = if eos_token_ids.len() == 1 {
        format!("eos: <[{}]>", eos_token_ids[0])
    } else {
        let ids: Vec<String> = eos_token_ids
            .iter()
            .map(|id| format!("<[{}]>", id))
            .collect();
        let alternation = ids.join(" | ");
        format!("eos: ( {} )", alternation)
    };

    // Get existing rules (everything after first line)
    let other_rules = if lines.len() > 1 {
        lines[1..].join("\n")
    } else {
        String::new()
    };

    // Combine: new start rule, existing rules, eos rule
    let final_grammar = format!("{}\n{}\n{}", new_start_line, other_rules, eos_line);

    TopLevelGrammar::from_lark_ascii(&final_grammar)
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
    let mut larks: Vec<String> = gram
        .grammars
        .iter()
        .filter_map(|g| g.lark_grammar.as_ref())
        .map(|s| s.clone())
        .collect();

    for g in &gram.grammars {
        if let Some(json_schema) = &g.json_schema {
            // Convert JSON schema to Lark string with text: %json {...} rule
            let schema_str = serde_json::to_string(json_schema).unwrap_or_default();
            larks.push(format!("start: text\ntext: %json {}", schema_str));
        }
    }

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
/// When working with Qwen-style tool tokens (e.g., `<\u200tool_call>`), llguidance uses
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
/// This function uses GrammarInput::compose() for sequential composition
/// Compose grammars based on constraint and tool settings
/// Returns a single TopLevelGrammar with proper precedence
/// This function uses GrammarInput::compose() for sequential composition
pub fn compose_grammars(
    constraint_grammars: Vec<TopLevelGrammar>,
    tool_grammar: Option<TopLevelGrammar>,
    tool_choice_required: bool,
    forced_tool_name: Option<String>,
    max_tokens: Option<usize>,
    guidance_tokens: &GuidanceTokens,
    reasoning_effort: Option<ReasoningEffort>,
) -> TopLevelGrammar {
    let reasoning_enabled = reasoning_effort
        .as_ref()
        .is_some_and(|effort| *effort != ReasoningEffort::None);

    // Handle tool-only case: return tool grammar directly (no constraint)
    if tool_grammar.is_some() && constraint_grammars.is_empty() {
        let mut grammar = tool_grammar.unwrap();
        if let Some(max) = max_tokens {
            grammar.max_tokens = Some(max);
        }
        // Add EOS termination to the tool-only grammar
        return eos_insertion(&grammar, &guidance_tokens.eos_token_ids);
    }
    
    // Build GrammarInput from constraint grammar(s)
    let grammar_input = if let Some(gram) = constraint_grammars.first().cloned() {
        // Extract the Lark grammar string from TopLevelGrammar
        let lark = get_lark_from_top_level_grammar(&gram);
        GrammarInput::Lark(lark)
    } else {
        // No constraint - use text as default
        let lark = chat_text_expression(!guidance_tokens.eos_token_ids.is_empty());
        GrammarInput::Lark(lark)
    };

    // Use GrammarInput::compose() for the composition
    let result = grammar_input.compose(guidance_tokens, tool_grammar, reasoning_effort);
    
    let mut grammar = match result {
        Ok(mut grammar) => {
            if let Some(max) = max_tokens {
                grammar.max_tokens = Some(max);
            }
            grammar
        }
        Err(e) => {
            crate::log_warn!("[llg] Grammar composition failed: {}", e);
            // Fallback to text grammar
            let lark = chat_text_expression(!guidance_tokens.eos_token_ids.is_empty());
            TopLevelGrammar::from_lark_ascii(&lark)
        }
    };

    // When tool_choice is required, modify the start rule to enforce tool_call first
    if tool_choice_required {
        let lark = get_lark_from_top_level_grammar(&grammar);
        let modified_lark = enforce_tool_call_first(&lark, reasoning_enabled, forced_tool_name.as_deref());
        grammar = TopLevelGrammar::from_lark_ascii(&modified_lark);
    }
    // Add EOS termination to the composed grammar (single point of entry)
    grammar = eos_insertion(&grammar, &guidance_tokens.eos_token_ids);
    
    grammar
}

/// Modify the start rule to enforce tool_call first when tool_choice is required
fn enforce_tool_call_first(lark: &str, reasoning_enabled: bool, forced_tool_name: Option<&str>) -> String {
    let lines: Vec<&str> = lark.lines().collect();
    if lines.is_empty() {
        return lark.to_string();
    }

    // Extract current start rule RHS
    let first_line = lines[0].trim();
    if !first_line.strip_prefix("start:").is_some() {
        return lark.to_string();
    };

    // Build new start rule based on configuration
    // Use tool_call as the mandatory marker (matches existing grammar structure)
    // Ignore forced_tool_name for now - just use tool_call
    let new_start_rhs = if reasoning_enabled {
        "reasoning_block tool_call".to_string()
    } else {
        "tool_call".to_string()
    };

    let new_start_line = format!("start: {}", new_start_rhs);
    
    // Join all lines back together
    let remaining_lines = if lines.len() > 1 { lines[1..].join("\n") } else { String::new() };
    
    format!("{}\n{}", new_start_line, remaining_lines)
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
    // Use GrammarInput::from_structured_outputs() instead of ConstraintBuilder
    let grammar_input = GrammarInput::from_structured_outputs(
        so.choice.clone(),
        so.regex.clone(),
        so.json.clone(),
        so.grammar.clone(),
        so.structural_tag.clone(),
    );
    
    match grammar_input {
        Ok(grammar_input) => {
            // Convert GrammarInput to TopLevelGrammar using build_base_grammar
            let guidance_tokens = GuidanceTokens::default();
            let grammar = grammar_input.build_base_grammar(&guidance_tokens)
                .map_err(|e| e.to_string())?;
            Ok(Some(grammar))
        }
        Err(GrammarError::TooManyConstraints) => {
            Err("structured_outputs must set exactly one of choice, regex, json, grammar, or structural_tag".to_string())
        }
        Err(GrammarError::InvalidGrammar(msg)) => {
            Err(msg)
        }
        Err(GrammarError::MissingJsonSchema) => {
            Err("response_format.json_schema is required for type=json_schema".to_string())
        }
        Err(GrammarError::UnsupportedFormat(msg)) => {
            Err(format!("unsupported response_format type: {}", msg))
        }
        Err(GrammarError::ToolGrammarError(msg)) => {
            Err(format!("tool grammar construction failed: {}", msg))
        }
    }
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

            Some(build_tool_grammar_for_parser(
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

    #[test]
    fn test_grammar_input_choice() {
        // Test GrammarInput::Choice variant
        let input = GrammarInput::Choice(vec!["option1".to_string(), "option2".to_string()]);
        assert!(matches!(input, GrammarInput::Choice(_)));
    }

    #[test]
    fn test_grammar_input_regex() {
        // Test GrammarInput::Regex variant
        let input = GrammarInput::Regex(r"[a-z]+".to_string());
        assert!(matches!(input, GrammarInput::Regex(_)));
    }

    #[test]
    fn test_grammar_input_json() {
        // Test GrammarInput::Json variant
        let input = GrammarInput::Json(serde_json::json!({"type": "object"}));
        assert!(matches!(input, GrammarInput::Json(_)));
    }

    #[test]
    fn test_grammar_input_lark() {
        // Test GrammarInput::Lark variant
        let input = GrammarInput::Lark("start: text\ntext: /(?s:.+?)/".to_string());
        assert!(matches!(input, GrammarInput::Lark(_)));
    }

    #[test]
    fn test_grammar_input_structural_tag() {
        // Test GrammarInput::StructuralTag variant
        let config = StructuralTagConfig {
            start_tag: "<tag>".to_string(),
            end_tag: "</tag>".to_string(),
            schema: serde_json::json!({"type": "object"}),
        };
        let input = GrammarInput::StructuralTag(config);
        assert!(matches!(input, GrammarInput::StructuralTag(_)));
    }

    #[test]
    fn test_grammar_input_reasoning() {
        // Test GrammarInput::Reasoning variant
        let base = GrammarInput::Choice(vec!["a".to_string(), "b".to_string()]);
        let input = GrammarInput::Reasoning {
            effort: ReasoningEffort::Low,
            base: Box::new(base),
        };
        assert!(matches!(input, GrammarInput::Reasoning { .. }));
    }

    #[test]
    fn test_grammar_input_constraint_then_tool() {
        // Test GrammarInput::ConstraintThenTool variant
        let constraint = GrammarInput::Json(serde_json::json!({"type": "object"}));
        let input = GrammarInput::ConstraintThenTool {
            constraint: Box::new(constraint),
            tool_name: None,
        };
        assert!(matches!(input, GrammarInput::ConstraintThenTool { .. }));
    }

    #[test]
    fn test_grammar_input_tool_then_constraint() {
        // Test GrammarInput::ToolThenConstraint variant
        let constraint = GrammarInput::Json(serde_json::json!({"type": "object"}));
        let input = GrammarInput::ToolThenConstraint {
            tool_name: None,
            constraint: Box::new(constraint),
        };
        assert!(matches!(input, GrammarInput::ToolThenConstraint { .. }));
    }
}
