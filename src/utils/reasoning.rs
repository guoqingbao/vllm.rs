// src/utils/reasoning.rs
//! Reasoning grammar builders and utilities
//!
//! This module provides utilities for building reasoning block grammars
//! used in structured output generation.

use crate::utils::special_tokens::SpecialTokens;
use llguidance::api::TopLevelGrammar;

/// Reasoning effort level for grammar generation
/// Controls whether to prepend reasoning_block to the grammar
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn from_str(s: String) -> Self {
        match s.to_lowercase().as_str() {
            "none" => Self::None,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            _ => Self::None
        }
    }
}

/// Builder for thinking grammar with reasoning block
/// This ensures reasoning_block has finite termination to prevent run-on generation
pub struct ThinkingGrammarBuilder {
    start_id: u32,
    end_id: u32,
    effort: Option<ReasoningEffort>,
}

impl ThinkingGrammarBuilder {
    pub fn new(start_id: u32, end_id: u32, effort: Option<ReasoningEffort>) -> Self {
        Self { start_id, end_id, effort }
    }

    /// Create thinking grammar from string token IDs
    pub fn from_string(start_id: u32, end_id: u32) -> Self {
        Self { start_id, end_id, effort: None }
    }

    /// Build the thinking grammar Lark string
    pub fn build(&self) -> String {
        thinking_grammar_with_reasoning_block(self.start_id, self.end_id, self.effort.clone())
    }

    /// Build as TopLevelGrammar
    pub fn build_grammar(&self) -> TopLevelGrammar {
        let lark = self.build();
        TopLevelGrammar::from_lark(lark)
    }
}

/// Build thinking grammar with reasoning block followed by finite termination
/// The reasoning_block must have finite grammar AFTER it to prevent run-on generation
/// Pattern: start: reasoning_block (text_with_eos | tool_call)*
pub fn thinking_grammar_with_reasoning_block(start_id: u32, end_id: u32, effort: Option<ReasoningEffort>) -> String {
    crate::log_debug!("[llg] thinking_grammar_with_reasoning_block() start_id={}, end_id={}, effort={:?}", start_id, end_id, effort);
    let reason_start = "start: reasoning_block* ";
    let reason_default = format!(r#"reasoning_block: <[{start_id}]> "\n" thinkgram "\n" <[{end_id}]> "\n"
thinkgram: /(?s:.*)/
"#);

    crate::log_debug!("[llg] thinking_grammar_with_reasoning_block() generated grammar:\n{}", reason_start);
    
    format!("{}{}", reason_start, reason_default)
}

/// Build a reasoning-aware grammar composer
/// Wraps a base composer with reasoning blocks when reasoning effort is enabled
pub fn build_reasoning_grammar(
    base_grammar: TopLevelGrammar,
    reasoning_effort: ReasoningEffort,
    special_tokens: &SpecialTokens,
) -> TopLevelGrammar {
    crate::log_debug!("[llg] build_reasoning_grammar() called with effort={:?}", reasoning_effort);
    
    if reasoning_effort == ReasoningEffort::None {
        return base_grammar;
    }

    let reasoning_start_ids = special_tokens.reasoning_start_ids();
    let reasoning_end_ids = special_tokens.reasoning_end_ids();

    if reasoning_start_ids.is_empty() || reasoning_end_ids.is_empty() {
        crate::log_warn!(
            "[llg] Reasoning effort {:?} set but no reasoning tokens found in special_tokens",
            reasoning_effort
        );
        return base_grammar;
    }

    let start_id = reasoning_start_ids[0];
    let end_id = reasoning_end_ids[0];
    crate::log_info!("[llg] build_reasoning_grammar() adding reasoning block with effort={:?}", reasoning_effort);
    let reasoning_lark = thinking_grammar_with_reasoning_block(start_id, end_id, None);
    let reasoning_gram = TopLevelGrammar::from_lark(reasoning_lark);

    // Merge reasoning block with base grammar
    // The reasoning block comes first, then the base grammar
    crate::utils::guidance::merge_top_level_grammars(
        vec![reasoning_gram, base_grammar],
        None,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_effort_from_str() {
        assert_eq!(ReasoningEffort::from_str("none".to_string()), ReasoningEffort::None);
        assert_eq!(ReasoningEffort::from_str("low".to_string()), ReasoningEffort::Low);
        assert_eq!(ReasoningEffort::from_str("medium".to_string()), ReasoningEffort::Medium);
        assert_eq!(ReasoningEffort::from_str("high".to_string()), ReasoningEffort::High);
        assert_eq!(ReasoningEffort::from_str("invalid".to_string()), ReasoningEffort::None);
    }

    #[test]
    fn test_thinking_grammar_builder() {
        let builder = ThinkingGrammarBuilder::new(151657, 151658, None);
        let lark = builder.build();
        assert!(lark.contains("reasoning_block"), "Should contain reasoning_block");
        assert!(lark.contains("<[151657]"), "Should contain start token ID");
        assert!(lark.contains("<[151658]"), "Should contain end token ID");
    }

    #[test]
    fn test_thinking_grammar_builder_from_string() {
        let builder = ThinkingGrammarBuilder::from_string(151657, 151658);
        let lark = builder.build();
        assert!(lark.contains("<[151657]>"), "Should contain start token ID");
        assert!(lark.contains("<[151658]>"), "Should contain end token ID");
    }

    #[test]
    fn test_thinking_grammar_builder_build_grammar() {
        let builder = ThinkingGrammarBuilder::new(151657, 151658, None);
        let grammar = builder.build_grammar();
        assert!(grammar.grammars.len() > 0, "Should have grammars");
    }
}
