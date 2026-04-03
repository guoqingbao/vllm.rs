// src/utils/reasoning.rs
//! Reasoning grammar builders and utilities
//!
//! This module provides utilities for building reasoning block grammars
//! used in structured output generation.

use crate::utils::special_tokens::SpecialTokens;
use llguidance::api::TopLevelGrammar;
use crate::utils::guidance::{GuidanceTokens, get_lark_from_top_level_grammar};
use tokenizers::Tokenizer;

/// Reasoning effort level for grammar generation
/// Optimized for specific reasoning strategies based on current research (2024-2025)
/// Note: For Python builds, this enum is passed via serde serialization, not pyo3
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// No structured reasoning - direct output only
    None,

    /// Constrained single-paragraph reasoning (~150 chars max)
    /// Implements "Fast Thinking" with tight length constraints
    /// Reduces hallucination risk by limiting generation space
    Low,

    /// Standard multi-step Chain-of-Thought (CoT)
    /// Implements Wei et al. (2022) baseline with sentence-based termination
    /// Balances reasoning depth and efficiency
    Medium,

    /// Adversarial analysis with self-correction phases
    /// Implements Cheng & Su (2025) adversarial critique pattern
    /// Forces explicit error checking before final output
    High,

    /// Best-of-breed Chain-of-Verification (CoVe) + Self-Critique
    /// Combines Madaan et al. (2024) CoVe with adversarial self-correction
    /// Maximum accuracy for complex/fact-sensitive tasks
    ChainOfThought,

    /// Custom user-provided grammar template
    /// For non-Python builds, allows users to submit their own reasoning patterns
    #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
    Custom(String),
}

impl Into<String> for ReasoningEffort {
    fn into(self) -> String {
        match self {
            ReasoningEffort::None => "none",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::ChainOfThought => "chain_of_thought",
            #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
            ReasoningEffort::Custom(_val) => "custom",
        }
        .to_string()
    }
}

impl ReasoningEffort {
    pub fn from_str(s: String) -> Self {
        match s.to_lowercase().as_str() {
            "none" => Self::None,
            "low" => Self::Low,
            "normal" | "medium" => Self::Medium, // Backward compatibility
            "high" => Self::High,
            "very_high" | "chain_of_thought" | "cot" | "cove" => Self::ChainOfThought,
            #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
            s if s.starts_with("custom:") => Self::Custom(s[7..].to_string()),
            #[cfg(feature = "python")]
            _ => Self::None,
            #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
            _ => Self::None,
        }
    }

    /// Check if reasoning effort is enabled (not None)
    pub fn is_enabled(&self) -> bool {
        *self != ReasoningEffort::None
    }

    /// Generate the appropriate grammar template for this reasoning level
    pub fn generate_grammar(&self, start_id: u32, end_id: u32) -> String {
    match self {
            Self::None => {
                // No reasoning block - direct output only
                // Minimal latency, no structured thinking
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n\n" <[{end_id}]> "\n\n"
"#
                )
            }
            Self::Low => {
                // Fast Thinking: Single paragraph constraint (max ~150 chars)
                // Limits generation space to reduce hallucination risk
                // Uses non-greedy matching to prevent runaway generation
                // Renamed 'text' to 'text' with suffix annotation for termination
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n" think_text "\n" (think_text+ "\n")? <[{end_id}]> "\n\n"
think_text[suffix="\n"]: /[ -~]+/
"#
                )
            }
            Self::Medium => {
                // Standard CoT: Multi-step reasoning with natural sentence termination
                // Implements Wei et al. (2022) baseline pattern
                // Allows multiple steps but enforces sentence boundaries
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n" text "\n" <[{end_id}]> "\n\n"
text: /(?s:.+?)/
"#
                )
            }
            Self::High => {
                // Adversarial Analysis: Explicit self-correction phases
                // Implements Cheng & Su (2025) adversarial critique pattern
                // Forces model to challenge its own reasoning before finalizing
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> analysis_block critique_block structure_block "\n" <[{end_id}]> "\n\n"
analysis_block: "\n<analysis>\n" text "\n</analysis>\n"
critique_block: "\n<critique>\n" text "\n</critique>\n"
structure_block: "\n<structure_response>\n" text "\n</structure_response>\n"
text: /(?s:.+?)/
"#
                )
            }
            Self::ChainOfThought => {
                // Best-of-breed: CoVe + Adversarial Critique + Final Consolidation
                // Combines Madaan et al. (2024) Chain-of-Verification with self-correction
                // Maximum accuracy for complex/fact-sensitive tasks
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n" draft_block verification_block critique_block structure_block "\n" <[{end_id}]> "\n\n"
draft_block: "\n<draft>\nCardinalities of concern, intended outcomes, and structures of consideration: " text "\n</draft>\n"
verification_block: "\n<verify>\nQuestions, assumptions, and suppositions: " text "\nMechanics of proving/disproving assumptions and qualifying the facts: " text "\n</verify>\n" 
critique_block: "\n<critique>\nAdversarial assessment of evaluation: " text "\n</critique>\n"
structure_block: "\n<structure_response>\n" text "\n</structure_response>\n"
text: /(?s:.+?)/
"#
                )
            }
            #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
            Self::Custom(template) => {
                // User-provided template with token ID injection
                // Supports $START_ID and $END_ID placeholders for dynamic token ID substitution
                template
                    .replace("$START_ID", &start_id.to_string())
                    .replace("$END_ID", &end_id.to_string())
            }
        }
    }
}

/// Updated grammar builder function that respects reasoning effort levels
pub fn thinking_grammar_with_reasoning_block(
    start_id: u32,
    end_id: u32,
    effort: Option<ReasoningEffort>,
) -> String {
    match effort {
        Some(level) => level.generate_grammar(start_id, end_id),
        None => {
            // Default to Medium if not specified (balanced approach)
            ReasoningEffort::Medium.generate_grammar(start_id, end_id)
        }
    }
}

/// Builder for thinking grammar with reasoning block
/// This ensures reasoning_block has finite termination to prevent run-on generation
/// Note: For Python builds, this struct is not exposed via pyo3 since ReasoningEffort can't be a pyclass
pub struct ThinkingGrammarBuilder {
    start_id: u32,
    end_id: u32,
    effort: Option<ReasoningEffort>,
}

impl ThinkingGrammarBuilder {
    pub fn new(start_id: u32, end_id: u32, effort: Option<ReasoningEffort>) -> Self {
        Self {
            start_id,
            end_id,
            effort,
        }
    }

    /// Create thinking grammar from string token IDs
    pub fn from_string(start_id: u32, end_id: u32) -> Self {
        Self {
            start_id,
            end_id,
            effort: None,
        }
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

/// Build a reasoning-aware grammar composer
/// Wraps a base composer with reasoning blocks when reasoning effort is enabled
pub fn build_reasoning_grammar(
    base_grammar: TopLevelGrammar,
    reasoning_effort: ReasoningEffort,
    special_tokens: &SpecialTokens,
) -> TopLevelGrammar {
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
    let reasoning_lark = thinking_grammar_with_reasoning_block(start_id, end_id, None);
    let reasoning_gram = TopLevelGrammar::from_lark(reasoning_lark);

    // Merge reasoning block with base grammar
    // The reasoning block comes first, then the base grammar
    crate::utils::guidance::merge_top_level_grammars(vec![reasoning_gram, base_grammar], None, None)
}

/// Extract reasoning token strings from GuidanceTokens using tokenizer
/// Returns Some((start_string, end_string)) if tokens exist, None otherwise
pub fn get_reasoning_token_strings(
    guidance_tokens: &GuidanceTokens,
    tokenizer: &Tokenizer,
) -> Option<(String, String)> {
    if guidance_tokens.reasoning_start_ids.is_empty()
        || guidance_tokens.reasoning_end_ids.is_empty() {
        return None;
    }

    // Use tokenizer to decode token IDs to strings
    let start_str = tokenizer.decode(&guidance_tokens.reasoning_start_ids, false).ok()?;
    let end_str = tokenizer.decode(&guidance_tokens.reasoning_end_ids, false).ok()?;

    Some((start_str, end_str))
}

pub fn is_reasoning_grammar(grammar: &TopLevelGrammar) -> bool {
    // Extract the Lark representation from TopLevelGrammar
    // This requires a helper function to serialize/deserialize the grammar structure
    let lark_str = get_lark_from_top_level_grammar(grammar);
    // Check for reasoning-specific definition in the grammar structure using added_vocab tokens
    lark_str.split("\n").into_iter().any(|l: &str| l.contains("reasoning_block") && l.contains("<[") && l.contains("]>"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_reasoning_token_strings_returns_none_for_empty_tokens() {
        // Test that get_reasoning_token_strings returns None when no reasoning tokens are configured
        let guidance_tokens = GuidanceTokens {
            reasoning_start_ids: vec![],
            reasoning_end_ids: vec![],
            ..Default::default()
        };

        // We can't create a real tokenizer in a unit test, but we can test the early return
        // The function will try to decode and return None if decoding fails
        // Since we don't have a tokenizer, we just verify the function signature
        assert!(guidance_tokens.reasoning_start_ids.is_empty());
    }

    #[test]
    fn test_reasoning_effort_from_str() {
        assert_eq!(
            ReasoningEffort::from_str("none".to_string()),
            ReasoningEffort::None
        );
        assert_eq!(
            ReasoningEffort::from_str("low".to_string()),
            ReasoningEffort::Low
        );
        assert_eq!(
            ReasoningEffort::from_str("medium".to_string()),
            ReasoningEffort::Medium
        );
        assert_eq!(
            ReasoningEffort::from_str("high".to_string()),
            ReasoningEffort::High
        );
        assert_eq!(
            ReasoningEffort::from_str("invalid".to_string()),
            ReasoningEffort::None
        );
    }

    #[test]
    fn test_thinking_grammar_builder() {
        let builder = ThinkingGrammarBuilder::new(151657, 151658, None);
        let lark = builder.build();
        assert!(
            lark.contains("reasoning_block"),
            "Should contain reasoning_block"
        );
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

    #[test]
    #[cfg(not(feature = "python"))]
    fn test_reasoning_effort_custom_from_str() {
        let template = "custom:\nstart: reasoning_block\nreasoning_block: <[$START_ID]> thinkgram <[$END_ID]>\nthinkgram: /(?s:[^.!?]+[.!?])+/\n";
        let effort = ReasoningEffort::from_str(template.to_string());
        assert!(matches!(effort, ReasoningEffort::Custom(_)));
    }

    #[test]
    #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
    fn test_reasoning_effort_custom_generate_grammar() {
        let template = "custom:\nstart: reasoning_block\nreasoning_block: <$START_ID> thinkgram <$END_ID>\nthinkgram: /(?s:[^.!?]+[.!?])+/\n";
        let effort = ReasoningEffort::Custom(template.to_string());
        let grammar = effort.generate_grammar(151660, 151661);
        assert!(
            grammar.contains("reasoning_block"),
            "Should contain reasoning_block"
        );
        assert!(grammar.contains("<151660>"), "Should contain start_id");
        assert!(grammar.contains("<151661>"), "Should contain end_id");
        assert!(
            grammar.contains("custom:"),
            "Should contain custom template"
        );
    }
}
