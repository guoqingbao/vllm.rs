// src/utils/reasoning.rs
//! Reasoning grammar builders and utilities
//!
//! This module provides utilities for building reasoning block grammars
//! used in structured output generation.

use crate::utils::special_tokens::SpecialTokens;
use llguidance::api::TopLevelGrammar;

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

impl ReasoningEffort {
    pub fn from_str(s: String) -> Self {
        match s.to_lowercase().as_str() {
            "none" => Self::None,
            "low" => Self::Low,
            "normal" | "medium" => Self::Medium, // Backward compatibility
            "high" => Self::High,
            "chain_of_thought" | "cot" | "cove" => Self::ChainOfThought,
            #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
            s if s.starts_with("custom:") => Self::Custom(s[7..].to_string()),
            #[cfg(feature = "python")]
            _ => Self::None,
            #[cfg(all(not(feature = "python"), not(feature = "pyo3")))]
            _ => Self::None,
        }
    }

    /// Generate the appropriate grammar template for this reasoning level
    pub fn generate_grammar(&self, start_id: u32, end_id: u32) -> String {
        match self {
            Self::None => {
                // No reasoning block - direct output only
                // Minimal latency, no structured thinking
                format!(
                    r#"start: reasoning_block text
text: /[\x09\x0A\x0D\x20-\x7E]*?/
reasoning_block: <[{}]> "\n" text "\n" <[{}]>
"#,
                    start_id, end_id
                )
            }
            Self::Low => {
                // Fast Thinking: Single paragraph constraint (max ~150 chars)
                // Limits generation space to reduce hallucination risk
                // Uses non-greedy matching to prevent runaway generation
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n" thinkgram "\n" <[{end_id}]> "\n"
thinkgram: /[\x09\x0A\x0D\x20-\x7E]+?{{1,300}}/
"#
                )
            }
            Self::Medium => {
                // Standard CoT: Multi-step reasoning with natural sentence termination
                // Implements Wei et al. (2022) baseline pattern
                // Allows multiple steps but enforces sentence boundaries
                format!(
                    r#"start: reasoning_block
reasoning_block: <[{start_id}]> "\n" thinkgram "\n" <[{end_id}]> "\n"
thinkgram: /[\x09\x0A\x0D\x20-\x7E]+?{{1,1200}}/
"#
                )
            }
            Self::High => {
                // Adversarial Analysis: Explicit self-correction phases
                // Implements Cheng & Su (2025) adversarial critique pattern
                // Forces model to challenge its own reasoning before finalizing
                format!(
                    r#"start: reasoning_block* analysis_block*
reasoning_block: <[{start_id}]> "\n" : analysis_block analysis_content critique_phase critique_content thinkgram "\n" <[{end_id}]> "\n"
analysis_block: "<ANALYZE>" "\n" analysis_content "\n" "</ANALYZE>" "\n"
analysis_content: /[\x09\x0A\x0D\x20-\x7E]*?{{1,2400}}/
critique_phase: "<CRITIQUE>" "\n" critique_content "\n" "</CRITIQUE>" "\n"
critique_content: /[\x09\x0A\x0D\x20-\x7E]*?{{1,1200}}/
thinkgram: "<STRUCTUREDANSWER>" "\n" /[\x09\x0A\x0D\x20-\x7E]*?{{1,3600}}/ "\n" "</STRUCTUREDANSWER>" "\n"
"#
                )
            }
            Self::ChainOfThought => {
                // Best-of-breed: CoVe + Adversarial Critique + Final Consolidation
                // Combines Madaan et al. (2024) Chain-of-Verification with self-correction
                // Maximum accuracy for complex/fact-sensitive tasks
                format!(
                    r#"start: reasoning_block+
reasoning_block: <[{start_id}]> "\n" draft_phase verification_phase critique_phase final_phase "\n" <[{end_id}]> "\n"
draft_phase: /(?s:[^.!?]+[.!?])+/
verification_phase: "<VERIFY>" "\n" verification_questions "\n" verification_answers "\n" "</VERIFY>" "\n"
verification_questions: /(?s:[^.!?]+[.!?])+/
verification_answers: /[\x09\x0A\x0D\x20-\x7E]*?/
critique_phase: "<CRITIQUE>" "\n" self_critique "\n" "</CRITIQUE>" "\n"
self_critique: /[\x09\x0A\x0D\x20-\x7E]*?/
final_phase: "<FINAL_ANSWER>" "\n" final_content "\n"
final_content: /[\x09\x0A\x0D\x20-\x7E]*?/
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
    crate::log_debug!(
        "[llg] build_reasoning_grammar() called with effort={:?}",
        reasoning_effort
    );

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
    crate::log_info!(
        "[llg] build_reasoning_grammar() adding reasoning block with effort={:?}",
        reasoning_effort
    );
    let reasoning_lark = thinking_grammar_with_reasoning_block(start_id, end_id, None);
    let reasoning_gram = TopLevelGrammar::from_lark(reasoning_lark);

    // Merge reasoning block with base grammar
    // The reasoning block comes first, then the base grammar
    crate::utils::guidance::merge_top_level_grammars(vec![reasoning_gram, base_grammar], None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

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
