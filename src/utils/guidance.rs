// src/utils/guidance.rs
//! Guided decoding support via llguidance.
//!
//! NOTE: This module is currently stubbed out due to API changes in llguidance >= 0.6.
//! The TopLevelGrammar::from_json_schema method is no longer available.
//! Guided decoding features are temporarily disabled.

use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

// Import toktrie from the crate root (it's re-exported by llguidance)
pub use toktrie::TokTrie;

pub struct GuidanceState {
    // Placeholder for future implementation
    _phantom: std::marker::PhantomData<()>,
}

impl GuidanceState {
    pub fn new(_toktrie: Arc<TokTrie>, _schema: Value) -> anyhow::Result<Self> {
        // Stubbed out - guided decoding temporarily disabled
        anyhow::bail!("Guided decoding is temporarily disabled due to llguidance API changes. \
                       The TopLevelGrammar::from_json_schema method is no longer available in llguidance >= 0.6")
    }

    pub fn compute_allowed_tokens(&mut self) -> anyhow::Result<AllowedTokens> {
        anyhow::bail!("Guided decoding is temporarily disabled")
    }

    pub fn commit_token(&mut self, _token: u32) -> anyhow::Result<()> {
        anyhow::bail!("Guided decoding is temporarily disabled")
    }
}

pub struct AllowedTokens {
    pub tokens: Vec<u32>,
    pub is_stopped: bool,
}

pub fn build_toktrie_from_tokenizer_bytes(bytes: &[u8]) -> anyhow::Result<TokTrie> {
    // Try to build TokTrie from bytes
    // The new API uses TokTrie::from() with TokRxInfo and words
    // For now, return an error as the exact migration path needs investigation
    anyhow::bail!("TokTrie construction from tokenizer bytes is temporarily disabled. \
                   The TokTrie::from_huggingface_bytes method is no longer available in toktrie >= 1.0. \
                   Input bytes length: {}", bytes.len())
}

pub fn load_toktrie_from_path(path: &Path) -> Option<TokTrie> {
    // Temporarily disabled - returns None
    crate::log_warn!("load_toktrie_from_path is disabled: {:?}", path);
    None
}
