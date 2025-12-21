// src/utils/guidance.rs
//! Guided decoding support via llguidance.

use llguidance::api::{Constraint, TopLevelGrammar};
use llguidance::toktrie::TokTrie;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

pub struct GuidanceState {
    constraint: Constraint,
}

impl GuidanceState {
    pub fn new(toktrie: Arc<TokTrie>, schema: Value) -> anyhow::Result<Self> {
        let grammar = TopLevelGrammar::from_json_schema(schema);
        let constraint = Constraint::new(toktrie, grammar);
        Ok(Self { constraint })
    }

    pub fn compute_allowed_tokens(&mut self) -> anyhow::Result<AllowedTokens> {
        let mask = self.constraint.compute_mask()?;
        let allowed = mask.sample_tokens();
        Ok(AllowedTokens {
            tokens: allowed,
            is_stopped: mask.is_stopped(),
        })
    }

    pub fn commit_token(&mut self, token: u32) -> anyhow::Result<()> {
        self.constraint.commit_token(Some(token))?;
        Ok(())
    }
}

pub struct AllowedTokens {
    pub tokens: Vec<u32>,
    pub is_stopped: bool,
}

pub fn build_toktrie_from_tokenizer_bytes(bytes: &[u8]) -> anyhow::Result<TokTrie> {
    Ok(TokTrie::from_huggingface_bytes(bytes))
}

pub fn load_toktrie_from_path(path: &Path) -> Option<TokTrie> {
    let bytes = std::fs::read(path).ok()?;
    build_toktrie_from_tokenizer_bytes(&bytes).ok()
}
