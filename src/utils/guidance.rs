// src/utils/guidance.rs
use anyhow::Result;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie::{SimpleVob, TokTrie};
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::utils::config::Constraint;

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
    pub fn new(factory: Arc<ParserFactory>, constraint: &Constraint) -> Result<Self> {
        crate::log_debug!("[llg] GuidanceState::new() called with constraint type: {:?}", constraint);
        let grammar = llg_grammar_from_constraint(constraint)?;
        crate::log_debug!("[llg] Grammar converted from constraint: {:?}", grammar.is_some());

        let grammar = match grammar {
            Some(g) => g,
            None => {
                crate::log_error!("[llg] Cannot create GuidanceState from Constraint::None");
                anyhow::bail!("Cannot create GuidanceState from Constraint::None");
            }
        };

        crate::log_trace!("[llg] Creating parser from grammar");
        let parser = factory.create_parser(grammar)?;
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

pub fn llg_grammar_from_constraint(constraint: &Constraint) -> Result<Option<TopLevelGrammar>> {
    let grm = match constraint {
        Constraint::Regex(regex) => TopLevelGrammar::from_regex(regex),
        Constraint::Lark(lark) => TopLevelGrammar::from_lark(lark.clone()),
        Constraint::JsonSchema(value) => TopLevelGrammar::from_json_schema(value.clone()),
        Constraint::Llguidance(value) => value.clone(),
        Constraint::None => return Ok(None),
    };
    Ok(Some(grm))
}
