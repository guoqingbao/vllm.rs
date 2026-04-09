// src/utils/special_tokens.rs
use tokenizers::Tokenizer;

const REASONING_START_TOKENS: &[&str] = &[
    "<thinking>",
    "<reasoning>",
    "<internal>",
    "<reflection>",
    "<think>",
    "<|think|>",
    "[THINK]",
    "<thought>",
    "<|channel>",
];

const REASONING_END_TOKENS: &[&str] = &[
    "</thinking>",
    "</internal>",
    "</think>",
    "<|/think|>",
    "[/THINK]",
    "</thought>",
    "<channel|>",
];

#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    reasoning_start_ids: Vec<u32>,
    reasoning_end_ids: Vec<u32>,
}

impl SpecialTokens {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let mut reasoning_start_ids =
            collect_candidate_token_ids(tokenizer, REASONING_START_TOKENS);
        let mut reasoning_end_ids = collect_candidate_token_ids(tokenizer, REASONING_END_TOKENS);

        sort_and_dedup(&mut reasoning_start_ids);
        sort_and_dedup(&mut reasoning_end_ids);

        Self {
            reasoning_start_ids,
            reasoning_end_ids,
        }
    }

    pub fn reasoning_start_ids(&self) -> Vec<u32> {
        self.reasoning_start_ids.clone()
    }

    pub fn reasoning_end_ids(&self) -> Vec<u32> {
        self.reasoning_end_ids.clone()
    }
}

fn collect_candidate_token_ids(tokenizer: &Tokenizer, candidates: &[&str]) -> Vec<u32> {
    candidates
        .iter()
        .filter_map(|candidate| candidate_token_id(tokenizer, candidate))
        .collect()
}

fn candidate_token_id(tokenizer: &Tokenizer, candidate: &str) -> Option<u32> {
    let encoding = tokenizer.encode(candidate, false).ok()?;
    let ids = encoding.get_ids();
    let tokens = encoding.get_tokens();

    if ids.len() == 1 && tokens.len() == 1 && tokens[0] == candidate {
        Some(ids[0])
    } else {
        None
    }
}

fn sort_and_dedup(ids: &mut Vec<u32>) {
    ids.sort_unstable();
    ids.dedup();
}

#[cfg(test)]
mod tests {
    use super::SpecialTokens;
    use tokenizers::{models::bpe::BPE, AddedToken, Tokenizer};

    #[test]
    fn special_tokens_collects_only_exact_single_token_candidates() {
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.add_special_tokens(&[
            AddedToken::from("<think>", true),
            AddedToken::from("</think>", true),
            AddedToken::from("<|assistant|>", true),
        ]);

        let special_tokens = SpecialTokens::new(&tokenizer);

        assert_eq!(special_tokens.reasoning_start_ids().len(), 1);
        assert_eq!(special_tokens.reasoning_end_ids().len(), 1);
    }
}
