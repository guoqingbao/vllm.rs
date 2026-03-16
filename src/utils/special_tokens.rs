// src/utils/special_tokens.rs
use tokenizers::Tokenizer;

const REASONING_START_TOKENS: &[&str] = &[
    "<thinking>",
    "<reasoning>",
    "<internal>",
    "<reflection>",
    "<think>",
];

const REASONING_END_TOKENS: &[&str] = &["</thinking>", "</internal>", "</think>"];

#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    reasoning_start_ids: Vec<u32>,
    reasoning_end_ids: Vec<u32>,
}

impl SpecialTokens {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let mut reasoning_start_ids = Vec::new();
        let mut reasoning_end_ids = Vec::new();

        for (token, id) in tokenizer.get_vocab(true) {
            match token.as_str() {
                token if REASONING_START_TOKENS.contains(&token) => reasoning_start_ids.push(id),
                token if REASONING_END_TOKENS.contains(&token) => reasoning_end_ids.push(id),
                _ => {}
            }
        }

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

fn sort_and_dedup(ids: &mut Vec<u32>) {
    ids.sort_unstable();
    ids.dedup();
}
