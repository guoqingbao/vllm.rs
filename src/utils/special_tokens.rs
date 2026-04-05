// src/utils/special_tokens.rs
use tokenizers::Tokenizer;

const BOS_TOKEN_STRINGS: &[&str] = &[
    "<s>", "<|im_start|>",
    "<start_of_turn>", "<|beginning_of_sentence|>",
    "<bos>",
    // Llama4 encapsulates the role between header IDs:
    // <|start_header_id|>user<|end_header_id|>\n\n
    "<|start_header_id|>", "<|end_header_id|>"
];

const REASONING_TOKEN_PAIRS: &[(&str, &str)] = &[
    ("<thinking>", "</thinking>"),
    ("<|thinking|>", "<|/thinking|>"),
    ("<reasoning>", "</reasoning>"),
    ("<internal>", "</internal>"),
    ("<reflection>", "</reflection>"),
    ("<|think|>", "<|/think|>"),
    ("<thought>", "</thought>"),
    // Leave the most common selection for last - fallthrough
    ("<think>", "</think>"),
];

const TOOL_CALL_TOKEN_PAIRS: &[(&str, &str)] = &[
    ("<‌tool_call>", "<‌/tool_call>"),
    ("<start_function_call>", "<end_function_call>"),
    ("[TOOL_CALLS]", "]")
];

#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    reasoning_start_ids: Vec<u32>,
    reasoning_end_ids: Vec<u32>,
    tool_call_start_ids: Vec<u32>,
    tool_call_end_ids: Vec<u32>,
    bos_token_ids: Vec<u32>
}

impl SpecialTokens {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let mut reasoning_start_ids = Vec::new();
        let mut reasoning_end_ids = Vec::new();
        let mut tool_call_start_ids = Vec::new();
        let mut tool_call_end_ids = Vec::new();
        let mut bos_token_ids = Vec::new();

        // First pass: collect all tokens without early abort
        for (id, token) in tokenizer.get_added_tokens_decoder().iter() {
            let content = token.content.as_str();
            
            // Collect BOS tokens
            for &bos_str in BOS_TOKEN_STRINGS.iter() {
                if content == bos_str {
                    bos_token_ids.push(*id)
                }
            }

            // Collect reasoning tokens
            for &(start, end) in REASONING_TOKEN_PAIRS.iter() {
                if content == start {
                    reasoning_start_ids.push(*id);
                }
                if content == end {
                    reasoning_end_ids.push(*id);
                }
            }
            
            // Collect tool call tokens
            for &(start, end) in TOOL_CALL_TOKEN_PAIRS.iter() {
                if content == start {
                    tool_call_start_ids.push(*id);
                }
                if content == end {
                    tool_call_end_ids.push(*id);
                }
            }
        }
        
        // Second pass: validate pair completeness
        let found_tool_pair = !tool_call_start_ids.is_empty() && !tool_call_end_ids.is_empty();

        // Fall back to vocab scan if needed
        if !found_tool_pair {
            for (token, id) in tokenizer.get_vocab(true) {
                let token_str = token.as_str();
                
                // Tool call start tokens
                for &(start, _) in TOOL_CALL_TOKEN_PAIRS.iter() {
                    if token_str == start && !tool_call_start_ids.contains(&id) {
                        tool_call_start_ids.push(id);
                    }
                }
                
                // Tool call end tokens
                for &(_, end) in TOOL_CALL_TOKEN_PAIRS.iter() {
                    if token_str == end && !tool_call_end_ids.contains(&id) {
                        tool_call_end_ids.push(id);
                    }
                }
                
                // Reasoning tokens using pairs
                for &(start, end) in REASONING_TOKEN_PAIRS.iter() {
                    if token_str == start && !reasoning_start_ids.contains(&id) {
                        reasoning_start_ids.push(id);
                    }
                    if token_str == end && !reasoning_end_ids.contains(&id) {
                        reasoning_end_ids.push(id);
                    }
                }
            }
        }

        sort_and_dedup(&mut reasoning_start_ids);
        sort_and_dedup(&mut reasoning_end_ids);
        sort_and_dedup(&mut tool_call_start_ids);
        sort_and_dedup(&mut tool_call_end_ids);

        Self {
            reasoning_start_ids,
            reasoning_end_ids,
            tool_call_start_ids,
            tool_call_end_ids,
            bos_token_ids
        }
    }

    pub fn bos_token_ids(&self) -> Vec<u32> {
        self.bos_token_ids.clone()
    }

    pub fn reasoning_start_ids(&self) -> Vec<u32> {
        self.reasoning_start_ids.clone()
    }

    pub fn reasoning_end_ids(&self) -> Vec<u32> {
        self.reasoning_end_ids.clone()
    }

    pub fn tool_call_start_ids(&self) -> Vec<u32> {
        self.tool_call_start_ids.clone()
    }

    pub fn tool_call_end_ids(&self) -> Vec<u32> {
        self.tool_call_end_ids.clone()
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

/// Helper function to sort and deduplicate token IDs
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
