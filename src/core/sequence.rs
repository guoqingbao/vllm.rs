// src/core/sequence.rs
use crate::utils::config::SamplingParams;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: usize,
    pub status: SequenceStatus,
    pub token_ids: Vec<u32>,
    pub output_ids: Vec<u32>,
    pub block_table: Vec<usize>,
    pub num_cached_tokens: usize,
    pub last_token: u32,
    pub block_size: usize,
    pub sampling_params: SamplingParams,
    pub prompt_length: usize,
}

impl Sequence {
    pub fn new(token_ids: Vec<u32>, block_size: usize, sampling_params: SamplingParams) -> Self {
        let prompt_length = token_ids.len();
        Self {
            id: 0, // Will be set by scheduler
            status: SequenceStatus::Waiting,
            token_ids: token_ids.clone(),
            output_ids: Vec::new(),
            block_table: Vec::new(),
            num_cached_tokens: 0,
            sampling_params,
            block_size,
            last_token: *token_ids.last().unwrap_or(&0),
            prompt_length,
        }
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn output_len(&self) -> usize {
        self.output_ids.len()
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn num_blocks(&self) -> usize {
        self.len().div_ceil(self.block_size)
    }

    pub fn last_block_num_tokens(&self) -> usize {
        self.len() - (self.num_blocks() - 1) * self.block_size
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / self.block_size
    }

    pub fn append_token(&mut self, token: u32) {
        self.token_ids.push(token);
        self.output_ids.push(token);
        self.last_token = token;
    }

    pub fn block(&self, index: usize) -> Vec<u32> {
        let start = index * self.block_size;
        let end = (index + 1) * self.block_size;
        self.token_ids[start..end.min(self.token_ids.len())].to_vec()
    }
}
