// src/core/sequence.rs
use crate::utils::config::SamplingParams;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
    Cached, //Finished but resources not freed
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sequence {
    pub id: usize,
    pub created_time: usize,
    pub status: SequenceStatus,
    pub token_ids: Vec<u32>,
    pub output_ids: Vec<u32>,
    pub block_table: Vec<u32>,
    pub num_cached_tokens: usize,
    pub last_token: u32,
    pub block_size: usize,
    pub sampling_params: SamplingParams,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DecodeSequence {
    pub id: usize,
    pub last_token: u32,
    pub len: usize,
    pub last_block_tokens: usize,
    pub block_table_last: u32,
    pub block_tables: Vec<u32>,
}

impl DecodeSequence {
    pub fn new(sequence: &Sequence) -> Self {
        let last_token = sequence.last_token;
        let len = sequence.len();
        let last_block_tokens = sequence.last_block_num_tokens();
        let block_table_last = *sequence.block_table.last().unwrap();
        DecodeSequence {
            id: sequence.id,
            last_token,
            len,
            last_block_tokens,
            block_table_last,
            block_tables: sequence.block_table.clone(),
        }
    }

    pub fn tokens_len(&self) -> usize {
        self.len
    }
}

pub trait ToDecodeInput {
    fn last_token(&self) -> u32;
    fn len(&self) -> usize;
    fn last_block_tokens(&self) -> usize;
    fn block_table_last(&self) -> u32;
    fn block_table(&self) -> &Vec<u32>;
    fn id(&self) -> usize;
}

impl ToDecodeInput for DecodeSequence {
    fn last_token(&self) -> u32 {
        self.last_token
    }

    fn len(&self) -> usize {
        self.len
    }

    fn last_block_tokens(&self) -> usize {
        self.last_block_tokens
    }

    fn block_table_last(&self) -> u32 {
        self.block_table_last
    }

    fn block_table(&self) -> &Vec<u32> {
        &self.block_tables
    }

    fn id(&self) -> usize {
        self.id
    }
}

impl ToDecodeInput for &Sequence {
    fn last_token(&self) -> u32 {
        self.last_token
    }

    fn len(&self) -> usize {
        self.tokens_len()
    }

    fn last_block_tokens(&self) -> usize {
        self.last_block_num_tokens()
    }

    fn block_table_last(&self) -> u32 {
        *self.block_table.last().unwrap()
    }

    fn block_table(&self) -> &Vec<u32> {
        &self.block_table
    }

    fn id(&self) -> usize {
        self.id
    }
}

impl Sequence {
    pub fn new(token_ids: Vec<u32>, block_size: usize, sampling_params: SamplingParams) -> Self {
        Self {
            id: 0, // Will be set by scheduler
            created_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as usize,
            status: SequenceStatus::Waiting,
            token_ids: token_ids.clone(),
            output_ids: Vec::new(),
            block_table: Vec::new(),
            num_cached_tokens: 0,
            sampling_params,
            block_size,
            last_token: *token_ids.last().unwrap_or(&0),
        }
    }

    pub fn tokens_len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn output_len(&self) -> usize {
        self.output_ids.len()
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished || self.status == SequenceStatus::Cached
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

    pub fn created_time(&self) -> usize {
        self.created_time
    }
}
