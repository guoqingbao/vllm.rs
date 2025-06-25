// src/core/scheduler.rs
use super::{
    block_manager::BlockManager,
    sequence::{Sequence, SequenceStatus},
};
use crate::utils::config::{Config, EngineConfig, EosToken};
use either::Either;
use std::collections::VecDeque;
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    block_manager: BlockManager,
    next_seq_id: usize,
    eos_token_id: Vec<u32>,
    cfg: EngineConfig,
}

impl Scheduler {
    pub fn new(econfig: &EngineConfig, config: &Config) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            block_manager: BlockManager::new(econfig.num_blocks, econfig.block_size),
            next_seq_id: 0,
            eos_token_id: match &config.eos_token_id {
                EosToken(Either::Left(Some(eos))) => vec![*eos],
                EosToken(Either::Right(Some(eos))) => eos.into_iter().map(|x| *x).collect(),
                _ => vec![],
            },
            cfg: econfig.clone(),
        }
    }

    pub fn add(&mut self, mut seq: Sequence) {
        seq.id = self.next_seq_id;
        self.next_seq_id += 1;
        self.waiting.push_back(seq);
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Schedule sequences and return their indexes in `running` along with prefill flag
    pub fn schedule(&mut self) -> (Vec<usize>, bool) {
        let mut scheduled_ids = Vec::new();
        let mut num_tokens = 0;

        // Prefill phase: move sequences from waiting to running if possible
        while let Some(mut seq) = self.waiting.pop_front() {
            if scheduled_ids.len() >= self.cfg.max_num_seqs
                || num_tokens + seq.len() > self.cfg.max_num_batched_tokens
                || !self.block_manager.can_allocate(&seq)
            {
                // Put it back and break out if cannot schedule more
                self.waiting.push_front(seq);
                break;
            }

            self.block_manager.allocate(&mut seq);
            seq.status = SequenceStatus::Running;
            num_tokens += seq.len() - seq.num_cached_tokens;
            self.running.push(seq);
            scheduled_ids.push(self.running.len() - 1); // index of newly pushed seq
        }

        if !scheduled_ids.is_empty() {
            return (scheduled_ids, true);
        }

        // Decode phase: pick sequences from running for decoding (up to max_num_seqs)
        let mut decode_ids = Vec::new();
        let mut preempt_ids = Vec::new();

        for (idx, seq) in self.running.iter().enumerate() {
            if !self.block_manager.can_append(seq) {
                preempt_ids.push(idx);
            }
        }

        for idx in preempt_ids.into_iter().rev() {
            let seq = self.running.remove(idx);
            self.waiting.push_back(seq);
        }

        for (idx, seq) in self.running.iter_mut().enumerate() {
            if decode_ids.len() >= self.cfg.max_num_seqs {
                break;
            }
            self.block_manager.may_append(seq);
            decode_ids.push(idx);
        }

        (decode_ids, false)
    }

    /// Provide immutable access to sequences by indexes (for model inference)
    pub fn get_sequences(&self, ids: &[usize]) -> Vec<&Sequence> {
        ids.iter().map(|&i| &self.running[i]).collect()
    }

    pub fn get_running(&self, id: usize) -> Option<&Sequence> {
        if id < self.running.len() {
            Some(&self.running[id])
        } else {
            None
        }
    }

    /// Postprocess output tokens and modify sequences by indexes
    pub fn postprocess(&mut self, ids: &[usize], output_ids: &[u32]) {
        for (i, &idx) in ids.iter().enumerate() {
            let seq = &mut self.running[idx];
            let token = output_ids[i];
            seq.append_token(token);

            if self.eos_token_id.contains(&token)
                || seq.output_ids.len() >= seq.sampling_params.max_tokens
            {
                seq.status = SequenceStatus::Finished;
                self.block_manager.deallocate(seq);
            }
        }
    }

    pub fn clear_finished(&mut self) {
        // Remove finished sequences from running vector
        self.running
            .retain(|seq| seq.status != SequenceStatus::Finished);
    }
}
