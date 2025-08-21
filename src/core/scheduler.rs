// src/core/scheduler.rs
use super::{
    block_manager::BlockManager,
    sequence::{Sequence, SequenceStatus},
};
use crate::utils::config::{Config, EngineConfig, EosTokenId};
use candle_core::Result;
use std::collections::VecDeque;
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    block_manager: BlockManager,
    next_seq_id: usize,
    eos_token_id: Vec<u32>,
    cfg: EngineConfig,
    cached_seqs: VecDeque<(usize, String)>,
}

impl Scheduler {
    pub fn new(econfig: &EngineConfig, config: &Config) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            block_manager: BlockManager::new(econfig.num_blocks, econfig.block_size),
            next_seq_id: 0,
            eos_token_id: match &config.eos_token_id {
                EosTokenId::Single(eos) => vec![*eos],
                EosTokenId::Multiple(eos) => eos.into_iter().map(|x| *x).collect(),
            },
            cfg: econfig.clone(),
            cached_seqs: VecDeque::new(),
        }
    }

    pub fn add(&mut self, mut seq: Sequence) -> usize {
        seq.id = self.next_seq_id;
        let id = seq.id;
        self.next_seq_id += 1;
        self.waiting.push_back(seq);
        id
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
                || num_tokens + seq.len() - seq.num_cached_tokens > self.cfg.max_num_batched_tokens
                || (seq.block_table.is_empty() && !self.block_manager.can_allocate(&seq))
            {
                // Put it back and break out if cannot schedule more
                self.waiting.push_front(seq);
                break;
            }

            if seq.block_table.is_empty() {
                self.block_manager.allocate(&mut seq);
            }
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

    pub fn get_running(&self, idx: usize) -> Option<&Sequence> {
        if idx < self.running.len() {
            Some(&self.running[idx])
        } else {
            None
        }
    }

    pub fn get_waiting(&self, idx: usize) -> Option<&Sequence> {
        if idx < self.waiting.len() {
            Some(&self.waiting[idx])
        } else {
            None
        }
    }

    /// Postprocess output tokens and modify sequences by indexes
    pub fn postprocess(
        &mut self,
        ids: &[usize],
        output_ids: &[u32],
        active_sessions: &VecDeque<(usize, String)>,
    ) {
        for (i, &idx) in ids.iter().enumerate() {
            let seq = &mut self.running[idx];
            let token = output_ids[i];
            seq.append_token(token);

            if self.eos_token_id.contains(&token)
                || seq.output_len() >= seq.sampling_params.max_tokens
                || seq.len() > self.cfg.max_num_batched_tokens
            {
                if let Some((_, v)) = active_sessions.iter().find(|(k, _)| *k == seq.id) {
                    seq.status = SequenceStatus::Cached;
                    seq.num_cached_tokens = seq.len();
                    self.cached_seqs.push_back((seq.id, v.clone()));
                    crate::log_warn!("\n\nSeq {} marked as cached (session_id {})", seq.id, v);
                } else {
                    seq.status = SequenceStatus::Finished;
                    self.block_manager.deallocate(seq);
                }
            }
        }
    }

    pub fn has_cache(&self, session_id: &String) -> bool {
        self.cached_seqs.iter().any(|(_, v)| v == session_id)
    }

    pub fn get_cache(&mut self, session_id: &String, new_tokens_ids: Vec<u32>) -> Result<usize> {
        if let Some((seq_id, _)) = self.cached_seqs.iter().find(|(_, v)| v == session_id) {
            for i in 0..self.waiting.len() {
                let seq: &mut Sequence = &mut self.waiting[i];
                if seq.id == *seq_id {
                    crate::log_warn!(
                        "\nSeq {} continued with {} new tokens (cached {} tokens, session_id {})",
                        seq.id,
                        new_tokens_ids.len(),
                        seq.token_ids.len(),
                        session_id.clone()
                    );
                    seq.token_ids.extend(new_tokens_ids.clone());
                    seq.status = SequenceStatus::Waiting;
                    seq.output_ids.clear();
                    if self.block_manager.can_append(seq) {
                        self.block_manager.ensure_allocate(seq);
                    } else {
                        seq.status = SequenceStatus::Finished;
                        crate::log_info!("Not enough space for seq {}", seq.id);
                    }
                    break;
                }
            }
            Ok(*seq_id)
        } else {
            candle_core::bail!("Cache for session {} not found!", session_id);
        }
    }

    pub fn clear_finished(&mut self) {
        let mut remove_ids = Vec::new();
        for i in 0..self.running.len() {
            let seq: &mut Sequence = &mut self.running[i];
            if seq.status != SequenceStatus::Cached {
                continue;
            }
            //push cached sessions back to waiting list
            let mut seq = seq.clone();
            // crate::log_info!("clear_finished {:?}", seq);
            seq.output_ids.clear();
            // seq.num_cached_tokens = seq.token_ids.len();
            remove_ids.push(seq.id);
            self.waiting.push_back(seq);
        }
        self.running.retain(|s| !remove_ids.contains(&s.id));

        // Remove finished sequences from running vector
        self.running
            .retain(|seq| seq.status != SequenceStatus::Finished);
        self.waiting
            .retain(|seq| seq.status != SequenceStatus::Finished);
    }

    pub fn release_all_waitings(&mut self) -> Vec<usize> {
        // Release all waiting sequences since there are no more resources (kv cache)
        let mut decode_ids = Vec::new();
        for i in 0..self.waiting.len() {
            let seq = &mut self.waiting[i];
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(seq);
            decode_ids.push(i);
        }
        self.cached_seqs.clear();
        // println!("{} waiting sequences released!", decode_ids.len());
        // assert!(decode_ids.len() > 0, "no more waiting");
        decode_ids
    }

    pub fn cancel(&mut self, seq_id: usize) {
        for i in 0..self.running.len() {
            let seq = &mut self.running[i];
            if seq.id == seq_id {
                seq.status = SequenceStatus::Finished;
                self.block_manager.deallocate(seq);
                break;
            }
        }
    }

    pub fn filter_prefill_finished(
        &mut self,
        scheduled_ids: &Vec<usize>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut finished_seqs = Vec::new();
        let mut remove_ids = Vec::new();
        const CHUNK_SIZE: usize = 8192;
        for (i, id) in scheduled_ids.iter().enumerate() {
            if *id < self.running.len() {
                let seq = &self.running[*id];
                if seq.len() < CHUNK_SIZE || seq.num_cached_tokens + CHUNK_SIZE >= seq.len() {
                    finished_seqs.push((i, seq.id));
                } else {
                    remove_ids.push(seq.id);
                    //unfinished due to chunked_prefill, push back to waiting list
                    let mut seq = seq.clone();
                    seq.num_cached_tokens += CHUNK_SIZE; //current prefilled CHUNK_SIZE
                    seq.status = SequenceStatus::Waiting;
                    crate::log_warn!(
                        "seq {} chunk prefilled {} (remain {} tokens)",
                        seq.id,
                        seq.num_cached_tokens,
                        seq.len() - seq.num_cached_tokens
                    );
                    self.waiting.push_back(seq);
                }
            }
        }
        self.running.retain(|s| !remove_ids.contains(&s.id));
        let (indices, finished_ids): (Vec<usize>, Vec<usize>) = finished_seqs.into_iter().unzip();
        let finished_indices: Vec<usize> = finished_ids
            .iter()
            .filter_map(|&target_id| self.running.iter().position(|seq| seq.id == target_id))
            .collect();
        (indices, finished_indices)
    }
}
