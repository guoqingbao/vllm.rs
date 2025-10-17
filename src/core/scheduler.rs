// src/core/scheduler.rs
use super::{
    block_manager::BlockManager,
    sequence::{Sequence, SequenceStatus},
};
use crate::utils::config::{Config, EngineConfig, EosTokenId};
use candle_core::Result;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    cached: Vec<Sequence>,
    block_manager: BlockManager,
    next_seq_id: usize,
    eos_token_id: Vec<u32>,
    cfg: EngineConfig,
    cached_seqs: VecDeque<(usize, String)>,
}

const MIN_NUM_SCHEDULED_REQS: usize = 5;

impl Scheduler {
    pub fn new(econfig: &EngineConfig, config: &Config) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            cached: Vec::new(),
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
    pub fn schedule(&mut self) -> candle_core::Result<(Vec<usize>, bool)> {
        let mut scheduled_ids = Vec::new();
        let mut num_tokens = 0;

        // Prefill phase: move sequences from waiting to running if possible
        while let Some(mut seq) = self.waiting.pop_front() {
            if scheduled_ids.len() >= std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS)
                || num_tokens + seq.len() >= self.cfg.max_num_batched_tokens - 1
                || (seq.block_table.is_empty() && !self.block_manager.can_allocate(&seq))
            {
                // Put it back and break out if cannot schedule more
                self.waiting.push_front(seq);
                break;
            }

            if seq.block_table.is_empty() {
                self.block_manager.allocate(&mut seq)?;
            }
            seq.status = SequenceStatus::Running;
            num_tokens += seq.len();
            self.running.push(seq);
            scheduled_ids.push(self.running.len() - 1); // index of newly pushed seq
        }

        if !scheduled_ids.is_empty() {
            return Ok((scheduled_ids, true));
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
            if decode_ids.len() >= std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS) {
                break;
            }
            self.block_manager.may_append(seq)?;
            decode_ids.push(idx);
        }

        Ok((decode_ids, false))
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
                || seq.output_len() >= seq.sampling_params.max_tokens.unwrap_or(16384)
                || seq.len() > self.cfg.max_num_batched_tokens
            {
                if let Some((_, v)) = active_sessions.iter().find(|(k, _)| *k == seq.id) {
                    seq.status = SequenceStatus::Cached;
                    seq.num_cached_tokens = seq.len();
                    if !self.cached_seqs.iter().any(|(_, v)| v == v) {
                        self.cached_seqs.push_back((seq.id, v.clone()));
                    }
                    // crate::log_info!(
                    //     "\n\nSeq {} - {} tokens cached (session_id {})",
                    //     seq.id,
                    //     seq.num_cached_tokens,
                    //     v
                    // );
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
            for i in 0..self.cached.len() {
                if self.cached[i].id == *seq_id {
                    crate::log_info!(
                        "\nSeq {} - continued with {} new tokens ({} cached tokens, session_id {})\n",
                        seq_id,
                        new_tokens_ids.len(),
                        self.cached[i].token_ids.len(),
                        session_id.clone()
                    );
                    let mut seq = self.cached.remove(i);
                    seq.token_ids.extend(new_tokens_ids.clone());
                    seq.status = SequenceStatus::Waiting; //active ths sequence (from cached to waiting)
                    seq.output_ids.clear();
                    //in context-cache, we dont' recreate sequences, so we need to update created_time
                    seq.created_time = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_millis() as usize;
                    if let Err(e) = self.block_manager.ensure_allocate(&mut seq) {
                        seq.status = SequenceStatus::Finished;
                        self.block_manager.deallocate(&seq);
                        if let Some(pos) = self.cached_seqs.iter().position(|(id, _)| id == seq_id)
                        {
                            self.cached_seqs.remove(pos);
                        }
                        candle_core::bail!("{:?}", e);
                    }
                    self.waiting.push_back(seq.clone());
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
            if seq.status == SequenceStatus::Cached {
                seq.output_ids.clear();
                remove_ids.push(seq.id);
                self.cached.push(seq.clone());
            }
        }
        self.running.retain(|s| !remove_ids.contains(&s.id));

        // Remove finished sequences from running vector
        self.running
            .retain(|seq| seq.status != SequenceStatus::Finished);
        self.waiting
            .retain(|seq| seq.status != SequenceStatus::Finished);
    }

    pub fn release_waitings(&mut self) {
        // Release all waiting sequences since there are no more resources (kv cache)
        let mut decode_ids = Vec::new();
        for i in 0..self.waiting.len() {
            let seq = &mut self.waiting[i];
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(seq);
            decode_ids.push(i);
        }
        self.waiting.clear();
        for i in 0..self.cached.len() {
            let seq = &mut self.cached[i];
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(seq);
        }
        self.cached.clear();
        self.cached_seqs.clear();
    }

    pub fn release_cache(&mut self, seq_id: usize) {
        if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
            let mut seq = self.cached.remove(pos);
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(&seq);
        }
        if let Some(pos) = self.cached_seqs.iter().position(|(id, _)| *id == seq_id) {
            self.cached_seqs.remove(pos);
        }
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
        self.release_cache(seq_id);
        self.running.retain(|seq| seq.id != seq_id);
        self.waiting.retain(|seq| seq.id != seq_id);
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
                    if seq.len() > CHUNK_SIZE {
                        crate::log_warn!(
                            "Seq {} - chunk prefill finished ({} tokens)",
                            seq.id,
                            seq.len()
                        );
                    }
                    finished_seqs.push((i, seq.id));
                } else {
                    remove_ids.push(seq.id);
                    //unfinished due to chunked_prefill, push back to waiting list
                    let mut seq = seq.clone();
                    seq.num_cached_tokens += CHUNK_SIZE; //current prefilled CHUNK_SIZE
                    seq.status = SequenceStatus::Waiting;
                    crate::log_warn!(
                        "Seq {} - chunk prefilled {} (remain {} tokens)",
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

    pub fn get_num_cached_tokens(&self) -> usize {
        let mut num_cached_tokens = 0;
        for i in 0..self.cached.len() {
            num_cached_tokens += self.cached[i].num_cached_tokens;
        }
        for i in 0..self.running.len() {
            num_cached_tokens += self.running[i].num_cached_tokens;
        }
        num_cached_tokens
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let free_blocks = self.block_manager.get_num_free_blocks();
        free_blocks * self.block_manager.get_block_size()
    }

    pub fn print_free_blocks(&self) {
        const SIZE_IN_GB: usize = 1024 * 1024 * 1024;
        let total_blocks = self.block_manager.get_num_total_blocks();
        let free_blocks = self.block_manager.get_num_free_blocks();
        let used_percent = 1.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks as f32);
        let kvcache_memory_gb = self.cfg.kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        crate::log_info!(
            "Kvcache: {} blocks ({} tokens) free, used {:.1}% ({:.2}GB/{:.1}GB)",
            free_blocks,
            free_blocks * self.block_manager.get_block_size(),
            used_percent * 100.0f32,
            used_percent * kvcache_memory_gb,
            kvcache_memory_gb,
        );
    }

    pub fn kv_cache_usage_percent(&self) -> f32 {
        let total_blocks = self.block_manager.get_num_total_blocks();
        let free_blocks = self.block_manager.get_num_free_blocks();
        100.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks as f32) * 100.0f32
    }
}
