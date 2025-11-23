// src/core/scheduler.rs
use super::runner::RunnerType;
use super::{
    block_manager::BlockManager,
    sequence::{Sequence, SequenceStatus},
};
use crate::transfer::{PdConfig, PdRole};
use crate::utils::config::{Config, EngineConfig, EosTokenId};
use candle_core::Result;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    cached: Vec<Sequence>,
    transferred: VecDeque<Sequence>,
    pub block_manager: BlockManager,
    next_seq_id: usize,
    eos_token_id: Vec<u32>,
    cfg: EngineConfig,
    cached_seqs: VecDeque<(usize, String)>,
    pd_config: Option<PdConfig>,
}

const MIN_NUM_SCHEDULED_REQS: usize = 5;
pub const KVCACHE_SWAP_THRESHOLD: f32 = 0.95f32; // over 95%
const SWAP_COOLING_PERIOD: usize = 5000; // 5 seconds cooling time to prevent frequent swap out/in
const MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP: usize = 1000; // to swap-in, at least 1000 kvcache tokens left for decoding
pub const PD_PREFILL_STATUS_CHECK_COOLING_PERIOD: usize = 500; // check prefill status on PD server every 1 second
pub const PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD: usize = 128; // do not transfer prefill length < 128

impl Scheduler {
    pub fn new(runners: Arc<RwLock<RunnerType>>, econfig: &EngineConfig, config: &Config) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            cached: Vec::new(),
            transferred: VecDeque::new(),
            block_manager: BlockManager::new(
                runners,
                econfig.num_blocks,
                (econfig.cpu_mem_fold.unwrap_or(1.0f32) * econfig.num_blocks as f32) as usize,
                econfig.block_size,
            ),
            next_seq_id: 0,
            eos_token_id: match &config.eos_token_id {
                EosTokenId::Single(eos) => vec![*eos],
                EosTokenId::Multiple(eos) => eos.into_iter().map(|x| *x).collect(),
            },
            cfg: econfig.clone(),
            cached_seqs: VecDeque::new(),
            pd_config: econfig.pd_config.clone(),
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
    pub fn schedule(&mut self) -> Result<(Vec<usize>, bool)> {
        let mut scheduled_ids = Vec::new();
        let mut num_tokens = 0;

        // PD server: Check for new incoming prefill requests
        if self.is_pd_server() {
            if let Ok((fit, Some(seq))) = self
                .block_manager
                .try_receive_prefill(self.get_available_kv_tokens())
            {
                let seq_id = seq.id;
                if !fit {
                    crate::log_warn!(
                        "Prefill request (Seq {}) enter pending status because it require {} KvCache tokens (left {}).",
                        seq_id,
                        seq.len() + 1,
                        self.get_available_kv_tokens(),
                    );
                } else {
                    crate::log_warn!(
                        "Prefill request (Seq {}, {} tokens) received from PD client.",
                        seq_id,
                        seq.len(),
                    );
                }
                // Add to waiting queue.
                self.waiting.push_back(seq);
            }
        }

        // Prefill phase: move sequences from waiting to running if possible
        while let Some(mut seq) = self.waiting.pop_front() {
            // We do not transfer context-cache request
            if self.is_pd_mode() && !self.is_pd_server() && self.try_transfer(&mut seq) {
                break;
            }

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

        // Client: Check for finished prefills
        if self.is_pd_mode() && !self.is_pd_server() {
            self.try_receive_kvcache()?;
        }

        // Swap back seq from cpu memory if possible
        #[cfg(feature = "cuda")]
        if preempt_ids.is_empty()
            && (self.kv_cache_usage_percent() < KVCACHE_SWAP_THRESHOLD * 0.9
                || (self.running.is_empty() && self.kv_cache_usage_percent() <= 0.3f32))
        {
            self.try_swap_in();
        } else if !preempt_ids.is_empty() && self.running.len() > 1 {
            // Requests unable to be processed at the current moment
            // If we only have one sequence running and it has been preempt,
            // swap out to cpu memory make non-sense
            // in such case, the only option is either waiting resources or abort it
            if let Some((idx, _)) = preempt_ids
                .iter()
                .map(|&i| (i, &self.running[i]))
                .min_by_key(|(_, seq)| seq.id)
            // swap-out the oldest sequence
            {
                self.try_swap_out(idx, true);
            }
        }

        let is_pd_server = self.is_pd_server();
        for (idx, seq) in self.running.iter_mut().enumerate() {
            if decode_ids.len() >= std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS) {
                break;
            }
            if !self.block_manager.can_append(&seq) {
                // filter out seq that unable to acquire resources
                continue;
            }
            if is_pd_server && seq.status == SequenceStatus::Cached {
                if let Ok(success) = self.block_manager.try_check_kvcache_release(seq.id) {
                    if success {
                        // Client successfully received kvcache and we need to release it on the server
                        crate::log_warn!("PD Server: release prefilled kvcache for Seq {}", seq.id);
                        seq.status = SequenceStatus::Finished;
                        self.block_manager.deallocate(seq);
                    }
                }
                // in PD server mode, we do not decode, filter out seq have been prefilled
                continue;
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

    pub fn get_seq_token_usage(&self, seq_id: usize) -> Result<usize> {
        // search waiting
        if let Some(item) = self.waiting.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // search running
        if let Some(item) = self.running.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // search cached
        if let Some(item) = self.cached.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // if nothing found
        Ok(0)
    }

    /// Postprocess output tokens and modify sequences by indexes
    pub fn postprocess(
        &mut self,
        ids: &[usize],
        output_ids: &[u32],
        active_sessions: &VecDeque<(usize, String)>,
    ) {
        for (i, &idx) in ids.iter().enumerate() {
            // Sequence may swapped out
            if idx >= self.running.len() {
                continue;
            }
            let seq_id = self.running[idx].id;
            let token = output_ids[i];

            // Since all reqeusts in PD server are prefill request, we need to finish and transfer
            // the kvcache in the first postprocess for each request.
            if self.is_pd_server() {
                match self
                    .block_manager
                    .try_send_kvcache(&self.running[idx], token)
                {
                    Ok(success) => {
                        crate::log_warn!(
                            "PD Server: transferred KV cache for seq {} ({})",
                            seq_id,
                            if success { "success" } else { "faild" }
                        );
                        let seq = &mut self.running[idx];
                        if success {
                            // if successed, we need to mantain the resources
                            // until the client ask explicitly to release or cache not sufficient
                            seq.status = SequenceStatus::Cached;
                            let cur_time = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time went backwards")
                                .as_millis() as usize;
                            let time_costs = cur_time - seq.created_time();
                            if time_costs / 100 > 0 && seq.len() > 0 {
                                crate::log_info!(
                                    "PD Prefilling [seq_id {}]: {} tokens in {:.2}s ({:.2} tokens/s)",
                                    seq_id,
                                    seq.len(),
                                    time_costs as f32 / 1000f32,
                                    seq.len() as f32 / (time_costs as f32 * 1.0f32 / 1000f32),
                                )
                            }
                        } else {
                            // release resources immediately if failed
                            seq.status = SequenceStatus::Finished;
                            self.block_manager.deallocate(seq);
                        }
                    }
                    Err(e) => {
                        crate::log_error!(
                            "PD Server: failed to transfer KV cache for seq {}: {}",
                            seq_id,
                            e
                        );
                        let seq = &mut self.running[idx];
                        seq.status = SequenceStatus::Finished;
                        self.block_manager.deallocate(seq);
                    }
                }

                continue; // Go to next sequence in postprocess
            }

            let seq = &mut self.running[idx];
            seq.append_token(token);

            if self.eos_token_id.contains(&token)
                || seq.output_len() >= seq.sampling_params.max_tokens.unwrap_or(16384)
                || seq.len() > self.cfg.max_num_batched_tokens
            {
                if let Some((_, v)) = active_sessions.iter().find(|(k, _)| *k == seq.id) {
                    self.swap_out_or_cache(idx, v.clone());
                } else {
                    // Resources for non context-cache requests will be removed immediately when finished
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
        let seq_map_entry = self
            .cached_seqs
            .iter()
            .find(|(_, v)| v == session_id)
            .map(|(id, _)| *id);
        if let Some(target_seq_id) = seq_map_entry {
            let cache_index = self.cached.iter().position(|s| s.id == target_seq_id);
            if let Some(i) = cache_index {
                // Found it in GPU memory
                crate::log_info!(
                    "\nSeq {} - continued with {} new tokens ({} cached tokens, session_id {})\n",
                    target_seq_id,
                    new_tokens_ids.len(),
                    self.cached[i].token_ids.len(),
                    session_id.clone()
                );
                let mut seq = self.cached.remove(i);
                if seq.status == SequenceStatus::Swapped && !self.block_manager.can_swap_in(&seq) {
                    self.cached.push(seq);
                    candle_core::bail!("Seq {} swapped out but currently no resources to swap in for execution, please request later!", target_seq_id);
                }

                seq.token_ids.extend(new_tokens_ids.clone());
                seq.output_ids.clear();
                //in context-cache, we dont' recreate sequences, so we need to update created_time
                seq.created_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize;

                let mut failed_reason = None;
                if let Err(e) = self.block_manager.ensure_allocate(&mut seq) {
                    failed_reason = Some(e);
                }

                // Swap in data from CPU (if swapped previously)
                if seq.status == SequenceStatus::Swapped && failed_reason.is_none() {
                    if let Err(e) = self.block_manager.swap_in(&mut seq) {
                        // swap in failed: mark finished and free gpu blocks
                        crate::log_warn!("Failed to swap in seq {}: {:?}", seq.id, e);
                        failed_reason = Some(e);
                    }
                }

                if let Some(e) = failed_reason {
                    seq.status = SequenceStatus::Finished;
                    self.block_manager.deallocate(&seq);
                    if let Some(pos) = self
                        .cached_seqs
                        .iter()
                        .position(|(id, _)| *id == target_seq_id)
                    {
                        self.cached_seqs.remove(pos);
                    }
                    candle_core::bail!("{:?}", e);
                }
                seq.status = SequenceStatus::Waiting; //active this sequence (from cached/swaped in to waiting)
                self.waiting.push_back(seq.clone());
                Ok(target_seq_id)
            } else {
                // Found in mapping, but missing in memory (Pollution/Eviction Desync)
                // Remove the stale mapping
                if let Some(pos) = self
                    .cached_seqs
                    .iter()
                    .position(|(id, _)| *id == target_seq_id)
                {
                    self.cached_seqs.remove(pos);
                }
                candle_core::bail!(
                    "Cache inconsistency: Session {} mapped to seq {} but data is missing.",
                    session_id,
                    target_seq_id
                );
            }
        } else {
            candle_core::bail!("Cache for session {} not found!", session_id);
        }
    }

    pub fn clear_finished(&mut self) {
        let is_pd_server = self.is_pd_server();
        let mut remove_ids = Vec::new();
        for i in 0..self.running.len() {
            let seq: &mut Sequence = &mut self.running[i];
            if !is_pd_server && seq.status == SequenceStatus::Cached {
                seq.output_ids.clear();
                remove_ids.push(seq.id);
                self.cached.push(seq.clone());
            }
            if is_pd_server && seq.status == SequenceStatus::Finished {
                self.print_free_blocks();
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
            // free gpu blocks and also free any CPU swap space
            self.block_manager.free_cpu_swap_for_seq(seq.id);
        }
        self.cached.clear();
        self.cached_seqs.clear();
    }

    pub fn release_cache(&mut self, seq_id: usize) {
        if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
            let mut seq = self.cached.remove(pos);
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(&seq);
            // also free cpu swap
            if seq.status == SequenceStatus::Swapped {
                self.block_manager.free_cpu_swap_for_seq(seq_id);
            }
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

    pub fn try_transfer(&mut self, seq: &mut Sequence) -> bool {
        if !self.is_suitable_for_transfer(&seq) {
            return false;
        }
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        if let Some(tm) = seq.swapped_time {
            if cur_time - tm < SWAP_COOLING_PERIOD {
                return false;
            }
        }
        if let Ok(success) = self.block_manager.try_transfer_prefill(&seq) {
            // Client: Offload prefill request to PD server
            seq.swapped_time = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize,
            );
            if success {
                crate::log_warn!(
                    "Prefill request (Seq {}, {} tokens) transfered to PD server.",
                    seq.id,
                    seq.len(),
                );
                seq.pd_first_token = None;
                self.transferred.push_back(seq.clone());
            } else {
                crate::log_warn!(
                    "Unable transfer prefill request (Seq {}) to PD server. Retry later...",
                    seq.id
                );
                self.waiting.push_front(seq.clone()); // push back, retry later
            }
            success
        } else {
            false
        }
    }

    pub fn try_swap_in(&mut self) {
        let available_kvcache_tokens = self.get_available_kv_tokens();
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        for i in 0..self.cached.len() {
            let seq = &self.cached[i];
            if seq.status != SequenceStatus::Swapped
                || cur_time - seq.swapped_time().unwrap_or(cur_time) < SWAP_COOLING_PERIOD
            {
                continue;
            }

            if !self.block_manager.can_swap_in(&seq)
                || (available_kvcache_tokens - seq.len() < MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP)
            {
                if !self.running.is_empty() {
                    // Wait for swap in
                    continue;
                }

                // KvCache not sufficent, try if we can swap-out cached seq
                if let Some((pos, s)) = self
                    .cached
                    .iter()
                    .enumerate()
                    .find(|(_, s)| s.status == SequenceStatus::Cached)
                {
                    crate::log_warn!("Insufficient KvCache, trying to swap out cached Seq {} for swapping in Seq {}!", s.id, seq.id);
                    self.try_swap_out(pos, true);
                    break;
                }

                let mut seq = self.cached.remove(i);
                seq.status = SequenceStatus::Finished;
                crate::log_error!("No KvCache left for swap in Seq {}!", seq.id);
                self.running.push(seq);
                break;
            }

            let mut seq = self.cached.remove(i);
            seq.swapped_time = Some(cur_time);
            seq.clear_block_table(); //reallocate block table since previous gpu blocks were freed

            if let Err(_) = self.block_manager.ensure_allocate(&mut seq) {
                continue;
            }

            // Swap in data from CPU (if swapped out previously)
            match self.block_manager.swap_in(&mut seq) {
                Ok(_) => {
                    seq.status = SequenceStatus::Running;
                    crate::log_warn!("Seq {} is swapped in for execution!", seq.id);
                }
                Err(e) => {
                    seq.status = SequenceStatus::Finished;
                    crate::log_error!("Seq {} swap in failed: {:?}!", seq.id, e);
                }
            }
            self.running.push(seq);
            break;
        }
    }

    // swap out one sequence a time
    pub fn try_swap_out(&mut self, idx: usize, is_running: bool) {
        if (is_running && idx >= self.running.len()) || (!is_running && idx >= self.cached.len()) {
            return;
        }

        let mut seq = if is_running {
            &mut self.running[idx]
        } else {
            &mut self.cached[idx]
        };

        // If sequence has blocks, attempt to swap to CPU.
        // If cannot swap, fallback.
        if !seq.block_table.is_empty()
            && seq.output_len() > 0
            && (seq.status == SequenceStatus::Running || seq.status == SequenceStatus::Cached)
            && self.block_manager.can_swap_out(&seq)
        {
            // make sure we have identical number of blocks when swapping in
            // for decoding
            if let Err(_) = self.block_manager.ensure_allocate(&mut seq) {
                return;
            }
            match self.block_manager.swap_out(&mut seq) {
                Ok(_) => {
                    let mut seq = if is_running {
                        self.running.remove(idx)
                    } else {
                        // Even though the cached sequence swapped out,
                        // no need remove it from cached list since it can be recoved
                        self.cached.remove(idx)
                    };
                    seq.status = SequenceStatus::Swapped;
                    // seq.num_cached_tokens = seq.len();
                    self.block_manager.deallocate(&seq);
                    // block table need to be reallocated when swapping in
                    self.cached.push(seq.clone());
                }
                Err(e) => {
                    crate::log_warn!("Swap out failed for seq {}: {:?}", seq.id, e);
                }
            }
        }
    }

    pub fn swap_out_or_cache(&mut self, idx: usize, v: String) {
        let kvcache_usage_percentage = self.kv_cache_usage_percent();
        let mut seq = &mut self.running[idx];
        if kvcache_usage_percentage > KVCACHE_SWAP_THRESHOLD
            && !seq.block_table.is_empty()
            && self.block_manager.can_swap_out(seq)
            && self.block_manager.ensure_allocate(&mut seq).is_ok()
        {
            // Reach the kvcache threashold and we have cpu memory to swap out
            match self.block_manager.swap_out(seq) {
                Ok(_) => {
                    let mut seq = self.running.remove(idx);
                    seq.status = SequenceStatus::Swapped;
                    self.block_manager.deallocate(&seq);
                    // block table need to be reallocated when swapping in
                    self.cached.push(seq.clone());
                }
                Err(e) => {
                    crate::log_warn!("Failed to swap out seq {}: {:?}", seq.id, e);
                    seq.status = SequenceStatus::Finished;
                    self.block_manager.deallocate(seq);
                }
            }
        } else {
            // Sufficient GPU KV Cache, no need to swap, mark as cached in GPU memory
            seq.status = SequenceStatus::Cached;
            seq.num_cached_tokens = seq.len();
            if !self.cached_seqs.iter().any(|(_, v_)| v_ == &v) {
                self.cached_seqs.push_back((seq.id, v.clone()));
            }
        }
    }

    pub fn is_pd_mode(&self) -> bool {
        self.pd_config.is_some()
    }

    pub fn is_pd_server(&self) -> bool {
        if let Some(p_cfg) = &self.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        }
    }

    pub fn is_suitable_for_transfer(&self, seq: &Sequence) -> bool {
        if seq.status == SequenceStatus::Swapped // swapped out sequence
            || seq.len() < PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD // prefill length < 128
            || (self.cfg.flash_context.unwrap_or(false) // Context-cache request
            && self.cached_seqs.iter().position(|(id, _)| id == &seq.id).is_some())
        {
            false
        } else {
            true
        }
    }

    /// Client: Check for finished prefills and move them to the running queue.
    pub fn try_receive_kvcache(&mut self) -> Result<()> {
        let mut finished_seq_ids = Vec::new();
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        for idx in 0..self.transferred.len() {
            let seq_id = self.transferred[idx].id;
            if cur_time - self.transferred[idx].swapped_time().unwrap_or(cur_time)
                < PD_PREFILL_STATUS_CHECK_COOLING_PERIOD
            {
                continue;
            }

            self.transferred[idx].swapped_time = Some(cur_time);
            let status = self.block_manager.try_check_prefill_status(seq_id);
            if status.is_err() || !status.unwrap_or(false) {
                continue;
            }
            // We have the data. Can we allocate space for it?
            if !self.block_manager.can_allocate(&self.transferred[idx]) {
                // Not enough memory right now. Put data back and try later.

                // KvCache not sufficent, try if we can swap-out cached seq
                if let Some((pos, s)) = self
                    .cached
                    .iter()
                    .enumerate()
                    .find(|(_, s)| s.status == SequenceStatus::Cached)
                {
                    crate::log_warn!("Insufficient KvCache, trying to swap out cached Seq {} for receiving KvCache for Seq {}!", s.id, seq_id);
                    self.try_swap_out(pos, true);
                    break;
                }

                crate::log_warn!(
                    "KvCache Transfer: Seq {} prefill finished on PD server, but no blocks to receive. Will retry.",
                    seq_id
                );
                // For simplicity, we just break and retry next cycle.
                break;
            }

            // Allocate GPU blocks for the sequence
            self.block_manager.allocate(&mut self.transferred[idx])?;

            // Perform the actual KV cache data transfer
            let mut success = false;
            match self
                .block_manager
                .try_receive_kvcache(&self.transferred[idx])
            {
                Ok((ret, first_token, sending_time)) => {
                    let seq = &mut self.transferred[idx];
                    success = ret;
                    if success {
                        // Update sequence and move to running
                        // The first token is generated on PD server,
                        // it has been transfered to client, but haven't been send to user
                        seq.append_token(first_token);
                        seq.pd_first_token = Some(first_token);
                        seq.status = SequenceStatus::Running;
                        self.running.push(seq.clone());
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis() as usize;
                        let transfer_duration = now - sending_time;

                        if transfer_duration > 10000 {
                            // Log a warning when a sequence takes an unusually long time to receive and swap-in.
                            // Possible causes: insufficient KV cache on server or client, or low communication bandwidth.
                            crate::log_warn!(
                                "KvCache Transfer: Seq {} prefill finished, but receive (with swap-in) time was unexpectedly long ({} s).",
                                seq.id,
                                transfer_duration / 1000
                            );
                        } else {
                            crate::log_info!(
                                "KvCache Transfer: Seq {} prefill finished and received in {} ms!",
                                seq.id,
                                transfer_duration
                            );
                        };

                        // Since KVCache Transfer involved here, the prefill speed might not accurate
                        // since receive remote kvcache requires sufficient local cache memory (sometime pending for kvcache)
                        // The actual prefill speed need to exclude the transfer time, for simplicity we didn't do that

                        self.block_manager.try_release_remote_kvcache(seq.id)?;
                    } else {
                        crate::log_error!(
                            "KvCache Transfer: Seq {} prefill finished but failed to receive. Aborting seq.",
                            seq.id,
                        );
                    }
                }
                Err(e) => {
                    crate::log_error!(
                        "KvCache Transfer: Failed to receive KV cache for seq {}: {}. Aborting seq.",
                        seq_id, e
                    );
                }
            }

            let seq = &mut self.transferred[idx];
            if !success {
                seq.status = SequenceStatus::Finished; // Mark as failed
                self.running.push(seq.clone());
            }
            finished_seq_ids.push(seq.id);
        }

        // Remove all processed sequences from the transferred queue
        self.transferred
            .retain(|s| !finished_seq_ids.contains(&s.id));
        Ok(())
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
        let used_percent =
            100.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks as f32) * 100.0f32;
        let kvcache_memory_gb = self.cfg.kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        #[cfg(feature = "cuda")]
        let cpu_swap_log = {
            let cpu_kvcache_memory_gb = kvcache_memory_gb * self.cfg.cpu_mem_fold.unwrap_or(1.0f32);
            format!(
                "CPU swap used {:.1}% ({:.2}GB/{:.2}GB)",
                self.block_manager.get_cpu_swap_usage() * 100.0f32,
                self.block_manager.get_cpu_swap_usage() * cpu_kvcache_memory_gb,
                cpu_kvcache_memory_gb,
            )
        };
        #[cfg(not(feature = "cuda"))]
        let cpu_swap_log = "".to_string();

        crate::log_info!(
            "GPU Kvcache: {} blocks ({} tokens) free, used {:.1}% ({:.2}GB/{:.2}GB); {}",
            free_blocks,
            free_blocks * self.block_manager.get_block_size(),
            used_percent,
            used_percent / 100.0f32 * kvcache_memory_gb,
            kvcache_memory_gb,
            cpu_swap_log,
        );
    }

    pub fn kv_cache_usage_percent(&self) -> f32 {
        let total_blocks = self.block_manager.get_num_total_blocks();
        let free_blocks = self.block_manager.get_num_free_blocks();
        1.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks as f32)
    }
}
