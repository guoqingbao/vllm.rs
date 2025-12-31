// src/core/scheduler.rs
use super::runner::RunnerType;
use super::{
    block_manager::BlockManager,
    prefix_cache::PrefixCacheConfig,
    sequence::{Sequence, SequenceStatus},
};
use crate::transfer::{PdConfig, PdRole};
use crate::utils::config::{Config, EngineConfig, EosTokenId};
use candle_core::Result;
use parking_lot::RwLock;
use regex::Regex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    cached: Vec<Sequence>,
    transferred: VecDeque<Sequence>,
    pub block_manager: BlockManager,
    next_seq_id: usize,
    eos_token_id: Vec<u32>,
    /// Token IDs that represent the end of a tool call (e.g., </tool_call> tokens)
    tool_call_end_token_ids: Vec<u32>,
    /// Token ID for } character (used for JSON tool call detection)
    json_end_token_id: Option<u32>,
    /// Tokenizer for decoding output to check JSON tool call patterns
    tokenizer: Option<Arc<Tokenizer>>,
    /// Regex for detecting JSON tool calls
    tool_call_regex: Regex,
    cfg: EngineConfig,
    pd_config: Option<PdConfig>,
    is_last_prefill: bool,
}

const MIN_NUM_SCHEDULED_REQS: usize = 5;
pub const KVCACHE_SWAP_THRESHOLD: f32 = 0.95f32; // over 95%
const SWAP_COOLING_PERIOD: usize = 5000; // 5 seconds cooling time to prevent frequent swap out/in
const MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP: usize = 1000; // to swap-in, at least 1000 kvcache tokens left for decoding
pub const PD_PREFILL_STATUS_CHECK_COOLING_PERIOD: usize = 500; // check prefill status on PD server every 1 second
pub const PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD: usize = 128; // do not transfer prefill length < 128

fn build_prefix_cache_config(econfig: &EngineConfig) -> PrefixCacheConfig {
    let enabled = econfig.prefix_cache.unwrap_or(false);
    if !enabled {
        return PrefixCacheConfig {
            enabled: false,
            max_cached_blocks: 0,
        };
    }

    let mut max_cached_blocks = if let Some(max_tokens) = econfig.prefix_cache_max_tokens {
        max_tokens / econfig.block_size
    } else {
        ((econfig.num_blocks as f32) * 0.25f32) as usize
    };

    if max_cached_blocks > econfig.num_blocks {
        max_cached_blocks = econfig.num_blocks;
    }

    if max_cached_blocks == 0 {
        crate::log_warn!("Prefix cache enabled but max cached blocks is 0; disabling.");
        return PrefixCacheConfig {
            enabled: false,
            max_cached_blocks: 0,
        };
    }

    crate::log_warn!(
        "Prefix cache enabled: {} blocks ({} tokens).",
        max_cached_blocks,
        max_cached_blocks * econfig.block_size
    );

    PrefixCacheConfig {
        enabled: true,
        max_cached_blocks,
    }
}

impl Scheduler {
    pub fn new(runners: Arc<RwLock<RunnerType>>, econfig: &EngineConfig, config: &Config) -> Self {
        let prefix_cache_cfg = build_prefix_cache_config(econfig);
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            cached: Vec::new(),
            transferred: VecDeque::new(),
            block_manager: BlockManager::new(
                runners,
                econfig.num_blocks,
                (econfig.cpu_mem_fold.unwrap_or(0.5f32) * econfig.num_blocks as f32) as usize,
                econfig.block_size,
                prefix_cache_cfg,
            ),
            next_seq_id: 0,
            eos_token_id: match &config.eos_token_id {
                Some(EosTokenId::Single(eos)) => vec![*eos],
                Some(EosTokenId::Multiple(eos)) => eos.into_iter().map(|x| *x).collect(),
                _ => vec![],
            },
            // Tool call end tokens will be set by engine after tokenizer is initialized
            tool_call_end_token_ids: Vec::new(),
            json_end_token_id: None,
            tokenizer: None,
            // Regex to match JSON tool call format: {"name": "...", "arguments": {...}}
            // We use (?s) to allow dot matching newlines
            tool_call_regex: Regex::new(r#"(?s)\{\s*"name"\s*:.*"arguments"\s*:.*\}\s*$"#).unwrap(),
            cfg: econfig.clone(),
            pd_config: econfig.pd_config.clone(),
            is_last_prefill: false,
        }
    }

    /// Set tool call end token IDs (called by engine after tokenizer is available)
    pub fn set_tool_call_end_tokens(&mut self, token_ids: Vec<u32>) {
        self.tool_call_end_token_ids = token_ids;
    }

    /// Set tokenizer for JSON tool call detection (called by engine after initialization)
    pub fn set_tokenizer(&mut self, tokenizer: Arc<Tokenizer>) {
        // Get the token ID for "}" character
        if let Ok(tokens) = tokenizer.encode("}", false) {
            if let Some(&token_id) = tokens.get_ids().last() {
                self.json_end_token_id = Some(token_id);
                crate::log_info!("JSON end token ID (}}) set to: {}", token_id);
            }
        }
        self.tokenizer = Some(tokenizer);
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
            // Try to transfer prefill requests to PD server when applicable
            if self.is_pd_mode() && !self.is_pd_server() && self.try_transfer(&mut seq) {
                break;
            }

            if scheduled_ids.len() >= std::cmp::max(self.cfg.max_num_seqs, MIN_NUM_SCHEDULED_REQS)
                || num_tokens + seq.len() >= self.cfg.max_num_batched_tokens - 1
                || (seq.block_table.is_empty() && !self.block_manager.can_allocate(&seq))
                // interleaved scheduling
                || (self.is_last_prefill && self.running.len() > 0)
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
            self.is_last_prefill = true;
            return Ok((scheduled_ids, true));
        }

        // Decode phase: pick sequences from running for decoding (up to max_num_seqs)
        let mut decode_ids = Vec::new();
        let mut preempt_ids = Vec::new();

        for (idx, seq) in self.running.iter().enumerate() {
            if !self.block_manager.can_append(seq)
                && seq.status != SequenceStatus::Swapped
                && seq.status != SequenceStatus::FinishSwapped
            {
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
        } else if !preempt_ids.is_empty() || self.kv_cache_usage_percent() > KVCACHE_SWAP_THRESHOLD
        {
            // Requests unable to be processed at the current moment
            // If we only have one sequence running and it has been preempt,
            // swap out to cpu memory make non-sense
            // in such case, the only option is either waiting resources or abort it
            let evicted = self.block_manager.evict_prefix_cache_blocks(1);
            if evicted > 0 {
                crate::log_warn!("Evicted {} prefix cache block(s) under pressure.", evicted);
            } else if !preempt_ids.is_empty() && self.running.len() > 1 {
                if let Some((idx, _)) = preempt_ids
                    .iter()
                    .map(|&i| (i, &self.running[i]))
                    .min_by_key(|(_, seq)| seq.id)
                // swap-out the oldest sequence
                {
                    crate::log_warn!("Trying to swap out preempt Seq {:?}", self.running[idx].id);
                    self.try_swap_out(idx, true);
                }
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

        self.is_last_prefill = false;
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

    pub fn find_seq_by_session_id(&self, session_id: &str) -> Option<(usize, SequenceStatus)> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.cached.iter())
            .find(|seq| seq.sampling_params.session_id.as_deref() == Some(session_id))
            .map(|seq| (seq.id, seq.status))
    }

    /// Postprocess output tokens and modify sequences by indexes
    pub fn postprocess(&mut self, ids: &[usize], output_ids: &[u32]) {
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

            // Check for tool call end token BEFORE checking EOS
            // When tools are enabled (mcp_mode.is_some()), finish stream at </tool_call>
            if self.running[idx].sampling_params.mcp_mode.is_some() {
                // Check if this is a tool call end (supports both XML </tool_call> and JSON } patterns)
                // We check BEFORE borrowing seq mutably
                let is_end = self.is_tool_call_end(token, idx);
                if is_end {
                    crate::log_info!(
                        "[Seq {}] Detected </tool_call> token {}, finishing for external handling",
                        seq_id,
                        token
                    );
                    let seq = &mut self.running[idx];
                    seq.append_token(token);
                    seq.is_tool_call_end = true;
                    // External tool mode: finish stream so client can handle tool calls
                    seq.status = SequenceStatus::Finished;
                    self.block_manager.cache_sequence(seq);
                    self.block_manager.deallocate(seq);
                    continue;
                }
            }

            let seq = &mut self.running[idx];

            if self.eos_token_id.contains(&token)
                || seq.output_len() >= seq.sampling_params.max_tokens.unwrap_or(16384)
                || seq.len() > self.cfg.max_num_batched_tokens
            {
                seq.status = SequenceStatus::Finished;
                self.block_manager.cache_sequence(seq);
                self.block_manager.deallocate(seq);
            } else {
                seq.append_token(token);
            }
        }
    }

    pub fn clear_finished(&mut self) {
        let is_pd_server = self.is_pd_server();
        for seq in &self.running {
            if seq.status == SequenceStatus::Finished && is_pd_server {
                self.print_free_blocks();
            }
        }
        self.running
            .retain(|seq| seq.status != SequenceStatus::Finished);
        self.waiting
            .retain(|seq| seq.status != SequenceStatus::Finished);
    }

    pub fn release_waitings(&mut self) {
        // Release all waiting sequences since there are no more resources (kv cache)
        for i in 0..self.waiting.len() {
            let seq = &mut self.waiting[i];
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(seq);
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
        self.block_manager.clear_prefix_cache();
    }

    pub fn release_cache(&mut self, seq_id: usize) {
        if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
            let mut seq = self.cached.remove(pos);
            // also free cpu swap
            if seq.status == SequenceStatus::Swapped || seq.status == SequenceStatus::FinishSwapped
            {
                self.block_manager.free_cpu_swap_for_seq(seq_id);
            }
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(&seq);
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

    #[allow(non_snake_case)]
    pub fn filter_prefill_finished(
        &mut self,
        scheduled_ids: &Vec<usize>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut finished_seqs = Vec::new();
        let mut remove_ids = Vec::new();
        let CHUNK_SIZE: usize =
            if self.cfg.flash_context.unwrap_or(false) && cfg!(feature = "flash-context") {
                4096
            } else {
                8192
            };
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
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        for i in 0..self.cached.len() {
            let (status, swapped_time, seq_id, seq_len) = {
                let seq = &self.cached[i];
                (
                    seq.status,
                    seq.swapped_time().unwrap_or(cur_time),
                    seq.id,
                    seq.len(),
                )
            };
            if status != SequenceStatus::Swapped || cur_time - swapped_time < SWAP_COOLING_PERIOD {
                continue;
            }

            let available_kvcache_tokens = self.get_available_kv_tokens();
            if !self.block_manager.can_swap_in(&self.cached[i])
                || (available_kvcache_tokens - seq_len < MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP)
            {
                if !self.running.is_empty() {
                    // Wait for swap in
                    continue;
                }

                let evicted = self.block_manager.evict_prefix_cache_blocks(1);
                if evicted > 0 {
                    crate::log_warn!(
                        "Evicted {} prefix cache block(s) for swap-in Seq {}.",
                        evicted,
                        seq_id
                    );
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
    pub fn try_swap_out(&mut self, idx: usize, is_running: bool) -> bool {
        if (cfg!(feature = "metal") || is_running && idx >= self.running.len())
            || (!is_running && idx >= self.cached.len())
        {
            return false;
        }

        let mut seq = if is_running {
            &mut self.running[idx]
        } else {
            &mut self.cached[idx]
        };

        // If sequence has blocks, attempt to swap to CPU.
        // If cannot swap, fallback.
        if !seq.block_table.is_empty()
            && (seq.status == SequenceStatus::Running || seq.status == SequenceStatus::Cached)
            && self.block_manager.can_swap_out(&seq)
        {
            // make sure we have identical number of blocks when swapping in
            // for decoding
            if let Err(_) = self.block_manager.ensure_allocate(&mut seq) {
                return false;
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
                    if seq.status == SequenceStatus::Running {
                        seq.status = SequenceStatus::Swapped;
                    } else {
                        seq.status = SequenceStatus::FinishSwapped;
                    }
                    // seq.num_cached_tokens = seq.len();
                    self.block_manager.deallocate(&seq);
                    // block table need to be reallocated when swapping in
                    self.cached.push(seq.clone());
                    return true;
                }
                Err(e) => {
                    crate::log_warn!("Swap out failed for seq {}: {:?}", seq.id, e);
                }
            }
        }
        return false;
    }

    pub fn try_swap_out_by_id(&mut self, seq_id: usize, is_running: bool) -> bool {
        if is_running {
            if let Some(pos) = self.running.iter().position(|seq| seq.id == seq_id) {
                return self.try_swap_out(pos, true);
            }
        } else {
            if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
                return self.try_swap_out(pos, false);
            }
        }
        false
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
            || seq.status == SequenceStatus::FinishSwapped // swapped out and finished sequence
            || seq.len() < PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD
        // prefill length < 128
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
            if !self
                .block_manager
                .can_allocate_without_prefix(&self.transferred[idx])
            {
                // Not enough memory right now. Put data back and try later.
                let evicted = self.block_manager.evict_prefix_cache_blocks(1);
                if evicted > 0 {
                    crate::log_warn!(
                        "Evicted {} prefix cache block(s) for Seq {} KvCache receiving!",
                        evicted,
                        seq_id
                    );
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
            self.block_manager
                .allocate_without_prefix(&mut self.transferred[idx])?;

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
        self.block_manager.prefix_cache_blocks() * self.block_manager.get_block_size()
    }

    pub fn evict_prefix_cache_until_free(&mut self, min_free_blocks: usize) -> usize {
        self.block_manager
            .evict_prefix_cache_until_free(min_free_blocks)
    }

    pub fn evict_prefix_cache_blocks(&mut self, blocks: usize) -> usize {
        self.block_manager.evict_prefix_cache_blocks(blocks)
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let free_blocks = self.block_manager.get_num_free_blocks();
        free_blocks * self.block_manager.get_block_size()
    }

    pub fn get_total_kv_tokens(&self) -> usize {
        let total_blocks = self.block_manager.get_num_total_blocks();
        total_blocks * self.block_manager.get_block_size()
    }

    pub fn get_cpu_swap_usage(&self) -> (f32, f32) {
        const SIZE_IN_GB: usize = 1024 * 1024 * 1024;
        let kvcache_memory_gb = self.cfg.kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        if cfg!(feature = "metal") {
            // Metal use unified memory, no cpu swap memory used
            (0f32, 0f32)
        } else {
            let cpu_kvcache_memory_gb = kvcache_memory_gb * self.cfg.cpu_mem_fold.unwrap_or(0.5f32);
            (
                self.block_manager.get_cpu_swap_usage() * cpu_kvcache_memory_gb,
                cpu_kvcache_memory_gb,
            )
        }
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
            let cpu_kvcache_memory_gb = kvcache_memory_gb * self.cfg.cpu_mem_fold.unwrap_or(0.5f32);
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

    /// Check if the given token is a tool call end token
    /// This supports both:
    /// 1. Explicit tool call end tokens (e.g., </tool_call> in XML format)
    /// 2. JSON end token "}" combined with Regex validation for {..."name":..., "arguments":...} pattern
    pub fn is_tool_call_end(&self, token: u32, idx: usize) -> bool {
        // 1. Check for explicit tool call end tokens (XML style)
        if self.tool_call_end_token_ids.contains(&token) {
            return true;
        }

        // 2. Check for JSON style tool call using Regex
        // This handles models like Qwen3 that output raw JSON without XML tags
        if self.json_end_token_id == Some(token) {
            if let Some(tokenizer) = &self.tokenizer {
                // Temporarily add the token to get complete output for decoding
                let mut temp_output = self.running[idx].output_ids.to_vec();
                temp_output.push(token);

                if let Ok(decoded) = tokenizer.decode(&temp_output, true) {
                    // Check for JSON tool call pattern using Regex
                    // The pattern matches if the decoded string ends with a valid JSON tool call structure
                    if self.tool_call_regex.is_match(&decoded) {
                        return true;
                    }
                }
            }
        }

        false
    }
}
