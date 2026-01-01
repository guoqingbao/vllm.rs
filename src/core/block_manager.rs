// src/core/block_manager.rs
use super::prefix_cache::{PrefixCache, PrefixCacheConfig, PrefixCacheUpdate};
use super::runner::RunnerType;
use super::sequence::{Sequence, SequenceStatus};
use crate::def_broadcast_message_to_runners;
use crate::runner::{receive_local, send_local, MessageType};
use candle_core::Result;
use interprocess::{local_socket::Stream as LocalStream, TryClone};
use parking_lot::RwLock;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Block {
    pub id: usize,
    pub ref_count: usize,
}

impl Block {
    pub fn reset(&mut self) {
        self.ref_count = 1;
    }
}

pub struct BlockManager {
    blocks: Vec<Block>,
    free_block_ids: VecDeque<usize>,
    used_block_ids: HashSet<usize>,
    // CPU blocks (for swapping)
    cpu_blocks: Vec<Block>,
    free_cpu_block_ids: VecDeque<usize>,
    // mapping for swapped sequences: seq.id -> Vec<cpu_block_id>
    swapped_map: HashMap<usize, Vec<usize>>,
    block_size: usize,
    runners: Arc<RwLock<RunnerType>>,
    prefix_cache: Option<PrefixCache>,
}

impl BlockManager {
    pub fn new(
        runners: Arc<RwLock<RunnerType>>,
        num_blocks: usize,
        num_cpu_blocks: usize,
        block_size: usize,
        prefix_cache: PrefixCacheConfig,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_block_ids = VecDeque::with_capacity(num_blocks);

        for i in 0..num_blocks {
            blocks.push(Block {
                id: i,
                ref_count: 0,
            });
            free_block_ids.push_back(i);
        }

        let mut cpu_blocks = Vec::with_capacity(num_cpu_blocks);
        let mut free_cpu_block_ids = VecDeque::with_capacity(num_cpu_blocks);
        for i in 0..num_cpu_blocks {
            cpu_blocks.push(Block {
                id: i,
                ref_count: 0,
            });
            free_cpu_block_ids.push_back(i);
        }

        let prefix_cache = if prefix_cache.enabled && prefix_cache.max_cached_blocks > 0 {
            Some(PrefixCache::new(block_size, prefix_cache))
        } else {
            None
        };

        Self {
            blocks,
            free_block_ids,
            used_block_ids: HashSet::new(),
            cpu_blocks,
            free_cpu_block_ids,
            swapped_map: HashMap::new(),
            block_size,
            runners,
            prefix_cache,
        }
    }

    fn allocate_block(&mut self, block_id: usize) -> &mut Block {
        let block = &mut self.blocks[block_id];
        assert_eq!(block.ref_count, 0);
        block.reset();
        self.free_block_ids.retain(|&id| id != block_id);
        self.used_block_ids.insert(block_id);
        block
    }

    fn allocate_fresh(&mut self, seq: &mut Sequence) -> Result<()> {
        for _ in 0..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available, retry later!"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
        if let Some(last_block_id) = seq.block_table.last() {
            let _ = self.try_clear_blocks(vec![*last_block_id]);
        }
        Ok(())
    }

    fn deallocate_block(&mut self, block_id: usize) {
        assert_eq!(self.blocks[block_id].ref_count, 0);
        if self.used_block_ids.remove(&block_id) {
            self.free_block_ids.push_back(block_id);
        }
    }

    pub fn required_blocks(&mut self, seq: &Sequence) -> usize {
        if let Some(prefix_cache) = self.prefix_cache.as_mut() {
            let prefix_match = prefix_cache.match_prefix(&seq.token_ids);
            let matched_blocks =
                self.adjusted_matched_blocks(seq.token_ids.len(), prefix_match.matched_blocks);
            seq.num_blocks().saturating_sub(matched_blocks)
        } else {
            seq.num_blocks()
        }
    }

    pub fn can_allocate(&mut self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= self.required_blocks(seq)
    }

    pub fn can_allocate_without_prefix(&self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= seq.num_blocks()
    }

    pub fn allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        assert!(seq.block_table.is_empty());
        if self.prefix_cache.is_some() {
            let mut prefix_cache = self.prefix_cache.take().unwrap();
            let result = self.allocate_with_prefix(seq, &mut prefix_cache);
            self.prefix_cache = Some(prefix_cache);
            result
        } else {
            self.allocate_fresh(seq)
        }
    }

    pub fn allocate_without_prefix(&mut self, seq: &mut Sequence) -> Result<()> {
        assert!(seq.block_table.is_empty());
        self.allocate_fresh(seq)
    }

    pub fn deallocate(&mut self, seq: &Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            self.decrement_block_ref(block_id as usize);
        }
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        let mut need_block: usize = 1;
        if seq.len() % self.block_size != 0 {
            need_block += 1;
        }
        self.free_block_ids.len() >= need_block
    }

    pub fn may_append(&mut self, seq: &mut Sequence) -> Result<()> {
        let len_mod = seq.len() % self.block_size;
        if len_mod == 1 {
            //approaching next block
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available, retry later!"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
        Ok(())
    }

    pub fn ensure_allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        let mut new_blocks = Vec::new();
        for i in seq.block_table.len()..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available, retry later!"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
            if i > seq.num_blocks() - 5 {
                new_blocks.push(block_id as u32);
            }
        }
        if !new_blocks.is_empty() {
            let _ = self.try_clear_blocks(new_blocks);
        }
        Ok(())
    }

    fn increment_block_ref(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        if block.ref_count == 0 {
            self.free_block_ids.retain(|&id| id != block_id);
            self.used_block_ids.insert(block_id);
        }
        block.ref_count += 1;
    }

    fn decrement_block_ref(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        block.ref_count = block.ref_count.saturating_sub(1);
        if block.ref_count == 0 {
            self.deallocate_block(block_id);
        }
    }

    fn adjusted_matched_blocks(&self, tokens_len: usize, matched_blocks: usize) -> usize {
        let full_blocks = tokens_len / self.block_size;
        if matched_blocks == full_blocks && tokens_len % self.block_size == 0 && matched_blocks > 0
        {
            matched_blocks - 1
        } else {
            matched_blocks
        }
    }

    fn allocate_with_prefix(
        &mut self,
        seq: &mut Sequence,
        prefix_cache: &mut PrefixCache,
    ) -> Result<()> {
        let tokens = &seq.token_ids;
        let mut matched_blocks = 0usize;
        let mut last_hash = None;

        if prefix_cache.enabled() {
            let prefix_match = prefix_cache.match_prefix(tokens);
            last_hash = prefix_match.last_hash;
            matched_blocks =
                self.adjusted_matched_blocks(tokens.len(), prefix_match.matched_blocks);
        }

        let cached_tokens = matched_blocks * self.block_size;
        if matched_blocks > 0 {
            crate::log_info!(
                "Prefix cache hit seq {} ({} cached tokens, {} blocks)",
                seq.id,
                cached_tokens,
                matched_blocks
            );
            if let Some(hash) = last_hash {
                let mut cached_blocks = prefix_cache.blocks_for_match(hash);
                cached_blocks.truncate(matched_blocks);
                for block_id in cached_blocks {
                    self.increment_block_ref(block_id);
                    seq.block_table.push(block_id as u32);
                }
            }
        } else if prefix_cache.enabled() && tokens.len() >= self.block_size {
            crate::log_info!("Prefix cache miss seq {} ({} tokens)", seq.id, tokens.len());
        }

        seq.num_cached_tokens = cached_tokens;

        let mut new_blocks = Vec::new();
        for _ in seq.block_table.len()..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available, retry later!"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
            new_blocks.push(block_id as u32);
        }
        if !new_blocks.is_empty() {
            let _ = self.try_clear_blocks(new_blocks);
        }

        Ok(())
    }

    pub fn cache_sequence(&mut self, seq: &Sequence) {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return;
        };
        if !prefix_cache.enabled() {
            return;
        }
        if matches!(
            seq.status,
            SequenceStatus::Swapped | SequenceStatus::FinishSwapped
        ) {
            return;
        }

        let tokens = &seq.token_ids;
        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return;
        }
        if seq.block_table.len() < full_blocks {
            return;
        }

        let blocks: Vec<usize> = seq
            .block_table
            .iter()
            .take(full_blocks)
            .map(|&id| id as usize)
            .collect();

        crate::log_info!(
            "Prefix cache insert seq {} ({} tokens, {} blocks)",
            seq.id,
            tokens.len(),
            full_blocks
        );

        let PrefixCacheUpdate { inserted, evicted } = prefix_cache.insert_prefix(tokens, &blocks);
        for block_id in inserted {
            self.increment_block_ref(block_id);
        }
        for block_id in evicted {
            self.decrement_block_ref(block_id);
        }
    }

    pub fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache
            .as_ref()
            .map_or(false, |cache| cache.enabled())
    }

    pub fn prefix_cache_blocks(&self) -> usize {
        self.prefix_cache
            .as_ref()
            .map_or(0, |cache| cache.cached_blocks())
    }

    pub fn clear_prefix_cache(&mut self) {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return;
        };
        let evicted = prefix_cache.clear();
        for block_id in evicted {
            self.decrement_block_ref(block_id);
        }
    }

    pub fn evict_prefix_cache_blocks(&mut self, num_blocks: usize) -> usize {
        let Some(prefix_cache) = self.prefix_cache.as_mut() else {
            return 0;
        };
        if num_blocks == 0 {
            return 0;
        }
        let evicted = prefix_cache.evict_blocks(num_blocks);
        let count = evicted.len();
        for block_id in evicted {
            self.decrement_block_ref(block_id);
        }
        count
    }

    pub fn evict_prefix_cache_until_free(&mut self, min_free_blocks: usize) -> usize {
        let mut total_evicted = 0;
        loop {
            if self.free_block_ids.len() >= min_free_blocks {
                break;
            }
            let evicted = {
                let Some(prefix_cache) = self.prefix_cache.as_mut() else {
                    break;
                };
                prefix_cache.evict_blocks(1)
            };
            if evicted.is_empty() {
                break;
            }
            total_evicted += evicted.len();
            for block_id in evicted {
                self.decrement_block_ref(block_id);
            }
        }
        total_evicted
    }

    pub fn get_num_total_blocks(&self) -> usize {
        self.blocks.len()
    }
    pub fn get_num_free_blocks(&self) -> usize {
        self.free_block_ids.len()
    }

    pub fn get_block_size(&self) -> usize {
        self.block_size
    }

    pub fn get_cpu_swap_usage(&self) -> f32 {
        let total_cpu_blocks = self.cpu_blocks.len();
        (total_cpu_blocks - self.free_cpu_block_ids.len()) as f32 / total_cpu_blocks as f32
    }

    // def try_transfer_prefill
    def_broadcast_message_to_runners!(
        pub, // visibility
        try_transfer_prefill, // function name to create
        transfer_prefill, // thread-mode method name
        (seq: &Sequence), // arguments
        MessageType::TransferPrefill, // message to send
        (seq.clone()), // message payload (must clone)
        MessageType::TransferPrefillResponse, // response to match
        bool // inner return type
    );

    // def try_receive_prefill
    def_broadcast_message_to_runners!(
        pub, // visibility
        try_receive_prefill,
        try_receive_prefill,
        (available_tokens: usize),
        MessageType::ReceivePrefill,
        (available_tokens),
        MessageType::ReceivePrefillResponse,
        (bool, Option<Sequence>)
    );

    // def try_check_prefill_status
    def_broadcast_message_to_runners!(
        pub,
        try_check_prefill_status,
        check_prefill_status,
        (seq_id: usize),
        MessageType::CheckPrefillStatus,
        (seq_id),
        MessageType::CheckPrefillStatusResponse,
        bool
    );

    // def try_swap_kvcache
    def_broadcast_message_to_runners!(
        pub,
        try_swap_kvcache,
        swap_kvcache,
        (mappings: HashMap<usize, usize>, swap_in: bool),
        MessageType::KVCacheSwap,
        ((mappings.clone(), swap_in)),
        MessageType::KVCacheSwapResponse,
        bool
    );

    // def try_send_kvcache
    def_broadcast_message_to_runners!(
        pub,
        try_send_kvcache,
        send_kvcache,
        (seq: &Sequence, token: u32),
        MessageType::KvCacheSend,
        ((seq.clone(), token)),
        MessageType::KvCacheSendResponse,
        bool
    );

    // def try_receive_kvcache
    def_broadcast_message_to_runners!(
        pub,
        try_receive_kvcache,
        receive_kvcache,
        (seq: &Sequence),
        MessageType::KvCacheReceive,
        (seq.clone()),
        MessageType::KvCacheReceiveResponse,
        (bool, u32, usize)
    );

    // After the client received prefill kvcache, it can call PD server to release
    // corresponding kvcache backup
    // def try_release_remote_kvcache
    def_broadcast_message_to_runners!(
        pub,
        try_release_remote_kvcache,
        release_remote_kvcache,
        (seq_id: usize),
        MessageType::KvCacheRelease,
        (seq_id),
        MessageType::KvCacheReleaseResponse,
        bool
    );

    // def try_check_kvcache_release
    def_broadcast_message_to_runners!(
        pub,
        try_check_kvcache_release,
        check_kvcache_release,
        (seq_id: usize),
        MessageType::CheckKvCacheRelease,
        (seq_id),
        MessageType::CheckKvCacheReleaseResponse,
        bool
    );

    // Zero specific blocks
    def_broadcast_message_to_runners!(
        pub,
        try_clear_blocks,
        clear_blocks,
        (block_ids: Vec<u32>),
        MessageType::ClearBlocks,
        (block_ids.clone()),
        MessageType::ClearBlocksResponse,
        bool
    );

    /// Can we swap-out `seq` (i.e., move its GPU blocks to CPU swap space)?
    #[allow(unused)]
    pub fn can_swap_out(&self, seq: &Sequence) -> bool {
        // Need at least as many free CPU blocks as the sequence currently owns.
        #[cfg(feature = "cuda")]
        {
            if seq
                .block_table
                .iter()
                .any(|&id| self.blocks[id as usize].ref_count > 1)
            {
                return false;
            }
            let needed = seq.num_blocks();
            self.free_cpu_block_ids.len() > needed
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Can we swap-in `seq` (i.e., bring its blocks back from CPU to GPU)?
    #[allow(unused)]
    pub fn can_swap_in(&self, seq: &Sequence) -> bool {
        // Need at least as many free GPU blocks as seq.num_blocks()
        #[cfg(feature = "cuda")]
        {
            self.free_block_ids.len() > seq.num_blocks()
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Swap out the GPU blocks of `seq` into CPU swap space.
    /// Caller need to deallocate the gpu blocks used in this seq
    pub fn swap_out(&mut self, seq: &mut Sequence) -> Result<()> {
        let num_blocks = seq.block_table.len();
        if self.free_cpu_block_ids.len() < num_blocks {
            candle_core::bail!("Not enough CPU swap blocks for seq {}", seq.id);
        }

        crate::log_warn!(
            "Swap out sequence {} ({} blocks) to CPU memory",
            seq.id,
            num_blocks,
        );

        // mapping GPU → CPU
        let mut mapping = std::collections::HashMap::new();
        let mut cpu_ids = Vec::with_capacity(num_blocks);

        for &gpu_bid_u32 in &seq.block_table {
            let gpu_bid = gpu_bid_u32 as usize;
            let cpu_bid = self
                .free_cpu_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free CPU swap blocks"))?;

            mapping.insert(gpu_bid, cpu_bid);
            cpu_ids.push(cpu_bid);
        }

        // Actual data copy GPU → CPU
        self.try_swap_kvcache(mapping.clone(), false)?;
        seq.swapped_time = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as usize,
        );

        // Update manager bookkeeping
        for (&gpu_bid, &cpu_bid) in &mapping {
            let gpu_block = &mut self.blocks[gpu_bid];
            let cpu_block = &mut self.cpu_blocks[cpu_bid];
            cpu_block.ref_count = gpu_block.ref_count;
        }

        self.swapped_map.insert(seq.id, cpu_ids);

        Ok(())
    }

    /// Need to preallocate new spaces for the seq before calling this function
    pub fn swap_in(&mut self, seq: &mut Sequence) -> Result<()> {
        let cpu_ids = self
            .swapped_map
            .remove(&seq.id)
            .ok_or_else(|| candle_core::Error::msg("No CPU-swap entry for seq"))?;

        if cpu_ids.len() > seq.block_table.len() {
            // push back and free
            self.swapped_map.insert(seq.id, cpu_ids);
            self.free_cpu_swap_for_seq(seq.id);
            candle_core::bail!("Insufficient GPU blocks to swap in sequence {}", seq.id);
        }

        // mapping CPU → GPU (reverse)
        let mapping: std::collections::HashMap<usize, usize> = cpu_ids
            .iter()
            .enumerate()
            .map(|(i, &cpu_id)| (cpu_id, seq.block_table[i] as usize))
            .collect();

        // Actual data copy CPU → GPU
        self.try_swap_kvcache(mapping.clone(), true)?;

        // Free CPU blocks now that data is back on GPU
        for cpu_bid in cpu_ids {
            let cpu_block = &mut self.cpu_blocks[cpu_bid];
            cpu_block.ref_count = 0;
            self.free_cpu_block_ids.push_back(cpu_bid);
        }

        Ok(())
    }

    /// Free CPU-side swap blocks for a seq (if any). Useful for aborts.
    pub fn free_cpu_swap_for_seq(&mut self, seq_id: usize) {
        if let Some(cpu_ids) = self.swapped_map.remove(&seq_id) {
            for cpu_bid in cpu_ids {
                let cpu_block = &mut self.cpu_blocks[cpu_bid];
                cpu_block.ref_count = 0;
                self.free_cpu_block_ids.push_back(cpu_bid);
            }
        }
    }
}
