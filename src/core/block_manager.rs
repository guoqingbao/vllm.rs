// src/core/block_manager.rs
use super::runner::RunnerType;
use super::sequence::Sequence;
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
}

impl BlockManager {
    pub fn new(
        runners: Arc<RwLock<RunnerType>>,
        num_blocks: usize,
        num_cpu_blocks: usize,
        block_size: usize,
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

        Self {
            blocks,
            free_block_ids,
            used_block_ids: HashSet::new(),
            cpu_blocks,
            free_cpu_block_ids,
            swapped_map: HashMap::new(),
            block_size,
            runners,
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

    fn deallocate_block(&mut self, block_id: usize) {
        assert_eq!(self.blocks[block_id].ref_count, 0);
        if self.used_block_ids.remove(&block_id) {
            self.free_block_ids.push_back(block_id);
        }
    }

    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= seq.num_blocks()
    }

    pub fn allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        assert!(seq.block_table.is_empty());
        for _ in 0..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
        Ok(())
    }

    pub fn deallocate(&mut self, seq: &Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            let block_id = block_id as usize;
            let block = &mut self.blocks[block_id];
            block.ref_count = block.ref_count.saturating_sub(1);
            if block.ref_count == 0 {
                self.deallocate_block(block_id);
            }
        }
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        let need_block = seq.len() % self.block_size == 1;
        self.free_block_ids.len() >= (need_block as usize)
    }

    pub fn may_append(&mut self, seq: &mut Sequence) -> Result<()> {
        let len_mod = seq.len() % self.block_size;
        if len_mod == 1 {
            //approaching next block
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
        Ok(())
    }

    pub fn ensure_allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        for _ in seq.block_table.len()..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .ok_or_else(|| candle_core::Error::msg("No free blocks available"))?;
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
        Ok(())
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

    pub fn kvcache_swap(&mut self, mappings: HashMap<usize, usize>, swap_in: bool) -> Result<()> {
        match &mut *self.runners.write() {
            RunnerType::Thread(model_runner) => model_runner.kvcache_swap(mappings, swap_in),
            RunnerType::Process(ref mut runner_streams) => {
                let cloned_streams: Vec<LocalStream> = runner_streams
                    .iter_mut()
                    .map(|s| s.try_clone().expect("clone failed"))
                    .collect();
                let all_result: Result<()> = cloned_streams
                    .into_par_iter()
                    .map(|mut stream| {
                        send_local(
                            &mut vec![stream.try_clone()?],
                            &MessageType::KVCacheSwap((mappings.clone(), swap_in)),
                            false,
                        )?;
                        let response = receive_local(&mut stream, false)?;
                        match response {
                            MessageType::KVCacheSwapResponse(success) => {
                                if success {
                                    Ok(())
                                } else {
                                    candle_core::bail!("Kv cache swap failed!")
                                }
                            }
                            other => {
                                candle_core::bail!("Unexpected response type: {:?}", other)
                            }
                        }
                    })
                    .collect();
                all_result
            }
        }
    }

    /// Can we swap-out `seq` (i.e., move its GPU blocks to CPU swap space)?
    pub fn can_swap_out(&self, seq: &Sequence) -> bool {
        // Need at least as many free CPU blocks as the sequence currently owns.
        let needed = seq.block_table.len();
        self.free_cpu_block_ids.len() >= needed
    }

    /// Can we swap-in `seq` (i.e., bring its blocks back from CPU to GPU)?
    pub fn can_swap_in(&self, seq: &Sequence) -> bool {
        // Need at least as many free GPU blocks as seq.num_blocks()
        self.free_block_ids.len() >= seq.num_blocks()
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
            num_blocks,
            seq.id
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
        self.kvcache_swap(mapping.clone(), false)?;
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
        self.kvcache_swap(mapping.clone(), true)?;

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
