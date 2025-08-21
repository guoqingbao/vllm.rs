// src/core/block_manager.rs
use super::sequence::Sequence;
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct Block {
    pub id: usize,
    pub ref_count: usize,
    pub hash: i64,
    pub token_ids: Vec<u32>,
}

impl Block {
    pub fn update(&mut self, hash: i64, token_ids: Vec<u32>) {
        assert!(hash != -1);
        self.hash = hash;
        self.token_ids = token_ids;
    }

    pub fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = -1;
        self.token_ids.clear();
    }
}

#[derive(Debug)]
pub struct BlockManager {
    blocks: Vec<Block>,
    free_block_ids: VecDeque<usize>,
    used_block_ids: HashSet<usize>,
    block_size: usize,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_block_ids = VecDeque::with_capacity(num_blocks);

        for i in 0..num_blocks {
            blocks.push(Block {
                id: i,
                ref_count: 0,
                hash: -1,
                token_ids: Vec::new(),
            });
            free_block_ids.push_back(i);
        }

        Self {
            blocks,
            free_block_ids,
            used_block_ids: HashSet::new(),
            block_size,
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
        self.used_block_ids.remove(&block_id);
        self.free_block_ids.push_back(block_id);
    }

    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= seq.num_blocks()
    }

    pub fn allocate(&mut self, seq: &mut Sequence) {
        assert!(seq.block_table.is_empty());
        for _ in 0..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .expect("No free blocks available");
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
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

    pub fn may_append(&mut self, seq: &mut Sequence) {
        let len_mod = seq.len() % self.block_size;
        if len_mod == 1 {
            //approaching next block
            let block_id = self
                .free_block_ids
                .pop_front()
                .expect("No free blocks available");
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
    }

    pub fn ensure_allocate(&mut self, seq: &mut Sequence) {
        for _ in seq.block_table.len()..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .pop_front()
                .expect("No free blocks available");
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        }
    }
}
