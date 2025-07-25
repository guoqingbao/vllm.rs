// src/core/block_manager.rs

use super::sequence::Sequence;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hasher as _;
use twox_hash::XxHash64;

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
    hash_to_block: HashMap<i64, usize>,
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
            hash_to_block: HashMap::new(),
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

        let mut block_hash: i64 = -1;
        let mut cache_miss = false;

        for i in 0..seq.num_blocks() {
            let token_ids = seq.block(i);
            block_hash = if token_ids.len() == self.block_size {
                self.compute_hash(&token_ids, block_hash)
            } else {
                -1
            };

            let mut block_id = usize::MAX;
            if let Some(&id) = self.hash_to_block.get(&block_hash) {
                if self.blocks[id].token_ids == token_ids {
                    block_id = id;
                }
            }

            if block_id == usize::MAX {
                cache_miss = true;
            }

            let block = if cache_miss {
                block_id = self
                    .free_block_ids
                    .pop_front()
                    .expect("No free blocks available");
                self.allocate_block(block_id)
            } else if self.used_block_ids.contains(&block_id) {
                let b = &mut self.blocks[block_id];
                b.ref_count += 1;
                b
            } else {
                self.allocate_block(block_id)
            };

            if block_hash != -1 {
                block.update(block_hash, token_ids.clone());
                self.hash_to_block.insert(block_hash, block_id);
            }

            seq.block_table.push(block_id as u32);
            if !cache_miss {
                seq.num_cached_tokens += self.block_size;
            }
        }
    }

    pub fn deallocate(&mut self, seq: &Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            let block_id = block_id as usize;
            let block = &mut self.blocks[block_id];
            block.ref_count = block.ref_count.saturating_sub(1);
            if block.ref_count == 0 {
                self.deallocate_block(block_id);
                if let Some(hash) = self.find_block_hash(block_id) {
                    self.hash_to_block.remove(&hash);
                }
            }
        }
    }

    fn find_block_hash(&self, block_id: usize) -> Option<i64> {
        self.hash_to_block
            .iter()
            .find(|&(_, &id)| id == block_id)
            .map(|(&hash, _)| hash)
    }

    pub fn compute_hash(&self, token_ids: &Vec<u32>, prefix_hash: i64) -> i64 {
        let mut hasher = XxHash64::with_seed(123456789);
        if prefix_hash != -1 {
            hasher.write(&prefix_hash.to_le_bytes());
        }
        let bytes: Vec<u8> = token_ids.iter().flat_map(|&id| id.to_le_bytes()).collect();
        hasher.write(&bytes);
        hasher.finish() as i64
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        let need_block = seq.len() % self.block_size == 1;
        self.free_block_ids.len() >= (need_block as usize)
    }

    pub fn may_append(&mut self, seq: &mut Sequence) {
        let len_mod = seq.len() % self.block_size;
        if len_mod == 1 {
            // assert!(last_block.hash != -1);
            let block_id = self
                .free_block_ids
                .pop_front()
                .expect("No free blocks available");
            self.allocate_block(block_id);
            seq.block_table.push(block_id as u32);
        } else if len_mod == 0 {
            // assert_eq!(last_block.hash, -1);
            let token_ids = seq.block(seq.num_blocks() - 1);
            let prefix = if seq.block_table.len() > 1 {
                self.blocks[seq.block_table[seq.block_table.len() - 2] as usize].hash
            } else {
                -1
            };
            let last_block_index = *seq.block_table.last().expect("Block table is empty") as usize;
            let h = self.compute_hash(&token_ids, prefix);
            self.blocks[last_block_index].update(h, token_ids.clone());
            self.hash_to_block
                .insert(h, self.blocks[last_block_index].id);
        }

        // else {
        //     assert_eq!(last_block.hash, -1);
        // }
    }
}
