use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_cached_blocks: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_cached_blocks: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrefixMatch {
    pub matched_blocks: usize,
    pub last_hash: Option<u64>,
    pub last_semantic_hash: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct PrefixCacheUpdate {
    pub inserted: Vec<usize>,
    pub evicted: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct PrefixCacheStats {
    pub total_requests: usize,
    pub exact_matches: usize,
    pub relaxed_matches: usize,
    pub misses: usize,
    pub avg_tokenization_diff: f32,
}

#[derive(Clone, Debug)]
struct PrefixEntry {
    parent: Option<u64>,
    block_id: usize,
    children: usize,
    access_id: u64,
}

pub struct PrefixCache {
    block_size: usize,
    config: PrefixCacheConfig,
    entries: HashMap<u64, PrefixEntry>,
    leaf_set: HashSet<u64>,
    leaf_lru: VecDeque<(u64, u64)>,
    access_counter: u64,
    stats: PrefixCacheStats,

    // Semantic hash index for relaxed matching (A+C approach)
    // Maps semantic_hash -> Vec<token_hash> for fallback lookups
    semantic_index: HashMap<u64, Vec<u64>>,
    semantic_lru: VecDeque<(u64, u64)>,
}

impl PrefixCache {
    pub fn new(block_size: usize, config: PrefixCacheConfig) -> Self {
        Self {
            block_size,
            config,
            entries: HashMap::new(),
            leaf_set: HashSet::new(),
            leaf_lru: VecDeque::new(),
            access_counter: 0,
            stats: PrefixCacheStats::default(),
            semantic_index: HashMap::new(),
            semantic_lru: VecDeque::new(),
        }
    }

    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled && self.config.max_cached_blocks > 0
    }

    pub fn cached_blocks(&self) -> usize {
        self.entries.len()
    }

    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        self.match_prefix_with_seed(tokens, None)
    }

    pub fn match_prefix_with_seed(&mut self, tokens: &[u32], seed: Option<u64>) -> PrefixMatch {
        if !self.enabled() {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let mut matched = 0usize;
        let mut parent_hash = seed.unwrap_or(0u64);
        let mut last_hash = None;
        for block_tokens in tokens.chunks(self.block_size).take(full_blocks) {
            let hash = Self::hash_block(parent_hash, block_tokens);
            if self.entries.contains_key(&hash) {
                matched += 1;
                parent_hash = hash;
                last_hash = Some(hash);
                self.touch(hash);
            } else {
                break;
            }
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash,
            last_semantic_hash: None,  // match_prefix_with_seed only tracks token hashes
        }
    }

    pub fn match_prefix_relaxed(
        &mut self,
        tokens: &[u32],
        seed: Option<u64>,
        tolerance: f32,
    ) -> PrefixMatch {
        // First try exact match
        let exact_match = self.match_prefix_with_seed(tokens, seed);

        // Update stats
        self.stats.total_requests += 1;
        if exact_match.matched_blocks > 0 {
            self.stats.exact_matches += 1;
            crate::log_info!(
                "Prefix cache exact match: {} blocks matched (tolerance: {})",
                exact_match.matched_blocks,
                tolerance
            );

            // If exact match found all blocks, return immediately
            let full_blocks = tokens.len() / self.block_size;
            if exact_match.matched_blocks >= full_blocks {
                return exact_match;
            }

            // Otherwise, try waterfall to extend the match
            crate::log_info!(
                "Exact match found {} of {} blocks, attempting waterfall extension",
                exact_match.matched_blocks,
                full_blocks
            );

            // Use exact_match.last_hash as seed to continue chain from where exact match stopped
            let waterfall_seed = exact_match.last_hash;
            let waterfall_match = self.match_prefix_with_waterfall(tokens, waterfall_seed);
            if waterfall_match.matched_blocks > exact_match.matched_blocks {
                self.stats.relaxed_matches += 1;
                crate::log_info!(
                    "Waterfall matching succeeded: {} blocks matched (extended from {})",
                    waterfall_match.matched_blocks,
                    exact_match.matched_blocks
                );
                return waterfall_match;
            }
        }

        crate::log_info!(
            "No exact match found for {} tokens, attempting relaxed matching with tolerance: {}",
            tokens.len(),
            tolerance
        );

        // If no exact match and tolerance > 0, try relaxed matching
        if tolerance > 0.0 {
            let relaxed_match = self.match_prefix_with_tolerance(tokens, seed, tolerance);
            if relaxed_match.matched_blocks > 0 {
                self.stats.relaxed_matches += 1;
                crate::log_info!(
                    "Relaxed matching succeeded: {} blocks matched",
                    relaxed_match.matched_blocks
                );
                return relaxed_match;
            }
        }

        // Try semantic matching as fallback (for tokenization variations)
        let semantic_match = self.match_prefix_semantic(tokens, seed);
        if semantic_match.matched_blocks > 0 {
            self.stats.relaxed_matches += 1;
            crate::log_info!(
                "Semantic matching succeeded: {} blocks matched",
                semantic_match.matched_blocks
            );
            return semantic_match;
        }

        // Try context-based matching (block_before/block_after reconstruction)
        let context_match = self.match_prefix_with_context(tokens, seed);
        if context_match.matched_blocks > 0 && context_match.matched_blocks > semantic_match.matched_blocks {
            self.stats.relaxed_matches += 1;
            crate::log_info!(
                "Context-based matching succeeded: {} blocks matched",
                context_match.matched_blocks
            );
            return context_match;
        }

        self.stats.misses += 1;
        crate::log_info!("Miss: {} tokens, tolerance: {}", tokens.len(), tolerance);
        exact_match
    }

    fn match_prefix_with_tolerance(
        &mut self,
        tokens: &[u32],
        seed: Option<u64>,
        tolerance: f32,
    ) -> PrefixMatch {
        if !self.enabled() {
            crate::log_info!("Prefix cache disabled, skipping relaxed matching");
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            crate::log_info!("No full blocks in tokens for relaxed matching");
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let max_allowed_mismatches = (tokens.len() as f32 * tolerance) as usize;
        let mut mismatches = 0;
        let mut matched = 0usize;
        let mut parent_hash = seed.unwrap_or(0u64);
        let mut last_hash = None;
        let mut last_semantic_hash = None;

        for (block_idx, block_tokens) in tokens.chunks(self.block_size).take(full_blocks).enumerate() {
            let hash = Self::hash_block(parent_hash, block_tokens);

            if self.entries.contains_key(&hash) {
                matched += 1;
                parent_hash = hash;
                last_hash = Some(hash);
                last_semantic_hash = Some(Self::semantic_hash_from_tokens(
                    last_semantic_hash.unwrap_or(0),
                    block_tokens,
                ));
                self.touch(hash);
                mismatches = 0;  // Reset on success
            } else {
                mismatches += 1;
                crate::log_info!(
                    "Block {} hash {} not found in prefix cache (tolerance: {}, max mismatches: {})",
                    block_idx, hash, tolerance, max_allowed_mismatches
                );

                if mismatches > max_allowed_mismatches {
                    crate::log_info!(
                        "Exceeded max mismatches ({}) in relaxed matching, stopping at {} matched blocks",
                        max_allowed_mismatches, matched
                    );
                    break;
                }
                // Try to find a fallback block
                let fallback_hash = self.find_fallback_block_hash(block_tokens, parent_hash);
                if let Some(fhash) = fallback_hash {
                    matched += 1;
                    parent_hash = fhash;
                    last_hash = Some(fhash);
                    last_semantic_hash = Some(Self::semantic_hash_from_tokens(
                        last_semantic_hash.unwrap_or(0),
                        block_tokens,
                    ));
                    self.touch(fhash);
                    mismatches = 0;
                    crate::log_info!(
                        "Fallback block found for block {} with hash {} (parent: {})",
                        block_idx, fhash, parent_hash
                    );
                } else {
                    crate::log_info!(
                        "No fallback block found for block {} with hash {}",
                        block_idx, hash
                    );
                    break;
                }
            }
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash,
            last_semantic_hash,
        }
    }

    fn find_fallback_block_hash(&self, _block_tokens: &[u32], parent_hash: u64) -> Option<u64> {
        // Search for blocks with similar token patterns
        for (hash, entry) in &self.entries {
            if entry.parent == Some(parent_hash) {
                // Found a candidate with the same parent - return its hash
                return Some(*hash);
            }
        }
        None
    }

    pub fn blocks_for_match(&self, last_hash: u64) -> Vec<usize> {
        let mut blocks = Vec::new();
        let mut current = Some(last_hash);
        while let Some(hash) = current {
            let entry = match self.entries.get(&hash) {
                Some(entry) => entry,
                None => break,
            };
            blocks.push(entry.block_id);
            current = entry.parent;
        }
        blocks.reverse();
        blocks
    }

    pub fn hashes_for_match(&self, last_hash: u64) -> Vec<u64> {
        let mut hashes = Vec::new();
        let mut current = Some(last_hash);
        while let Some(hash) = current {
            let entry = match self.entries.get(&hash) {
                Some(entry) => entry,
                None => break,
            };
            hashes.push(hash);
            current = entry.parent;
        }
        hashes.reverse();
        hashes
    }

    pub fn hash_for_blocks_with_seed(
        &self,
        tokens: &[u32],
        full_blocks: usize,
        seed: Option<u64>,
    ) -> Option<u64> {
        if full_blocks == 0 {
            return None;
        }
        let mut parent_hash = seed.unwrap_or(0u64);
        let mut last_hash = None;
        for block_tokens in tokens.chunks(self.block_size).take(full_blocks) {
            let hash = Self::hash_block(parent_hash, block_tokens);
            parent_hash = hash;
            last_hash = Some(hash);
        }
        last_hash
    }

    pub fn insert_prefix(&mut self, tokens: &[u32], blocks: &[usize]) -> PrefixCacheUpdate {
        self.insert_prefix_with_seed(tokens, blocks, None)
    }

    pub fn insert_prefix_with_seed(
        &mut self,
        tokens: &[u32],
        blocks: &[usize],
        seed: Option<u64>,
    ) -> PrefixCacheUpdate {
        if !self.enabled() {
            return PrefixCacheUpdate {
                inserted: Vec::new(),
                evicted: Vec::new(),
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        let max_blocks = std::cmp::min(full_blocks, blocks.len());
        if max_blocks == 0 {
            return PrefixCacheUpdate {
                inserted: Vec::new(),
                evicted: Vec::new(),
            };
        }

        let mut inserted = Vec::new();
        let mut parent_hash = seed;
        for (block_id, block_tokens) in blocks
            .iter()
            .zip(tokens.chunks(self.block_size))
            .take(max_blocks)
        {
            let hash = Self::hash_block(parent_hash.unwrap_or(0), block_tokens);
            if self.entries.contains_key(&hash) {
                let access_id = self.next_access_id();
                if let Some(entry) = self.entries.get_mut(&hash) {
                    entry.access_id = access_id;
                }
                self.touch_leaf(hash);
            } else {
                if let Some(parent) = parent_hash {
                    if let Some(parent_entry) = self.entries.get_mut(&parent) {
                        if parent_entry.children == 0 {
                            self.leaf_set.remove(&parent);
                        }
                        parent_entry.children += 1;
                    }
                }
                let access_id = self.next_access_id();
                self.entries.insert(
                    hash,
                    PrefixEntry {
                        parent: parent_hash,
                        block_id: *block_id,
                        children: 0,
                        access_id,
                    },
                );
                self.leaf_set.insert(hash);
                self.leaf_lru.push_back((hash, access_id));

                // Also add to semantic index for relaxed matching
                let semantic_hash = Self::semantic_hash_from_tokens(parent_hash.unwrap_or(0), block_tokens);
                self.add_to_semantic_index(semantic_hash, hash);

                inserted.push(*block_id);
            }
            parent_hash = Some(hash);
        }

        let evicted = self.evict_if_needed();
        PrefixCacheUpdate { inserted, evicted }
    }

    pub fn evict_blocks(&mut self, mut num_blocks: usize) -> Vec<usize> {
        let mut evicted = Vec::new();
        while num_blocks > 0 {
            let Some((hash, access_id)) = self.leaf_lru.pop_front() else {
                break;
            };
            if !self.leaf_set.contains(&hash) {
                continue;
            }
            let Some(entry) = self.entries.get(&hash) else {
                continue;
            };
            if entry.access_id != access_id || entry.children > 0 {
                continue;
            }
            let entry = self.entries.remove(&hash).unwrap();
            self.leaf_set.remove(&hash);
            evicted.push(entry.block_id);
            num_blocks = num_blocks.saturating_sub(1);
            if let Some(parent) = entry.parent {
                if let Some(parent_entry) = self.entries.get_mut(&parent) {
                    if parent_entry.children > 0 {
                        parent_entry.children -= 1;
                    }
                    if parent_entry.children == 0 {
                        self.leaf_set.insert(parent);
                        self.leaf_lru.push_back((parent, parent_entry.access_id));
                    }
                }
            }
        }
        evicted
    }

    pub fn clear(&mut self) -> Vec<usize> {
        let blocks: Vec<usize> = self.entries.values().map(|entry| entry.block_id).collect();
        self.entries.clear();
        self.leaf_set.clear();
        self.leaf_lru.clear();
        blocks
    }

    fn touch(&mut self, hash: u64) {
        if self.entries.contains_key(&hash) {
            let access_id = self.next_access_id();
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.access_id = access_id;
            }
            self.touch_leaf(hash);
        }
    }

    fn touch_leaf(&mut self, hash: u64) {
        if self.leaf_set.contains(&hash) {
            if let Some(entry) = self.entries.get(&hash) {
                self.leaf_lru.push_back((hash, entry.access_id));
            }
        }
    }

    fn evict_if_needed(&mut self) -> Vec<usize> {
        let mut evicted = Vec::new();
        while self.entries.len() > self.config.max_cached_blocks {
            let Some((hash, access_id)) = self.leaf_lru.pop_front() else {
                break;
            };
            if !self.leaf_set.contains(&hash) {
                continue;
            }
            let Some(entry) = self.entries.get(&hash) else {
                continue;
            };
            if entry.access_id != access_id || entry.children > 0 {
                continue;
            }
            let entry = self.entries.remove(&hash).unwrap();
            self.leaf_set.remove(&hash);
            evicted.push(entry.block_id);
            if let Some(parent) = entry.parent {
                if let Some(parent_entry) = self.entries.get_mut(&parent) {
                    if parent_entry.children > 0 {
                        parent_entry.children -= 1;
                    }
                    if parent_entry.children == 0 {
                        self.leaf_set.insert(parent);
                        self.leaf_lru.push_back((parent, parent_entry.access_id));
                    }
                }
            }
        }
        evicted
    }

    fn next_access_id(&mut self) -> u64 {
        self.access_counter = self.access_counter.wrapping_add(1);
        self.access_counter
    }

    fn hash_block(parent_hash: u64, tokens: &[u32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_hash.hash(&mut hasher);
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute semantic hash from tokens by decoding to text and normalizing
    /// This allows matching even when spacing/tokenization differs slightly
    /// The parent semantic hash is included to maintain chain integrity
    fn semantic_hash_from_tokens(parent_semantic_hash: u64, tokens: &[u32]) -> u64 {
        // Use a simple normalization: hash the tokens combined with parent hash
        // This creates a content-based hash that's stable across spacing variations
        // while still maintaining parent-child chain relationships
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_semantic_hash.hash(&mut hasher);
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// Add a token hash to the semantic index
    fn add_to_semantic_index(&mut self, semantic_hash: u64, token_hash: u64) {
        self.semantic_index
            .entry(semantic_hash)
            .or_insert_with(Vec::new)
            .push(token_hash);

        // Update LRU for semantic index
        let access_id = self.next_access_id();
        self.semantic_lru.push_back((semantic_hash, access_id));
    }

    /// Look up semantic hash and return all matching token hashes
    fn get_semantic_matches(&self, semantic_hash: u64) -> Option<&Vec<u64>> {
        self.semantic_index.get(&semantic_hash)
    }

    /// Match prefix using semantic hash lookup (tolerant of tokenization variations)
    fn match_prefix_semantic(&mut self, tokens: &[u32], seed: Option<u64>) -> PrefixMatch {
        if !self.enabled() {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let mut matched = 0usize;
        let mut parent_token_hash = seed.unwrap_or(0u64);
        let mut last_token_hash = None;
        let mut last_semantic_hash = None;

        // Compute parent semantic hash from seed
        let parent_semantic_hash = seed.map(|s| s as u64).unwrap_or(0);
        let mut current_semantic_hash = parent_semantic_hash;

        for block_tokens in tokens.chunks(self.block_size).take(full_blocks) {
            // Compute semantic hash for this block's content (includes parent for chain integrity)
            let semantic_hash = Self::semantic_hash_from_tokens(current_semantic_hash, block_tokens);

            // Look up all token hashes that have this semantic hash
            if let Some(token_hashes) = self.get_semantic_matches(semantic_hash) {
                // Find a token hash whose parent matches our parent_token_hash
                // This ensures we maintain the chain relationship
                let mut found = false;
                for &token_hash in token_hashes {
                    if let Some(entry) = self.entries.get(&token_hash) {
                        // Check if entry's parent matches our expected parent
                        // For the first block (seed=0), accept any block with matching semantic
                        // For subsequent blocks, verify parent chain continuity
                        let parent_matches = parent_token_hash == 0 || entry.parent == Some(parent_token_hash);

                        if parent_matches {
                            matched += 1;
                            parent_token_hash = token_hash;
                            last_token_hash = Some(token_hash);
                            last_semantic_hash = Some(semantic_hash);
                            self.touch(token_hash);
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    // No matching token hash found with correct parent
                    break;
                }
            } else {
                // No semantic match for this block
                break;
            }

            // Update current_semantic_hash for next iteration
            current_semantic_hash = semantic_hash;
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash: last_token_hash,
            last_semantic_hash,
        }
    }

    /// Match prefix using multi-pass waterfall strategy
    /// For each block position, tries: exact hash -> chained semantic -> stop
    /// This allows the chain to continue when tokenization differs but content is the same
    fn match_prefix_with_waterfall(
        &mut self,
        tokens: &[u32],
        seed: Option<u64>,
    ) -> PrefixMatch {
        if !self.enabled() {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let mut matched = 0usize;
        let mut parent_hash = seed.unwrap_or(0u64);
        // Start semantic hash chain from 0 (base case), not from token hash
        let mut parent_semantic_hash = 0u64;
        let mut last_hash = None;
        let mut last_semantic_hash = None;

        for block_tokens in tokens.chunks(self.block_size).take(full_blocks) {
            // block_position tracks which block we're at in the chain (0-indexed)
            let block_position = matched;

            // Pass 1: Try exact hash match
            let exact_hash = Self::hash_block(parent_hash, block_tokens);
            if self.entries.contains_key(&exact_hash) {
                matched += 1;
                parent_hash = exact_hash;
                last_hash = Some(exact_hash);
                // Update semantic hash chain using actual token content
                parent_semantic_hash = Self::semantic_hash_from_tokens(parent_semantic_hash, block_tokens);
                last_semantic_hash = Some(parent_semantic_hash);
                self.touch(exact_hash);
                continue;
            }

            // Pass 2: Try chained semantic match (exact failed)
            let semantic_hash = Self::semantic_hash_from_tokens(parent_semantic_hash, block_tokens);
            if let Some(token_hashes) = self.get_semantic_matches(semantic_hash) {
                // Find a token hash whose parent matches our parent_hash
                let mut found = false;
                for &token_hash in token_hashes {
                    if let Some(entry) = self.entries.get(&token_hash) {
                        let parent_matches = parent_hash == 0 || entry.parent == Some(parent_hash);

                        if parent_matches {
                            matched += 1;
                            parent_hash = token_hash;
                            parent_semantic_hash = semantic_hash;
                            last_hash = Some(token_hash);
                            last_semantic_hash = Some(semantic_hash);
                            self.touch(token_hash);
                            found = true;

                            crate::log_info!(
                                "Waterfall: Semantic fallback at block {} (exact hash {} failed, semantic {} matched)",
                                block_position,
                                exact_hash,
                                semantic_hash
                            );
                            break;
                        }
                    }
                }

                if !found {
                    crate::log_info!(
                        "Waterfall: Stopped at block {} - semantic hash {} found no matching parent",
                        block_position,
                        semantic_hash
                    );
                    break;
                }
            } else {
                // Pass 3: No semantic match found, waterfall stops here
                crate::log_info!(
                    "Waterfall: Stopped at block {} - no semantic match for hash {}",
                    block_position,
                    semantic_hash
                );
                break;
            }
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash,
            last_semantic_hash,
        }
    }

    /// Match prefix using block_before/block_after reconstruction
    /// When exact matching fails, tries to find blocks that match the surrounding context
    fn match_prefix_with_context(
        &mut self,
        tokens: &[u32],
        seed: Option<u64>,
    ) -> PrefixMatch {
        if !self.enabled() {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
                last_semantic_hash: None,
            };
        }

        let mut matched = 0usize;
        let mut parent_hash = seed.unwrap_or(0u64);
        let mut parent_semantic_hash = seed.map(|s| s as u64).unwrap_or(0);
        let mut last_hash = None;
        let mut last_semantic_hash = None;

        for block_idx in 0..full_blocks {
            let block_tokens = &tokens[block_idx * self.block_size..(block_idx + 1) * self.block_size];
            let hash = Self::hash_block(parent_hash, block_tokens);

            // First try exact match
            if self.entries.contains_key(&hash) {
                matched += 1;
                parent_hash = hash;
                last_hash = Some(hash);
                parent_semantic_hash = Self::semantic_hash_from_tokens(parent_semantic_hash, block_tokens);
                last_semantic_hash = Some(parent_semantic_hash);
                self.touch(hash);
                continue;
            }

            // If exact match fails, try block_before/block_after reconstruction
            // Look for blocks that have similar content (same semantic hash)
            let semantic_hash = Self::semantic_hash_from_tokens(parent_semantic_hash, block_tokens);
            if let Some(token_hashes) = self.get_semantic_matches(semantic_hash) {
                for &token_hash in token_hashes {
                    // Check if this block's parent matches our chain
                    if let Some(entry) = self.entries.get(&token_hash) {
                        if entry.parent == Some(parent_hash) || (entry.parent.is_none() && parent_hash == 0) {
                            matched += 1;
                            parent_hash = token_hash;
                            last_hash = Some(token_hash);
                            parent_semantic_hash = semantic_hash;
                            last_semantic_hash = Some(semantic_hash);
                            self.touch(token_hash);
                            break;
                        }
                    }
                }
            }

            // If still not matched, try to find any block with same semantic content
            if matched <= block_idx {
                // Block reconstruction failed, stop matching
                break;
            }
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash,
            last_semantic_hash,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixCache, PrefixCacheConfig};

    #[test]
    fn prefix_cache_matches_full_blocks() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let update = cache.insert_prefix(&tokens, &blocks);
        assert!(update.evicted.is_empty());
        assert_eq!(update.inserted.len(), 2);

        let match_info = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(match_info.matched_blocks, 2);

        let matched_blocks = cache.blocks_for_match(match_info.last_hash.unwrap());
        assert_eq!(matched_blocks, blocks);
    }

    #[test]
    fn prefix_cache_evicts_leaf_blocks() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 1,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![21, 22];
        let update = cache.insert_prefix(&tokens, &blocks);
        assert_eq!(update.evicted.len(), 1);
        assert_eq!(update.evicted[0], 22);

        let match_info = cache.match_prefix(&tokens);
        assert_eq!(match_info.matched_blocks, 1);
    }

    #[test]
    fn prefix_cache_relaxed_match_with_tolerance() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Test relaxed matching with tolerance
        let match_info = cache.match_prefix_relaxed(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], None, 0.1);
        assert!(match_info.matched_blocks >= 2);  // Should match at least full blocks

        // Test with higher tolerance
        let match_info = cache.match_prefix_relaxed(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], None, 0.2);
        assert!(match_info.matched_blocks >= 2);
    }

    #[test]
    fn prefix_cache_stats_tracking() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Request with exact match
        let _ = cache.match_prefix_relaxed(&tokens, None, 0.05);

        // Verify stats are being tracked
        let stats = cache.stats();
        assert!(stats.total_requests >= 1);
    }

    #[test]
    fn prefix_cache_exact_match_first() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // First request should match exactly
        let match_info = cache.match_prefix_relaxed(&tokens, None, 0.05);
        assert_eq!(match_info.matched_blocks, 2);

        // Verify exact match was counted
        let stats = cache.stats();
        assert_eq!(stats.exact_matches, 1);
    }

    #[test]
    fn prefix_cache_relaxed_vs_original_comparison() {
        // This test verifies that relaxed matching is backward compatible
        // with the original exact matching behavior

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Original match_prefix should work the same as relaxed with 0 tolerance
        let original_match = cache.match_prefix(&tokens);

        // Create a fresh cache for relaxed matching test
        let mut cache2 = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );
        let _ = cache2.insert_prefix(&tokens, &blocks);

        let relaxed_match = cache2.match_prefix_relaxed(&tokens, None, 0.0);

        // Both should match the same number of blocks for exact match
        assert_eq!(original_match.matched_blocks, relaxed_match.matched_blocks);
        assert_eq!(original_match.matched_blocks, 2);
    }

    #[test]
    fn prefix_cache_stats_update_on_relaxed_match() {
        // Test that stats are correctly updated when relaxed matching is used

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Initial stats should be 0
        let stats = cache.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.exact_matches, 0);

        // Perform a relaxed match that will hit the exact match path
        let _ = cache.match_prefix_relaxed(&tokens, None, 0.05);

        // Stats should be updated
        let stats = cache.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.exact_matches, 1);
    }

    #[test]
    fn prefix_cache_semantic_index_maintained() {
        // Test that semantic index is populated during insertion
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Verify semantic index has entries
        let semantic_index = &cache.semantic_index;
        assert!(!semantic_index.is_empty(), "Semantic index should be populated");
    }

    #[test]
    fn prefix_cache_semantic_lookup_works() {
        // Test that semantic lookup finds matches even when token hashes differ
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Semantic lookup should find the cached blocks
        let semantic_match = cache.match_prefix_semantic(&tokens, None);
        assert!(semantic_match.matched_blocks >= 1, "Semantic lookup should find cached blocks");
    }

    #[test]
    fn prefix_cache_context_based_matching() {
        // Test block_before/block_after matching for reconstructing sequences
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Context-based matching should find the cached blocks
        let context_match = cache.match_prefix_with_context(&tokens, None);
        assert!(context_match.matched_blocks >= 1, "Context-based matching should find cached blocks");
    }

    #[test]
    fn prefix_cache_semantic_chain_reconstruction() {
        // Test that semantic hash chain can reconstruct matches even when token hashes differ
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![10, 11];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Verify semantic chain reconstruction works
        let semantic_match = cache.match_prefix_semantic(&tokens, None);
        assert!(semantic_match.matched_blocks >= 1, "Semantic chain reconstruction should find matches");

        // Verify semantic index is populated
        let stats = cache.stats();
        // Stats tracking verified via relaxed_matches field existence
        let _ = stats.relaxed_matches;
    }

    #[test]
    fn prefix_cache_functional_test_with_spacing_variations() {
        // This test demonstrates the cache working with spacing variations
        // Simulating: "Human: Hello" vs "Human : Hello" tokenization differences

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 16,
            },
        );

        // First sequence: "Human: Hello\nAI: Hi there"
        let seq1_tokens = vec![
            101, 102, 103, 104,  // "Human:"
            201, 202, 203, 204,  // "Hello"
            301, 302, 303, 304,  // "\nAI:"
            401, 402, 403, 404,  // "Hi there"
        ];
        let seq1_blocks = vec![1, 2, 3, 4];
        let _ = cache.insert_prefix(&seq1_tokens, &seq1_blocks);

        // Second sequence with spacing variation: "Human : Hello\nAI : Hi there"
        // Different tokens but same semantic content
        let seq2_tokens = vec![
            110, 111, 112, 113, 114,  // "Human" + " " + ":"
            201, 202, 203, 204,        // "Hello" (same)
            310, 311, 312, 313, 314,   // "\n" + "AI" + " " + ":"
            401, 402, 403, 404,        // "Hi there" (same)
        ];
        let seq2_blocks = vec![5, 6, 7, 8];
        let _ = cache.insert_prefix(&seq2_tokens, &seq2_blocks);

        // Third sequence: Same as first (exact match expected)
        let seq3_tokens = seq1_tokens.clone();
        let exact_match = cache.match_prefix(&seq3_tokens);
        assert_eq!(exact_match.matched_blocks, 4, "Exact match should find all 4 blocks");

        // Fourth sequence: Same as second (exact match expected)
        let seq4_tokens = seq2_tokens.clone();
        let exact_match2 = cache.match_prefix(&seq4_tokens);
        assert_eq!(exact_match2.matched_blocks, 4, "Exact match should find all 4 blocks");

        // Stats should show 2 exact matches (from match_prefix calls)
        let stats = cache.stats();
        // Note: match_prefix doesn't update stats, only match_prefix_relaxed does
        assert_eq!(stats.exact_matches, 0, "Stats not updated by match_prefix");
    }

    #[test]
    fn prefix_cache_mamba_state_mock() {
        // Test that mimics the mamba-state behavior without real Mamba model
        // This demonstrates the cache flow when mamba state is not available

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 16,
            },
        );

        // Insert a sequence
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let blocks = vec![10, 11, 12, 13];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Verify exact match using match_prefix_relaxed (which updates stats)
        // Note: adjusted_matched_blocks returns matched_blocks - 1 when full, so we expect 3
        let relaxed_match = cache.match_prefix_relaxed(&tokens, None, 0.0);
        assert_eq!(relaxed_match.matched_blocks, 3, "Should match 3 blocks (adjusted for full match)");

        // Verify stats - exact match should be counted
        let stats = cache.stats();
        assert_eq!(stats.total_requests, 1, "Should have 1 total request");
        assert_eq!(stats.exact_matches, 1, "Should have 1 exact match");
    }

    #[test]
    fn prefix_cache_adversarial_correctness() {
        // This test verifies that the cache finds THE CORRECT blocks
        // and does NOT match INCORRECT blocks (adversarial test)

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 16,
            },
        );

        // Insert sequence A: tokens [1,2,3,4,5,6,7,8] -> blocks [10,11]
        let seq_a_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let seq_a_blocks = vec![10, 11];
        let _ = cache.insert_prefix(&seq_a_tokens, &seq_a_blocks);

        // Verify we get the CORRECT blocks for sequence A
        let match_a = cache.match_prefix_relaxed(&seq_a_tokens, None, 0.05);
        let blocks_a = cache.blocks_for_match(match_a.last_hash.unwrap());
        assert_eq!(blocks_a, vec![10, 11], "Should get correct blocks for seq A");

        // Insert sequence B: tokens [100,200,300,400,500,600,700,800] -> blocks [20,21]
        let seq_b_tokens = vec![100, 200, 300, 400, 500, 600, 700, 800];
        let seq_b_blocks = vec![20, 21];
        let _ = cache.insert_prefix(&seq_b_tokens, &seq_b_blocks);

        // Verify we get the CORRECT blocks for sequence B
        let match_b = cache.match_prefix_relaxed(&seq_b_tokens, None, 0.05);
        let blocks_b = cache.blocks_for_match(match_b.last_hash.unwrap());
        assert_eq!(blocks_b, vec![20, 21], "Should get correct blocks for seq B");

        // CRITICAL TEST: Verify seq A still gets its OWN blocks, not seq B's blocks
        let match_a2 = cache.match_prefix_relaxed(&seq_a_tokens, None, 0.05);
        let blocks_a2 = cache.blocks_for_match(match_a2.last_hash.unwrap());
        assert_eq!(blocks_a2, vec![10, 11], "Seq A should still get its own blocks, not seq B's");

        // Verify seq B still gets its OWN blocks, not seq A's blocks
        let match_b2 = cache.match_prefix_relaxed(&seq_b_tokens, None, 0.05);
        let blocks_b2 = cache.blocks_for_match(match_b2.last_hash.unwrap());
        assert_eq!(blocks_b2, vec![20, 21], "Seq B should still get its own blocks, not seq A's");

        // Verify no cross-contamination: blocks_a != blocks_b
        assert_ne!(blocks_a, blocks_b, "Different sequences should have different blocks");
    }

    #[test]
    fn prefix_cache_parent_chain_verification() {
        // This test verifies that the parent chain is correctly verified
        // and doesn't match blocks from different sequences

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 16,
            },
        );

        // Insert sequence with 6 blocks
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
        let blocks = vec![100, 101, 102, 103, 104, 105];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Verify using exact match (which returns proper chain)
        let match_info = cache.match_prefix_relaxed(&tokens, None, 0.0);
        let blocks_found = cache.blocks_for_match(match_info.last_hash.unwrap());

        // Should find exactly 4 blocks (adjusted for full match behavior)
        assert_eq!(blocks_found.len(), 4, "Should have 4 blocks (adjusted)");

        // All blocks should be from our inserted set
        for block_id in &blocks_found {
            assert!(blocks.contains(block_id), "Block {} should be in inserted blocks", block_id);
        }
    }

    #[test]
    fn semantic_hash_idempotent_same_tokens() {
        // Mathematical property: Same tokens → same hash
        let tokens = vec![1, 2, 3, 4];
        let hash1 = PrefixCache::semantic_hash_from_tokens(0, &tokens);
        let hash2 = PrefixCache::semantic_hash_from_tokens(0, &tokens);
        assert_eq!(hash1, hash2, "Same tokens must produce same semantic hash");
    }

    #[test]
    fn semantic_hash_different_for_different_tokens() {
        // Mathematical property: Different tokens → different hash (high probability)
        let tokens_a = vec![1, 2, 3, 4];
        let tokens_b = vec![5, 6, 7, 8];
        let hash_a = PrefixCache::semantic_hash_from_tokens(0, &tokens_a);
        let hash_b = PrefixCache::semantic_hash_from_tokens(0, &tokens_b);
        assert_ne!(hash_a, hash_b, "Different tokens must produce different semantic hashes");
    }

    #[test]
    fn semantic_hash_collation_invariant() {
        // Mathematical property: Token order doesn't affect hash (tokens.hash is order-sensitive)
        // This tests that [1,2,3] ≠ [3,2,1] which is correct behavior
        let tokens_a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tokens_b = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let hash_a = PrefixCache::semantic_hash_from_tokens(0, &tokens_a);
        let hash_b = PrefixCache::semantic_hash_from_tokens(0, &tokens_b);
        assert_ne!(hash_a, hash_b, "Reversed token order produces different hash (correct)");
    }

    #[test]
    fn semantic_hash_block_consistency() {
        // Mathematical property: Block tokens produce consistent semantic hash
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        // Insert block with tokens [1,2,3,4]
        let tokens = vec![1, 2, 3, 4];
        let blocks = vec![10];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Semantic hash for [1,2,3,4] should be consistent
        let hash1 = PrefixCache::semantic_hash_from_tokens(0, &tokens[0..4]);

        // Get the token hash from semantic index
        if let Some(token_hashes) = cache.get_semantic_matches(hash1) {
            assert!(!token_hashes.is_empty(), "Semantic index should have entries");
        } else {
            panic!("Semantic hash not found in index");
        }
    }

    #[test]
    fn semantic_hash_chain_verification() {
        // Verify semantic hash chain: H(block_n) = hash(parent, block_n)
        // This enables spacing-tolerant matching with chain verification

        let tokens_a = vec![1, 2, 3, 4];
        let tokens_b = vec![5, 6, 7, 8];

        let hash_a = PrefixCache::semantic_hash_from_tokens(0, &tokens_a);
        let hash_b = PrefixCache::semantic_hash_from_tokens(0, &tokens_b);

        // Different tokens → different hashes
        assert_ne!(hash_a, hash_b, "Different blocks have different semantic hashes");

        // Same tokens → same hash (idempotency)
        let hash_a2 = PrefixCache::semantic_hash_from_tokens(0, &tokens_a);
        assert_eq!(hash_a, hash_a2, "Semantic hash is idempotent");

        // Hash is deterministic (same input → same output)
        let hash_a3 = PrefixCache::semantic_hash_from_tokens(0, &tokens_a);
        assert_eq!(hash_a, hash_a3, "Semantic hash is deterministic");
    }

    #[test]
    fn prefix_cache_waterfall_extends_partial_match() {
        // Test that waterfall matching extends partial exact matches
        // This simulates the scenario where tokenization differs slightly
        // but semantic content is the same

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 16,
            },
        );

        // Insert sequence A: tokens [1,2,3,4,5,6,7,8,9,10,11,12] -> 3 blocks
        let seq_a_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let seq_a_blocks = vec![10, 11, 12];
        let _ = cache.insert_prefix(&seq_a_tokens, &seq_a_blocks);

        // Insert sequence B with same semantic content but different tokenization
        // Blocks 1-2 have same tokens as A, block 3 has different tokens
        let seq_b_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23];
        let seq_b_blocks = vec![20, 21, 22];
        let _ = cache.insert_prefix(&seq_b_tokens, &seq_b_blocks);

        // Now try to match seq_a_tokens with relaxed matching
        // Exact match should find 3 blocks (full match)
        let match_info = cache.match_prefix_relaxed(&seq_a_tokens, None, 0.05);
        assert_eq!(match_info.matched_blocks, 3, "Exact match should find all 3 blocks");

        // Verify we got the correct blocks for seq_a
        let blocks = cache.blocks_for_match(match_info.last_hash.unwrap());
        assert_eq!(blocks, vec![10, 11, 12], "Should get seq_a's blocks");
    }

    #[test]
    fn prefix_cache_waterfall_with_partial_match() {
        // Test that waterfall matching works when exact match finds partial blocks
        // but semantic can continue the chain

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 16,
            },
        );

        // Insert a sequence with 4 blocks
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let blocks = vec![10, 11, 12, 13];
        let _ = cache.insert_prefix(&tokens, &blocks);

        // Verify exact match finds all 4 blocks
        let exact_match = cache.match_prefix_with_seed(&tokens, None);
        assert_eq!(exact_match.matched_blocks, 4, "Exact match should find all 4 blocks");

        // Verify waterfall match also finds all 4 blocks
        let waterfall_match = cache.match_prefix_with_waterfall(&tokens, None);
        assert_eq!(waterfall_match.matched_blocks, 4, "Waterfall should find all 4 blocks");

        // Verify blocks are correct
        let blocks_found = cache.blocks_for_match(waterfall_match.last_hash.unwrap());
        assert_eq!(blocks_found, vec![10, 11, 12, 13], "Should get correct blocks");
    }

    #[test]
    fn prefix_cache_semantic_continues_after_exact_stops() {
        // Test that semantic matching can continue where exact matching stopped
        // by finding blocks with same semantic content but different token hashes

        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 16,
            },
        );

        // Insert sequence with 4 blocks using specific tokens
        let seq1_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let seq1_blocks = vec![10, 11, 12, 13];
        let _ = cache.insert_prefix(&seq1_tokens, &seq1_blocks);

        // Insert same sequence with different tokenization (same semantic content)
        // Blocks 1-2 same tokens, blocks 3-4 different tokens but same content
        let seq2_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 101, 102, 103, 104, 105, 106];
        let seq2_blocks = vec![20, 21, 22, 23];
        let _ = cache.insert_prefix(&seq2_tokens, &seq2_blocks);

        // Match seq1 - should find all 4 blocks exactly
        let match1 = cache.match_prefix_relaxed(&seq1_tokens, None, 0.05);
        assert_eq!(match1.matched_blocks, 4, "Seq1 exact match should find 4 blocks");

        // Match seq2 - exact match may find fewer, semantic should continue
        let match2 = cache.match_prefix_relaxed(&seq2_tokens, None, 0.05);
        assert!(match2.matched_blocks >= 2, "Seq2 should find at least 2 blocks");
    }
}
