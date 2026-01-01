# Prefix Cache (KV Reuse)

Prefix cache lets vLLM.rs reuse KV cache blocks from prior requests when a new
prompt shares a prefix. This accelerates consecutive requests with overlapping
history (for example, chat sessions that replay the same system + earlier turns).

## How it works
- Finished sequences contribute full KV blocks to a global prefix cache.
- New requests find the longest cached prefix (block-aligned) and reuse those blocks.
- Remaining tokens are prefetched as usual, with KV writes continuing after the cached prefix.

Prefix cache is block-granular: only full KV blocks are reused. If the common
prefix ends mid-block, the tail of that block is recomputed. When a prompt is
fully cached at block boundaries, the last block is recomputed to ensure a
non-empty prefill step for correct sampling.

## Flags
- `--prefix-cache`: enable prefix cache.
- `--prefix-cache-max-tokens <N>`: cap cache size in tokens (rounded down to block size).

If `--prefix-cache-max-tokens` is not set, a default of ~25% of the GPU KV blocks
is reserved for cached prefixes.

## Notes
- Prefix cache uses the same KV memory pool as active sequences. A larger cache
  reduces the maximum number of concurrent tokens available for new requests.
- Cached KV reuse is automatic; no `session_id` is required.
- Sliding window attention limits how much cached context is effectively used.
