# Performance Research Report: vllm.rs + attention.rs vs sglang
## Qwen 3.5 MoE — Unquantized and FP8

---

## A) Executive Summary

### Top 5 Root Causes of Performance Gap

1. **Unfused MoE GEMM pipeline (critical, ~2× gap)**: vllm.rs executes 3 _separate_ [moe_gemm](file:///Users/bobking/attention.rs/src/moe.rs#6-166) / [moe_gemm_fp8](file:///Users/bobking/attention.rs/src/moe.rs#591-606) calls (gate, up, down) per MoE layer. Each call involves multiple kernel launches (sort → compute expert offsets → GEMM → scatter). sglang uses a **single fused call** ([trtllm_bf16_moe](file:///Users/bobking/flashinfer/flashinfer/fused_moe/core.py#2237-2331) or [trtllm_fp8_block_scale_moe](file:///Users/bobking/flashinfer/flashinfer/fused_moe/core.py#2521-2621)) that handles routing, GEMM1 (gate+up fused), activation, GEMM2 (down), and output combine in one dispatch.

2. **Excessive kernel launches in FP8 CUTLASS path (~5-7 launches per GEMM)**: Our FP8 path on SM90+ requires: `fp8_quantize_per_token_group` → `gather_rows` (shuffle inputs) → `gather_rows` (shuffle scales) → `calculate_expert_offsets` → `grouped_gemm` → `scatter_rows`. This is done 3 times (gate, up, down) = **15-21 kernel launches** per MoE layer vs sglang's **~1-2 launches**.

3. **No FlashAttention v3/v4 for attention layers**: sglang uses FA3 (`sgl_kernel.flash_attn`) with FP8 KV cache support and FA4 as an option. Our stack uses FlashAttention v2 / flashinfer via [PagedAttention](file:///Users/bobking/attention.rs/src/paged_attention.rs#11-26). FA3 provides significant speedups especially with FP8 KV quantization and paged KV cache.

4. **No attention state merge for MLA/DeepSeek chunked prefill**: For MLA models (e.g., DeepSeek), sglang uses [merge_state_v2](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#300-303) to combine partial attention outputs (prefix from cached KV + suffix from new tokens) using log-sum-exp statistics. This enables efficient chunked prefill for complex architectures. Our approach lacks this mechanism for MLA.

5. **Hand-written GEMM kernels vs CUTLASS/TRT-LLM optimized grouped GEMM**: For unquantized MoE, our [moe_gemm_wmma.cu](file:///Users/bobking/attention.rs/src/kernels/src/moe_gemm_wmma.cu) uses WMMA (Tensor Cores) with 32×32 tile sizes and 4 warps. sglang's fused TRT-LLM kernel uses highly tuned CUTLASS with auto-tuned tile configurations, persistent kernels, and warp-specialized scheduling.

### Top 5 Changes for Largest Performance Wins

1. **Integrate the Full Fused MoE Pipeline** — Eliminate all separate kernel calls (routing + GEMM1 + act + GEMM2 + combine) by porting flashinfer's [cutlass_fused_moe](file:///Users/bobking/flashinfer/flashinfer/fused_moe/core.py#506-665) directly via C FFI. This provides the massive benefit of TRT-LLM's heuristic auto-tuner which dynamically dispatches perfectly scaled small-M warp-specialized kernels during decoding.

2. **Integrate TRT-LLM Batched GEMM for Linear Layers (SM100+)** — Replace standard grouped GEMMs parsing Linear or MLP layers with flashinfer's `TrtllmGenBatchedGemmRunner` on Blackwell (SM100+). This grants the same auto-tuner benefits (zero padding) for non-MoE layers. Note: this engine is compiled strictly for SM100+, so Hopper (SM90) must rely on the Fused MoE path for autotuning.

3. **Integrate FA3 for attention** — Replace FlashAttention v2 prefill with FA3 varlen API, add FP8 KV cache support, and use paged KV cache with FA3's native page table support.

4. **Add [merge_state](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/merge_state.py#26-47) kernel for MLA/DeepSeek chunked prefill** — Port the [merge_state_v2](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#300-303) kernel to enable efficient chunked prefill specifically for DeepSeek/MLA architectures, where prefix attention results must be accurately merged with new chunk results.

5. **Reduce kernel launch overhead via stream-ordered operations** — Batch auxiliary operations (quantize, gather, scatter) and use CUDA graphs for the decode path to minimize launch latency.

---

## B) Concrete Call-Path Maps

### sglang Path: Qwen 3.5 MoE Forward

```
Qwen3_5ForCausalLM.forward()                    [qwen3_5.py:714]
  └─ Qwen3_5ForCausalLM.model.forward()         [qwen3_5.py:714-766]
       └─ for layer in self.layers:
            ├─ Qwen3_5AttentionDecoderLayer.forward()  [qwen3_5.py:605-650]
            │    ├─ layer_communicator.prepare_attn()   # fused RMSNorm+residual
            │    ├─ self_attention()                    [qwen3_5.py:573-603]
            │    │    ├─ qkv_proj(hidden_states)        # QKVParallelLinear
            │    │    ├─ q_norm, k_norm                 # RMSNorm (QK norm)
            │    │    ├─ rotary_emb(q, k, positions)    # YaRN RoPE
            │    │    ├─ attn_backend.forward_extend()  # FA3/FA4 with paged KV
            │    │    │    ├─ flash_attn_varlen_func()   # prefix: causal attention on new tokens
            │    │    │    ├─ flash_attn_with_kvcache()   # cached KV lookup
            │    │    │    └─ merge_state_v2()            # merge prefix+suffix outputs
            │    │    └─ o_proj(attn_output)             # RowParallelLinear
            │    ├─ layer_communicator.prepare_mlp()     # fused RMSNorm+residual
            │    └─ Qwen3MoeSparseMoeBlock.forward()    [qwen3_moe.py:265-281]
            │         ├─ forward_normal()                [qwen3_moe.py:293-314]
            │         │    ├─ gate(hidden_states)         # ReplicatedLinear → router logits
            │         │    ├─ TopK → topk_softmax         # top-k gating
            │         │    └─ experts.forward()           # FusedMoE via MoeRunner
            │         │         └─ MoeRunner.run()        [moe_runner/runner.py:73-107]
            │         │              └─ fused_func()      # registered fused op
            │         │                   └─ fused_experts_none_to_flashinfer_trtllm()
            │         │                        ├─ trtllm_bf16_moe()        # BF16 unquantized
            │         │                        │    (single fused: routing + GEMM1[gate||up]
            │         │                        │     + SiLU + GEMM2[down] + combine)
            │         │                        └─ trtllm_fp8_block_scale_moe()  # FP8
            │         │                             (+ per_token_group_quant_fp8()     )
            │         │                             (single fused: same pipeline as BF16)
            │         └─ tensor_model_parallel_all_reduce()
            │
            └─ Qwen3_5LinearDecoderLayer.forward()  [qwen3_5.py:358-403]
                 ├─ (same structure, linear_attn instead of self_attention)
                 └─ (same MoE block)
```

**Key sglang kernels for MoE:**
| Operation | Kernel/Function | Source |
|---|---|---|
| Routing + Top-K | `topk_softmax` (fused inside trtllm kernel) | flashinfer |
| BF16 Grouped GEMM | [trtllm_bf16_moe](file:///Users/bobking/flashinfer/flashinfer/fused_moe/core.py#2237-2331) | flashinfer → CUTLASS 3.x |
| FP8 Grouped GEMM | [trtllm_fp8_block_scale_moe](file:///Users/bobking/flashinfer/flashinfer/fused_moe/core.py#2521-2621) | flashinfer → CUTLASS 3.x |
| FP8 Input Quantize | `per_token_group_quant_fp8` | sgl_kernel |
| Attention (prefill) | `flash_attn_varlen_func` (FA3/FA4) | sgl_kernel |
| Attention (decode) | `flash_attn_with_kvcache` (FA3/FA4) | sgl_kernel |
| State merge (MLA/DeepSeek) | [merge_state_v2](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#300-303) | sgl_kernel |

---

### vllm.rs Path: Qwen 3.5 MoE Forward

```
Qwen3_5MoEForCausalLM::forward()                [qwen3_5_moe.rs:672]
  └─ forward_inner()                             [qwen3_5_moe.rs:592-670]
       ├─ get_attention_causal_mask()              # host-side mask construction
       ├─ embed_tokens.forward()                   # Embedding
       ├─ resolve_seq_slots()                      # Mamba cache slot resolution
       └─ for layer in self.layers:
            └─ Qwen3_5MoEDecoderLayer::forward()  [qwen3_5_moe.rs:236-296]
                 ├─ input_layernorm.forward()       # RMSNorm (F32)
                 ├─ Attention::forward()            # full attention layers
                 │    ├─ QKV projection              # Linear
                 │    ├─ rotary_emb                   # Scaling RoPE
                 │    ├─ reshape_and_cache()          # KV cache update
                 │    │    └─ ffi::reshape_and_cache() # CUDA kernel
                 │    ├─ PagedAttention                # decode: paged attention
                 │    │    └─ cuda_fwd_t()
                 │    │         └─ ffi::paged_attention_v2() # vLLM paged attn kernel
                 │    └─ flash_attn_with_kv_cache()   # prefill: FA v2
                 │         └─ ffi::flash_attn_varlen_fwd() 
                 │    └─ o_proj                        # Linear
                 ├─ OR GatedDeltaNet::forward()    # linear attention layers
                 │    └─ (Mamba-like recurrent)
                 ├─ residual + attn_output
                 ├─ post_attention_layernorm()      # RMSNorm
                 ├─ shared_expert (if applicable)   # Shared expert gate + MLP
                 └─ MoeOrMlp::forward()             [qwen3_5_moe.rs:38-45]
                      ├─ FusedMoe::forward()         [moe.rs:300-370]
                      │    ├─ gate.forward()           # Linear → router logits
                      │    ├─ topk_softmax()           # attention_rs::topk
                      │    ├─ topk_ids.sort()          # CUDA sort (prefill) or CPU sort
                      │    ├─ moe_gemm(xs, gate_w)     # ① Gate projection
                      │    │    └─ (prefill): ffi::moe_gemm_wmma()  # WMMA kernel
                      │    │    └─ (decode):  ffi::moe_gemv()       # GEMV kernel
                      │    ├─ moe_gemm(xs, up_w)       # ② Up projection
                      │    │    └─ (same kernels as gate)
                      │    ├─ SiLU activation           # up * silu(gate)
                      │    ├─ moe_gemm(act, down_w)    # ③ Down projection (with topk_weights)
                      │    │    └─ (same kernels + weighted scatter)
                      │    └─ reshape + sum over experts
                      │
                      └─ FusedMoeFp8::forward()      [moe.rs:1191-1273]
                           ├─ (same routing as FusedMoe)
                           ├─ moe_gemm_fp8(xs, gate)   # ① Gate (FP8)
                           │    └─ CUTLASS path (SM90+, block_size=128):
                           │         ├─ fp8_quantize_per_token_group()   # kernel 1
                           │         ├─ gather_rows (input)               # kernel 2
                           │         ├─ gather_rows (scales)              # kernel 3
                           │         ├─ calculate_expert_offsets()        # kernel 4
                           │         ├─ grouped_gemm()                   # kernel 5 (CUTLASS)
                           │         └─ scatter_rows()                   # kernel 6
                           │    └─ Fallback (non-CUTLASS):
                           │         └─ moe_gemm_wmma_fp8()             # WMMA with FP8 dequant
                           ├─ moe_gemm_fp8(xs, up)     # ② Up (FP8) — same 6 launches
                           ├─ SiLU activation
                           ├─ moe_gemm_fp8(act, down)  # ③ Down (FP8) — same 6 launches
                           └─ reshape + sum
```

**Key vllm.rs kernels for MoE:**
| Operation | Kernel | Source |
|---|---|---|
| Routing | `topk_softmax` | `attention_rs::topk` |
| Sort | radix sort (CUDA) or `sort_last_dim` (CPU) | `attention_rs::sort` |
| BF16 GEMM (prefill) | `moe_gemm_wmma` (WMMA 32×32 tiles) | [moe_gemm_wmma.cu](file:///Users/bobking/attention.rs/src/kernels/src/moe_gemm_wmma.cu) |
| BF16 GEMM (decode) | `moe_gemv` (vectorized) | [moe_gemv.cu](file:///Users/bobking/attention.rs/src/kernels/src/moe_gemv.cu) |
| BF16 GEMM (small batch fallback) | `moe_gemm_vectorized_kernel` (shared mem) | [moe_gemm.cu](file:///Users/bobking/attention.rs/src/kernels/src/moe_gemm.cu) |
| FP8 GEMM (prefill, SM90+) | CUTLASS grouped GEMM | [fp8_moe_cutlass.cu](file:///Users/bobking/attention.rs/src/kernels/src/fp8_moe_cutlass.cu) |
| FP8 GEMM (prefill, <SM90) | `moe_gemm_wmma_fp8` | [moe_gemm_wmma.cu](file:///Users/bobking/attention.rs/src/kernels/src/moe_gemm_wmma.cu) |
| FP8 quantize | `fp8_quantize_per_token_group` | [fp8_moe_cutlass.cu](file:///Users/bobking/attention.rs/src/kernels/src/fp8_moe_cutlass.cu) |
| FP8 gather/scatter | `gather_rows_kernel` / `scatter_rows_kernel` | [fp8_moe_cutlass.cu](file:///Users/bobking/attention.rs/src/kernels/src/fp8_moe_cutlass.cu) |
| Expert offsets | `moe_fp8_calculate_expert_offsets` | helper kernel |
| Attention (decode) | `paged_attention_v2` | vLLM paged attn kernel |
| Attention (prefill) | `flash_attn_varlen_fwd` (FA v2) | flash-attention |
| KV cache update | [reshape_and_cache](file:///Users/bobking/attention.rs/src/paged_attention.rs#1290-1319) | [paged_attention.rs](file:///Users/bobking/attention.rs/src/paged_attention.rs) |

---

## C) Kernel Comparison Table

### MoE GEMM Comparison — Unquantized (BF16)

| Feature | vllm.rs (attention.rs) | sglang (flashinfer TRT-LLM) |
|---|---|---|
| **Kernel type** | Hand-written WMMA (prefill) / GEMV (decode) | CUTLASS 3.x grouped GEMM (TRT-LLM) |
| **Tile size** | 32×32×16 (WMMA_M×N×K), 4 warps | Auto-tuned via CUTLASS tile scheduler |
| **Tensor core usage** | WMMA API (older, less optimal) | WGMMA (SM90) / UMMA (SM100) |
| **Fusion level** | None — 3 separate kernel calls | Fully fused: routing+GEMM1+act+GEMM2+combine |
| **Gate+Up fusion** | Separate calls | Fused into single GEMM1 (2× wide N) |
| **Activation fusion** | Separate host-side SiLU | Fused epilogue |
| **Token routing** | Sort → per-token indexing | Expert-offset based dispatch inside kernel |
| **Kernel launches (per MoE layer)** | 9+ (3 GEMMs × 3 steps each) | 1-2 |
| **Decode path** | `moe_gemv` (custom CUDA) | Same fused kernel (adapts to small M) |

### MoE GEMM Comparison — FP8

| Feature | vllm.rs (attention.rs) | sglang (flashinfer TRT-LLM) |
|---|---|---|
| **Kernel type** | CUTLASS 3.x grouped (SM90+) or WMMA fallback | CUTLASS 3.x via [trtllm_fp8_block_scale_moe](file:///Users/bobking/flashinfer/flashinfer/fused_moe/core.py#2521-2621) |
| **Input quantization** | Separate `fp8_quantize_per_token_group` kernel | Integrated (or thin `per_token_group_quant_fp8`) |
| **Data shuffle (gather)** | Separate `gather_rows_kernel` | Not needed — handled inside fused kernel |
| **Expert offset calc** | Separate `calculate_expert_offsets` kernel | Built into fused kernel arg setup |
| **Output scatter** | Separate `scatter_rows_kernel` with weight multiply | Fused epilogue with weighted combine |
| **Block scale layout** | Supports both row-major (SM90) and col-major (SM100) | Same, handled by flashinfer |
| **Kernel launches per GEMM** | 5-6 | 1 (fused) |
| **Kernel launches per MoE layer** | 15-18+ | 2-3 |
| **Gate+Up fusion** | Separate calls | Fused |

### Attention Comparison

| Feature | vllm.rs (attention.rs) | sglang |
|---|---|---|
| **Prefill kernel** | FlashAttention v2 varlen | FA3 or FA4 varlen |
| **Decode kernel** | vLLM paged_attention_v2 | FA3 `flash_attn_with_kvcache` |
| **FP8 KV cache** | Not supported | Supported (FA3 with k_descale/v_descale) |
| **Chunked prefill (MLA)**| Basic (mask-based) | Local attention blocks + [merge_state_v2](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#300-303) |
| **State merge (MLA)** | Not available | [merge_state_v2](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#300-303) kernel for prefix+suffix |
| **KV cache layout** | Paged (block_size-based, via [reshape_and_cache](file:///Users/bobking/attention.rs/src/paged_attention.rs#1290-1319)) | Paged (page_size-based, page table indices) |
| **CUDA graph (decode)** | Supported | Supported, with FA3/FA4 compatibility |

---

## D) Porting Plan into attention.rs

### Phase 0: Research Validation (This Report)
- [x] Complete call-path reconstruction (both projects)
- [x] Identify kernel bottlenecks
- [x] Document fusion opportunities
- [ ] **Action**: Profile on CUDA machine to confirm kernel launch counts/times

### Phase 1: Full CUTLASS Fused MoE Integration (Estimated: 3-4 weeks)
**Goal**: Achieve sglang-level fusion with a single-call MoE, fixing the M=1 padding bottleneck.

#### What to do:
1. **Build System Changes**:
   - Update [attention.rs/src/kernels/build.rs](file:///Users/bobking/attention.rs/src/kernels/build.rs) to compile [flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu](file:///Users/bobking/flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu).
   - Compile the heuristic autotuner core: [flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp](file:///Users/bobking/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp).

2. **New CUDA FFI Wrapper (`flashinfer_moe_adapter.cu`)**:
   - Instantiate `tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner`.
   - Wrap `Runner::run` taking concatenated gate+up and down weights.

3. **Rust Side Restructure (`attention_rs::moe`)**:
   - Delete manual sorting/routing logic (the C++ runner does this).
   - Collapse 3 GEMM calls into 1 FFI call.

### Phase 2: Standalone TRT-LLM GEMM Integration for Linear Layers (SM100+) (Estimated: 2 weeks)
**Goal**: Fix padding waste on standard M=1 Linear layers for Blackwell GPUs.

#### What to do:
1. **Identify Target Layers**: Non-MoE MLPs, QKV linear, or O-proj linear.
2. **Build System**: Compile [flashinfer/csrc/trtllm_batched_gemm_runner.cu](file:///Users/bobking/flashinfer/csrc/trtllm_batched_gemm_runner.cu). Ensure the `flashinferMetaInfo.h` cubins are fetched.
3. **C FFI Wrapper (`flashinfer_linear_adapter.cu`)**:
   - Instantiate `tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner`.
   - Expose the auto-tuned [run](file:///Users/bobking/sglang/python/sglang/srt/layers/moe/moe_runner/runner.py#73-108) method for M=1 decode.
4. **Hardware Branching**: Only invoke this path if `cudaGetDeviceProperties` major version is >= 10.

### Phase 3: Attention Upgrades & MLA Merging (Estimated: 4-6 weeks)
**Goal**: Integrate FA3, FP8 KV cache, and support DeepSeek/MLA chunked prefill.

#### What to do:
1. **Integrate FA3 kernels** from `sgl_kernel`:
   - `flash_attn_varlen_func` for prefill
   - `flash_attn_with_kvcache` for decode (replaces `paged_attention_v2`)
   - Wire up through `PagedAttention::cuda_fwd_t()`

2. **FP8 KV cache support**:
   - Add k_scale/v_scale parameters to [reshape_and_cache()](file:///Users/bobking/attention.rs/src/paged_attention.rs#1290-1319)
   - Quantize K/V to FP8 during cache insertion
   - Dequantize during attention (FA3 handles this natively)

3. **Port [merge_state](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/merge_state.py#26-47) (For MLA/DeepSeek)**:
   - New CUDA kernel in attention.rs: `merge_state.cu`
   - Rust wrapper: `attention_rs::merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse)`
   - Enable chunked prefill with state merging specifically for MLA architectures.

4. **Chunked prefill strategy (MLA)**:
   - Implement local attention block decomposition (similar to sglang's [make_local_attention_virtual_batches](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#142-292))
   - Process chunks independently, merge results

### Phase 4: MLA Chunked Prefill & State Merge (Estimated: 2-3 weeks)
**Goal**: Enable chunking for MLA/DeepSeek prefill blocks matching sglang's local attention execution.

#### What to do:
1. Implement local attention block decomposition (similar to sglang's [make_local_attention_virtual_batches](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#142-292)).
2. Process chunks independently, merge results using [merge_state_v2](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#300-303).

### What to Port (CUDA/C++ only) vs What Not to Port

| Port ✅ | Don't Port ❌ |
|---|---|
| CUTLASS Fused MoE Runner (`MoE::Runner`) | Triton MoE kernels |
| TRT-LLM Batched GEMM Runner for SM100 | standalone [group_gemm_sm90.cuh](file:///Users/bobking/flashinfer/include/flashinfer/gemm/group_gemm_sm90.cuh) (No autotune) |
| FA3 C++ kernels | Python scheduling logic |
| [merge_state](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/merge_state.py#26-47) CUDA kernel | Token dispatcher abstractions |
| FP8 quantization kernels | DeepEP/MoonCake EP logic |

---

## E) Code Examples (Design-Level)

### Suggested Rust API — Fused MoE (replaces `FusedMoe::forward`)

```rust
// attention_rs::moe (new function wrapping the C++ Runner)

/// Fully Fused MoE Pipeline (flashinfer CUTLASS MoE runner)
/// Handles formatting, memory allocation lookup, and single-dispatch inference.
///
/// # Arguments
/// * `input` - [M, K] input activations
/// * `gate_up_weights` - [num_experts, 2*N, K] concatenated
/// * `down_weights` - [num_experts, K, N]
/// * `topk` - experts per token
///
/// # Returns
/// * [M, N] output (fully combined and un-permuted)
pub fn flashinfer_fused_moe_bf16(
    input: &Tensor,
    gate_up_weights: &Tensor,
    down_weights: &Tensor,
    topk: usize,
) -> Result<Tensor>;

/// FP8 variant with block-wise scales (Major-K layouts)
pub fn flashinfer_fused_moe_fp8(
    input: &Tensor,
    gate_up_weights: &Tensor,
    down_weights: &Tensor,
    gate_up_scales: &Tensor,
    down_scales: &Tensor,
    topk: usize,
) -> Result<Tensor>;
```

### Suggested Rust API — Attention State Merge (MLA/DeepSeek)

```rust
// attention_rs::merge_state (new module)

/// Merge two partial attention outputs using log-sum-exp statistics.
/// Used for MLA/DeepSeek chunked prefill: merges prefix (cached) and suffix (new) results.
///
/// Given two attention outputs with their LSE (log-sum-exp) statistics:
///   merged_output = (exp(lse_a) * output_a + exp(lse_b) * output_b)
///                   / (exp(lse_a) + exp(lse_b))
///
/// # Arguments
/// * `prefix_output` - [num_tokens, num_heads, head_dim] from cached KV
/// * `prefix_lse` - [num_tokens, num_heads] log-sum-exp stats
/// * `suffix_output` - [num_tokens, num_heads, head_dim] from new tokens
/// * `suffix_lse` - [num_tokens, num_heads] log-sum-exp stats
///
/// # Returns
/// * Merged output [num_tokens, num_heads, head_dim]
pub fn merge_state(
    prefix_output: &Tensor,
    prefix_lse: &Tensor,
    suffix_output: &Tensor,
    suffix_lse: &Tensor,
) -> Result<Tensor>;
```

### Example CUDA Wrapper Skeleton — Fused MoE FFI

```cpp
// flashinfer_moe_adapter.cu (new file in attention.rs/src/kernels/src/)

#include <cuda_runtime.h>
#include "flashinfer/trtllm/fused_moe/runner.h"

extern "C" void flashinfer_cutlass_fused_moe_bf16(
    const void* input,
    const void* gate_up_weights,
    const void* down_weights,
    void* output,
    void* workspace,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int topk,
    cudaStream_t stream
) {
    using namespace tensorrt_llm::kernels::trtllmgen_moe;
    
    // 1. Initialize the TRT-LLM wrapped MoE Runner
    // The runner internalizes the heuristic auto-tuner which checks SM
    // arch and dispatches the most minimal padded block config.
    MoE::Runner runner(true /* bfloat16 */, false /* fp8 */);

    // 2. Setup arguments mapping pointers to network geometry
    MoERunnerArgs args;
    args.m = num_tokens;
    args.n = intermediate_dim;
    args.k = hidden_dim;
    // ... mapped rest of params

    // 3. Dispatch the fully fused pipeline
    runner.run(args, workspace, stream);
}
```

### Integration into Existing Call Sites (Pseudocode)

```rust
// In vllm.rs/src/models/layers/moe.rs — FusedMoe::forward()
pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
    // OLD: 3 separate GEMM calls + manual rust routing
    // let router_logits = self.gate.forward(&xs)?;
    // let (topk_weights, topk_ids) = topk_softmax(&router_logits, self.topk)?;
    // ... gate, up, down GEMMs

    // NEW: Single fused call covering routing, sorting, GEMM1, GEMM2
    let ys = flashinfer_fused_moe_bf16(
        xs,
        &self.gate_up_w,  // pre-concatenated [experts, 2*inter, hidden]
        &self.down_w,
        self.topk,
    )?;

    Ok(ys)
}
```

---

## F) Validation Strategy

### Microbenchmarks (First Priority)

| Benchmark | What to Measure | Expected Result |
|---|---|---|
| **MoE routing** | Top-k softmax + sort time | Baseline comparison |
| **Grouped GEMM (BF16)** | Fused vs 3-call for M=1,8,64,512,2048 | Fused should be 1.5-2× faster |
| **Grouped GEMM (FP8)** | Fused gather/scatter vs separate | 2-3× fewer kernel launches |
| **Attention prefill** | FA2 vs FA3 varlen, seq_len=512,2048,8192 | FA3 should be 1.3-1.5× faster |
| **Attention decode** | paged_attn_v2 vs FA3 with_kvcache | FA3 should match or beat |
| **merge_state (MLA)** | Throughput for various num_tokens×num_heads | Verify numerical correctness |

### Metrics to Capture

- **Per-kernel time**: Use `cudaEventRecord` around each kernel
- **Kernel launch count**: Count via CUPTI or Nsight Systems trace
- **Memory bandwidth utilization**: Nsight Compute for key kernels
- **GPU occupancy**: Nsight Compute per kernel
- **End-to-end latency**: Time per layer, time per MoE block

### Tools

| Tool | Usage |
|---|---|
| **Nsight Systems** | System-wide timeline: identify launch gaps, sync points, CPU-GPU overlaps |
| **Nsight Compute** | Per-kernel analysis: occupancy, memory throughput, arithmetic intensity |
| **CUPTI** | Programmatic kernel timing and launch counting |
| **Custom tracing** | `RUST_LOG=debug` + timing macros around attention_rs calls |

### Compile-Time Tracing (macOS-Friendly)

Add static path verification now without needing GPU:

```rust
// In attention_rs (macOS-compatible logging)
#[cfg(not(feature = "cuda"))]
pub fn moe_gemm_fused_gate_up(/* ... */) -> Result<Tensor> {
    tracing::info!(
        "moe_gemm_fused_gate_up: input=({},{}), weights=({},{},{}), topk={}, prefill={}",
        input.dim(0)?, input.dim(1)?,
        weights.dim(0)?, weights.dim(1)?, weights.dim(2)?,
        topk, is_prefill
    );
    candle_core::bail!("CUDA required for moe_gemm_fused_gate_up")
}
```

Use `cargo check --features cuda` to verify FFI signatures match, and `cargo test` with mock tensors to validate Rust-side logic (tensor shapes, layouts, routing).

### Correctness Validation Plan

1. **Unit tests**: Compare fused kernel output vs reference (3 separate calls) on small inputs
2. **Numerical accuracy**: Track max/mean absolute error, verify within FP16/BF16 tolerance
3. **Full model test**: Run Qwen 3.5 MoE with both old and new paths, compare logits
4. **Regression test**: Ensure GGUF and ISQ paths remain unaffected

---

## Appendix: Files Inspected

| File | Project | Relevance |
|---|---|---|
| [qwen3_5_moe.rs](file:///Users/bobking/vllm.rs/src/models/qwen3_5_moe.rs) | vllm.rs | Model forward, decoder layer |
| [moe.rs](file:///Users/bobking/vllm.rs/src/models/layers/moe.rs) | vllm.rs | FusedMoe/FusedMoeFp8 forward |
| [moe.rs](file:///Users/bobking/attention.rs/src/moe.rs) | attention.rs | Rust wrappers for MoE GEMM |
| [moe_gemm.cu](file:///Users/bobking/attention.rs/src/kernels/src/moe_gemm.cu) | attention.rs | Vectorized MoE GEMM kernel |
| [moe_gemm_wmma.cu](file:///Users/bobking/attention.rs/src/kernels/src/moe_gemm_wmma.cu) | attention.rs | WMMA-based grouped GEMM |
| [fp8_moe_cutlass.cu](file:///Users/bobking/attention.rs/src/kernels/src/fp8_moe_cutlass.cu) | attention.rs | CUTLASS FP8 grouped GEMM |
| [paged_attention.rs](file:///Users/bobking/attention.rs/src/paged_attention.rs) | attention.rs | Paged attention + KV cache |
| [attention.rs](file:///Users/bobking/vllm.rs/src/models/layers/attention.rs) | vllm.rs | Attention layer wrapper |
| [qwen3_moe.py](file:///Users/bobking/sglang/python/sglang/srt/models/qwen3_moe.py) | sglang | Qwen3 MoE model + sparse block |
| [qwen3_5.py](file:///Users/bobking/sglang/python/sglang/srt/models/qwen3_5.py) | sglang | Qwen3.5 hybrid model |
| [layer.py](file:///Users/bobking/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py) | sglang | MoE impl dispatch |
| [runner.py](file:///Users/bobking/sglang/python/sglang/srt/layers/moe/moe_runner/runner.py) | sglang | MoE runner orchestration |
| [flashinfer_trtllm.py](file:///Users/bobking/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py) | sglang | FlashInfer TRT-LLM MoE dispatch |
| [flashattention_backend.py](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/flashattention_backend.py) | sglang | FA3/FA4 attention backend |
| [merge_state.py](file:///Users/bobking/sglang/python/sglang/srt/layers/attention/merge_state.py) | sglang | Attention state merge |
