# Integration Guide: Flashinfer CUTLASS Fused MoE into vllm.rs

This document provides a detailed plan and technical guide for integrating the `cutlass_fused_moe` pathway from flashinfer into our Rust stack (`attention.rs` and `vllm.rs`). 

Unlike our current setup which requires 3 separate `moe_gemm` calls (gate, up, down) with multiple kernel launches for FP8 routing, the CUTLASS Fused MoE handles the *entire* MoE pipeline—routing, GEMM 1 (gate+up), activation, GEMM 2 (down), and combine—in a highly optimized, auto-tuned pipeline that supports SM89 (Ada), SM90 (Hopper), and SM100 architectures.

---

## 1. High-Level Workflow of `cutlass_fused_moe`

To understand how to integrate it, we must understand what `cutlass_fused_moe` actually does under the hood. It is not just one CUDA kernel; it is a pipeline orchestrated by the host runner (`MoERunner` in Python, or the C++ equivalent we must build).

The pipeline executed by the CUTLASS MoE runner:

1.  **Routing (if not pre-computed)**: Computes `TopK` indices and expert assignments.
2.  **Permutation / Scatter**: Shuffles the input tokens (`hidden_states`) so that tokens belonging to the same expert are contiguous in memory.
3.  **Grouped GEMM 1 (Gate + Up)**: A single CUTLASS `GemmGrouped` kernel computes both the gate and up projections simultaneously using concatenated weights.
4.  **Activation**: A fused epilogue computes `Up * SiLU(Gate)` (or GeGLU, etc.).
5.  **Grouped GEMM 2 (Down)**: Another CUTLASS `GemmGrouped` kernel computes the down projection.
6.  **Finalize / Combine**: Un-shuffles the outputs back to their original token ordering and multiplies by the router weights (`token_final_scales`).

---

## 2. Build System Dependencies: `.cu` vs `.h`

Currently, `attention.rs` pulls flashinfer from git during `build.rs` and merely includes its headers (e.g., `#include <flashinfer/attention/hopper/variants.cuh>`).

For the CUTLASS Fused MoE, **header-only inclusion is not enough**. The CUTLASS MoE backend in flashinfer relies on a significant amount of compiled C++ Code from Nvidia's TRT-LLM `batched_gemm` infrastructure.

### Files needed from Flashinfer source (`csrc/`):

To build the CUTLASS MoE runner, we must compile specific `.cu` and `.cpp` files from the flashinfer repository and link them into `attention.rs`.

**Required Includes Directories (add to `cc::Build` in `build.rs`):**
- `flashinfer/csrc/nv_internal/` (Contains the embedded TRT-LLM headers)
- `flashinfer/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/`
- `flashinfer/csrc/fused_moe/cutlass_backend/` (Contains the MoE specific headers)

**Required Source Files (add via `.file()` in `build.rs`):**
We must compile the CUTLASS heuristic solver and the instantiation of the fused MoE.
1.  `flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp`
2.  `flashinfer/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu` (This is the file that explicitly instantiates the massive CUTLASS templates for BF16/FP8 combinations).

*Note: Compiling `cutlass_fused_moe_instantiation.cu` will be extremely heavy (it instantiates large templates). It requires high RAM and takes significant time. `nvcc_thread_patterns` should be aware of this.*

---

## 3. The C++/CUDA FFI Wrapper (`attention.rs/src/kernels`)

Since flashinfer's `cutlass_fused_moe` is heavily engineered for PyBind11/Torch, we need to bypass the Python layer and write a direct C FFI wrapper that `attention.rs` can call.

Create a new file: `attention.rs/src/kernels/src/flashinfer_moe_adapter.cu`

```cpp
#include <cuda_runtime.h>
#include <iostream>

// Include the deep tensorrt_llm runner from flashinfer
#include "flashinfer/trtllm/fused_moe/runner.h"

extern "C" {

// Simplified C interface for Rust to call
int flashinfer_cutlass_fused_moe_bf16(
    void* input,                           // [num_tokens, hidden_size]
    int32_t* token_selected_experts,       // [num_tokens, top_k]
    float* token_final_scales,             // [num_tokens, top_k]
    void* fc1_expert_weights,              // [num_experts, 2 * intermediate_size, hidden_size]
    void* fc2_expert_weights,              // [num_experts, hidden_size, intermediate_size]
    void* output,                          // [num_tokens, hidden_size]
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t intermediate_size,
    int32_t num_experts,
    int32_t top_k,
    cudaStream_t stream
) {
    try {
        using namespace tensorrt_llm::kernels::trtllmgen_moe;
        using namespace batchedGemm::trtllm::gen;

        // 1. Initialize the MoE configuration arguments
        MoE::MoERunnerArgs args;
        args.hidden_states = input;
        args.gemm1_weights = fc1_expert_weights;
        args.gemm2_weights = fc2_expert_weights;
        args.output = output;
        
        args.num_tokens = num_tokens;
        args.num_experts = num_experts;
        args.hidden_size = hidden_size;
        args.intermediate_size = intermediate_size;
        args.top_k = top_k;
        
        // Setup data types
        args.mDtypeElt = Dtype::Bfloat16;
        args.mDtypeExpW = Dtype::Bfloat16;
        args.mDtypeOut = Dtype::Bfloat16;
        args.activation_type = MoE::ActivationType::Swiglu;
        args.do_finalize = true;

        // 2. Instantiate the Runner
        // DtypeElt, DtypeW, useDeepSeekFp8
        MoE::Runner runner(Dtype::Bfloat16, Dtype::Bfloat16, false);

        // 3. Allocate Workspace
        // The runner requires a workspace struct for intermediate tensor permutations
        MoE::MoEWorkspace workspace;
        
        // NOTE: In a real implementation, you MUST query the required workspace size:
        // auto [ws1, ws2] = runner.getWorkspaceSizeInBytes(args, configIndex);
        // And allocate device memory for workspace.permuted_hidden_states, etc.
        // For brevity, allocation is omitted here.
        // allocate_workspace(workspace, num_tokens, hidden_size, intermediate_size, ...);

        // 4. Run the auto-tuner to get the best kernel config
        int64_t configIndex = runner.getDefaultValidConfigIndex(
            top_k, hidden_size, intermediate_size, num_experts, num_tokens);

        // 5. Execute the fused MoE pipeline
        int device = 0;
        cudaGetDevice(&device);
        runner.run(args, workspace, device, stream, configIndex, /*enable_pdl=*/true);

        return 0; // Success
    } catch (const std::exception& e) {
        std::cerr << "FlashInfer MoE Error: " << e.what() << std::endl;
        return -1;
    }
}

} // extern "C"
```

---

## 4. Rust Wrapper (`attention.rs/src/moe.rs`)

Next, we bind the C FFI function in Rust.

```rust
use candle_core::{Tensor, Result, Error};
use std::ffi::c_void;

extern "C" {
    fn flashinfer_cutlass_fused_moe_bf16(
        input: *const c_void,
        token_selected_experts: *const i32,
        token_final_scales: *const f32,
        fc1_expert_weights: *const c_void,
        fc2_expert_weights: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_size: i32,
        intermediate_size: i32,
        num_experts: i32,
        top_k: i32,
        stream: cudaStream_t,
    ) -> i32;
}

/// Executes the fully fused CUTLASS MoE pipeline (BF16).
/// 
/// `fc1_expert_weights` must be pre-concatenated: [num_experts, 2 * intermediate, hidden]
pub fn cutlass_fused_moe_bf16(
    input: &Tensor,
    token_selected_experts: &Tensor,
    token_final_scales: &Tensor,
    fc1_expert_weights: &Tensor,
    fc2_expert_weights: &Tensor,
    top_k: usize,
) -> Result<Tensor> {
    let (num_tokens, hidden_size) = input.dims2()?;
    let (num_experts, hidden_size_2, intermediate_size) = fc2_expert_weights.dims3()?;
    
    // Allocate output tensor
    let device = input.device();
    let output = Tensor::zeros((num_tokens, hidden_size), input.dtype(), device)?;

    // Get pointers
    let input_ptr = get_cuda_ptr(input)?;
    let output_ptr = get_cuda_ptr(&output)?;
    // ... get other pointers

    unsafe {
        let stream = get_current_stream(device);
        let status = flashinfer_cutlass_fused_moe_bf16(
            input_ptr,
            expert_idx_ptr,
            scales_ptr,
            fc1_ptr,
            fc2_ptr,
            output_ptr,
            num_tokens as i32,
            hidden_size as i32,
            intermediate_size as i32,
            num_experts as i32,
            top_k as i32,
            stream
        );

        if status != 0 {
            return Err(Error::Msg("CUTLASS Fused MoE execution failed".to_string()));
        }
    }

    Ok(output)
}
```

---

## 5. Integration into `vllm.rs` (`models/layers/moe.rs`)

Finally, we update the `FusedMoe` forward pass. We no longer need the three distinct `moe_gemm` calls or the complex Rust-side routing sorting logic.

```rust
// vllm.rs/src/models/layers/moe.rs

pub struct FusedMoe {
    pub gate_up_w: Tensor, // [num_experts, 2 * intermediate_size, hidden_size]
    pub down_w: Tensor,    // [num_experts, hidden_size, intermediate_size]
    pub gate: ReplicatedLinear, // For calculating routing logits
    pub topk: usize,
    // ...
}

impl FusedMoe {
    pub fn forward(&self, xs: &Tensor, _is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        
        // 1. Calculate routing logits (still done in Rust/standard GEMM)
        let router_logits = self.gate.forward(&xs)?;
        
        // 2. Get top-k experts and their weights (scales)
        // Note: We don't need sorting anymore! The fused kernel handles token permutation.
        let (token_final_scales, token_selected_experts) = topk_softmax(&router_logits, self.topk)?;

        // 3. Dispatch to the single fused kernel
        // This handles: permute -> gate/up gemm -> silu -> down gemm -> scatter output
        let ys = attention_rs::moe::cutlass_fused_moe_bf16(
            xs,
            &token_selected_experts,
            &token_final_scales,
            &self.gate_up_w,
            &self.down_w,
            self.topk,
        )?;

        // 4. Distributed all-reduce if necessary
        let ys = self.tensor_model_parallel_all_reduce(ys)?;

        Ok(ys)
    }
}
```

### FP8 Support (Next Step)
The exact same workflow applies to FP8, using the `MoE::Runner` configured with `Dtype::E4m3` (or `Dtype::E2m1` for block scaling). The C FFI wrapper would need extra arguments for the quantization scale tensors (e.g., `fc1_expert_weights_scale`), but the Core `Runner::run` API remains exactly the same. 

## Summary Checklist for Attention.rs maintainers:
- [ ] Modify `build.rs` to add `flashinfer/csrc/nv_internal/` to includes.
- [ ] Modify `build.rs` to compile `cutlass_fused_moe_instantiation.cu` and `cutlass_heuristic.cpp`.
- [ ] Create `flashinfer_moe_adapter.cu` encapsulating the TRT-LLM `MoE::Runner` logic.
- [ ] Implement C++ workspace allocation logic based on `Runner::getWorkspaceSizeInBytes`.
- [ ] Bind function in `attention.rs/src/moe.rs`.
- [ ] Update `vllm.rs` `FusedMoe` to use the single call.
