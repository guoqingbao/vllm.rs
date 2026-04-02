// src/models/layers/moe.rs
use crate::models::layers::distributed::{shard, AllReduce, Comm};
use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::config::QuantConfig;
use attention_rs::moe;
use attention_rs::moe::moe_gemm_fp8;
use candle_core::Module;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Result, Tensor, D,
};
use candle_nn::var_builder::Shard;
use either::Either;
use std::rc::Rc;
use std::sync::Arc;

/// Quantization method for MoE experts
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantMethod {
    /// No quantization (FP32/FP16/BF16)
    None,
    /// FP8 quantization
    Fp8,
    /// ISQ quantization (GGUF)
    Isq,
    /// QBlock quantization
    QBlock,
    /// QKT quantization
    QKT,
}

impl std::fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantMethod::None => write!(f, "none"),
            QuantMethod::Fp8 => write!(f, "fp8"),
            QuantMethod::Isq => write!(f, "isq"),
            QuantMethod::QBlock => write!(f, "qblock"),
            QuantMethod::QKT => write!(f, "qkt"),
        }
    }
}

/// Expert parameter mapping entry
/// (param_name, weight_name, expert_id, shard_id)
pub type ExpertMappingEntry = (String, String, usize, String);

/// Build initial global physical to logical expert mapping
///
/// This mirrors Python vLLM's `EplbState.build_initial_global_physical_to_logical_map`.
/// Creates a mapping from physical expert IDs to logical expert IDs.
///
/// For redundant experts, the mapping wraps around using modulo:
/// physical_to_logical[physical_id] = physical_id % num_routed_experts
///
/// # Arguments
/// * `num_routed_experts` - Number of routed (logical) experts
/// * `num_redundant_experts` - Number of redundant experts
///
/// # Returns
/// A vector where index is physical expert ID and value is logical expert ID
pub fn build_initial_global_physical_to_logical_map(
    num_routed_experts: usize,
    num_redundant_experts: usize,
) -> Vec<usize> {
    // Start with identity mapping for routed experts
    let mut map: Vec<usize> = (0..num_routed_experts).collect();

    // Add redundant experts mapping (wrap around using modulo)
    for i in 0..num_redundant_experts {
        map.push(i % num_routed_experts);
    }

    map
}

/// Check if any parameter names contain the base_layer prefix
///
/// # Arguments
/// * `params` - List of (param_name, param_tensor) tuples
///
/// # Returns
/// true if any parameter name contains ".base_layer."
pub fn has_base_layer_prefix(params: &[(&str, candle_core::Tensor)]) -> bool {
    params.iter().any(|(name, _)| name.contains(".base_layer."))
}

/// Generate expert parameter mapping for Qwen3Next/Qwen3.5 MoE models
///
/// This mirrors the Python vLLM `SharedFusedMoE.make_expert_params_mapping` behavior.
/// Returns a list of tuples: (param_name, weight_name, expert_id, shard_id)
///
/// - param_name: The parameter name in the model (e.g., "experts.w13_" or "experts.w2_")
/// - weight_name: The weight name in the checkpoint (e.g., "experts.0.gate_proj" or "experts.0.up_proj")
/// - expert_id: The physical expert ID (0 to num_physical_experts-1)
/// - shard_id: The shard identifier ("w1", "w2", or "w3")
pub fn make_expert_params_mapping(
    num_experts: usize,
    num_redundant_experts: usize,
) -> Vec<ExpertMappingEntry> {
    let num_physical_experts = num_experts + num_redundant_experts;

    // Build physical to logical expert mapping
    let physical_to_logical_map = build_initial_global_physical_to_logical_map(num_experts, num_redundant_experts);

    let mut mappings = Vec::new();

    for expert_id in 0..num_physical_experts {
        let logical_expert_id = physical_to_logical_map[expert_id];

        // For each expert, generate mappings for w1, w2, w3 shards
        // w1/w3 correspond to gate_proj/up_proj (combined as w13)
        // w2 corresponds to down_proj

        // Gate projection (w1)
        mappings.push((
            format!("experts.w13_"),
            format!("experts.{logical_expert_id}.gate_proj"),
            expert_id,
            "w1".to_string(),
        ));

        // Up projection (w3)
        mappings.push((
            format!("experts.w13_"),
            format!("experts.{logical_expert_id}.up_proj"),
            expert_id,
            "w3".to_string(),
        ));

        // Down projection (w2)
        mappings.push((
            format!("experts.w2_"),
            format!("experts.{logical_expert_id}.down_proj"),
            expert_id,
            "w2".to_string(),
        ));
    }

    mappings
}

/// Generate expert parameter mapping with base_layer prefix support
///
/// This mirrors Python vLLM's behavior where checkpoints may have weights
/// under a `base_layer.` prefix (e.g., from PEFT adapters).
///
/// # Arguments
/// * `num_experts` - Number of routed experts
/// * `num_redundant_experts` - Number of redundant experts
/// * `has_base_layer` - Whether the checkpoint uses base_layer prefix
///
/// # Returns
/// List of (param_name, weight_name, expert_id, shard_id) tuples
pub fn make_expert_params_mapping_with_base_layer(
    num_experts: usize,
    num_redundant_experts: usize,
    has_base_layer: bool,
) -> Vec<ExpertMappingEntry> {
    let num_physical_experts = num_experts + num_redundant_experts;

    let base_layer_prefix = if has_base_layer { "base_layer." } else { "" };
    let physical_to_logical_map = build_initial_global_physical_to_logical_map(num_experts, num_redundant_experts);

    let mut mappings = Vec::new();

    for expert_id in 0..num_physical_experts {
        let logical_expert_id = physical_to_logical_map[expert_id];

        // Gate projection (w1)
        mappings.push((
            format!("experts.{base_layer_prefix}w13_"),
            format!("experts.{logical_expert_id}.gate_proj.{base_layer_prefix}"),
            expert_id,
            "w1".to_string(),
        ));

        // Up projection (w3)
        mappings.push((
            format!("experts.{base_layer_prefix}w13_"),
            format!("experts.{logical_expert_id}.up_proj.{base_layer_prefix}"),
            expert_id,
            "w3".to_string(),
        ));

        // Down projection (w2)
        mappings.push((
            format!("experts.{base_layer_prefix}w2_"),
            format!("experts.{logical_expert_id}.down_proj.{base_layer_prefix}"),
            expert_id,
            "w2".to_string(),
        ));
    }

    mappings
}

/// Generate expert parameter mapping for FP8 weight scales
///
/// FP8 quantized experts have weight_scale/weight_scale_inv tensors
/// that need to be mapped to the correct experts.
///
/// # Arguments
/// * `num_experts` - Number of routed experts
/// * `num_redundant_experts` - Number of redundant experts
/// * `is_weight_scale` - Whether this is for weight scale (vs activation scale)
///
/// # Returns
/// List of (param_name, weight_name, expert_id, shard_id) tuples for FP8 scales
pub fn make_expert_fp8_weight_scale_mapping(
    num_experts: usize,
    num_redundant_experts: usize,
    is_weight_scale: bool,
) -> Vec<ExpertMappingEntry> {
    let num_physical_experts = num_experts + num_redundant_experts;
    let physical_to_logical_map = build_initial_global_physical_to_logical_map(num_experts, num_redundant_experts);

    let scale_name = if is_weight_scale { "weight_scale_inv" } else { "weight_scale" };

    let mut mappings = Vec::new();

    for expert_id in 0..num_physical_experts {
        let logical_expert_id = physical_to_logical_map[expert_id];

        // Gate projection weight scale (w1)
        mappings.push((
            format!("experts.w13_{scale_name}"),
            format!("experts.{logical_expert_id}.gate_proj.{scale_name}"),
            expert_id,
            "w1".to_string(),
        ));

        // Up projection weight scale (w3)
        mappings.push((
            format!("experts.w13_{scale_name}"),
            format!("experts.{logical_expert_id}.up_proj.{scale_name}"),
            expert_id,
            "w3".to_string(),
        ));

        // Down projection weight scale (w2)
        mappings.push((
            format!("experts.w2_{scale_name}"),
            format!("experts.{logical_expert_id}.down_proj.{scale_name}"),
            expert_id,
            "w2".to_string(),
        ));
    }

    mappings
}

/// Generate expert parameter mapping for FP8 activation scales
///
/// # Arguments
/// * `num_experts` - Number of routed experts
/// * `num_redundant_experts` - Number of redundant experts
///
/// # Returns
/// List of (param_name, weight_name, expert_id, shard_id) tuples for FP8 activation scales
pub fn make_expert_fp8_activation_scale_mapping(
    num_experts: usize,
    num_redundant_experts: usize,
) -> Vec<ExpertMappingEntry> {
    let num_physical_experts = num_experts + num_redundant_experts;
    let physical_to_logical_map = build_initial_global_physical_to_logical_map(num_experts, num_redundant_experts);

    let scale_name = "activation_scale_inv";

    let mut mappings = Vec::new();

    for expert_id in 0..num_physical_experts {
        let logical_expert_id = physical_to_logical_map[expert_id];

        // Gate projection activation scale (w1)
        mappings.push((
            format!("experts.w13_{scale_name}"),
            format!("experts.{logical_expert_id}.gate_proj.{scale_name}"),
            expert_id,
            "w1".to_string(),
        ));

        // Up projection activation scale (w3)
        mappings.push((
            format!("experts.w13_{scale_name}"),
            format!("experts.{logical_expert_id}.up_proj.{scale_name}"),
            expert_id,
            "w3".to_string(),
        ));

        // Down projection activation scale (w2)
        mappings.push((
            format!("experts.w2_{scale_name}"),
            format!("experts.{logical_expert_id}.down_proj.{scale_name}"),
            expert_id,
            "w2".to_string(),
        ));
    }

    mappings
}

/// Load fused expert weights using expert parameter mapping
///
/// This helper function mirrors Python vLLM's `load_fused_expert_weights` behavior.
/// It loads expert weights using the VarBuilderX path resolution based on expert mapping.
///
/// # Arguments
/// * `vb` - VarBuilderX for the base module path
/// * `experts_vb` - VarBuilderX for the experts submodule path
/// * `expert_id` - The physical expert ID to load
/// * `weight_name` - The weight name in checkpoint (e.g., "gate_proj", "up_proj", "down_proj")
/// * `shard_id` - The shard identifier (e.g., "w1", "w3", "w2")
/// * `num_experts` - Total number of routed experts in the model
///
/// # Returns
/// * `Result<Tensor>` - The loaded tensor
pub fn load_fused_expert_weights(
    _vb: &VarBuilderX,
    experts_vb: &VarBuilderX,
    expert_id: usize,
    weight_name: &str,
    _shard_id: &str,
    num_experts: usize,
) -> Result<Tensor> {
    // Get the logical expert ID from physical expert ID
    let physical_to_logical_map = build_initial_global_physical_to_logical_map(num_experts, 0);
    let logical_expert_id = physical_to_logical_map[expert_id];

    // Construct the path to the expert's weight
    let expert_vb = experts_vb.pp(format!("{}", logical_expert_id).as_str());
    let weight_vb = expert_vb.pp(weight_name);

    // Load the weight tensor
    weight_vb.get_with_hints_dtype((), "weight", Shard::default(), DType::F32)
}

#[derive(Clone, Copy, Debug)]
enum PackedGateUpLayout {
    // [experts, hidden, 2*intermediate]
    HiddenPacked,
    // [experts, 2*intermediate, hidden]
    InterPacked,
}

#[derive(Clone, Copy, Debug)]
enum PackedDownLayout {
    // [experts, intermediate, hidden] -> transpose to [experts, hidden, intermediate]
    InterHidden,
    // [experts, hidden, intermediate] -> already in expected GEMM layout.
    HiddenInter,
}

fn resolve_packed_gate_up_layout(cfg: &Config) -> Result<PackedGateUpLayout> {
    let arch = cfg
        .architectures
        .as_ref()
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("");

    // Qwen3.5 MoE / Qwen3Next checkpoints store gate_up as [experts, 2*intermediate, hidden].
    if matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
    ) {
        return Ok(PackedGateUpLayout::InterPacked);
    }

    let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
    if cfg.hidden_size == moe_cfg.moe_intermediate_size * 2 {
        candle_core::bail!(
            "Ambiguous packed gate_up_proj layout for arch {:?}: hidden_size ({}) == 2 * moe_intermediate_size ({}). \
Please add an architecture-specific layout mapping.",
            arch,
            cfg.hidden_size,
            moe_cfg.moe_intermediate_size
        );
    }

    Ok(PackedGateUpLayout::HiddenPacked)
}

fn resolve_packed_down_layout(cfg: &Config) -> PackedDownLayout {
    let arch = cfg
        .architectures
        .as_ref()
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("");

    // Qwen3.5 MoE / Qwen3Next checkpoints store down_proj as [experts, hidden, intermediate].
    if matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
    ) {
        PackedDownLayout::HiddenInter
    } else {
        PackedDownLayout::InterHidden
    }
}

#[allow(dead_code)]
pub struct FusedMoe {
    gate: Linear,
    gate_up_w: Tensor,
    down_w: Tensor,
    w_size_n: usize,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoe {
        /// Load expert weights using expert parameter mapping
        /// This is the core method that uses the expert mapping functions to load weights
        /// from checkpoints with varying weight naming conventions.
        pub fn load_packed_with_mapping(
            cfg: &Config,
            experts_vb: &VarBuilderX,
            comm: Rc<Comm>,
            num_experts: usize,
            num_redundant_experts: usize,
        ) -> Result<(Tensor, Tensor, Tensor)> {
            let num_physical_experts = num_experts + num_redundant_experts;
            let mut gate_experts = Vec::with_capacity(num_physical_experts);
            let mut up_experts = Vec::with_capacity(num_physical_experts);
            let mut down_experts = Vec::with_capacity(num_physical_experts);

            // Get the logical expert mapping
            let physical_to_logical_map = build_initial_global_physical_to_logical_map(num_experts, num_redundant_experts);

            crate::log_info!("MoE experts: using per-expert checkpoint format with logical mapping");
            crate::log_info!("MoE experts: num_experts={}, num_redundant={}, num_physical={}", num_experts, num_redundant_experts, num_physical_experts);

            for physical_expert_id in 0..num_physical_experts {
                let logical_expert_id = physical_to_logical_map[physical_expert_id];

                // Construct the expert path using logical expert ID
                let expert_vb = experts_vb.pp(format!("{}", logical_expert_id).as_str());

                // Load gate_proj (w1 shard)
                let gate_expert = expert_vb.pp("gate_proj").get_with_hints_dtype(
                    (cfg.moe_cfg.as_ref().unwrap().moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ).map_err(|e| candle_core::Error::msg(format!("Failed to load gate_proj for logical_expert_id={} (physical_expert_id={}): {}", logical_expert_id, physical_expert_id, e)))?;
                gate_experts.push(gate_expert);

                // Load up_proj (w3 shard)
                let up_expert = expert_vb.pp("up_proj").get_with_hints_dtype(
                    (cfg.moe_cfg.as_ref().unwrap().moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ).map_err(|e| candle_core::Error::msg(format!("Failed to load up_proj for logical_expert_id={} (physical_expert_id={}): {}", logical_expert_id, physical_expert_id, e)))?;
                up_experts.push(up_expert);

                // Load down_proj (w2 shard)
                let down_expert = expert_vb.pp("down_proj").get_with_hints_dtype(
                    (cfg.hidden_size, cfg.moe_cfg.as_ref().unwrap().moe_intermediate_size),
                    "weight",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::F32,
                ).map_err(|e| candle_core::Error::msg(format!("Failed to load down_proj for logical_expert_id={} (physical_expert_id={}): {}", logical_expert_id, physical_expert_id, e)))?;
                down_experts.push(down_expert);
            }

            Ok((
                Tensor::stack(&gate_experts, 0)?,
                Tensor::stack(&up_experts, 0)?,
                Tensor::stack(&down_experts, 0)?,
            ))
        }

        /// Legacy method - loads experts directly by physical index without mapping
        /// Use this only for checkpoints where experts are stored with physical indices.
        /// For new checkpoints with logical expert mapping, use load_packed instead.
        pub fn load_packed_physical(
        cfg: &Config,
        experts_vb: VarBuilderX,
        comm: Rc<Comm>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();
        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        let (gate_experts, up_experts, down_experts) = if experts_vb.has_key("gate_up_proj") {
            match &experts_vb.0 {
                Either::Left(vb) => {
                    let gate_up_layout = resolve_packed_gate_up_layout(cfg)?;
                    let (gate_expert, up_expert) = match gate_up_layout {
                        // [experts, hidden, 2*intermediate]
                        PackedGateUpLayout::HiddenPacked => {
                            let gate = vb
                                .get_with_hints(
                                    (
                                        num_experts,
                                        cfg.hidden_size,
                                        moe_cfg.moe_intermediate_size * 2,
                                    ),
                                    "gate_up_proj",
                                    shard(2, comm.rank(), comm.world_size() * 2),
                                )?
                                .t()?
                                .contiguous()?;

                            let up = vb
                                .get_with_hints(
                                    (
                                        num_experts,
                                        cfg.hidden_size,
                                        moe_cfg.moe_intermediate_size * 2,
                                    ),
                                    "gate_up_proj",
                                    shard(
                                        2,
                                        comm.rank() + comm.world_size(),
                                        comm.world_size() * 2,
                                    ),
                                )?
                                .t()?
                                .contiguous()?;
                            (gate, up)
                        }
                        // [experts, 2*intermediate, hidden]
                        PackedGateUpLayout::InterPacked => {
                            let gate = vb
                                .get_with_hints(
                                    (
                                        num_experts,
                                        moe_cfg.moe_intermediate_size * 2,
                                        cfg.hidden_size,
                                    ),
                                    "gate_up_proj",
                                    shard(1, comm.rank(), comm.world_size() * 2),
                                )?
                                .contiguous()?;
                            let up = vb
                                .get_with_hints(
                                    (
                                        num_experts,
                                        moe_cfg.moe_intermediate_size * 2,
                                        cfg.hidden_size,
                                    ),
                                    "gate_up_proj",
                                    shard(
                                        1,
                                        comm.rank() + comm.world_size(),
                                        comm.world_size() * 2,
                                    ),
                                )?
                                .contiguous()?;
                            (gate, up)
                        }
                    };
                    let down_layout = resolve_packed_down_layout(cfg);
                    let down_expert = match down_layout {
                        PackedDownLayout::InterHidden => vb
                            .get_with_hints(
                                (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                                "down_proj",
                                shard(1, comm.rank(), comm.world_size()),
                            )?
                            .t()?
                            .contiguous()?,
                        PackedDownLayout::HiddenInter => vb
                            .get_with_hints(
                                (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                                "down_proj",
                                shard(2, comm.rank(), comm.world_size()),
                            )?
                            .contiguous()?,
                    };
                    let (_, gate_n, gate_k) = gate_expert.dims3()?;
                    let (_, up_n, up_k) = up_expert.dims3()?;
                    let (_, down_n, down_k) = down_expert.dims3()?;
                    if gate_n != up_n
                        || gate_k != up_k
                        || gate_k != cfg.hidden_size
                        || down_n != cfg.hidden_size
                        || down_k != gate_n
                    {
                        candle_core::bail!(
                            "Invalid packed MoE tensor shapes after loading: gate={:?}, up={:?}, down={:?}, hidden_size={}, arch={:?}. \
This usually means packed down_proj / gate_up_proj layout was interpreted incorrectly.",
                            gate_expert.shape(),
                            up_expert.shape(),
                            down_expert.shape(),
                            cfg.hidden_size,
                            cfg.architectures
                        );
                    }
                    (gate_expert, up_expert, down_expert)
                }
                _ => candle_core::bail!("invalid varbuild or quant config!"),
            }
        } else {
            for i in 0..num_experts {
                let experts_vb = experts_vb.pp(format!("{}", i).as_str());
                match &experts_vb.0 {
                    Either::Left(vb) => {
                        // n x k format
                        let gate_expert = vb.pp("gate_proj").get_with_hints(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                            shard(0, comm.rank(), comm.world_size()),
                        )?;
                        let up_expert = vb.pp("up_proj").get_with_hints(
                            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "weight",
                            shard(0, comm.rank(), comm.world_size()),
                        )?;
                        let down_expert = vb.pp("down_proj").get_with_hints(
                            (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                            "weight",
                            shard(1, comm.rank(), comm.world_size()),
                        )?;
                        gate_experts.push(gate_expert);
                        up_experts.push(up_expert);
                        down_experts.push(down_expert);
                    }
                    _ => candle_core::bail!("invalid varbuild or quant config!"),
                }
            }
            (
                Tensor::stack(&gate_experts, 0)?,
                Tensor::stack(&up_experts, 0)?,
                Tensor::stack(&down_experts, 0)?,
            )
        };
        Ok((gate_experts, up_experts, down_experts))
    }

    /// Load expert weights with automatic dispatch based on checkpoint format
    ///
    /// This function uses tensor dimension checking to determine the checkpoint
    /// format, similar to Python vLLM's approach:
    /// - Fused experts (gate_up_proj): 3D tensor shape (num_experts, hidden, 2*intermediate)
    /// - Per-expert (gate_proj/up_proj/down_proj): 2D tensor shape (intermediate, hidden)
    ///
    /// The dispatch logic:
    /// 1. Try load_packed_with_mapping() first (per-expert with logical mapping)
    /// 2. If that fails, fallback to load_packed_physical() (packed or per-expert with physical IDs)
    pub fn load_packed(
        cfg: &Config,
        experts_vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_method: &QuantMethod,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        // Determine checkpoint format by checking tensor dimensions
        // Fused experts (gate_up_proj): 3D tensor (num_experts, hidden, 2*intermediate)
        // Per-expert (gate_proj/up_proj/down_proj): 2D tensors (intermediate, hidden)
        //
        // We check tensor dimensions by attempting to load a sample expert weight
        // and examining its shape, similar to Python vLLM's approach.

        // First try experts.0.gate_proj (safetensors per-expert format)
        let expert_vb = experts_vb.pp("0");
        let has_expert_weights = match expert_vb.0 {
            either::Either::Left(ref vb) => vb.contains_tensor("gate_proj"),
            either::Either::Right(ref vb) => vb.tensor_shape("gate_proj").is_some(),
        };

        if has_expert_weights {
            // Per-expert format: experts.0.gate_proj exists
            crate::log_info!("MoE experts: trying per-expert checkpoint format (experts.0.gate_proj)");
            crate::log_info!("MoE experts: quant_method={}, num_experts={}", quant_method, num_experts);
            match Self::load_packed_with_mapping(cfg, &experts_vb, comm.clone(), num_experts, 0) {
                Ok(tensors) => {
                    crate::log_info!("MoE experts: per-expert with logical mapping succeeded");
                    return Ok(tensors);
                }
                Err(e) => {
                    crate::log_info!("MoE experts: per-expert with logical mapping failed: {:?}", e);
                    crate::log_info!("MoE experts: falling back to physical per-expert loading");
                }
            }
            // Fallback to physical per-expert loading
            Self::load_packed_physical(cfg, experts_vb, comm)
        } else {
            // Fused expert format: gate_up_proj exists
            crate::log_info!("MoE experts: using fused expert checkpoint format (gate_up_proj detected)");
            crate::log_info!("MoE experts: quant_method={}, num_experts={}", quant_method, num_experts);
            Self::load_packed_physical(cfg, experts_vb, comm)
        }
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        assert!(
            cfg.quantization_config.is_none(),
            "Invalid quantization format!"
        );
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &None,
            &None,
            dtype,
        )?;

        let (gate_w, up_w, down_w) = Self::load_packed(cfg, vb.pp("experts"), comm.clone(), &QuantMethod::None)?;
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 1)?;
        let world_size = comm.world_size();
        let w_size_n = gate_up_w.dim(1)? / 2;
        Ok(Self {
            gate,
            gate_up_w,
            down_w,
            w_size_n,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) = attention_rs::topk::topk_softmax(
            &router_logits.to_dtype(DType::F32)?,
            self.num_experts_per_tok,
        )?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }

        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        //out (M, top_k, N)
        let gate_up = moe::moe_gemm(
            &xs,
            &self.gate_up_w,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?;

        let gate = gate_up
            .narrow(candle_core::D::Minus1, 0, self.w_size_n)?
            .contiguous()?;
        let up = gate_up
            .narrow(candle_core::D::Minus1, self.w_size_n, self.w_size_n)?
            .contiguous()?;
        //(M * top_k, N // 2)
        let down_inputs = (up * gate.apply(&self.act)?)?;

        //view(M, top_k, K) -> sum -> (M, K)
        let mut ys = moe::moe_gemm(
            &down_inputs,
            &self.down_w,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}

pub struct FusedMoeGGUF {
    gate: Linear,
    gate_experts: Arc<QTensor>,
    up_experts: Arc<QTensor>,
    down_experts: Arc<QTensor>,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeGGUF {
    pub fn new_repack(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let gate_ws = match &vb.pp("ffn_gate_inp").0 {
            Either::Right(v) => v
                .get((num_experts, cfg.hidden_size), "weight")?
                .dequantize(v.device())?,
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        let gate = Linear::new(gate_ws, None, &None)?;

        let (gate_experts, up_experts, down_experts) = match &vb.0 {
            Either::Right(v) => (
                v.pp("ffn_gate_exps")
                    .get(
                        (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                    )?
                    .dequantize_f16(&v.device())?,
                v.pp("ffn_up_exps")
                    .get(
                        (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                    )?
                    .dequantize_f16(&v.device())?,
                v.pp("ffn_down_exps")
                    .get(
                        (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                        "weight",
                    )?
                    .dequantize_f16(&v.device())?,
            ),
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        let (ggml_dtype, block_size) = (GgmlDType::Q4K, GgmlDType::Q4K.block_size());

        let moe_intermediate_chunk =
            if moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0 {
                ((moe_cfg.moe_intermediate_size / comm.world_size() + block_size - 1) / block_size)
                    * block_size
            } else {
                moe_cfg.moe_intermediate_size / comm.world_size()
            };

        let cur_chunk_size = if comm.rank() * moe_intermediate_chunk + moe_intermediate_chunk
            < moe_cfg.moe_intermediate_size
        {
            moe_intermediate_chunk
        } else {
            moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk
        };

        assert!(cur_chunk_size > 0 && cur_chunk_size % block_size == 0,
            "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}! \n \
            \t*****Tips: you may try these gglm types: `q8_0` (recommend), `q4_0`, `q4_1`, `q5_0`, `q5_1` (with smaller block_size 32)",
            moe_cfg.moe_intermediate_size,
            comm.world_size(),
            block_size
        );

        let (gate_experts, up_experts, down_experts) = (
            gate_experts
                .narrow(1, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
            up_experts
                .narrow(1, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
            down_experts
                .narrow(2, comm.rank() * moe_intermediate_chunk, cur_chunk_size)?
                .contiguous()?,
        );
        let gate_experts = Arc::new(QTensor::quantize(&gate_experts, ggml_dtype)?);
        let up_experts = Arc::new(QTensor::quantize(&up_experts, ggml_dtype)?);
        let down_experts = Arc::new(QTensor::quantize(&down_experts, GgmlDType::Q8_0)?);

        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        if comm.world_size() > 1 {
            return Self::new_repack(cfg, vb, comm.clone(), dtype);
        }
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();
        let gate_ws = match &vb.pp("ffn_gate_inp").0 {
            Either::Right(v) => v
                .get((num_experts, cfg.hidden_size), "weight")?
                .dequantize(v.device())?,
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        let gate = Linear::new(gate_ws, None, &None)?;

        let (gate_experts, up_experts, down_experts) = match &vb.0 {
            Either::Right(v) => (
                v.pp("ffn_gate_exps").get(
                    (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                )?,
                v.pp("ffn_up_exps").get(
                    (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                )?,
                v.pp("ffn_down_exps").get(
                    (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                )?,
            ),
            _ => {
                panic!("Invalid varbuilder!");
            }
        };

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: cfg.hidden_act,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size: 1,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) =
            attention_rs::topk::topk_softmax(&router_logits, self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }
        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        let ys = {
            let gate = moe::moe_gemm_gguf(
                &xs,
                &self.gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let up = moe::moe_gemm_gguf(
                &xs,
                &self.up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;

            let down_inputs = (up * gate.apply(&self.act)?)?;
            moe::moe_gemm_gguf(
                &down_inputs,
                &self.down_experts,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?
        };
        let mut ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}

pub struct FusedMoeISQ {
    gate: Linear,
    gate_experts: QTensor,
    up_experts: QTensor,
    down_experts: QTensor,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeISQ {
    pub fn new(cfg: &Config, vb: VarBuilderX, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let mut quant_type = match cfg.quant.as_ref().unwrap().as_str() {
            "q40" | "q4_0" => GgmlDType::Q4_0,
            "q4" | "q41" | "q4_1" => GgmlDType::Q4_1,
            "q50" | "q5_0" => GgmlDType::Q5_0,
            "q5" | "q51" | "q5_1" => GgmlDType::Q5_1,
            "q8" | "q80" | "q8_0" => GgmlDType::Q8_0,
            "q2k" | "q2_k" => GgmlDType::Q2K,
            "q3k" | "q3_k" => GgmlDType::Q3K,
            "q4k" | "q4_k" => GgmlDType::Q4K,
            "q5k" | "q5_k" => GgmlDType::Q5K,
            "q6k" | "q6_k" => GgmlDType::Q6K,
            _ => panic!("Unsupported GGML data type!"),
        };

        let get_moe_intermediate_chunk = |blk_size: usize| -> usize {
            let base = moe_cfg.moe_intermediate_size / comm.world_size();
            if base % blk_size != 0 {
                ((base + blk_size - 1) / blk_size) * blk_size
            } else {
                base
            }
        };

        let mut block_size = quant_type.block_size();
        if comm.world_size() > 1
            && moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0
        {
            //in case of the experts unable to be split under qkk format,
            //and asymetric split also not workable, switch to q8_0
            let chunk = get_moe_intermediate_chunk(block_size);
            if (moe_cfg.moe_intermediate_size - chunk) % (comm.world_size() - 1) != 0 {
                quant_type = GgmlDType::Q8_0;
                block_size = quant_type.block_size();
            }
        }
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &None,
            &None,
            DType::F32,
        )?;

        let (gate_experts, up_experts, down_experts) = if moe_cfg.moe_intermediate_size
            / comm.world_size()
            % block_size
            == 0
        {
            FusedMoe::load_packed(cfg, vb.pp("experts"), comm.clone(), &QuantMethod::Isq)?
        } else {
            let experts_vb = vb.pp("experts");
            let mut gate_experts = Vec::with_capacity(num_experts);
            let mut up_experts = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);

            let moe_intermediate_chunk = get_moe_intermediate_chunk(block_size);

            let (gate_experts, up_experts, down_experts) = if experts_vb.has_key("gate_up_proj") {
                match &experts_vb.0 {
                    Either::Left(vb) => {
                        let gate_up_layout = resolve_packed_gate_up_layout(cfg)?;
                        let (gate_expert, up_expert) = match gate_up_layout {
                            // [experts, hidden, 2*intermediate]
                            PackedGateUpLayout::HiddenPacked => {
                                let gate = vb
                                    .get_with_hints(
                                        (
                                            num_experts,
                                            cfg.hidden_size,
                                            moe_cfg.moe_intermediate_size * 2,
                                        ),
                                        "gate_up_proj",
                                        shard(2, 0, 2),
                                    )?
                                    .t()?
                                    .contiguous()?;
                                let up = vb
                                    .get_with_hints(
                                        (
                                            num_experts,
                                            cfg.hidden_size,
                                            moe_cfg.moe_intermediate_size * 2,
                                        ),
                                        "gate_up_proj",
                                        shard(2, 1, 2),
                                    )?
                                    .t()?
                                    .contiguous()?;
                                (gate, up)
                            }
                            // [experts, 2*intermediate, hidden]
                            PackedGateUpLayout::InterPacked => {
                                let gate = vb
                                    .get_with_hints(
                                        (
                                            num_experts,
                                            moe_cfg.moe_intermediate_size * 2,
                                            cfg.hidden_size,
                                        ),
                                        "gate_up_proj",
                                        shard(1, 0, 2),
                                    )?
                                    .contiguous()?;
                                let up = vb
                                    .get_with_hints(
                                        (
                                            num_experts,
                                            moe_cfg.moe_intermediate_size * 2,
                                            cfg.hidden_size,
                                        ),
                                        "gate_up_proj",
                                        shard(1, 1, 2),
                                    )?
                                    .contiguous()?;
                                (gate, up)
                            }
                        };

                        let down_expert = match vb.get_with_hints(
                            (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "down_proj",
                            Shard::default(),
                        ) {
                            // Layout A: [experts, intermediate, hidden] -> transpose to [experts, hidden, intermediate]
                            Ok(w) => w.t()?.contiguous()?,
                            // Layout B: [experts, hidden, intermediate] -> already in expected GEMM layout.
                            Err(_) => vb
                                .get_with_hints(
                                    (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                                    "down_proj",
                                    Shard::default(),
                                )?
                                .contiguous()?,
                        };
                        (gate_expert, up_expert, down_expert)
                    }
                    Either::Right(_) => panic!("invalid varbuild!"),
                }
            } else {
                //pack experts
                for i in 0..num_experts {
                    let experts_vb = experts_vb.pp(format!("{}", i).as_str());
                    match &experts_vb.0 {
                        Either::Left(vb) => {
                            let gate_expert = vb.pp("gate_proj").get_with_hints(
                                (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                                "weight",
                                Shard::default(),
                            )?;
                            let up_expert = vb.pp("up_proj").get_with_hints(
                                (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                                "weight",
                                Shard::default(),
                            )?;
                            let down_expert = vb.pp("down_proj").get_with_hints(
                                (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                                "weight",
                                Shard::default(),
                            )?;
                            gate_experts.push(gate_expert);
                            up_experts.push(up_expert);
                            down_experts.push(down_expert);
                        }
                        Either::Right(_) => panic!("invalid varbuild!"),
                    }
                }

                (
                    Tensor::stack(&gate_experts, 0)?,
                    Tensor::stack(&up_experts, 0)?,
                    Tensor::stack(&down_experts, 0)?,
                )
            };

            let mut last_remain_size = moe_intermediate_chunk;
            if comm.rank() * moe_intermediate_chunk + moe_intermediate_chunk
                < moe_cfg.moe_intermediate_size
            {
            } else {
                last_remain_size =
                    moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk;
                assert!(last_remain_size > 0 && last_remain_size % block_size == 0,
                    "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}! \n \
                    \t*****Tips: you may try these gglm types: `q8_0` (recommend), `q4_0`, `q4_1`, `q5_0`, `q5_1` (with smaller block_size 32)",
                    moe_cfg.moe_intermediate_size,
                    comm.world_size(),
                    block_size
                );
            };

            let gate_experts =
                gate_experts.narrow(1, comm.rank() * moe_intermediate_chunk, last_remain_size)?;
            let up_experts =
                up_experts.narrow(1, comm.rank() * moe_intermediate_chunk, last_remain_size)?;
            let down_experts =
                down_experts.narrow(2, comm.rank() * moe_intermediate_chunk, last_remain_size)?;

            (gate_experts, up_experts, down_experts)
        };

        let gate_experts = QTensor::quantize(&gate_experts, quant_type)?;
        let up_experts = QTensor::quantize(&up_experts, quant_type)?;
        let down_experts = QTensor::quantize(&down_experts, GgmlDType::Q8_0)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) =
            attention_rs::topk::topk_softmax(&router_logits, self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }
        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        let ys = {
            let gate = moe::moe_gemm_gguf(
                &xs,
                &self.gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let up = moe::moe_gemm_gguf(
                &xs,
                &self.up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let down_inputs = (up * gate.apply(&self.act)?)?;
            moe::moe_gemm_gguf(
                &down_inputs,
                &self.down_experts,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?
        };
        let mut ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}

pub struct FusedMoeFp8 {
    gate: Linear,
    gate_up_experts: Tensor,
    gate_up_experts_scale: Tensor,
    down_experts: Tensor,
    down_experts_scale: Tensor,
    w_size_n: usize,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    block_size: Vec<usize>,
}

impl FusedMoeFp8 {
    pub fn new(
        cfg: &Config,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        let moe_cfg = cfg.moe_cfg.as_ref().expect("MoE config is not available!");
        let num_experts = moe_cfg.num_experts.unwrap();

        let block_size = quant_cfg
            .weight_block_size
            .clone()
            .unwrap_or(vec![128, 128]);
        if block_size.len() != 2 {
            candle_core::bail!("FusedMoeFp8: weight_block_size must have 2 elements");
        }
        let by = block_size[0]; // for scale_n
        let bx = block_size[1]; // for scale_k

        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &None,
            &None,
            dtype,
        )?;

        let experts_vb = vb.pp("experts");

        let (
            gate_experts,
            gate_experts_scale,
            up_experts,
            up_experts_scale,
            down_experts,
            down_experts_scale,
        ) = if experts_vb.has_key("gate_up_proj") {
            // Qwen3 VL approach.
            match &experts_vb.0 {
                Either::Left(vb) => {
                    let gate_weight = vb
                        .get_with_hints_dtype(
                            (
                                num_experts,
                                cfg.hidden_size,
                                moe_cfg.moe_intermediate_size * 2,
                            ),
                            "gate_up_proj",
                            shard(2, comm.rank(), comm.world_size() * 2),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let up_weight = vb
                        .get_with_hints_dtype(
                            (
                                num_experts,
                                cfg.hidden_size,
                                moe_cfg.moe_intermediate_size * 2,
                            ),
                            "gate_up_proj",
                            shard(2, comm.rank() + comm.world_size(), comm.world_size() * 2),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let scale_n = (cfg.hidden_size + by - 1) / by;
                    let scale_k = (moe_cfg.moe_intermediate_size * 2 + bx - 1) / bx;

                    let gate_up_scale = vb.get_with_hints_dtype(
                        (num_experts, scale_n, scale_k),
                        "gate_up_proj_scale_inv",
                        Default::default(),
                        DType::F32,
                    )?;

                    let inter_blocks = moe_cfg.moe_intermediate_size / bx;
                    let local_inter_blocks = inter_blocks / comm.world_size();
                    let start_blocks = comm.rank() * local_inter_blocks;

                    let gate_s_t = gate_up_scale.narrow(2, 0, inter_blocks)?.contiguous()?;
                    let up_s_t = gate_up_scale
                        .narrow(2, inter_blocks, inter_blocks)?
                        .contiguous()?;

                    let gate_s = gate_s_t
                        .narrow(2, start_blocks, local_inter_blocks)?
                        .t()?
                        .contiguous()?;
                    let up_s = up_s_t
                        .narrow(2, start_blocks, local_inter_blocks)?
                        .t()?
                        .contiguous()?;

                    let down_weight = vb
                        .get_with_hints_dtype(
                            (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "down_proj",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let down_s = vb
                        .get_with_hints_dtype(
                            (num_experts, scale_k / 2, scale_n),
                            "down_proj_scale_inv",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    (gate_weight, gate_s, up_weight, up_s, down_weight, down_s)
                }
                _ => candle_core::bail!("FusedMoeFp8: Invalid varbuilder for packed loading"),
            }
        } else {
            // Per-expert loading
            let mut gate_experts = Vec::with_capacity(num_experts);
            let mut gate_experts_scale = Vec::with_capacity(num_experts);
            let mut up_experts = Vec::with_capacity(num_experts);
            let mut up_experts_scale = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);
            let mut down_experts_scale = Vec::with_capacity(num_experts);
            for i in 0..num_experts {
                let expert_vb = experts_vb.pp(format!("{}", i).as_str());
                let gate_weight = expert_vb.pp("gate_proj").get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let sn = (moe_cfg.moe_intermediate_size + by - 1) / by;
                let sk = (cfg.hidden_size + bx - 1) / bx;
                let gate_s = match expert_vb.pp("gate_proj").get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp("gate_proj").get_with_hints_dtype(
                        (sn, sk),
                        "weight_scale_inv",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                let up_weight = expert_vb.pp("up_proj").get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let sn = (moe_cfg.moe_intermediate_size + by - 1) / by;
                let sk = (cfg.hidden_size + bx - 1) / bx;
                let up_s = match expert_vb.pp("up_proj").get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp("up_proj").get_with_hints_dtype(
                        (sn, sk),
                        "weight_scale_inv",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                let down_weight = expert_vb.pp("down_proj").get_with_hints_dtype(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let sn = (cfg.hidden_size + by - 1) / by;
                let sk = (moe_cfg.moe_intermediate_size + bx - 1) / bx;
                let down_s = match expert_vb.pp("down_proj").get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp("down_proj").get_with_hints_dtype(
                        (sn, sk),
                        "weight_scale_inv",
                        shard(1, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                gate_experts.push(gate_weight);
                gate_experts_scale.push(gate_s);
                up_experts.push(up_weight);
                up_experts_scale.push(up_s);
                down_experts.push(down_weight);
                down_experts_scale.push(down_s);
            }

            (
                Tensor::stack(&gate_experts, 0)?,
                Tensor::stack(&gate_experts_scale, 0)?,
                Tensor::stack(&up_experts, 0)?,
                Tensor::stack(&up_experts_scale, 0)?,
                Tensor::stack(&down_experts, 0)?,
                Tensor::stack(&down_experts_scale, 0)?,
            )
        };
        let gate_up_experts = Tensor::cat(&[&gate_experts, &up_experts], 1)?;
        let gate_up_experts_scale = Tensor::cat(&[&gate_experts_scale, &up_experts_scale], 1)?;
        let w_size_n = gate_up_experts.dim(1)? / 2;
        Ok(Self {
            gate,
            gate_up_experts,
            gate_up_experts_scale,
            down_experts,
            down_experts_scale,
            w_size_n,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            block_size: vec![by, bx],
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) = attention_rs::topk::topk_softmax(
            &router_logits.to_dtype(DType::F32)?,
            self.num_experts_per_tok,
        )?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }

        let xs = if xs.dtype() == DType::F32 {
            xs.to_dtype(self.dtype)?
        } else {
            xs.clone()
        };

        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        let gate_up = moe_gemm_fp8(
            &xs,
            &self.gate_up_experts,
            &self.gate_up_experts_scale,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            self.block_size[0],
            self.block_size[1],
            is_prefill,
        )?;

        let gate = gate_up
            .narrow(candle_core::D::Minus1, 0, self.w_size_n)?
            .contiguous()?;
        let up = gate_up
            .narrow(candle_core::D::Minus1, self.w_size_n, self.w_size_n)?
            .contiguous()?;
        let down_inputs = (up * gate.apply(&self.act)?)?;

        let mut ys = moe_gemm_fp8(
            &down_inputs,
            &self.down_experts,
            &self.down_experts_scale,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            self.block_size[0],
            self.block_size[1],
            is_prefill,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }
}
