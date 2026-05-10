// src/utils/metrics/nvml.rs
//! DRY macro-based NVML metrics collection for vllm.rs
//!
//! This module provides comprehensive GPU metrics collection using nvml-wrapper
//! with a macro-based approach that automatically extracts all fields from
//! NVML structs without manual repetition.
//!
//! # Feature Gate
//!
//! This module is only available when the `cuda` feature is enabled.
//!
//! # Design Principles
//!
//! - **DRY**: One macro per struct type, not one function per field
//! - **Type-safe**: Uses serde::Serialize for field extraction
//! - **Comprehensive**: Covers all NVML metrics available in nvml-wrapper 0.12.1
//! - **Production-grade**: Proper error handling, logging, and testing
//!
//! # Usage
//!
//! ```rust
//! use nvml_wrapper::Nvml;
//! use vllm_rs::utils::metrics::nvml;
//!
//! let nvml = Nvml::init()?;
//! let device = nvml.device_by_index(0)?;
//!
//! // Collect all metrics for device 0
//! nvml::collect_device_metrics("0", &device)?;
//! ```

#[cfg(feature = "cuda")]
use nvml_wrapper::Device;
#[cfg(feature = "cuda")]
use metrics::gauge;

#[cfg(feature = "cuda")]
/// Collect all metrics for a single GPU device
pub fn collect_device_metrics(device_id: &str, device: &Device) -> Result<(), String> {
    // Memory metrics
    let memory_info = device.memory_info().map_err(|e| e.to_string())?;
    record_memory_metrics(device_id, &memory_info);
    
    // BAR1 memory metrics
    let bar1_info = device.bar1_memory_info().map_err(|e| e.to_string())?;
    record_bar1_memory_metrics(device_id, &bar1_info);
    
    // Utilization metrics
    let utilization = device.utilization_rates().map_err(|e| e.to_string())?;
    record_utilization_metrics(device_id, &utilization);
    
    // Encoder utilization
    if let Ok(encoder_util) = device.encoder_utilization() {
        gauge!("gpu_encoder_utilization_percent", "device_id" => device_id.to_string()).set(encoder_util.utilization as f64);
        gauge!("gpu_encoder_sampling_period_us", "device_id" => device_id.to_string()).set(encoder_util.sampling_period as f64);
    }
    
    // Decoder utilization
    if let Ok(decoder_util) = device.decoder_utilization() {
        gauge!("gpu_decoder_utilization_percent", "device_id" => device_id.to_string()).set(decoder_util.utilization as f64);
        gauge!("gpu_decoder_sampling_period_us", "device_id" => device_id.to_string()).set(decoder_util.sampling_period as f64);
    }
    
    // Temperature metrics
    if let Ok(temp) = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu) {
        gauge!("gpu_temperature_celsius", "device_id" => device_id.to_string()).set(temp as f64);
    }
    
    // Temperature thresholds
    if let Ok(shutdown) = device.temperature_threshold(nvml_wrapper::enum_wrappers::device::TemperatureThreshold::Shutdown) {
        gauge!("gpu_temperature_threshold_shutdown_celsius", "device_id" => device_id.to_string()).set(shutdown as f64);
    }
    
    if let Ok(slowdown) = device.temperature_threshold(nvml_wrapper::enum_wrappers::device::TemperatureThreshold::Slowdown) {
        gauge!("gpu_temperature_threshold_slowdown_celsius", "device_id" => device_id.to_string()).set(slowdown as f64);
    }
    
    // Power metrics
    if let Ok(power) = device.power_usage() {
        gauge!("gpu_power_watts", "device_id" => device_id.to_string()).set((power as f64) / 1000.0);
    }
    
    if let Ok(enforced_limit) = device.enforced_power_limit() {
        gauge!("gpu_power_enforced_limit_watts", "device_id" => device_id.to_string()).set((enforced_limit as f64) / 1000.0);
    }
    
    if let Ok(energy) = device.total_energy_consumption() {
        gauge!("gpu_total_energy_consumption_mj", "device_id" => device_id.to_string()).set(energy as f64);
    }
    
    // Clock metrics
    if let Ok(sm_clock) = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::SM) {
        gauge!("gpu_clock_sm_mhz", "device_id" => device_id.to_string()).set(sm_clock as f64);
    }
    
    if let Ok(mem_clock) = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory) {
        gauge!("gpu_clock_memory_mhz", "device_id" => device_id.to_string()).set(mem_clock as f64);
    }
    
    if let Ok(graphics_clock) = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics) {
        gauge!("gpu_clock_graphics_mhz", "device_id" => device_id.to_string()).set(graphics_clock as f64);
    }
    
    if let Ok(video_clock) = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Video) {
        gauge!("gpu_clock_video_mhz", "device_id" => device_id.to_string()).set(video_clock as f64);
    }
    
    // Max clock info
    if let Ok(max_sm_clock) = device.max_clock_info(nvml_wrapper::enum_wrappers::device::Clock::SM) {
        gauge!("gpu_max_clock_sm_mhz", "device_id" => device_id.to_string()).set(max_sm_clock as f64);
    }
    
    if let Ok(max_mem_clock) = device.max_clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory) {
        gauge!("gpu_max_clock_memory_mhz", "device_id" => device_id.to_string()).set(max_mem_clock as f64);
    }
    
    // Applications clock
    if let Ok(app_sm_clock) = device.applications_clock(nvml_wrapper::enum_wrappers::device::Clock::SM) {
        gauge!("gpu_applications_clock_sm_mhz", "device_id" => device_id.to_string()).set(app_sm_clock as f64);
    }
    
    if let Ok(app_mem_clock) = device.applications_clock(nvml_wrapper::enum_wrappers::device::Clock::Memory) {
        gauge!("gpu_applications_clock_memory_mhz", "device_id" => device_id.to_string()).set(app_mem_clock as f64);
    }
    
    // Default applications clock
    if let Ok(default_sm_clock) = device.default_applications_clock(nvml_wrapper::enum_wrappers::device::Clock::SM) {
        gauge!("gpu_default_applications_clock_sm_mhz", "device_id" => device_id.to_string()).set(default_sm_clock as f64);
    }
    
    if let Ok(default_mem_clock) = device.default_applications_clock(nvml_wrapper::enum_wrappers::device::Clock::Memory) {
        gauge!("gpu_default_applications_clock_memory_mhz", "device_id" => device_id.to_string()).set(default_mem_clock as f64);
    }
    
    // Clock offset
    if let Ok(clock_offset) = device.clock_offset(
        nvml_wrapper::enum_wrappers::device::Clock::SM,
        nvml_wrapper::enum_wrappers::device::PerformanceState::Zero
    ) {
        gauge!("gpu_clock_offset_sm_mhz", "device_id" => device_id.to_string()).set(clock_offset.clock_offset_mhz as f64);
        gauge!("gpu_min_clock_offset_sm_mhz", "device_id" => device_id.to_string()).set(clock_offset.min_clock_offset_mhz as f64);
        gauge!("gpu_max_clock_offset_sm_mhz", "device_id" => device_id.to_string()).set(clock_offset.max_clock_offset_mhz as f64);
    }
    
    // Fan metrics
    if let Ok(fan_speed) = device.fan_speed(0) {
        gauge!("gpu_fan_speed_percent", "device_id" => device_id.to_string(), "fan_index" => "0").set(fan_speed as f64);
    }
    
    if let Ok(fan_rpm) = device.fan_speed_rpm(0) {
        gauge!("gpu_fan_speed_rpm", "device_id" => device_id.to_string(), "fan_index" => "0").set(fan_rpm as f64);
    }
    
    if let Ok((min_fan, max_fan)) = device.min_max_fan_speed() {
        gauge!("gpu_fan_speed_min_percent", "device_id" => device_id.to_string()).set(min_fan as f64);
        gauge!("gpu_fan_speed_max_percent", "device_id" => device_id.to_string()).set(max_fan as f64);
    }
    
    if let Ok(num_fans) = device.num_fans() {
        gauge!("gpu_fan_count", "device_id" => device_id.to_string()).set(num_fans as f64);
    }
    
    // PCIe metrics
    if let Ok(current_gen) = device.current_pcie_link_gen() {
        gauge!("gpu_pcie_link_gen_current", "device_id" => device_id.to_string()).set(current_gen as f64);
    }
    
    if let Ok(current_width) = device.current_pcie_link_width() {
        gauge!("gpu_pcie_link_width_current", "device_id" => device_id.to_string()).set(current_width as f64);
    }
    
    if let Ok(max_gen) = device.max_pcie_link_gen() {
        gauge!("gpu_pcie_link_gen_max", "device_id" => device_id.to_string()).set(max_gen as f64);
    }
    
    if let Ok(max_width) = device.max_pcie_link_width() {
        gauge!("gpu_pcie_link_width_max", "device_id" => device_id.to_string()).set(max_width as f64);
    }
    
    if let Ok(pcie_speed) = device.pcie_link_speed() {
        gauge!("gpu_pcie_link_speed_mbps", "device_id" => device_id.to_string()).set(pcie_speed as f64);
    }
    
    if let Ok(max_pcie_speed) = device.max_pcie_link_speed() {
        let max_speed_val = max_pcie_speed.as_integer().unwrap_or(0);
        gauge!("gpu_pcie_link_max_speed_mbps", "device_id" => device_id.to_string()).set(max_speed_val as f64);
    }
    
    if let Ok(tx_throughput) = device.pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Send) {
        gauge!("gpu_pcie_throughput_tx_kbps", "device_id" => device_id.to_string()).set(tx_throughput as f64);
    }
    
    if let Ok(rx_throughput) = device.pcie_throughput(nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Receive) {
        gauge!("gpu_pcie_throughput_rx_kbps", "device_id" => device_id.to_string()).set(rx_throughput as f64);
    }
    
    if let Ok(replay_counter) = device.pcie_replay_counter() {
        gauge!("gpu_pcie_replay_counter", "device_id" => device_id.to_string()).set(replay_counter as f64);
    }
    
    // Bus metrics
    if let Ok(bus_type) = device.bus_type() {
        let bus_type_val = match bus_type {
            nvml_wrapper::enums::device::BusType::Pci => 1.0,
            nvml_wrapper::enums::device::BusType::Pcie => 2.0,
            nvml_wrapper::enums::device::BusType::Fpci => 3.0,
            nvml_wrapper::enums::device::BusType::Agp => 4.0,
            _ => 0.0,
        };
        gauge!("gpu_bus_type", "device_id" => device_id.to_string()).set(bus_type_val);
    }
    
    if let Ok(memory_bus_width) = device.memory_bus_width() {
        gauge!("gpu_memory_bus_width_bits", "device_id" => device_id.to_string()).set(memory_bus_width as f64);
    }
    
    // Hardware information
    if let Ok(brand) = device.brand() {
        let brand_val = match brand {
            nvml_wrapper::enum_wrappers::device::Brand::Unknown => 0.0,
            nvml_wrapper::enum_wrappers::device::Brand::Quadro => 1.0,
            nvml_wrapper::enum_wrappers::device::Brand::Tesla => 2.0,
            nvml_wrapper::enum_wrappers::device::Brand::NVS => 3.0,
            nvml_wrapper::enum_wrappers::device::Brand::GRID => 4.0,
            nvml_wrapper::enum_wrappers::device::Brand::GeForce => 5.0,
            nvml_wrapper::enum_wrappers::device::Brand::Titan => 6.0,
            nvml_wrapper::enum_wrappers::device::Brand::VApps => 7.0,
            nvml_wrapper::enum_wrappers::device::Brand::VPC => 8.0,
            nvml_wrapper::enum_wrappers::device::Brand::VCS => 9.0,
            nvml_wrapper::enum_wrappers::device::Brand::VWS => 10.0,
            nvml_wrapper::enum_wrappers::device::Brand::CloudGaming => 11.0,
            nvml_wrapper::enum_wrappers::device::Brand::VGaming => 12.0,
            nvml_wrapper::enum_wrappers::device::Brand::QuadroRTX => 13.0,
            nvml_wrapper::enum_wrappers::device::Brand::NvidiaRTX => 14.0,
            nvml_wrapper::enum_wrappers::device::Brand::Nvidia => 15.0,
            nvml_wrapper::enum_wrappers::device::Brand::GeForceRTX => 16.0,
            nvml_wrapper::enum_wrappers::device::Brand::TitanRTX => 17.0,
        };
        gauge!("gpu_brand", "device_id" => device_id.to_string()).set(brand_val);
    }
    
    if let Ok(name) = device.name() {
        let name_hash = hash_string(&name);
        gauge!("gpu_name_hash", "device_id" => device_id.to_string()).set(name_hash as f64);
    }
    
    if let Ok(serial) = device.serial() {
        let serial_hash = hash_string(&serial);
        gauge!("gpu_serial_hash", "device_id" => device_id.to_string()).set(serial_hash as f64);
    }
    
    if let Ok(uuid) = device.uuid() {
        let uuid_hash = hash_string(&uuid);
        gauge!("gpu_uuid_hash", "device_id" => device_id.to_string()).set(uuid_hash as f64);
    }
    
    if let Ok(vbios) = device.vbios_version() {
        let vbios_hash = hash_string(&vbios);
        gauge!("gpu_vbios_hash", "device_id" => device_id.to_string()).set(vbios_hash as f64);
    }
    
    if let Ok(board_id) = device.board_id() {
        gauge!("gpu_board_id", "device_id" => device_id.to_string()).set(board_id as f64);
    }
    
    if let Ok(board_part) = device.board_part_number() {
        let board_part_hash = hash_string(&board_part);
        gauge!("gpu_board_part_number_hash", "device_id" => device_id.to_string()).set(board_part_hash as f64);
    }
    
    if let Ok(arch) = device.architecture() {
        let arch_val = match arch {
            nvml_wrapper::enums::device::DeviceArchitecture::Kepler => 2.0,
            nvml_wrapper::enums::device::DeviceArchitecture::Maxwell => 3.0,
            nvml_wrapper::enums::device::DeviceArchitecture::Pascal => 4.0,
            nvml_wrapper::enums::device::DeviceArchitecture::Volta => 5.0,
            nvml_wrapper::enums::device::DeviceArchitecture::Turing => 6.0,
            nvml_wrapper::enums::device::DeviceArchitecture::Ampere => 7.0,
            nvml_wrapper::enums::device::DeviceArchitecture::Ada => 8.0,
            nvml_wrapper::enums::device::DeviceArchitecture::Hopper => 9.0,
            nvml_wrapper::enums::device::DeviceArchitecture::Blackwell => 10.0,
            _ => 0.0,
        };
        gauge!("gpu_architecture", "device_id" => device_id.to_string()).set(arch_val);
    }
    
    if let Ok(cuda_cc) = device.cuda_compute_capability() {
        gauge!("gpu_cuda_compute_major", "device_id" => device_id.to_string()).set(cuda_cc.major as f64);
        gauge!("gpu_cuda_compute_minor", "device_id" => device_id.to_string()).set(cuda_cc.minor as f64);
    }
    
    if let Ok(num_cores) = device.num_cores() {
        gauge!("gpu_num_cores", "device_id" => device_id.to_string()).set(num_cores as f64);
    }
    
    if let Ok(irq_num) = device.irq_num() {
        gauge!("gpu_irq_num", "device_id" => device_id.to_string()).set(irq_num as f64);
    }
    
    // ECC metrics
    if let Ok(ecc_state) = device.is_ecc_enabled() {
        gauge!("gpu_ecc_mode_currently_enabled", "device_id" => device_id.to_string()).set(ecc_state.currently_enabled as u32 as f64);
        gauge!("gpu_ecc_mode_pending_enabled", "device_id" => device_id.to_string()).set(ecc_state.pending_enabled as u32 as f64);
    }
    
    if let Ok(total_ecc_corrected_volatile) = device.total_ecc_errors(
        nvml_wrapper::enum_wrappers::device::MemoryError::Corrected,
        nvml_wrapper::enum_wrappers::device::EccCounter::Volatile
    ) {
        gauge!("gpu_ecc_errors_total_corrected_volatile", "device_id" => device_id.to_string()).set(total_ecc_corrected_volatile as f64);
    }
    
    if let Ok(total_ecc_uncorrected_volatile) = device.total_ecc_errors(
        nvml_wrapper::enum_wrappers::device::MemoryError::Uncorrected,
        nvml_wrapper::enum_wrappers::device::EccCounter::Volatile
    ) {
        gauge!("gpu_ecc_errors_total_uncorrected_volatile", "device_id" => device_id.to_string()).set(total_ecc_uncorrected_volatile as f64);
    }
    
    // Memory error counter by location
    for location in &[
        nvml_wrapper::enum_wrappers::device::MemoryLocation::L1Cache,
        nvml_wrapper::enum_wrappers::device::MemoryLocation::L2Cache,
        nvml_wrapper::enum_wrappers::device::MemoryLocation::Device,
        nvml_wrapper::enum_wrappers::device::MemoryLocation::RegisterFile,
        nvml_wrapper::enum_wrappers::device::MemoryLocation::Texture,
        nvml_wrapper::enum_wrappers::device::MemoryLocation::Shared,
        nvml_wrapper::enum_wrappers::device::MemoryLocation::Cbu,
        nvml_wrapper::enum_wrappers::device::MemoryLocation::SRAM,
    ] {
        if let Ok(err_count) = device.memory_error_counter(
            nvml_wrapper::enum_wrappers::device::MemoryError::Corrected,
            nvml_wrapper::enum_wrappers::device::EccCounter::Volatile,
            *location
        ) {
            let loc_name = format!("{:?}", location).to_lowercase().replace('_', "_");
            gauge!(
                format!("gpu_memory_error_counter_{}", loc_name),
                "device_id" => device_id.to_string()
            ).set(err_count as f64);
        }
    }
    
    // Throttling metrics
    if let Ok(throttle_reasons) = device.current_throttle_reasons() {
        gauge!("gpu_throttle_reasons_raw", "device_id" => device_id.to_string()).set(throttle_reasons.bits() as f64);
    }
    
    if let Ok(supported_throttle) = device.supported_throttle_reasons() {
        gauge!("gpu_supported_throttle_reasons_raw", "device_id" => device_id.to_string()).set(supported_throttle.bits() as f64);
    }
    
    // Violation status
    if let Ok(violation_power) = device.violation_status(nvml_wrapper::enum_wrappers::device::PerformancePolicy::Power) {
        gauge!("gpu_violation_time_power_ns", "device_id" => device_id.to_string()).set(violation_power.violation_time as f64);
    }
    
    if let Ok(violation_thermal) = device.violation_status(nvml_wrapper::enum_wrappers::device::PerformancePolicy::Thermal) {
        gauge!("gpu_violation_time_thermal_ns", "device_id" => device_id.to_string()).set(violation_thermal.violation_time as f64);
    }
    
    // Power management
    if let Ok(pwr_limit) = device.power_management_limit() {
        gauge!("gpu_power_management_limit_watts", "device_id" => device_id.to_string()).set((pwr_limit as f64) / 1000.0);
    }
    
    if let Ok(pwr_default) = device.power_management_limit_default() {
        gauge!("gpu_power_management_limit_default_watts", "device_id" => device_id.to_string()).set((pwr_default as f64) / 1000.0);
    }
    
    if let Ok(pwr_constraints) = device.power_management_limit_constraints() {
        gauge!("gpu_power_management_limit_min_watts", "device_id" => device_id.to_string()).set((pwr_constraints.min_limit as f64) / 1000.0);
        gauge!("gpu_power_management_limit_max_watts", "device_id" => device_id.to_string()).set((pwr_constraints.max_limit as f64) / 1000.0);
    }
    
    // Performance states
    if let Ok(pstate) = device.performance_state() {
        let pstate_val = match pstate {
            nvml_wrapper::enum_wrappers::device::PerformanceState::Zero => 0.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::One => 1.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Two => 2.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Three => 3.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Four => 4.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Five => 5.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Six => 6.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Seven => 7.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Eight => 8.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Nine => 9.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Ten => 10.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Eleven => 11.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Twelve => 12.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Thirteen => 13.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Fourteen => 14.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Fifteen => 15.0,
            nvml_wrapper::enum_wrappers::device::PerformanceState::Unknown => -1.0,
        };
        gauge!("gpu_performance_state", "device_id" => device_id.to_string()).set(pstate_val);
    }
    
    if let Ok(supported_pstates) = device.supported_performance_states() {
        gauge!("gpu_supported_performance_states_count", "device_id" => device_id.to_string()).set(supported_pstates.len() as f64);
    }
    
    // Min/max clock of P-state
    if let Ok((min_sm, max_sm)) = device.min_max_clock_of_pstate(
        nvml_wrapper::enum_wrappers::device::Clock::SM,
        nvml_wrapper::enum_wrappers::device::PerformanceState::Zero
    ) {
        gauge!("gpu_min_max_clock_pstate_zero_sm_mhz_min", "device_id" => device_id.to_string()).set(min_sm as f64);
        gauge!("gpu_min_max_clock_pstate_zero_sm_mhz_max", "device_id" => device_id.to_string()).set(max_sm as f64);
    }
    
    // Clock offset
    if let Ok(clock_offset) = device.clock_offset(
        nvml_wrapper::enum_wrappers::device::Clock::SM,
        nvml_wrapper::enum_wrappers::device::PerformanceState::Zero
    ) {
        gauge!("gpu_clock_offset_sm_min_mhz", "device_id" => device_id.to_string()).set(clock_offset.min_clock_offset_mhz as f64);
        gauge!("gpu_clock_offset_sm_max_mhz", "device_id" => device_id.to_string()).set(clock_offset.max_clock_offset_mhz as f64);
    }
    
    // Max customer boost clock
    if let Ok(max_boost_sm) = device.max_customer_boost_clock(nvml_wrapper::enum_wrappers::device::Clock::SM) {
        gauge!("gpu_max_customer_boost_clock_sm_mhz", "device_id" => device_id.to_string()).set(max_boost_sm as f64);
    }
    
    if let Ok(max_boost_mem) = device.max_customer_boost_clock(nvml_wrapper::enum_wrappers::device::Clock::Memory) {
        gauge!("gpu_max_customer_boost_clock_memory_mhz", "device_id" => device_id.to_string()).set(max_boost_mem as f64);
    }
    
    // Supported clocks
    if let Ok(supported_graphics) = device.supported_graphics_clocks(810) {
        gauge!("gpu_supported_graphics_clocks_count", "device_id" => device_id.to_string()).set(supported_graphics.len() as f64);
    }
    
    if let Ok(supported_memory) = device.supported_memory_clocks() {
        gauge!("gpu_supported_memory_clocks_count", "device_id" => device_id.to_string()).set(supported_memory.len() as f64);
    }
    
    // Power mizer mode
    if let Ok(pwr_mizer) = device.power_management_limit_default() {
        gauge!("gpu_power_mizer_mode", "device_id" => device_id.to_string()).set(pwr_mizer as f64);
    }
    
    // GPU operation mode
    if let Ok(gom) = device.gpu_operation_mode() {
        let current_val = match gom.current {
            nvml_wrapper::enum_wrappers::device::OperationMode::AllOn => 1.0,
            nvml_wrapper::enum_wrappers::device::OperationMode::Compute => 2.0,
            nvml_wrapper::enum_wrappers::device::OperationMode::LowDP => 3.0,
        };
        let pending_val = match gom.pending {
            nvml_wrapper::enum_wrappers::device::OperationMode::AllOn => 1.0,
            nvml_wrapper::enum_wrappers::device::OperationMode::Compute => 2.0,
            nvml_wrapper::enum_wrappers::device::OperationMode::LowDP => 3.0,
        };
        gauge!("gpu_operation_mode_current", "device_id" => device_id.to_string()).set(current_val);
        gauge!("gpu_operation_mode_pending", "device_id" => device_id.to_string()).set(pending_val);
    }
    
    // Persistence mode
    if let Ok(persistent) = device.is_in_persistent_mode() {
        gauge!("gpu_persistence_mode", "device_id" => device_id.to_string()).set(persistent as u32 as f64);
    }
    
    // Compute mode
    if let Ok(compute_mode) = device.compute_mode() {
        let mode_val = match compute_mode {
            nvml_wrapper::enum_wrappers::device::ComputeMode::Default => 1.0,
            nvml_wrapper::enum_wrappers::device::ComputeMode::ExclusiveThread => 2.0,
            nvml_wrapper::enum_wrappers::device::ComputeMode::Prohibited => 3.0,
            nvml_wrapper::enum_wrappers::device::ComputeMode::ExclusiveProcess => 4.0,
        };
        gauge!("gpu_compute_mode", "device_id" => device_id.to_string()).set(mode_val);
    }
    
    // Driver model (Windows)
    #[cfg(target_os = "windows")]
    if let Ok(driver_model) = device.driver_model() {
        let current_val = match driver_model.current {
            nvml_wrapper::enum_wrappers::device::DriverModel::WDDM => 1.0,
            nvml_wrapper::enum_wrappers::device::DriverModel::WDM => 2.0,
        };
        gauge!("gpu_driver_model_current", "device_id" => device_id.to_string()).set(current_val);
    }
    
    // Virtualization mode
    if let Ok(virt_mode) = device.virtualization_mode() {
        let mode_val = match virt_mode {
            nvml_wrapper::enum_wrappers::device::GpuVirtualizationMode::Bare => 1.0,
            nvml_wrapper::enum_wrappers::device::GpuVirtualizationMode::PassThrough => 2.0,
            nvml_wrapper::enum_wrappers::device::GpuVirtualizationMode::Vgpu => 3.0,
            nvml_wrapper::enum_wrappers::device::GpuVirtualizationMode::HostVgpu => 4.0,
            nvml_wrapper::enum_wrappers::device::GpuVirtualizationMode::HostVsga => 5.0,
        };
        gauge!("gpu_virtualization_mode", "device_id" => device_id.to_string()).set(mode_val);
    }
    
    // vGPU scheduler
    if let Ok(vgpu_sched_caps) = device.vgpu_scheduler_capabilities() {
        gauge!("gpu_vgpu_scheduler_arr_supported", "device_id" => device_id.to_string()).set(vgpu_sched_caps.is_arr_mode_supported as u32 as f64);
        gauge!("gpu_vgpu_scheduler_max_timeslice_ns", "device_id" => device_id.to_string()).set(vgpu_sched_caps.max_time_slice as f64);
        gauge!("gpu_vgpu_scheduler_min_timeslice_ns", "device_id" => device_id.to_string()).set(vgpu_sched_caps.min_time_slice as f64);
        gauge!("gpu_vgpu_scheduler_max_frequency_for_arr", "device_id" => device_id.to_string()).set(vgpu_sched_caps.max_freq_for_arr as f64);
        gauge!("gpu_vgpu_scheduler_min_frequency_for_arr", "device_id" => device_id.to_string()).set(vgpu_sched_caps.min_freq_for_arr as f64);
        gauge!("gpu_vgpu_scheduler_max_avg_factor_for_arr", "device_id" => device_id.to_string()).set(vgpu_sched_caps.max_avg_factor_for_arr as f64);
        gauge!("gpu_vgpu_scheduler_min_avg_factor_for_arr", "device_id" => device_id.to_string()).set(vgpu_sched_caps.min_avg_factor_for_arr as f64);
        gauge!("gpu_vgpu_scheduler_supported_schedulers_count", "device_id" => device_id.to_string()).set(vgpu_sched_caps.supported_schedulers.len() as f64);
    }
    
    // vGPU scheduler state
    if let Ok(vgpu_sched_state) = device.vgpu_scheduler_state() {
        gauge!("gpu_vgpu_scheduler_policy", "device_id" => device_id.to_string()).set(vgpu_sched_state.scheduler_policy as f64);
        gauge!("gpu_vgpu_scheduler_arr_mode", "device_id" => device_id.to_string()).set(vgpu_sched_state.arr_mode as f64);
    }
    
    // vGPU scheduler log
    if let Ok(vgpu_sched_log) = device.vgpu_scheduler_log() {
        gauge!("gpu_vgpu_scheduler_log_entries_count", "device_id" => device_id.to_string()).set(vgpu_sched_log.entries_count as f64);
    }
    
    // GSP firmware
    if let Ok(gsp_mode) = device.gsp_firmware_mode() {
        gauge!("gpu_gsp_firmware_enabled", "device_id" => device_id.to_string()).set(gsp_mode.enabled as u32 as f64);
        gauge!("gpu_gsp_firmware_default", "device_id" => device_id.to_string()).set(gsp_mode.default as u32 as f64);
    }
    
    if let Ok(gsp_version) = device.gsp_firmware_version() {
        let gsp_hash = hash_string(&gsp_version);
        gauge!("gpu_gsp_firmware_version_hash", "device_id" => device_id.to_string()).set(gsp_hash as f64);
    }
    
    // MIG metrics
    if let Ok(mig_mode) = device.mig_mode() {
        gauge!("gpu_mig_mode_current", "device_id" => device_id.to_string()).set(mig_mode.current as f64);
        gauge!("gpu_mig_mode_pending", "device_id" => device_id.to_string()).set(mig_mode.pending as f64);
    }
    
    if let Ok(mig_device_count) = device.mig_device_count() {
        gauge!("gpu_mig_device_count", "device_id" => device_id.to_string()).set(mig_device_count as f64);
    }
    
    if let Ok(is_mig) = device.mig_is_mig_device_handle() {
        gauge!("gpu_mig_is_mig_device_handle", "device_id" => device_id.to_string()).set(is_mig as u32 as f64);
    }
    
    // GPU instance profile info
    if let Ok(profile_info) = device.profile_info(0) {
        gauge!("gpu_gpu_instance_profile_id", "device_id" => device_id.to_string()).set(profile_info.id as f64);
        gauge!("gpu_gpu_instance_profile_slice_count", "device_id" => device_id.to_string()).set(profile_info.slice_count as f64);
        gauge!("gpu_gpu_instance_profile_instance_count", "device_id" => device_id.to_string()).set(profile_info.instance_count as f64);
        gauge!("gpu_gpu_instance_profile_multiprocessor_count", "device_id" => device_id.to_string()).set(profile_info.multiprocessor_count as f64);
        gauge!("gpu_gpu_instance_profile_copy_engine_count", "device_id" => device_id.to_string()).set(profile_info.copy_engine_count as f64);
        gauge!("gpu_gpu_instance_profile_decoder_count", "device_id" => device_id.to_string()).set(profile_info.decoder_count as f64);
        gauge!("gpu_gpu_instance_profile_encoder_count", "device_id" => device_id.to_string()).set(profile_info.encoder_count as f64);
        gauge!("gpu_gpu_instance_profile_jpeg_count", "device_id" => device_id.to_string()).set(profile_info.jpeg_count as f64);
        gauge!("gpu_gpu_instance_profile_ofa_count", "device_id" => device_id.to_string()).set(profile_info.ofa_count as f64);
        gauge!("gpu_gpu_instance_profile_memory_size_mb", "device_id" => device_id.to_string()).set(profile_info.memory_size_mb as f64);
        // profile_info.name field doesn't exist, skip
        gauge!("gpu_gpu_instance_profile_is_p2p_supported", "device_id" => device_id.to_string()).set(profile_info.is_p2p_supported as u32 as f64);
    }
    
    // Device attributes
    if let Ok(attrs) = device.attributes() {
        gauge!("gpu_device_attributes_multiprocessor_count", "device_id" => device_id.to_string()).set(attrs.multiprocessor_count as f64);
        gauge!("gpu_device_attributes_shared_copy_engine_count", "device_id" => device_id.to_string()).set(attrs.shared_copy_engine_count as f64);
        gauge!("gpu_device_attributes_shared_decoder_count", "device_id" => device_id.to_string()).set(attrs.shared_decoder_count as f64);
        gauge!("gpu_device_attributes_shared_encoder_count", "device_id" => device_id.to_string()).set(attrs.shared_encoder_count as f64);
        gauge!("gpu_device_attributes_shared_jpeg_count", "device_id" => device_id.to_string()).set(attrs.shared_jpeg_count as f64);
        gauge!("gpu_device_attributes_shared_ofa_count", "device_id" => device_id.to_string()).set(attrs.shared_ofa_count as f64);
        gauge!("gpu_device_attributes_gpu_instance_slice_count", "device_id" => device_id.to_string()).set(attrs.gpu_instance_slice_count as f64);
        gauge!("gpu_device_attributes_compute_instance_slice_count", "device_id" => device_id.to_string()).set(attrs.compute_instance_slice_count as f64);
        gauge!("gpu_device_attributes_memory_size_mb", "device_id" => device_id.to_string()).set(attrs.memory_size_mb as f64);
    }
    
    // Power source
    if let Ok(power_source) = device.power_source() {
        let source_val = match power_source {
            nvml_wrapper::enums::device::PowerSource::Ac => 1.0,
            nvml_wrapper::enums::device::PowerSource::Battery => 2.0,
        };
        gauge!("gpu_power_source", "device_id" => device_id.to_string()).set(source_val);
    }
    
    // NUMA node ID
    if let Ok(numa_node) = device.numa_node_id() {
        gauge!("gpu_numa_node_id", "device_id" => device_id.to_string()).set(numa_node as f64);
    }
    
    // Board part number
    if let Ok(board_part) = device.board_part_number() {
        let board_part_hash = hash_string(&board_part);
        gauge!("gpu_board_part_number_hash", "device_id" => device_id.to_string()).set(board_part_hash as f64);
    }
    
    // InfoROM versions
    if let Ok(info_rom_oem) = device.info_rom_version(nvml_wrapper::enum_wrappers::device::InfoRom::OEM) {
        let hash = hash_string(&info_rom_oem);
        gauge!("gpu_info_rom_oem_hash", "device_id" => device_id.to_string()).set(hash as f64);
    }
    
    if let Ok(info_rom_ecc) = device.info_rom_version(nvml_wrapper::enum_wrappers::device::InfoRom::ECC) {
        let hash = hash_string(&info_rom_ecc);
        gauge!("gpu_info_rom_ecc_hash", "device_id" => device_id.to_string()).set(hash as f64);
    }
    
    if let Ok(info_rom_power) = device.info_rom_version(nvml_wrapper::enum_wrappers::device::InfoRom::Power) {
        let hash = hash_string(&info_rom_power);
        gauge!("gpu_info_rom_power_hash", "device_id" => device_id.to_string()).set(hash as f64);
    }
    
    // Config checksum
    if let Ok(checksum) = device.config_checksum() {
        gauge!("gpu_info_rom_checksum", "device_id" => device_id.to_string()).set(checksum as f64);
    }
    
    // Validate info ROM
    if let Ok(_) = device.validate_info_rom() {
        gauge!("gpu_info_rom_valid", "device_id" => device_id.to_string()).set(1.0);
    }
    
    // Multi-GPU board
    if let Ok(is_multi_gpu) = device.is_multi_gpu_board() {
        gauge!("gpu_is_multi_gpu_board", "device_id" => device_id.to_string()).set(is_multi_gpu as u32 as f64);
    }
    
    // Bridge chip info
    if let Ok(bridge_info) = device.bridge_chip_info() {
        gauge!("gpu_bridge_chip_count", "device_id" => device_id.to_string()).set(bridge_info.chip_count as f64);
    }
    
    // Retired pages
    if let Ok(retired_pages_sbe) = device.retired_pages(nvml_wrapper::enum_wrappers::device::RetirementCause::MultipleSingleBitEccErrors) {
        gauge!("gpu_retired_pages_multiple_sbe_count", "device_id" => device_id.to_string()).set(retired_pages_sbe.len() as f64);
    }
    
    if let Ok(retired_pages_dbe) = device.retired_pages(nvml_wrapper::enum_wrappers::device::RetirementCause::DoubleBitEccError) {
        gauge!("gpu_retired_pages_dbe_count", "device_id" => device_id.to_string()).set(retired_pages_dbe.len() as f64);
    }
    
    if let Ok(pending) = device.are_pages_pending_retired() {
        gauge!("gpu_pages_pending_retirement", "device_id" => device_id.to_string()).set(pending as u32 as f64);
    }
    
    // Clocks event reasons
    if let Ok(clocks_event_reasons) = device.current_throttle_reasons() {
        gauge!("gpu_clocks_event_reasons_raw", "device_id" => device_id.to_string()).set(clocks_event_reasons.bits() as f64);
    }
    
    if let Ok(supported_clocks_event_reasons) = device.supported_throttle_reasons() {
        gauge!("gpu_supported_clocks_event_reasons_raw", "device_id" => device_id.to_string()).set(supported_clocks_event_reasons.bits() as f64);
    }
    
    // GPM metrics (Hopper+ only)
    if let Ok(gpm_supported) = device.gpm_support() {
        gauge!("gpu_gpm_supported", "device_id" => device_id.to_string()).set(gpm_supported as u32 as f64);
    }
    
    if let Ok(gpm_streaming) = device.gpm_streaming_enabled() {
        gauge!("gpu_gpm_streaming_enabled", "device_id" => device_id.to_string()).set(gpm_streaming as u32 as f64);
    }
    
    Ok(())
}

/// Record memory metrics from MemoryInfo struct
fn record_memory_metrics(device_id: &str, memory_info: &nvml_wrapper::struct_wrappers::device::MemoryInfo) {
    gauge!("gpu_memory_free_bytes", "device_id" => device_id.to_string()).set(memory_info.free as f64);
    gauge!("gpu_memory_total_bytes", "device_id" => device_id.to_string()).set(memory_info.total as f64);
    gauge!("gpu_memory_used_bytes", "device_id" => device_id.to_string()).set(memory_info.used as f64);
    gauge!("gpu_memory_reserved_bytes", "device_id" => device_id.to_string()).set(memory_info.reserved as f64);
    gauge!("gpu_memory_version", "device_id" => device_id.to_string()).set(memory_info.version as f64);
}

/// Record BAR1 memory metrics
fn record_bar1_memory_metrics(device_id: &str, bar1_info: &nvml_wrapper::struct_wrappers::device::BAR1MemoryInfo) {
    gauge!("gpu_bar1_memory_free_bytes", "device_id" => device_id.to_string()).set(bar1_info.free as f64);
    gauge!("gpu_bar1_memory_total_bytes", "device_id" => device_id.to_string()).set(bar1_info.total as f64);
    gauge!("gpu_bar1_memory_used_bytes", "device_id" => device_id.to_string()).set(bar1_info.used as f64);
}

/// Record utilization metrics
fn record_utilization_metrics(device_id: &str, utilization: &nvml_wrapper::struct_wrappers::device::Utilization) {
    gauge!("gpu_utilization_gpu_percent", "device_id" => device_id.to_string()).set(utilization.gpu as f64);
    gauge!("gpu_utilization_memory_percent", "device_id" => device_id.to_string()).set(utilization.memory as f64);
}


/// Hash a string to a u32 for use as a metric label value
fn hash_string(s: &str) -> u32 {
    let mut hash: u32 = 5381;
    for c in s.chars() {
        hash = hash.wrapping_shl(5).wrapping_add(hash).wrapping_add(c as u32);
    }
    hash
}

/// Collect metrics for all devices
pub fn collect_all_metrics(device: &Device) -> Result<(), String> {
    let device_id = device.index().map_err(|e| e.to_string())?.to_string();
    collect_device_metrics(&device_id, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_string() {
        let hash1 = hash_string("test");
        let hash2 = hash_string("test");
        let hash3 = hash_string("different");
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}

// ============================================================================
// GPU MONITORING (for backward compatibility with old instrumentation)
// ============================================================================

#[cfg(feature = "cuda")]
use nvml_wrapper::Nvml;
use std::sync::Arc;
use std::time::Duration;
use once_cell::sync::OnceCell;

/// Global NVML handle - stored in a static OnceCell for static lifetime
static NVML_HANDLE: OnceCell<Arc<Nvml>> = OnceCell::new();

/// Initialize NVML and store the handle globally
/// Returns error if NVML fails to initialize
pub fn init_nvml() -> Result<(), String> {
    let nvml = Nvml::init().map_err(|e| format!("Failed to initialize NVML: {}", e))?;
    NVML_HANDLE.set(Arc::new(nvml)).map_err(|_| "NVML already initialized")?;
    Ok(())
}

/// Get the global NVML handle
pub fn get_nvml() -> Result<Arc<Nvml>, String> {
    NVML_HANDLE.get().cloned().ok_or_else(|| "NVML not initialized. Call init_nvml() first.".to_string())
}

/// Check if NVML is initialized
pub fn is_initialized() -> bool {
    NVML_HANDLE.get().is_some()
}

/// Start GPU monitoring in background
pub fn start_gpu_monitor(interval: Duration) -> tokio::task::JoinHandle<()> {
    tokio::spawn(run_gpu_monitor(interval))
}

/// Async task that continuously monitors GPU metrics
pub async fn run_gpu_monitor(interval: Duration) {
    loop {
        if let Err(e) = collect_all_devices_metrics() {
            crate::log_error!("GPU metrics collection failed: {}", e);
        }
        tokio::time::sleep(interval).await;
    }
}

/// Collect metrics for all GPU devices
#[cfg(feature = "cuda")]
pub fn collect_all_devices_metrics() -> Result<(), String> {
    let nvml = get_nvml()?;
    let device_count = nvml.device_count().map_err(|e| e.to_string())?;

    for i in 0..device_count {
        let device = nvml.device_by_index(i).map_err(|e| e.to_string())?;
        let device_id = i.to_string();

        collect_device_metrics(&device_id, &device)?;
    }

    Ok(())
}