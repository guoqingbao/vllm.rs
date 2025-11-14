// src/core/transfer/cuda.rs
use super::CudaIpcMemHandle;
use candle_core::cuda_backend::cudarc::driver::sys::{lib, CUdeviceptr, CUipcMemHandle};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{Device, Result, Tensor, WithDType};
use std::collections::HashMap;
use std::mem::{ManuallyDrop, MaybeUninit};
/// (Server) Gets a serializable IPC handle for a GPU tensor's memory.
pub(super) fn get_ipc_handle<T: WithDType + candle_core::cuda_backend::CudaDType>(
    tensor: &Tensor,
) -> Result<CudaIpcMemHandle> {
    use candle_core::Storage;
    let (storage, _) = tensor.storage_and_layout();
    let Storage::Cuda(src_storage) = &*storage else {
        candle_core::bail!("Invalid source kvcache storage!")
    };
    let ptr = src_storage.as_cuda_slice::<T>()?.device_ptr();

    let mut handle = MaybeUninit::<CUipcMemHandle>::uninit();
    let handle = unsafe {
        lib()
            .cuIpcGetMemHandle(handle.as_mut_ptr(), *ptr)
            .result()
            .map_err(|e| candle_core::Error::Msg(format!("cuIpcGetMemHandle failed: {e:?}")))?;
        handle.assume_init()
    };

    Ok(CudaIpcMemHandle(
        handle.reserved.to_vec(),
        tensor.shape().dims().to_vec(),
        tensor.dtype().into(),
    ))
}

/// (Client) Opens an IPC handle to get a local tensor pointing to remote GPU memory.
pub(super) fn open_ipc_handle<T: WithDType + candle_core::cuda_backend::CudaDType>(
    handle: &CudaIpcMemHandle,
    device: &Device,
) -> Result<ManuallyDrop<Tensor>> {
    use candle_core::cuda_backend::cudarc::driver::CudaSlice;

    let mut ptr: CUdeviceptr = 0;
    use core::ffi::c_char;
    if handle.0.len() != 64 {
        candle_core::bail!("Invalid CUipcMemHandle handle!");
    }
    let raw_array: [i8; 64] = handle.0.clone().try_into().unwrap();
    let handle_raw = CUipcMemHandle {
        reserved: raw_array.map(|b| b as c_char),
    };
    unsafe {
        lib()
            .cuIpcOpenMemHandle_v2(&mut ptr, handle_raw, 1)
            .result()
            .map_err(|e| candle_core::Error::Msg(format!("cuIpcOpenMemHandle_v2 failed: {e:?}")))?;
    }
    let dev = device.as_cuda_device()?;
    let src_slice = unsafe {
        let slice: CudaSlice<T> = dev.upgrade_device_ptr(ptr, handle.1.iter().sum());
        // std::mem::ManuallyDrop::new(slice)
        slice
    };

    let slice = candle_core::CudaStorage::wrap_cuda_slice(src_slice, dev.clone());
    // We created a virtual Tensor, it stored the remote mem handle, so we should not release it
    Ok(ManuallyDrop::new(Tensor::from_storage(
        candle_core::Storage::Cuda(slice),
        handle.1.clone(),
    )?))
}

/// (Server) Copies specific blocks from a GPU tensor to a new, contiguous CPU tensor.
pub(super) fn copy_blocks_to_cpu(
    gpu_tensor: &Tensor,
    mapping: &HashMap<usize, usize>,
    num_blocks: usize,
) -> Result<Tensor> {
    // Create a new destination tensor on the CPU
    let mut cpu_shape = gpu_tensor.shape().dims().to_vec();
    cpu_shape[0] = num_blocks;
    let cpu_tensor = Tensor::zeros(cpu_shape, gpu_tensor.dtype(), &Device::Cpu)?;

    // Use swap_blocks to copy from sparse locations in `gpu_tensor` to
    //    contiguous locations in `cpu_tensor`.
    //    mapping: { server_block_id -> contiguous_index (0, 1, 2...) }
    attention_rs::cache::swap_blocks(gpu_tensor, &cpu_tensor, mapping)?;

    Ok(cpu_tensor)
}

/// (Client) Converts raw bytes back into a CPU tensor for HtoD copy.
pub(super) fn bytes_to_cpu_tensor(
    bytes: &Vec<u8>,
    num_blocks: usize,
    gpu_tensor_template: &Tensor, // Used for shape/dtype
) -> Result<Tensor> {
    let dtype = gpu_tensor_template.dtype();
    let mut cpu_shape = gpu_tensor_template.shape().dims().to_vec();
    cpu_shape[0] = num_blocks;

    // This is a candle-specific way to create a tensor from raw bytes
    Tensor::from_raw_buffer(bytes, dtype, &cpu_shape, &Device::Cpu)
}

/// (Server) Converts a CPU tensor to raw bytes for network transfer.
pub(super) fn cpu_tensor_to_bytes<T: WithDType>(cpu_tensor: &Tensor) -> Result<Vec<u8>> {
    use candle_core::Storage;
    if !cpu_tensor.is_contiguous() {
        candle_core::bail!("CPU tensor must be contiguous to serialize");
    }
    let (storage, _) = cpu_tensor.storage_and_layout();
    let Storage::Cpu(src_storage) = &*storage else {
        candle_core::bail!("Invalid source kvcache storage!")
    };
    let src_slice: &[T] = src_storage.as_slice()?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            src_slice.as_ptr() as *const u8,
            src_slice.len() * std::mem::size_of::<T>(),
        )
    };
    Ok(bytes.to_vec())
}
