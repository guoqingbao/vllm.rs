// src/core/transfer/mod.rs
use crate::core::sequence::Sequence;
use crate::runner::SerializableDType;
use crate::transfer::comm::Communicator;
use candle_core::{DType, Result, Tensor, WithDType};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::JoinHandle;
// Sub-modules for implementation details
mod comm;
#[cfg(feature = "cuda")]
mod cuda_remote;
#[cfg(feature = "python")]
use pyo3::pyclass;

/// Defines the role of the current inference engine instance.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PdRole {
    /// The main instance, handles decoding and orchestrates prefills.
    Client = 1,
    /// A worker instance, dedicated to executing prefills.
    Server = 2,
}

/// The mechanism used to transfer KV cache data.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PdMethod {
    /// Use CUDA IPC handles for D2D transfer (fastest, local machine only).
    LocalIpc = 1,
    /// Use TCP for remote transfer (inter-machine).
    RemoteTcp = 2,
}

#[cfg(not(feature = "python"))]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PdConfig {
    /// Is this instance a Client or a PDServer?
    pub role: PdRole,
    /// The chosen transfer method.
    pub method: PdMethod,
    // The network address for the PD Server or client to listen on/connect to (e.g., "0.0.0.0:9000")
    pub url: Option<String>,
}

/// Configuration for the Transfer sub-system.
#[cfg(feature = "python")]
#[pyclass]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PdConfig {
    /// Is this instance a Client or a PDServer?
    #[pyo3(get, set)]
    pub role: PdRole,
    /// The chosen transfer method.
    #[pyo3(get, set)]
    pub method: PdMethod,
    // The network address for the PD Server or client to listen on/connect to (e.g., "0.0.0.0:9000")
    #[pyo3(get, set)]
    pub url: Option<String>,
}

/// Serializable handle for a CUDA IPC memory region.
/// This is a wrapper around the `cudaIpcMemHandle_t` struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaIpcMemHandle(Vec<i8>, Vec<usize>, SerializableDType); // Simplified as bytes. See cuda.rs for real impl.

/// A handle abstracting *how* to get the KV cache data.
/// This is serialized and sent from the PDServer to the Client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KVTransferHandle {
    /// For local D2D copy. Contains IPC handles for all layers' K/V tensors.
    LocalIpc {
        /// `Vec` is over `num_layers`. Each `(k, v)` is an IPC handle.
        /// These handles are specific to the server's *rank*.
        layer_handles: Vec<(CudaIpcMemHandle, CudaIpcMemHandle)>,
        /// The *server's* GPU block IDs that contain the data for this sequence.
        server_block_ids: Vec<u32>,
    },
    /// For remote HtoD copy. Contains the raw KV cache data.
    RemoteTcp {
        /// Raw bytes of the KV cache data, copied block-by-block from the server's CPU.
        /// `Vec` is over `num_layers`. `(k_bytes, v_bytes)`.
        /// The inner `Vec<u8>` contains all blocks for that layer, concatenated.
        layer_data: Vec<(Vec<u8>, Vec<u8>)>,
        /// The number of blocks (tokens) in the sequence.
        num_blocks: usize,
    },
}

/// Data sent from the PDServer to the Client upon prefill completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinishedPrefillData {
    pub seq_id: usize,
    pub first_token: u32,
    pub transfer_handle: KVTransferHandle,
}

/// Messages for communication between Client and PDServer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferMessage {
    /// Client -> Server: Request to prefill a new sequence.
    TransferPrefill(Sequence),
    /// Server -> Client: Prefill finished, contains KV cache handle.
    TransferKvCache(FinishedPrefillData),
}

/// The main Transfer struct that orchestrates PD disaggregation.
#[allow(dead_code)]
pub struct Transfer {
    config: PdConfig,
    /// (Client-side) Holds completed prefill data from the server.
    finished_data: Arc<RwLock<HashMap<usize, FinishedPrefillData>>>,
    /// (Server-side) Holds incoming prefill requests from the client.
    pending_prefills: Arc<Mutex<VecDeque<Sequence>>>,
    /// Handle to the background communication thread.
    comm_handle: Option<JoinHandle<()>>,
    rank: usize,
    communicator: Communicator,
}

impl Transfer {
    /// Creates a new Transfer instance and starts its communication thread.
    pub fn new(
        config: PdConfig,
        rank: usize,
        model_loaded: Arc<AtomicBool>,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<Self> {
        let finished_data = Arc::new(RwLock::new(HashMap::new()));
        let pending_prefills = Arc::new(Mutex::new(VecDeque::new()));

        // Clone Arcs for the background thread
        let thread_finished_data = finished_data.clone();
        let thread_pending_prefills = pending_prefills.clone();
        let communicator = Communicator::new(config.clone(), config.role.clone(), rank);
        // Start the background communication thread
        let thread_comm = communicator.clone();
        let comm_handle = Some(std::thread::spawn(move || {
            thread_comm.run_listener_loop(
                thread_pending_prefills,
                thread_finished_data,
                model_loaded,
                stop_flag,
            );
        }));

        Ok(Self {
            config,
            finished_data,
            pending_prefills,
            comm_handle,
            rank,
            communicator,
        })
    }

    pub fn is_client(&self) -> bool {
        matches!(self.config.role, PdRole::Client)
    }

    pub fn is_server(&self) -> bool {
        matches!(self.config.role, PdRole::Server)
    }

    // --- Client-side API ---

    /// (Client) Sends a sequence to the PDServer for prefill.
    pub fn transfer_prefill(&self, seq: &Sequence) -> Result<bool> {
        if !self.is_client() {
            return Ok(false);
        }
        crate::log_warn!("transfer_prefill Seq {}", seq.id);
        self.communicator
            .send(&TransferMessage::TransferPrefill(seq.clone()))
    }

    /// (Client) Checks if a specific prefill has finished.
    pub fn check_prefill_finished(&self, seq_id: usize) -> Result<bool> {
        {
            let guard = self.finished_data.read();
            crate::log_warn!(
                "check_prefill_finished Seq {}, existing {:?}",
                seq_id,
                guard.keys()
            );
        }

        Ok(self.finished_data.write().contains_key(&seq_id))
    }

    /// (Client) Receives the KV cache data and copies it into local GPU blocks.
    pub fn receive_kv_cache(
        &self,
        seq: &Sequence,
        local_gpu_cache: &Vec<(Tensor, Tensor)>,
    ) -> Result<(bool, u32)> {
        crate::log_warn!("receive_kv_cache Seq {}", seq.id);

        let status = self.check_prefill_finished(seq.id)?;
        if !status {
            candle_core::bail!("Unable to receive kvcache from the PD server since this sequence is not prefill completed!")
        }

        fn read_data<T: WithDType + candle_core::cuda_backend::CudaDType>(
            sf: &Transfer,
            seq: &Sequence,
            local_gpu_cache: &Vec<(Tensor, Tensor)>,
        ) -> Result<(bool, u32)> {
            let local_gpu_ids = seq.block_table.clone();
            let local_device = local_gpu_cache[0].0.device();

            let data_guard = sf.finished_data.read();
            let data = &data_guard.get(&seq.id).unwrap();
            let token = data.first_token;
            match &data.transfer_handle {
                KVTransferHandle::LocalIpc {
                    layer_handles,
                    server_block_ids,
                } => {
                    // This mapping copies from the server's block N to the client's block N.
                    let mapping: HashMap<usize, usize> = server_block_ids
                        .iter()
                        .zip(local_gpu_ids)
                        .map(|(server_id, local_id)| (*server_id as usize, local_id as usize))
                        .collect();

                    #[cfg(feature = "cuda")]
                    for (i, (k_handle, v_handle)) in layer_handles.iter().enumerate() {
                        // Open the remote IPC handles to get local pointers to remote memory
                        let remote_k_tensor =
                            cuda_remote::open_ipc_handle::<T>(k_handle, local_device)?;
                        let remote_v_tensor =
                            cuda_remote::open_ipc_handle::<T>(v_handle, local_device)?;

                        crate::log_warn!("got remote_k_tensor {:?}", remote_k_tensor);
                        crate::log_warn!("got remote_v_tensor {:?}", remote_v_tensor);

                        // Use swap_blocks to perform a D2D (peer) copy
                        // Copy from: remote_k_tensor, To: local_gpu_cache
                        attention_rs::cache::swap_blocks(
                            &remote_k_tensor,
                            &local_gpu_cache[i].0,
                            &mapping,
                        )?;
                        attention_rs::cache::swap_blocks(
                            &remote_v_tensor,
                            &local_gpu_cache[i].1,
                            &mapping,
                        )?;
                    }
                }
                KVTransferHandle::RemoteTcp {
                    layer_data,
                    num_blocks,
                } => {
                    if local_gpu_ids.len() < *num_blocks {
                        candle_core::bail!("Not enough blocks allocated to receive KV cache");
                    }

                    // This mapping copies from the *N-th* block in the received data
                    // to the client's *N-th* allocated block.
                    let mapping: HashMap<usize, usize> = (0..*num_blocks)
                        .zip(local_gpu_ids)
                        .map(|(i, local_id)| (i, local_id as usize))
                        .collect();

                    #[cfg(feature = "cuda")]
                    for (i, (k_bytes, v_bytes)) in layer_data.into_iter().enumerate() {
                        // Reconstruct CPU tensors from raw bytes
                        let (local_k_cache, local_v_cache) = &local_gpu_cache[i];

                        let remote_k_tensor =
                            cuda_remote::bytes_to_cpu_tensor(k_bytes, *num_blocks, local_k_cache)?;
                        let remote_v_tensor =
                            cuda_remote::bytes_to_cpu_tensor(v_bytes, *num_blocks, local_v_cache)?;

                        // Use swap_blocks to perform an HtoD copy
                        // Copy from: remote_k_tensor (CPU), To: local_k_cache (GPU)
                        attention_rs::cache::swap_blocks(
                            &remote_k_tensor,
                            &local_k_cache,
                            &mapping,
                        )?;
                        attention_rs::cache::swap_blocks(
                            &remote_v_tensor,
                            &local_v_cache,
                            &mapping,
                        )?;
                    }
                }
            }
            Ok((true, token))
        }
        let dtype = local_gpu_cache[0].0.dtype();
        match dtype {
            DType::F16 => read_data::<half::f16>(&self, seq, local_gpu_cache),
            DType::BF16 => read_data::<half::bf16>(&self, seq, local_gpu_cache),
            DType::U8 => read_data::<u8>(&self, seq, local_gpu_cache),
            _ => candle_core::bail!("Invalid kvcache dtype!"),
        }
    }

    // --- Server-side API ---

    /// (Server) Tries to receive a new prefill request from the queue.
    pub fn try_receive_prefill_request(&self) -> Option<Sequence> {
        self.pending_prefills.lock().pop_front()
    }

    /// (Server) Creates the KVTransferHandle and sends it to the client.
    pub fn transfer_kv_cache(
        &self,
        seq: &Sequence,
        server_gpu_cache: &Vec<(Tensor, Tensor)>,
        first_token: u32,
    ) -> Result<bool> {
        crate::log_warn!(
            "transfer_kv_cache Seq {}, first token {}, config {:?}",
            seq.id,
            first_token,
            self.config,
        );

        // if !self.is_server() {
        //     return Ok(false);
        // }

        fn transfer_data<T: WithDType + candle_core::cuda_backend::CudaDType>(
            sf: &Transfer,
            config: &PdConfig,
            seq: &Sequence,
            server_gpu_cache: &Vec<(Tensor, Tensor)>,
            first_token: u32,
        ) -> Result<bool> {
            let transfer_handle = match config.method {
                PdMethod::LocalIpc => {
                    let mut layer_handles = Vec::new();
                    #[cfg(feature = "cuda")]
                    for (k_tensor, v_tensor) in server_gpu_cache.iter() {
                        // Get IPC handles for the *entire* layer tensors
                        let k_handle = cuda_remote::get_ipc_handle::<T>(k_tensor)?;
                        let v_handle = cuda_remote::get_ipc_handle::<T>(v_tensor)?;
                        layer_handles.push((k_handle, v_handle));
                    }
                    crate::log_warn!(
                        "KVTransferHandle::LocalIpc Seq {}, block_table {:?}",
                        seq.id,
                        seq.block_table.clone()
                    );
                    KVTransferHandle::LocalIpc {
                        layer_handles,
                        server_block_ids: seq.block_table.clone(),
                    }
                }
                PdMethod::RemoteTcp => {
                    let mut layer_data = Vec::new();
                    let server_block_ids = &seq.block_table;

                    // This mapping copies *from* the server's block IDs *to*
                    // a contiguous index (0, 1, 2...) in a *new CPU tensor*.
                    let mapping: HashMap<usize, usize> = server_block_ids
                        .iter()
                        .enumerate()
                        .map(|(i, &server_id)| (server_id as usize, i))
                        .collect();

                    #[cfg(feature = "cuda")]
                    for (k_tensor, v_tensor) in server_gpu_cache.iter() {
                        // Copy blocks from GPU to a new contiguous CPU tensor
                        let k_cpu_tensor = cuda_remote::copy_blocks_to_cpu(
                            k_tensor,
                            &mapping,
                            server_block_ids.len(),
                        )?;
                        let v_cpu_tensor = cuda_remote::copy_blocks_to_cpu(
                            v_tensor,
                            &mapping,
                            server_block_ids.len(),
                        )?;

                        // Get raw bytes from the CPU tensors
                        layer_data.push((
                            cuda_remote::cpu_tensor_to_bytes::<T>(&k_cpu_tensor)?,
                            cuda_remote::cpu_tensor_to_bytes::<T>(&v_cpu_tensor)?,
                        ));
                    }
                    KVTransferHandle::RemoteTcp {
                        layer_data,
                        num_blocks: seq.block_table.len(),
                    }
                }
            };

            let msg = TransferMessage::TransferKvCache(FinishedPrefillData {
                seq_id: seq.id,
                first_token,
                transfer_handle,
            });
            crate::log_warn!(
                "Sending TransferMessage::TransferKvCache for Seq {}",
                seq.id,
            );
            // Send the finished data back to the client
            sf.communicator.send(&msg)
        }

        let dtype = server_gpu_cache[0].0.dtype();
        match dtype {
            DType::F16 => {
                transfer_data::<half::f16>(&self, &self.config, seq, server_gpu_cache, first_token)?
            }
            DType::BF16 => transfer_data::<half::bf16>(
                &self,
                &self.config,
                seq,
                server_gpu_cache,
                first_token,
            )?,
            DType::U8 => {
                transfer_data::<u8>(&self, &self.config, seq, server_gpu_cache, first_token)?
            }
            _ => candle_core::bail!("Invalid kvcache dtype!"),
        };
        Ok(true)
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        // TODO: Add logic to gracefully shut down the comm_handle thread
    }
}
