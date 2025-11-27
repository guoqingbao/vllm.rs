use crate::core::sequence::{DecodeSequence, Sequence};
use crate::models::layers::distributed::Id;
use crate::utils::config::{Config, EngineConfig, ModelType};
use crate::utils::downloader::ModelPaths;
#[cfg(feature = "nccl")]
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use candle_core::DType;
use interprocess::local_socket::Stream as LocalStream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RunnerInitRequest {
    pub rank: usize,
    pub dev_id: usize,
    pub num_shards: usize,
    pub model_type: ModelType,
    pub config: Config,
    pub econfig: EngineConfig,
    pub model_pathes: ModelPaths,
    pub is_gguf: bool,
    pub dtype: SerializableDType,
    pub is_rope_i: bool,
    #[cfg(feature = "nccl")]
    pub nccl_id: NcclId,
}

#[derive(Debug, Clone)]
pub struct NcclId(pub Id);

#[cfg(feature = "nccl")]
impl Serialize for NcclId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Detect if JSON serializer
        if serializer.is_human_readable() {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.0.internal().as_ptr() as *const u8,
                    self.0.internal().len(),
                )
            };
            let encoded = STANDARD_NO_PAD.encode(bytes);
            serializer.serialize_str(&encoded)
        } else {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.0.internal().as_ptr() as *const u8,
                    self.0.internal().len(),
                )
            };
            serializer.serialize_bytes(bytes)
        }
    }
}

#[cfg(feature = "nccl")]
impl<'de> Deserialize<'de> for NcclId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s: &str = Deserialize::deserialize(deserializer)?;
            let bytes = STANDARD_NO_PAD
                .decode(s)
                .map_err(serde::de::Error::custom)?;
            if bytes.len() != 128 {
                return Err(serde::de::Error::custom(format!(
                    "Expected 128 bytes but got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0i8; 128];
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), arr.as_mut_ptr() as *mut u8, 128);
            }
            Ok(NcclId(Id::uninit(arr)))
        } else {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = NcclId;

                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "128-byte NCCL ID")
                }

                fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
                where
                    E: serde::de::Error,
                {
                    if v.len() != 128 {
                        return Err(E::custom(format!("Expected 128 bytes but got {}", v.len())));
                    }
                    let mut arr = [0i8; 128];
                    unsafe {
                        std::ptr::copy_nonoverlapping(v.as_ptr(), arr.as_mut_ptr() as *mut u8, 128);
                    }
                    Ok(NcclId(Id::uninit(arr)))
                }
            }

            deserializer.deserialize_bytes(Visitor)
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(u8)]
pub enum SerializableDType {
    U8 = 0,
    U32 = 1,
    I64 = 2,
    BF16 = 3,
    F16 = 4,
    F32 = 5,
    F64 = 6,
}

impl From<DType> for SerializableDType {
    fn from(dt: DType) -> Self {
        match dt {
            DType::U8 => Self::U8,
            DType::U32 => Self::U32,
            DType::I64 => Self::I64,
            DType::BF16 => Self::BF16,
            DType::F16 => Self::F16,
            DType::F32 => Self::F32,
            DType::F64 => Self::F64,
        }
    }
}

impl From<SerializableDType> for DType {
    fn from(sdt: SerializableDType) -> Self {
        match sdt {
            SerializableDType::U8 => DType::U8,
            SerializableDType::U32 => DType::U32,
            SerializableDType::I64 => DType::I64,
            SerializableDType::BF16 => DType::BF16,
            SerializableDType::F16 => DType::F16,
            SerializableDType::F32 => DType::F32,
            SerializableDType::F64 => DType::F64,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct InitAck {
    pub ok: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MessageType {
    /// Sent by main process to initialize the runner.
    Init(RunnerInitRequest),

    /// Sent by runner in response to `Init` with initialization status.
    InitAck(bool),

    LoadingProgress((usize, usize)),

    /// Sent by main process to request prefill on sequences.
    RunPrefill((Vec<Sequence>, bool)),

    /// Sent by main process to request inference on sequences.
    RunDecode((Vec<DecodeSequence>, bool)),

    /// Sent by runner in response to `Run` with generated token IDs.
    RunResponse(Vec<u32>),

    /// Sent by main process to notify the finished decoding sequences.
    FinishDecode(usize),

    /// Optional: runner can send back an error message.
    Error(String),

    Heartbeat,

    // Prefill transfer to PD server
    TransferPrefill(Sequence),
    TransferPrefillResponse(bool),

    // Prefill transfer receive
    ReceivePrefill(usize),
    ReceivePrefillResponse((bool, Option<Sequence>)),

    // Client: Check PD prefill status
    CheckPrefillStatus(usize),
    CheckPrefillStatusResponse(bool),

    KVCacheSwap((HashMap<usize, usize>, bool)),

    KVCacheSwapResponse(bool),

    // send kvcache to client (seq_id, first_token)
    KvCacheSend((Sequence, u32)),
    KvCacheSendResponse(bool),

    // receive kvcache from PD server
    KvCacheReceive(Sequence),
    KvCacheReceiveResponse((bool, u32, usize)),

    // notify PD server to release kvcache
    KvCacheRelease(usize),
    KvCacheReleaseResponse(bool),

    // Server: Check if a prefilled seq need to release kvcache
    CheckKvCacheRelease(usize),
    CheckKvCacheReleaseResponse(bool),

    ClearBlocks(Vec<u32>),
    ClearBlocksResponse(bool),

    UsableMemoryLeft(EngineConfig),
    /// shutdown subprocesses
    Shutdown,
}

//inter-node communication
pub fn send_local(
    streams: &mut Vec<LocalStream>,
    message: &MessageType,
    use_json: bool,
) -> std::io::Result<()> {
    let serialized = if use_json {
        serde_json::to_vec(message).expect("JSON serialization failed")
    } else {
        bincode::serialize(message).expect("Bincode serialization failed")
    };

    for stream in streams.iter_mut() {
        stream.write_all(&(serialized.len() as u32).to_le_bytes())?;
        stream.write_all(&serialized)?;
        stream.flush()?; // Ensure data is sent immediately
                         // Wait for acknowledgment
        let mut ack_buf = [0u8; 1];
        if let Err(e) = stream.read_exact(&mut ack_buf) {
            eprintln!(
                "Timeout waiting for acknowledgment from subprocess: {:?}",
                e
            );
        } else if ack_buf[0] != 1 {
            eprintln!("Unexpected acknowledgment value from subprocess");
        }
    }
    Ok(())
}

pub fn receive_local(stream: &mut LocalStream, use_json: bool) -> std::io::Result<MessageType> {
    let mut length_buf = [0u8; 4];
    stream.read_exact(&mut length_buf)?;
    let length = u32::from_le_bytes(length_buf) as usize;

    let mut serialized = vec![0u8; length];
    stream.read_exact(&mut serialized)?;

    let message: MessageType = if use_json {
        serde_json::from_slice(&serialized).expect("JSON deserialization failed")
    } else {
        bincode::deserialize(&serialized).expect("Bincode deserialization failed")
    };

    // Send acknowledgment
    stream.write_all(&[1])?;
    stream.flush()?;
    Ok(message)
}

pub fn send_and_expect_ack(
    stream: &mut LocalStream,
    msg: &MessageType,
    stage: &str,
    rank: usize,
) -> candle_core::Result<()> {
    use interprocess::TryClone;
    send_local(&mut vec![stream.try_clone()?], msg, true)?;

    crate::log_info!("Waiting runner {} {} response...", rank, stage);

    match receive_local(stream, false)? {
        MessageType::InitAck(true) => Ok(()),
        _ => candle_core::bail!("Runner {} failed during {}", rank, stage),
    }
}

///
/// Defines a function that broadcasts an operation to all runners and expects a `Result<T>`.
/// It handles both `Thread` (direct call) and `Process` (IPC message) runners.
///
/// In Process mode, it expects the response variant to contain the value `T`.
/// It collects all values and verifies they are identical before returning one.
///
#[macro_export]
macro_rules! def_broadcast_message_to_runners {
    (
        // The visibility (e.g., `pub`)
        $vis:vis,
        // The name of the function to create (e.g., `try_receive_kv_cache`)
        $fn_name:ident,
        // The name of the method on the thread-mode runner (e.g., `receive_kv_cache`)
        $thread_fn_name:ident,
        // The arguments for the function (e.g., `(seq: Sequence)`)
        ($($arg_name:ident: $arg_type:ty),*),
        // The MessageType variant to send (e.g., `MessageType::KvCacheReceive`)
        $msg_variant:path,
        // The expression to build the message payload (e.g., `(seq.clone())`)
        ($($msg_arg:expr),*),
        // The MessageType response variant to match (e.g., `MessageType::KvCacheReceiveResponse`)
        $resp_variant:path,
        // The inner return type (e.g., `u32`)
        $return_ty:ty
    ) => {
        $vis fn $fn_name(&self, $($arg_name: $arg_type),*) -> Result<$return_ty>
        where
            $return_ty: std::fmt::Debug + Send,
        {
            match &mut *self.runners.write() {
                RunnerType::Thread(model_runner) => {
                    // Thread Mode: Call the method directly.
                    model_runner.$thread_fn_name($($arg_name),*)
                }
                RunnerType::Process(ref mut runner_streams) => {
                    // Process Mode: Broadcast to all subprocess runners.
                    let cloned_streams: Vec<LocalStream> = runner_streams
                        .iter_mut()
                        .map(|s| s.try_clone().expect("Failed to clone runner stream"))
                        .collect();

                    // Use Rayon for parallel broadcast
                    let all_results: Result<Vec<$return_ty>> = cloned_streams
                        .into_par_iter()
                        .map(|mut stream| {
                            // Send the message
                            send_local(
                                &mut vec![stream.try_clone()?],
                                &$msg_variant($($msg_arg),*),
                                false,
                            )?;

                            // Wait for the response
                            let response = receive_local(&mut stream, false)?;
                            match response {
                                // Match on the expected response containing the value
                                $resp_variant(value) => {
                                    Ok(value)
                                }
                                other => {
                                    candle_core::bail!("Unexpected response for {}: {:?}", stringify!($fn_name), other)
                                }
                            }
                        })
                        .collect(); // Collects into a Result<Vec<T>>

                    // Check that all ranks returned the same value
                    match all_results {
                        Ok(mut values) => {
                            if values.is_empty() {
                                candle_core::bail!("No values received from runners for {}", stringify!($fn_name));
                            }
                            // Pop first element to return, then check rest for consistency
                            let first_val = values.pop().unwrap();
                            Ok(first_val)
                        }
                        Err(e) => Err(e),
                    }
                }
            }
        }
    };
}
