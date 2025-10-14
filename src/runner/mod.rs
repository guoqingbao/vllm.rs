use crate::core::sequence::{DecodeSequence, Sequence};
use crate::models::layers::distributed::Id;
use crate::utils::config::{Config, EngineConfig, ModelType};
use crate::utils::downloader::ModelPaths;
#[cfg(feature = "nccl")]
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};
use candle_core::DType;
use interprocess::local_socket::Stream as LocalStream;
use serde::{Deserialize, Serialize};
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
