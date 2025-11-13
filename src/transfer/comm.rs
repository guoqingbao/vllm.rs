// src/core/transfer/comm.rs
use super::{FinishedPrefillData, PdConfig, PdRole, TransferMessage};
use bincode;
use candle_core::Result;
use interprocess::local_socket::traits::Listener;
use interprocess::local_socket::traits::Stream;
use interprocess::local_socket::{
    GenericNamespaced, ListenerOptions, Stream as LocalStream, ToNsName,
};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

/// An internal enum to abstract over the two stream types.
/// It implements Read and Write to be used generically.
enum CommStream {
    Local(LocalStream),
    Remote(TcpStream),
}

impl Read for CommStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            CommStream::Local(s) => s.read(buf),
            CommStream::Remote(s) => s.read(buf),
        }
    }
}

impl Write for CommStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            CommStream::Local(s) => s.write(buf),
            CommStream::Remote(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            CommStream::Local(s) => s.flush(),
            CommStream::Remote(s) => s.flush(),
        }
    }
}

/// The main Communicator struct.
/// This holds a thread-safe, persistent connection (once established)
/// and can be cloned to share between the main thread (for sending)
/// and the listener thread (for receiving).
#[derive(Clone)]
pub struct Communicator {
    /// The stream is None until a connection is established.
    /// Arc<Mutex<...>> allows sharing between the listener and main threads.
    stream: Arc<Mutex<Option<CommStream>>>,
    config: PdConfig,
    role: PdRole,
    rank: usize,
}

impl Communicator {
    /// Creates a new Communicator, ready to connect.
    pub fn new(config: PdConfig, role: PdRole, rank: usize) -> Self {
        Self {
            stream: Arc::new(Mutex::new(None)),
            config,
            role,
            rank,
        }
    }

    /// Public method to send a message.
    /// This is called from the *main thread* (e.g., Scheduler).
    /// It locks the stream, sends the data, and unlocks.
    pub fn send(&self, msg: &TransferMessage) -> Result<bool> {
        let mut guard = self.stream.lock();
        if let Some(stream) = guard.as_mut() {
            send_message_generic(stream, msg)
        } else {
            candle_core::bail!(
                "[{:?} Rank {}] Communicator not connected, cannot send message.",
                self.role,
                self.rank
            );
        }
    }

    /// Internal method to receive a message.
    /// This is called from the *listener thread* in a loop.
    fn receive(&self) -> Result<TransferMessage> {
        let mut guard = self.stream.lock();
        if let Some(stream) = guard.as_mut() {
            receive_message_generic(stream)
        } else {
            // This should ideally not happen if called from run_listener_loop
            // as the stream is guaranteed to be Some.
            candle_core::bail!(
                "[{:?} Rank {}] Communicator not connected, cannot receive message.",
                self.role,
                self.rank
            );
        }
    }

    /// The main listener loop, intended to be run in a separate thread.
    /// It handles establishing the connection and then transitions
    /// into a loop to receive and dispatch messages.
    /// It also handles connection drops and retries.
    pub fn run_listener_loop(
        &self,
        pending_prefills: Arc<Mutex<VecDeque<crate::core::sequence::Sequence>>>,
        finished_data: Arc<RwLock<HashMap<usize, FinishedPrefillData>>>,
    ) {
        // --- Outer loop: Connection Establishing ---
        loop {
            match self.establish_connection() {
                Ok(_) => {
                    crate::log_info!(
                        "[{:?} Rank {}] Connection established. Starting listener...",
                        self.role,
                        self.rank
                    );
                }
                Err(e) => {
                    crate::log_error!(
                        "[{:?} Rank {}] Failed to establish connection: {}. Retrying in 2s...",
                        self.role,
                        self.rank,
                        e
                    );
                    std::thread::sleep(Duration::from_secs(2));
                    continue; // Retry connection
                }
            }

            // --- Inner loop: Message Receiving ---
            // Now that connection is established, self.stream is Some.
            loop {
                match self.receive() {
                    Ok(msg) => self.handle_received_message(msg, &pending_prefills, &finished_data),
                    Err(e) => {
                        crate::log_error!(
                            "[{:?} Rank {}] Connection error: {}. Re-establishing...",
                            self.role,
                            self.rank,
                            e
                        );
                        *self.stream.lock() = None; // Clear the dead stream
                        break; // Break inner loop to re-establish connection
                    }
                }
            }
        }
    }

    /// Blocks until a connection is established based on role and config.
    /// On success, it populates `self.stream`.
    fn establish_connection(&self) -> Result<()> {
        let new_stream = if let Some(url) = &self.config.url {
            // --- Remote (TCP) Mode ---
            match self.role {
                PdRole::Client => {
                    let stream = TcpStream::connect(url)?;
                    stream.set_nodelay(true)?; // Disable Nagle's algorithm for low latency
                    crate::log_info!(
                        "[PD Client Rank {}] Connected to TCP server at {}",
                        self.rank,
                        url
                    );
                    CommStream::Remote(stream)
                }
                PdRole::Server => {
                    let listener = TcpListener::bind(url)?;
                    crate::log_info!(
                        "[PD Server Rank {}] TCP listener bound. Waiting for client on {}",
                        self.rank,
                        url
                    );
                    let (stream, addr) = listener.accept()?;
                    stream.set_nodelay(true)?;
                    crate::log_info!(
                        "[PD Server Rank {}] Accepted TCP connection from {}",
                        self.rank,
                        addr
                    );
                    CommStream::Remote(stream)
                }
            }
        } else {
            // --- Local (IPC) Mode ---
            let sock_name = format!("@vllm-rs-transfer-{}", self.rank);
            let ns_name = sock_name.clone().to_ns_name::<GenericNamespaced>()?;

            match self.role {
                PdRole::Client => {
                    // Retry connecting to the local socket
                    loop {
                        match LocalStream::connect(ns_name.clone()) {
                            Ok(stream) => {
                                crate::log_info!(
                                    "[PD Client Rank {}] Connected to LocalIPC server at {}",
                                    self.rank,
                                    sock_name
                                );
                                break CommStream::Local(stream);
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                                // Server might not be ready, wait and retry
                                std::thread::sleep(Duration::from_millis(100));
                                continue;
                            }
                            Err(e) => return Err(e.into()), // Other error
                        }
                    }
                }
                PdRole::Server => {
                    let listener = ListenerOptions::new().name(ns_name).create_sync()?;
                    crate::log_info!(
                        "[PD Server Rank {}] LocalIPC listener created. Waiting for client at {}",
                        self.rank,
                        sock_name
                    );
                    let stream = listener.accept()?;
                    crate::log_info!(
                        "[PD Server Rank {}] Accepted LocalIPC connection",
                        self.rank
                    );
                    CommStream::Local(stream)
                }
            }
        };

        // Store the newly established stream
        *self.stream.lock() = Some(new_stream);
        Ok(())
    }

    /// Internal helper to dispatch received messages to the correct queue.
    fn handle_received_message(
        &self,
        msg: TransferMessage,
        pending_prefills: &Arc<Mutex<VecDeque<crate::core::sequence::Sequence>>>,
        finished_data: &Arc<RwLock<HashMap<usize, FinishedPrefillData>>>,
    ) {
        match (&self.role, msg) {
            // Client receives KV cache
            (PdRole::Client, TransferMessage::TransferKvCache(data)) => {
                finished_data.write().insert(data.seq_id, data);
            }
            // Server receives Prefill request
            (PdRole::Server, TransferMessage::TransferPrefill(seq)) => {
                pending_prefills.lock().push_back(seq);
            }
            // Mismatched messages (warn and drop)
            (PdRole::Client, TransferMessage::TransferPrefill(_)) => {
                crate::log_warn!(
                    "[PD Client Rank {}] Received unexpected TransferPrefill msg",
                    self.rank
                );
            }
            (PdRole::Server, TransferMessage::TransferKvCache(_)) => {
                crate::log_warn!(
                    "[PD Server Rank {}] Received unexpected TransferKvCache msg",
                    self.rank
                );
            }
        }
    }
}

/// Generic, standardized function to send a message.
/// Uses a 4-byte LE length prefix followed by bincode data.
fn send_message_generic(stream: &mut (impl Read + Write), msg: &TransferMessage) -> Result<bool> {
    let serialized: Vec<u8> = bincode::serialize(msg).map_err(candle_core::Error::wrap)?;
    let len = serialized.len() as u32;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&serialized)?;
    stream.flush()?;
    Ok(true)
}

/// Generic, standardized function to receive a message.
/// Reads a 4-byte LE length prefix then bincode data.
fn receive_message_generic(stream: &mut (impl Read + Write)) -> Result<TransferMessage> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    if len > 1_000_000_000 {
        // 1GB sanity check
        candle_core::bail!("Message size too large: {} bytes", len);
    }

    let mut msg_buf = vec![0u8; len];
    stream.read_exact(&mut msg_buf)?;

    let msg = bincode::deserialize(&msg_buf).map_err(candle_core::Error::wrap)?;
    Ok(msg)
}
