pub mod block_manager;
pub mod engine;
pub mod runner;
pub mod scheduler;
pub mod sequence;

#[derive(Debug)]
pub struct GenerationOutput {
    pub seq_id: usize,
    pub prompt_length: usize,
    pub decode_start_time: usize,
    pub decoded_length: usize,
    pub decode_output: String,
}
