//src/core/engine.rs
use super::runner::ModelRunner;
use super::scheduler::Scheduler;
use super::sequence::Sequence;
use crate::utils::config::{Config, EngineConfig, SamplingParams};
use crate::utils::{hub_load_local_safetensors, new_device};
use candle_core::{DType, Device, Result};
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub struct LLMEngine {
    model_runner: ModelRunner,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    device: Device,
    econfig: EngineConfig,
}

impl LLMEngine {
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Self> {
        let device = new_device(1)?;
        let weight_files =
            hub_load_local_safetensors(&econfig.model_path, "model.safetensors.index.json")?;
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(&weight_files, dtype, &device)
                .unwrap()
        };
        let config_path = format!("{}/config.json", econfig.model_path);
        let mut config: Config =
            serde_json::from_slice(&std::fs::read(config_path).map_err(candle_core::Error::wrap)?)
                .map_err(candle_core::Error::wrap)?;
        config.quant = econfig.quant.clone();

        let mut econfig = econfig.clone();
        econfig.max_model_len = std::cmp::min(
            econfig.max_model_len,
            config
                .max_model_len
                .unwrap_or(config.max_position_embeddings),
        );

        let tokenizer_file = if econfig.tokenizer.is_some() {
            econfig.tokenizer.clone().unwrap()
        } else {
            econfig.model_path.clone() + "/tokenizer.json"
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(candle_core::Error::wrap)?;

        let model_runner = ModelRunner::new(vb, &econfig, &config, dtype, device.clone())?;
        let scheduler = Scheduler::new(&econfig);

        Ok(Self {
            model_runner,
            scheduler,
            tokenizer,
            device,
            econfig,
        })
    }

    pub fn add_request(&mut self, prompt: &str, params: SamplingParams) -> Result<()> {
        let tokens = self.tokenizer.encode(prompt, true).expect("encode failed!");
        let token_ids: Vec<u32> = tokens.get_ids().iter().map(|&x| x as u32).collect();
        let seq = Sequence::new(token_ids, self.econfig.block_size, params);
        self.scheduler.add(seq);
        Ok(())
    }

    pub fn step(&mut self) -> Result<Vec<(usize, Vec<u32>)>> {
        // Get scheduled sequence indexes and prefill flag
        let (scheduled_ids, is_prefill) = self.scheduler.schedule();
        if scheduled_ids.is_empty() {
            return Ok(vec![]);
        }

        // Get immutable references to scheduled sequences for model_runner
        let seqs = self.scheduler.get_sequences(&scheduled_ids);

        // Run model on the scheduled sequences
        let output_ids = self.model_runner.run(&seqs, is_prefill)?;

        // Postprocess sequences by modifying them inside the scheduler
        self.scheduler.postprocess(&scheduled_ids, &output_ids);

        // Collect outputs of finished sequences
        let outputs: Vec<_> = scheduled_ids
            .iter()
            .filter_map(|&idx| match self.scheduler.get_running(idx) {
                Some(s) => {
                    if s.is_finished() {
                        println!("seq {} finished", s.id);
                        Some((s.id, s.output_ids.clone()))
                    } else {
                        // let output = self
                        //     .tokenizer
                        //     .decode(&[s.last_token], true)
                        //     .expect("unable to decode!");
                        // print!("{}", output);
                        // use std::io::{self, Write};
                        // std::io::stdout().flush();
                        None
                    }
                }
                _ => None,
            })
            .collect();
        self.scheduler.clear_finished();
        Ok(outputs)
    }

    pub fn generate(&mut self, prompts: &[&str], params: &SamplingParams) -> Result<Vec<String>> {
        for prompt in prompts {
            let prompt = format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            );
            self.add_request(&prompt.as_str(), params.clone())?;
        }

        let mut outputs = HashMap::new();
        while !self.scheduler.is_finished() {
            let step_output = self.step()?;
            for (seq_id, token_ids) in step_output {
                outputs.insert(seq_id, token_ids);
            }
        }

        let mut sorted_outputs: Vec<_> = outputs.into_iter().collect();
        sorted_outputs.sort_by_key(|(seq_id, _)| *seq_id);

        let mut results = vec![];
        for (_, token_ids) in sorted_outputs {
            let output = self
                .tokenizer
                .decode(&token_ids, true)
                .expect("unable to decode!");
            results.push(output);
        }

        Ok(results)
    }
}
