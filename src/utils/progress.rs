use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::{thread, time};
pub trait ProgressLike: Send + Sync {
    fn get_progress(&self) -> (usize, usize);
    fn set_progress(&mut self, p: usize);
}

pub struct ProgressReporter {
    pub rank: usize,
    pub progress: usize,
}

impl ProgressLike for ProgressReporter {
    fn get_progress(&self) -> (usize, usize) {
        (self.rank, self.progress)
    }

    fn set_progress(&mut self, p: usize) {
        self.progress = p;
    }
}

impl ProgressReporter {
    pub fn new(rank: usize) -> Self {
        Self { rank, progress: 0 }
    }
}

unsafe impl Send for ProgressReporter {}
unsafe impl Sync for ProgressReporter {}

pub struct Progress {
    m: MultiProgress,
    bars: Vec<ProgressBar>,
    size: usize,
}

impl Progress {
    pub fn new(n: usize, size: usize) -> Progress {
        let m = MultiProgress::new();
        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:60.cyan/blue} {pos:>4}/{len:4} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        let mut bars = Vec::<ProgressBar>::new();
        for i in 0..n {
            let pb = m.add(ProgressBar::new(size as u64));
            pb.set_style(sty.clone());
            if n > 1 {
                pb.set_message(format!("On Rank {} Device", i));
            }
            bars.push(pb);
        }

        // if n > 1 {
        //     m.println(format!("Loading model in {} ranks!", n)).unwrap();
        // }
        Self { m, bars, size }
    }

    pub fn update(&self, idx: usize, progress: usize) {
        if idx < self.bars.len() && progress > 0 {
            let pos = self.bars[idx].position();
            self.bars[idx].inc(progress as u64 - pos);
            if self.bars.len() > 1 {
                if progress >= self.size {
                    self.bars[idx].set_message(format!("On Rank {} Device Finished", idx));
                } else {
                    self.bars[idx].set_message(format!("On Rank {} Device", idx));
                }
            }
        }
    }

    pub fn finish(&self) {
        for idx in 0..self.bars.len() {
            let pos = self.bars[idx].position();
            self.bars[idx].inc(self.size as u64 - pos);
            if self.bars.len() > 1 {
                self.bars[idx].set_message(format!("On Rank {} Device Finished", idx));
            }
        }
        self.m.clear().unwrap();
    }
}

#[allow(unused_variables)]
pub fn progress_worker(
    num_shards: usize,
    length: usize,
    progress_reporter: &Vec<Arc<RwLock<ProgressReporter>>>,
) -> std::thread::JoinHandle<()> {
    let mut finished_map = HashMap::<usize, usize>::new();
    let reporters = progress_reporter.clone();
    let progress_bar = Some(Progress::new(num_shards, length));
    let handle = thread::spawn(move || loop {
        {
            let _ = thread::sleep(time::Duration::from_millis(500 as u64));
            for i in 0..num_shards {
                let (rank, progress) = reporters[i].read().get_progress();
                finished_map.insert(rank, progress);
                progress_bar.as_ref().unwrap().update(rank, progress);
            }

            if finished_map.values().all(|v| v >= &length) {
                progress_bar.as_ref().unwrap().finish();
                break;
            }
        }
    });

    handle
}
