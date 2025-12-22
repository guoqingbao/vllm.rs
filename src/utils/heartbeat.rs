use super::command::CommandManager;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::{process, thread, time};

pub fn heartbeat_worker(
    num_subprocess: Option<usize>,
    is_daemon: bool,
    stop_flag: Arc<AtomicBool>,
    uuid: &str,
) -> std::thread::JoinHandle<()> {
    let uuid_str = uuid.to_string();
    let handle = thread::spawn(move || {
        let flag_clone = Arc::clone(&stop_flag);
        let sock_name = format!("{}@vllm-rs-runner-heartbeat", uuid_str);
        let mut connect_retry_count = 0;
        let mut command_manager = if is_daemon {
            let mut manager = CommandManager::new_command(&sock_name, None, is_daemon);
            while !flag_clone.load(Ordering::Relaxed) {
                if manager.is_ok() {
                    break;
                } else if connect_retry_count < 120 {
                    connect_retry_count += 1;
                    crate::log_info!(
                        "Retry connect to main process' command channel ({:?})!",
                        manager
                    );
                    let _ = thread::sleep(time::Duration::from_millis(1000 as u64));
                    manager = CommandManager::new_command(&sock_name, None, is_daemon);
                    continue;
                } else {
                    crate::log_warn!("{:?}", manager);
                    break;
                }
            }
            manager
        } else {
            CommandManager::new_command(&sock_name, num_subprocess, is_daemon)
        };

        let mut heartbeat_error_count = 0;
        if let Ok(manager) = command_manager.as_mut() {
            crate::log_info!("enter heartbeat processing loop ({:?})", manager);
            while !flag_clone.load(Ordering::Relaxed) {
                let alive_result = manager.heartbeat(is_daemon);
                if alive_result.is_err() {
                    if !flag_clone.load(Ordering::Relaxed) {
                        crate::log_warn!("{:?}", alive_result);
                    }
                    if heartbeat_error_count > 5 {
                        crate::log_error!(
                            "heartbeat detection failed, exit the current process because of {:?}",
                            alive_result
                        );
                        process::abort();
                    }
                    heartbeat_error_count += 1;
                }

                let _ = thread::sleep(time::Duration::from_millis(1000 as u64));
            }
        } else {
            crate::log_error!(
                "Failed to initialize command manager: {:?}",
                command_manager.err()
            );
        }
    });
    handle
}
