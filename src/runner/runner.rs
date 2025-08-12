use interprocess::local_socket::traits::Stream;
use interprocess::local_socket::Stream as LocalStream;
use interprocess::local_socket::{GenericNamespaced, ToNsName};
use interprocess::TryClone;
use parking_lot::RwLock;
use std::io::Write;
use std::rc::Rc;
use std::sync::Arc;
use vllm_rs::core::runner::{ModelRunner, Seqs};
use vllm_rs::models::layers::distributed::Comm;
use vllm_rs::models::layers::VarBuilderX;
use vllm_rs::runner::{receive_local, send_local, MessageType};
use vllm_rs::utils::new_device;
use vllm_rs::utils::progress::{ProgressLike, ProgressReporter, RemoteProgressReporter};

fn main() -> anyhow::Result<()> {
    vllm_rs::log_info!("runner started");

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args: Vec<String> = std::env::args().collect();
    let sock = args
        .iter()
        .position(|s| s == "--sock")
        .and_then(|i| args.get(i + 1))
        .expect("Socket name missing");
    let sock_name = sock.clone().to_ns_name::<GenericNamespaced>()?;
    let mut stream = LocalStream::connect(sock_name.clone());

    loop {
        if stream.is_ok() {
            break;
        }
        vllm_rs::log_info!("Runner retry connecting to socket: {}", sock);
        stream = LocalStream::connect(sock_name.clone());
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let mut stream = stream.expect("Failed to connect to socket");
    stream.write_all(b"ready\n")?;
    stream.flush()?;

    vllm_rs::log_info!("Runner connected to socket: {}", sock);
    let msg = receive_local(&mut stream)?;
    let runner = match msg {
        MessageType::Init(init_req) => {
            vllm_rs::log_info!("Received init request: {:?}", init_req);
            // Use init_req.rank to pick device
            let device = new_device(init_req.dev_id)?;

            #[cfg(feature = "nccl")]
            let comm = Rc::new(
                Comm::from_rank(
                    device.as_cuda_device().unwrap().cuda_device(),
                    init_req.rank,
                    init_req.num_shards,
                    init_req.nccl_id.0,
                )
                .unwrap(),
            );

            #[cfg(not(feature = "nccl"))]
            let comm = Rc::new(Comm::default());

            vllm_rs::log_info!("Loading model at rank {}", init_req.rank);

            let progress_sock_name = "@vllm-rs-progress".to_string();

            let progress_reporter = match RemoteProgressReporter::new(
                init_req.rank,
                init_req.num_shards,
                progress_sock_name,
                true,
            ) {
                Ok(reporter) => {
                    let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                        Arc::new(RwLock::new(Box::new(reporter)));
                    reporter
                }
                _ => {
                    vllm_rs::log_error!("Unable to create remote progress reporter!");
                    let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                        Arc::new(RwLock::new(Box::new(ProgressReporter::new(init_req.rank))));
                    reporter
                }
            };

            let vb = VarBuilderX::new(
                &init_req.econfig.model_path,
                init_req.dtype.into(),
                &device,
                &progress_reporter,
            )?;
            #[allow(unused_mut)]
            let mut runner = ModelRunner::new(
                init_req.model_type,
                &vb,
                comm,
                &init_req.econfig,
                &init_req.config,
                init_req.dtype.into(),
                init_req.is_rope_i,
                device,
                progress_reporter,
            )?;

            vllm_rs::log_info!("Runner at rank {} created!", init_req.rank);

            // Optional warmup
            #[cfg(all(feature = "cuda", feature = "graph"))]
            match runner.warmup_capture() {
                Ok(_) => eprintln!("Cuda graph captured"),
                Err(e) => eprintln!("Graph capture failed: {:?}", e),
            }

            send_local(&mut vec![stream.try_clone()?], &MessageType::InitAck(true))?;
            runner
        }
        _ => {
            vllm_rs::log_error!("Unexpected message type: {:?}", msg);
            panic!("Unexpected message type");
        }
    };

    loop {
        match receive_local(&mut stream) {
            Ok(MessageType::Shutdown) => {
                vllm_rs::log_info!("Runner exit");
                break;
            }
            Ok(MessageType::RunPrefill((sequences, is_prefill))) => {
                let outputs = runner.run(
                    Seqs::SeqRefs(&sequences.iter().collect::<Vec<_>>()),
                    is_prefill,
                )?;
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RunResponse(outputs),
                )?;
            }
            Ok(MessageType::RunDecode((sequences, is_prefill))) => {
                let outputs = runner.run(Seqs::DecodeVec(&sequences), is_prefill)?;
                send_local(
                    &mut vec![stream.try_clone()?],
                    &MessageType::RunResponse(outputs),
                )?;
            }
            Ok(MessageType::LoadingProgress(_)) => {
                vllm_rs::log_info!("Received loading progress message");
            }
            Err(e) => {
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    vllm_rs::log_error!("Runner exit with error: {:?}", e);
                }
                break;
            }
            _ => {
                vllm_rs::log_error!("Unexpected message type");
            }
        }
    }
    vllm_rs::log_info!("Runner finished");
    Ok(())
}
