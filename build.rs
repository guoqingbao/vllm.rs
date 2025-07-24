use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only run this for maturin builds
    if env::var("CARGO_FEATURE_PYTHON").is_ok() {
        let profile = env::var("PROFILE").expect("PROFILE env var not set by Cargo");

        // Collect all enabled features from environment variables
        let features: Vec<String> = env::vars()
            .filter_map(|(key, _)| {
                if let Some(stripped) = key.strip_prefix("CARGO_FEATURE_") {
                    Some(stripped.replace('_', "-").to_lowercase())
                } else {
                    None
                }
            })
            .collect();

        println!("Running features: {:?}", features);
        let mut build_cmd = Command::new("cargo");

        build_cmd.arg("build")
            .arg("--bin").arg("runner");

        if profile == "release" {
            build_cmd.arg("--release");
        }

        if !features.is_empty() {
            build_cmd.arg("--features").arg(features.join(","));
        }

        // Inject env var to detect recursion
        build_cmd.env("RUNNING_RUNNER_BUILD", "1");

        // Run cargo build with the same profile and features
        let status = build_cmd
            .status()
            .expect("Failed to execute cargo build for runner");

        if !status.success() {
            panic!("Failed to build runner binary");
        }

        // OUT_DIR = target/{profile}/build/<hash>/out
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_dir = out_dir.ancestors().nth(4).unwrap(); // gets target/
        let binary_dir = target_dir.join(&profile);

        let binary_name = if cfg!(windows) {
            "runner.exe"
        } else {
            "runner"
        };

        let runner_binary = binary_dir.join(&binary_name);
        if !runner_binary.exists() {
            panic!("Runner binary not found at {:?}", runner_binary);
        }

        let dest_path = PathBuf::from("runner_bin").join(&binary_name);
        fs::create_dir_all("runner_bin").unwrap();
        fs::copy(&runner_binary, &dest_path)
            .expect("Failed to copy runner binary into package directory");
    }
}
