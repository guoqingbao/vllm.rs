#!/usr/bin/env bash
set -e

# Default build options
RELEASE=""
PROFILE="release"
FEATURES=""

# Arrays to hold build and run args separately
RUN_ARGS=()

# Parse build arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --release)
      RELEASE="--release"
      PROFILE="release"
      shift
      ;;
    --features)
      FEATURES="$2"
      shift 2
      ;;
    --) # Separator: remaining args go to runtime
      shift
      RUN_ARGS+=("$@")
      break
      ;;
    *) # Anything unknown is forwarded as a runtime arg
      RUN_ARGS+=("$1")
      shift
      ;;
  esac
done

IS_METAL=false
if [[ "$FEATURES" == *"metal"* ]]; then
  IS_METAL=true
fi

# Echo config
echo "Building with profile: $PROFILE"
echo "Features: $FEATURES"
echo "Runtime arguments: ${RUN_ARGS[*]}"

# Build both binaries in one cargo command
FEATURES_RUNNER=$(echo "$FEATURES" | sed -E 's/\bpython\b//g' | xargs)
if [[ "$IS_METAL" == true ]]; then
  echo "Building vllm-rs binary..."
  cargo build $RELEASE --bin vllm-rs --features "$FEATURES_RUNNER"
else
  echo "Building vllm-rs and runner binaries..."
  cargo build $RELEASE --bin vllm-rs --bin runner --features "$FEATURES_RUNNER"
fi

# Run the program with runtime args
cargo run $RELEASE --bin vllm-rs --features "$FEATURES" -- "${RUN_ARGS[@]}"
