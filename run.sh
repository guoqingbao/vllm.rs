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

# Echo config
echo "Building with profile: $PROFILE"
echo "Features: $FEATURES"
echo "Runtime arguments: ${RUN_ARGS[*]}"

# Step 1: Build runner binary
FEATURES_RUNNER=$(echo "$FEATURES" | sed -E 's/\bpython\b//g' | xargs)
echo "Building runner binary..."
cargo build $RELEASE --bin runner --features "$FEATURES_RUNNER"

FEATURES=$(echo "$FEATURES" | sed -E 's/\bflash-attn\b//g' | xargs)
# Step 2: Run the program with runtime args
cargo run $RELEASE --features "$FEATURES" -- "${RUN_ARGS[@]}"
