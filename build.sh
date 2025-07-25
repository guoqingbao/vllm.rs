#!/usr/bin/env bash
set -e

# Default build options
RELEASE=""
PROFILE="release"
FEATURES=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --release)
      RELEASE="--release"
      PROFILE="release"
      ;;
    --features)
      FEATURES="$2"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
  shift
done

# Echo config
echo "Building with profile: $PROFILE"
echo "Features: $FEATURES"

# Step 1: Build runner binary
echo "Building runner binary..."
cargo build $RELEASE --bin runner --features "$FEATURES"

# Step 2: Copy runner binary into vllm_rs/runner_bin
#echo "Copying runner binary..."
#BIN_NAME="runner"
#[[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]] && BIN_NAME="runner.exe"

#RUNNER_BINARY="target/$PROFILE/$BIN_NAME"
#DEST_DIR="python/vllm_rs"

#cp "$RUNNER_BINARY" "$DEST_DIR/runner"
# mkdir -p "$DEST_DIR"
# cp "$RUNNER_BINARY" "$DEST_DIR/"
# cp "vllm_rs.pyi" "vllm_rs/__init__.pyi"
# cp "python/vllm_rs/__init__.py" "vllm_rs/"
# touch "vllm_rs/py.typed"

#echo "✅ Done. Runner binary copied to $DEST_DIR/"

# Step 3: Build Python package with maturin
echo "Building Python extension with maturin..."
maturin build $RELEASE --features "$FEATURES"

# Step 4: remove temporary vllm_rs/runner_bin
#echo "Cleaning up temporary files..."
#rm "$DEST_DIR/runner"