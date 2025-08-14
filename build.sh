#!/usr/bin/env bash
set -e

# Default build options
RELEASE=""
PROFILE="release"
FEATURES=""
PUBLISH=false

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
    publish)
      PUBLISH=true
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
echo "Publish: $PUBLISH"

# Step 1: Build runner binary
FEATURES_RUNNER=$(echo "$FEATURES" | sed -E 's/\bpython\b//g' | xargs)
echo "Building runner binary..."
cargo build $RELEASE --bin runner --features "$FEATURES_RUNNER"

# Step 2: Copy runner binary into vllm_rs/
echo "Copying runner binary..."
BIN_NAME="runner"
[[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]] && BIN_NAME="runner.exe"

RUNNER_BINARY="target/$PROFILE/$BIN_NAME"
DEST_DIR="vllm_rs"

mkdir -p "$DEST_DIR"
cp "$RUNNER_BINARY" "$DEST_DIR"
chmod 755 "$DEST_DIR/runner"
cp "vllm_rs.pyi" "$DEST_DIR/__init__.pyi"
chmod 755 "$DEST_DIR/__init__.pyi"
touch "$DEST_DIR/py.typed"
chmod 755 "$DEST_DIR/py.typed"
cp "python/__init__.py" "$DEST_DIR/__init__.py"
chmod 755 "$DEST_DIR/__init__.py"
cp "ReadMe.md" "$DEST_DIR/ReadMe.md"
chmod 755 "$DEST_DIR/ReadMe.md"

echo "✅ Done. Runner binary copied to $DEST_DIR/"

# Step 3: Build or publish Python package with maturin
echo "Building Python extension with maturin..."

# Remove 'flash-attn' if present
FEATURES=$(echo "$FEATURES" | sed -E 's/\bflash-attn\b//g' | xargs)
echo "Python extension features: $FEATURES"

if [ "$PUBLISH" = true ]; then
  echo "Publishing package to PyPI..."
  maturin publish --features "$FEATURES" --username __token__
else
  maturin build $RELEASE --features "$FEATURES"
fi

# Step 4: Clean up
echo "Cleaning up temporary files..."
rm "$DEST_DIR/runner"
rm "$DEST_DIR/__init__.py"
rm "$DEST_DIR/__init__.pyi"
rm "$DEST_DIR/py.typed"
rm "$DEST_DIR/ReadMe.md"
rm -r "$DEST_DIR"

echo "✅ ${PUBLISH:+Publish}Build complete."
