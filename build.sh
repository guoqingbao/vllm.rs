#!/usr/bin/env bash
set -euo pipefail

# Default build options
RELEASE=""
PROFILE="release"
FEATURES=""
PUBLISH=false

INSTALL=false
DST="/usr/local/bin"

usage() {
  cat <<EOF
Usage: $0 [--debug|--release] [--features "<feat1 feat2 ...>"] [publish] [--install] [--dst <dir>]

Options:
  --debug            Build debug profile
  --release          Build release profile (default)
  --features <...>   Cargo/Maturin feature list (string)
  publish            Publish to PyPI via maturin publish
  --install          Force --release build and copy runner + vllm-rs into --dst (default: /usr/local/bin)
  --dst <dir>        Destination directory for --install (default: /usr/local/bin)
EOF
}

# Helper to compute binary names cross-platform
bin_name() {
  local name="$1"
  if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "win32" || "${OSTYPE:-}" == "cygwin" ]]; then
    echo "${name}.exe"
  else
    echo "$name"
  fi
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --debug)
      RELEASE=""
      PROFILE="debug"
      ;;
    --release)
      RELEASE="--release"
      PROFILE="release"
      ;;
    --features)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --features requires a value"
        usage
        exit 1
      fi
      FEATURES="$2"
      shift
      ;;
    publish)
      PUBLISH=true
      ;;
    --install)
      INSTALL=true
      ;;
    --dst)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --dst requires a value"
        usage
        exit 1
      fi
      DST="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

# If install, force release build (binary installs should be release)
if [[ "$INSTALL" == true ]]; then
  RELEASE="--release"
  PROFILE="release"
fi

RUNNER_BIN="$(bin_name runner)"
VLLM_RS_BIN="$(bin_name vllm-rs)"

# Echo config
echo "Building with profile: $PROFILE"
echo "Cargo release flag: ${RELEASE:-<none>}"
echo "Features: $FEATURES"
echo "Publish: $PUBLISH"
echo "Install: $INSTALL"
echo "Dst: $DST"

# -------------------------------------------------------------------
# INSTALL FAST PATH: skip ALL python/maturin and vllm_rs staging
# -------------------------------------------------------------------
if [[ "$INSTALL" == true ]]; then
  echo "Binary-only install requested; skipping maturin and python package staging."

  # Build vllm-rs
  echo "Building vllm-rs binary..."
  cargo build $RELEASE --bin vllm-rs --features "$FEATURES"

  # Build runner unless metal
  if [[ "$FEATURES" == *"metal"* ]]; then
    echo "Metal feature detected. Skipping runner build."
  else
    FEATURES_RUNNER=$(echo "$FEATURES" | sed -E 's/\bpython\b//g' | xargs)
    echo "Building runner binary..."
    cargo build $RELEASE --bin runner --features "$FEATURES_RUNNER"
  fi

  # Install binaries
  echo "Installing binaries to: $DST"
  mkdir -p "$DST"

  VLLM_RS_PATH="target/$PROFILE/$VLLM_RS_BIN"
  if [[ ! -f "$VLLM_RS_PATH" ]]; then
    echo "Error: vllm-rs binary not found at $VLLM_RS_PATH"
    exit 1
  fi
  install -m 755 "$VLLM_RS_PATH" "$DST/vllm-rs"

  if [[ "$FEATURES" != *"metal"* ]]; then
    RUNNER_PATH="target/$PROFILE/$RUNNER_BIN"
    if [[ ! -f "$RUNNER_PATH" ]]; then
      echo "Error: runner binary not found at $RUNNER_PATH"
      exit 1
    fi
    install -m 755 "$RUNNER_PATH" "$DST/runner"
  fi

  echo "✅ Install complete."
  exit 0
fi

# -------------------------------------------------------------------
# NON-INSTALL PATH: python package staging + maturin build/publish
# -------------------------------------------------------------------
DEST_DIR="vllm_rs"
mkdir -p "$DEST_DIR"

# If metal feature detected: skip runner build/copy
if [[ "$FEATURES" == *"metal"* ]]; then
  echo "Metal feature detected. Skipping runner build and copy."
else
  FEATURES_RUNNER=$(echo "$FEATURES" | sed -E 's/\bpython\b//g' | xargs)
  echo "Building runner binary..."
  cargo build $RELEASE --bin runner --features "$FEATURES_RUNNER"

  echo "Copying runner binary into $DEST_DIR/ ..."
  RUNNER_BINARY="target/$PROFILE/$RUNNER_BIN"
  cp "$RUNNER_BINARY" "$DEST_DIR/runner"
  chmod 755 "$DEST_DIR/runner"
  if command -v patchelf >/dev/null 2>&1; then
    echo "Patching runner rpath for bundled libs..."
    patchelf --set-rpath '$ORIGIN:$ORIGIN/../vllm_rs.libs' "$DEST_DIR/runner"
  else
    echo "Warning: patchelf not found; runner may need LD_LIBRARY_PATH to find bundled libs."
  fi
  echo "✅ Done. Runner binary copied to $DEST_DIR/"
fi

# Staging files for python package
cp "vllm_rs.pyi" "$DEST_DIR/__init__.pyi"
chmod 755 "$DEST_DIR/__init__.pyi"
touch "$DEST_DIR/py.typed"
chmod 755 "$DEST_DIR/py.typed"
cp "python/__init__.py" "$DEST_DIR/__init__.py"
chmod 755 "$DEST_DIR/__init__.py"
cp "ReadMe.md" "$DEST_DIR/ReadMe.md"
chmod 755 "$DEST_DIR/ReadMe.md"
cp "example/server.py" "$DEST_DIR/server.py"
chmod 755 "$DEST_DIR/server.py"
cp "example/chat.py" "$DEST_DIR/chat.py"
chmod 755 "$DEST_DIR/chat.py"
cp "example/completion.py" "$DEST_DIR/completion.py"
chmod 755 "$DEST_DIR/completion.py"

# Build or publish Python package with maturin
echo "Building Python extension with maturin..."

FEATURES=$(echo "$FEATURES" | sed -E 's/\bflash-attn\b//g' | xargs)
FEATURES=$(echo "$FEATURES" | sed -E 's/\bflash-context\b//g' | xargs)
echo "Python extension features: $FEATURES"

if [[ "$PUBLISH" == true ]]; then
  echo "Publishing package to PyPI..."
  maturin publish --features "$FEATURES" --username __token__
else
  maturin build $RELEASE --features "$FEATURES"
fi

# Clean up staging directory
echo "Cleaning up temporary files..."
if [[ "$FEATURES" != *"metal"* ]]; then
  rm -f "$DEST_DIR/runner"
fi
rm -f "$DEST_DIR/__init__.py" \
      "$DEST_DIR/__init__.pyi" \
      "$DEST_DIR/py.typed" \
      "$DEST_DIR/ReadMe.md" \
      "$DEST_DIR/server.py" \
      "$DEST_DIR/chat.py" \
      "$DEST_DIR/completion.py"
rm -rf "$DEST_DIR"

echo "✅ ${PUBLISH:+Publish }Build complete."
