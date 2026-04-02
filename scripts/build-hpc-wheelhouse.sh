#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELHOUSE_DIR="${ROOT_DIR}/vendor/wheels"
TMBED_SOURCE_DIR="${ROOT_DIR}/tmbed"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
PYTHON_IMAGE="${PYTHON_IMAGE:-python:3.12.10-slim}"
CLEAN_WHEELHOUSE=0

log() {
  printf '[MetaMP Wheelhouse] %s\n' "$*"
}

die() {
  printf '[MetaMP Wheelhouse][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/build-hpc-wheelhouse.sh [options]

Build a Linux-compatible offline wheelhouse for the SQLite HPC TMbed workflow.
This is intended to run on a machine with Docker and internet access.

Options:
  --wheelhouse-dir PATH    Output directory for wheels. Default: vendor/wheels
  --tmbed-source-dir PATH  Local TMbed source tree to build into a wheel. Default: ./tmbed
  --docker-bin PATH        Container CLI to use. Default: docker
  --docker-platform VALUE  Docker target platform. Default: linux/amd64
  --python-image IMAGE     Builder image. Default: python:3.12.10-slim
  --clean                  Remove the wheelhouse before downloading.
  -h, --help               Show this help.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

write_filtered_ml_requirements() {
  local source_requirements="$ROOT_DIR/requirements-ml.txt"
  local filtered_requirements="$WHEELHOUSE_DIR/requirements-ml.local.txt"
  require_file "$source_requirements"
  grep -viE '^[[:space:]]*tmbed([[:space:]]|@|=|>|<)' "$source_requirements" > "$filtered_requirements"
}

main() {
  require_command "$DOCKER_BIN"
  require_file "$ROOT_DIR/requirements.txt"
  require_file "$ROOT_DIR/requirements-ml.txt"

  mkdir -p "$WHEELHOUSE_DIR"
  WHEELHOUSE_DIR="$(cd "$WHEELHOUSE_DIR" && pwd)"
  mkdir -p "$WHEELHOUSE_DIR"
  if [[ "$CLEAN_WHEELHOUSE" -eq 1 ]]; then
    log "Cleaning wheelhouse at $WHEELHOUSE_DIR"
    find "$WHEELHOUSE_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi

  write_filtered_ml_requirements

  local tmbed_mount_args=()
  local tmbed_container_path=""
  if [[ -d "$TMBED_SOURCE_DIR/tmbed" ]]; then
    TMBED_SOURCE_DIR="$(cd "$TMBED_SOURCE_DIR" && pwd)"
    tmbed_mount_args=(-v "$TMBED_SOURCE_DIR:/tmbedsrc")
    tmbed_container_path="/tmbedsrc"
  fi

  log "Building wheelhouse with ${DOCKER_BIN} (${DOCKER_PLATFORM}, ${PYTHON_IMAGE})"
  "$DOCKER_BIN" run --rm \
    --platform "$DOCKER_PLATFORM" \
    -v "$ROOT_DIR:/work" \
    -v "$WHEELHOUSE_DIR:/wheelhouse" \
    "${tmbed_mount_args[@]}" \
    -w /work \
    -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
    "$PYTHON_IMAGE" \
    /bin/bash -lc "
      set -euo pipefail
      apt-get update
      apt-get install -y --no-install-recommends build-essential gcc g++ gfortran git libffi-dev libpq-dev libssl-dev
      python -m pip install --upgrade pip
      python -m pip wheel \
        --wheel-dir /wheelhouse \
        -r /work/requirements.txt \
        -r /wheelhouse/requirements-ml.local.txt
      if [ -n '${tmbed_container_path}' ] && [ -d '${tmbed_container_path}/tmbed' ]; then
        python -m pip wheel --no-deps \
          --wheel-dir /wheelhouse \
          '${tmbed_container_path}'
      fi
      rm -rf /var/lib/apt/lists/*
    "

  log "Wheelhouse ready at $WHEELHOUSE_DIR"
  log "Copy this directory to the HPC and pass --wheelhouse-dir with the target path."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheelhouse-dir)
      WHEELHOUSE_DIR="$2"
      shift 2
      ;;
    --tmbed-source-dir)
      TMBED_SOURCE_DIR="$2"
      shift 2
      ;;
    --docker-bin)
      DOCKER_BIN="$2"
      shift 2
      ;;
    --docker-platform)
      DOCKER_PLATFORM="$2"
      shift 2
      ;;
    --python-image)
      PYTHON_IMAGE="$2"
      shift 2
      ;;
    --clean)
      CLEAN_WHEELHOUSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

main
