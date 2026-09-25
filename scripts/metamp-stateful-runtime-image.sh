#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"

DOCKER_BIN="${DOCKER_BIN:-}"
ENV_FILE="$DEFAULT_ENV_FILE"
SNAPSHOT_DIR=""
IMAGE_REF=""
NO_CACHE=0

log() {
  printf '[MetaMP Runtime Image] %s\n' "$*"
}

die() {
  printf '[MetaMP Runtime Image][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-stateful-runtime-image.sh build [options]
  ./scripts/metamp-stateful-runtime-image.sh push [options]

Commands:
  build     Build a runtime-assets image from an exported MetaMP snapshot.
  push      Build and push that runtime-assets image.

Options:
  --env-file PATH      Use a different env file. Default: .env.docker.deployment
  --snapshot-dir PATH  Snapshot directory exported by metamp-snapshot.sh
  --image-ref REF      Full image ref override. Default: <REGISTRY_NAMESPACE>/<SNAPSHOT_ASSETS_IMAGE_NAME>:<IMAGE_TAG>
  --no-cache           Build without Docker cache.
  -h, --help           Show this help.
EOF
}

resolve_docker_bin() {
  if [[ -n "$DOCKER_BIN" && -x "$DOCKER_BIN" ]]; then
    return 0
  fi
  for candidate in docker podman; do
    if command -v "$candidate" >/dev/null 2>&1; then
      DOCKER_BIN="$(command -v "$candidate")"
      return 0
    fi
  done
  for candidate in \
    /Applications/Docker.app/Contents/Resources/bin/docker \
    /usr/local/bin/docker \
    /opt/homebrew/bin/docker
  do
    if [[ -x "$candidate" ]]; then
      DOCKER_BIN="$candidate"
      return 0
    fi
  done
  die "Docker was not found on PATH. Set DOCKER_BIN explicitly if needed."
}

load_env_file() {
  [[ -f "$ENV_FILE" ]] || die "Env file not found: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

resolve_image_ref() {
  if [[ -n "$IMAGE_REF" ]]; then
    printf '%s\n' "$IMAGE_REF"
    return 0
  fi
  printf '%s/%s:%s\n' \
    "${REGISTRY_NAMESPACE:-ebenco36}" \
    "${SNAPSHOT_ASSETS_IMAGE_NAME:-metamp_runtime_snapshot}" \
    "${IMAGE_TAG:-latest}"
}

build_context() {
  [[ -n "$SNAPSHOT_DIR" ]] || die "Provide --snapshot-dir."
  [[ -d "$SNAPSHOT_DIR" ]] || die "Snapshot directory not found: $SNAPSHOT_DIR"

  local tmp_dir
  tmp_dir="$(mktemp -d)"

  cat >"$tmp_dir/Dockerfile" <<'EOF'
FROM alpine:3.20
WORKDIR /runtime-data
COPY datasets ./datasets
COPY models ./models
COPY tm_predictions ./tm_predictions
EOF

  mkdir -p "$tmp_dir/datasets" "$tmp_dir/models" "$tmp_dir/tm_predictions"

  if [[ -d "$SNAPSHOT_DIR/datasets" ]]; then
    cp -R "$SNAPSHOT_DIR/datasets/." "$tmp_dir/datasets/"
  fi
  if [[ -d "$SNAPSHOT_DIR/data/models" ]]; then
    cp -R "$SNAPSHOT_DIR/data/models/." "$tmp_dir/models/"
  fi
  if [[ -d "$SNAPSHOT_DIR/data/tm_predictions" ]]; then
    cp -R "$SNAPSHOT_DIR/data/tm_predictions/." "$tmp_dir/tm_predictions/"
  fi

  printf '%s\n' "$tmp_dir"
}

build_image() {
  resolve_docker_bin
  load_env_file
  local image_ref build_dir
  image_ref="$(resolve_image_ref)"
  build_dir="$(build_context)"
  trap "rm -rf '$build_dir'" EXIT

  local build_args=(build -t "$image_ref")
  if [[ "$NO_CACHE" -eq 1 ]]; then
    build_args+=(--no-cache)
  fi
  build_args+=("$build_dir")

  log "Building runtime snapshot image $image_ref" >&2
  "$DOCKER_BIN" "${build_args[@]}" >&2
  printf '%s\n' "$image_ref"
}

COMMAND="${1:-build}"
if [[ "$COMMAND" == "-h" || "$COMMAND" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --snapshot-dir)
      SNAPSHOT_DIR="$2"
      shift 2
      ;;
    --image-ref)
      IMAGE_REF="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=1
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

case "$COMMAND" in
  build)
    build_image >/dev/null
    ;;
  push)
    resolve_docker_bin
    image_ref="$(build_image)"
    log "Pushing runtime snapshot image $image_ref"
    "$DOCKER_BIN" push "$image_ref"
    ;;
  *)
    die "Unknown command: $COMMAND"
    ;;
esac
