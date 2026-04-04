#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"

ENV_FILE="$DEFAULT_ENV_FILE"
SNAPSHOT_DIR=""
TOP_MODELS=1
NO_CACHE=0
SKIP_FRONTEND=0

log() {
  printf '[MetaMP Release] %s\n' "$*"
}

die() {
  printf '[MetaMP Release][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-release.sh publish [options]

Commands:
  publish   Push the current MetaMP images and export a matching runtime snapshot.

Options:
  --snapshot-dir PATH    Optional export target for the runtime snapshot.
  --top-models N         Keep the top N production ML bundles in the snapshot. Default: 1
  --env-file PATH        Use a different env file. Default: .env.docker.deployment
  --no-cache             Build images without cache before pushing.
  --skip-frontend        Skip frontend image build/push.
  -h, --help             Show this help.
EOF
}

COMMAND="${1:-publish}"
if [[ "$COMMAND" == "-h" || "$COMMAND" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot-dir)
      SNAPSHOT_DIR="$2"
      shift 2
      ;;
    --top-models)
      TOP_MODELS="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --skip-frontend)
      SKIP_FRONTEND=1
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
  publish)
    log "Pushing MetaMP application images..."
    image_args=(push --env-file "$ENV_FILE")
    if [[ "$NO_CACHE" -eq 1 ]]; then
      image_args+=(--no-cache)
    fi
    if [[ "$SKIP_FRONTEND" -eq 1 ]]; then
      image_args+=(--skip-frontend)
    fi
    "$ROOT_DIR/scripts/metamp-stack-images.sh" "${image_args[@]}"

    log "Exporting matching runtime snapshot..."
    snapshot_args=(export --env-file "$ENV_FILE" --top-models "$TOP_MODELS")
    if [[ -n "$SNAPSHOT_DIR" ]]; then
      snapshot_args+=(--snapshot-dir "$SNAPSHOT_DIR")
    fi
    "$ROOT_DIR/scripts/metamp-snapshot.sh" "${snapshot_args[@]}"

    log "Release publish flow completed."
    ;;
  *)
    die "Unknown command: $COMMAND"
    ;;
esac
