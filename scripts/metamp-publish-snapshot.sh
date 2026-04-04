#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"
DEFAULT_SNAPSHOT_ROOT="$ROOT_DIR/release-snapshots"

ENV_FILE="$DEFAULT_ENV_FILE"
SNAPSHOT_DIR=""
TOP_MODELS=1
SKIP_FRONTEND=0
NO_CACHE=0

log() {
  printf '[MetaMP Snapshot Publish] %s\n' "$*"
}

die() {
  printf '[MetaMP Snapshot Publish][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-publish-snapshot.sh build [options]
  ./scripts/metamp-publish-snapshot.sh push [options]

Commands:
  build     Export a snapshot and build the snapshot-ready images locally.
  push      Export a snapshot and push the snapshot-ready images to the configured registry.

Options:
  --snapshot-dir PATH   Reuse or write a snapshot directory. Default: ./release-snapshots/metamp-snapshot-<timestamp>
  --top-models N        Keep only the top N production ML bundles in the exported snapshot. Default: 1
  --env-file PATH       Use a different Docker env file. Default: .env.docker.deployment
  --skip-frontend       Skip frontend image build/push
  --no-cache            Build images without Docker cache
  -h, --help            Show this help
EOF
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
    --skip-frontend)
      SKIP_FRONTEND=1
      shift
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

if [[ -z "$SNAPSHOT_DIR" ]]; then
  timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  SNAPSHOT_DIR="$DEFAULT_SNAPSHOT_ROOT/metamp-snapshot-$timestamp"
fi

snapshot_args=(export --env-file "$ENV_FILE" --snapshot-dir "$SNAPSHOT_DIR" --top-models "$TOP_MODELS")
stack_args=("$COMMAND" --env-file "$ENV_FILE")
postgres_args=("$COMMAND" --env-file "$ENV_FILE" --snapshot-dir "$SNAPSHOT_DIR")
runtime_args=("$COMMAND" --env-file "$ENV_FILE" --snapshot-dir "$SNAPSHOT_DIR")

if [[ "$SKIP_FRONTEND" -eq 1 ]]; then
  stack_args+=(--skip-frontend)
fi

if [[ "$NO_CACHE" -eq 1 ]]; then
  stack_args+=(--no-cache)
  postgres_args+=(--no-cache)
  runtime_args+=(--no-cache)
fi

log "Exporting runtime snapshot to $SNAPSHOT_DIR"
bash "$ROOT_DIR/scripts/metamp-snapshot.sh" "${snapshot_args[@]}"

log "Building or pushing application images"
bash "$ROOT_DIR/scripts/metamp-stack-images.sh" "${stack_args[@]}"

log "Building or pushing stateful PostgreSQL snapshot image"
bash "$ROOT_DIR/scripts/metamp-stateful-postgres-image.sh" "${postgres_args[@]}"

log "Building or pushing runtime snapshot assets image"
bash "$ROOT_DIR/scripts/metamp-stateful-runtime-image.sh" "${runtime_args[@]}"

log "Snapshot publish flow complete."
log "Use $ROOT_DIR/docker-compose.snapshot.yml with the pushed images for a no-background-jobs runtime."
