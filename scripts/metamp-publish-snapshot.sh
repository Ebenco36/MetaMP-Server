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
DOCKER_BIN="${DOCKER_BIN:-}"
DOCKER_COMPOSE_BIN="${DOCKER_COMPOSE_BIN:-}"
COMPOSE_RUNNER_MODE=""

log() {
  printf '[MetaMP Snapshot Publish] %s\n' "$*"
}

die() {
  printf '[MetaMP Snapshot Publish][error] %s\n' "$*" >&2
  exit 1
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

resolve_compose_runner() {
  if [[ -n "$DOCKER_COMPOSE_BIN" && -x "$DOCKER_COMPOSE_BIN" ]]; then
    COMPOSE_RUNNER_MODE="standalone"
    return 0
  fi
  resolve_docker_bin
  if "$DOCKER_BIN" compose version >/dev/null 2>&1; then
    COMPOSE_RUNNER_MODE="plugin"
    return 0
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_BIN="$(command -v docker-compose)"
    COMPOSE_RUNNER_MODE="standalone"
    return 0
  fi
  die "Docker/Podman was found, but no Compose runner is available."
}

run_reviewer_compose() {
  resolve_compose_runner
  local args=(
    --project-name "${COMPOSE_PROJECT_NAME:-metamp-reviewer}"
    --env-file "$ENV_FILE"
    -f "$ROOT_DIR/docker-compose.snapshot.yml"
  )
  if [[ "$COMPOSE_RUNNER_MODE" == "plugin" ]]; then
    "$DOCKER_BIN" compose "${args[@]}" "$@"
  else
    "$DOCKER_COMPOSE_BIN" "${args[@]}" "$@"
  fi
}

ensure_reviewer_stack_is_not_running() {
  local running_services
  running_services="$(run_reviewer_compose ps --status running --services 2>/dev/null || true)"
  if [[ -n "$running_services" ]]; then
    die "The reviewer stack is currently running and occupies the published container names/ports. Stop it first with: docker compose --project-name ${COMPOSE_PROJECT_NAME:-metamp-reviewer} --env-file $ENV_FILE -f $ROOT_DIR/docker-compose.snapshot.yml down"
  fi
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

ensure_reviewer_stack_is_not_running

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
