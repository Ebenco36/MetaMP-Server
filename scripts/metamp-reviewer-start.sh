#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"
COMPOSE_FILE="$ROOT_DIR/docker-compose.snapshot.yml"
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-metamp-reviewer}"

ENV_FILE="$DEFAULT_ENV_FILE"
DOCKER_BIN="${DOCKER_BIN:-}"
DOCKER_COMPOSE_BIN="${DOCKER_COMPOSE_BIN:-}"
COMPOSE_RUNNER_MODE=""
KEEP_DATA=0

log() {
  printf '[MetaMP Reviewer] %s\n' "$*"
}

die() {
  printf '[MetaMP Reviewer][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-reviewer-start.sh [options]

Options:
  --env-file PATH   Use a different Docker env file. Default: .env.docker.deployment
  --keep-data       Reuse the existing reviewer-stack volumes instead of resetting them
  -h, --help        Show this help
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

load_env_file() {
  [[ -f "$ENV_FILE" ]] || die "Env file not found: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  export POSTGRES_IMAGE="${POSTGRES_IMAGE_OVERRIDE:-${REGISTRY_NAMESPACE:-ebenco36}/${POSTGRES_STATE_IMAGE_NAME:-metamp_postgres_state}:${IMAGE_TAG:-latest}}"
  export SNAPSHOT_ASSETS_IMAGE="${SNAPSHOT_ASSETS_IMAGE_OVERRIDE:-${REGISTRY_NAMESPACE:-ebenco36}/${SNAPSHOT_ASSETS_IMAGE_NAME:-metamp_runtime_snapshot}:${IMAGE_TAG:-latest}}"
}

run_compose() {
  resolve_compose_runner
  local args=(
    --project-name "$PROJECT_NAME"
    --env-file "$ENV_FILE"
    -f "$COMPOSE_FILE"
  )
  if [[ "$COMPOSE_RUNNER_MODE" == "plugin" ]]; then
    "$DOCKER_BIN" compose "${args[@]}" "$@"
  else
    "$DOCKER_COMPOSE_BIN" "${args[@]}" "$@"
  fi
}

wait_for_backend() {
  local port="${FLASK_RUN_PORT:-5400}"
  local url="http://127.0.0.1:${port}/api/v1/health/ready"
  local attempts=90
  for _ in $(seq 1 "$attempts"); do
    if curl --silent --fail "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  die "Backend did not become ready in time."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --keep-data)
      KEEP_DATA=1
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

load_env_file

log "Using snapshot PostgreSQL image: $POSTGRES_IMAGE"
log "Using snapshot runtime image: $SNAPSHOT_ASSETS_IMAGE"
log "Pulling latest reviewer images..."
run_compose pull postgres redis snapshot-assets flask-app frontend

if [[ "$KEEP_DATA" -ne 1 ]]; then
  log "Resetting reviewer stack volumes so the exact published snapshot is restored..."
  run_compose down -v --remove-orphans || true
fi

log "Starting MetaMP reviewer stack..."
run_compose up -d
wait_for_backend

log "MetaMP is ready."
log "Frontend: http://localhost/"
log "Backend:  http://localhost:${FLASK_RUN_PORT:-5400}/api/v1/health/ready"
