#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"
DOCKER_BIN="${DOCKER_BIN:-}"
DOCKER_COMPOSE_BIN="${DOCKER_COMPOSE_BIN:-}"
COMPOSE_RUNNER_MODE=""
ENV_FILE="$DEFAULT_ENV_FILE"
NO_CACHE=0
SKIP_FRONTEND=0

log() {
  printf '[MetaMP Images] %s\n' "$*"
}

die() {
  printf '[MetaMP Images][error] %s\n' "$*" >&2
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
  die "Docker/Podman was found, but no Compose runner is available. Install 'docker compose' or 'docker-compose', or set DOCKER_COMPOSE_BIN explicitly."
}

load_env_file() {
  [[ -f "$ENV_FILE" ]] || die "Env file not found: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

compose_args() {
  printf '%s\n' --env-file "$ENV_FILE" -f "$ROOT_DIR/docker-compose.yml"
}

run_compose() {
  resolve_compose_runner
  local args=()
  while IFS= read -r line; do
    args+=("$line")
  done < <(compose_args)
  if [[ "$COMPOSE_RUNNER_MODE" == "plugin" ]]; then
    "$DOCKER_BIN" compose "${args[@]}" "$@"
  else
    "$DOCKER_COMPOSE_BIN" "${args[@]}" "$@"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-stack-images.sh build [options]
  ./scripts/metamp-stack-images.sh push [options]

Commands:
  build     Build the MetaMP backend images (and optionally the frontend image).
  push      Build and push the MetaMP backend images (and optionally the frontend image).

Options:
  --env-file PATH    Use a different env file. Default: .env.docker.deployment
  --no-cache         Build images without cache
  --skip-frontend    Skip frontend image build/push
  -h, --help         Show this help
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

load_env_file

build_backend_images() {
  local build_args=(build)
  if [[ "$NO_CACHE" -eq 1 ]]; then
    build_args+=(--no-cache)
  fi
  build_args+=(flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat)
  log "Building MetaMP backend images in namespace ${REGISTRY_NAMESPACE:-ebenco36}"
  run_compose "${build_args[@]}"
}

push_backend_images() {
  log "Pushing MetaMP backend images"
  run_compose push flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat
}

build_or_push_frontend() {
  if [[ "$SKIP_FRONTEND" -eq 1 ]]; then
    return 0
  fi
  local frontend_command="build"
  local frontend_args=("$frontend_command")
  if [[ "$COMMAND" == "push" ]]; then
    frontend_command="push"
    frontend_args=("$frontend_command")
  fi
  if [[ "$NO_CACHE" -eq 1 ]]; then
    frontend_args+=(--no-cache)
  fi
  bash "$ROOT_DIR/scripts/metamp-frontend-image.sh" "${frontend_args[@]}"
}

case "$COMMAND" in
  build)
    build_backend_images
    build_or_push_frontend
    log "Backend images use ${REGISTRY_NAMESPACE:-ebenco36}/${APP_IMAGE_NAME:-mpvis_app}:${IMAGE_TAG:-latest} and ${REGISTRY_NAMESPACE:-ebenco36}/${ML_IMAGE_NAME:-mpvis_app_ml}:${IMAGE_TAG:-latest}"
    log "Redis and PostgreSQL remain upstream images by default."
    ;;
  push)
    build_backend_images
    push_backend_images
    build_or_push_frontend
    log "Pushed MetaMP custom images to ${REGISTRY_NAMESPACE:-ebenco36}"
    log "Redis and PostgreSQL remain upstream images unless you intentionally mirror them yourself."
    ;;
  *)
    die "Unknown command: $COMMAND"
    ;;
esac
