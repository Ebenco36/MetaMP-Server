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
FORCE_PUSH=0
SCOPE="all"
PUSH_STATE_FILE="$ROOT_DIR/.metamp-image-push-state"

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
  --scope VALUE      Push scope: all, backend, frontend, ml-backend, lean-backend
  --force-push       Push selected images even when the local tag matches the last pushed image ID
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
    --scope)
      SCOPE="$2"
      shift 2
      ;;
    --force-push)
      FORCE_PUSH=1
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

case "$SCOPE" in
  all|backend|frontend|ml-backend|lean-backend)
    ;;
  *)
    die "Unsupported scope: $SCOPE"
    ;;
esac

backend_services_for_scope() {
  case "$SCOPE" in
    all|backend)
      printf '%s\n' flask-app celery-worker
      ;;
    ml-backend)
      printf '%s\n' flask-app
      ;;
    lean-backend)
      printf '%s\n' celery-worker
      ;;
    frontend)
      ;;
  esac
}

should_build_frontend() {
  [[ "$SKIP_FRONTEND" -ne 1 ]] && [[ "$SCOPE" == "all" || "$SCOPE" == "frontend" ]]
}

ensure_push_state_file() {
  touch "$PUSH_STATE_FILE"
}

get_local_image_id() {
  local image_ref="$1"
  resolve_docker_bin
  "$DOCKER_BIN" image inspect "$image_ref" --format '{{.Id}}' 2>/dev/null || true
}

get_recorded_image_id() {
  local image_ref="$1"
  [[ -f "$PUSH_STATE_FILE" ]] || return 0
  awk -F'|' -v ref="$image_ref" '$1 == ref {print $2}' "$PUSH_STATE_FILE" | tail -n 1
}

record_pushed_image_id() {
  local image_ref="$1"
  local image_id="$2"
  local tmp_file
  ensure_push_state_file
  tmp_file="$(mktemp)"
  awk -F'|' -v ref="$image_ref" '$1 != ref {print $0}' "$PUSH_STATE_FILE" >"$tmp_file" || true
  printf '%s|%s\n' "$image_ref" "$image_id" >>"$tmp_file"
  mv "$tmp_file" "$PUSH_STATE_FILE"
}

push_image_if_needed() {
  local image_ref="$1"
  local image_id recorded_id
  image_id="$(get_local_image_id "$image_ref")"
  [[ -n "$image_id" ]] || die "Local image tag not found: $image_ref"
  recorded_id="$(get_recorded_image_id "$image_ref")"

  if [[ "$FORCE_PUSH" -ne 1 && -n "$recorded_id" && "$recorded_id" == "$image_id" ]]; then
    log "Skipping unchanged image: $image_ref"
    return 0
  fi

  resolve_docker_bin
  log "Pushing $image_ref"
  "$DOCKER_BIN" push "$image_ref"
  record_pushed_image_id "$image_ref" "$image_id"
}

build_backend_images() {
  local build_args=(build)
  local services=()
  while IFS= read -r service; do
    [[ -n "$service" ]] && services+=("$service")
  done < <(backend_services_for_scope)

  if [[ "${#services[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ "$NO_CACHE" -eq 1 ]]; then
    build_args+=(--no-cache)
  fi
  # Only build the distinct image-producing services once.
  # flask-app -> ${ML_IMAGE_NAME}
  # celery-worker -> ${APP_IMAGE_NAME}
  # celery-worker-ml, celery-worker-tm, and celery-beat reuse those tags.
  build_args+=("${services[@]}")
  log "Building MetaMP images in namespace ${REGISTRY_NAMESPACE:-ebenco36} for scope '$SCOPE'"
  run_compose "${build_args[@]}"
}

push_backend_images() {
  case "$SCOPE" in
    all|backend|ml-backend)
      if [[ "$SCOPE" != "lean-backend" ]]; then
        push_image_if_needed "${REGISTRY_NAMESPACE:-ebenco36}/${ML_IMAGE_NAME:-mpvis_app_ml}:${IMAGE_TAG:-latest}"
      fi
      ;;
  esac

  case "$SCOPE" in
    all|backend|lean-backend)
      push_image_if_needed "${REGISTRY_NAMESPACE:-ebenco36}/${APP_IMAGE_NAME:-mpvis_app}:${IMAGE_TAG:-latest}"
      ;;
  esac
}

build_or_push_frontend() {
  if ! should_build_frontend; then
    return 0
  fi
  local frontend_args=(build)
  if [[ "$NO_CACHE" -eq 1 ]]; then
    frontend_args+=(--no-cache)
  fi
  bash "$ROOT_DIR/scripts/metamp-frontend-image.sh" "${frontend_args[@]}"

  if [[ "$COMMAND" == "push" ]]; then
    push_image_if_needed "${REGISTRY_NAMESPACE:-ebenco36}/${FRONTEND_IMAGE_NAME:-mpfrontend}:${FRONTEND_IMAGE_TAG:-latest}"
  fi
}

case "$COMMAND" in
  build)
    build_backend_images
    build_or_push_frontend
    log "Image scope '$SCOPE' built successfully."
    log "Backend images use ${REGISTRY_NAMESPACE:-ebenco36}/${APP_IMAGE_NAME:-mpvis_app}:${IMAGE_TAG:-latest} and ${REGISTRY_NAMESPACE:-ebenco36}/${ML_IMAGE_NAME:-mpvis_app_ml}:${IMAGE_TAG:-latest}"
    log "Redis and PostgreSQL remain upstream images by default."
    ;;
  push)
    build_backend_images
    push_backend_images
    build_or_push_frontend
    log "Pushed MetaMP image scope '$SCOPE' to ${REGISTRY_NAMESPACE:-ebenco36}"
    log "Redis and PostgreSQL remain upstream images unless you intentionally mirror them yourself."
    ;;
  *)
    die "Unknown command: $COMMAND"
    ;;
esac
