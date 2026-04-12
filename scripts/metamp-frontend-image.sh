#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"

ENV_FILE="$DEFAULT_ENV_FILE"
PUSH_IMAGE=0
NO_CACHE=0

log() {
  printf '[MetaMP Frontend] %s\n' "$*"
}

die() {
  printf '[MetaMP Frontend][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-frontend-image.sh build [options]
  ./scripts/metamp-frontend-image.sh push [options]

Commands:
  build     Build the frontend image from the local MPVisualization workspace.
  push      Build and push the frontend image to the configured registry namespace.

Options:
  --env-file PATH   Use a different env file. Default: .env.docker.deployment
  --no-cache        Build the image without cache
  -h, --help        Show this help
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

load_env_file() {
  [[ -f "$ENV_FILE" ]] || die "Env file not found: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

build_frontend_image() {
  local context="${FRONTEND_BUILD_CONTEXT:-../MPVisualization}"
  local dockerfile="$ROOT_DIR/docker/frontend.local.Dockerfile"
  local namespace="${REGISTRY_NAMESPACE:-ebenco36}"
  local image_name="${FRONTEND_IMAGE_NAME:-mpfrontend}"
  local image_tag="${FRONTEND_IMAGE_TAG:-latest}"
  local image_ref="${namespace}/${image_name}:${image_tag}"

  [[ -d "$context" ]] || die "Frontend build context not found: $context"
  [[ -f "$dockerfile" ]] || die "Frontend Dockerfile not found: $dockerfile"

  local build_args=(
    build
    -f "$dockerfile"
    -t "$image_ref"
    --build-arg "VITE_MPV_APP_URL=${FRONTEND_VITE_API_URL:-http://localhost:5400/api/v1/}"
    --build-arg "VITE_APP_MPV_MOCK_URL=${FRONTEND_VITE_MOCK_URL:-http://localhost:5400/api/v1/}"
  )

  if [[ "$NO_CACHE" -eq 1 ]]; then
    build_args+=(--no-cache)
  fi

  build_args+=("$context")

  log "Building frontend image ${image_ref} from ${context}"
  docker "${build_args[@]}"

  if [[ "$PUSH_IMAGE" -eq 1 ]]; then
    log "Pushing frontend image ${image_ref}"
    docker push "$image_ref"
  else
    log "Built frontend image ${image_ref}"
  fi
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
      [[ $# -ge 2 ]] || die "Missing value for $1"
      ENV_FILE="$2"
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

require_command docker
load_env_file

case "$COMMAND" in
  build)
    build_frontend_image
    ;;
  push)
    PUSH_IMAGE=1
    build_frontend_image
    ;;
  *)
    usage
    exit 1
    ;;
esac
