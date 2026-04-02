#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"
DEFAULT_SNAPSHOT_ROOT="$ROOT_DIR/release-snapshots"
DEFAULT_RUNTIME_COMPOSE_FILE="$ROOT_DIR/docker-compose.snapshot.yml"
DOCKER_BIN="${DOCKER_BIN:-}"
DOCKER_COMPOSE_BIN="${DOCKER_COMPOSE_BIN:-}"
COMPOSE_RUNNER_MODE=""
ENV_FILE="$DEFAULT_ENV_FILE"
SNAPSHOT_DIR=""
TOP_MODELS=5
WITH_FRONTEND=0
WITH_LOCAL_FRONTEND=0
WITH_BACKGROUND_JOBS=0
SKIP_BUILD=0

log() {
  printf '[MetaMP Snapshot] %s\n' "$*"
}

die() {
  printf '[MetaMP Snapshot][error] %s\n' "$*" >&2
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

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-snapshot.sh export [options]
  ./scripts/metamp-snapshot.sh load --snapshot-dir PATH [options]

Commands:
  export    Export a reusable MetaMP runtime snapshot from the running stack.
  load      Load a previously exported snapshot into the running stack.

Options:
  --snapshot-dir PATH       Snapshot directory. Default export target: ./release-snapshots/metamp-snapshot-<timestamp>
  --top-models N            Keep only the top N production ML bundles. Default: 5
  --env-file PATH           Use a different Docker env file. Default: .env.docker.deployment
  --with-frontend           Start the frontend service when loading.
  --with-local-frontend     Start the frontend via docker-compose.local-frontend.yml when loading.
  --with-background-jobs    Also start celery workers and celery-beat during snapshot load.
  --skip-build              Skip image refresh during load.
  -h, --help                Show this help.
EOF
}

load_env_file() {
  [[ -f "$ENV_FILE" ]] || die "Env file not found: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  RUNTIME_DATASET_DIR="${INGESTION_DATASET_BASE_DIR:-/var/app/data/datasets}"
}

compose_args() {
  local args=(
    --env-file "$ENV_FILE"
    -f "$ROOT_DIR/docker-compose.yml"
  )
  if [[ "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
    args+=(-f "$ROOT_DIR/docker-compose.local-frontend.yml")
  fi
  printf '%s\n' "${args[@]}"
}

runtime_compose_args() {
  local args=(
    --env-file "$ENV_FILE"
  )
  if [[ "$WITH_BACKGROUND_JOBS" -eq 1 ]]; then
    args+=(-f "$ROOT_DIR/docker-compose.yml")
  else
    args+=(-f "$DEFAULT_RUNTIME_COMPOSE_FILE")
  fi
  if [[ "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
    args+=(-f "$ROOT_DIR/docker-compose.local-frontend.yml")
  fi
  printf '%s\n' "${args[@]}"
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

run_runtime_compose() {
  resolve_compose_runner
  local args=()
  while IFS= read -r line; do
    args+=("$line")
  done < <(runtime_compose_args)
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

start_runtime_for_load() {
  if [[ "$SKIP_BUILD" -ne 1 ]]; then
    log "Pulling runtime images for snapshot load..."
    local pull_services=(postgres redis flask-app)
    if [[ "$WITH_BACKGROUND_JOBS" -eq 1 ]]; then
      pull_services+=(celery-worker celery-worker-ml celery-worker-tm celery-beat)
    fi
    if [[ "$WITH_FRONTEND" -eq 1 && "$WITH_LOCAL_FRONTEND" -eq 0 ]]; then
      pull_services+=(frontend)
    fi
    run_runtime_compose pull "${pull_services[@]}"

    if [[ "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
      log "Building local frontend image from the MPVisualization workspace..."
      bash "$ROOT_DIR/scripts/metamp-frontend-image.sh" build --env-file "$ENV_FILE"
    fi
  fi

  local services=(postgres redis flask-app)
  if [[ "$WITH_BACKGROUND_JOBS" -eq 1 ]]; then
    services+=(celery-worker celery-worker-ml celery-worker-tm celery-beat)
  fi
  if [[ "$WITH_FRONTEND" -eq 1 ]]; then
    services+=(frontend)
  fi

  if [[ "$WITH_BACKGROUND_JOBS" -eq 1 ]]; then
    log "Starting runtime services with background jobs enabled..."
  else
    log "Starting snapshot runtime without background jobs..."
  fi
  run_runtime_compose up -d "${services[@]}"
  wait_for_backend
}

export_snapshot() {
  load_env_file

  local timestamp
  timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  local target_dir="${SNAPSHOT_DIR:-$DEFAULT_SNAPSHOT_ROOT/metamp-snapshot-$timestamp}"
  mkdir -p "$target_dir"

  log "Preparing trimmed runtime snapshot assets inside flask-app..."
  run_compose exec -T flask-app python -m src.Commands.runtime_snapshot export \
    --source-root /var/app \
    --output-dir /tmp/metamp_snapshot_export \
    --top-models "$TOP_MODELS"

  log "Copying dataset and model assets to $target_dir ..."
  mkdir -p "$target_dir"
  run_compose exec -T flask-app sh -lc 'cd /tmp/metamp_snapshot_export && tar -cf - .' | tar -xf - -C "$target_dir"

  log "Exporting PostgreSQL dump..."
  mkdir -p "$target_dir/initdb"
  run_compose exec -T postgres sh -lc 'export PGPASSWORD="${POSTGRES_PASSWORD:-postgres}"; pg_dump -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-mpvis_db}" -Fc' > "$target_dir/initdb/all_tables.dump"

  log "Snapshot exported successfully: $target_dir"
}

load_snapshot() {
  load_env_file
  [[ -n "$SNAPSHOT_DIR" ]] || die "--snapshot-dir is required for load."
  [[ -d "$SNAPSHOT_DIR" ]] || die "Snapshot directory not found: $SNAPSHOT_DIR"

  start_runtime_for_load

  if [[ -d "$SNAPSHOT_DIR/datasets" ]]; then
    log "Loading snapshot datasets into flask-app runtime volume..."
    run_runtime_compose exec -T flask-app sh -lc "mkdir -p '$RUNTIME_DATASET_DIR'"
    run_runtime_compose cp "$SNAPSHOT_DIR/datasets/." "flask-app:$RUNTIME_DATASET_DIR"
    run_runtime_compose exec -T -u root flask-app sh -lc "chmod -R a+rwX '$RUNTIME_DATASET_DIR' || true"
  fi

  if [[ -d "$SNAPSHOT_DIR/data/models" ]]; then
    log "Loading snapshot ML artifacts into flask-app runtime volume..."
    run_runtime_compose exec -T flask-app sh -lc "mkdir -p /var/app/data/models"
    run_runtime_compose cp "$SNAPSHOT_DIR/data/models/." "flask-app:/var/app/data/models"
    run_runtime_compose exec -T -u root flask-app sh -lc "chmod -R a+rwX /var/app/data/models || true"
  fi

  if [[ -f "$SNAPSHOT_DIR/initdb/all_tables.dump" ]]; then
    log "Restoring PostgreSQL database dump..."
    run_runtime_compose cp "$SNAPSHOT_DIR/initdb/all_tables.dump" "postgres:/tmp/metamp_snapshot.dump"
    run_runtime_compose exec -T postgres sh -lc 'export PGPASSWORD="${POSTGRES_PASSWORD:-postgres}"; pg_restore --clean --if-exists --no-owner -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-mpvis_db}" /tmp/metamp_snapshot.dump'
  else
    log "No all_tables.dump found in snapshot; skipping DB restore."
  fi

  log "Snapshot load completed."
  log "Suggested next check: docker compose --env-file $ENV_FILE exec -T flask-app env FLASK_APP=manage.py flask validate-dashboard-regressions"
}

COMMAND="${1:-}"
if [[ "$COMMAND" == "-h" || "$COMMAND" == "--help" ]]; then
  usage
  exit 0
fi
[[ -n "$COMMAND" ]] || {
  usage
  exit 1
}
shift || true

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
    --with-frontend)
      WITH_FRONTEND=1
      shift
      ;;
    --with-local-frontend)
      WITH_LOCAL_FRONTEND=1
      WITH_FRONTEND=1
      shift
      ;;
    --with-background-jobs)
      WITH_BACKGROUND_JOBS=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
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
  export)
    export_snapshot
    ;;
  load)
    load_snapshot
    ;;
  *)
    die "Unknown command: $COMMAND"
    ;;
esac
