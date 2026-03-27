#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"
STATE_FILE="$ROOT_DIR/.metamp-launcher.state"

ENV_FILE="$DEFAULT_ENV_FILE"
WITH_FRONTEND=0
WITH_LOCAL_FRONTEND=0
SKIP_BUILD=0
NO_CACHE=0
SKIP_SYNC=0
SKIP_ML=0
SKIP_TM=0

log() {
  printf '[MetaMP] %s\n' "$*"
}

warn() {
  printf '[MetaMP][warn] %s\n' "$*" >&2
}

die() {
  printf '[MetaMP][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-pipeline.sh up [options]
  ./scripts/metamp-pipeline.sh status [options]
  ./scripts/metamp-pipeline.sh logs [service] [options]
  ./scripts/metamp-pipeline.sh down [options]

Commands:
  up        Build, start, initialize, and queue the production pipeline.
  status    Show container state and latest queued task status.
  logs      Tail logs for a service. Defaults to flask-app.
  down      Stop the stack.

Options:
  --env-file PATH            Use a different env file. Default: .env.docker.deployment
  --with-frontend            Start the frontend service too.
  --with-local-frontend      Start the frontend with docker-compose.local-frontend.yml.
  --skip-build               Skip docker compose build.
  --no-cache                 Build images without cache.
  --skip-sync                Skip the initial protein database sync.
  --skip-ml                  Skip queueing the ML training task.
  --skip-tm                  Skip queueing the normalized TM annotation sync task.
  -h, --help                 Show this help.

Examples:
  ./scripts/metamp-pipeline.sh up
  ./scripts/metamp-pipeline.sh up --with-frontend
  ./scripts/metamp-pipeline.sh status
  ./scripts/metamp-pipeline.sh logs celery-worker-ml
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

run_compose() {
  local args=()
  while IFS= read -r line; do
    args+=("$line")
  done < <(compose_args)
  docker compose "${args[@]}" "$@"
}

wait_for_backend() {
  local port="${FLASK_RUN_PORT:-5400}"
  local url="http://127.0.0.1:${port}/api/v1/health/ready"
  local attempts=90

  log "Waiting for backend readiness at ${url}"
  for _ in $(seq 1 "$attempts"); do
    if curl --silent --fail "$url" >/dev/null 2>&1; then
      log "Backend is ready."
      return 0
    fi
    sleep 2
  done

  die "Backend did not become ready in time."
}

queue_ml_job() {
  local output
  output="$(run_compose exec -T flask-app env FLASK_APP=manage.py flask queue-machine-learning-job)"
  printf '%s\n' "$output" >&2
  printf '%s\n' "$output" | awk -F': ' 'END {print $NF}'
}

queue_tm_annotation_sync() {
  local args=(exec -T flask-app env FLASK_APP=manage.py flask queue-tmalphafold-sync --with-tmdet --with-tmbed --tmbed-refresh)
  if [[ "${TM_PREDICTION_USE_GPU:-false}" == "true" || "${TMALPHAFOLD_TMBED_USE_GPU:-false}" == "true" ]]; then
    args+=(--tmbed-use-gpu)
  fi
  if [[ -n "${TM_PREDICTION_BATCH_SIZE:-}" ]]; then
    args+=(--tmbed-batch-size "${TM_PREDICTION_BATCH_SIZE}")
  fi
  if [[ -n "${TM_PREDICTION_MAX_WORKERS:-}" ]]; then
    args+=(--tmbed-max-workers "${TM_PREDICTION_MAX_WORKERS}")
  fi
  local output
  output="$(run_compose "${args[@]}")"
  printf '%s\n' "$output" >&2
  printf '%s\n' "$output" | awk -F': ' 'END {print $NF}'
}

write_state() {
  local ml_task_id="${1:-}"
  local tm_annotation_task_id="${2:-}"
  cat >"$STATE_FILE" <<EOF
ENV_FILE=$ENV_FILE
WITH_FRONTEND=$WITH_FRONTEND
WITH_LOCAL_FRONTEND=$WITH_LOCAL_FRONTEND
ML_TASK_ID=$ml_task_id
TM_ANNOTATION_TASK_ID=$tm_annotation_task_id
LAST_BOOTSTRAP_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
}

show_stateful_task_status() {
  local label="$1"
  local task_id="$2"
  if [[ -z "$task_id" ]]; then
    return 0
  fi

  log "$label task status ($task_id):"
  run_compose exec -T flask-app env FLASK_APP=manage.py flask celery-task-status --task-id "$task_id" || true
}

start_stack() {
  load_env_file
  require_command docker
  require_command curl

  local services=(postgres redis flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat)
  local build_services=(flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat)

  if [[ "$WITH_FRONTEND" -eq 1 || "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
    services+=(frontend)
  fi
  if [[ "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
    build_services+=(frontend)
  fi

  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    log "Building service images..."
    local build_args=(build)
    if [[ "$NO_CACHE" -eq 1 ]]; then
      build_args+=(--no-cache)
    fi
    build_args+=("${build_services[@]}")
    run_compose "${build_args[@]}"
  else
    log "Skipping image build."
  fi

  log "Starting service stack..."
  run_compose up -d "${services[@]}"

  wait_for_backend

  if [[ "$SKIP_SYNC" -eq 0 ]]; then
    log "Running initial protein database sync..."
    run_compose exec -T flask-app env FLASK_APP=manage.py flask sync-protein-database
  else
    warn "Skipping initial protein database sync. Queued jobs will use the current database state."
  fi

  local tm_annotation_task_id=""
  local ml_task_id=""

  if [[ "$SKIP_TM" -eq 0 ]]; then
    log "Queueing normalized TM annotation sync task..."
    tm_annotation_task_id="$(queue_tm_annotation_sync)"
  else
    log "Skipping normalized TM annotation sync."
  fi

  if [[ "$SKIP_ML" -eq 0 ]]; then
    log "Queueing machine learning training task..."
    ml_task_id="$(queue_ml_job)"
  else
    log "Skipping machine learning task."
  fi

  write_state "$ml_task_id" "$tm_annotation_task_id"

  log "Bootstrap complete."
  log "API: http://127.0.0.1:${FLASK_RUN_PORT:-5400}/api/v1/health/ready"
  if [[ "$WITH_FRONTEND" -eq 1 || "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
    log "Frontend: http://127.0.0.1/"
  fi
  [[ -n "$ml_task_id" ]] && log "Machine learning task id: $ml_task_id"
  [[ -n "$tm_annotation_task_id" ]] && log "TM annotation sync task id: $tm_annotation_task_id"
  log "Use './scripts/metamp-pipeline.sh status' to inspect progress."
}

show_status() {
  load_env_file
  log "Container status:"
  run_compose ps || true

  log "Backend readiness:"
  curl --silent "http://127.0.0.1:${FLASK_RUN_PORT:-5400}/api/v1/health/ready" || true
  printf '\n'

  log "Latest protein refresh status:"
  run_compose exec -T flask-app env FLASK_APP=manage.py flask protein-refresh-status || true

  log "Latest protein database sync status:"
  run_compose exec -T flask-app env FLASK_APP=manage.py flask protein-database-sync-status || true

  log "Latest TM annotation sync status:"
  run_compose exec -T flask-app env FLASK_APP=manage.py flask tm-prediction-status || true

  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    show_stateful_task_status "Machine learning" "${ML_TASK_ID:-}"
    show_stateful_task_status "TM annotation sync" "${TM_ANNOTATION_TASK_ID:-}"
  else
    warn "No launcher state file found yet."
  fi
}

tail_logs() {
  load_env_file
  local service="${1:-flask-app}"
  run_compose logs -f "$service"
}

stop_stack() {
  load_env_file
  log "Stopping service stack..."
  run_compose down
}

COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
  usage
  exit 1
fi
if [[ "$COMMAND" == "-h" || "$COMMAND" == "--help" || "$COMMAND" == "help" ]]; then
  usage
  exit 0
fi
shift || true

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="${2:-}"
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
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift
      ;;
    --skip-ml)
      SKIP_ML=1
      shift
      ;;
    --skip-tm)
      SKIP_TM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

case "$COMMAND" in
  up)
    start_stack
    ;;
  status)
    show_status
    ;;
  logs)
    tail_logs "${POSITIONAL[0]:-flask-app}"
    ;;
  down)
    stop_stack
    ;;
  *)
    usage
    die "Unknown command: $COMMAND"
    ;;
esac
