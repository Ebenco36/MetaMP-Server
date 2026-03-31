#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"
STATE_FILE="$ROOT_DIR/.metamp-production-bootstrap.state"
BOOTSTRAP_MARKER_PATH="/var/app/data/bootstrap/production-bootstrap-state.json"
RUNTIME_DATASET_DIR=""
DOCKER_BIN="${DOCKER_BIN:-}"

ENV_FILE="$DEFAULT_ENV_FILE"
WITH_FRONTEND=0
WITH_LOCAL_FRONTEND=0
SKIP_BUILD=0
NO_CACHE=0
FORCE_BOOTSTRAP=0
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-10800}"

log() {
  printf '[MetaMP Bootstrap] %s\n' "$*"
}

warn() {
  printf '[MetaMP Bootstrap][warn] %s\n' "$*" >&2
}

die() {
  printf '[MetaMP Bootstrap][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-production-bootstrap.sh run [options]
  ./scripts/metamp-production-bootstrap.sh status [options]
  ./scripts/metamp-production-bootstrap.sh logs [service] [options]
  ./scripts/metamp-production-bootstrap.sh reset [options]

Commands:
  run       Build, start, fully bootstrap, normalize TM predictions, validate, and mark the local production state.
  status    Show service state, bootstrap marker state, and latest maintenance status.
  logs      Tail logs for a service. Defaults to flask-app.
  reset     Remove the bootstrap marker so the next run executes the full bootstrap again.

Options:
  --env-file PATH            Use a different env file. Default: .env.docker.deployment
  --with-frontend            Start the frontend service too.
  --with-local-frontend      Start the frontend with docker-compose.local-frontend.yml.
  --skip-build               Skip docker compose build.
  --no-cache                 Build images without cache.
  --force-bootstrap          Ignore any existing bootstrap marker and rerun the full process.
  --wait-timeout SECONDS     Max wait time for long ML/TM tasks. Default: 10800
  -h, --help                 Show this help.
EOF
}

refresh_frontend_image() {
  if [[ "$WITH_FRONTEND" -ne 1 || "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
    return 0
  fi
  log "Refreshing frontend image from the configured registry..."
  run_compose pull frontend
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
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
  die "No supported container CLI was found. Set DOCKER_BIN to docker or podman explicitly."
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

run_compose() {
  resolve_docker_bin
  local args=()
  while IFS= read -r line; do
    args+=("$line")
  done < <(compose_args)
  "$DOCKER_BIN" compose "${args[@]}" "$@"
}

flask_exec() {
  run_compose exec -T flask-app env FLASK_APP=manage.py flask "$@"
}

flask_run_once() {
  run_compose run --rm -T flask-app env FLASK_APP=manage.py flask "$@"
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

restart_runtime_services() {
  local services=(flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat)

  log "Starting application services..."
  run_compose up -d "${services[@]}"
  wait_for_backend
}

seed_runtime_datasets() {
  local source_dir="$ROOT_DIR/datasets"

  if [[ ! -d "$source_dir" ]]; then
    warn "No local datasets directory found at $source_dir; skipping runtime dataset seeding."
    return 0
  fi

  log "Seeding runtime dataset directory at ${RUNTIME_DATASET_DIR} from local snapshots..."
  run_compose exec -T flask-app sh -lc "mkdir -p '$RUNTIME_DATASET_DIR'"
  run_compose cp "$source_dir/." "flask-app:$RUNTIME_DATASET_DIR"
  run_compose exec -T -u root flask-app sh -lc "chmod -R a+rwX '$RUNTIME_DATASET_DIR' || true"
}

queue_ml_job() {
  local output
  output="$(flask_exec queue-machine-learning-job)"
  printf '%s\n' "$output" >&2
  printf '%s\n' "$output" | awk -F': ' 'END {print $NF}'
}

queue_tmalphafold_sync() {
  local args=(queue-tmalphafold-sync --with-tmdet --with-tmbed --tmbed-refresh)
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
  output="$(flask_exec "${args[@]}")"
  printf '%s\n' "$output" >&2
  printf '%s\n' "$output" | awk -F': ' 'END {print $NF}'
}

get_task_status() {
  local task_id="$1"
  local output
  output="$(flask_exec celery-task-status --task-id "$task_id" 2>/dev/null || true)"
  printf '%s\n' "$output" | grep -m1 '"status"' | awk -F'"' '{print $4}'
}

wait_for_task() {
  local label="$1"
  local task_id="$2"
  local started_at
  started_at="$(date +%s)"

  [[ -n "$task_id" ]] || die "Missing task id for ${label}"

  log "Waiting for ${label} task ${task_id}"
  while true; do
    local status
    status="$(get_task_status "$task_id")"

    case "$status" in
      succeeded|success)
        log "${label} task ${task_id} succeeded."
        return 0
        ;;
      failed|failure)
        warn "${label} task ${task_id} failed."
        flask_exec celery-task-status --task-id "$task_id" || true
        return 1
        ;;
      skipped)
        warn "${label} task ${task_id} was skipped."
        return 0
        ;;
      started|running|PENDING|pending|"")
        ;;
      *)
        log "${label} task ${task_id} current status: ${status}"
        ;;
    esac

    if (( "$(date +%s)" - started_at > WAIT_TIMEOUT_SECONDS )); then
      die "${label} task ${task_id} did not finish within ${WAIT_TIMEOUT_SECONDS} seconds."
    fi
    sleep 10
  done
}

bootstrap_marker_exists() {
  run_compose exec -T flask-app sh -lc "test -f '$BOOTSTRAP_MARKER_PATH'" >/dev/null 2>&1
}

show_bootstrap_marker() {
  run_compose exec -T flask-app sh -lc "cat '$BOOTSTRAP_MARKER_PATH'" || true
}

write_bootstrap_marker() {
  local ml_task_id="$1"
  local tmalphafold_task_id="$2"
  local tmp_file
  tmp_file="$(mktemp)"

  cat >"$tmp_file" <<EOF
{
  "status": "ready",
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "env_file": "$ENV_FILE",
  "ml_task_id": "$ml_task_id",
  "tmalphafold_task_id": "$tmalphafold_task_id",
  "maintenance_schedule": "celery-beat every 30 days",
  "validator_command": "python -m flask --app manage.py validate-dashboard-regressions"
}
EOF

  run_compose exec -T flask-app sh -lc "mkdir -p \"$(dirname "$BOOTSTRAP_MARKER_PATH")\""
  run_compose cp "$tmp_file" "flask-app:$BOOTSTRAP_MARKER_PATH" >/dev/null
  rm -f "$tmp_file"
}

clear_bootstrap_marker() {
  run_compose exec -T flask-app sh -lc "rm -f '$BOOTSTRAP_MARKER_PATH'" || true
}

write_state_file() {
  local ml_task_id="$1"
  local tmalphafold_task_id="$2"
  cat >"$STATE_FILE" <<EOF
ENV_FILE=$ENV_FILE
WITH_FRONTEND=$WITH_FRONTEND
WITH_LOCAL_FRONTEND=$WITH_LOCAL_FRONTEND
ML_TASK_ID=$ml_task_id
TMALPHAFOLD_TASK_ID=$tmalphafold_task_id
LAST_BOOTSTRAP_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
}

start_stack() {
  load_env_file
  require_command curl
  resolve_docker_bin

  local services=(postgres redis flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat)
  local build_services=(flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat)

  if [[ "$WITH_FRONTEND" -eq 1 || "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
    services+=(frontend)
  fi
  if [[ "$WITH_LOCAL_FRONTEND" -eq 1 ]]; then
    build_services+=(frontend)
  fi

  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    local build_args=(build)
    if [[ "$NO_CACHE" -eq 1 ]]; then
      build_args+=(--no-cache)
    fi
    build_args+=("${build_services[@]}")
    log "Building service images..."
    run_compose "${build_args[@]}"
  else
    log "Skipping image build."
  fi

  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    refresh_frontend_image
  fi

  log "Starting service stack..."
  run_compose up -d "${services[@]}"
  wait_for_backend
}

run_bootstrap() {
  start_stack

  if bootstrap_marker_exists && [[ "$FORCE_BOOTSTRAP" -eq 0 ]]; then
    log "Existing bootstrap marker found. Reusing local production state."
    show_bootstrap_marker
    return 0
  fi

  seed_runtime_datasets

  log "Stopping application services to avoid database locks during schema sync..."
  run_compose stop flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat >/dev/null

  log "Running full production bootstrap..."
  flask_run_once sync-protein-database

  restart_runtime_services

  log "Dropping legacy TM predictor columns from membrane_proteins..."
  flask_exec drop-legacy-tm-predictor-columns

  log "Queueing full TMAlphaFold normalized sync..."
  local tmalphafold_task_id
  tmalphafold_task_id="$(queue_tmalphafold_sync)"

  log "Queueing ML training..."
  local ml_task_id
  ml_task_id="$(queue_ml_job)"

  wait_for_task "Machine learning" "$ml_task_id"
  wait_for_task "TMAlphaFold sync" "$tmalphafold_task_id"

  log "Exporting discrepancy benchmark..."
  flask_exec export-discrepancy-benchmark

  log "Running regression validator..."
  flask_exec validate-dashboard-regressions

  write_bootstrap_marker "$ml_task_id" "$tmalphafold_task_id"
  write_state_file "$ml_task_id" "$tmalphafold_task_id"

  log "Production bootstrap completed successfully."
  log "Subsequent monthly maintenance is handled by celery-beat via shared-task-monthly-production-maintenance."
}

show_status() {
  load_env_file
  log "Container status:"
  run_compose ps || true

  log "Backend readiness:"
  curl --silent "http://127.0.0.1:${FLASK_RUN_PORT:-5400}/api/v1/health/ready" || true
  printf '\n'

  log "Bootstrap marker:"
  if bootstrap_marker_exists; then
    show_bootstrap_marker
  else
    warn "No bootstrap marker found."
  fi

  log "Latest maintenance status:"
  flask_exec production-maintenance-status || true

  log "Latest protein database sync status:"
  flask_exec protein-database-sync-status || true

  log "Latest TM annotation sync status:"
  flask_exec tm-prediction-status || true
}

tail_logs() {
  load_env_file
  local service="${1:-flask-app}"
  run_compose logs -f "$service"
}

reset_bootstrap() {
  load_env_file
  clear_bootstrap_marker
  rm -f "$STATE_FILE"
  log "Bootstrap marker cleared. The next run will execute the full bootstrap."
}

COMMAND="${1:-run}"
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
    --with-frontend)
      WITH_FRONTEND=1
      shift
      ;;
    --with-local-frontend)
      WITH_LOCAL_FRONTEND=1
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
    --force-bootstrap)
      FORCE_BOOTSTRAP=1
      shift
      ;;
    --wait-timeout)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      WAIT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$COMMAND" == "logs" && -z "${LOG_SERVICE_SET:-}" ]]; then
        LOG_SERVICE="$1"
        LOG_SERVICE_SET=1
        shift
      else
        die "Unknown option or argument: $1"
      fi
      ;;
  esac
done

case "$COMMAND" in
  run)
    run_bootstrap
    ;;
  status)
    show_status
    ;;
  logs)
    tail_logs "${LOG_SERVICE:-flask-app}"
    ;;
  reset)
    reset_bootstrap
    ;;
  *)
    usage
    exit 1
    ;;
esac
