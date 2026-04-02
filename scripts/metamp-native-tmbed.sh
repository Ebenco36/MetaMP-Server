#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="$ROOT_DIR/.env.docker.deployment"
RUNTIME_ROOT="$ROOT_DIR/release-snapshots/native-tmbed-runtime"
VENV_DIR=""
PYTHON_BIN=""
DEVICE_MODE="auto"
BATCH_SIZE=""
MAX_WORKERS=""
LIMIT=""
CSV_OUT=""
RUN_ALL=0
INCLUDE_COMPLETED=0
RETRY_ERRORS=0
COMMAND_NAME="${1:-}"
shift || true
PDB_CODES=()
PDB_CODE_FILE=""
DETECTED_BACKEND=""
HOST_UNAME_S="$(uname -s)"
HOST_UNAME_M="$(uname -m)"

log() {
  printf '[MetaMP Native TMbed] %s\n' "$*"
}

die() {
  printf '[MetaMP Native TMbed][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/metamp-native-tmbed.sh doctor [options]
  bash scripts/metamp-native-tmbed.sh sync [options]
  bash scripts/metamp-native-tmbed.sh fallback [options]

This wrapper runs TMbed outside Docker so MetaMP can use:
- Apple Silicon MPS on native macOS
- CUDA on native Linux
- CPU everywhere else

Commands:
  doctor     Show the detected native TMbed runtime and device availability.
  sync       Run the direct TMbed sync command.
  fallback   Run the verified fallback pipeline restricted to TMbed.

Options:
  --env-file PATH      Env file to source. Default: .env.docker.deployment
  --runtime-root PATH  Local runtime/cache directory. Default: release-snapshots/native-tmbed-runtime
  --venv-dir PATH      Virtualenv directory to use.
  --python-bin PATH    Explicit Python interpreter to use.
  --device MODE        One of: auto, gpu, cpu. Default: auto
  --all                Run across all eligible targets.
  --limit N            Restrict the run to the first N targets.
  --pdb-code CODE      Restrict the run to one or more PDB codes. Repeat as needed.
  --pdb-code-file PATH Read one or more PDB codes from a file.
  --batch-size N       Optional TMbed batch size override.
  --max-workers N      Optional TMbed worker-count override.
  --csv-out PATH       Optional resumable/output CSV path for sync mode.
  --include-completed  Rerun TMbed even if MetaMP already has rows.
  --retry-errors      Retry stored MetaMP fallback error rows without forcing successful rows to rerun.
  -h, --help           Show this help.

Examples:
  bash scripts/metamp-native-tmbed.sh doctor
  bash scripts/metamp-native-tmbed.sh sync --all --device auto
  bash scripts/metamp-native-tmbed.sh fallback --all --device gpu --batch-size 10
  bash scripts/metamp-native-tmbed.sh sync --pdb-code 6N7G --device cpu --include-completed
EOF
}

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

normalize_pdb_code() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
}

load_pdb_codes_from_file() {
  [[ -n "$PDB_CODE_FILE" ]] || return 0
  require_file "$PDB_CODE_FILE"
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local token
    for token in ${raw_line//,/ }; do
      token="$(normalize_pdb_code "$token")"
      if [[ -n "$token" ]]; then
        PDB_CODES+=("$token")
      fi
    done
  done < "$PDB_CODE_FILE"
}

load_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

resolve_python_runtime() {
  if [[ -n "$PYTHON_BIN" ]]; then
    [[ -x "$PYTHON_BIN" ]] || die "Python interpreter not found or not executable: $PYTHON_BIN"
    return 0
  fi

  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    VENV_DIR="${VIRTUAL_ENV}"
    return 0
  fi

  if [[ -n "$VENV_DIR" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
    return 0
  fi

  local candidate
  for candidate in \
    "$ROOT_DIR/.mpvis" \
    "$ROOT_DIR/.venv" \
    "$ROOT_DIR/.venv_mpvis" \
    "$ROOT_DIR/.venv312"
  do
    if [[ -x "$candidate/bin/python" ]]; then
      VENV_DIR="$candidate"
      PYTHON_BIN="$candidate/bin/python"
      return 0
    fi
  done

  die "No usable native Python environment found. Activate a venv, pass --python-bin, or pass --venv-dir."
}

detect_native_backend() {
  DETECTED_BACKEND="$("$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception:
    print("cpu")
    raise SystemExit(0)

try:
    if torch.cuda.is_available():
        print("cuda")
        raise SystemExit(0)
except Exception:
    pass

try:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("mps")
        raise SystemExit(0)
except Exception:
    pass

print("cpu")
PY
)"
}

prepare_runtime_paths() {
  mkdir -p \
    "$RUNTIME_ROOT" \
    "$RUNTIME_ROOT/data/models/semi-supervised" \
    "$RUNTIME_ROOT/data/tmbed-models" \
    "$RUNTIME_ROOT/data/tm_predictions/external"
}

rewrite_container_hosts_for_native() {
  export DB_HOST="${DB_HOST:-postgres}"
  export REDIS_HOST="${REDIS_HOST:-redis}"

  if [[ "$DB_HOST" == "postgres" ]]; then
    export DB_HOST="127.0.0.1"
  fi
  if [[ "$REDIS_HOST" == "redis" ]]; then
    export REDIS_HOST="127.0.0.1"
  fi

  if [[ -n "${DATABASE_URL:-}" ]]; then
    DATABASE_URL="${DATABASE_URL//@postgres:/@127.0.0.1:}"
  elif [[ -n "${DB_USER:-}" && -n "${DB_PASSWORD:-}" && -n "${DB_NAME:-}" ]]; then
    DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT:-5432}/${DB_NAME}"
  fi
  export DATABASE_URL="${DATABASE_URL:-}"

  if [[ -n "${CELERY_BROKER_URL:-}" ]]; then
    CELERY_BROKER_URL="${CELERY_BROKER_URL//redis:\/\/redis:/redis:\/\/127.0.0.1:}"
  else
    CELERY_BROKER_URL="redis://${REDIS_HOST}:6379/0"
  fi
  export CELERY_BROKER_URL

  if [[ -n "${CELERY_RESULT_BACKEND:-}" ]]; then
    CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND//redis:\/\/redis:/redis:\/\/127.0.0.1:}"
  else
    CELERY_RESULT_BACKEND="redis://${REDIS_HOST}:6379/0"
  fi
  export CELERY_RESULT_BACKEND
}

prepare_native_env() {
  load_env_file
  prepare_runtime_paths
  rewrite_container_hosts_for_native
  detect_native_backend

  export FLASK_APP=manage.py
  export FLASK_ENV=production
  export APP_SETTINGS=config.config.ProductionConfig
  export INGESTION_DATASET_BASE_DIR="$ROOT_DIR/datasets"
  if [[ -f "$ROOT_DIR/datasets/expert_annotation_predicted.csv" ]]; then
    export DASHBOARD_ANNOTATION_DATASET_PATH="$ROOT_DIR/datasets/expert_annotation_predicted.csv"
  fi
  export SEMI_SUPERVISED_MODEL_DIR="$RUNTIME_ROOT/data/models/semi-supervised"
  export LIVE_GROUP_PREDICTIONS_PATH="$RUNTIME_ROOT/data/models/live_group_predictions.csv"
  export TM_PREDICTION_OUTPUT_CSV="$RUNTIME_ROOT/data/tm_predictions/tm_summary.csv"
  export OPTIONAL_TM_PREDICTION_BASE_DIR="$RUNTIME_ROOT/data/tm_predictions/external"
  export TMBED_MODEL_DIR="$RUNTIME_ROOT/data/tmbed-models"
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  case "$DEVICE_MODE" in
    auto|gpu)
      export TM_PREDICTION_USE_GPU=true
      ;;
    cpu)
      export TM_PREDICTION_USE_GPU=false
      ;;
    *)
      die "Unsupported --device value: $DEVICE_MODE"
      ;;
  esac

  if [[ "$DETECTED_BACKEND" == "mps" || ( "$HOST_UNAME_S" == "Darwin" && "$HOST_UNAME_M" == "arm64" && "$DEVICE_MODE" != "cpu" ) ]]; then
    if [[ -n "$BATCH_SIZE" ]]; then
      if [[ "$BATCH_SIZE" -gt 1 ]]; then
        log "Apple Silicon native TMbed run detected; capping outer batch size from $BATCH_SIZE to 1 for stability."
        BATCH_SIZE="1"
      fi
    else
      BATCH_SIZE="1"
      log "Apple Silicon native TMbed run detected; using outer batch size 1 for stability."
    fi
    if [[ -z "$MAX_WORKERS" ]]; then
      MAX_WORKERS="1"
    fi
  fi

  if [[ -n "$BATCH_SIZE" ]]; then
    export TM_PREDICTION_BATCH_SIZE="$BATCH_SIZE"
  fi
  if [[ -n "$MAX_WORKERS" ]]; then
    export TM_PREDICTION_MAX_WORKERS="$MAX_WORKERS"
  fi
}

run_flask() {
  "$PYTHON_BIN" -m flask --app manage.py "$@"
}

append_scope_args() {
  local -n _args_ref=$1

  if [[ "$RUN_ALL" -eq 1 ]]; then
    _args_ref+=(--all)
  fi

  if [[ -n "$LIMIT" ]]; then
    _args_ref+=(--limit "$LIMIT")
  fi

  local pdb_code
  for pdb_code in "${PDB_CODES[@]}"; do
    _args_ref+=(--pdb-code "$pdb_code")
  done

  if [[ -n "$BATCH_SIZE" ]]; then
    _args_ref+=(--batch-size "$BATCH_SIZE")
  fi

  if [[ -n "$MAX_WORKERS" ]]; then
    _args_ref+=(--max-workers "$MAX_WORKERS")
  fi
}

append_device_args() {
  local -n _args_ref=$1
  case "$DEVICE_MODE" in
    auto|gpu)
      _args_ref+=(--use-gpu)
      ;;
    cpu)
      _args_ref+=(--no-gpu)
      ;;
  esac
}

require_explicit_scope() {
  if [[ "$RUN_ALL" -eq 0 && -z "$LIMIT" && "${#PDB_CODES[@]}" -eq 0 ]]; then
    die "Provide --all, --limit, or at least one --pdb-code so the TMbed scope is explicit."
  fi
}

run_doctor() {
  log "Inspecting native TMbed runtime ..."
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import platform
import sys

payload = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "machine": platform.machine(),
  "database_url": os.getenv("DATABASE_URL"),
  "redis_host": os.getenv("REDIS_HOST"),
  "tmbed_model_dir": os.getenv("TMBED_MODEL_DIR"),
  "tm_prediction_use_gpu": os.getenv("TM_PREDICTION_USE_GPU"),
    "tm_prediction_batch_size": os.getenv("TM_PREDICTION_BATCH_SIZE"),
    "tm_prediction_max_workers": os.getenv("TM_PREDICTION_MAX_WORKERS"),
  "tmbed_importable": importlib.util.find_spec("tmbed") is not None,
}

try:
    import torch
    payload["torch_importable"] = True
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["mps_available"] = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
except Exception as exc:
    payload["torch_importable"] = False
    payload["torch_error"] = str(exc)

print(json.dumps(payload, indent=2))
PY
}

run_sync() {
  require_explicit_scope
  local args=(sync-tmbed-predictions)
  append_scope_args args
  append_device_args args
  if [[ "$INCLUDE_COMPLETED" -eq 1 ]]; then
    args+=(--refresh)
  fi
  if [[ -n "$CSV_OUT" ]]; then
    args+=(--csv-out "$CSV_OUT")
  fi
  log "Running native TMbed sync ..."
  run_flask "${args[@]}"
}

run_fallback() {
  require_explicit_scope
  local args=(run-verified-tm-fallbacks --mode fallback_only --fallback-method TMbed)
  append_scope_args args
  append_device_args args
  if [[ "$INCLUDE_COMPLETED" -eq 1 ]]; then
    args+=(--include-completed)
  fi
  if [[ "$RETRY_ERRORS" -eq 1 ]]; then
    args+=(--retry-errors)
  fi
  log "Running native TMbed fallback pipeline ..."
  run_flask "${args[@]}"
}

require_project_inputs() {
  require_file "$ROOT_DIR/manage.py"
  require_file "$ROOT_DIR/requirements.txt"
  require_file "$ROOT_DIR/requirements-ml.txt"
  [[ -d "$ROOT_DIR/datasets" ]] || die "Required dataset directory not found: $ROOT_DIR/datasets"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      DEVICE_MODE="$2"
      shift 2
      ;;
    --all)
      RUN_ALL=1
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --pdb-code)
      PDB_CODES+=("$(normalize_pdb_code "$2")")
      shift 2
      ;;
    --pdb-code-file)
      PDB_CODE_FILE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --csv-out)
      CSV_OUT="$2"
      shift 2
      ;;
    --include-completed|--refresh)
      INCLUDE_COMPLETED=1
      shift
      ;;
    --retry-errors)
      RETRY_ERRORS=1
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

case "$COMMAND_NAME" in
  doctor|sync|fallback)
    ;;
  ""|-h|--help)
    usage
    exit 0
    ;;
  *)
    die "Unknown command: $COMMAND_NAME"
    ;;
esac

cd "$ROOT_DIR"
require_project_inputs
load_pdb_codes_from_file
resolve_python_runtime
prepare_native_env

log "Python runtime: $PYTHON_BIN"
if [[ -n "$VENV_DIR" ]]; then
  log "Virtualenv: $VENV_DIR"
fi
log "Runtime root: $RUNTIME_ROOT"
log "Device mode: $DEVICE_MODE"
log "Detected backend: ${DETECTED_BACKEND:-unknown}"
log "Database URL: ${DATABASE_URL:-<unset>}"

case "$COMMAND_NAME" in
  doctor)
    run_doctor
    ;;
  sync)
    run_sync
    ;;
  fallback)
    run_fallback
    ;;
esac
