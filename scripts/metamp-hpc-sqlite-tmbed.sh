#!/usr/bin/env bash
#SBATCH --job-name=metamp_sqlite_tmbed
#SBATCH --output=metamp_sqlite_tmbed.out
#SBATCH --error=metamp_sqlite_tmbed.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"
DEFAULT_RUNTIME_ROOT="$ROOT_DIR/release-snapshots/sqlite-tmbed-runtime"

ENV_FILE="$DEFAULT_ENV_FILE"
RUNTIME_ROOT="$DEFAULT_RUNTIME_ROOT"
SQLITE_PATH=""
CSV_EXPORT_PATH=""
REBUILD_DB=0
INCLUDE_COMPLETED=0
USE_GPU_MODE="${USE_GPU_MODE:-auto}"
TMBED_BATCH_SIZE="${TMBED_BATCH_SIZE:-1}"
TMBED_MAX_WORKERS="${TMBED_MAX_WORKERS:-1}"

log() {
  printf '[MetaMP SQLite TMbed] %s\n' "$*"
}

die() {
  printf '[MetaMP SQLite TMbed][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/metamp-hpc-sqlite-tmbed.sh
  bash scripts/metamp-hpc-sqlite-tmbed.sh [options]

This script creates a SQLite-backed MetaMP runtime, loads the checked-in dataset snapshots,
runs the TMbed-only fallback pipeline, and exports membrane_protein_tmalphafold_predictions to CSV.

Options:
  --env-file PATH          Use a different env file for shared defaults. Default: .env.docker.deployment
  --runtime-root PATH      Runtime working directory. Default: release-snapshots/sqlite-tmbed-runtime
  --sqlite-path PATH       SQLite database file path. Default: <runtime-root>/metamp_hpc.sqlite
  --csv-export PATH        CSV export path. Default: <runtime-root>/membrane_protein_tmalphafold_predictions.csv
  --rebuild-db             Remove the SQLite DB and reload datasets before the TMbed run.
  --include-completed      Rerun TMbed even if MetaMP already has SQLite rows for a record.
  --gpu-mode MODE          One of: auto, on, off. Default: auto
  --tmbed-batch-size N     TMbed batch size. Default: 1
  --tmbed-max-workers N    TMbed worker count. Default: 1
  -h, --help               Show this help.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

resolve_gpu_flag() {
  case "$USE_GPU_MODE" in
    on)
      printf '%s\n' "true"
      ;;
    off)
      printf '%s\n' "false"
      ;;
    auto)
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        printf '%s\n' "true"
      else
        printf '%s\n' "false"
      fi
      ;;
    *)
      die "Unsupported --gpu-mode value: $USE_GPU_MODE"
      ;;
  esac
}

load_base_env() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

prepare_runtime_paths() {
  mkdir -p "$RUNTIME_ROOT"
  if [[ -z "$SQLITE_PATH" ]]; then
    SQLITE_PATH="$RUNTIME_ROOT/metamp_hpc.sqlite"
  fi
  if [[ -z "$CSV_EXPORT_PATH" ]]; then
    CSV_EXPORT_PATH="$RUNTIME_ROOT/membrane_protein_tmalphafold_predictions.csv"
  fi
  mkdir -p \
    "$RUNTIME_ROOT/data/models/semi-supervised" \
    "$RUNTIME_ROOT/data/tmbed-models" \
    "$RUNTIME_ROOT/data/tm_predictions/external"
}

prepare_python_env() {
  load_base_env
  prepare_runtime_paths

  export FLASK_APP=manage.py
  export FLASK_ENV=production
  export APP_SETTINGS=config.config.ProductionConfig
  export DATABASE_URL="sqlite:///$SQLITE_PATH"
  export INGESTION_DATASET_BASE_DIR="$ROOT_DIR/datasets"
  export DASHBOARD_ANNOTATION_DATASET_PATH="$ROOT_DIR/datasets/expert_annotation_predicted.csv"
  export SEMI_SUPERVISED_MODEL_DIR="$RUNTIME_ROOT/data/models/semi-supervised"
  export LIVE_GROUP_PREDICTIONS_PATH="$RUNTIME_ROOT/data/models/live_group_predictions.csv"
  export TM_PREDICTION_OUTPUT_CSV="$RUNTIME_ROOT/data/tm_predictions/tm_summary.csv"
  export OPTIONAL_TM_PREDICTION_BASE_DIR="$RUNTIME_ROOT/data/tm_predictions/external"
  export TMBED_MODEL_DIR="$RUNTIME_ROOT/data/tmbed-models"
  export TM_PREDICTION_BATCH_SIZE="$TMBED_BATCH_SIZE"
  export TM_PREDICTION_MAX_WORKERS="$TMBED_MAX_WORKERS"
  export TM_PREDICTION_USE_GPU="$(resolve_gpu_flag)"
  export TMALPHAFOLD_TMBED_USE_GPU="$TM_PREDICTION_USE_GPU"
}

run_flask() {
  python3 -m flask "$@"
}

bootstrap_sqlite_db() {
  if [[ "$REBUILD_DB" -eq 1 && -f "$SQLITE_PATH" ]]; then
    log "Removing existing SQLite database at $SQLITE_PATH"
    rm -f "$SQLITE_PATH"
  fi

  if [[ ! -f "$SQLITE_PATH" ]]; then
    log "Creating SQLite database and loading dataset snapshots..."
    run_flask load-protein-datasets --clear-db --no-seed-defaults
  else
    log "SQLite database already exists; syncing schema only."
    run_flask sync-protein-schema
  fi
}

run_tmbed_only() {
  local args=(run-verified-tm-fallbacks --mode fallback_only --all --fallback-method TMbed)
  if [[ "$TM_PREDICTION_USE_GPU" == "true" ]]; then
    args+=(--use-gpu)
  else
    args+=(--no-gpu)
  fi
  args+=(--batch-size "$TMBED_BATCH_SIZE" --max-workers "$TMBED_MAX_WORKERS")
  if [[ "$INCLUDE_COMPLETED" -eq 1 ]]; then
    args+=(--include-completed)
  fi

  log "Running TMbed-only fallback pipeline against SQLite ..."
  run_flask "${args[@]}"
}

export_predictions_csv() {
  log "Exporting membrane_protein_tmalphafold_predictions to $CSV_EXPORT_PATH"
  run_flask export-tmalphafold-predictions-csv --output-path "$CSV_EXPORT_PATH"
}

main() {
  require_command python3
  prepare_python_env

  log "SQLite database: $SQLITE_PATH"
  log "CSV export path: $CSV_EXPORT_PATH"
  log "TMbed GPU enabled: $TM_PREDICTION_USE_GPU"

  bootstrap_sqlite_db
  run_tmbed_only
  export_predictions_csv

  log "SQLite TMbed workflow completed."
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
    --sqlite-path)
      SQLITE_PATH="$2"
      shift 2
      ;;
    --csv-export)
      CSV_EXPORT_PATH="$2"
      shift 2
      ;;
    --rebuild-db)
      REBUILD_DB=1
      shift
      ;;
    --include-completed)
      INCLUDE_COMPLETED=1
      shift
      ;;
    --gpu-mode)
      USE_GPU_MODE="$2"
      shift 2
      ;;
    --tmbed-batch-size)
      TMBED_BATCH_SIZE="$2"
      shift 2
      ;;
    --tmbed-max-workers)
      TMBED_MAX_WORKERS="$2"
      shift 2
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

main
