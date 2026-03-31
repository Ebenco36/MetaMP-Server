#!/usr/bin/env bash
#SBATCH --job-name=metamp_full_release
#SBATCH --output=metamp_full_release.out
#SBATCH --error=metamp_full_release.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=main
#SBATCH --nodelist=hpc-node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"
DEFAULT_RELEASE_ROOT="$ROOT_DIR/release-snapshots"

ENV_FILE="$DEFAULT_ENV_FILE"
SNAPSHOT_DIR=""
SNAPSHOT_ARCHIVE=""
TOP_MODELS=5
NO_CACHE=0
FORCE_BOOTSTRAP=1
PUSH_BACKEND_IMAGES=1
PUSH_STATEFUL_POSTGRES=1
RUN_TMBED=1
RUN_TMDET=1
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-432000}"
IMAGE_TAG_OVERRIDE=""
REGISTRY_NAMESPACE_OVERRIDE=""
USE_GPU_MODE="${USE_GPU_MODE:-auto}"
TMBED_BATCH_SIZE="${TMBED_BATCH_SIZE:-1}"
TMBED_MAX_WORKERS="${TMBED_MAX_WORKERS:-1}"
VALIDATE_AFTER_RUN=1
TEMP_ENV_FILE=""

log() {
  printf '[MetaMP HPC] %s\n' "$*"
}

die() {
  printf '[MetaMP HPC][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/metamp-hpc-run.sh
  bash scripts/metamp-hpc-run.sh [options]

This script performs the full backend production flow:
  1. build and bootstrap the stack
  2. run the verified local fallback TM methods in staged order
  3. validate the dashboard/runtime
  4. export a full runtime snapshot
  5. archive that snapshot as a zip file
  6. push backend-only application images
  7. optionally build and push a stateful PostgreSQL image from the live dump

Options:
  --env-file PATH           Use a different env file. Default: .env.docker.deployment
  --snapshot-dir PATH       Export snapshot directory. Default: release-snapshots/metamp-hpc-<timestamp>
  --snapshot-archive PATH   Zip archive path. Default: <snapshot-dir>.zip
  --top-models N            Retain top N ML bundles in snapshot. Default: 5
  --image-tag TAG           Override IMAGE_TAG for the release images.
  --registry-namespace NS   Override REGISTRY_NAMESPACE for the release images.
  --no-cache                Build without Docker cache.
  --skip-image-push         Do not push backend application images.
  --skip-stateful-postgres  Do not build/push the stateful PostgreSQL image.
  --skip-tmbed              Skip TMbed in the fallback stage.
  --skip-tmdet              Skip TMDET in the fallback stage.
  --gpu-mode MODE           One of: auto, on, off. Default: auto
  --tmbed-batch-size N      TMbed batch size for the HPC fallback pass. Default: 1
  --tmbed-max-workers N     TMbed worker count for the HPC fallback pass. Default: 1
  --wait-timeout SECONDS    Bootstrap async wait timeout. Default: 432000
  --skip-validate           Skip validate-dashboard-regressions after pipeline completion.
  -h, --help                Show this help.
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

load_env_file() {
  [[ -f "$ENV_FILE" ]] || die "Env file not found: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

prepare_runtime_env_file() {
  if [[ -z "$IMAGE_TAG_OVERRIDE" && -z "$REGISTRY_NAMESPACE_OVERRIDE" ]]; then
    return 0
  fi

  TEMP_ENV_FILE="$(mktemp "${ROOT_DIR}/.env.hpc.release.XXXXXX")"
  cp "$ENV_FILE" "$TEMP_ENV_FILE"
  if [[ -n "$IMAGE_TAG_OVERRIDE" ]]; then
    python3 - "$TEMP_ENV_FILE" IMAGE_TAG "$IMAGE_TAG_OVERRIDE" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text().splitlines()
updated = False
for index, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[index] = f"{key}={value}"
        updated = True
        break
if not updated:
    lines.append(f"{key}={value}")
path.write_text("\n".join(lines) + "\n")
PY
  fi
  if [[ -n "$REGISTRY_NAMESPACE_OVERRIDE" ]]; then
    python3 - "$TEMP_ENV_FILE" REGISTRY_NAMESPACE "$REGISTRY_NAMESPACE_OVERRIDE" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text().splitlines()
updated = False
for index, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[index] = f"{key}={value}"
        updated = True
        break
if not updated:
    lines.append(f"{key}={value}")
path.write_text("\n".join(lines) + "\n")
PY
  fi
  ENV_FILE="$TEMP_ENV_FILE"
}

compose_args() {
  printf '%s\n' --env-file "$ENV_FILE" -f "$ROOT_DIR/docker-compose.yml"
}

run_compose() {
  local args=()
  while IFS= read -r line; do
    args+=("$line")
  done < <(compose_args)
  docker compose "${args[@]}" "$@"
}

flask_exec() {
  run_compose exec -T flask-app env FLASK_APP=manage.py flask "$@"
}

default_snapshot_paths() {
  local timestamp
  timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  if [[ -z "$SNAPSHOT_DIR" ]]; then
    SNAPSHOT_DIR="$DEFAULT_RELEASE_ROOT/metamp-hpc-$timestamp"
  fi
  if [[ -z "$SNAPSHOT_ARCHIVE" ]]; then
    SNAPSHOT_ARCHIVE="${SNAPSHOT_DIR}.zip"
  fi
}

archive_snapshot_zip() {
  local snapshot_dir="$1"
  local archive_path="$2"
  mkdir -p "$(dirname "$archive_path")"
  python3 - "$snapshot_dir" "$archive_path" <<'PY'
import pathlib
import shutil
import sys

snapshot_dir = pathlib.Path(sys.argv[1]).resolve()
archive_path = pathlib.Path(sys.argv[2]).resolve()
archive_path.parent.mkdir(parents=True, exist_ok=True)
base_name = str(archive_path.with_suffix(""))
root_dir = str(snapshot_dir.parent)
base_dir = snapshot_dir.name
created = shutil.make_archive(base_name, "zip", root_dir=root_dir, base_dir=base_dir)
print(created)
PY
}

run_bootstrap() {
  local bootstrap_args=(run --env-file "$ENV_FILE" --wait-timeout "$WAIT_TIMEOUT_SECONDS")
  if [[ "$FORCE_BOOTSTRAP" -eq 1 ]]; then
    bootstrap_args+=(--force-bootstrap)
  fi
  if [[ "$NO_CACHE" -eq 1 ]]; then
    bootstrap_args+=(--no-cache)
  fi
  log "Running full production bootstrap..."
  "$ROOT_DIR/scripts/metamp-production-bootstrap.sh" "${bootstrap_args[@]}"
}

run_verified_fallback_stage() {
  local predictor="$1"
  local args=(run-verified-tm-fallbacks --mode tmalphafold_first --all --fallback-method "$predictor")
  case "$predictor" in
    TMbed)
      if [[ "$(resolve_gpu_flag)" == "true" ]]; then
        args+=(--use-gpu)
      else
        args+=(--no-gpu)
      fi
      args+=(--batch-size "$TMBED_BATCH_SIZE" --max-workers "$TMBED_MAX_WORKERS")
      ;;
  esac
  log "Running verified fallback stage for $predictor ..."
  flask_exec "${args[@]}"
}

write_release_env() {
  local output_path="$1"
  local postgres_image_ref="$2"
  cat >"$output_path" <<EOF
REGISTRY_NAMESPACE=${REGISTRY_NAMESPACE}
IMAGE_TAG=${IMAGE_TAG}
POSTGRES_IMAGE=${postgres_image_ref}
FLASK_APP_IMAGE=${REGISTRY_NAMESPACE}/${ML_IMAGE_NAME}:${IMAGE_TAG}
BACKEND_APP_IMAGE=${REGISTRY_NAMESPACE}/${APP_IMAGE_NAME}:${IMAGE_TAG}
BACKEND_ML_IMAGE=${REGISTRY_NAMESPACE}/${ML_IMAGE_NAME}:${IMAGE_TAG}
SNAPSHOT_DIR=${SNAPSHOT_DIR}
SNAPSHOT_ARCHIVE=${SNAPSHOT_ARCHIVE}
EOF
}

main() {
  require_command bash
  require_command docker
  require_command python3

  prepare_runtime_env_file
  trap '[[ -n "$TEMP_ENV_FILE" && -f "$TEMP_ENV_FILE" ]] && rm -f "$TEMP_ENV_FILE"' EXIT
  load_env_file
  export TM_PREDICTION_USE_GPU="$(resolve_gpu_flag)"
  export TMALPHAFOLD_TMBED_USE_GPU="$TM_PREDICTION_USE_GPU"
  export TM_PREDICTION_BATCH_SIZE="$TMBED_BATCH_SIZE"
  export TM_PREDICTION_MAX_WORKERS="$TMBED_MAX_WORKERS"

  default_snapshot_paths

  log "Release namespace: ${REGISTRY_NAMESPACE}"
  log "Release image tag: ${IMAGE_TAG}"
  log "TMbed GPU enabled: ${TM_PREDICTION_USE_GPU}"

  run_bootstrap

  log "Ensuring schema is in sync before fallback stages..."
  flask_exec sync-protein-schema

  run_verified_fallback_stage DeepTMHMM
  run_verified_fallback_stage TMHMM
  if [[ "$RUN_TMDET" -eq 1 ]]; then
    run_verified_fallback_stage TMDET
  fi
  if [[ "$RUN_TMBED" -eq 1 ]]; then
    run_verified_fallback_stage TMbed
  fi

  if [[ "$VALIDATE_AFTER_RUN" -eq 1 ]]; then
    log "Running dashboard regression validation..."
    flask_exec validate-dashboard-regressions
  fi

  log "Exporting runtime snapshot to $SNAPSHOT_DIR ..."
  "$ROOT_DIR/scripts/metamp-snapshot.sh" export --env-file "$ENV_FILE" --snapshot-dir "$SNAPSHOT_DIR" --top-models "$TOP_MODELS"

  log "Archiving runtime snapshot to $SNAPSHOT_ARCHIVE ..."
  archive_snapshot_zip "$SNAPSHOT_DIR" "$SNAPSHOT_ARCHIVE" >/dev/null

  if [[ "$PUSH_BACKEND_IMAGES" -eq 1 ]]; then
    log "Building and pushing backend-only application images ..."
    image_args=(push --env-file "$ENV_FILE" --skip-frontend)
    if [[ "$NO_CACHE" -eq 1 ]]; then
      image_args+=(--no-cache)
    fi
    "$ROOT_DIR/scripts/metamp-stack-images.sh" "${image_args[@]}"
  fi

  local postgres_image_ref="postgres:latest"
  if [[ "$PUSH_STATEFUL_POSTGRES" -eq 1 ]]; then
    postgres_image_ref="${REGISTRY_NAMESPACE}/${POSTGRES_STATE_IMAGE_NAME:-metamp_postgres_state}:${IMAGE_TAG}"
    log "Building and pushing stateful PostgreSQL image $postgres_image_ref ..."
    "$ROOT_DIR/scripts/metamp-stateful-postgres-image.sh" \
      push \
      --env-file "$ENV_FILE" \
      --snapshot-dir "$SNAPSHOT_DIR" \
      --image-ref "$postgres_image_ref"
  fi

  local manifest_path="$SNAPSHOT_DIR/release-backend.env"
  write_release_env "$manifest_path" "$postgres_image_ref"

  log "MetaMP HPC release flow completed."
  log "Snapshot directory: $SNAPSHOT_DIR"
  log "Snapshot zip: $SNAPSHOT_ARCHIVE"
  log "Backend release env: $manifest_path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --snapshot-dir)
      SNAPSHOT_DIR="$2"
      shift 2
      ;;
    --snapshot-archive)
      SNAPSHOT_ARCHIVE="$2"
      shift 2
      ;;
    --top-models)
      TOP_MODELS="$2"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG_OVERRIDE="$2"
      shift 2
      ;;
    --registry-namespace)
      REGISTRY_NAMESPACE_OVERRIDE="$2"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --skip-image-push)
      PUSH_BACKEND_IMAGES=0
      shift
      ;;
    --skip-stateful-postgres)
      PUSH_STATEFUL_POSTGRES=0
      shift
      ;;
    --skip-tmbed)
      RUN_TMBED=0
      shift
      ;;
    --skip-tmdet)
      RUN_TMDET=0
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
    --wait-timeout)
      WAIT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --skip-validate)
      VALIDATE_AFTER_RUN=0
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

main
