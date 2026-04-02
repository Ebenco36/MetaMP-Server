#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NATIVE_TMBED_SCRIPT="$ROOT_DIR/scripts/metamp-native-tmbed.sh"

DOCKER_BIN="${DOCKER_BIN:-}"
APP_CONTAINER_NAME="${APP_CONTAINER_NAME:-testmpvis_app}"
FALLBACK_MODE="${FALLBACK_MODE:-fallback_only}"
RUN_ALL=0
LIMIT=""
INCLUDE_COMPLETED=0
RETRY_ERRORS=0
DOCKER_BATCH_SIZE="${DOCKER_BATCH_SIZE:-25}"
DOCKER_MAX_WORKERS="${DOCKER_MAX_WORKERS:-1}"
NATIVE_DEVICE="${NATIVE_DEVICE:-auto}"
WITH_DEEPTMHMM=1
WITH_TMHMM=1
WITH_TMDET=1
WITH_TMBED=1
PDB_CODES=()
TMALPHAFOLD_METHODS="${TMALPHAFOLD_METHODS:-}"
WITH_TMALPHAFOLD_TMDET=1
TMALPHAFOLD_MAX_WORKERS="${TMALPHAFOLD_MAX_WORKERS:-8}"
TMALPHAFOLD_TIMEOUT="${TMALPHAFOLD_TIMEOUT:-30}"
TMALPHAFOLD_REFRESH=0
TMALPHAFOLD_RETRY_ERRORS=0
TMALPHAFOLD_BACKFILL_SEQUENCES=1

log() {
  printf '[MetaMP Hybrid Fallbacks] %s\n' "$*"
}

die() {
  printf '[MetaMP Hybrid Fallbacks][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/metamp-hybrid-verified-fallbacks.sh [options]

Run the verified local fallback pipeline through one command:
- DeepTMHMM, TMHMM, and TMDET run inside Docker
- TMbed runs natively on the host for macOS MPS / Linux CUDA / CPU

Options:
  --mode MODE              One of: fallback_only, tmalphafold_first. Default: fallback_only
  --all                    Run across all eligible targets.
  --limit N                Restrict the run to the first N targets.
  --pdb-code CODE          Restrict the run to one or more PDB codes. Repeat as needed.
  --include-completed      Rerun completed rows too.
  --retry-errors          Retry stored MetaMP fallback error rows without forcing successful rows to rerun.
  --native-device MODE     TMbed native device mode: auto, gpu, cpu. Default: auto
  --docker-batch-size N    Batch size for Docker fallback stage. Default: 25
  --docker-max-workers N   Max workers for Docker fallback stage. Default: 1
  --skip-deeptmhmm         Skip DeepTMHMM.
  --skip-tmhmm             Skip TMHMM.
  --skip-tmdet             Skip TMDET.
  --skip-tmbed             Skip TMbed.
  --methods CSV            Optional comma-separated TMAlphaFold method list for tmalphafold_first mode.
  --without-tmdet          Exclude TMDET from the TMAlphaFold stage.
  --tmalphafold-max-workers N
                          TMAlphaFold concurrent request count. Default: 8
  --tmalphafold-timeout N  TMAlphaFold request timeout in seconds. Default: 30
  --tmalphafold-refresh    Refetch TMAlphaFold rows even when successful upstream rows already exist.
  --tmalphafold-retry-errors
                          Retry stored TMAlphaFold error rows too.
  --skip-backfill-sequences
                          Disable TMAlphaFold sequence backfill.
  -h, --help               Show this help.

Examples:
  bash scripts/metamp-hybrid-verified-fallbacks.sh --all
  bash scripts/metamp-hybrid-verified-fallbacks.sh --pdb-code 6N7G
  bash scripts/metamp-hybrid-verified-fallbacks.sh --pdb-code 6N7G --include-completed
  bash scripts/metamp-hybrid-verified-fallbacks.sh --all --skip-tmbed
EOF
}

normalize_pdb_code() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
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
  die "Docker CLI not found. Set DOCKER_BIN explicitly if needed."
}

require_scope() {
  if [[ "$RUN_ALL" -eq 0 && -z "$LIMIT" && "${#PDB_CODES[@]}" -eq 0 ]]; then
    die "Provide --all, --limit, or at least one --pdb-code so the fallback scope is explicit."
  fi
}

require_inputs() {
  [[ -x "$NATIVE_TMBED_SCRIPT" ]] || die "Native TMbed runner not found or not executable: $NATIVE_TMBED_SCRIPT"
}

build_python_fallback_scope_snippet() {
  cat <<'PY'
import json
from app import app
from src.Jobs.LoadProteinPredictions import determine_tmalphafold_fallback_targets

methods = json.loads(__METHODS_JSON__)
pdb_codes = json.loads(__PDB_CODES_JSON__)
limit = __LIMIT__
with_tmdet = __WITH_TMDET__

with app.app_context():
    summary = determine_tmalphafold_fallback_targets(
        methods=methods,
        with_tmdet=with_tmdet,
        pdb_codes=pdb_codes or None,
        limit=limit,
        progress_callback=None,
    )

print(json.dumps(summary.get("pdb_codes") or []))
PY
}

get_tmalphafold_fallback_codes_file() {
  local output_path="$1"
  local methods_json pdb_codes_json limit_literal with_tmdet_literal script_body

  if [[ -n "$TMALPHAFOLD_METHODS" ]]; then
    methods_json="$(python3 - <<PY
import json
print(json.dumps([item.strip() for item in "${TMALPHAFOLD_METHODS}".split(",") if item.strip()]))
PY
)"
  else
    methods_json="null"
  fi

  if [[ "${#PDB_CODES[@]}" -gt 0 ]]; then
    pdb_codes_json="$(python3 - "${PDB_CODES[@]}" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
)"
  else
    pdb_codes_json="null"
  fi

  if [[ -n "$LIMIT" ]]; then
    limit_literal="$LIMIT"
  else
    limit_literal="None"
  fi

  if [[ "$WITH_TMALPHAFOLD_TMDET" -eq 1 ]]; then
    with_tmdet_literal="True"
  else
    with_tmdet_literal="False"
  fi

  script_body="$(build_python_fallback_scope_snippet)"
  script_body="${script_body/__METHODS_JSON__/$methods_json}"
  script_body="${script_body/__PDB_CODES_JSON__/$pdb_codes_json}"
  script_body="${script_body/__LIMIT__/$limit_literal}"
  script_body="${script_body/__WITH_TMDET__/$with_tmdet_literal}"

  local result_json
  result_json="$("$DOCKER_BIN" exec "$APP_CONTAINER_NAME" python -c "$script_body")"

  python3 - "$output_path" "$result_json" <<'PY'
import json
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
payload = json.loads(sys.argv[2])
output_path.write_text("\n".join(str(item).strip().upper() for item in payload if str(item).strip()) + "\n")
PY
}

docker_args_for_scope() {
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

  if [[ "$INCLUDE_COMPLETED" -eq 1 ]]; then
    _args_ref+=(--include-completed)
  fi
  if [[ "$RETRY_ERRORS" -eq 1 ]]; then
    _args_ref+=(--retry-errors)
  fi
}

native_args_for_scope() {
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

  if [[ "$INCLUDE_COMPLETED" -eq 1 ]]; then
    _args_ref+=(--include-completed)
  fi
  if [[ "$RETRY_ERRORS" -eq 1 ]]; then
    _args_ref+=(--retry-errors)
  fi
}

run_with_optional_caffeinate() {
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i "$@"
  else
    "$@"
  fi
}

run_docker_stage() {
  local methods=()
  [[ "$WITH_DEEPTMHMM" -eq 1 ]] && methods+=(DeepTMHMM)
  [[ "$WITH_TMHMM" -eq 1 ]] && methods+=(TMHMM)
  [[ "$WITH_TMDET" -eq 1 ]] && methods+=(TMDET)

  if [[ "${#methods[@]}" -eq 0 ]]; then
    log "Skipping Docker fallback stage."
    return 0
  fi

  resolve_docker_bin

  local args=(
    exec
    "$APP_CONTAINER_NAME"
    env
    FLASK_APP=manage.py
    flask
    run-verified-tm-fallbacks
    --mode
    "$FALLBACK_MODE"
    --batch-size
    "$DOCKER_BATCH_SIZE"
    --max-workers
    "$DOCKER_MAX_WORKERS"
  )

  local method
  for method in "${methods[@]}"; do
    args+=(--fallback-method "$method")
  done

  docker_args_for_scope args

  if [[ "$FALLBACK_MODE" == "tmalphafold_first" ]]; then
    if [[ -n "$TMALPHAFOLD_METHODS" ]]; then
      args+=(--methods "$TMALPHAFOLD_METHODS")
    fi
    if [[ "$WITH_TMALPHAFOLD_TMDET" -eq 1 ]]; then
      args+=(--with-tmdet)
    else
      args+=(--without-tmdet)
    fi
    args+=(--tmalphafold-max-workers "$TMALPHAFOLD_MAX_WORKERS")
    args+=(--tmalphafold-timeout "$TMALPHAFOLD_TIMEOUT")
    if [[ "$TMALPHAFOLD_REFRESH" -eq 1 ]]; then
      args+=(--tmalphafold-refresh)
    fi
    if [[ "$TMALPHAFOLD_RETRY_ERRORS" -eq 1 ]]; then
      args+=(--tmalphafold-retry-errors)
    else
      args+=(--skip-tmalphafold-errors)
    fi
    if [[ "$TMALPHAFOLD_BACKFILL_SEQUENCES" -eq 1 ]]; then
      args+=(--backfill-sequences)
    else
      args+=(--skip-backfill-sequences)
    fi
  fi

  log "Running Docker fallback stage for: ${methods[*]}"
  run_with_optional_caffeinate "$DOCKER_BIN" "${args[@]}"
}

run_native_tmbed_stage() {
  if [[ "$WITH_TMBED" -ne 1 ]]; then
    log "Skipping native TMbed stage."
    return 0
  fi

  local args=(
    "$NATIVE_TMBED_SCRIPT"
    fallback
    --device
    "$NATIVE_DEVICE"
  )

  local fallback_scope_file=""
  if [[ "$FALLBACK_MODE" == "tmalphafold_first" ]]; then
    fallback_scope_file="$(mktemp "$ROOT_DIR/.metamp-tmalphafold-fallback-codes.XXXXXX")"
    get_tmalphafold_fallback_codes_file "$fallback_scope_file"
    if [[ ! -s "$fallback_scope_file" ]]; then
      log "TMAlphaFold fallback scope is empty for TMbed; skipping native TMbed stage."
      rm -f "$fallback_scope_file"
      return 0
    fi
    args+=(--pdb-code-file "$fallback_scope_file")
    if [[ "$INCLUDE_COMPLETED" -eq 1 ]]; then
      args+=(--include-completed)
    fi
  else
    native_args_for_scope args
  fi

  log "Running native TMbed stage ..."
  run_with_optional_caffeinate bash "${args[@]}"
  if [[ -n "$fallback_scope_file" ]]; then
    rm -f "$fallback_scope_file"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      FALLBACK_MODE="$2"
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
    --include-completed)
      INCLUDE_COMPLETED=1
      shift
      ;;
    --retry-errors)
      RETRY_ERRORS=1
      shift
      ;;
    --native-device)
      NATIVE_DEVICE="$2"
      shift 2
      ;;
    --docker-batch-size)
      DOCKER_BATCH_SIZE="$2"
      shift 2
      ;;
    --docker-max-workers)
      DOCKER_MAX_WORKERS="$2"
      shift 2
      ;;
    --skip-deeptmhmm)
      WITH_DEEPTMHMM=0
      shift
      ;;
    --skip-tmhmm)
      WITH_TMHMM=0
      shift
      ;;
    --skip-tmdet)
      WITH_TMDET=0
      shift
      ;;
    --skip-tmbed)
      WITH_TMBED=0
      shift
      ;;
    --methods)
      TMALPHAFOLD_METHODS="$2"
      shift 2
      ;;
    --with-tmdet)
      WITH_TMALPHAFOLD_TMDET=1
      shift
      ;;
    --without-tmdet)
      WITH_TMALPHAFOLD_TMDET=0
      shift
      ;;
    --tmalphafold-max-workers)
      TMALPHAFOLD_MAX_WORKERS="$2"
      shift 2
      ;;
    --tmalphafold-timeout)
      TMALPHAFOLD_TIMEOUT="$2"
      shift 2
      ;;
    --tmalphafold-refresh)
      TMALPHAFOLD_REFRESH=1
      shift
      ;;
    --tmalphafold-retry-errors)
      TMALPHAFOLD_RETRY_ERRORS=1
      shift
      ;;
    --skip-backfill-sequences)
      TMALPHAFOLD_BACKFILL_SEQUENCES=0
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

cd "$ROOT_DIR"
require_inputs
require_scope

case "$FALLBACK_MODE" in
  fallback_only|tmalphafold_first)
    ;;
  *)
    die "Unsupported --mode value: $FALLBACK_MODE"
    ;;
esac

log "Mode: $FALLBACK_MODE"
log "Scope: all=$RUN_ALL limit=${LIMIT:-<none>} pdb_codes=${#PDB_CODES[@]}"
log "Docker stage: DeepTMHMM=$WITH_DEEPTMHMM TMHMM=$WITH_TMHMM TMDET=$WITH_TMDET batch_size=$DOCKER_BATCH_SIZE max_workers=$DOCKER_MAX_WORKERS"
log "Native TMbed stage: enabled=$WITH_TMBED device=$NATIVE_DEVICE"

run_docker_stage
run_native_tmbed_stage

log "Hybrid verified fallback pipeline completed."
