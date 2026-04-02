#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.docker.deployment"

DOCKER_BIN="${DOCKER_BIN:-}"
ENV_FILE="$DEFAULT_ENV_FILE"
SNAPSHOT_DIR=""
DUMP_PATH=""
IMAGE_REF=""
NO_CACHE=0

log() {
  printf '[MetaMP Postgres Image] %s\n' "$*"
}

die() {
  printf '[MetaMP Postgres Image][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/metamp-stateful-postgres-image.sh build [options]
  ./scripts/metamp-stateful-postgres-image.sh push [options]

Commands:
  build     Build a PostgreSQL image that restores the exported MetaMP dump on first container init.
  push      Build and push that PostgreSQL image.

Options:
  --env-file PATH      Use a different env file. Default: .env.docker.deployment
  --snapshot-dir PATH  Snapshot directory containing initdb/all_tables.dump
  --dump-path PATH     Explicit dump path. Overrides --snapshot-dir when both are provided.
  --image-ref REF      Full image ref override. Default: <REGISTRY_NAMESPACE>/<POSTGRES_STATE_IMAGE_NAME>:<IMAGE_TAG>
  --no-cache           Build without Docker cache.
  -h, --help           Show this help.
EOF
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

load_env_file() {
  [[ -f "$ENV_FILE" ]] || die "Env file not found: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

resolve_dump_path() {
  if [[ -n "$DUMP_PATH" ]]; then
    [[ -f "$DUMP_PATH" ]] || die "Dump path not found: $DUMP_PATH"
    printf '%s\n' "$DUMP_PATH"
    return 0
  fi
  [[ -n "$SNAPSHOT_DIR" ]] || die "Provide --snapshot-dir or --dump-path."
  local candidate="$SNAPSHOT_DIR/initdb/all_tables.dump"
  [[ -f "$candidate" ]] || die "Snapshot dump not found: $candidate"
  printf '%s\n' "$candidate"
}

resolve_image_ref() {
  if [[ -n "$IMAGE_REF" ]]; then
    printf '%s\n' "$IMAGE_REF"
    return 0
  fi
  printf '%s/%s:%s\n' \
    "${REGISTRY_NAMESPACE:-ebenco36}" \
    "${POSTGRES_STATE_IMAGE_NAME:-metamp_postgres_state}" \
    "${IMAGE_TAG:-latest}"
}

build_context() {
  local dump_source="$1"
  local tmp_dir
  tmp_dir="$(mktemp -d)"

  cat >"$tmp_dir/Dockerfile" <<EOF
FROM ${POSTGRES_BASE_IMAGE:-postgres:latest}
COPY all_tables.dump /docker-entrypoint-initdb.d/010-metamp.dump
COPY restore_dump.sh /docker-entrypoint-initdb.d/020-restore_dump.sh
RUN chmod 0644 /docker-entrypoint-initdb.d/010-metamp.dump && \\
    chmod 0755 /docker-entrypoint-initdb.d/020-restore_dump.sh
EOF

  cat >"$tmp_dir/restore_dump.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

export PGPASSWORD="${POSTGRES_PASSWORD:-postgres}"

if [[ ! -f /docker-entrypoint-initdb.d/010-metamp.dump ]]; then
  echo "[MetaMP Postgres Init] Dump file is missing; nothing to restore." >&2
  exit 0
fi

echo "[MetaMP Postgres Init] Restoring MetaMP dump into ${POSTGRES_DB:-postgres} ..."
pg_restore \
  --clean \
  --if-exists \
  --no-owner \
  -U "${POSTGRES_USER:-postgres}" \
  -d "${POSTGRES_DB:-postgres}" \
  /docker-entrypoint-initdb.d/010-metamp.dump

echo "[MetaMP Postgres Init] Restore complete."
EOF

  cp "$dump_source" "$tmp_dir/all_tables.dump"
  printf '%s\n' "$tmp_dir"
}

build_image() {
  resolve_docker_bin
  load_env_file
  local dump_source image_ref build_dir
  dump_source="$(resolve_dump_path)"
  image_ref="$(resolve_image_ref)"
  build_dir="$(build_context "$dump_source")"
  trap "rm -rf '$build_dir'" EXIT

  local build_args=(build -t "$image_ref")
  if [[ "$NO_CACHE" -eq 1 ]]; then
    build_args+=(--no-cache)
  fi
  build_args+=("$build_dir")

  log "Building stateful PostgreSQL image $image_ref" >&2
  "$DOCKER_BIN" "${build_args[@]}" >&2
  printf '%s\n' "$image_ref"
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
    --snapshot-dir)
      SNAPSHOT_DIR="$2"
      shift 2
      ;;
    --dump-path)
      DUMP_PATH="$2"
      shift 2
      ;;
    --image-ref)
      IMAGE_REF="$2"
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

case "$COMMAND" in
  build)
    build_image >/dev/null
    ;;
  push)
    resolve_docker_bin
    image_ref="$(build_image)"
    log "Pushing stateful PostgreSQL image $image_ref"
    "$DOCKER_BIN" push "$image_ref"
    ;;
  *)
    die "Unknown command: $COMMAND"
    ;;
esac
