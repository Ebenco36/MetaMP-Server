# MetaMP Command Workflow

This guide explains the command-line workflow for MetaMP from a clean machine to a fully bootstrapped application.

It has two goals:
- give operators one low-stress command that brings up the stack
- explain the manual steps underneath that command for debugging and maintenance

## Choose One Workflow

MetaMP now supports two operational paths:

- `Load from snapshot` for most users. This is the fastest and lightest route. It restores a prepared database dump, curated dataset snapshots, and a trimmed production ML bundle set without rerunning full ingestion.
- `Build from scratch` for maintainers. This reruns ingestion, database sync, ML, TM prediction, and validation from source workflows.

If you already have a prepared MetaMP runtime snapshot, prefer the snapshot workflow below.

## Fastest Path: Load From Snapshot

Export a reusable snapshot from a prepared MetaMP instance:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-snapshot.sh export --top-models 1
```

What this exports:
- a PostgreSQL dump for fast restore
- only the runtime-required dataset snapshots, not all historical ingestion files
- production ML manifests, figures, tables, and only the top retained model bundles
- the current live group-prediction artifact and compact ML publication outputs

Load a prepared snapshot on another machine:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-snapshot.sh load --snapshot-dir /path/to/metamp-snapshot-YYYYMMDDTHHMMSSZ
```

Load a prepared snapshot with background jobs still disabled by default, but with the standard frontend service:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-snapshot.sh load --snapshot-dir /path/to/metamp-snapshot-YYYYMMDDTHHMMSSZ --with-frontend
```

Opt in to Celery workers and Celery beat only when you explicitly want them:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-snapshot.sh load --snapshot-dir /path/to/metamp-snapshot-YYYYMMDDTHHMMSSZ --with-frontend --with-background-jobs
```

Load and start with the local frontend overlay:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-snapshot.sh load --snapshot-dir /path/to/metamp-snapshot-YYYYMMDDTHHMMSSZ --with-local-frontend
```

Operational notes:
- snapshot export keeps only the top `N` production ML bundles, default `1`, to reduce space while still preserving reviewer-facing ML figures and tables
- snapshot load pulls the configured runtime images, restores the database dump directly, and copies the retained model/runtime assets into the live container volume
- snapshot load now starts a lightweight runtime by default: `postgres`, `redis`, `flask-app`, and optional `frontend`
- Celery workers and `celery-beat` stay off unless `--with-background-jobs` is supplied explicitly
- this path is intended for users who want a working MetaMP deployment without rerunning the full curation pipeline
- this is the recommended publication/demo workflow
- prebuilt Docker images do not contain your live PostgreSQL volume by default; the snapshot is what carries the prepared database state, curated datasets, and trimmed ML/runtime assets

For external transmembrane predictors such as `Phobius`, `TOPCONS`, and `CCTOP`, see:

- [TM_PREDICTOR_EXTERNAL_WORKFLOW.md](TM_PREDICTOR_EXTERNAL_WORKFLOW.md)

That runbook now includes:

- queued production jobs
- status commands
- default production file paths
- direct CLI fallback commands

## Verified Local Fallbacks

MetaMP currently has four locally verified fallback predictors that can be used when `TMAlphaFold` is missing coverage or when you want to bypass `TMAlphaFold` entirely:

- `TMbed`
- `DeepTMHMM`
- `TMHMM`
- `TMDET`

These are orchestrated through one resumable command:

```bash
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode tmalphafold_first --pdb-code 1PTH
```

What this does:
- checks `TMAlphaFold` for the selected scope first
- identifies proteins still missing at least one upstream `TMAlphaFold` method
- runs only the verified local fallback methods for those proteins
- skips already completed `MetaMP` fallback rows so reruns continue from where they stopped

Run the verified local fallbacks directly, without going through `TMAlphaFold`:

```bash
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode fallback_only --pdb-code 1PTH
```

Run across all eligible proteins:

```bash
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode fallback_only --all
```

Run across all eligible proteins, but restrict bulk execution to the safer subset:

```bash
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode fallback_only --all --bulk-safe-only
```

Force recomputation even when `MetaMP` already stores successful fallback rows:

```bash
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode fallback_only --pdb-code 1PTH --include-completed
```

Useful options:

```bash
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode tmalphafold_first --all
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode tmalphafold_first --limit 100
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode fallback_only --fallback-method TMHMM --fallback-method TMDET --pdb-code 1PTH
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode fallback_only --no-gpu --batch-size 8 --max-workers 1 --all
docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode fallback_only --all --all-verified-methods
```

Notes:
- targeted runs default to `TMbed`, `DeepTMHMM`, `TMHMM`, and `TMDET`
- bulk `--all` now defaults to the full verified local fallback set and stages it method-by-method against each method's actual missing `MetaMP` coverage
- use `--bulk-safe-only` to restrict bulk runs to `DeepTMHMM` and `TMHMM`
- explicit `--fallback-method` values always override the default bulk selector
- resume behavior is the default; `--include-completed` disables it
- `TMbed` and `DeepTMHMM` run in the local ML runtime
- `TMHMM` and `TMDET` run through the verified MetaMP wrappers
- recorded `MetaMP` error rows also count as completed for resumable exports, so impossible cases such as oversized `TMDET` structures are skipped on later reruns unless `--include-completed` is used
- if a sequence-based fallback candidate has no usable sequence retrievable from RCSB, MetaMP records a `MetaMP` error row for that predictor so later bulk reruns do not keep surfacing the same unrunnable protein
- if a run fetches missing protein sequences first, those sequences are now persisted only into tables that actually expose the target sequence column in that environment
- `TMDET` uses a persistent runtime under `/var/app/data/tm_predictions/external/tmdet/runtime`
- by default, a cold `TMDET` cache now follows TMDET's normal documented auto-install path on the requested structure instead of silently running a dedicated warmup structure
- if you explicitly want prewarm behavior for bulk runs, set `METAMP_TMDET_ENABLE_PREWARM=1`

Useful TMDET runtime checks:

```bash
docker exec testmpvis_app sh -lc 'find /var/app/data/tm_predictions/external/tmdet/runtime/ccd -type f | wc -l'
docker exec testmpvis_app sh -lc 'test -f /var/app/data/tm_predictions/external/tmdet/runtime/.ccd_ready && cat /var/app/data/tm_predictions/external/tmdet/runtime/.ccd_ready'
```

Targeted TMDET-only verification:

```bash
caffeinate -i docker exec testmpvis_app env FLASK_APP=manage.py flask run-verified-tm-fallbacks --mode fallback_only --pdb-code 6N7G --fallback-method TMDET --batch-size 1 --include-completed
```
- if you are upgrading an older database and want `membrane_proteins.sequence_sequence` restored explicitly, run:

```bash
docker exec testmpvis_app env FLASK_APP=manage.py flask sync-protein-schema
```

No-Docker HPC path for TMbed only:

```bash
cd /path/to/MetaMP-Server
bash scripts/metamp-hpc-sqlite-tmbed.sh --rebuild-db
```

What it does:
- creates a SQLite-backed MetaMP runtime
- loads the checked-in dataset snapshots into SQLite
- runs the TMbed-only fallback pipeline
- exports `membrane_protein_tmalphafold_predictions` to CSV for later PostgreSQL import

Useful options:

```bash
bash scripts/metamp-hpc-sqlite-tmbed.sh --runtime-root /scratch/$USER/metamp-sqlite
bash scripts/metamp-hpc-sqlite-tmbed.sh --gpu-mode auto --tmbed-batch-size 1 --tmbed-max-workers 1
bash scripts/metamp-hpc-sqlite-tmbed.sh --include-completed
python3 -m flask export-tmalphafold-predictions-csv --output-path /path/to/membrane_protein_tmalphafold_predictions.csv
```

If the HPC cannot reach PyPI/GitHub, build a Linux wheelhouse first on a machine with Docker and internet access:

```bash
cd /path/to/MetaMP-Server
bash scripts/build-hpc-wheelhouse.sh --clean
```

Then copy `vendor/wheels/` to the HPC and run the SQLite TMbed job with the offline wheelhouse. If the HPC checkout does not include the local `tmbed/` source tree, also provide a Linux-compatible `tmbed-*.whl` in the wheelhouse:

```bash
sbatch scripts/metamp-hpc-sqlite-tmbed.sh \
  --rebuild-db \
  --recreate-venv \
  --runtime-root /scratch/$USER/metamp-sqlite \
  --wheelhouse-dir /scratch/$USER/metamp-wheelhouse
```

## Publish and Reuse Images

MetaMP now has a dedicated stack-image helper for publishing the custom application images used by the production stack.

Build the custom backend and frontend images:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-stack-images.sh build
```

Build and push the custom backend and frontend images to the configured registry namespace:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-stack-images.sh push
```

Push the MetaMP images and export a matching runtime snapshot in one release flow:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-release.sh publish
```

Publish a snapshot-ready distribution stack that pairs the pushed application images with a stateful PostgreSQL image built from the current snapshot:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-publish-snapshot.sh push
```

Useful options:

```bash
./scripts/metamp-stack-images.sh push --skip-frontend
./scripts/metamp-stack-images.sh push --no-cache
./scripts/metamp-stack-images.sh push --env-file .env.docker.deployment
./scripts/metamp-release.sh publish --snapshot-dir /path/to/output/metamp-snapshot-release --top-models 1
./scripts/metamp-publish-snapshot.sh push --snapshot-dir /path/to/output/metamp-snapshot-release --top-models 1
```

Important:
- this publishes the custom MetaMP application images only
- `postgres` and `redis` remain upstream images by default
- prepared runtime data should be distributed with a MetaMP snapshot, not assumed to live inside the pushed images
- the lightest reviewer workflow is therefore: pull the configured images and run `./scripts/metamp-snapshot.sh load --snapshot-dir ... --with-frontend`
- if you want a Docker-registry-friendly database snapshot as well, pair the pushed app images with `./scripts/metamp-stateful-postgres-image.sh push ...` or use `./scripts/metamp-publish-snapshot.sh push`

Build a backend-ready PostgreSQL image from an exported live MetaMP dump:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-stateful-postgres-image.sh push --snapshot-dir /path/to/metamp-snapshot-YYYYMMDDTHHMMSSZ
```

Notes:
- this creates a PostgreSQL image that restores `initdb/all_tables.dump` on first container initialization
- pair it with `POSTGRES_IMAGE=<namespace>/<POSTGRES_STATE_IMAGE_NAME>:<IMAGE_TAG>` when launching the backend stack
- this is the lowest-friction way to ship backend images together with the live database state

Run the full HPC backend release flow from a fresh checkout:

```bash
cd /path/to/MetaMP-Server
sbatch scripts/metamp-hpc-run.sh
```

Useful options:

```bash
bash scripts/metamp-hpc-run.sh --image-tag 2026-03-31-hpc
bash scripts/metamp-hpc-run.sh --snapshot-dir /scratch/$USER/metamp-release --snapshot-archive /scratch/$USER/metamp-release.zip
bash scripts/metamp-hpc-run.sh --skip-image-push --skip-stateful-postgres
bash scripts/metamp-hpc-run.sh --skip-tmbed
bash scripts/metamp-hpc-run.sh --gpu-mode auto --tmbed-batch-size 1 --tmbed-max-workers 1
```

What the HPC script does:
- runs the full production bootstrap
- syncs the schema explicitly
- stages verified local fallbacks in order: `DeepTMHMM`, `TMHMM`, `TMDET`, `TMbed`
- exports the runtime snapshot and archives it as a `.zip`
- pushes backend-only application images
- builds and pushes a stateful PostgreSQL image from the live dump
- writes `release-backend.env` into the snapshot directory with the exact backend image refs to reuse

## Build From Scratch

For maintainers or refresh jobs, use the production bootstrap script.

Backend only:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-production-bootstrap.sh run
```

Backend plus local frontend:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-production-bootstrap.sh run --with-local-frontend
```

Backend plus frontend service from the normal compose setup:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-production-bootstrap.sh run --with-frontend
```

Build the registry frontend image from the local `MPVisualization` workspace:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-frontend-image.sh build
```

Build and push the frontend image to the configured registry namespace:

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-frontend-image.sh push
```

What this command does:
- builds the required Docker images
- refreshes the standard frontend image when `--with-frontend` is used
- starts `postgres`, `redis`, `flask-app`, `celery-worker`, `celery-worker-ml`, and `celery-beat`
- optionally starts `frontend`
- waits for the backend readiness endpoint to respond
- seeds the runtime dataset volume from local dataset snapshots
- runs the full ingestion and database sync
- queues the machine-learning workflow
- queues TM prediction backfill
- exports the discrepancy benchmark
- runs the regression validator
- writes a bootstrap marker so later starts do not repeat the expensive full bootstrap unless asked

## Step-By-Step Workflow

This section explains the same flow one layer at a time.

### 1. Start the Stack

```bash
docker compose --env-file .env.docker.deployment up -d postgres redis flask-app celery-worker celery-worker-ml celery-beat
```

What this does:
- starts the database
- starts Redis for Celery and caching
- starts the Flask API
- starts the general worker, ML worker, and scheduler

If you also want the standard frontend service:

```bash
docker compose --env-file .env.docker.deployment up -d frontend
```

If you want the local frontend overlay:

```bash
docker compose --env-file .env.docker.deployment -f docker-compose.yml -f docker-compose.local-frontend.yml up -d frontend
```

### 2. Wait for the API

```bash
curl --fail http://127.0.0.1:5400/api/v1/health/ready
```

What this does:
- confirms the API container is up
- confirms the application is ready to accept CLI and HTTP work

### 3. Refresh Datasets From Sources

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask refresh-protein-datasets
```

What this does:
- fetches and normalizes MPStruc, PDB, OPM, and UniProt source data
- writes processed artifacts to the dataset area
- runs the configured ingestion stages

Important:
- this updates local artifacts only
- it does not load them into PostgreSQL yet

If you are restoring from a prepared snapshot instead of rebuilding from source, skip this step and use the snapshot workflow at the top of this file.

### 4. Sync the Database

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask sync-protein-database
```

What this does:
- refreshes ingestion artifacts
- synchronizes schema expectations
- loads the curated dataset into PostgreSQL
- seeds default records when needed

This is the main command that makes the application data usable.

If you already restored `all_tables.dump` through the snapshot workflow, do not rerun this unless you intentionally want to overwrite the restored state with the local CSV datasets.

### 5. Queue the Machine-Learning Pipeline

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask queue-machine-learning-job
```

What this does:
- queues the ML workflow on `celery-worker-ml`
- trains the structural-group classification pipeline
- produces ML outputs used by MetaMP services
- runs time-split evaluation outputs for stronger validation artifacts

How to check whether the job is queued or running:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask celery-list-jobs
docker compose --env-file .env.docker.deployment logs -f celery-worker-ml
```

Typical workflow:
- queue the job and copy the returned task id
- run `celery-list-jobs` to see whether it is in `reserved`, `scheduled`, or `active`
- follow `celery-worker-ml` logs to watch training start, progress, and completion

Inspect one specific task:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask celery-task-status --task-id <task-id>
```

### 6. Queue TM Prediction Backfill

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask backfill-tm-predictions
```

What this does:
- queues TM prediction for sequences in the curated dataset
- runs TMbed
- runs DeepTMHMM too when configured and available
- stores TM counts and TM boundaries for later display and reconciliation
- mirrors each completed TMbed batch into the normalized prediction store immediately
- lets interrupted reruns continue without losing already completed TMbed batches

Useful variants:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask backfill-tm-predictions --tmbed-only
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask backfill-tm-predictions --batch-size 10 --max-workers 1
```

How to check whether the TM job is queued or running:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask celery-list-jobs
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask tm-prediction-status
docker compose --env-file .env.docker.deployment logs -f celery-worker
docker compose --env-file .env.docker.deployment logs -f celery-worker-tm
```

Typical workflow:
- queue the job and copy the returned task id
- run `tm-prediction-status` for the latest TM pipeline summary
- use `celery-list-jobs` to see whether it is waiting or active
- follow `celery-worker` and `celery-worker-tm` logs for batch execution details

Direct local TMbed rerun with GPU:

```bash
cd /path/to/MetaMP-Server
PYTHONPATH="$PWD" ./.mpvis/bin/flask --app manage.py sync-tmbed-predictions --all --use-gpu
```

Use this when:
- you are running TMbed outside Docker on the local machine
- you want resumable reruns after an interrupted long TMbed session

Native host TMbed runner for macOS MPS, Linux CUDA, or CPU:

```bash
cd /path/to/MetaMP-Server
bash scripts/metamp-native-tmbed.sh doctor
bash scripts/metamp-native-tmbed.sh sync --all --device auto
bash scripts/metamp-native-tmbed.sh fallback --all --device auto --batch-size 10
```

Device notes:
- `--device auto` prefers CUDA on Linux, MPS on Apple Silicon, then CPU
- `--device gpu` forces MetaMP to request GPU-capable TMbed execution and still lets TMbed fall back to CPU if needed
- `--device cpu` disables GPU use explicitly

Examples:

```bash
bash scripts/metamp-native-tmbed.sh sync --pdb-code 6N7G --device auto --include-completed
bash scripts/metamp-native-tmbed.sh fallback --all --device gpu --batch-size 10 --max-workers 1
bash scripts/metamp-native-tmbed.sh sync --all --device cpu
```

Single hybrid command for all verified fallbacks:

```bash
bash scripts/metamp-hybrid-verified-fallbacks.sh --all
```

What it does:
- runs `DeepTMHMM`, `TMHMM`, and `TMDET` inside Docker
- runs `TMbed` natively on the host
- keeps the same scope across both stages
- supports resumable reruns with `--include-completed`

Examples:

```bash
bash scripts/metamp-hybrid-verified-fallbacks.sh --all
bash scripts/metamp-hybrid-verified-fallbacks.sh --pdb-code 6N7G
bash scripts/metamp-hybrid-verified-fallbacks.sh --pdb-code 6N7G --include-completed
bash scripts/metamp-hybrid-verified-fallbacks.sh --all --skip-tmbed
bash scripts/metamp-hybrid-verified-fallbacks.sh --all --native-device auto --docker-batch-size 25
```

Single hybrid command with TMAlphaFold first, then local fallbacks:

```bash
bash scripts/metamp-hybrid-verified-fallbacks.sh --mode tmalphafold_first --all
```

Examples:

```bash
bash scripts/metamp-hybrid-verified-fallbacks.sh --mode tmalphafold_first --all
bash scripts/metamp-hybrid-verified-fallbacks.sh --mode tmalphafold_first --pdb-code 6N7G
bash scripts/metamp-hybrid-verified-fallbacks.sh --mode tmalphafold_first --pdb-code 6N7G --include-completed
bash scripts/metamp-hybrid-verified-fallbacks.sh --mode tmalphafold_first --all --without-tmdet
```

### 6b. Run All TMAlphaFold Methods and Persist Them

Direct sync across all eligible UniProt-backed records:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask sync-tmalphafold-predictions --all --with-tmdet --with-tmbed --tmbed-refresh
```

What this does:
- fetches all configured TMAlphaFold sequence methods
- also includes TMDET by default for structure-based membrane-plane predictions
- stores normalized TMAlphaFold outputs in the MetaMP database
- appends a TMbed rerun and mirrors TMbed rows into the normalized prediction store
- clears record payload caches after completion so the frontend sees the latest values

Queue the same full production sync in the background:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask queue-tmalphafold-sync --with-tmdet --with-tmbed --tmbed-refresh
```

Check the latest TMAlphaFold/TM annotation task status:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask tm-prediction-status
```

Notes:
- the default TMAlphaFold method list already includes all configured sequence and auxiliary TMAlphaFold methods
- use `--methods method1,method2,...` only when you intentionally want a subset
- use the queued command for production runs; use the direct command when you want foreground logs and an immediate JSON summary

How to monitor the queued TMAlphaFold workflow:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask celery-list-jobs
docker compose --env-file .env.docker.deployment logs -f celery-worker-tm
docker compose --env-file .env.docker.deployment logs -f flask-app
```

Typical workflow:
- queue the job and copy the returned task id
- use `celery-list-jobs` to see whether it is `reserved`, `scheduled`, or `active`
- follow `celery-worker-tm` for the task execution logs
- follow `flask-app` if you also want API-side queue submission and export/status request logs

### 7. Export Benchmark Artifacts

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask export-discrepancy-benchmark
```

What this does:
- exports the expert-support discrepancy benchmark
- captures source labels, MetaMP normalized labels, predictor outputs, and review metadata
- writes reproducible release-grade benchmark artifacts

For the high-confidence subset:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask export-high-confidence-subset --format json
```

What this does:
- exports the subset of entries that pass stronger confidence criteria
- gives a cleaner set for downstream analysis or demonstrations

### 7b. Work with the Discrepancy Review Queue

The discrepancy review queue is now server-side paginated and searchable, so the frontend no longer loads the full queue at once.

Fetch the first review page:

```bash
curl "http://127.0.0.1:5400/api/v1/discrepancy-reviews?disagreement_only=true&page=1&per_page=25"
```

Search the queue:

```bash
curl "http://127.0.0.1:5400/api/v1/discrepancy-reviews?disagreement_only=true&search=1AFO&page=1&per_page=25"
```

Fetch summary counts for the same filtered search:

```bash
curl "http://127.0.0.1:5400/api/v1/discrepancy-reviews/summary?disagreement_only=true&search=bitopic"
```

Export the filtered queue:

```bash
curl -L "http://127.0.0.1:5400/api/v1/discrepancy-reviews/export?disagreement_only=true&search=bitopic&format=csv" -o metamp_discrepancy_review_queue.csv
curl -L "http://127.0.0.1:5400/api/v1/discrepancy-reviews/export?disagreement_only=true&format=json" -o metamp_discrepancy_review_queue.json
curl -L "http://127.0.0.1:5400/api/v1/discrepancy-reviews/export?disagreement_only=true&format=xlsx" -o metamp_discrepancy_review_queue.xlsx
curl -L "http://127.0.0.1:5400/api/v1/discrepancy-reviews/export?disagreement_only=true&format=tsv" -o metamp_discrepancy_review_queue.tsv
```

Operational notes:
- `Group (MPstruc)` in discrepancy payloads now comes from the live `membrane_proteins.group` column when available
- queue responses return `items`, `pagination`, `filters`, and `export_formats`
- supported queue export formats are `json`, `csv`, `xlsx`, and `tsv`

### 8. Run Validation

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask validate-protein-datasets
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask validate-dashboard-regressions
```

What this does:
- checks curated dataset integrity
- checks known operator-critical cases like `EGFR`, replacement resolution, and enriched detail records
- confirms that the most important API behaviors still work

### 9. Use the Bootstrap Marker

The bootstrap script writes:

```text
/var/app/data/bootstrap/production-bootstrap-state.json
```

What this does:
- marks that the heavy first-time bootstrap already finished
- allows later runs to skip expensive repeated initialization
- lets containers start quickly and rely on existing local artifacts

## One-Script Commands

These are the commands most operators should use.

### Full first-time bootstrap

```bash
./scripts/metamp-production-bootstrap.sh run
```

Use this when:
- you are setting up MetaMP locally for the first time
- you want backend services and the full data/ML bootstrapping

### Full first-time bootstrap with local frontend

```bash
./scripts/metamp-production-bootstrap.sh run --with-local-frontend
```

Use this when:
- you want the backend plus the current local `MPVisualization` frontend

### Force a full rebuild and rebootstrap

```bash
./scripts/metamp-production-bootstrap.sh run --force-bootstrap
```

Use this when:
- you changed ingestion logic
- you changed ML logic
- you want to ignore the existing bootstrap marker

### Build without cache

```bash
./scripts/metamp-production-bootstrap.sh run --force-bootstrap --no-cache
```

Use this when:
- Docker image caching is hiding a change you expect to see

### Run in the background on macOS

```bash
nohup caffeinate -dimsu ./scripts/metamp-production-bootstrap.sh run --with-frontend > bootstrap-full.log 2>&1 < /dev/null &
```

Notes:
- keep the redirection exactly as shown; `> 2>&` is invalid and will exit immediately
- the bootstrap script intentionally exits with status `1` when a required step fails, including ML/TM task failures
- watch progress with `tail -f bootstrap-full.log`

### Inspect current status

```bash
./scripts/metamp-production-bootstrap.sh status
```

What it shows:
- running service state
- bootstrap marker state
- latest maintenance information

### Tail logs

```bash
./scripts/metamp-production-bootstrap.sh logs
./scripts/metamp-production-bootstrap.sh logs flask-app
./scripts/metamp-production-bootstrap.sh logs celery-worker-ml
./scripts/metamp-production-bootstrap.sh logs celery-worker
./scripts/metamp-production-bootstrap.sh logs celery-worker-tm
./scripts/metamp-production-bootstrap.sh logs celery-beat
```

What this does:
- helps you inspect startup, ingestion, ML, and validation behavior

### Reset bootstrap state

```bash
./scripts/metamp-production-bootstrap.sh reset
```

What this does:
- removes the bootstrap marker
- makes the next `run` perform the heavy initialization again

## Flask CLI Commands

All commands are registered in [manage.py](/Users/awotoroebenezer/Desktop/MetaMP-Server/manage.py).

Use them like this:

```bash
env FLASK_APP=manage.py flask <command>
```

Inside Docker, the common form is:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask <command>
```

### Core data commands

- `refresh-protein-datasets`
  Refresh source artifacts only.

- `sync-protein-schema`
  Sync database schema expectations for the protein dataset.

- `load-protein-datasets`
  Load the current curated dataset into PostgreSQL.

- `sync-protein-database`
  Full ingestion plus schema plus load path.

- `protein-refresh-status`
  Show latest artifact refresh status.

- `protein-database-sync-status`
  Show latest full DB sync status.

### ML and inference commands

- `queue-machine-learning-job`
  Queue the main ML pipeline.

- `backfill-tm-predictions`
  Queue TM prediction backfill.

- `tm-prediction-status`
  Show the latest TM backfill task state.

- `celery-list-jobs`
  Show active, reserved, and scheduled Celery jobs across workers.

- `celery-task-status --task-id <id>`
  Inspect a specific Celery task.

- `celery-revoke-task --task-id <id> [--terminate]`
  Revoke a queued task, or terminate a running one.

- `celery-purge-queued --yes`
  Purge waiting Celery tasks from the broker queues.

### Validation and benchmark commands

- `validate-protein-datasets`
  Run dataset validation checks.

- `validate-dashboard-regressions`
  Run dashboard-critical regression checks.

- `export-discrepancy-benchmark`
  Export the discrepancy benchmark.

- `discrepancy-benchmark-status`
  Show latest benchmark export metadata.

- `export-high-confidence-subset --format json|csv`
  Export the stronger-confidence subset.

### Maintenance commands

- `queue-production-maintenance`
  Queue the scheduled maintenance chain manually.

- `production-maintenance-status`
  Show the last maintenance run state.

### TMAlphaFold-specific commands

- `sync-tmalphafold-predictions --all --with-tmdet --with-tmbed --tmbed-refresh`
  Run the full TMAlphaFold normalization workflow immediately and persist it.

- `queue-tmalphafold-sync --with-tmdet --with-tmbed --tmbed-refresh`
  Queue the same TMAlphaFold normalization workflow on Celery for production use.

- `sync-tmbed-predictions --all`
  Run TMbed directly without the TMAlphaFold phase.

## Docker Compose Commands

Use this default form:

```bash
docker compose --env-file .env.docker.deployment <command>
```

Examples:

```bash
docker compose --env-file .env.docker.deployment ps
docker compose --env-file .env.docker.deployment logs -f flask-app
docker compose --env-file .env.docker.deployment logs -f celery-worker
docker compose --env-file .env.docker.deployment logs -f celery-worker-ml
docker compose --env-file .env.docker.deployment logs -f celery-worker-tm
docker compose --env-file .env.docker.deployment logs -f celery-beat
docker compose --env-file .env.docker.deployment down
docker compose --env-file .env.docker.deployment build flask-app celery-worker celery-worker-ml celery-worker-tm celery-beat
```

## Queue and Log Monitoring

Use these commands when you want to answer three practical questions:
- did the command queue a task?
- is the task waiting or running?
- which container logs should I follow?

Queue the ML pipeline:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask queue-machine-learning-job
```

List current Celery work across workers:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask celery-list-jobs
```

Inspect one task by id:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask celery-task-status --task-id <task-id>
```

Tail the most useful logs:

```bash
docker compose --env-file .env.docker.deployment logs -f celery-worker-ml
docker compose --env-file .env.docker.deployment logs -f celery-worker-tm
docker compose --env-file .env.docker.deployment logs -f celery-worker
docker compose --env-file .env.docker.deployment logs -f flask-app
```

Service guide:
- `celery-worker-ml`: machine-learning training and evaluation jobs
- `celery-worker-tm`: queued TMAlphaFold and TM prediction normalization jobs
- `celery-worker`: general background jobs and non-ML Celery work
- `flask-app`: API-side queue submission, CLI output, and request handling
- `celery-beat`: scheduled maintenance and periodic jobs

Typical monitoring workflow:
- run the queue command and copy the returned task id
- run `celery-list-jobs` to see whether the task is `reserved`, `scheduled`, or `active`
- follow the matching worker logs
- use `celery-task-status --task-id <task-id>` to confirm final success or failure

## Publication Snapshot Export

Run the live publication extractor after the ML and TM artifacts you want are already present:

```bash
cd /path/to/MetaMP-Server
python3 scripts/extract_live_metrics.py
```

What this does:
- reads the currently running MetaMP Docker containers
- copies live ML artifacts, expert benchmark exports, and production figures
- recomputes manuscript-ready tables under `.metamp-publication/`
- renders publication figures, including the model-registry figure set
- updates `.metamp-publication/latest`

Capture the extractor output to a logfile:

```bash
cd /path/to/MetaMP-Server
python3 scripts/extract_live_metrics.py 2>&1 | tee extract-live-metrics.log
```

Important:
- `extract_live_metrics.py` is a foreground host-side script, not a queued Celery task
- you track it in the terminal where you launched it, or through the `tee` log file above
- it expects the target MetaMP containers to already be running

Recommended end-to-end workflow:

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask queue-machine-learning-job
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask celery-list-jobs
docker compose --env-file .env.docker.deployment logs -f celery-worker-ml
```

After the ML job completes:

```bash
cd /path/to/MetaMP-Server
python3 scripts/extract_live_metrics.py 2>&1 | tee extract-live-metrics.log
```

Useful outputs:
- `.metamp-publication/latest/metadata/publication_manifest.json`
- `.metamp-publication/latest/metadata/live_metrics.json`
- `.metamp-publication/latest/derived/tables/`
- `.metamp-publication/latest/copied/publication_figures/`
- `.metamp-publication/latest/copied/publication_figures/model_registry/`

## Monthly Maintenance

After the first bootstrap, the stack does not need to redo the full heavy initialization on every start.

Instead:
- the running containers can use the already prepared local state
- `celery-beat` schedules the monthly production maintenance task
- that maintenance task refreshes sources, queues ML/TM work, exports benchmarks, and reruns validation

That means the intended lifecycle is:

1. run the bootstrap once
2. keep and reuse the generated local state
3. let monthly maintenance keep the installation current

## Practical Workflows

### Easiest path for a new user

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-production-bootstrap.sh run --with-local-frontend
```

### Backend only for operators

```bash
cd /path/to/MetaMP-Server
./scripts/metamp-production-bootstrap.sh run
```

### Re-run just the data sync manually

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask sync-protein-database
```

### Re-run just ML manually

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask queue-machine-learning-job
```

### Re-run just TM predictions manually

```bash
docker compose --env-file .env.docker.deployment exec -T flask-app env FLASK_APP=manage.py flask backfill-tm-predictions
```

For frontend building
```bash
bash ./scripts/metamp-frontend-image.sh push --env-file .env.docker.deployment
chmod +x ./scripts/metamp-frontend-image.sh
./scripts/metamp-frontend-image.sh push --env-file .env.docker.deployment


nohup caffeinate -dimsu ./scripts/metamp-production-bootstrap.sh run \
  --with-frontend \
  --force-bootstrap \
  --stages pdb,quantitative,opm,uniprot,clean_valid,validate_release \
  --refresh-stages pdb,quantitative,opm,uniprot,clean_valid,validate_release \
  > bootstrap-full.log 2>&1 < /dev/null &


```
